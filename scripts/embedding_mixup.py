"""Embedding-space Mixup / CutMix for the v4 multi-scale LR head.

Research question
-----------------
The prior `augmented_head_training.py` experiment (T3 = D4-expand) was a
*redundant* augmentation: every expanded sample was a geometric rotation of an
existing training sample, which only adds slight noise in the final embedding.
It yielded -0.038 w-F1 (ROLLBACK).

**Mixup is fundamentally different:** it interpolates *between classes*, which
acts as a Vicinal Risk Minimization regularizer and a label-smoothing prior.
Cross-class mixup is widely reported to give +1-3 pp on small medical datasets.

This script runs Mixup and CutMix entirely in **embedding space** (no backbone
retrain). Budget: pure local, ~15 min.

Treatments (person-LOPO, 35 folds, seed 42, 3-way geo-mean ensemble as v4):

  M0 — sanity (averaged head only, == v4)           baseline == 0.6887
  M1 — standard Mixup (any pair, lambda~Beta(0.4,0.4))
  M2 — class-restricted Mixup (only within-class pairs; regularization only)
  M3 — minority-boost Mixup (oversample pairs involving SucheOko / Diabetes)
  M4 — CutMix on 9-tile embeddings (swap half the tiles between scans of
       different classes)

Each Mi uses the same treatment on ALL THREE components of the v4 ensemble:
  A) DINOv2-B @ 90 nm/px
  B) DINOv2-B @ 45 nm/px
  C) BiomedCLIP @ 90 nm/px

The LR head is trained via soft-label cross-entropy (torch, LBFGS). For M0 we
use the sklearn path so that it exactly reproduces v4's T1 sanity.

Outputs
-------
  cache/mixup_predictions.json
  reports/EMBEDDING_MIXUP.md
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import CLASSES, person_id  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
CACHE.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

N_CLASSES = len(CLASSES)
EPS = 1e-12
V4_BASELINE = 0.6887

# Minority classes (smallest training populations)
MINORITY = {CLASSES.index("SucheOko"), CLASSES.index("Diabetes")}

# Mixup hyperparameters (keep fixed & documented)
BETA_ALPHA = 0.4
N_MIX_FACTOR = 1  # create N mixup samples (same N as originals)
CUTMIX_FRAC = 0.5  # fraction of tiles swapped in CutMix
SEED = 42

DEVICE = "cpu"  # tiny head, cpu is fine and fully reproducible


# ---------------------------------------------------------------------------
# I/O helpers (copied from augmented_head_training.py)
# ---------------------------------------------------------------------------

def _resolve_paths(arr) -> list[str]:
    return [str(Path(str(p)).resolve()) for p in arr]


def align_to_reference(paths_ref, paths_src, X_src):
    src_idx = {p: i for i, p in enumerate(paths_src)}
    order = np.array([src_idx[p] for p in paths_ref])
    return X_src[order]


def _mean_pool_tiles(X_tiles, tile_to_scan, n_scans):
    D = X_tiles.shape[1]
    out = np.zeros((n_scans, D), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int64)
    for i, s in enumerate(tile_to_scan):
        out[s] += X_tiles[i]
        counts[s] += 1
    counts = np.maximum(counts, 1)
    out /= counts[:, None]
    return out


# ---------------------------------------------------------------------------
# Soft-label LR head (torch) for mixup/cutmix
# ---------------------------------------------------------------------------

def fit_predict_soft_lr(
    X_train: np.ndarray, Y_train: np.ndarray,  # Y_train: (n, N_CLASSES) soft
    X_test: np.ndarray,
    sample_weight: np.ndarray | None = None,
    l2: float = 1.0,
    max_iter: int = 500,
    class_weight: bool = True,
) -> np.ndarray:
    """Linear head trained with softmax cross-entropy on soft labels.

    Mirrors sklearn LogisticRegression(C=1.0, solver='lbfgs',
    class_weight='balanced') but supports soft targets.
    """
    # L2-normalize & standard-scale (match v4)
    X_tr_n = normalize(X_train, norm="l2", axis=1)
    X_te_n = normalize(X_test, norm="l2", axis=1)
    sc = StandardScaler().fit(X_tr_n)
    X_tr_s = np.nan_to_num(sc.transform(X_tr_n)).astype(np.float32)
    X_te_s = np.nan_to_num(sc.transform(X_te_n)).astype(np.float32)

    Xtr = torch.from_numpy(X_tr_s).to(DEVICE)
    Ytr = torch.from_numpy(Y_train.astype(np.float32)).to(DEVICE)
    Xte = torch.from_numpy(X_te_s).to(DEVICE)

    D = Xtr.shape[1]
    W = torch.zeros(D, N_CLASSES, device=DEVICE, requires_grad=True)
    b = torch.zeros(N_CLASSES, device=DEVICE, requires_grad=True)

    # Class weights mimicking sklearn "balanced" on the *hard* argmax of Y
    if class_weight:
        hard = Y_train.argmax(axis=1)
        n = len(hard)
        counts = np.bincount(hard, minlength=N_CLASSES).astype(np.float64)
        cw_np = np.where(counts > 0, n / (N_CLASSES * counts), 0.0)
        cw = torch.from_numpy(cw_np.astype(np.float32)).to(DEVICE)
    else:
        cw = torch.ones(N_CLASSES, device=DEVICE)

    if sample_weight is None:
        sw = torch.ones(len(Y_train), device=DEVICE)
    else:
        sw = torch.from_numpy(sample_weight.astype(np.float32)).to(DEVICE)

    # Effective per-sample weight = sw * (sum over classes of Y_i * cw_c)
    eff_w = sw * (Ytr * cw.unsqueeze(0)).sum(dim=1)
    eff_w = eff_w / (eff_w.mean() + EPS)  # normalize scale

    opt = torch.optim.LBFGS(
        [W, b], max_iter=max_iter, tolerance_grad=1e-6,
        tolerance_change=1e-8, history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        logits = Xtr @ W + b
        logp = F.log_softmax(logits, dim=1)
        # Soft-target cross-entropy per sample
        loss_i = -(Ytr * logp).sum(dim=1)
        loss = (eff_w * loss_i).mean()
        # L2 regularization on W only (sklearn style: penalty 1/C * 0.5 ||W||^2)
        reg = 0.5 * (W * W).sum() / l2
        loss = loss + reg / len(Y_train)
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        logits_te = Xte @ W + b
        proba = F.softmax(logits_te, dim=1).cpu().numpy()
    return proba  # always dense over full N_CLASSES


def fit_predict_averaged_sklearn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Exact v4 T1 sanity path via sklearn."""
    X_train_n = normalize(X_train, norm="l2", axis=1)
    X_test_n = normalize(X_test, norm="l2", axis=1)
    sc = StandardScaler().fit(X_train_n)
    X_train_s = np.nan_to_num(sc.transform(X_train_n))
    X_test_s = np.nan_to_num(sc.transform(X_test_n))
    clf = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0,
                             solver="lbfgs", n_jobs=4, random_state=42)
    clf.fit(X_train_s, y_train)
    proba = clf.predict_proba(X_test_s)
    full = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    for ci, cls in enumerate(clf.classes_):
        full[:, cls] = proba[:, ci]
    return full


# ---------------------------------------------------------------------------
# Mixup samplers
# ---------------------------------------------------------------------------

def make_mixup(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
    mode: str = "any", n_mix: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create mixup augmentations.

    mode:
        "any"      — sample pairs uniformly (standard Mixup)
        "within"   — only within-class pairs (regularization only)
        "minority" — upweight pairs involving minority classes
    """
    n = len(y)
    if n_mix is None:
        n_mix = n
    lambdas = rng.beta(BETA_ALPHA, BETA_ALPHA, size=n_mix).astype(np.float32)

    y_oh = np.zeros((n, N_CLASSES), dtype=np.float32)
    y_oh[np.arange(n), y] = 1.0

    idx_i = np.empty(n_mix, dtype=np.int64)
    idx_j = np.empty(n_mix, dtype=np.int64)

    if mode == "any":
        idx_i = rng.integers(0, n, size=n_mix)
        idx_j = rng.integers(0, n, size=n_mix)
    elif mode == "within":
        # Per-class index pools
        class_pool = {c: np.where(y == c)[0] for c in range(N_CLASSES)}
        # Draw a class c with probability proportional to class size (so larger
        # classes get more mixup samples; small classes cannot dominate).
        class_sizes = np.array([len(class_pool[c]) for c in range(N_CLASSES)],
                               dtype=np.float64)
        class_p = class_sizes / class_sizes.sum()
        chosen = rng.choice(N_CLASSES, size=n_mix, p=class_p)
        for k, c in enumerate(chosen):
            pool = class_pool[int(c)]
            if len(pool) < 2:
                idx_i[k] = pool[0]; idx_j[k] = pool[0]
            else:
                a, b = rng.choice(pool, size=2, replace=False)
                idx_i[k] = a; idx_j[k] = b
    elif mode == "minority":
        # Half the mixup samples: one index must be in a minority class
        minor_mask = np.isin(y, list(MINORITY))
        minor_idx = np.where(minor_mask)[0]
        if len(minor_idx) == 0:
            return make_mixup(X, y, rng, mode="any", n_mix=n_mix)
        n_half = n_mix // 2
        idx_i[:n_half] = rng.choice(minor_idx, size=n_half, replace=True)
        idx_j[:n_half] = rng.integers(0, n, size=n_half)
        idx_i[n_half:] = rng.integers(0, n, size=n_mix - n_half)
        idx_j[n_half:] = rng.integers(0, n, size=n_mix - n_half)
    else:
        raise ValueError(mode)

    X_mix = lambdas[:, None] * X[idx_i] + (1.0 - lambdas)[:, None] * X[idx_j]
    Y_mix = lambdas[:, None] * y_oh[idx_i] + (1.0 - lambdas)[:, None] * y_oh[idx_j]
    return X_mix.astype(np.float32), Y_mix.astype(np.float32)


def make_cutmix_tiles(
    tile_emb: dict[int, np.ndarray],  # scan_idx -> (n_tiles, D)
    scan_y: np.ndarray,
    rng: np.random.Generator,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CutMix analog: build synthetic scan-embeddings by swapping ~half the
    tiles between two scans of different classes, then mean-pool.

    Returns (n_cut, D) scan embeddings and (n_cut, N_CLASSES) soft labels.
    """
    Xs, Ys = [], []
    train_idx = np.asarray(train_idx)
    for si in train_idx:
        tiles_i = tile_emb[int(si)]  # (ni, D)
        yi = int(scan_y[si])
        # Pick a partner from another class
        other = [int(sj) for sj in train_idx
                 if int(scan_y[sj]) != yi]
        if not other:
            continue
        sj = int(rng.choice(other))
        tiles_j = tile_emb[sj]
        yj = int(scan_y[sj])
        ni = tiles_i.shape[0]
        nj = tiles_j.shape[0]
        # Swap floor(ni * CUTMIX_FRAC) tiles from i with tiles from j
        n_swap = max(1, int(round(ni * CUTMIX_FRAC)))
        keep = rng.choice(ni, size=ni - n_swap, replace=False) if ni > n_swap else np.array([], dtype=int)
        swap_from_j = rng.choice(nj, size=n_swap, replace=True)
        new_tiles = np.vstack([
            tiles_i[keep] if len(keep) else np.zeros((0, tiles_i.shape[1]), dtype=tiles_i.dtype),
            tiles_j[swap_from_j],
        ])
        # Mean-pool to get scan embedding (same as v4 aggregation)
        emb = new_tiles.mean(axis=0)
        # Soft label proportional to tile counts
        frac_j = n_swap / ni
        frac_i = 1.0 - frac_j
        y_oh = np.zeros(N_CLASSES, dtype=np.float32)
        y_oh[yi] += frac_i
        y_oh[yj] += frac_j
        Xs.append(emb)
        Ys.append(y_oh)
    if not Xs:
        return (np.zeros((0, tiles_i.shape[1]), dtype=np.float32),
                np.zeros((0, N_CLASSES), dtype=np.float32))
    return np.stack(Xs).astype(np.float32), np.stack(Ys).astype(np.float32)


# ---------------------------------------------------------------------------
# LOPO drivers
# ---------------------------------------------------------------------------

def lopo_averaged_sk(X, y, groups):
    P = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups):
        P[va] = fit_predict_averaged_sklearn(X[tr], y[tr], X[va])
    return P


def lopo_mixup(X, y, groups, mode: str, seed: int = SEED) -> np.ndarray:
    """Train LR head on original + mixup samples, per fold."""
    rng_master = np.random.default_rng(seed)
    P = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    for fold_i, (tr, va) in enumerate(leave_one_patient_out(groups)):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        Xtr = X[tr]
        ytr = y[tr]
        # Hard label matrix for originals
        Ytr = np.zeros((len(ytr), N_CLASSES), dtype=np.float32)
        Ytr[np.arange(len(ytr)), ytr] = 1.0
        # Mixup samples (same count as original)
        Xmix, Ymix = make_mixup(Xtr, ytr, rng, mode=mode, n_mix=len(ytr))
        X_exp = np.vstack([Xtr.astype(np.float32), Xmix])
        Y_exp = np.vstack([Ytr, Ymix])
        # Downweight mixup samples slightly (lambda=0.5 informative weight)
        sw = np.concatenate([np.ones(len(ytr)),
                              np.full(len(Xmix), 0.5)]).astype(np.float32)
        P[va] = fit_predict_soft_lr(X_exp, Y_exp, X[va], sample_weight=sw)
    return P


def lopo_cutmix_tiles(
    X_tiles: np.ndarray, tile_to_scan: np.ndarray,
    scan_y: np.ndarray, groups: np.ndarray,
    seed: int = SEED,
) -> np.ndarray:
    """CutMix: train on original scan embeddings + synthetic scans built by
    swapping half the tiles between scans of different classes."""
    rng_master = np.random.default_rng(seed)
    n_scans = len(scan_y)
    # Build scan -> tile_emb dict once
    tile_emb: dict[int, np.ndarray] = {}
    for ti, si in enumerate(tile_to_scan):
        tile_emb.setdefault(int(si), []).append(X_tiles[ti])
    tile_emb = {k: np.stack(v).astype(np.float32) for k, v in tile_emb.items()}
    # Also need scan-level averages (v4 uses these)
    X_scan = _mean_pool_tiles(X_tiles, tile_to_scan, n_scans)

    P = np.zeros((n_scans, N_CLASSES), dtype=np.float64)
    for fold_i, (tr, va) in enumerate(leave_one_patient_out(groups)):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        # Original labels as one-hot
        ytr = scan_y[tr]
        Ytr = np.zeros((len(ytr), N_CLASSES), dtype=np.float32)
        Ytr[np.arange(len(ytr)), ytr] = 1.0
        # CutMix synthetic samples (one per scan in train fold)
        Xcut, Ycut = make_cutmix_tiles(tile_emb, scan_y, rng, tr)
        X_exp = np.vstack([X_scan[tr].astype(np.float32), Xcut])
        Y_exp = np.vstack([Ytr, Ycut])
        sw = np.concatenate([np.ones(len(tr)),
                              np.full(len(Xcut), 0.5)]).astype(np.float32)
        P[va] = fit_predict_soft_lr(X_exp, Y_exp, X_scan[va], sample_weight=sw)
    return P


# ---------------------------------------------------------------------------
# Ensemble & metrics
# ---------------------------------------------------------------------------

def geom_mean(probs_list: list[np.ndarray]) -> np.ndarray:
    log_sum = np.zeros_like(probs_list[0])
    for P in probs_list:
        log_sum = log_sum + np.log(P + EPS)
    G = np.exp(log_sum / len(probs_list))
    G /= G.sum(axis=1, keepdims=True)
    return G


def metrics_of(P: np.ndarray, y: np.ndarray) -> dict:
    pred = P.argmax(axis=1)
    return {
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(
            y, pred, average=None, labels=list(range(N_CLASSES)), zero_division=0,
        ).tolist(),
    }


def bootstrap_delta(P_new: np.ndarray, P_base: np.ndarray, y: np.ndarray,
                    n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.zeros(n_boot, dtype=np.float64)
    pred_new = P_new.argmax(axis=1)
    pred_base = P_base.argmax(axis=1)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        f_new = f1_score(yb, pred_new[idx], average="weighted", zero_division=0)
        f_base = f1_score(yb, pred_base[idx], average="weighted", zero_division=0)
        deltas[b] = f_new - f_base
    return {
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std()),
        "delta_p05": float(np.percentile(deltas, 5)),
        "delta_p95": float(np.percentile(deltas, 95)),
        "p_gt_zero": float((deltas > 0).mean()),
    }


def _verdict(delta: float, abs_wf1: float, p_gt: float) -> str:
    if abs_wf1 >= 0.70 and p_gt >= 0.90:
        return f"CHAMPION_CANDIDATE (wF1={abs_wf1:.4f}, p(>0)={p_gt:.2f})"
    if delta >= 0.02 and p_gt >= 0.90:
        return f"PROMOTE (delta={delta:+.4f}, p(>0)={p_gt:.2f})"
    if delta >= 0.01:
        return f"INCONCLUSIVE_POSITIVE (delta={delta:+.4f}, p(>0)={p_gt:.2f})"
    if delta <= -0.01:
        return f"ROLLBACK (delta={delta:+.4f}, p(>0)={p_gt:.2f})"
    return f"NOISE_FLOOR (delta={delta:+.4f}, p(>0)={p_gt:.2f})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print("Embedding Mixup / CutMix for v4 multi-scale ensemble")
    print(f"v4 baseline (honest person-LOPO weighted F1): {V4_BASELINE}")
    print("=" * 78)

    # --- Load v4 scan-level features ---
    print("\n[load] v4 caches")
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz",
                  allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz",
                  allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz",
                  allow_pickle=True)

    paths_90 = _resolve_paths(z90["scan_paths"])
    paths_45 = _resolve_paths(z45["scan_paths"])
    paths_bc = _resolve_paths(zbc["scan_paths"])

    y = np.asarray(z90["scan_y"], dtype=np.int64)
    groups = np.array([person_id(Path(p)) for p in paths_90])
    n_scans = len(y)
    n_persons = len(np.unique(groups))
    print(f"  n_scans={n_scans}  n_persons={n_persons}  "
          f"class counts={np.bincount(y).tolist()}")

    # Scan-level features
    X90 = _mean_pool_tiles(z90["X"], z90["tile_to_scan"], len(paths_90))
    X45_raw = _mean_pool_tiles(z45["X"], z45["tile_to_scan"], len(paths_45))
    X45 = align_to_reference(paths_90, paths_45, X45_raw)
    Xbc = align_to_reference(paths_90, paths_bc,
                              zbc["X_scan"].astype(np.float32))
    print(f"  averaged features: X90={X90.shape} X45={X45.shape} Xbc={Xbc.shape}")

    # Tile-level for CutMix (need index alignment to paths_90 too)
    # For z45 and zbc the tile ordering differs — but we only need CutMix on
    # the DINOv2 90 stream for the scalar-DINOv2 cutmix experiment. We apply
    # CutMix to ALL THREE streams using each stream's own tile cache.
    def _stream_tiles(z, paths):
        # z['X'] tiles, z['tile_to_scan'] indexes within paths order; remap
        tile_to_scan_ref = np.array(
            [paths_90.index(paths[int(s)]) for s in z["tile_to_scan"]],
            dtype=np.int64,
        )
        return z["X"].astype(np.float32), tile_to_scan_ref

    T90_X, T90_t2s = _stream_tiles(z90, paths_90)
    T45_X, T45_t2s = _stream_tiles(z45, paths_45)
    # For BiomedCLIP we need per-tile embeddings; use the per-view cache
    zbc_pv = np.load(CACHE / "tta_perview_biomedclip_afmhot_t512_n9_d4.npz",
                     allow_pickle=True)
    # Reduce per-view -> per-tile by averaging d4 views
    # view_to_tile is per-scan tile index; build (scan, tile) keys
    pv_paths_bc = _resolve_paths(zbc_pv["scan_paths"])
    Xpv = zbc_pv["X"].astype(np.float32)
    v2s = zbc_pv["view_to_scan"].astype(np.int64)
    v2t = zbc_pv["view_to_tile"].astype(np.int64)
    # Build tile-level bc embeddings
    tile_key_to_embs: dict[tuple[int, int], list] = {}
    for i in range(Xpv.shape[0]):
        tile_key_to_embs.setdefault((int(v2s[i]), int(v2t[i])), []).append(Xpv[i])
    bc_tile_X = []
    bc_tile_t2s = []  # scan index in pv_paths_bc order
    for (si, ti), embs in tile_key_to_embs.items():
        bc_tile_X.append(np.stack(embs).mean(axis=0))
        bc_tile_t2s.append(si)
    bc_tile_X = np.stack(bc_tile_X).astype(np.float32)
    bc_tile_t2s_pv = np.array(bc_tile_t2s, dtype=np.int64)
    # Remap pv scan -> paths_90 scan idx
    pv_to_ref = np.array([paths_90.index(p) for p in pv_paths_bc], dtype=np.int64)
    Tbc_t2s = pv_to_ref[bc_tile_t2s_pv]
    Tbc_X = bc_tile_X
    print(f"  tile counts: 90={T90_X.shape} 45={T45_X.shape} bc={Tbc_X.shape}")

    # ======================================================================
    # M0 — sanity (sklearn T1)
    # ======================================================================
    print("\n[M0] averaged-head sanity (sklearn)")
    P90_m0 = lopo_averaged_sk(X90, y, groups)
    P45_m0 = lopo_averaged_sk(X45, y, groups)
    Pbc_m0 = lopo_averaged_sk(Xbc, y, groups)
    G_m0 = geom_mean([P90_m0, P45_m0, Pbc_m0])
    m_m0 = metrics_of(G_m0, y)
    print(f"  M0 W-F1={m_m0['weighted_f1']:.4f}  M-F1={m_m0['macro_f1']:.4f}  "
          f"(expect ~{V4_BASELINE})")

    # ======================================================================
    # M1..M3 — Mixup variants
    # ======================================================================
    results = {"M0_sanity": m_m0}
    probs = {"M0": G_m0}

    for label, mode in [("M1_mixup_any", "any"),
                         ("M2_mixup_within", "within"),
                         ("M3_mixup_minority", "minority")]:
        print(f"\n[{label}] mode={mode}")
        P90 = lopo_mixup(X90, y, groups, mode=mode, seed=SEED)
        P45 = lopo_mixup(X45, y, groups, mode=mode, seed=SEED + 1)
        Pbc = lopo_mixup(Xbc, y, groups, mode=mode, seed=SEED + 2)
        G = geom_mean([P90, P45, Pbc])
        m = metrics_of(G, y)
        print(f"  {label}  W-F1={m['weighted_f1']:.4f}  M-F1={m['macro_f1']:.4f}")
        results[label] = m
        probs[label[:2]] = G

    # ======================================================================
    # M4 — CutMix on tile embeddings
    # ======================================================================
    print("\n[M4_cutmix] tile-swap CutMix (frac=0.5)")
    P90_m4 = lopo_cutmix_tiles(T90_X, T90_t2s, y, groups, seed=SEED)
    P45_m4 = lopo_cutmix_tiles(T45_X, T45_t2s, y, groups, seed=SEED + 1)
    Pbc_m4 = lopo_cutmix_tiles(Tbc_X, Tbc_t2s, y, groups, seed=SEED + 2)
    G_m4 = geom_mean([P90_m4, P45_m4, Pbc_m4])
    m_m4 = metrics_of(G_m4, y)
    print(f"  M4 W-F1={m_m4['weighted_f1']:.4f}  M-F1={m_m4['macro_f1']:.4f}")
    results["M4_cutmix"] = m_m4
    probs["M4"] = G_m4

    # ======================================================================
    # Bootstrap (vs M0 sanity == v4 reproduction)
    # ======================================================================
    print("\n[bootstrap] 1000 x paired vs M0 sanity (which == v4)")
    boot = {}
    for key in ["M1", "M2", "M3", "M4"]:
        b = bootstrap_delta(probs[key], probs["M0"], y, n_boot=1000, seed=42)
        boot[key] = b
        print(f"  {key}: delta_mean={b['delta_mean']:+.4f}  "
              f"P(delta>0)={b['p_gt_zero']:.3f}  "
              f"[{b['delta_p05']:+.4f}, {b['delta_p95']:+.4f}]")

    # ======================================================================
    # Verdicts
    # ======================================================================
    verdicts = {}
    print("\nVerdicts (promotion rule: wF1>=0.70 AND p>0.90, or delta>=+0.02 AND p>=0.90):")
    for key in ["M1", "M2", "M3", "M4"]:
        label_full = [k for k in results if k.startswith(key)][0]
        wf1 = results[label_full]["weighted_f1"]
        verdicts[key] = _verdict(boot[key]["delta_mean"], wf1, boot[key]["p_gt_zero"])
        print(f"  {key}: {verdicts[key]}")

    # ======================================================================
    # Save
    # ======================================================================
    out = {
        "v4_baseline_weighted_f1": V4_BASELINE,
        "n_scans": int(n_scans),
        "n_persons": int(n_persons),
        "hyperparams": {
            "beta_alpha": BETA_ALPHA,
            "n_mix_factor": N_MIX_FACTOR,
            "cutmix_frac": CUTMIX_FRAC,
            "seed": SEED,
            "mixup_sample_weight": 0.5,
            "head": "torch LBFGS softmax soft-CE (L2=1, balanced class weights)",
        },
        "results": results,
        "bootstrap_vs_M0": boot,
        "verdicts": verdicts,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (CACHE / "mixup_predictions.json").write_text(json.dumps(out, indent=2))
    print(f"\n[saved] cache/mixup_predictions.json")

    _write_markdown(out)
    print(f"[saved] reports/EMBEDDING_MIXUP.md")
    print(f"\n[done] total elapsed: {time.time() - t0:.1f}s")
    return out


def _write_markdown(out: dict):
    L: list[str] = []
    L.append("# Embedding-space Mixup / CutMix — Results\n")
    L.append("**Question:** Does Mixup / CutMix in *embedding space* (on top of frozen "
             "DINOv2-B + BiomedCLIP features) improve the v4 LR-head ensemble?\n")
    L.append("Rationale: prior `augmented_head_training.py` T3 (D4-expand) failed because "
             "rotated views live close to the original in embedding space. Mixup "
             "interpolates *between classes*, a fundamentally different regularizer.\n")

    L.append("## Treatments\n")
    L.append("- **M0 sanity** — averaged-head sklearn LR (== v4 T1, must reproduce 0.6887).\n")
    L.append("- **M1 Mixup (any)** — add N mixup samples per fold, λ ~ Beta(0.4, 0.4), any pair.\n")
    L.append("- **M2 Mixup (within)** — only within-class pairs (pure regularization, no soft labels).\n")
    L.append("- **M3 Mixup (minority)** — 50%% of pairs anchored on a minority-class sample (SucheOko or Diabetes).\n")
    L.append("- **M4 CutMix tiles** — build synthetic scan by swapping half the tiles of scan A with tiles of scan B (different class); mean-pool; soft label by tile-fraction.\n")

    L.append("## Setup\n")
    hp = out["hyperparams"]
    L.append(f"- Person-level LOPO, {out['n_persons']} folds, {out['n_scans']} scans.\n")
    L.append(f"- Mixup α = {hp['beta_alpha']}, mixup:original ratio = {hp['n_mix_factor']}, "
             f"mixup sample_weight = {hp['mixup_sample_weight']}.\n")
    L.append(f"- Head: {hp['head']}.\n")
    L.append(f"- 3-way geometric-mean ensemble over DINOv2-90, DINOv2-45, BiomedCLIP-90.\n")
    L.append(f"- v4 reference weighted F1 (reports): **{out['v4_baseline_weighted_f1']}**.\n")

    L.append("## Ensemble-level results\n")
    L.append("| Treatment | Weighted F1 | Macro F1 | Δ vs v4 |")
    L.append("|---|---:|---:|---:|")
    for key in ["M0_sanity", "M1_mixup_any", "M2_mixup_within",
                "M3_mixup_minority", "M4_cutmix"]:
        m = out["results"][key]
        delta = m["weighted_f1"] - out["v4_baseline_weighted_f1"]
        L.append(f"| **{key}** | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} | {delta:+.4f} |")
    L.append("")

    L.append("## Per-class F1\n")
    hdr = "| Treatment | " + " | ".join(CLASSES) + " |"
    L.append(hdr)
    L.append("|---|" + "|".join([":---:"] * N_CLASSES) + "|")
    for key in ["M0_sanity", "M1_mixup_any", "M2_mixup_within",
                "M3_mixup_minority", "M4_cutmix"]:
        pcf1 = out["results"][key]["per_class_f1"]
        L.append(f"| **{key}** | " + " | ".join(f"{v:.4f}" for v in pcf1) + " |")
    L.append("")

    L.append("## Paired bootstrap (1000×) vs M0 sanity\n")
    L.append("| Treatment | Δ mean | Δ p05 | Δ p95 | P(Δ > 0) |")
    L.append("|---|---:|---:|---:|---:|")
    for key in ["M1", "M2", "M3", "M4"]:
        b = out["bootstrap_vs_M0"][key]
        L.append(f"| **{key}** | {b['delta_mean']:+.4f} | {b['delta_p05']:+.4f} | "
                 f"{b['delta_p95']:+.4f} | {b['p_gt_zero']:.3f} |")
    L.append("")

    L.append("## Verdicts\n")
    for key, v in out["verdicts"].items():
        L.append(f"- **{key}**: {v}")
    L.append("")
    L.append("\nPromotion rule: **wF1 ≥ 0.70 AND P(Δ>0) ≥ 0.90** → champion candidate; "
             "Δ ≥ +0.02 AND P(Δ>0) ≥ 0.90 → promote; |Δ| < 0.01 → noise floor.\n")

    (REPORTS / "EMBEDDING_MIXUP.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
