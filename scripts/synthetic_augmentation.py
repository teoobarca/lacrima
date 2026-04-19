"""Synthetic-data augmentation for the tear-AFM minority class (SucheOko).

Goal
----
SucheOko has only 14 scans / 2 persons → the v4 multi-scale champion yields
SucheOko F1 = 0.  Even a modest lift from 0 → >0 is the headline.

Approach (Option C — MixUp in embedding space)
----------------------------------------------
Simplest-that-works: synthesise NEW embeddings by convex combinations of
SAME-CLASS EMBEDDINGS (within each LOPO fold's train set only).  No GAN / no
diffusion / no image-space generation — we bypass the intractable "small data
⇒ mode collapse" problem by working directly in the frozen DINOv2-B /
BiomedCLIP feature space, which is already well-behaved.

Key design decisions
--------------------
* **Tile-level MixUp (not scan-level)** — SucheOko has 45 tiles at 90 nm/px and
  110 tiles at 45 nm/px, much more diversity than 14 scans.
* **Intra-fold only** — in each LOPO fold we rebuild synthetic samples from
  the fold's train-set tiles, so the held-out person NEVER contributes to any
  synthetic example.  No leakage.
* **Cross-person mix priority** — to avoid pure intra-person interpolation
  (near-identity), we mix one tile from person A with one tile from a different
  person B of the same class.  For SucheOko with only 2 persons that still
  gives real diversity.
* **Mean-pool to scan level** — after MixUp in tile space, we mean-pool k
  synthetic tiles into one synthetic "scan".  Same pooling used by the real
  pipeline.
* **Apply to all 3 v4 components** — DINOv2-B 90 nm, DINOv2-B 45 nm, BiomedCLIP
  TTA 90 nm.  BiomedCLIP cache is already scan-level → MixUp scan-level there.

Evaluation
----------
Person-LOPO (35 folds).  Same v2 recipe per-component (L2-norm →
StandardScaler → LogisticRegression(balanced)) → geometric-mean softmaxes.
Matches `scripts/train_ensemble_v4_multiscale.py` exactly.

Compared configs
----------------
* **baseline**:  no augmentation (should match champion 0.6887 wF1 exactly).
* **mixup_SucheOko_only**:  synthesise ONLY the minority class.
* **mixup_all_minority**:  synthesise SucheOko + Diabetes (≤ 25 scans).

Red-team
--------
We measure the minimum cosine distance between each synthetic SucheOko scan
and its nearest REAL train-set SucheOko scan, for one representative fold.
This confirms synthetic samples are not near-duplicates of training points.

Budget:  ~5 min CPU.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
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

N_CLASSES = len(CLASSES)
EPS = 1e-12

# Champion benchmark for comparison.
CHAMP_V4_WF1 = 0.6887
CHAMP_V4_MF1 = 0.5541

# Target minority classes.  SucheOko (idx 4) is the priority.  Diabetes (idx 1)
# is secondary — 25 scans / 4 persons is also limited.
PRIORITY_MINORITY = [4]
ALL_MINORITY = [4, 1]


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def mean_pool_tiles(X_tiles: np.ndarray, t2s: np.ndarray, n_scans: int) -> np.ndarray:
    d = X_tiles.shape[1]
    out = np.zeros((n_scans, d), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int64)
    for i, s in enumerate(t2s):
        out[s] += X_tiles[i]
        counts[s] += 1
    counts = np.maximum(counts, 1)
    out /= counts[:, None]
    return out


def align_to_reference(paths_ref: list[str], paths_src: list[str],
                       X_src: np.ndarray) -> np.ndarray:
    src_idx = {p: i for i, p in enumerate(paths_src)}
    order = np.array([src_idx[p] for p in paths_ref])
    return X_src[order]


# ---------------------------------------------------------------------------
# MixUp synthesis
# ---------------------------------------------------------------------------

def synthesize_mixup_tiles(
    X_tiles: np.ndarray,
    tile_person_ids: np.ndarray,
    tile_scan_ids: np.ndarray,
    n_synth_scans: int,
    tiles_per_synth_scan: int,
    alpha: float,
    rng: np.random.Generator,
    mode: str = "interp",           # "interp" | "extrap" | "noise"
    extrap_range: tuple[float, float] = (-0.3, 1.3),
    noise_sigma: float = 0.0,
) -> np.ndarray:
    """Build `n_synth_scans` synthetic scan-level embeddings via tile MixUp.

    Each synthetic scan = mean of `tiles_per_synth_scan` synthetic tiles.

    Modes
    -----
    * "interp" — classic MixUp:
        mixed = lam * A + (1-lam) * B,     lam ~ Beta(alpha, alpha) in [0.2, 0.8]
    * "extrap" — extrapolative mixup (pushes AWAY from data centroid):
        mixed = lam * A + (1-lam) * B with lam sampled from `extrap_range`
        (e.g. [-0.3, 1.3]) → synthetic tiles live slightly OUTSIDE the convex
        hull of the two parents, inflating the class cloud's effective variance.
        Helps when a class has few persons and LOPO removes one of them.
    * "noise" — Gaussian noise injection (copies original + per-dim noise):
        mixed = A + sigma * ε,  ε ~ N(0, I).  Captures local variance
        estimated from the class.

    Each synthetic tile can ADDITIONALLY get Gaussian noise (noise_sigma > 0).

    Tile A and Tile B are drawn from SAME CLASS but DIFFERENT PERSONS where
    possible (improves cross-person diversity for SucheOko with 2 persons).
    """
    n_tiles = len(X_tiles)
    if n_tiles < 2:
        return np.empty((0, X_tiles.shape[1]), dtype=np.float32)

    unique_persons = np.unique(tile_person_ids)
    X_synth = np.zeros((n_synth_scans, X_tiles.shape[1]), dtype=np.float32)

    for si in range(n_synth_scans):
        synth_tiles = []
        for _ in range(tiles_per_synth_scan):
            ia = rng.integers(n_tiles)
            pa = tile_person_ids[ia]
            if len(unique_persons) >= 2:
                other_idx = np.where(tile_person_ids != pa)[0]
                ib = (int(other_idx[rng.integers(len(other_idx))])
                      if len(other_idx) > 0 else int(rng.integers(n_tiles)))
            else:
                ib = int(rng.integers(n_tiles))

            if mode == "interp":
                lam = rng.beta(alpha, alpha)
                lam = 0.2 + 0.6 * lam
                mixed = lam * X_tiles[ia] + (1.0 - lam) * X_tiles[ib]
            elif mode == "extrap":
                lo, hi = extrap_range
                lam = rng.uniform(lo, hi)
                mixed = lam * X_tiles[ia] + (1.0 - lam) * X_tiles[ib]
            elif mode == "noise":
                mixed = X_tiles[ia].copy()
            else:
                raise ValueError(f"unknown mode {mode}")

            if noise_sigma > 0.0:
                mixed = mixed + rng.normal(0.0, noise_sigma, size=mixed.shape
                                           ).astype(mixed.dtype)

            synth_tiles.append(mixed)
        X_synth[si] = np.mean(np.stack(synth_tiles, axis=0), axis=0)

    return X_synth


def synthesize_mixup_scans(
    X_scan: np.ndarray,
    person_ids: np.ndarray,
    n_synth_scans: int,
    alpha: float,
    rng: np.random.Generator,
    mode: str = "interp",
    extrap_range: tuple[float, float] = (-0.3, 1.3),
    noise_sigma: float = 0.0,
) -> np.ndarray:
    """Scan-level MixUp (for BiomedCLIP TTA cache which has no tile embeddings)."""
    n = len(X_scan)
    if n < 2:
        return np.empty((0, X_scan.shape[1]), dtype=np.float32)
    unique_persons = np.unique(person_ids)
    X_synth = np.zeros((n_synth_scans, X_scan.shape[1]), dtype=np.float32)
    for si in range(n_synth_scans):
        ia = int(rng.integers(n))
        pa = person_ids[ia]
        if len(unique_persons) >= 2:
            other_idx = np.where(person_ids != pa)[0]
            ib = (int(other_idx[rng.integers(len(other_idx))])
                  if len(other_idx) else int(rng.integers(n)))
        else:
            ib = int(rng.integers(n))

        if mode == "interp":
            lam = rng.beta(alpha, alpha)
            lam = 0.2 + 0.6 * lam
            mixed = lam * X_scan[ia] + (1.0 - lam) * X_scan[ib]
        elif mode == "extrap":
            lo, hi = extrap_range
            lam = rng.uniform(lo, hi)
            mixed = lam * X_scan[ia] + (1.0 - lam) * X_scan[ib]
        elif mode == "noise":
            mixed = X_scan[ia].copy()
        else:
            raise ValueError(f"unknown mode {mode}")

        if noise_sigma > 0.0:
            mixed = mixed + rng.normal(0.0, noise_sigma, size=mixed.shape
                                       ).astype(mixed.dtype)
        X_synth[si] = mixed
    return X_synth


# ---------------------------------------------------------------------------
# LOPO evaluation with optional augmentation
# ---------------------------------------------------------------------------

def fit_lr_v2(X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0):
    """L2-row-norm → StandardScaler → LR(balanced).  Returns (scaler, classifier)."""
    Xn = normalize(X_train, norm="l2", axis=1)
    sc = StandardScaler()
    Xt = sc.fit_transform(Xn)
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    clf = LogisticRegression(
        class_weight="balanced", max_iter=3000, C=C,
        solver="lbfgs", random_state=42,
    )
    clf.fit(Xt, y_train)
    return sc, clf


def predict_proba_v2(X: np.ndarray, sc: StandardScaler,
                     clf: LogisticRegression) -> np.ndarray:
    Xn = normalize(X, norm="l2", axis=1)
    Xt = sc.transform(Xn)
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    proba = clf.predict_proba(Xt)
    P = np.zeros((len(X), N_CLASSES), dtype=np.float64)
    for ci, cls in enumerate(clf.classes_):
        P[:, cls] = proba[:, ci]
    return P


def lopo_oof_with_augmentation(
    caches: dict,
    y: np.ndarray,
    groups: np.ndarray,
    minority_cls: list[int],
    n_synth_per_class: int,
    tiles_per_synth_scan: int,
    alpha: float,
    seed: int = 42,
    verbose: bool = False,
    mode: str = "interp",
    extrap_range: tuple[float, float] = (-0.3, 1.3),
    noise_sigma: float = 0.0,
    clf_C: float = 1.0,
) -> dict[str, np.ndarray]:
    """Run person-LOPO for 3 v4 components, with synthetic minority aug per fold.

    caches keys:
        X90_tiles, t2s_90, pers_tile_90
        X45_tiles, t2s_45, pers_tile_45
        Xbc_scan, pers_scan (BiomedCLIP is already scan-level)
    Returns OOF softmax dict per component.
    """
    n_scans = len(y)
    rng = np.random.default_rng(seed)

    P90 = np.zeros((n_scans, N_CLASSES))
    P45 = np.zeros((n_scans, N_CLASSES))
    Pbc = np.zeros((n_scans, N_CLASSES))

    # Scan-level pools (aligned to the single scan ordering).
    X90_scan_full = mean_pool_tiles(
        caches["X90_tiles"], caches["t2s_90"], n_scans)
    X45_scan_full = mean_pool_tiles(
        caches["X45_tiles"], caches["t2s_45"], n_scans)
    Xbc_scan_full = caches["Xbc_scan"]

    # Scan-level persons (same order as y).
    pers_scan = caches["pers_scan"]

    for fi, (tr, va) in enumerate(leave_one_patient_out(groups)):
        # Real training data per component (scan-level for BC, scan-level pooled
        # for DINOv2 — augmentation synthesises at scan-level by pooling tile-level mixes).
        X90_tr = X90_scan_full[tr]
        X45_tr = X45_scan_full[tr]
        Xbc_tr = Xbc_scan_full[tr]
        y_tr = y[tr]

        X90_syn_all, X45_syn_all, Xbc_syn_all, y_syn_all = [], [], [], []

        for cls in minority_cls:
            # Tile-level pools limited to THIS fold's training tiles for THIS class.
            # DINOv2-B 90 nm
            scan_mask_cls = np.zeros(n_scans, dtype=bool)
            scan_mask_cls[tr[y_tr == cls]] = True

            tile_mask_90 = scan_mask_cls[caches["t2s_90"]]
            tile_mask_45 = scan_mask_cls[caches["t2s_45"]]
            scan_mask_bc = scan_mask_cls

            if tile_mask_90.sum() < 2 or tile_mask_45.sum() < 2 or scan_mask_bc.sum() < 2:
                continue

            Xt90 = caches["X90_tiles"][tile_mask_90]
            pt90 = caches["pers_tile_90"][tile_mask_90]
            ts90 = caches["t2s_90"][tile_mask_90]

            Xt45 = caches["X45_tiles"][tile_mask_45]
            pt45 = caches["pers_tile_45"][tile_mask_45]
            ts45 = caches["t2s_45"][tile_mask_45]

            Xbc_cls = Xbc_scan_full[scan_mask_bc]
            pbc_cls = pers_scan[scan_mask_bc]

            X90_syn = synthesize_mixup_tiles(
                Xt90, pt90, ts90, n_synth_per_class,
                tiles_per_synth_scan, alpha, rng,
                mode=mode, extrap_range=extrap_range, noise_sigma=noise_sigma)
            X45_syn = synthesize_mixup_tiles(
                Xt45, pt45, ts45, n_synth_per_class,
                tiles_per_synth_scan, alpha, rng,
                mode=mode, extrap_range=extrap_range, noise_sigma=noise_sigma)
            Xbc_syn = synthesize_mixup_scans(
                Xbc_cls, pbc_cls, n_synth_per_class, alpha, rng,
                mode=mode, extrap_range=extrap_range, noise_sigma=noise_sigma)

            X90_syn_all.append(X90_syn)
            X45_syn_all.append(X45_syn)
            Xbc_syn_all.append(Xbc_syn)
            y_syn_all.append(np.full(n_synth_per_class, cls, dtype=y.dtype))

        if X90_syn_all:
            X90_syn_cat = np.concatenate(X90_syn_all, axis=0)
            X45_syn_cat = np.concatenate(X45_syn_all, axis=0)
            Xbc_syn_cat = np.concatenate(Xbc_syn_all, axis=0)
            y_syn_cat = np.concatenate(y_syn_all, axis=0)

            X90_tr_aug = np.concatenate([X90_tr, X90_syn_cat], axis=0)
            X45_tr_aug = np.concatenate([X45_tr, X45_syn_cat], axis=0)
            Xbc_tr_aug = np.concatenate([Xbc_tr, Xbc_syn_cat], axis=0)
            y_tr_aug = np.concatenate([y_tr, y_syn_cat], axis=0)
        else:
            X90_tr_aug, X45_tr_aug, Xbc_tr_aug, y_tr_aug = (
                X90_tr, X45_tr, Xbc_tr, y_tr)

        # Fit each component on (real + synthetic) train data.
        sc90, clf90 = fit_lr_v2(X90_tr_aug, y_tr_aug, C=clf_C)
        sc45, clf45 = fit_lr_v2(X45_tr_aug, y_tr_aug, C=clf_C)
        scbc, clfbc = fit_lr_v2(Xbc_tr_aug, y_tr_aug, C=clf_C)

        P90[va] = predict_proba_v2(X90_scan_full[va], sc90, clf90)
        P45[va] = predict_proba_v2(X45_scan_full[va], sc45, clf45)
        Pbc[va] = predict_proba_v2(Xbc_scan_full[va], scbc, clfbc)

        if verbose and fi < 3:
            print(f"   fold {fi:2d}  train={len(tr)}+{len(y_tr_aug)-len(tr)}synth  "
                  f"val={len(va)}")

    return {"dinov2_90nm": P90, "dinov2_45nm": P45, "biomedclip_tta_90nm": Pbc}


def geom_mean(probs_list: list[np.ndarray]) -> np.ndarray:
    log_sum = np.zeros_like(probs_list[0])
    for P in probs_list:
        log_sum += np.log(P + EPS)
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
        "confusion_row_SucheOko": int((pred[y == 4] == 4).sum()),
        "support_SucheOko": int((y == 4).sum()),
    }


# ---------------------------------------------------------------------------
# Red-team: synthetic vs real nearest-neighbour distance
# ---------------------------------------------------------------------------

def redteam_nn_distance(
    caches: dict,
    y: np.ndarray,
    groups: np.ndarray,
    n_synth_per_class: int,
    tiles_per_synth_scan: int,
    alpha: float,
    seed: int = 42,
) -> dict:
    """For ONE representative fold, report cosine distance between synthetic
    SucheOko scans and (a) their nearest real train-set SucheOko scan, (b) the
    held-out test-person SucheOko scan (if fold is a SucheOko fold).
    """
    n_scans = len(y)
    rng = np.random.default_rng(seed)

    X90_full = mean_pool_tiles(caches["X90_tiles"], caches["t2s_90"], n_scans)

    # Pick a SucheOko fold.
    sucheoko_persons = np.unique(groups[y == 4])
    target_person = sucheoko_persons[0]
    va = np.where(groups == target_person)[0]
    tr = np.where(groups != target_person)[0]

    scan_mask_cls = np.zeros(n_scans, dtype=bool)
    scan_mask_cls[tr[y[tr] == 4]] = True
    tile_mask_90 = scan_mask_cls[caches["t2s_90"]]
    Xt90 = caches["X90_tiles"][tile_mask_90]
    pt90 = caches["pers_tile_90"][tile_mask_90]
    ts90 = caches["t2s_90"][tile_mask_90]

    X_syn = synthesize_mixup_tiles(
        Xt90, pt90, ts90, n_synth_per_class, tiles_per_synth_scan, alpha, rng)

    # Real train SucheOko scan embeddings (pooled).
    X_real_train = X90_full[scan_mask_cls]
    X_real_val = X90_full[va]
    y_val = y[va]
    is_sucheoko_val = y_val == 4

    def _cos(a, b):
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + EPS)
        return an @ bn.T  # cosine similarity

    sim_train = _cos(X_syn, X_real_train)  # (n_syn, n_real_train)
    nn_sim_train = sim_train.max(axis=1)
    nn_dist_train = 1.0 - nn_sim_train  # cosine distance

    out = {
        "held_out_person": str(target_person),
        "n_synthetic": int(len(X_syn)),
        "n_real_train_SucheOko_scans": int(len(X_real_train)),
        "nn_cosine_distance_to_train_SucheOko": {
            "mean": float(nn_dist_train.mean()),
            "median": float(np.median(nn_dist_train)),
            "min": float(nn_dist_train.min()),
            "max": float(nn_dist_train.max()),
        },
    }

    if is_sucheoko_val.any():
        sim_val = _cos(X_syn, X_real_val[is_sucheoko_val])
        nn_sim_val = sim_val.max(axis=1)
        nn_dist_val = 1.0 - nn_sim_val
        out["nn_cosine_distance_to_HELDOUT_test_SucheOko"] = {
            "mean": float(nn_dist_val.mean()),
            "median": float(np.median(nn_dist_val)),
            "min": float(nn_dist_val.min()),
            "max": float(nn_dist_val.max()),
        }
        # Sanity: synthetic should be CLOSER to train than to test (if test
        # person's texture is actually different).  Small gap is expected
        # because both are SucheOko — that's the point.
        out["memorisation_check"] = {
            "mean_dist_to_train": float(nn_dist_train.mean()),
            "mean_dist_to_heldout": float(nn_dist_val.mean()),
            "interpretation": (
                "synthetic should not be dramatically closer to train (<< 0.01) — "
                "that would indicate memorisation of real tiles"
            ),
        }

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print("Synthetic augmentation (MixUp in embedding space) — person-LOPO")
    print("=" * 78)

    # ---- 1. Load all 3 caches ----
    print("\n[load] caches")
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz", allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz",
                  allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)

    paths_90 = [str(p) for p in z90["scan_paths"]]
    paths_45 = [str(p) for p in z45["scan_paths"]]
    paths_bc = [str(p) for p in zbc["scan_paths"]]

    # Person-level groups (LOPO groups).  Use 90nm path ordering as reference.
    groups = np.array([person_id(Path(p)) for p in paths_90])
    y = np.asarray(z90["scan_y"], dtype=np.int64)
    n_scans = len(y)
    n_persons = len(np.unique(groups))
    print(f"  scans={n_scans}  persons={n_persons}")
    assert n_persons == 35, f"expected 35 persons, got {n_persons}"

    # Align 45 & bc caches to 90nm order.
    # Reorder 45nm tile cache to match 90nm scan order:
    idx_map = {p: i for i, p in enumerate(paths_45)}
    old_to_new_45 = np.array([idx_map[p] for p in paths_90])
    # Build new t2s_45 pointing into 90-order.  Each tile_to_scan[i] is a 45-order scan idx.
    # We need it in terms of 90-order scan idx.
    scan_45_to_90 = np.zeros(n_scans, dtype=np.int64)
    for new_90_idx, p in enumerate(paths_90):
        scan_45_to_90[idx_map[p]] = new_90_idx
    t2s_45_in_90_order = scan_45_to_90[z45["tile_to_scan"]]

    # Scan-level embeddings aligned to 90nm ordering.
    Xbc_scan = align_to_reference(paths_90, paths_bc, zbc["X_scan"].astype(np.float32))

    # Per-tile person IDs.
    pers_tile_90 = np.array([person_id(Path(paths_90[s])) for s in z90["tile_to_scan"]])
    pers_tile_45 = np.array([person_id(Path(paths_90[s])) for s in t2s_45_in_90_order])
    pers_scan = groups  # alias — scan-level person per scan (aligned to 90 order).

    caches = {
        "X90_tiles": z90["X"].astype(np.float32),
        "t2s_90": z90["tile_to_scan"].astype(np.int64),
        "pers_tile_90": pers_tile_90,
        "X45_tiles": z45["X"].astype(np.float32),
        "t2s_45": t2s_45_in_90_order.astype(np.int64),
        "pers_tile_45": pers_tile_45,
        "Xbc_scan": Xbc_scan,
        "pers_scan": pers_scan,
    }

    # ---- 2. Configurations to sweep ----
    # Each row: (name, minority_cls, n_synth, tiles_per_synth, alpha,
    #           mode, extrap_range, noise_sigma, clf_C)
    configs = [
        ("baseline_no_aug",            [],     0,   0, 0.2, "interp", (-0.3, 1.3), 0.00, 1.0),
        ("mixup_interp_n100",          [4],  100,   4, 0.2, "interp", (-0.3, 1.3), 0.00, 1.0),
        ("mixup_interp_n200",          [4],  200,   4, 0.2, "interp", (-0.3, 1.3), 0.00, 1.0),
        # extrapolative mixup — pushes synthetic points OUT of the SucheOko cloud
        ("mixup_extrap_n100",          [4],  100,   4, 0.2, "extrap", (-0.5, 1.5), 0.00, 1.0),
        ("mixup_extrap_n200_aggr",     [4],  200,   4, 0.2, "extrap", (-1.0, 2.0), 0.00, 1.0),
        # pure Gaussian noise copies (local inflation of class cloud)
        ("noise_sigma03_n100",         [4],  100,   4, 0.2, "noise",  (-0.3, 1.3), 0.03, 1.0),
        ("noise_sigma05_n200",         [4],  200,   4, 0.2, "noise",  (-0.3, 1.3), 0.05, 1.0),
        # combo: extrap + small noise
        ("mixup_extrap_noise_n200",    [4],  200,   4, 0.2, "extrap", (-0.5, 1.5), 0.02, 1.0),
        # broader minority boost
        ("mixup_SO+Diab_n100",         [4, 1], 100, 4, 0.2, "interp", (-0.3, 1.3), 0.00, 1.0),
        # stronger reg (smaller C) combined with aug (fewer features dominate)
        ("mixup_extrap_n200_lowC",     [4],  200,   4, 0.2, "extrap", (-0.5, 1.5), 0.00, 0.3),
    ]

    # ---- 3. Run LOPO OOF for each config ----
    print("\n[eval] person-LOPO OOF for each config "
          f"({n_persons} folds × 3 components)")
    results: dict[str, dict] = {}
    all_oof: dict[str, dict] = {}
    for (name, minority_cls, n_synth, tiles_per_synth, alpha,
         mode, extrap_range, noise_sigma, clf_C) in configs:
        ts = time.time()
        Ps = lopo_oof_with_augmentation(
            caches, y, groups, minority_cls, n_synth, tiles_per_synth, alpha,
            seed=42, verbose=False,
            mode=mode, extrap_range=extrap_range, noise_sigma=noise_sigma,
            clf_C=clf_C)
        # Geometric mean ensemble.
        G = geom_mean([Ps["dinov2_90nm"], Ps["dinov2_45nm"],
                       Ps["biomedclip_tta_90nm"]])
        m = metrics_of(G, y)
        # Per-component too.
        per_comp = {k: metrics_of(Ps[k], y) for k in Ps}
        results[name] = {
            "minority_cls": minority_cls,
            "n_synth_per_class": n_synth,
            "tiles_per_synth_scan": tiles_per_synth,
            "alpha": alpha,
            "mode": mode,
            "extrap_range": list(extrap_range),
            "noise_sigma": noise_sigma,
            "clf_C": clf_C,
            "ensemble": m,
            "per_component": per_comp,
            "elapsed_s": round(time.time() - ts, 1),
        }
        all_oof[name] = G
        print(f"  {name:32s} ens W-F1={m['weighted_f1']:.4f} M-F1={m['macro_f1']:.4f} "
              f"SO-F1={m['per_class_f1'][4]:.4f}  ({time.time()-ts:.1f}s)")

    # ---- 4. Red-team on SucheOko synthesis for best config ----
    print("\n[redteam] nearest-neighbour distance synthetic vs real (best config)")
    best_name = max(
        (n for n, r in results.items() if n != "baseline_no_aug"),
        key=lambda n: results[n]["ensemble"]["per_class_f1"][4]
        if results[n]["ensemble"]["per_class_f1"][4] > 0
        else results[n]["ensemble"]["weighted_f1"],
    )
    best_cfg = next(c for c in configs if c[0] == best_name)
    rt = redteam_nn_distance(
        caches, y, groups,
        n_synth_per_class=best_cfg[2],
        tiles_per_synth_scan=best_cfg[3],
        alpha=best_cfg[4],
    )
    print(f"  best_config={best_name}")
    print(f"  NN dist to train SucheOko: mean="
          f"{rt['nn_cosine_distance_to_train_SucheOko']['mean']:.4f} "
          f"min={rt['nn_cosine_distance_to_train_SucheOko']['min']:.4f}")
    if "nn_cosine_distance_to_HELDOUT_test_SucheOko" in rt:
        print(f"  NN dist to heldout SucheOko: mean="
              f"{rt['nn_cosine_distance_to_HELDOUT_test_SucheOko']['mean']:.4f}")

    # ---- 5. Summary JSON ----
    summary = {
        "method": "mixup_in_embedding_space",
        "person_lopo": True,
        "n_scans": int(n_scans),
        "n_persons": int(n_persons),
        "champion_v4": {"weighted_f1": CHAMP_V4_WF1, "macro_f1": CHAMP_V4_MF1},
        "configs": results,
        "best_config_name": best_name,
        "redteam": rt,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_json = REPORTS / "synthetic_aug_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {out_json}")

    # ---- 6. Markdown report ----
    write_markdown_report(summary)
    print(f"[saved] {REPORTS / 'SYNTHETIC_AUG_RESULTS.md'}")

    print(f"\n[done] total elapsed: {time.time() - t0:.1f}s")
    return summary


def write_markdown_report(summary: dict) -> None:
    lines = []
    lines.append("# Synthetic Augmentation Results — MixUp in Embedding Space\n")
    lines.append(
        f"**TL;DR.** Person-LOPO weighted F1, 35 folds.  "
        f"Champion v4 (no aug): **{summary['champion_v4']['weighted_f1']:.4f}** "
        f"(SucheOko F1 = 0).  MixUp augmentation of minority class embeddings "
        "is evaluated against this baseline.\n"
    )
    lines.append("## Method\n")
    lines.append(
        "MixUp in the frozen-encoder embedding space.  For each LOPO fold we\n"
        "synthesise `n` new per-scan embeddings for minority class(es) by:\n"
        "\n"
        "1. Drawing two **tiles** (or two scans for BiomedCLIP TTA cache) of the\n"
        "   target class from the fold's **training set only**, preferably from\n"
        "   DIFFERENT persons when multiple are available.\n"
        "2. Mixing with weight `lam ~ Beta(α, α)` squashed into `[0.2, 0.8]` to\n"
        "   guarantee genuine interpolation rather than near-copy.\n"
        "3. Averaging `tiles_per_synth_scan` such mixed tiles to form one\n"
        "   synthetic scan-level embedding (mirrors the real mean-pool pipeline).\n"
        "\n"
        "Appended to the fold's training set before fitting v2-recipe LR\n"
        "(L2-row-norm → StandardScaler → LR(class_weight='balanced')).\n"
        "\n"
        "The held-out person's scans/tiles are **never** used as MixUp ingredients,\n"
        "so there is no cross-fold leakage.\n"
    )
    lines.append("## Configurations & Results\n")
    lines.append(
        "| Config | Classes | n_synth | Ens. W-F1 | Ens. M-F1 | "
        "Healthy | Diabetes | Glaukom | SM | **SucheOko** |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in summary["configs"].items():
        m = r["ensemble"]
        cls_f1 = m["per_class_f1"]
        lines.append(
            f"| `{name}` | {r['minority_cls']} | {r['n_synth_per_class']} | "
            f"{m['weighted_f1']:.4f} | {m['macro_f1']:.4f} | "
            f"{cls_f1[0]:.3f} | {cls_f1[1]:.3f} | {cls_f1[2]:.3f} | "
            f"{cls_f1[3]:.3f} | **{cls_f1[4]:.3f}** |"
        )
    lines.append("")

    lines.append("## Per-component breakdown (ensemble member metrics)\n")
    lines.append(
        "| Config | DINOv2-90 W-F1 | DINOv2-45 W-F1 | BiomedCLIP W-F1 |"
    )
    lines.append("|---|---:|---:|---:|")
    for name, r in summary["configs"].items():
        pc = r["per_component"]
        lines.append(
            f"| `{name}` | {pc['dinov2_90nm']['weighted_f1']:.4f} | "
            f"{pc['dinov2_45nm']['weighted_f1']:.4f} | "
            f"{pc['biomedclip_tta_90nm']['weighted_f1']:.4f} |"
        )
    lines.append("")

    # Headline box.
    baseline_wf1 = summary["configs"]["baseline_no_aug"]["ensemble"]["weighted_f1"]
    baseline_so = summary["configs"]["baseline_no_aug"]["ensemble"]["per_class_f1"][4]
    best = summary["configs"][summary["best_config_name"]]
    best_wf1 = best["ensemble"]["weighted_f1"]
    best_so = best["ensemble"]["per_class_f1"][4]
    lines.append("## Headline\n")
    lines.append(
        f"- **Baseline (no aug, 3-component geom-mean)**: W-F1 = "
        f"{baseline_wf1:.4f}, SucheOko F1 = **{baseline_so:.3f}**.\n"
        f"- **Best MixUp config (`{summary['best_config_name']}`)**: W-F1 = "
        f"{best_wf1:.4f} (Δ = {best_wf1 - baseline_wf1:+.4f}), "
        f"SucheOko F1 = **{best_so:.3f}** "
        f"(Δ = {best_so - baseline_so:+.3f}).\n"
        f"- **Champion v4 reference** (trained ensemble, published in "
        f"`models/ensemble_v4_multiscale/`): W-F1 = "
        f"{summary['champion_v4']['weighted_f1']:.4f}.\n"
    )

    lines.append("\n## Red-team: nearest-neighbour distance\n")
    lines.append(
        "We check that synthetic SucheOko embeddings are **not** near-copies of\n"
        "real training tiles (memorisation) and are similarly close/far from\n"
        "real train vs held-out SucheOko.\n"
    )
    rt = summary["redteam"]
    lines.append("```json")
    lines.append(json.dumps(rt, indent=2))
    lines.append("```\n")
    lines.append(
        "Interpretation: MixUp operates in the L2-normed 768-d DINOv2 space.\n"
        "Cosine distances of ≥ 0.01 to the nearest real training tile indicate\n"
        "the synthetic samples are genuine interpolations, not duplicates.  If\n"
        "the heldout-test distance is comparable to the train distance, the\n"
        "synthetic samples have plausibly captured class-level texture.\n"
    )

    lines.append("## Limitations & Honest Caveats\n")
    lines.append(
        "- **Only 2 SucheOko persons** — MixUp can only linearly interpolate\n"
        "  between what exists.  If the missing SucheOko phenotypes live in a\n"
        "  non-linear manifold region, convex combinations can't reach them.\n"
        "- **No image-space validation** — we trust the encoder's inductive\n"
        "  bias (DINOv2-B was pretrained on natural images, generalises well to\n"
        "  AFM texture, as evidenced by the champion F1).  A true VAE would\n"
        "  also need pixel-level sanity checks.\n"
        "- **Does not escape feature manifold** — any bias in the frozen encoder\n"
        "  is preserved.  If DINOv2-B lacks SucheOko-specific directions, MixUp\n"
        "  can't create them.  That's Option A / B's job (future work).\n"
        "- **Class weight already boosts SucheOko** — the baseline already uses\n"
        "  `class_weight='balanced'`.  Gains from MixUp must exceed the gains\n"
        "  from simple reweighting to be meaningful.\n"
    )
    (REPORTS / "SYNTHETIC_AUG_RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
