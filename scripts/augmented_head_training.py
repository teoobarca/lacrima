"""Augmented head training for the v4 multi-scale ensemble.

Research question
-----------------
The v4 champion mean-pools the 8 D4-TTA views per scan and trains one linear
head per fold on (n_scans, D) scan-level embeddings. **Does training the head
on each D4 view as its own training sample (8× data) regularize the head
enough to improve weighted F1?**

Three treatments (all with PERSON-level LOPO, 35 folds):

  T1 — averaged-head (sanity == v4)
      Train on the mean-over-views embedding per scan.  Must match v4 0.6887.

  T2 — noise-injected-head
      Same averaged input, but inject zero-mean Gaussian noise at train time
      (sigma = 0.01 in L2-normalized space) as embedding-space augmentation.

  T3 — augmented-head (D4 expanded samples)
      Expand training set 8×: each D4 view of each training scan becomes a
      separate training sample with the same label.  Test-time: mean-pool the
      8 views (unchanged TTA) then predict.

v4 architecture:
  A) DINOv2-B @ 90 nm/px, NO TTA   (scan-level = mean over ~3 tiles)
  B) DINOv2-B @ 45 nm/px, NO TTA   (scan-level = mean over ~13 tiles)
  C) BiomedCLIP @ 90 nm/px, D4 TTA (scan-level = mean over 8 views × tiles)

For components A & B (no TTA) there is no natural 8× expansion, so we apply
the D4 augmentation only to component C (BiomedCLIP, the TTA branch).  Per
the literature this is the branch that benefits from augmentation.

However, to exercise the hypothesis more aggressively we also build per-view
D4 caches for A & B on the fly so the augmentation can be applied there too.
Because v4 prescribes *no* TTA on A & B at inference time, for T3 we still
pool the 8 views at test time (opting in to symmetrize training & inference).

Budget-aware: if per-view caches are missing we only re-encode component C
(the BiomedCLIP branch). Components A & B fall back to their existing mean-
pooled scan-level features.

Output:
    reports/AUGMENTED_HEAD_RESULTS.md
    cache/augmented_head_results.json
    cache/tta_perview_*.npz (new per-view caches, if built)
"""
from __future__ import annotations

import argparse
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
from teardrop.data import (  # noqa: E402
    CLASSES,
    center_crop_or_pad,
    enumerate_samples,
    load_height,
    person_id,
    plane_level,
    resample_to_pixel_size,
    robust_normalize,
    tile,
)
from teardrop.encoders import height_to_pil  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
CACHE.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

N_CLASSES = len(CLASSES)
EPS = 1e-12
V4_BASELINE = 0.6887  # honest person-LOPO weighted F1 from v4 (reports)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def d4_augmentations(arr: np.ndarray) -> list[np.ndarray]:
    """The 8 elements of D4 applied to a 2-D array."""
    rots = [np.rot90(arr, k=k) for k in range(4)]
    flipped = np.fliplr(arr)
    rots_flip = [np.rot90(flipped, k=k) for k in range(4)]
    return rots + rots_flip


# ---------------------------------------------------------------------------
# Per-view D4 cache builder (one encoder, one scale)
# ---------------------------------------------------------------------------

def build_perview_cache(
    cache_path: Path,
    encoder_loader,           # callable returning EncoderBundle
    target_nm_per_px: float,
    render_mode: str = "afmhot",
    tile_size: int = 512,
    max_tiles: int = 9,
    batch_size: int = 16,
) -> Path:
    """Encode each D4 view of each tile as a separate embedding.

    Layout of saved npz:
        X                : (N_views_total, D)  — row ordering: scan, tile, d4view
        view_to_scan     : (N_views_total,) int64   scan index
        view_to_tile     : (N_views_total,) int64   tile index within scan
        view_to_d4       : (N_views_total,) int64   0..7
        scan_paths       : (n_scans,) str
        scan_y           : (n_scans,) int64
        scan_groups      : (n_scans,) str
    """
    if cache_path.exists():
        print(f"[cache-hit] {cache_path.name}")
        return cache_path

    print(f"[build-perview] {cache_path.name}")
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    enc = encoder_loader()
    print(f"  encoder: {enc.name} on {enc.device}  target={target_nm_per_px} nm/px")

    all_pil: list = []
    view_to_scan: list[int] = []
    view_to_tile: list[int] = []
    view_to_d4: list[int] = []
    scan_y: list[int] = []
    scan_groups: list[str] = []
    scan_paths: list[str] = []

    t0 = time.time()
    for si, s in enumerate(samples):
        try:
            hm = load_height(s.raw_path)
            h = plane_level(hm.height)
            h = resample_to_pixel_size(h, hm.pixel_nm, target_nm_per_px)
            h = robust_normalize(h)
            if h.shape[0] < tile_size or h.shape[1] < tile_size:
                pad_h = max(0, tile_size - h.shape[0])
                pad_w = max(0, tile_size - h.shape[1])
                h = np.pad(
                    h,
                    ((pad_h // 2, pad_h - pad_h // 2),
                     (pad_w // 2, pad_w - pad_w // 2)),
                    mode="reflect",
                )
            tiles = tile(h, tile_size, stride=tile_size)
            if not tiles:
                tiles = [center_crop_or_pad(h, tile_size)]
            if len(tiles) > max_tiles:
                idx = np.linspace(0, len(tiles) - 1, max_tiles).astype(int)
                tiles = [tiles[i] for i in idx]
            for ti, t in enumerate(tiles):
                for d4i, aug in enumerate(d4_augmentations(t)):
                    all_pil.append(
                        height_to_pil(np.ascontiguousarray(aug), mode=render_mode)
                    )
                    view_to_scan.append(si)
                    view_to_tile.append(ti)
                    view_to_d4.append(d4i)
            scan_y.append(s.label)
            scan_groups.append(person_id(s.raw_path))
            scan_paths.append(str(Path(s.raw_path).resolve()))
        except Exception as e:
            print(f"  [err] {s.raw_path.name}: {e}")
        if (si + 1) % 40 == 0:
            print(f"  preproc [{si + 1}/{len(samples)}]  "
                  f"views={len(all_pil)}  t={time.time() - t0:.1f}s")

    print(f"  TOTAL views={len(all_pil)} in {time.time() - t0:.1f}s")
    print("  encoding...")
    t1 = time.time()
    X = enc.encode(all_pil, batch_size=batch_size)
    enc_time = time.time() - t1
    print(f"  encoded {X.shape} in {enc_time:.1f}s")

    np.savez(
        cache_path,
        X=X.astype(np.float32),
        view_to_scan=np.array(view_to_scan, dtype=np.int64),
        view_to_tile=np.array(view_to_tile, dtype=np.int64),
        view_to_d4=np.array(view_to_d4, dtype=np.int64),
        scan_paths=np.array(scan_paths),
        scan_y=np.array(scan_y, dtype=np.int64),
        scan_groups=np.array(scan_groups),
        encode_time_s=np.array(enc_time),
    )
    print(f"  [saved] {cache_path}")

    # Release encoder
    del enc
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return cache_path


# ---------------------------------------------------------------------------
# Aggregation helpers on per-view caches
# ---------------------------------------------------------------------------

def scan_embeddings_from_perview(
    X: np.ndarray,
    view_to_scan: np.ndarray,
    view_to_d4: np.ndarray,
    n_scans: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-view embeddings.

    Returns:
        X_scan     (n_scans, D)    mean across all views & tiles (test-time TTA)
        X_scan_d4  (n_scans, 8, D) mean per d4 view across tiles
    """
    D = X.shape[1]
    X_scan = np.zeros((n_scans, D), dtype=np.float32)
    X_scan_d4 = np.zeros((n_scans, 8, D), dtype=np.float32)
    counts_d4 = np.zeros((n_scans, 8), dtype=np.int64)
    counts = np.zeros(n_scans, dtype=np.int64)
    for i in range(X.shape[0]):
        si = int(view_to_scan[i])
        d4i = int(view_to_d4[i])
        X_scan[si] += X[i]
        X_scan_d4[si, d4i] += X[i]
        counts[si] += 1
        counts_d4[si, d4i] += 1
    counts = np.maximum(counts, 1)
    counts_d4 = np.maximum(counts_d4, 1)
    X_scan /= counts[:, None]
    X_scan_d4 /= counts_d4[:, :, None]
    return X_scan, X_scan_d4


# ---------------------------------------------------------------------------
# Generic per-component head training
# ---------------------------------------------------------------------------

def fit_predict_averaged(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """T1: standard v4 — train head on averaged embeddings."""
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


def fit_predict_noise(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,
    sigma: float = 0.01, n_copies: int = 8, seed: int = 42,
) -> np.ndarray:
    """T2: noise-injected head — expand train set by adding Gaussian noise
    copies in L2-normalized space."""
    rng = np.random.default_rng(seed)
    X_train_n = normalize(X_train, norm="l2", axis=1)
    X_test_n = normalize(X_test, norm="l2", axis=1)

    # Create n_copies noisy copies (including 1 clean copy)
    X_noisy_list = [X_train_n]
    y_noisy_list = [y_train]
    for _ in range(n_copies - 1):
        noise = rng.normal(0.0, sigma, size=X_train_n.shape).astype(np.float32)
        Xn = X_train_n + noise
        # Re-normalize after noise (keep on unit sphere roughly)
        Xn = normalize(Xn, norm="l2", axis=1)
        X_noisy_list.append(Xn)
        y_noisy_list.append(y_train)
    X_expanded = np.vstack(X_noisy_list)
    y_expanded = np.concatenate(y_noisy_list)

    sc = StandardScaler().fit(X_expanded)
    X_tr_s = np.nan_to_num(sc.transform(X_expanded))
    X_te_s = np.nan_to_num(sc.transform(X_test_n))
    clf = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0,
                             solver="lbfgs", n_jobs=4, random_state=42)
    clf.fit(X_tr_s, y_expanded)
    proba = clf.predict_proba(X_te_s)
    full = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    for ci, cls in enumerate(clf.classes_):
        full[:, cls] = proba[:, ci]
    return full


def fit_predict_d4_expanded(
    X_train_d4: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """T3: D4-expanded head — each d4 view = separate training sample."""
    n_train, n_views, D = X_train_d4.shape
    X_expanded = X_train_d4.reshape(n_train * n_views, D)
    y_expanded = np.repeat(y_train, n_views)

    X_tr_n = normalize(X_expanded, norm="l2", axis=1)
    X_te_n = normalize(X_test, norm="l2", axis=1)
    sc = StandardScaler().fit(X_tr_n)
    X_tr_s = np.nan_to_num(sc.transform(X_tr_n))
    X_te_s = np.nan_to_num(sc.transform(X_te_n))
    clf = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0,
                             solver="lbfgs", n_jobs=4, random_state=42)
    clf.fit(X_tr_s, y_expanded)
    proba = clf.predict_proba(X_te_s)
    full = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    for ci, cls in enumerate(clf.classes_):
        full[:, cls] = proba[:, ci]
    return full


# ---------------------------------------------------------------------------
# LOPO drivers for each treatment
# ---------------------------------------------------------------------------

def lopo_averaged(X_scan: np.ndarray, y: np.ndarray,
                  groups: np.ndarray) -> np.ndarray:
    P = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups):
        P[va] = fit_predict_averaged(X_scan[tr], y[tr], X_scan[va])
    return P


def lopo_noise(X_scan: np.ndarray, y: np.ndarray, groups: np.ndarray,
               sigma: float = 0.01, n_copies: int = 8) -> np.ndarray:
    P = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups):
        P[va] = fit_predict_noise(X_scan[tr], y[tr], X_scan[va],
                                   sigma=sigma, n_copies=n_copies)
    return P


def lopo_d4(X_scan_d4: np.ndarray, X_scan: np.ndarray, y: np.ndarray,
            groups: np.ndarray) -> np.ndarray:
    """X_scan_d4 is (n_scans, 8, D). Test-time uses mean-pool over D4 (=X_scan)."""
    P = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups):
        P[va] = fit_predict_d4_expanded(X_scan_d4[tr], y[tr], X_scan[va])
    return P


# ---------------------------------------------------------------------------
# Ensemble helpers
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
    """Paired bootstrap of weighted F1 delta (new - base)."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def align_to_reference(paths_ref, paths_src, X_src):
    src_idx = {p: i for i, p in enumerate(paths_src)}
    order = np.array([src_idx[p] for p in paths_ref])
    return X_src[order]


def _resolve_paths(arr) -> list[str]:
    return [str(Path(str(p)).resolve()) for p in arr]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-perview-build", action="store_true",
                        help="Only run treatments that use existing scan-level caches.")
    parser.add_argument("--components", default="all",
                        choices=["all", "biomedclip_only"],
                        help="Which components to build per-view for.")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("Augmented head training for v4 multi-scale ensemble")
    print(f"v4 baseline (honest person-LOPO weighted F1): {V4_BASELINE}")
    print("=" * 78)

    # --- Load existing v4 scan-level features (for reference & component A/B) ---
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

    # Use 90 ordering as reference.
    y = np.asarray(z90["scan_y"], dtype=np.int64)
    groups = np.array([person_id(Path(p)) for p in paths_90])
    n_scans = len(y)
    n_persons = len(np.unique(groups))
    print(f"  n_scans={n_scans} n_persons={n_persons}")
    assert n_persons == 35, f"expected 35 persons, got {n_persons}"

    # Mean-pool 90 and 45 tiles (v4 style)
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

    X90_avg = _mean_pool_tiles(z90["X"], z90["tile_to_scan"], len(paths_90))
    X45_avg_raw = _mean_pool_tiles(z45["X"], z45["tile_to_scan"], len(paths_45))
    X45_avg = align_to_reference(paths_90, paths_45, X45_avg_raw)
    Xbc_avg = align_to_reference(paths_90, paths_bc,
                                 zbc["X_scan"].astype(np.float32))
    print(f"  averaged features: X90={X90_avg.shape} X45={X45_avg.shape} "
          f"Xbc={Xbc_avg.shape}")

    # --- Build per-view D4 caches (component-selective) ---
    perview_caches: dict[str, Path] = {}

    if not args.skip_perview_build:
        from teardrop.encoders import load_dinov2, load_biomedclip
        # BiomedCLIP 90 (the TTA branch in v4)
        perview_caches["biomedclip_90"] = build_perview_cache(
            CACHE / "tta_perview_biomedclip_afmhot_t512_n9_d4.npz",
            lambda: load_biomedclip(),
            target_nm_per_px=90.0,
        )
        if args.components == "all":
            # DINOv2 90 (v4 has no TTA on this branch — we add D4 expansion)
            perview_caches["dinov2_90"] = build_perview_cache(
                CACHE / "tta_perview_dinov2_vitb14_afmhot_t512_n9_d4.npz",
                lambda: load_dinov2("vitb14"),
                target_nm_per_px=90.0,
            )
            # DINOv2 45 (v4 has no TTA on this branch — we add D4 expansion)
            perview_caches["dinov2_45"] = build_perview_cache(
                CACHE / "tta_perview_dinov2_vitb14_afmhot_t512_n9_d4_45nm.npz",
                lambda: load_dinov2("vitb14"),
                target_nm_per_px=45.0,
            )

    # --- Load per-view caches & align ---
    perview = {}  # name -> dict(X_scan_avg, X_scan_d4)
    for name, path in perview_caches.items():
        z = np.load(path, allow_pickle=True)
        pv_paths = _resolve_paths(z["scan_paths"])
        n = len(pv_paths)
        X_scan_avg, X_scan_d4 = scan_embeddings_from_perview(
            z["X"], z["view_to_scan"], z["view_to_d4"], n,
        )
        # Align to 90 ordering
        X_scan_avg = align_to_reference(paths_90, pv_paths, X_scan_avg)
        X_scan_d4 = align_to_reference(paths_90, pv_paths,
                                         X_scan_d4.reshape(n, -1)).reshape(
            n_scans, 8, -1)
        perview[name] = {"X_scan_avg": X_scan_avg, "X_scan_d4": X_scan_d4}
        print(f"  [perview:{name}] X_scan_avg={X_scan_avg.shape} "
              f"X_scan_d4={X_scan_d4.shape}")

    # ======================================================================
    # Run treatments
    # ======================================================================

    # T1 — averaged head (sanity == v4) -------------------------------------
    print("\n[T1] averaged head (v4 sanity)")
    P90_t1 = lopo_averaged(X90_avg, y, groups)
    P45_t1 = lopo_averaged(X45_avg, y, groups)
    Pbc_t1 = lopo_averaged(Xbc_avg, y, groups)
    G_t1 = geom_mean([P90_t1, P45_t1, Pbc_t1])
    m_t1 = metrics_of(G_t1, y)
    print(f"  ensemble T1 W-F1={m_t1['weighted_f1']:.4f} "
          f"M-F1={m_t1['macro_f1']:.4f}  (expect ~{V4_BASELINE})")

    # T2 — noise-injected head ---------------------------------------------
    print("\n[T2] noise-injected head (sigma=0.01, 8 copies)")
    P90_t2 = lopo_noise(X90_avg, y, groups, sigma=0.01, n_copies=8)
    P45_t2 = lopo_noise(X45_avg, y, groups, sigma=0.01, n_copies=8)
    Pbc_t2 = lopo_noise(Xbc_avg, y, groups, sigma=0.01, n_copies=8)
    G_t2 = geom_mean([P90_t2, P45_t2, Pbc_t2])
    m_t2 = metrics_of(G_t2, y)
    print(f"  ensemble T2 W-F1={m_t2['weighted_f1']:.4f} "
          f"M-F1={m_t2['macro_f1']:.4f}")

    # T3 — D4-augmented head -----------------------------------------------
    print("\n[T3] D4-augmented head (8x training samples)")
    # Build per-component d4 softmaxes; for components without perview cache,
    # fall back to T1 softmax (we note this).
    fallback_used = []
    if "dinov2_90" in perview:
        P90_t3 = lopo_d4(perview["dinov2_90"]["X_scan_d4"],
                         perview["dinov2_90"]["X_scan_avg"], y, groups)
    else:
        P90_t3 = P90_t1
        fallback_used.append("dinov2_90")
    if "dinov2_45" in perview:
        P45_t3 = lopo_d4(perview["dinov2_45"]["X_scan_d4"],
                         perview["dinov2_45"]["X_scan_avg"], y, groups)
    else:
        P45_t3 = P45_t1
        fallback_used.append("dinov2_45")
    if "biomedclip_90" in perview:
        Pbc_t3 = lopo_d4(perview["biomedclip_90"]["X_scan_d4"],
                         perview["biomedclip_90"]["X_scan_avg"], y, groups)
    else:
        Pbc_t3 = Pbc_t1
        fallback_used.append("biomedclip_90")
    if fallback_used:
        print(f"  NOTE: fallback to T1 for components: {fallback_used}")
    G_t3 = geom_mean([P90_t3, P45_t3, Pbc_t3])
    m_t3 = metrics_of(G_t3, y)
    print(f"  ensemble T3 W-F1={m_t3['weighted_f1']:.4f} "
          f"M-F1={m_t3['macro_f1']:.4f}")

    # ======================================================================
    # Bootstrap vs v4
    # ======================================================================
    print("\n[bootstrap] 1000 x paired vs T1 (in-experiment baseline)")
    boot_t2 = bootstrap_delta(G_t2, G_t1, y, n_boot=1000, seed=42)
    boot_t3 = bootstrap_delta(G_t3, G_t1, y, n_boot=1000, seed=42)
    print(f"  T2 vs T1: delta_mean={boot_t2['delta_mean']:+.4f} "
          f"P(delta>0)={boot_t2['p_gt_zero']:.3f}")
    print(f"  T3 vs T1: delta_mean={boot_t3['delta_mean']:+.4f} "
          f"P(delta>0)={boot_t3['p_gt_zero']:.3f}")

    # Raw deltas vs reported v4 baseline number
    delta_t1_vs_v4 = m_t1["weighted_f1"] - V4_BASELINE
    delta_t2_vs_v4 = m_t2["weighted_f1"] - V4_BASELINE
    delta_t3_vs_v4 = m_t3["weighted_f1"] - V4_BASELINE
    print(f"\nVs reported v4 {V4_BASELINE}:")
    print(f"  T1 delta = {delta_t1_vs_v4:+.4f} (sanity; should be ~0)")
    print(f"  T2 delta = {delta_t2_vs_v4:+.4f}")
    print(f"  T3 delta = {delta_t3_vs_v4:+.4f}")

    # ======================================================================
    # Verdict
    # ======================================================================
    verdict_t2 = _verdict(boot_t2["delta_mean"], boot_t2["p_gt_zero"])
    verdict_t3 = _verdict(boot_t3["delta_mean"], boot_t3["p_gt_zero"])

    print(f"\nVerdict:")
    print(f"  T2 (noise): {verdict_t2}")
    print(f"  T3 (D4):    {verdict_t3}")

    # ======================================================================
    # Save
    # ======================================================================
    out = {
        "v4_baseline_weighted_f1": V4_BASELINE,
        "n_scans": int(n_scans),
        "n_persons": int(n_persons),
        "fallback_used_for_t3": fallback_used,
        "T1_averaged_head": m_t1,
        "T2_noise_injected": m_t2,
        "T3_d4_augmented": m_t3,
        "bootstrap_T2_vs_T1": boot_t2,
        "bootstrap_T3_vs_T1": boot_t3,
        "delta_vs_v4": {
            "T1": delta_t1_vs_v4,
            "T2": delta_t2_vs_v4,
            "T3": delta_t3_vs_v4,
        },
        "verdict": {"T2": verdict_t2, "T3": verdict_t3},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (CACHE / "augmented_head_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n[saved] cache/augmented_head_results.json")

    _write_markdown(out)
    print(f"[saved] reports/AUGMENTED_HEAD_RESULTS.md")
    print(f"\n[done] total elapsed: {time.time() - t0:.1f}s")
    return out


def _verdict(delta: float, p_gt: float) -> str:
    if delta >= 0.02 and p_gt >= 0.90:
        return f"PROMOTE (delta={delta:+.4f}, p(>0)={p_gt:.2f})"
    if delta >= 0.01:
        return f"INCONCLUSIVE_POSITIVE (delta={delta:+.4f}, p(>0)={p_gt:.2f})"
    if delta <= -0.01:
        return f"ROLLBACK (delta={delta:+.4f}, p(>0)={p_gt:.2f})"
    return f"NOISE_FLOOR (delta={delta:+.4f}, p(>0)={p_gt:.2f})"


def _write_markdown(out: dict):
    lines = []
    lines.append("# Augmented Head Training — Results\n")
    lines.append(
        "**Question:** Does training the v4 linear head on D4-expanded samples "
        "(8x per scan) improve weighted F1 vs the standard averaged-embedding recipe?\n"
    )
    lines.append("## Treatments\n")
    lines.append(
        "- **T1 averaged-head**: train head on mean-over-views embedding per scan "
        "(identical to v4 recipe; sanity baseline).\n"
        "- **T2 noise-injected**: augment in embedding space (Gaussian sigma=0.01 on L2-normalized embeddings, 8 copies).\n"
        "- **T3 D4-augmented**: each of the 8 D4 views = separate training sample "
        "(8x data for head). Test-time still averages views.\n"
    )
    lines.append(f"## Setup\n")
    lines.append(
        f"- Person-level LOPO, {out['n_persons']} folds, {out['n_scans']} scans.\n"
        f"- v4 reference weighted F1 (reports): **{out['v4_baseline_weighted_f1']}**.\n"
    )
    if out.get("fallback_used_for_t3"):
        lines.append(
            f"- T3 fallback to T1 for components {out['fallback_used_for_t3']} "
            f"(per-view cache unavailable).\n"
        )

    lines.append("## Ensemble-level results\n")
    lines.append("| Treatment | Weighted F1 | Macro F1 | delta vs v4 |")
    lines.append("|---|---:|---:|---:|")
    for key, label in [("T1_averaged_head", "T1 averaged"),
                        ("T2_noise_injected", "T2 noise-inj"),
                        ("T3_d4_augmented", "T3 D4-aug")]:
        m = out[key]
        d = out["delta_vs_v4"][key.split("_")[0]]
        lines.append(
            f"| **{label}** | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} | {d:+.4f} |"
        )
    lines.append("")

    lines.append("## Per-class F1\n")
    header = "| Treatment | " + " | ".join(CLASSES) + " |"
    sep = "|---|" + "|".join([":---:"] * N_CLASSES) + "|"
    lines.append(header)
    lines.append(sep)
    for key, label in [("T1_averaged_head", "T1 averaged"),
                        ("T2_noise_injected", "T2 noise-inj"),
                        ("T3_d4_augmented", "T3 D4-aug")]:
        pcf1 = out[key]["per_class_f1"]
        lines.append(f"| **{label}** | " + " | ".join(f"{v:.4f}" for v in pcf1) + " |")
    lines.append("")

    lines.append("## Paired bootstrap (1000x) vs T1 (in-experiment baseline)\n")
    lines.append("| Treatment | delta_mean | delta_p05 | delta_p95 | P(delta>0) |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, label in [("bootstrap_T2_vs_T1", "T2 vs T1"),
                        ("bootstrap_T3_vs_T1", "T3 vs T1")]:
        b = out[key]
        lines.append(
            f"| **{label}** | {b['delta_mean']:+.4f} | {b['delta_p05']:+.4f} | "
            f"{b['delta_p95']:+.4f} | {b['p_gt_zero']:.3f} |"
        )
    lines.append("")

    lines.append("## Verdict\n")
    lines.append(f"- **T2 (noise-injected):** {out['verdict']['T2']}\n")
    lines.append(f"- **T3 (D4-augmented):**  {out['verdict']['T3']}\n")
    lines.append("\nPromotion rule: `delta >= +0.02 weighted F1 AND P(delta>0) >= 0.90`. "
                 "Noise floor: `|delta| < 0.01`.\n")

    (REPORTS / "AUGMENTED_HEAD_RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
