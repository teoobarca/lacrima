"""Train and save the v4 multi-scale champion ensemble on all 240 scans.

Recipe (Wave 7 Config D, confirmed by `multiscale_experiment.py` and red-team
approved in `reports/RED_TEAM_MULTISCALE.md`):

  Three components, each independently trained with the v2 recipe
  (L2-normalize row-wise -> StandardScaler -> LogisticRegression(balanced)):

    A) DINOv2-B at 90 nm/px, NO TTA  (tiled embeddings, mean-pooled per scan)
    B) DINOv2-B at 45 nm/px, NO TTA  (finer-scale lipid/mucin texture)
    C) BiomedCLIP at 90 nm/px, D4 TTA (72 views per scan, already scan-level)

  Ensemble = geometric mean of 3 softmaxes, renormalized, argmax.

Why no TTA on DINOv2 branches? Empirically, D4 TTA at 45 nm/px REGRESSES the
weighted F1 by 0.022 — fine-scale features are orientation-specific so
averaging over rotations washes out signal. See `reports/MULTISCALE_TTA_RESULTS.md`.

Honest person-LOPO weighted F1: 0.6887
Honest person-LOPO macro F1:    0.5541

Saves to: models/ensemble_v4_multiscale/
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.data import CLASSES, person_id  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
MODEL_DIR = ROOT / "models" / "ensemble_v4_multiscale"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-9


def _mean_pool_tiles(X_tiles: np.ndarray, tile_to_scan: np.ndarray,
                     n_scans: int) -> np.ndarray:
    """Mean-pool per-tile embeddings (N_tiles, D) -> (n_scans, D)."""
    D = X_tiles.shape[1]
    out = np.zeros((n_scans, D), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int64)
    for i, s in enumerate(tile_to_scan):
        out[s] += X_tiles[i]
        counts[s] += 1
    counts = np.maximum(counts, 1)
    out /= counts[:, None]
    return out


def _fit_component(X: np.ndarray, y: np.ndarray, name: str):
    X_norm = normalize(X, norm="l2", axis=1)
    sc = StandardScaler().fit(X_norm)
    X_std = sc.transform(X_norm)
    clf = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0,
                             solver="lbfgs", n_jobs=4, random_state=42)
    clf.fit(X_std, y)
    train_acc = (clf.predict(X_std) == y).mean()
    print(f"  [{name}] D={X.shape[1]} train-acc={train_acc:.4f}")
    return sc, clf


def _softmax_from_lr(X: np.ndarray, sc: StandardScaler,
                     clf: LogisticRegression) -> np.ndarray:
    X_norm = normalize(X, norm="l2", axis=1)
    X_std = sc.transform(X_norm)
    return clf.predict_proba(X_std)


def main():
    print("[load] caches")
    # 90 nm/px DINOv2-B tiled (non-TTA)
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz",
                  allow_pickle=True)
    # 45 nm/px DINOv2-B tiled (non-TTA)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz",
                  allow_pickle=True)
    # BiomedCLIP 90 nm/px D4 TTA (already scan-level)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz",
                  allow_pickle=True)

    y = z90["scan_y"]
    scan_paths = z90["scan_paths"]
    scan_groups = z90["scan_groups"]
    n_scans = len(y)
    print(f"  n_scans={n_scans}")

    # Sanity: alignment of scan_paths across caches
    if not np.array_equal(z45["scan_paths"], scan_paths):
        raise RuntimeError("scan_paths mismatch between 90nm and 45nm caches")
    if not np.array_equal(zbc["scan_paths"], scan_paths):
        raise RuntimeError("scan_paths mismatch between 90nm and BiomedCLIP TTA caches")
    if not np.array_equal(z45["scan_y"], y) or not np.array_equal(zbc["scan_y"], y):
        raise RuntimeError("label mismatch across caches")

    # Mean-pool tile embeddings to scan level
    X90 = _mean_pool_tiles(z90["X"], z90["tile_to_scan"], n_scans)
    X45 = _mean_pool_tiles(z45["X"], z45["tile_to_scan"], n_scans)
    Xbc = zbc["X_scan"].astype(np.float32)
    print(f"  X90={X90.shape}  X45={X45.shape}  Xbc={Xbc.shape}")

    # Fit 3 components with the v2 recipe
    print("[fit] components")
    sc90, clf90 = _fit_component(X90, y, "dinov2b_90nm")
    sc45, clf45 = _fit_component(X45, y, "dinov2b_45nm")
    scbc, clfbc = _fit_component(Xbc, y, "biomedclip_tta")

    # Sanity: training F1 (memorization check on 240 samples)
    p90 = _softmax_from_lr(X90, sc90, clf90)
    p45 = _softmax_from_lr(X45, sc45, clf45)
    pbc = _softmax_from_lr(Xbc, scbc, clfbc)
    log_avg = (np.log(p90 + EPS) + np.log(p45 + EPS) + np.log(pbc + EPS)) / 3.0
    p_geom = np.exp(log_avg - log_avg.max(axis=1, keepdims=True))
    p_geom /= p_geom.sum(axis=1, keepdims=True)
    preds = p_geom.argmax(axis=1)
    train_f1_w = f1_score(y, preds, average="weighted")
    train_f1_m = f1_score(y, preds, average="macro")
    print(f"  training geometric-mean F1 (expect ~1.0): "
          f"weighted={train_f1_w:.4f} macro={train_f1_m:.4f}")

    # Save components
    comp_specs = [
        ("dinov2b_90nm",   clf90, sc90, "dinov2_vitb14", 90.0, False),
        ("dinov2b_45nm",   clf45, sc45, "dinov2_vitb14", 45.0, False),
        ("biomedclip_tta", clfbc, scbc, "biomedclip",    90.0, True),
    ]
    for name, clf, sc, enc_name, nm_per_px, uses_tta in comp_specs:
        comp_dir = MODEL_DIR / name
        comp_dir.mkdir(exist_ok=True)
        np.savez(
            comp_dir / "classifier.npz",
            scaler_means=sc.mean_.astype(np.float32),
            scaler_scales=sc.scale_.astype(np.float32),
            lr_coef=clf.coef_.astype(np.float32),
            lr_intercept=clf.intercept_.astype(np.float32),
        )
        with open(comp_dir / "meta.json", "w") as f:
            json.dump({
                "kind": "single",
                "encoder_name": enc_name,
                "classes": CLASSES,
                "config": {
                    "target_nm_per_px": nm_per_px,
                    "tile_size": 512,
                    "max_tiles": 9,
                    "render_mode": "afmhot",
                    "tta_group": "D4" if uses_tta else "none",
                    "preprocessing": "l2_normalize_then_standardscaler",
                },
            }, f, indent=2)
        print(f"  [saved] {comp_dir}")

    # Top-level meta
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump({
            "kind": "ensemble_v4",
            "components": ["dinov2b_90nm", "dinov2b_45nm", "biomedclip_tta"],
            "classes": CLASSES,
            "config": {
                "ensemble_method": "geometric_mean",
                "preprocessing": "l2_normalize_row -> standardscaler",
                "tile_size": 512,
                "max_tiles": 9,
                "render_mode": "afmhot",
                "scales_nm_per_px": {
                    "dinov2b_90nm": 90.0,
                    "dinov2b_45nm": 45.0,
                    "biomedclip_tta": 90.0,
                },
                "tta_group": {
                    "dinov2b_90nm": "none",
                    "dinov2b_45nm": "none",
                    "biomedclip_tta": "D4",
                },
                "honest_lopo_weighted_f1": 0.6887,
                "honest_lopo_macro_f1": 0.5541,
                "trained_on_n_scans": int(n_scans),
                "trained_on_n_persons": int(len({person_id(Path(str(p))) for p in scan_paths})),
                "provenance": "Wave 7 Config D — multi-scale champion, red-team approved",
            },
        }, f, indent=2)

    # Copy predict.py (written separately)
    print(f"[done] bundle saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
