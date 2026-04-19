"""Train final model on FULL dataset (no CV) and save for inference.

Uses DINOv2-B tiled embeddings + balanced LogisticRegression.
This is the shippable model — we do NOT split out validation because we want
all 240 scans for the final model. LOPO numbers (reported in RESULTS.md) are
our honest expected performance.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.data import CLASSES, enumerate_samples, person_id
from teardrop.encoders import load_dinov2
from teardrop.infer import ClassifierBundle, preprocess_and_tile_spm
from teardrop.encoders import height_to_pil

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
MODEL_DIR = ROOT / "models" / "dinov2b_tiled_v1"


def main():
    # Try to load cached tiled embeddings (produced by baseline_tiled_ensemble.py)
    cache_file = CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz"
    if not cache_file.exists():
        print(f"Missing cache: {cache_file}")
        print("Run: .venv/bin/python scripts/baseline_tiled_ensemble.py dinov2_vitb14")
        sys.exit(1)

    z = np.load(cache_file, allow_pickle=True)
    X_tiles = z["X"]                        # (n_tiles, 768)
    tile_to_scan = z["tile_to_scan"]
    scan_y = z["scan_y"]                    # (n_scans,)
    scan_paths = z["scan_paths"].tolist()

    n_scans = len(scan_y)
    embed_dim = X_tiles.shape[1]

    # Aggregate to scan level
    scan_emb = np.zeros((n_scans, embed_dim), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int32)
    for ti, si in enumerate(tile_to_scan):
        scan_emb[si] += X_tiles[ti]
        counts[si] += 1
    scan_emb /= np.maximum(counts, 1)[:, None]

    print(f"Training on {n_scans} scans, embed_dim={embed_dim}")
    print(f"Class distribution: {dict(zip(*np.unique(scan_y, return_counts=True)))}")

    # Fit scaler + LR
    scaler = StandardScaler()
    X_std = scaler.fit_transform(scan_emb)

    clf = LogisticRegression(
        class_weight="balanced", max_iter=3000, C=1.0,
        solver="lbfgs", n_jobs=4, random_state=42,
    )
    clf.fit(X_std, scan_y)

    # Sanity: training F1
    from sklearn.metrics import f1_score
    train_preds = clf.predict(X_std)
    print(f"Training F1 (seen by model): {f1_score(scan_y, train_preds, average='weighted'):.4f}")

    # Save
    encoder = load_dinov2("vitb14")
    bundle = ClassifierBundle(
        encoder=encoder,
        scaler_means=scaler.mean_.astype(np.float32),
        scaler_scales=scaler.scale_.astype(np.float32),
        lr_coef=clf.coef_.astype(np.float32),
        lr_intercept=clf.intercept_.astype(np.float32),
        classes=CLASSES,
        config={
            "target_nm_per_px": 90.0,
            "tile_size": 512,
            "max_tiles": 9,
            "render_mode": "afmhot",
            "encoder": "dinov2_vitb14",
            "classifier": "LogisticRegression_balanced",
            "trained_on_n_scans": int(n_scans),
            "honest_lopo_f1": 0.615,
        },
    )
    bundle.save(MODEL_DIR)
    print(f"[saved] {MODEL_DIR}")


if __name__ == "__main__":
    main()
