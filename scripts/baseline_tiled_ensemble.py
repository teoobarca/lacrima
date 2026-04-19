"""Tile-level baseline: each scan → multiple tiles → embeddings, then aggregate per-scan.

Idea:
- Original 1024×1024 scan → 4 non-overlapping 512×512 tiles (or more with overlap)
- Each tile is an independent training sample (4× more data!)
- At inference: predict on each tile, aggregate via mean of softmax probabilities
- This effectively augments minority classes (SucheOko 14 scans → 56 tiles)

Plus: ensemble of multiple encoders concatenated.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.cv import patient_stratified_kfold, leave_one_patient_out
from teardrop.data import (
    CLASSES, enumerate_samples, load_height, plane_level, resample_to_pixel_size,
    robust_normalize, tile,
)
from teardrop.encoders import (
    EncoderBundle, height_to_pil, load_biomedclip, load_dinov2,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def preprocess_to_tiles(
    raw_path: Path,
    target_nm_per_px: float = 90.0,
    tile_size: int = 512,
    stride: int | None = None,
    max_tiles: int = 9,
) -> list[np.ndarray]:
    """Load → plane level → resample → normalize → tile."""
    hm = load_height(raw_path)
    h = plane_level(hm.height)
    h = resample_to_pixel_size(h, hm.pixel_nm, target_nm_per_px)
    h = robust_normalize(h)
    if h.shape[0] < tile_size or h.shape[1] < tile_size:
        # Pad small scans by reflection
        pad_h = max(0, tile_size - h.shape[0])
        pad_w = max(0, tile_size - h.shape[1])
        h = np.pad(h, ((pad_h // 2, pad_h - pad_h // 2),
                       (pad_w // 2, pad_w - pad_w // 2)), mode="reflect")
    tiles = tile(h, tile_size, stride=stride or tile_size)
    if not tiles:  # edge case — somehow shape mismatch
        return [h[:tile_size, :tile_size]]
    if len(tiles) > max_tiles:
        # Keep evenly-spaced subset to limit memory
        idx = np.linspace(0, len(tiles) - 1, max_tiles).astype(int)
        tiles = [tiles[i] for i in idx]
    return tiles


def build_tile_embeddings(samples, encoder: EncoderBundle, render_mode: str = "afmhot",
                          tile_size: int = 512, max_tiles: int = 9):
    """For each scan, extract tile embeddings.

    Returns: X (n_tiles_total, embed_dim), tile_to_scan (n_tiles_total,) idx into samples,
             scan_y (n_scans,), scan_groups (n_scans,)
    """
    cache = CACHE_DIR / f"tiled_emb_{encoder.name}_{render_mode}_t{tile_size}_n{max_tiles}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"[cache] {cache}")
        return z["X"], z["tile_to_scan"], z["scan_y"], z["scan_groups"], z["scan_paths"].tolist()

    print(f"Tiling+encoding with {encoder.name} (render={render_mode}, tile={tile_size}, max_tiles={max_tiles})...")

    all_pil, tile_to_scan = [], []
    scan_y, scan_groups, scan_paths = [], [], []

    t0 = time.time()
    for si, s in enumerate(samples):
        try:
            tiles = preprocess_to_tiles(
                s.raw_path, target_nm_per_px=90.0,
                tile_size=tile_size, max_tiles=max_tiles,
            )
            for t in tiles:
                all_pil.append(height_to_pil(t, mode=render_mode))
                tile_to_scan.append(si)
            scan_y.append(s.label)
            scan_groups.append(s.patient)
            scan_paths.append(str(s.raw_path))
        except Exception as e:
            print(f"  [err] {s.raw_path.name}: {e}")
        if (si + 1) % 40 == 0:
            print(f"  preproc [{si + 1}/{len(samples)}] {len(all_pil)} tiles {time.time()-t0:.1f}s")

    print(f"  total {len(all_pil)} tiles from {len(samples)} scans in {time.time()-t0:.1f}s")
    print(f"  encoding...")
    t1 = time.time()
    X = encoder.encode(all_pil, batch_size=16)
    print(f"  encoded {X.shape} in {time.time()-t1:.1f}s")

    out = dict(
        X=X,
        tile_to_scan=np.array(tile_to_scan),
        scan_y=np.array(scan_y),
        scan_groups=np.array(scan_groups),
        scan_paths=np.array(scan_paths),
    )
    np.savez(cache, **out)
    print(f"[saved] {cache}")
    return out["X"], out["tile_to_scan"], out["scan_y"], out["scan_groups"], scan_paths


def evaluate_tile_lr(X, tile_to_scan, scan_y, scan_groups, n_splits=5):
    """Train classifier at TILE level, predict at SCAN level by mean-of-probas."""
    n_classes = len(np.unique(scan_y))
    print(f"\nTile features: {X.shape}  scans: {len(scan_y)}  patients: {len(np.unique(scan_groups))}")

    # ---- KFold (split BY SCAN, not by tile) ----
    print(f"\n--- StratifiedGroupKFold (k={n_splits}, scan-level split) ---")
    fold_f1s = []
    oof_scan_preds = np.full(len(scan_y), -1, dtype=int)
    oof_scan_probs = np.zeros((len(scan_y), n_classes))

    for fi, (tr_scans, va_scans) in enumerate(
        patient_stratified_kfold(scan_y, scan_groups, n_splits, seed=42)
    ):
        # Map scan indices to tile indices
        tr_tiles = np.where(np.isin(tile_to_scan, tr_scans))[0]
        va_tiles = np.where(np.isin(tile_to_scan, va_scans))[0]

        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr_tiles])
        Xv = scaler.transform(X[va_tiles])

        # Tile-level training labels: copy scan label to all its tiles
        tr_y_tiles = scan_y[tile_to_scan[tr_tiles]]

        clf = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0,
                                 solver="lbfgs", n_jobs=4)
        clf.fit(Xt, tr_y_tiles)

        # Tile-level probability predictions
        tile_probs = clf.predict_proba(Xv)  # (n_val_tiles, n_classes)

        # Aggregate to scan level: average tile probs per scan
        for si in va_scans:
            mask = tile_to_scan[va_tiles] == si
            if mask.any():
                avg = tile_probs[mask].mean(axis=0)
                oof_scan_probs[si] = avg
                oof_scan_preds[si] = avg.argmax()

        # Per-fold F1
        fold_mask = np.isin(np.arange(len(scan_y)), va_scans)
        f1 = f1_score(scan_y[fold_mask], oof_scan_preds[fold_mask], average="weighted")
        f1m = f1_score(scan_y[fold_mask], oof_scan_preds[fold_mask], average="macro")
        present = sorted(set(scan_y[fold_mask]))
        print(f"  Fold {fi}: weighted F1={f1:.4f}  macro F1={f1m:.4f}  classes={present}  "
              f"n_scans={len(va_scans)} n_tiles={len(va_tiles)}")
        fold_f1s.append(f1)

    print(f"  Mean weighted F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    mask = oof_scan_preds >= 0
    print(f"\n  OOF scan-level aggregate (n={mask.sum()}):")
    print(f"    Weighted F1: {f1_score(scan_y[mask], oof_scan_preds[mask], average='weighted'):.4f}")
    print(f"    Macro F1:    {f1_score(scan_y[mask], oof_scan_preds[mask], average='macro'):.4f}")
    print("\n" + classification_report(scan_y[mask], oof_scan_preds[mask],
                                       target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(scan_y[mask], oof_scan_preds[mask], labels=list(range(n_classes)))
    print("Confusion matrix:")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    # ---- LOPO ----
    print(f"\n--- LOPO ---")
    lopo_scan_preds = np.full(len(scan_y), -1, dtype=int)
    for tr_scans, va_scans in leave_one_patient_out(scan_groups):
        tr_tiles = np.where(np.isin(tile_to_scan, tr_scans))[0]
        va_tiles = np.where(np.isin(tile_to_scan, va_scans))[0]

        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr_tiles])
        Xv = scaler.transform(X[va_tiles])
        tr_y_tiles = scan_y[tile_to_scan[tr_tiles]]

        clf = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0,
                                 solver="lbfgs", n_jobs=4)
        clf.fit(Xt, tr_y_tiles)

        tile_probs = clf.predict_proba(Xv)
        for si in va_scans:
            m = tile_to_scan[va_tiles] == si
            if m.any():
                lopo_scan_preds[si] = tile_probs[m].mean(axis=0).argmax()

    f1w = f1_score(scan_y, lopo_scan_preds, average="weighted")
    f1m = f1_score(scan_y, lopo_scan_preds, average="macro")
    print(f"  LOPO weighted F1: {f1w:.4f}")
    print(f"  LOPO macro F1:    {f1m:.4f}")
    print("\n" + classification_report(scan_y, lopo_scan_preds, target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(scan_y, lopo_scan_preds, labels=list(range(n_classes)))
    print("Confusion matrix:")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    return {"kfold_mean_f1": float(np.mean(fold_f1s)),
            "lopo_f1": float(f1w), "lopo_macro_f1": float(f1m)}


def main():
    encoder_name = sys.argv[1] if len(sys.argv) > 1 else "dinov2_vits14"
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    print(f"Loaded {len(samples)} samples")

    if encoder_name == "biomedclip":
        enc = load_biomedclip()
    elif encoder_name == "dinov2_vitb14":
        enc = load_dinov2("vitb14")
    else:
        enc = load_dinov2("vits14")

    print(f"Encoder: {enc.name}")
    X, t2s, sy, sg, _ = build_tile_embeddings(samples, enc, render_mode="afmhot",
                                               tile_size=512, max_tiles=9)
    metrics = evaluate_tile_lr(X, t2s, sy, sg, n_splits=5)
    print(f"\n>>> Final metrics for {enc.name} (tiled): {metrics}")


if __name__ == "__main__":
    main()
