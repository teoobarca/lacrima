"""Baseline: frozen foundation model + linear probe, patient-level CV.

Tries multiple encoders (DINOv2, BiomedCLIP, OpenCLIP) and reports which wins.

Usage:
    .venv/bin/python scripts/baseline_foundation_probe.py [encoder]
        encoder ∈ {dinov2_vitb14, dinov2_vits14, biomedclip, openclip}
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
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.data import CLASSES, enumerate_samples, preprocess_spm
from teardrop.cv import patient_stratified_kfold, leave_one_patient_out
from teardrop.encoders import (
    EncoderBundle, height_to_pil, load_biomedclip, load_dinov2, load_openclip,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def get_encoder(name: str) -> EncoderBundle:
    if name == "dinov2_vitb14":
        return load_dinov2("vitb14")
    if name == "dinov2_vits14":
        return load_dinov2("vits14")
    if name == "biomedclip":
        return load_biomedclip()
    if name == "openclip":
        return load_openclip("ViT-L-14", "laion2b_s32b_b82k")
    raise ValueError(name)


def build_embeddings(samples, encoder: EncoderBundle, render_mode: str = "afmhot",
                     target_nm_per_px: float = 90.0, crop_size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    cache = CACHE_DIR / f"emb_{encoder.name}_{render_mode}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"[cache] {cache}")
        return z["X"], z["y"], z["groups"], z["paths"].tolist()

    print(f"Building embeddings with {encoder.name} (render={render_mode})...")
    pil_images, y, groups, paths = [], [], [], []
    t0 = time.time()
    for i, s in enumerate(samples):
        try:
            h = preprocess_spm(s.raw_path, target_nm_per_px=target_nm_per_px,
                               crop_size=crop_size)
            pil = height_to_pil(h, mode=render_mode)
            pil_images.append(pil)
            y.append(s.label)
            groups.append(s.patient)
            paths.append(str(s.raw_path))
        except Exception as e:
            print(f"  [err] {s.raw_path.name}: {e}")
        if (i + 1) % 40 == 0:
            print(f"  preproc [{i + 1}/{len(samples)}] {time.time() - t0:.1f}s")

    print(f"  preprocessed {len(pil_images)} images in {time.time() - t0:.1f}s")
    print(f"  encoding with {encoder.name}...")
    t1 = time.time()
    X = encoder.encode(pil_images, batch_size=16)
    print(f"  encoded {X.shape} in {time.time() - t1:.1f}s")

    y = np.array(y)
    groups = np.array(groups)
    paths = np.array(paths)
    np.savez(cache, X=X, y=y, groups=groups, paths=paths)
    print(f"[saved] {cache}")
    return X, y, groups, paths.tolist()


def evaluate_linear_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> dict:
    """Standardize → logistic regression with balanced class weights."""
    n_classes = len(np.unique(y))
    print(f"\nFeatures: {X.shape}  classes: {n_classes}  patients: {len(np.unique(groups))}")

    # ---- KFold ----
    print(f"\n--- StratifiedGroupKFold (k={n_splits}) ---")
    fold_f1s = []
    oof_preds = np.full(len(y), -1, dtype=int)
    for fi, (tr, va) in enumerate(patient_stratified_kfold(y, groups, n_splits, seed=42)):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr])
        Xv = scaler.transform(X[va])
        clf = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0,
            solver="lbfgs", n_jobs=4,
        )
        clf.fit(Xt, y[tr])
        preds = clf.predict(Xv)
        oof_preds[va] = preds
        f1 = f1_score(y[va], preds, average="weighted")
        f1m = f1_score(y[va], preds, average="macro")
        present = sorted(set(y[va]))
        print(f"  Fold {fi}: weighted F1={f1:.4f}  macro F1={f1m:.4f}  classes_in_val={present}")
        fold_f1s.append(f1)
    print(f"  Mean weighted F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    mask = oof_preds >= 0
    print(f"\n  OOF aggregate (n={mask.sum()}):")
    print(f"    Weighted F1: {f1_score(y[mask], oof_preds[mask], average='weighted'):.4f}")
    print(f"    Macro F1:    {f1_score(y[mask], oof_preds[mask], average='macro'):.4f}")
    print("\n" + classification_report(y[mask], oof_preds[mask],
                                       target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(y[mask], oof_preds[mask], labels=list(range(n_classes)))
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm_df.to_string())

    # ---- LOPO ----
    print(f"\n--- Leave-One-Patient-Out (n={len(np.unique(groups))} patients) ---")
    lopo_preds = np.full(len(y), -1, dtype=int)
    for tr, va in leave_one_patient_out(groups):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr])
        Xv = scaler.transform(X[va])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0,
                                 solver="lbfgs", n_jobs=4)
        clf.fit(Xt, y[tr])
        lopo_preds[va] = clf.predict(Xv)
    f1w = f1_score(y, lopo_preds, average="weighted")
    f1m = f1_score(y, lopo_preds, average="macro")
    print(f"  LOPO weighted F1: {f1w:.4f}")
    print(f"  LOPO macro F1:    {f1m:.4f}")
    print("\n" + classification_report(y, lopo_preds, target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(y, lopo_preds, labels=list(range(n_classes)))
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm_df.to_string())

    return {"kfold_mean_f1": float(np.mean(fold_f1s)),
            "lopo_f1": float(f1w), "lopo_macro_f1": float(f1m)}


def main():
    encoder_name = sys.argv[1] if len(sys.argv) > 1 else "dinov2_vits14"
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    print(f"Loaded {len(samples)} samples")

    enc = get_encoder(encoder_name)
    print(f"Encoder: {enc.name} (dim={enc.embed_dim}, device={enc.device})")

    X, y, groups, _ = build_embeddings(samples, enc, render_mode="afmhot")
    metrics = evaluate_linear_probe(X, y, groups, n_splits=5)
    print(f"\n>>> Final metrics for {enc.name}: {metrics}")


if __name__ == "__main__":
    main()
