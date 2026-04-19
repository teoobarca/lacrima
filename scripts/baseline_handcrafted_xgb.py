"""Baseline: handcrafted features → XGBoost, evaluated by patient-level CV.

Usage:
    .venv/bin/python scripts/baseline_handcrafted_xgb.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.data import (
    CLASSES, enumerate_samples, preprocess_spm, samples_dataframe,
)
from teardrop.features import extract_all_features
from teardrop.cv import patient_stratified_kfold, leave_one_patient_out

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "features_handcrafted.parquet"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def build_feature_matrix(samples, target_nm_per_px=90.0, crop_size=512):
    """Extract features for every sample. Cached as parquet."""
    if CACHE.exists():
        print(f"[cache hit] {CACHE}")
        return pd.read_parquet(CACHE)
    CACHE.parent.mkdir(exist_ok=True)

    rows = []
    t0 = time.time()
    for i, s in enumerate(samples):
        try:
            h = preprocess_spm(s.raw_path, target_nm_per_px=target_nm_per_px,
                               crop_size=crop_size)
            feats = extract_all_features(h)
            row = {"raw": str(s.raw_path), "cls": s.cls, "label": s.label,
                   "patient": s.patient, **feats}
            rows.append(row)
        except Exception as e:
            print(f"  [err] {s.raw_path.name}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(samples)}] {time.time() - t0:.1f}s elapsed")

    df = pd.DataFrame(rows)
    df.to_parquet(CACHE)
    print(f"[saved] {CACHE} ({len(df)} rows, {df.shape[1]} cols)")
    return df


def evaluate_xgb(df: pd.DataFrame, n_splits: int = 5):
    """Patient-stratified k-fold + LOPO eval. Reports weighted F1."""
    feature_cols = [c for c in df.columns if c not in ("raw", "cls", "label", "patient")]
    X = df[feature_cols].values
    y = df["label"].values
    groups = df["patient"].values

    print(f"\n{'='*60}\nFeature matrix: X={X.shape}, n_classes={len(np.unique(y))}, n_patients={len(np.unique(groups))}")

    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
    sample_weights = np.array([cw[label] for label in y])

    base_params = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=1.5,
        reg_alpha=0.5,
        random_state=42,
        n_jobs=4,
        objective="multi:softprob",
        num_class=len(np.unique(y)),
        tree_method="hist",
    )

    # ---- k-fold ----
    print(f"\n--- StratifiedGroupKFold (k={n_splits}) ---")
    fold_f1s = []
    oof_preds = np.full(len(y), -1, dtype=int)
    oof_probs = np.zeros((len(y), len(np.unique(y))), dtype=np.float64)
    for fi, (tr, va) in enumerate(patient_stratified_kfold(y, groups, n_splits, seed=42)):
        clf = XGBClassifier(**base_params)
        clf.fit(X[tr], y[tr], sample_weight=sample_weights[tr])
        probs = clf.predict_proba(X[va])
        preds = probs.argmax(axis=1)
        oof_preds[va] = preds
        oof_probs[va] = probs
        f1 = f1_score(y[va], preds, average="weighted")
        f1m = f1_score(y[va], preds, average="macro")
        present = sorted(set(y[va]))
        print(f"  Fold {fi}: weighted F1={f1:.4f}  macro F1={f1m:.4f}  "
              f"classes_in_val={present}  n={len(va)}")
        fold_f1s.append(f1)
    print(f"  Mean weighted F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    # Aggregate OOF report (over all samples that got a prediction)
    mask = oof_preds >= 0
    print(f"\n  OOF aggregate (n={mask.sum()}):")
    print(f"    Weighted F1: {f1_score(y[mask], oof_preds[mask], average='weighted'):.4f}")
    print(f"    Macro F1:    {f1_score(y[mask], oof_preds[mask], average='macro'):.4f}")
    print("\n" + classification_report(y[mask], oof_preds[mask],
                                       target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(y[mask], oof_preds[mask], labels=list(range(len(CLASSES))))
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm_df.to_string())

    # ---- LOPO ----
    print(f"\n--- Leave-One-Patient-Out (n={len(np.unique(groups))} patients) ---")
    lopo_preds = np.full(len(y), -1, dtype=int)
    for tr, va in leave_one_patient_out(groups):
        clf = XGBClassifier(**base_params)
        clf.fit(X[tr], y[tr], sample_weight=sample_weights[tr])
        lopo_preds[va] = clf.predict(X[va])
    print(f"  LOPO weighted F1: {f1_score(y, lopo_preds, average='weighted'):.4f}")
    print(f"  LOPO macro F1:    {f1_score(y, lopo_preds, average='macro'):.4f}")
    print("\n" + classification_report(y, lopo_preds, target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(y, lopo_preds, labels=list(range(len(CLASSES))))
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm_df.to_string())

    return {
        "kfold_mean_f1": float(np.mean(fold_f1s)),
        "lopo_f1": float(f1_score(y, lopo_preds, average="weighted")),
        "lopo_macro_f1": float(f1_score(y, lopo_preds, average="macro")),
    }


def main() -> None:
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    print(f"Enumerated {len(samples)} samples")

    df = build_feature_matrix(samples)
    print(df.head())

    metrics = evaluate_xgb(df, n_splits=5)
    print(f"\nFinal metrics: {metrics}")


if __name__ == "__main__":
    main()
