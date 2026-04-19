"""Baseline: Topological Data Analysis (cubical persistent homology) features.

Extracts ~1015 TDA features per scan (sublevel + superlevel persistence diagrams
for H_0 and H_1, at two scales, vectorized via persistence images + landscapes
+ summary stats + threshold-based Betti curves).

Evaluations:
    1. TDA standalone -> Logistic Regression (StandardScaler + balanced)
    2. TDA standalone -> XGBoost (matched to baseline_handcrafted_xgb.py params)
    3. TDA + DINOv2-S tiled (mean-pooled to scan-level) -> LR + XGB

Usage:
    .venv/bin/python scripts/baseline_tda.py
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
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out
from teardrop.data import CLASSES, enumerate_samples, preprocess_spm
from teardrop.topology import persistence_features

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "cache" / "features_tda.parquet"
DINO_PATH = ROOT / "cache" / "tiled_emb_dinov2_vits14_afmhot_t512_n9.npz"


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(samples, target_nm_per_px=90.0, crop_size=512):
    """Extract TDA features for every sample. Cached as parquet."""
    if CACHE_PATH.exists():
        print(f"[cache hit] {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH)

    CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
    rows = []
    t0 = time.time()
    for i, s in enumerate(samples):
        try:
            h = preprocess_spm(s.raw_path, target_nm_per_px=target_nm_per_px,
                               crop_size=crop_size)
            feats = persistence_features(h)
            row = {"raw": str(s.raw_path), "cls": s.cls, "label": s.label,
                   "patient": s.patient, **feats}
            rows.append(row)
        except Exception as e:
            print(f"  [err] {s.raw_path.name}: {e}")
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(samples) - i - 1)
            print(f"  [{i + 1}/{len(samples)}] {elapsed:.1f}s elapsed, ETA {eta:.1f}s")

    df = pd.DataFrame(rows)
    df.to_parquet(CACHE_PATH)
    print(f"[saved] {CACHE_PATH} ({len(df)} rows, {df.shape[1]} cols)")
    return df


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_lr(X, y, groups, name: str):
    n_classes = len(np.unique(y))
    # Drop features that are constant across the full training set --
    # they contribute nothing and cause NaN after z-scoring.
    var = X.var(axis=0)
    keep = var > 1e-12
    X_use = X[:, keep]
    print(f"\n[{name} | LR] Leave-One-Patient-Out -- {len(np.unique(groups))} patients "
          f"(dropped {int((~keep).sum())} zero-variance cols, using {X_use.shape[1]})")
    preds = np.full(len(y), -1, dtype=int)
    for tr, va in leave_one_patient_out(groups):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X_use[tr])
        Xv = scaler.transform(X_use[va])
        # Paranoia: in LOPO a fold-specific column may still be ~constant.
        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
        clf = LogisticRegression(class_weight="balanced", max_iter=4000, C=1.0,
                                 solver="lbfgs", n_jobs=4)
        clf.fit(Xt, y[tr])
        preds[va] = clf.predict(Xv)
    f1w = f1_score(y, preds, average="weighted")
    f1m = f1_score(y, preds, average="macro")
    f1pc = f1_score(y, preds, average=None, labels=list(range(n_classes)))
    print(f"  weighted F1: {f1w:.4f}  macro F1: {f1m:.4f}")
    print(f"  per-class F1: " + ", ".join(f"{CLASSES[i]}={v:.3f}" for i, v in enumerate(f1pc)))
    print(classification_report(y, preds, target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(y, preds, labels=list(range(n_classes)))
    print("  Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
    return {"weighted_f1": float(f1w), "macro_f1": float(f1m),
            "per_class_f1": [float(v) for v in f1pc], "preds": preds}


def _eval_xgb(X, y, groups, name: str, return_importances: bool = False,
              feature_names: list[str] | None = None):
    n_classes = len(np.unique(y))
    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
    sample_weights = np.array([cw[label] for label in y])
    base_params = dict(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.7,
        reg_lambda=1.5, reg_alpha=0.5,
        random_state=42, n_jobs=4,
        objective="multi:softprob", num_class=n_classes,
        tree_method="hist",
    )

    print(f"\n[{name} | XGB] Leave-One-Patient-Out -- {len(np.unique(groups))} patients")
    preds = np.full(len(y), -1, dtype=int)
    importances = np.zeros(X.shape[1], dtype=np.float64) if return_importances else None
    n_folds = 0
    for tr, va in leave_one_patient_out(groups):
        clf = XGBClassifier(**base_params)
        clf.fit(X[tr], y[tr], sample_weight=sample_weights[tr])
        preds[va] = clf.predict(X[va])
        if importances is not None:
            importances += clf.feature_importances_
            n_folds += 1
    f1w = f1_score(y, preds, average="weighted")
    f1m = f1_score(y, preds, average="macro")
    f1pc = f1_score(y, preds, average=None, labels=list(range(n_classes)))
    print(f"  weighted F1: {f1w:.4f}  macro F1: {f1m:.4f}")
    print(f"  per-class F1: " + ", ".join(f"{CLASSES[i]}={v:.3f}" for i, v in enumerate(f1pc)))
    print(classification_report(y, preds, target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(y, preds, labels=list(range(n_classes)))
    print("  Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    if importances is not None and feature_names is not None:
        importances /= max(n_folds, 1)
        top_idx = np.argsort(importances)[::-1][:25]
        print("\n  Top 25 features by mean XGB importance (across LOPO folds):")
        for i, idx in enumerate(top_idx):
            print(f"    {i+1:2d}. {feature_names[idx]:35s}  {importances[idx]:.5f}")

    out = {"weighted_f1": float(f1w), "macro_f1": float(f1m),
           "per_class_f1": [float(v) for v in f1pc], "preds": preds}
    if importances is not None:
        out["importances"] = importances
    return out


# ---------------------------------------------------------------------------
# DINOv2 helpers
# ---------------------------------------------------------------------------

def load_dinov2_scan_level(samples) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-pool DINOv2-S tiles to scan-level. Returns (X, y, groups)."""
    if not DINO_PATH.exists():
        return None, None, None
    z = np.load(DINO_PATH, allow_pickle=True)
    X_tiles = z["X"]
    tile_to_scan = z["tile_to_scan"]
    scan_y = z["scan_y"]
    scan_groups = z["scan_groups"]
    n_scans = len(samples)
    out = np.zeros((n_scans, X_tiles.shape[1]), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int32)
    for ti, si in enumerate(tile_to_scan):
        out[si] += X_tiles[ti]
        counts[si] += 1
    out /= np.maximum(counts, 1)[:, None]
    return out, scan_y, scan_groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    print(f"Enumerated {len(samples)} samples from TRAIN_SET")

    df = build_feature_matrix(samples)
    print(f"\nFeature dataframe: {df.shape}")
    print(f"Class distribution: {df['cls'].value_counts().to_dict()}")

    feature_cols = [c for c in df.columns if c not in ("raw", "cls", "label", "patient")]

    # Reorder to match enumerate_samples order via raw-path mapping.
    path_to_idx = {str(s.raw_path): i for i, s in enumerate(samples)}
    df["si"] = df["raw"].map(path_to_idx)
    df = df.dropna(subset=["si"]).sort_values("si").reset_index(drop=True)

    X_tda = df[feature_cols].values.astype(np.float32)
    # Replace any NaN/Inf with finite values (paranoia for downstream classifiers).
    X_tda = np.nan_to_num(X_tda, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].values.astype(int)
    groups = df["patient"].values

    print(f"\nTDA matrix: X_tda={X_tda.shape}, n_classes={len(np.unique(y))}, "
          f"n_patients={len(np.unique(groups))}")

    results: dict[str, dict] = {}

    # ----- TDA standalone -----
    print("\n" + "=" * 72)
    print("BLOCK 1 -- TDA features only")
    print("=" * 72)
    results["TDA_LR"] = _eval_lr(X_tda, y, groups, "TDA")
    results["TDA_XGB"] = _eval_xgb(X_tda, y, groups, "TDA",
                                   return_importances=True,
                                   feature_names=feature_cols)

    # ----- DINOv2 standalone -----
    print("\n" + "=" * 72)
    print("BLOCK 2 -- DINOv2-S (tiled, mean-pooled) only")
    print("=" * 72)
    X_dino, y_d, g_d = load_dinov2_scan_level(samples)
    if X_dino is None:
        print("DINOv2 cache not found; skipping DINOv2 + concat blocks.")
    else:
        # Sanity check: scan order should match.
        assert np.array_equal(y_d, y), "DINOv2 scan_y != enumerated labels"
        assert np.array_equal(g_d, groups), "DINOv2 scan_groups != enumerated patients"
        print(f"DINOv2 matrix: X_dino={X_dino.shape}")
        results["DINO_LR"] = _eval_lr(X_dino, y, groups, "DINOv2-S")
        results["DINO_XGB"] = _eval_xgb(X_dino, y, groups, "DINOv2-S")

        # ----- Concat -----
        print("\n" + "=" * 72)
        print("BLOCK 3 -- TDA + DINOv2-S concat")
        print("=" * 72)
        X_concat = np.concatenate([X_tda, X_dino], axis=1)
        print(f"Concat matrix: X_concat={X_concat.shape}")
        results["CONCAT_LR"] = _eval_lr(X_concat, y, groups, "TDA+DINO")
        results["CONCAT_XGB"] = _eval_xgb(X_concat, y, groups, "TDA+DINO")

    # ----- Summary -----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'config':25s}  {'weighted F1':>12s} {'macro F1':>10s}")
    for name, m in results.items():
        print(f"{name:25s}  {m['weighted_f1']:12.4f} {m['macro_f1']:10.4f}")

    # ----- Per-class F1 table -----
    print(f"\n{'config':25s}  " + "  ".join(f"{c:>18s}" for c in CLASSES))
    for name, m in results.items():
        cells = "  ".join(f"{v:18.4f}" for v in m["per_class_f1"])
        print(f"{name:25s}  {cells}")

    return results, feature_cols


if __name__ == "__main__":
    main()
