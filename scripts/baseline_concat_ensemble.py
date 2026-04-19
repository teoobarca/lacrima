"""Concat ensemble: DINOv2 + BiomedCLIP + handcrafted features → XGBoost.

Loads cached tiled embeddings + cached handcrafted features and trains a unified
classifier on concatenated representation.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.cv import patient_stratified_kfold, leave_one_patient_out
from teardrop.data import CLASSES, enumerate_samples

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"


def load_tiled(name: str):
    """Load cached tiled embeddings npz."""
    path = CACHE / f"tiled_emb_{name}_afmhot_t512_n9.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return z["X"], z["tile_to_scan"], z["scan_y"], z["scan_groups"], z["scan_paths"].tolist()


def aggregate_tiles_to_scan(X_tiles: np.ndarray, tile_to_scan: np.ndarray,
                            n_scans: int) -> np.ndarray:
    """Mean-pool tile embeddings to scan-level embeddings."""
    embed_dim = X_tiles.shape[1]
    out = np.zeros((n_scans, embed_dim), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int32)
    for ti, si in enumerate(tile_to_scan):
        out[si] += X_tiles[ti]
        counts[si] += 1
    counts = np.maximum(counts, 1)
    out /= counts[:, None]
    return out


def evaluate(X, y, groups, name: str, n_splits: int = 5):
    n_classes = len(np.unique(y))
    print(f"\n=== {name} ===")
    print(f"X: {X.shape}  patients: {len(np.unique(groups))}")

    # ---- LR baseline ----
    print(f"\n[LR] LOPO ---")
    lopo_preds = np.full(len(y), -1, dtype=int)
    for tr, va in leave_one_patient_out(groups):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr])
        Xv = scaler.transform(X[va])
        clf = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0,
                                 solver="lbfgs", n_jobs=4)
        clf.fit(Xt, y[tr])
        lopo_preds[va] = clf.predict(Xv)
    f1w = f1_score(y, lopo_preds, average="weighted")
    f1m = f1_score(y, lopo_preds, average="macro")
    print(f"  [LR] LOPO weighted F1: {f1w:.4f}  macro F1: {f1m:.4f}")
    print(classification_report(y, lopo_preds, target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(y, lopo_preds, labels=list(range(n_classes)))
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    # ---- XGBoost ----
    print(f"\n[XGB] LOPO ---")
    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
    sample_weights = np.array([cw[label] for label in y])
    xgb_params = dict(
        n_estimators=500, max_depth=4, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.6,
        reg_lambda=2.0, reg_alpha=0.5,
        random_state=42, n_jobs=4,
        objective="multi:softprob", num_class=n_classes,
        tree_method="hist",
    )
    lopo_xgb = np.full(len(y), -1, dtype=int)
    for tr, va in leave_one_patient_out(groups):
        clf = XGBClassifier(**xgb_params)
        clf.fit(X[tr], y[tr], sample_weight=sample_weights[tr])
        lopo_xgb[va] = clf.predict(X[va])
    f1w_x = f1_score(y, lopo_xgb, average="weighted")
    f1m_x = f1_score(y, lopo_xgb, average="macro")
    print(f"  [XGB] LOPO weighted F1: {f1w_x:.4f}  macro F1: {f1m_x:.4f}")
    print(classification_report(y, lopo_xgb, target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(y, lopo_xgb, labels=list(range(n_classes)))
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    return {"lr_f1": float(f1w), "lr_macro_f1": float(f1m),
            "xgb_f1": float(f1w_x), "xgb_macro_f1": float(f1m_x)}


def main():
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    n_scans = len(samples)

    parts = {}

    # DINOv2-S tiled
    d = load_tiled("dinov2_vits14")
    if d is None:
        print("Run baseline_tiled_ensemble.py dinov2_vits14 first")
        return
    Xd, t2s, y, groups, _ = d
    parts["dinov2_vits14_tiled"] = aggregate_tiles_to_scan(Xd, t2s, n_scans)

    # BiomedCLIP tiled (if exists)
    b = load_tiled("biomedclip")
    if b is not None:
        Xb, t2s_b, _, _, _ = b
        parts["biomedclip_tiled"] = aggregate_tiles_to_scan(Xb, t2s_b, n_scans)

    # DINOv2-B tiled (if exists)
    db = load_tiled("dinov2_vitb14")
    if db is not None:
        Xdb, t2s_db, _, _, _ = db
        parts["dinov2_vitb14_tiled"] = aggregate_tiles_to_scan(Xdb, t2s_db, n_scans)

    # Handcrafted
    hc_path = CACHE / "features_handcrafted.parquet"
    if hc_path.exists():
        hc = pd.read_parquet(hc_path)
        # Reorder to match enumerate_samples order via raw path
        path_to_idx = {str(s.raw_path): i for i, s in enumerate(samples)}
        hc["si"] = hc["raw"].map(path_to_idx)
        hc = hc.dropna(subset=["si"]).sort_values("si")
        hc_cols = [c for c in hc.columns if c not in ("raw", "cls", "label", "patient", "si")]
        Xh = np.zeros((n_scans, len(hc_cols)), dtype=np.float32)
        for i, row in hc.iterrows():
            Xh[int(row["si"])] = row[hc_cols].values.astype(np.float32)
        parts["handcrafted"] = Xh

    print(f"\nAvailable parts: {list(parts.keys())}")
    print(f"Per-part shapes: " + ", ".join(f"{k}={v.shape}" for k, v in parts.items()))

    results = {}
    # Individual parts
    for name, X in parts.items():
        results[name] = evaluate(X, y, groups, name)

    # Concat all
    if len(parts) >= 2:
        X_all = np.concatenate(list(parts.values()), axis=1)
        results["CONCAT_ALL"] = evaluate(X_all, y, groups, "CONCAT_ALL")

    # Concat just neural (DINOv2 + BiomedCLIP)
    neural = [k for k in parts if "tiled" in k]
    if len(neural) >= 2:
        X_neural = np.concatenate([parts[k] for k in neural], axis=1)
        results["CONCAT_NEURAL"] = evaluate(X_neural, y, groups, "CONCAT_NEURAL")

    print("\n\n========== SUMMARY ==========")
    print(f"{'config':30s}  {'LR_F1':>8s} {'LR_mF1':>8s}  {'XGB_F1':>8s} {'XGB_mF1':>8s}")
    for name, m in results.items():
        print(f"{name:30s}  {m['lr_f1']:8.4f} {m['lr_macro_f1']:8.4f}  "
              f"{m['xgb_f1']:8.4f} {m['xgb_macro_f1']:8.4f}")


if __name__ == "__main__":
    main()
