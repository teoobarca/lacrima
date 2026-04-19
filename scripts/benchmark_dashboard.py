"""Unified benchmark dashboard — evaluates ALL existing cached models on person-LOPO.

Produces a single table comparing:
- Raw handcrafted features (XGB, LR)
- Each foundation encoder alone (DINOv2-S/B, BiomedCLIP)
- Each foundation encoder tiled
- TTA variants
- Combinations

Used as the canonical evaluation harness after every new model is added to cache/.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.data import CLASSES, enumerate_samples, person_id
from teardrop.cv import leave_one_patient_out

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"


def aggregate_tiles(X_tiles: np.ndarray, tile_to_scan: np.ndarray, n_scans: int) -> np.ndarray:
    out = np.zeros((n_scans, X_tiles.shape[1]), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int32)
    for ti, si in enumerate(tile_to_scan):
        out[si] += X_tiles[ti]
        counts[si] += 1
    return out / np.maximum(counts, 1)[:, None]


def load_scan_level(name: str, n_scans: int) -> np.ndarray | None:
    """Try several cache patterns to load scan-level embedding matrix."""
    # TTA: already scan-level
    tta_path = CACHE / f"tta_emb_{name}_afmhot_t512_n9_d4.npz"
    if tta_path.exists():
        z = np.load(tta_path, allow_pickle=True)
        if "X_scan" in z.files:
            return z["X_scan"]
        X = z["X"]
        if X.shape[0] == n_scans:
            return X
        return aggregate_tiles(X, z["tile_to_scan"], n_scans)

    # Tiled non-TTA
    tiled_path = CACHE / f"tiled_emb_{name}_afmhot_t512_n9.npz"
    if tiled_path.exists():
        z = np.load(tiled_path, allow_pickle=True)
        return aggregate_tiles(z["X"], z["tile_to_scan"], n_scans)

    # Single crop
    single_path = CACHE / f"emb_{name}_afmhot.npz"
    if single_path.exists():
        z = np.load(single_path, allow_pickle=True)
        return z["X"]

    return None


def load_handcrafted(samples, n_scans: int) -> tuple[np.ndarray | None, list[str]]:
    path = CACHE / "features_handcrafted.parquet"
    if not path.exists():
        return None, []
    df = pd.read_parquet(path)
    path_to_idx = {str(s.raw_path): i for i, s in enumerate(samples)}
    df["si"] = df["raw"].map(path_to_idx)
    df = df.dropna(subset=["si"]).sort_values("si")
    feature_cols = [c for c in df.columns if c not in ("raw", "cls", "label", "patient", "si")]
    X = np.zeros((n_scans, len(feature_cols)), dtype=np.float32)
    for _, row in df.iterrows():
        X[int(row["si"])] = row[feature_cols].values.astype(np.float32)
    return X, feature_cols


def load_tda(samples, n_scans: int) -> np.ndarray | None:
    path = CACHE / "features_tda.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    path_to_idx = {str(s.raw_path): i for i, s in enumerate(samples)}
    df["si"] = df["raw"].map(path_to_idx)
    df = df.dropna(subset=["si"]).sort_values("si")
    feature_cols = [c for c in df.columns if c not in ("raw", "cls", "label", "patient", "si")]
    X = np.zeros((n_scans, len(feature_cols)), dtype=np.float32)
    for _, row in df.iterrows():
        X[int(row["si"])] = row[feature_cols].values.astype(np.float32)
    # TDA may have NaN — replace
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def load_advanced(samples, n_scans: int) -> np.ndarray | None:
    path = CACHE / "features_advanced.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    path_to_idx = {str(s.raw_path): i for i, s in enumerate(samples)}
    df["si"] = df["raw"].map(path_to_idx)
    df = df.dropna(subset=["si"]).sort_values("si")
    feature_cols = [c for c in df.columns if c not in ("raw", "cls", "label", "patient", "si")]
    X = np.zeros((n_scans, len(feature_cols)), dtype=np.float32)
    for _, row in df.iterrows():
        X[int(row["si"])] = row[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def lopo_lr(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """Person-LOPO LR evaluation. Returns (weighted_F1, macro_F1)."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    lopo_preds = np.full(len(y), -1, dtype=int)
    for tr, va in leave_one_patient_out(groups):
        scaler = StandardScaler()
        Xt = np.nan_to_num(scaler.fit_transform(X[tr]), nan=0.0, posinf=0.0, neginf=0.0)
        Xv = np.nan_to_num(scaler.transform(X[va]), nan=0.0, posinf=0.0, neginf=0.0)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0,
                                 solver="lbfgs", n_jobs=4)
        clf.fit(Xt, y[tr])
        lopo_preds[va] = clf.predict(Xv)
    return (
        f1_score(y, lopo_preds, average="weighted"),
        f1_score(y, lopo_preds, average="macro"),
    )


def main():
    print("Building benchmark dashboard...")
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    n_scans = len(samples)
    y = np.array([s.label for s in samples])
    groups = np.array([s.person for s in samples])
    print(f"n_scans={n_scans}, n_classes={len(CLASSES)}, n_persons={len(np.unique(groups))}")

    rows = []

    # Foundation encoders — scan-level
    encoder_configs = [
        ("dinov2_vits14", "single-crop"),
        ("dinov2_vitb14", "single-crop"),
        ("biomedclip", "single-crop"),
    ]
    for enc_name, tag in encoder_configs:
        path = CACHE / f"emb_{enc_name}_afmhot.npz"
        if path.exists():
            z = np.load(path, allow_pickle=True)
            X = z["X"]
            t0 = time.time()
            f1w, f1m = lopo_lr(X, y, groups)
            rows.append({"config": f"{enc_name} {tag}", "f1_weighted": f1w,
                         "f1_macro": f1m, "dim": X.shape[1], "seconds": time.time() - t0})

    # Tiled (mean-pool) versions
    for enc_name in ["dinov2_vits14", "dinov2_vitb14", "biomedclip"]:
        path = CACHE / f"tiled_emb_{enc_name}_afmhot_t512_n9.npz"
        if path.exists():
            z = np.load(path, allow_pickle=True)
            X = aggregate_tiles(z["X"], z["tile_to_scan"], n_scans)
            t0 = time.time()
            f1w, f1m = lopo_lr(X, y, groups)
            rows.append({"config": f"{enc_name} tiled (mean-pool)", "f1_weighted": f1w,
                         "f1_macro": f1m, "dim": X.shape[1], "seconds": time.time() - t0})

    # TTA versions
    for enc_name in ["dinov2_vitb14", "biomedclip"]:
        path = CACHE / f"tta_emb_{enc_name}_afmhot_t512_n9_d4.npz"
        if path.exists():
            z = np.load(path, allow_pickle=True)
            X = z["X_scan"] if "X_scan" in z.files else z["X"]
            if X.shape[0] != n_scans:
                X = aggregate_tiles(X, z["tile_to_scan"], n_scans)
            t0 = time.time()
            f1w, f1m = lopo_lr(X, y, groups)
            rows.append({"config": f"{enc_name} TTA (D4, mean-pool)", "f1_weighted": f1w,
                         "f1_macro": f1m, "dim": X.shape[1], "seconds": time.time() - t0})

    # Ensemble: softmax average of DINOv2-B tiled + BiomedCLIP tiled (NO TTA)
    def load_tiled_explicit(name: str) -> np.ndarray | None:
        p = CACHE / f"tiled_emb_{name}_afmhot_t512_n9.npz"
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=True)
        return aggregate_tiles(z["X"], z["tile_to_scan"], n_scans)

    dinov2b_tiled = load_tiled_explicit("dinov2_vitb14")
    biomedclip_tiled = load_tiled_explicit("biomedclip")
    if dinov2b_tiled is not None and biomedclip_tiled is not None:
        # do LOPO with ensemble: get softmax from each, average, argmax
        from sklearn.metrics import f1_score as _f1
        preds = np.full(n_scans, -1, dtype=int)
        for tr, va in leave_one_patient_out(groups):
            # DINOv2-B
            sc1 = StandardScaler()
            Xt1 = sc1.fit_transform(dinov2b_tiled[tr])
            Xv1 = sc1.transform(dinov2b_tiled[va])
            clf1 = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, n_jobs=4)
            clf1.fit(Xt1, y[tr])
            p1 = clf1.predict_proba(Xv1)
            # BiomedCLIP
            sc2 = StandardScaler()
            Xt2 = sc2.fit_transform(biomedclip_tiled[tr])
            Xv2 = sc2.transform(biomedclip_tiled[va])
            clf2 = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, n_jobs=4)
            clf2.fit(Xt2, y[tr])
            p2 = clf2.predict_proba(Xv2)
            preds[va] = (0.5 * p1 + 0.5 * p2).argmax(axis=1)
        f1w = _f1(y, preds, average="weighted")
        f1m = _f1(y, preds, average="macro")
        rows.append({"config": "ENSEMBLE DINOv2-B + BiomedCLIP (tiled, no TTA)",
                     "f1_weighted": f1w, "f1_macro": f1m,
                     "dim": dinov2b_tiled.shape[1] + biomedclip_tiled.shape[1],
                     "seconds": 0.0})

    # Same but with TTA features where available
    dinov2b_tta = load_scan_level("dinov2_vitb14", n_scans)  # falls back to tiled
    tta_dinov2b_path = CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz"
    tta_biomedclip_path = CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz"
    if tta_dinov2b_path.exists() and tta_biomedclip_path.exists():
        zd = np.load(tta_dinov2b_path, allow_pickle=True)
        zb = np.load(tta_biomedclip_path, allow_pickle=True)
        Xd = zd["X_scan"] if "X_scan" in zd.files else zd["X"]
        Xb = zb["X_scan"] if "X_scan" in zb.files else zb["X"]
        preds = np.full(n_scans, -1, dtype=int)
        for tr, va in leave_one_patient_out(groups):
            sc1 = StandardScaler(); Xt1 = sc1.fit_transform(Xd[tr]); Xv1 = sc1.transform(Xd[va])
            clf1 = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, n_jobs=4)
            clf1.fit(Xt1, y[tr])
            p1 = clf1.predict_proba(Xv1)
            sc2 = StandardScaler(); Xt2 = sc2.fit_transform(Xb[tr]); Xv2 = sc2.transform(Xb[va])
            clf2 = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, n_jobs=4)
            clf2.fit(Xt2, y[tr])
            p2 = clf2.predict_proba(Xv2)
            preds[va] = (0.5 * p1 + 0.5 * p2).argmax(axis=1)
        f1w = f1_score(y, preds, average="weighted")
        f1m = f1_score(y, preds, average="macro")
        rows.append({"config": "★ ENSEMBLE TTA (shipped champion)",
                     "f1_weighted": f1w, "f1_macro": f1m,
                     "dim": Xd.shape[1] + Xb.shape[1], "seconds": 0.0})

    # Handcrafted
    X_hc, _ = load_handcrafted(samples, n_scans)
    if X_hc is not None:
        f1w, f1m = lopo_lr(X_hc, y, groups)
        rows.append({"config": "handcrafted (94 feat) + LR", "f1_weighted": f1w,
                     "f1_macro": f1m, "dim": X_hc.shape[1], "seconds": 0.0})

    # TDA
    X_tda = load_tda(samples, n_scans)
    if X_tda is not None:
        f1w, f1m = lopo_lr(X_tda, y, groups)
        rows.append({"config": "TDA (1015 feat) + LR", "f1_weighted": f1w,
                     "f1_macro": f1m, "dim": X_tda.shape[1], "seconds": 0.0})

    # Advanced features (if cached by Wave 5)
    X_adv = load_advanced(samples, n_scans)
    if X_adv is not None:
        f1w, f1m = lopo_lr(X_adv, y, groups)
        rows.append({"config": "advanced features + LR", "f1_weighted": f1w,
                     "f1_macro": f1m, "dim": X_adv.shape[1], "seconds": 0.0})

    df = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 100)
    print("BENCHMARK DASHBOARD — person-level LOPO (35 persons), Logistic Regression head")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out_path = ROOT / "reports" / "BENCHMARK_DASHBOARD.md"
    with open(out_path, "w") as f:
        f.write("# Benchmark Dashboard — person-level LOPO F1\n\n")
        f.write("Single-classifier (Logistic Regression, `class_weight='balanced'`) evaluation "
                "on person-level LOPO with all cached feature sets.\n\n")
        f.write("Last updated automatically by `scripts/benchmark_dashboard.py`.\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
