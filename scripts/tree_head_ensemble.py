"""Tree-based head as alternative to LR for v4 fusion.

LR has linear decision boundaries. ExtraTrees / RandomForest have non-linear,
complementary error patterns. If they add diversity to v4, fusion may help.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
EPS = 1e-9


def _mean_pool(X_tiles, tile_to_scan, n_scans):
    out = np.zeros((n_scans, X_tiles.shape[1]), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int64)
    for i, s in enumerate(tile_to_scan):
        out[s] += X_tiles[i]
        counts[s] += 1
    out /= np.maximum(counts, 1)[:, None]
    return out


def fit_lopo(X, y, groups, head_factory, n_classes=5):
    n = len(y)
    oof = np.zeros((n, n_classes), dtype=np.float32)
    for p in np.unique(groups):
        train_mask = groups != p
        test_mask = ~train_mask
        Xt_n = normalize(X[train_mask], norm="l2", axis=1)
        sc = StandardScaler().fit(Xt_n)
        Xt_s = sc.transform(Xt_n)
        clf = head_factory()
        clf.fit(Xt_s, y[train_mask])
        Xe_n = normalize(X[test_mask], norm="l2", axis=1)
        p_partial = clf.predict_proba(sc.transform(Xe_n))
        full = np.zeros((p_partial.shape[0], n_classes), dtype=np.float32)
        full[:, clf.classes_] = p_partial
        oof[test_mask] = full
    return oof


def main():
    print("[load]")
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz", allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz", allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)
    y = z90["scan_y"]
    groups = z90["scan_groups"]
    n = len(y)
    X90 = _mean_pool(z90["X"], z90["tile_to_scan"], n)
    X45 = _mean_pool(z45["X"], z45["tile_to_scan"], n)
    Xbc = zbc["X_scan"].astype(np.float32)
    encoders = [("dinov2_90", X90), ("dinov2_45", X45), ("biomedclip", Xbc)]

    head_factories = {
        "LR": lambda: LogisticRegression(class_weight="balanced", max_iter=3000,
                                         C=1.0, solver="lbfgs", random_state=42),
        "ET": lambda: ExtraTreesClassifier(n_estimators=300, max_depth=None,
                                          class_weight="balanced",
                                          random_state=42, n_jobs=-1),
        "RF": lambda: RandomForestClassifier(n_estimators=300, max_depth=None,
                                            class_weight="balanced",
                                            random_state=42, n_jobs=-1),
    }

    # Compute per-encoder per-head OOF
    results = {}
    for hname, hf in head_factories.items():
        for ename, X in encoders:
            print(f"  fit {hname} on {ename}")
            results[(hname, ename)] = fit_lopo(X, y, groups, hf)

    # Per-head v4 (3-encoder geomean)
    print()
    for hname in head_factories:
        log_p = sum(np.log(results[(hname, e)] + EPS) for e, _ in encoders) / 3.0
        p = np.exp(log_p - log_p.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        pred = p.argmax(axis=1)
        f1w = f1_score(y, pred, average="weighted")
        f1m = f1_score(y, pred, average="macro")
        print(f"[v4-{hname}] wF1={f1w:.4f} mF1={f1m:.4f}")

    # LR + ET fusion (per-encoder mean of LR & ET, then geomean across encoders)
    log_p = 0
    for ename, _ in encoders:
        per_enc = (np.log(results[("LR", ename)] + EPS) +
                   np.log(results[("ET", ename)] + EPS)) / 2.0
        log_p = log_p + per_enc
    log_p = log_p / 3.0
    p = np.exp(log_p - log_p.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)
    f1w = f1_score(y, pred, average="weighted")
    f1m = f1_score(y, pred, average="macro")
    print(f"\n[v4 LR+ET fusion] wF1={f1w:.4f} mF1={f1m:.4f}")

    # 9-way ensemble: 3 encoders × 3 heads
    log_p = sum(np.log(results[(h, e)] + EPS) for h in head_factories for e, _ in encoders) / 9.0
    p = np.exp(log_p - log_p.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)
    f1w = f1_score(y, pred, average="weighted")
    f1m = f1_score(y, pred, average="macro")
    print(f"[v4 9-way (3enc × 3head)] wF1={f1w:.4f} mF1={f1m:.4f}")

    # baseline (LR only)
    log_p = sum(np.log(results[("LR", e)] + EPS) for e, _ in encoders) / 3.0
    p_b = np.exp(log_p - log_p.max(axis=1, keepdims=True))
    p_b /= p_b.sum(axis=1, keepdims=True)
    pred_b = p_b.argmax(axis=1)
    f1w_b = f1_score(y, pred_b, average="weighted")
    print(f"\n[BASELINE LR-only] wF1={f1w_b:.4f}")


if __name__ == "__main__":
    main()
