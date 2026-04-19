"""Weighted geometric mean of v4 encoders.

Default v4 uses uniform 1/3 weight. Test: weight by per-encoder standalone F1
(soft attention to stronger encoders). Use INNER LOPO to compute weights, then
OUTER LOPO eval — proper nested protocol.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
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


def fit_predict(X_train, y_train, X_test, n_classes=5):
    Xt_n = normalize(X_train, norm="l2", axis=1)
    sc = StandardScaler().fit(Xt_n)
    Xt_s = sc.transform(Xt_n)
    clf = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0,
                             solver="lbfgs", random_state=42)
    clf.fit(Xt_s, y_train)
    Xe_n = normalize(X_test, norm="l2", axis=1)
    p_partial = clf.predict_proba(sc.transform(Xe_n))
    # Pad missing classes with 0
    p_full = np.zeros((p_partial.shape[0], n_classes), dtype=np.float32)
    p_full[:, clf.classes_] = p_partial
    return p_full


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
    persons = np.unique(groups)
    n_enc = len(encoders)

    # Outer LOPO with inner-fold weight selection
    oof_baseline = np.zeros((n, 5), dtype=np.float32)
    oof_weighted = np.zeros((n, 5), dtype=np.float32)
    oof_optimal = np.zeros((n, 5), dtype=np.float32)
    weights_hist = []

    for outer_p in persons:
        outer_train_mask = groups != outer_p
        outer_test_mask = ~outer_train_mask
        outer_train_persons = np.unique(groups[outer_train_mask])

        # Per-encoder probs on outer test (used for both baseline and weighted)
        per_enc_outer = []
        for ename, X in encoders:
            p = fit_predict(X[outer_train_mask], y[outer_train_mask], X[outer_test_mask])
            per_enc_outer.append(p)

        # Baseline: uniform geomean
        log_b = sum(np.log(p + EPS) for p in per_enc_outer) / n_enc
        p_b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
        p_b /= p_b.sum(axis=1, keepdims=True)
        oof_baseline[outer_test_mask] = p_b

        # Inner LOPO on outer_train_persons → per-encoder F1
        per_enc_inner_f1 = []
        for ename, X in encoders:
            inner_oof = np.zeros((sum(outer_train_mask), 5), dtype=np.float32)
            X_outer_train = X[outer_train_mask]
            y_outer_train = y[outer_train_mask]
            g_outer_train = groups[outer_train_mask]
            inner_persons = np.unique(g_outer_train)
            for inner_p in inner_persons:
                inner_train_mask = g_outer_train != inner_p
                inner_test_mask = ~inner_train_mask
                p = fit_predict(X_outer_train[inner_train_mask],
                                y_outer_train[inner_train_mask],
                                X_outer_train[inner_test_mask])
                inner_oof[inner_test_mask] = p
            inner_pred = inner_oof.argmax(axis=1)
            f1 = f1_score(y_outer_train, inner_pred, average="weighted", zero_division=0)
            per_enc_inner_f1.append(f1)

        # Weighted geomean: weights ∝ inner F1
        f1_arr = np.array(per_enc_inner_f1)
        # sharpen via softmax
        weights = np.exp(f1_arr * 5)  # scale 5 = soft attention
        weights /= weights.sum()
        weights_hist.append(weights)

        log_w = sum(w * np.log(p + EPS) for w, p in zip(weights, per_enc_outer))
        p_w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
        p_w /= p_w.sum(axis=1, keepdims=True)
        oof_weighted[outer_test_mask] = p_w

        # OPTIMAL weight per fold (oracle — for upper bound diagnostic)
        # Try {0,1,2} weight grid for each encoder, pick best by inner OOF
        # Skip for time; use weighted only

    pred_b = oof_baseline.argmax(axis=1)
    pred_w = oof_weighted.argmax(axis=1)
    f1w_b = f1_score(y, pred_b, average="weighted")
    f1m_b = f1_score(y, pred_b, average="macro")
    f1w_w = f1_score(y, pred_w, average="weighted")
    f1m_w = f1_score(y, pred_w, average="macro")

    print(f"\n[baseline uniform geomean]   wF1={f1w_b:.4f} mF1={f1m_b:.4f}")
    print(f"[weighted by inner F1 sm5]   wF1={f1w_w:.4f} mF1={f1m_w:.4f}  Δ={f1w_w-f1w_b:+.4f}")

    avg_w = np.mean(weights_hist, axis=0)
    print(f"\nMean weights across folds: dinov2_90={avg_w[0]:.3f} dinov2_45={avg_w[1]:.3f} biomedclip={avg_w[2]:.3f}")

    # Bootstrap
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(1000):
        sampled = rng.choice(persons, size=len(persons), replace=True)
        mask = np.isin(groups, sampled)
        f_b = f1_score(y[mask], pred_b[mask], average="weighted", zero_division=0)
        f_v = f1_score(y[mask], pred_w[mask], average="weighted", zero_division=0)
        deltas.append(f_v - f_b)
    d = np.array(deltas)
    print(f"\n[bootstrap 1000x]  mean Δ={d.mean():+.4f}  CI95=[{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]  P(Δ>0)={(d>0).mean():.3f}")


if __name__ == "__main__":
    main()
