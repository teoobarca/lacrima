"""Multi-seed v4 ensemble via person-LOPO.

For each LOPO fold, fit each of v4's 3 LR heads with 5 random_state seeds
(different solver init). Average their softmaxes to get a more stable per-fold
prediction. Should reduce LR variance noise.

Expected gain: +0.5-1.5 pp wF1.
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

from teardrop.data import CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
EPS = 1e-9
SEEDS = [42, 7, 13, 99, 2024]


def _mean_pool_tiles(X_tiles, tile_to_scan, n_scans):
    D = X_tiles.shape[1]
    out = np.zeros((n_scans, D), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int64)
    for i, s in enumerate(tile_to_scan):
        out[s] += X_tiles[i]
        counts[s] += 1
    out /= np.maximum(counts, 1)[:, None]
    return out


def fit_predict_lopo(X, y, groups, seed):
    """Standard person-LOPO with given LR seed. Returns OOF probas (N, 5)."""
    n = len(y)
    oof = np.zeros((n, 5), dtype=np.float32)
    unique_persons = np.unique(groups)
    for p in unique_persons:
        train_mask = groups != p
        test_mask = ~train_mask
        Xt, yt = X[train_mask], y[train_mask]
        Xt_n = normalize(Xt, norm="l2", axis=1)
        sc = StandardScaler().fit(Xt_n)
        Xt_s = sc.transform(Xt_n)
        clf = LogisticRegression(class_weight="balanced", max_iter=3000,
                                 C=1.0, solver="lbfgs", random_state=seed)
        clf.fit(Xt_s, yt)
        Xe = X[test_mask]
        Xe_n = normalize(Xe, norm="l2", axis=1)
        Xe_s = sc.transform(Xe_n)
        oof[test_mask] = clf.predict_proba(Xe_s)
    return oof


def main():
    print("[load] caches")
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz", allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz", allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)

    y = z90["scan_y"]
    groups = z90["scan_groups"]
    n = len(y)

    X90 = _mean_pool_tiles(z90["X"], z90["tile_to_scan"], n)
    X45 = _mean_pool_tiles(z45["X"], z45["tile_to_scan"], n)
    Xbc = zbc["X_scan"].astype(np.float32)
    print(f"  n_scans={n} groups={len(np.unique(groups))}")

    # Per-encoder, per-seed OOF probas
    encoders = [("dinov2_90", X90), ("dinov2_45", X45), ("biomedclip", Xbc)]
    per_seed_per_enc = {}  # (enc_name, seed) -> (240, 5)
    for ename, X in encoders:
        for seed in SEEDS:
            print(f"  fit {ename} seed={seed}")
            per_seed_per_enc[(ename, seed)] = fit_predict_lopo(X, y, groups, seed)

    # Baseline v4: seed=42 only, geometric mean
    p_baseline = np.exp(
        (np.log(per_seed_per_enc[("dinov2_90", 42)] + EPS) +
         np.log(per_seed_per_enc[("dinov2_45", 42)] + EPS) +
         np.log(per_seed_per_enc[("biomedclip", 42)] + EPS)) / 3.0
    )
    p_baseline /= p_baseline.sum(axis=1, keepdims=True)
    pred_baseline = p_baseline.argmax(axis=1)
    f1w_b = f1_score(y, pred_baseline, average="weighted")
    f1m_b = f1_score(y, pred_baseline, average="macro")
    print(f"\n[baseline v4 (seed=42)] wF1={f1w_b:.4f} mF1={f1m_b:.4f}")

    # Variant 1: per-encoder average across 5 seeds, then geometric mean
    avg_per_enc = {ename: np.mean([per_seed_per_enc[(ename, s)] for s in SEEDS], axis=0)
                   for ename, _ in encoders}
    p_v1 = np.exp(sum(np.log(avg_per_enc[e] + EPS) for e, _ in encoders) / 3.0)
    p_v1 /= p_v1.sum(axis=1, keepdims=True)
    pred_v1 = p_v1.argmax(axis=1)
    f1w_v1 = f1_score(y, pred_v1, average="weighted")
    f1m_v1 = f1_score(y, pred_v1, average="macro")
    print(f"[v1 per-encoder mean of 5 seeds → geomean] wF1={f1w_v1:.4f} mF1={f1m_v1:.4f}  Δ={f1w_v1-f1w_b:+.4f}")

    # Variant 2: full 15-way geometric mean (3 enc × 5 seeds)
    log_sum = sum(np.log(per_seed_per_enc[(e, s)] + EPS) for e, _ in encoders for s in SEEDS) / 15.0
    p_v2 = np.exp(log_sum)
    p_v2 /= p_v2.sum(axis=1, keepdims=True)
    pred_v2 = p_v2.argmax(axis=1)
    f1w_v2 = f1_score(y, pred_v2, average="weighted")
    f1m_v2 = f1_score(y, pred_v2, average="macro")
    print(f"[v2 full 15-way geomean]                  wF1={f1w_v2:.4f} mF1={f1m_v2:.4f}  Δ={f1w_v2-f1w_b:+.4f}")

    # Variant 3: per-seed v4, then majority vote
    seed_v4_preds = []
    for s in SEEDS:
        ps = np.exp(sum(np.log(per_seed_per_enc[(e, s)] + EPS) for e, _ in encoders) / 3.0)
        ps /= ps.sum(axis=1, keepdims=True)
        seed_v4_preds.append(ps.argmax(axis=1))
    seed_v4_preds = np.array(seed_v4_preds)  # (5, 240)
    from scipy.stats import mode
    pred_v3, _ = mode(seed_v4_preds, axis=0, keepdims=False)
    f1w_v3 = f1_score(y, pred_v3, average="weighted")
    f1m_v3 = f1_score(y, pred_v3, average="macro")
    print(f"[v3 per-seed v4 majority vote]            wF1={f1w_v3:.4f} mF1={f1m_v3:.4f}  Δ={f1w_v3-f1w_b:+.4f}")

    # Bootstrap CI for best variant vs baseline
    from sklearn.utils import resample
    candidates = [("v1", pred_v1), ("v2", pred_v2), ("v3", pred_v3)]
    print("\n[bootstrap 1000x paired vs baseline]")
    rng = np.random.default_rng(0)
    persons = np.unique(groups)
    for name, pred_v in candidates:
        deltas = []
        for _ in range(1000):
            sample_persons = rng.choice(persons, size=len(persons), replace=True)
            mask = np.isin(groups, sample_persons)
            f_b = f1_score(y[mask], pred_baseline[mask], average="weighted", zero_division=0)
            f_v = f1_score(y[mask], pred_v[mask], average="weighted", zero_division=0)
            deltas.append(f_v - f_b)
        d = np.array(deltas)
        print(f"  {name}: mean Δ={d.mean():+.4f}  CI95=[{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]  P(Δ>0)={(d>0).mean():.3f}")

    # Save
    np.savez(CACHE / "multiseed_v4_predictions.npz",
             baseline_proba=p_baseline, v1_proba=p_v1, v2_proba=p_v2,
             v3_pred=pred_v3, y=y, groups=groups)
    print("\nSaved cache/multiseed_v4_predictions.npz")


if __name__ == "__main__":
    main()
