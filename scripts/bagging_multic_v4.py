"""Bagging + multi-C v4 ensemble.

LR with fixed solver+seed is deterministic, so multi-seed gives zero variance.
Real diversity sources: (a) bootstrap subsample of training persons, (b) different
regularization strength C. Combine both.

Each member: bootstrap sample of train persons (with replacement) + LR with
C ∈ {0.1, 0.3, 1.0, 3.0, 10.0}. Gives 25 base learners per encoder.
Average softmaxes per encoder, then geometric mean across encoders.
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

C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
N_BAGS = 5
RNG = np.random.default_rng(42)


def _mean_pool_tiles(X_tiles, tile_to_scan, n_scans):
    D = X_tiles.shape[1]
    out = np.zeros((n_scans, D), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int64)
    for i, s in enumerate(tile_to_scan):
        out[s] += X_tiles[i]
        counts[s] += 1
    out /= np.maximum(counts, 1)[:, None]
    return out


def fit_predict_lopo_bagging(X, y, groups, c_grid, n_bags):
    n = len(y)
    K = 5
    oof = np.zeros((n, K), dtype=np.float32)
    persons = np.unique(groups)
    for p in persons:
        train_mask = groups != p
        test_mask = ~train_mask
        train_persons = np.unique(groups[train_mask])
        Xe = X[test_mask]

        ensemble_sum = np.zeros((Xe.shape[0], K), dtype=np.float32)
        n_members = 0
        for bag in range(n_bags):
            # bootstrap sample of training persons (with replacement)
            sampled = RNG.choice(train_persons, size=len(train_persons), replace=True)
            mask = np.isin(groups, sampled) & train_mask
            Xt = X[mask]
            yt = y[mask]
            # Need ALL 5 classes present (otherwise LR predict_proba is wrong shape)
            if len(np.unique(yt)) < 5:
                continue
            Xt_n = normalize(Xt, norm="l2", axis=1)
            sc = StandardScaler().fit(Xt_n)
            Xt_s = sc.transform(Xt_n)
            Xe_n = normalize(Xe, norm="l2", axis=1)
            Xe_s = sc.transform(Xe_n)
            for C in c_grid:
                clf = LogisticRegression(class_weight="balanced", max_iter=3000,
                                         C=C, solver="lbfgs", random_state=42)
                clf.fit(Xt_s, yt)
                ensemble_sum += clf.predict_proba(Xe_s)
                n_members += 1
        if n_members > 0:
            oof[test_mask] = ensemble_sum / n_members
    return oof


def fit_predict_lopo_basic(X, y, groups):
    n = len(y)
    oof = np.zeros((n, 5), dtype=np.float32)
    for p in np.unique(groups):
        train_mask = groups != p
        test_mask = ~train_mask
        Xt, yt = X[train_mask], y[train_mask]
        Xt_n = normalize(Xt, norm="l2", axis=1)
        sc = StandardScaler().fit(Xt_n)
        Xt_s = sc.transform(Xt_n)
        clf = LogisticRegression(class_weight="balanced", max_iter=3000,
                                 C=1.0, solver="lbfgs", random_state=42)
        clf.fit(Xt_s, yt)
        Xe_n = normalize(X[test_mask], norm="l2", axis=1)
        oof[test_mask] = clf.predict_proba(sc.transform(Xe_n))
    return oof


def main():
    print("[load]")
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz", allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz", allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)

    y = z90["scan_y"]
    groups = z90["scan_groups"]
    n = len(y)

    X90 = _mean_pool_tiles(z90["X"], z90["tile_to_scan"], n)
    X45 = _mean_pool_tiles(z45["X"], z45["tile_to_scan"], n)
    Xbc = zbc["X_scan"].astype(np.float32)
    print(f"  n={n} groups={len(np.unique(groups))}")

    encoders = [("dinov2_90", X90), ("dinov2_45", X45), ("biomedclip", Xbc)]

    # baseline (basic v4)
    print("\n[baseline]")
    p_basic = {}
    for ename, X in encoders:
        print(f"  fit basic {ename}")
        p_basic[ename] = fit_predict_lopo_basic(X, y, groups)
    p_b = np.exp(sum(np.log(p_basic[e] + EPS) for e, _ in encoders) / 3.0)
    p_b /= p_b.sum(axis=1, keepdims=True)
    pred_b = p_b.argmax(axis=1)
    f1w_b = f1_score(y, pred_b, average="weighted")
    f1m_b = f1_score(y, pred_b, average="macro")
    print(f"  baseline v4 wF1={f1w_b:.4f} mF1={f1m_b:.4f}")

    # bagging
    print("\n[bagging+multi-C]")
    p_bag = {}
    for ename, X in encoders:
        print(f"  fit bagging {ename}  (n_bags={N_BAGS} × n_C={len(C_GRID)} = {N_BAGS*len(C_GRID)} members)")
        p_bag[ename] = fit_predict_lopo_bagging(X, y, groups, C_GRID, N_BAGS)
    p_bg = np.exp(sum(np.log(p_bag[e] + EPS) for e, _ in encoders) / 3.0)
    p_bg /= p_bg.sum(axis=1, keepdims=True)
    pred_bg = p_bg.argmax(axis=1)
    f1w = f1_score(y, pred_bg, average="weighted")
    f1m = f1_score(y, pred_bg, average="macro")
    print(f"  bagging v4 wF1={f1w:.4f} mF1={f1m:.4f}  Δ_wF1={f1w-f1w_b:+.4f}  Δ_mF1={f1m-f1m_b:+.4f}")

    # bootstrap
    rng = np.random.default_rng(0)
    persons = np.unique(groups)
    deltas = []
    for _ in range(1000):
        sampled = rng.choice(persons, size=len(persons), replace=True)
        mask = np.isin(groups, sampled)
        f_b = f1_score(y[mask], pred_b[mask], average="weighted", zero_division=0)
        f_v = f1_score(y[mask], pred_bg[mask], average="weighted", zero_division=0)
        deltas.append(f_v - f_b)
    d = np.array(deltas)
    print(f"\n[bootstrap 1000x bagging vs baseline]")
    print(f"  mean Δ={d.mean():+.4f}  CI95=[{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]")
    print(f"  P(Δ>0)={(d>0).mean():.3f}")

    np.savez(CACHE / "bagging_multic_predictions.npz",
             baseline=p_b, bagging=p_bg, y=y, groups=groups)
    print("\nSaved cache/bagging_multic_predictions.npz")


if __name__ == "__main__":
    main()
