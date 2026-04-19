"""H8 — Add TDA or handcrafted as a 3rd ensemble component (geom mean).

Rationale: STATE.md says TDA gives +16% relative on Glaukom but hurts flat-concat.
As a 3rd SOFTMAX component in a GEOMETRIC-MEAN ensemble, it might act as a
sparse Glaukom-specialist without destroying the majority classes (since geom
mean is sensitive to any zero probability, the TDA LR probas have to be smooth
— that's fine with StandardScaler+L2+LR).

Combos tested:
  - H2 winner base = {D(TTA) L2+std LR, B(TTA) L2+std LR} geometric mean
  - + TDA (l2_then_std + LR)
  - + handcrafted (l2_then_std + LR)
  - + both

All uniform-geometric-mean, no weight tuning.

Cost: ~20s — purely linear algebra on cached features.
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
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import CLASSES, enumerate_samples, person_id  # noqa: E402

CACHE = ROOT / "cache"
N_CLASSES = len(CLASSES)


def lopo_predict(X, y, groups, pipeline="l2_then_std", C=1.0):
    n = len(y)
    P = np.zeros((n, N_CLASSES), dtype=np.float64)
    var = X.var(axis=0)
    keep = var > 1e-12
    X_k = X[:, keep]
    for tr, va in leave_one_patient_out(groups):
        Xt = X_k[tr].copy(); Xv = X_k[va].copy()
        if pipeline == "l2_then_std":
            Xt = normalize(Xt, axis=1); Xv = normalize(Xv, axis=1)
            sc = StandardScaler(); Xt = sc.fit_transform(Xt); Xv = sc.transform(Xv)
        elif pipeline == "std":
            sc = StandardScaler(); Xt = sc.fit_transform(Xt); Xv = sc.transform(Xv)
        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
        clf = LogisticRegression(class_weight="balanced", max_iter=3000, C=C, solver="lbfgs")
        clf.fit(Xt, y[tr])
        proba = clf.predict_proba(Xv)
        p_full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
        for ci, cls in enumerate(clf.classes_):
            p_full[:, cls] = proba[:, ci]
        P[va] = p_full
    return P


def f1_of(P, y):
    pred = P.argmax(axis=1)
    return (
        float(f1_score(y, pred, average="weighted", zero_division=0)),
        float(f1_score(y, pred, average="macro", zero_division=0)),
    )


def align_by_path(paths_target, paths_src, X_src, y_src, g_src):
    pmap = {p: i for i, p in enumerate(paths_src)}
    order = np.array([pmap[p] for p in paths_target])
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    return X_src[inv], y_src[inv], g_src[inv]


def main():
    print("=" * 72)
    print("H8: 3-way geom-mean ensemble (TTA D + TTA B + TDA/HC)")
    print("=" * 72)

    # Load TTA embeddings
    zd = np.load(CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz", allow_pickle=True)
    zb = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)
    Xd = zd["X_scan"]; y = zd["scan_y"]; groups = zd["scan_groups"]
    pd_ = [str(Path(p).resolve()) for p in zd["scan_paths"]]
    Xb = zb["X_scan"]; yb = zb["scan_y"]; gb = zb["scan_groups"]
    pb_ = [str(Path(p).resolve()) for p in zb["scan_paths"]]
    Xb, yb, gb = align_by_path(pd_, pb_, Xb, yb, gb)

    # Load TDA / HC features. Align to the same scan ordering via raw path.
    tda = pd.read_parquet(CACHE / "features_tda.parquet")
    hc = pd.read_parquet(CACHE / "features_handcrafted.parquet")
    # Their "raw" column is the path. Build path->index maps.
    tda_paths = [str(Path(p).resolve()) for p in tda["raw"].tolist()]
    hc_paths = [str(Path(p).resolve()) for p in hc["raw"].tolist()]

    tda_feat_cols = [c for c in tda.columns if c not in ("raw", "cls", "label", "patient")]
    hc_feat_cols = [c for c in hc.columns if c not in ("raw", "cls", "label", "patient")]

    X_tda_all = tda[tda_feat_cols].to_numpy(dtype=np.float64)
    X_hc_all = hc[hc_feat_cols].to_numpy(dtype=np.float64)

    # Clean NaN/inf
    X_tda_all = np.nan_to_num(X_tda_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_hc_all = np.nan_to_num(X_hc_all, nan=0.0, posinf=0.0, neginf=0.0)

    # Align to pd_ order
    tda_map = {p: i for i, p in enumerate(tda_paths)}
    hc_map = {p: i for i, p in enumerate(hc_paths)}
    tda_idx = [tda_map.get(p) for p in pd_]
    hc_idx = [hc_map.get(p) for p in pd_]
    if None in tda_idx:
        missing = [p for p, i in zip(pd_, tda_idx) if i is None][:3]
        print(f"WARNING: {sum(1 for i in tda_idx if i is None)} TDA paths missing, e.g. {missing}")
    if None in hc_idx:
        missing = [p for p, i in zip(pd_, hc_idx) if i is None][:3]
        print(f"WARNING: {sum(1 for i in hc_idx if i is None)} HC paths missing, e.g. {missing}")
    tda_idx = [i if i is not None else 0 for i in tda_idx]
    hc_idx = [i if i is not None else 0 for i in hc_idx]
    X_tda = X_tda_all[tda_idx]
    X_hc = X_hc_all[hc_idx]

    print(f"Shapes: D={Xd.shape} B={Xb.shape} TDA={X_tda.shape} HC={X_hc.shape}")

    # Run LOPO for each source
    t0 = time.time()
    print("\nLOPO per encoder/feature (l2_then_std + LR C=1.0)...")
    Pd = lopo_predict(Xd, y, groups, "l2_then_std", 1.0)
    Pb = lopo_predict(Xb, y, groups, "l2_then_std", 1.0)
    P_tda = lopo_predict(X_tda, y, groups, "l2_then_std", 1.0)
    P_hc = lopo_predict(X_hc, y, groups, "l2_then_std", 1.0)

    print(f"  D       W-F1={f1_of(Pd, y)[0]:.4f}")
    print(f"  B       W-F1={f1_of(Pb, y)[0]:.4f}")
    print(f"  TDA     W-F1={f1_of(P_tda, y)[0]:.4f}")
    print(f"  HC      W-F1={f1_of(P_hc, y)[0]:.4f}")

    eps = 1e-12

    def gm(ps):
        s = np.zeros_like(ps[0])
        for p in ps:
            s = s + np.log(p + eps)
        s = s / len(ps)
        r = np.exp(s)
        r = r / r.sum(axis=1, keepdims=True)
        return r

    def am(ps):
        return np.mean(ps, axis=0)

    combos = {
        "D+B (geom)": gm([Pd, Pb]),
        "D+B+TDA (geom)": gm([Pd, Pb, P_tda]),
        "D+B+HC (geom)": gm([Pd, Pb, P_hc]),
        "D+B+TDA+HC (geom)": gm([Pd, Pb, P_tda, P_hc]),
        "D+B (arith)": am([Pd, Pb]),
        "D+B+TDA (arith)": am([Pd, Pb, P_tda]),
        "D+B+HC (arith)": am([Pd, Pb, P_hc]),
        "D+B+TDA+HC (arith)": am([Pd, Pb, P_tda, P_hc]),
    }
    print()
    for name, P in combos.items():
        w, m = f1_of(P, y)
        print(f"  {name:25s} W-F1={w:.4f} M-F1={m:.4f}")

    # Per-class breakdown for the best geom combo
    best_name = max(combos, key=lambda n: f1_of(combos[n], y)[0])
    best_P = combos[best_name]
    best_w, best_m = f1_of(best_P, y)
    print(f"\nBest: {best_name} W-F1={best_w:.4f} M-F1={best_m:.4f}")

    # Print per-class F1 breakdown for best combo
    from sklearn.metrics import classification_report
    pred = best_P.argmax(axis=1)
    print("\nPer-class (best combo):")
    print(classification_report(y, pred, target_names=CLASSES, zero_division=0))

    print(f"\nelapsed: {time.time() - t0:.1f}s")
    champ = 0.6458
    h2_champ = 0.6562
    print(f"\nVs non-TTA champion 0.6458: {best_w - champ:+.4f}")
    print(f"Vs H2-winner 0.6562: {best_w - h2_champ:+.4f}")

    import json
    (ROOT / "reports" / "autoresearch_h8_3way.json").write_text(json.dumps({
        "results": {n: list(f1_of(p, y)) for n, p in combos.items()},
        "best": best_name,
    }, indent=2))


if __name__ == "__main__":
    main()
