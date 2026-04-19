"""H2 — L2-normalize embeddings before StandardScaler + LR.

Rationale: DINOv2/BiomedCLIP embeddings live roughly on a hypersphere; cosine
similarity is the natural metric. StandardScaler destroys the unit-norm structure.
L2-normalize first, then StandardScaler, then LR might be better (or worse).

Variants tested:
  baseline: StandardScaler -> LR (current champion path)
  l2_then_std: L2-normalize rows -> StandardScaler -> LR
  l2_only: L2-normalize rows -> LR (no scaler)
  std_then_l2: StandardScaler -> L2-normalize rows -> LR
  also: test effect on both encoders separately AND the ensemble

Using cached TTA embeddings. Cost: ~10s.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import CLASSES  # noqa: E402

CACHE = ROOT / "cache"
N_CLASSES = len(CLASSES)


def lopo_predict(X, y, groups, pipeline="std"):
    n = len(y)
    P = np.zeros((n, N_CLASSES), dtype=np.float64)
    var = X.var(axis=0)
    keep = var > 1e-12
    X_k = X[:, keep]
    for tr, va in leave_one_patient_out(groups):
        Xt = X_k[tr].copy()
        Xv = X_k[va].copy()
        if pipeline == "std":
            sc = StandardScaler()
            Xt = sc.fit_transform(Xt)
            Xv = sc.transform(Xv)
        elif pipeline == "l2_then_std":
            Xt = normalize(Xt, axis=1)
            Xv = normalize(Xv, axis=1)
            sc = StandardScaler()
            Xt = sc.fit_transform(Xt)
            Xv = sc.transform(Xv)
        elif pipeline == "l2_only":
            Xt = normalize(Xt, axis=1)
            Xv = normalize(Xv, axis=1)
        elif pipeline == "std_then_l2":
            sc = StandardScaler()
            Xt = sc.fit_transform(Xt)
            Xv = sc.transform(Xv)
            Xt = normalize(Xt, axis=1)
            Xv = normalize(Xv, axis=1)
        elif pipeline == "none":
            pass
        else:
            raise ValueError(pipeline)
        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
        clf = LogisticRegression(
            class_weight="balanced", max_iter=3000, C=1.0, solver="lbfgs",
        )
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
    print("H2: L2-normalize effect on LR head")
    print("=" * 72)

    zd = np.load(CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz", allow_pickle=True)
    zb = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)

    Xd = zd["X_scan"]; yd = zd["scan_y"]; gd = zd["scan_groups"]
    pd_ = [str(Path(p).resolve()) for p in zd["scan_paths"]]
    Xb = zb["X_scan"]; yb = zb["scan_y"]; gb = zb["scan_groups"]
    pb_ = [str(Path(p).resolve()) for p in zb["scan_paths"]]

    Xb, yb, gb = align_by_path(pd_, pb_, Xb, yb, gb)
    y = yd; groups = gd

    pipelines = ["std", "l2_then_std", "l2_only", "std_then_l2", "none"]
    results = []
    t0 = time.time()
    for pipe in pipelines:
        print(f"\nPipeline: {pipe}")
        Pd = lopo_predict(Xd, y, groups, pipeline=pipe)
        wd, md = f1_of(Pd, y)
        Pb = lopo_predict(Xb, y, groups, pipeline=pipe)
        wb, mb = f1_of(Pb, y)
        # arithmetic and geometric ensembles
        P_ari = 0.5 * (Pd + Pb)
        wa, ma = f1_of(P_ari, y)
        eps = 1e-12
        P_geo = np.exp(0.5 * (np.log(Pd + eps) + np.log(Pb + eps)))
        P_geo /= P_geo.sum(axis=1, keepdims=True)
        wg, mg = f1_of(P_geo, y)

        print(f"  DINOv2-B:   W-F1={wd:.4f}  M-F1={md:.4f}")
        print(f"  BiomedCLIP: W-F1={wb:.4f}  M-F1={mb:.4f}")
        print(f"  Ens arith:  W-F1={wa:.4f}  M-F1={ma:.4f}")
        print(f"  Ens geom:   W-F1={wg:.4f}  M-F1={mg:.4f}")
        results.append({
            "pipeline": pipe,
            "dinov2b_wf1": wd, "dinov2b_mf1": md,
            "biomedclip_wf1": wb, "biomedclip_mf1": mb,
            "ens_arith_wf1": wa, "ens_arith_mf1": ma,
            "ens_geom_wf1": wg, "ens_geom_mf1": mg,
        })

    print(f"\nelapsed: {time.time() - t0:.1f}s")
    champ = 0.6458
    print(f"\n{'Pipeline':20s} {'D-WF1':>8s} {'B-WF1':>8s} {'Ens-arith':>10s} {'Ens-geom':>10s}")
    for r in results:
        print(f"  {r['pipeline']:18s} {r['dinov2b_wf1']:>8.4f} {r['biomedclip_wf1']:>8.4f} "
              f"{r['ens_arith_wf1']:>10.4f} {r['ens_geom_wf1']:>10.4f}")

    best = max(results, key=lambda r: max(r["ens_arith_wf1"], r["ens_geom_wf1"]))
    best_wf1 = max(best["ens_arith_wf1"], best["ens_geom_wf1"])
    print(f"\nBest pipeline (by ensemble W-F1): {best['pipeline']} -> {best_wf1:.4f} "
          f"(delta vs champ {champ} = {best_wf1 - champ:+.4f})")

    import json
    (ROOT / "reports" / "autoresearch_h2_l2norm.json").write_text(
        __import__("json").dumps({"champion_baseline": champ, "results": results}, indent=2)
    )
    print(f"[saved] reports/autoresearch_h2_l2norm.json")


if __name__ == "__main__":
    main()
