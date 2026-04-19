"""H3 — Per-scan z-score and per-encoder-feature standardization via train-set only.

STATE.md insight: "Patient = dominant latent variable. UMAP shows patient clusters
stronger than class clusters." Can we partially remove the patient-axis from the
embedding?

Approach (LEAKAGE-AWARE): within each LOPO outer fold:
  - Compute the per-person mean embedding across the 34 training persons
  - Subtract global train mean from each train scan (already done by StandardScaler)
  - Also normalize within-scan (L2-norm) to remove magnitude differences (H2 winner path)
  - Optional: subtract the patient mean per training scan (but on test we don't know)

The cleanest version is: center feature-dim by train mean, scale by train std,
L2-normalize rows. That is the H2 l2_then_std winner.

Here we test a NEW idea: instead of L2-normalizing the whole scan embedding, remove
a low-rank nuisance subspace identified by the TRAINING persons' mean embeddings.

Protocol:
  1. Stack training scan embeddings X_tr (n_tr, D)
  2. For each training person p, compute its mean embedding mu_p
  3. Center X_tr by subtracting global mean, get centered mu_p's
  4. Take top-k principal components of {mu_p - mu_global} — these are directions
     that explain BETWEEN-PERSON variance
  5. Project out top-k dirs from the full-feature space
  6. Then proceed: StandardScaler + LR

k ∈ {1, 3, 5, 8, 12}. Apply to DINOv2-B and BiomedCLIP TTA.

No tuning on OOF — we pre-register k=3 as the primary. Report all k for transparency.

Cost: ~30s, all operations linear algebra + LR.
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


def project_out_person_dirs(X_tr, groups_tr, X_va, k: int):
    """Return (X_tr_proj, X_va_proj) with top-k person-mean directions removed."""
    # Compute training per-person means
    unique_p = np.unique(groups_tr)
    mu_list = []
    for p in unique_p:
        mu_list.append(X_tr[groups_tr == p].mean(axis=0))
    mu = np.stack(mu_list, axis=0)  # (n_persons, D)
    mu_centered = mu - mu.mean(axis=0, keepdims=True)
    # PCA via SVD
    U, S, Vt = np.linalg.svd(mu_centered, full_matrices=False)
    # Top-k directions are the first k rows of Vt
    k = min(k, Vt.shape[0])
    dirs = Vt[:k]  # (k, D)
    # Project out: X_proj = X - (X @ dirs.T) @ dirs
    X_tr_proj = X_tr - (X_tr @ dirs.T) @ dirs
    X_va_proj = X_va - (X_va @ dirs.T) @ dirs
    return X_tr_proj, X_va_proj


def lopo_predict_with_person_proj(X, y, groups, k: int, pipeline="l2_then_std", C=1.0):
    n = len(y)
    P = np.zeros((n, N_CLASSES), dtype=np.float64)
    var = X.var(axis=0)
    keep = var > 1e-12
    X_k = X[:, keep]
    for tr, va in leave_one_patient_out(groups):
        Xt_raw = X_k[tr].copy()
        Xv_raw = X_k[va].copy()
        if k > 0:
            Xt_raw, Xv_raw = project_out_person_dirs(Xt_raw, groups[tr], Xv_raw, k)

        # Pipeline (same as H2 winner)
        if pipeline == "l2_then_std":
            Xt_raw = normalize(Xt_raw, axis=1)
            Xv_raw = normalize(Xv_raw, axis=1)
            sc = StandardScaler()
            Xt = sc.fit_transform(Xt_raw); Xv = sc.transform(Xv_raw)
        elif pipeline == "std":
            sc = StandardScaler()
            Xt = sc.fit_transform(Xt_raw); Xv = sc.transform(Xv_raw)
        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)

        clf = LogisticRegression(
            class_weight="balanced", max_iter=3000, C=C, solver="lbfgs",
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
    print("H3: Project-out top-k between-person directions from embeddings")
    print("=" * 72)

    zd = np.load(CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz", allow_pickle=True)
    zb = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)
    Xd = zd["X_scan"]; y = zd["scan_y"]; groups = zd["scan_groups"]
    pd_ = [str(Path(p).resolve()) for p in zd["scan_paths"]]
    Xb = zb["X_scan"]; yb = zb["scan_y"]; gb = zb["scan_groups"]
    pb_ = [str(Path(p).resolve()) for p in zb["scan_paths"]]
    Xb, yb, gb = align_by_path(pd_, pb_, Xb, yb, gb)

    t0 = time.time()
    print(f"\n{'k':>4s}  {'D W-F1':>8s}  {'B W-F1':>8s}  {'arith':>8s}  {'geom':>8s}")
    results = []
    for k in [0, 1, 2, 3, 5, 8, 12]:
        Pd = lopo_predict_with_person_proj(Xd, y, groups, k, pipeline="l2_then_std")
        Pb = lopo_predict_with_person_proj(Xb, y, groups, k, pipeline="l2_then_std")
        wd, md = f1_of(Pd, y)
        wb, mb = f1_of(Pb, y)
        P_ari = 0.5 * (Pd + Pb)
        wa, ma = f1_of(P_ari, y)
        eps = 1e-12
        P_geo = np.exp(0.5 * (np.log(Pd + eps) + np.log(Pb + eps)))
        P_geo /= P_geo.sum(axis=1, keepdims=True)
        wg, mg = f1_of(P_geo, y)
        print(f"  {k:>3d}  {wd:>8.4f}  {wb:>8.4f}  {wa:>8.4f}  {wg:>8.4f}")
        results.append({"k": k, "D_wf1": wd, "B_wf1": wb, "arith_wf1": wa, "geom_wf1": wg,
                        "D_mf1": md, "B_mf1": mb, "arith_mf1": ma, "geom_mf1": mg})

    print(f"\nelapsed: {time.time() - t0:.1f}s")

    champ = 0.6458
    champ_geom = 0.6562
    best = max(results, key=lambda r: r["geom_wf1"])
    print(f"\nBest k (by geom ens W-F1): k={best['k']} -> {best['geom_wf1']:.4f} "
          f"(delta vs 0.6458 = {best['geom_wf1']-champ:+.4f}, delta vs H2 0.6562 = {best['geom_wf1']-champ_geom:+.4f})")

    import json
    (ROOT / "reports" / "autoresearch_h3_person_proj.json").write_text(
        json.dumps({"champion_baseline": champ, "h2_winner": champ_geom, "results": results}, indent=2)
    )
    print(f"[saved] reports/autoresearch_h3_person_proj.json")


if __name__ == "__main__":
    main()
