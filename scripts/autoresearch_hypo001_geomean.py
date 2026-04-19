"""H1 — Geometric-mean (log-prob average) ensemble vs arithmetic-mean.

Uses cached D4-TTA embeddings for DINOv2-B and BiomedCLIP.

Compares (person-LOPO, raw argmax):
  A. Arithmetic mean of softmaxes (current 0.6458 champion)
  B. Geometric mean of softmaxes (equivalent to averaging log-probs, then softmax)
  C. Arithmetic mean of logits
  D. Geometric mean of logits (softmax-of-log-probs-average; same as B, numerically more stable)

Rationale: arithmetic mean is biased toward the higher-confidence model; geometric
mean is the "consensus of independent experts" operation in Bayesian log-odds
space. At the 0.6458 ceiling, ~0.005-0.015 can come from a better combiner.

Cost: <30s (no encoding, all cached).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import CLASSES  # noqa: E402

CACHE = ROOT / "cache"
N_CLASSES = len(CLASSES)


def lopo_predict(X, y, groups, return_logits=False):
    n = len(y)
    P = np.zeros((n, N_CLASSES), dtype=np.float64)
    L = np.zeros((n, N_CLASSES), dtype=np.float64)
    var = X.var(axis=0)
    keep = var > 1e-12
    X_k = X[:, keep]
    for tr, va in leave_one_patient_out(groups):
        sc = StandardScaler()
        Xt = sc.fit_transform(X_k[tr])
        Xv = sc.transform(X_k[va])
        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
        clf = LogisticRegression(
            class_weight="balanced", max_iter=3000, C=1.0, solver="lbfgs",
        )
        clf.fit(Xt, y[tr])
        proba = clf.predict_proba(Xv)
        logits = clf.decision_function(Xv)
        if logits.ndim == 1:
            # 2-class fallback — not expected
            logits = np.vstack([-logits, logits]).T
        p_full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
        l_full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
        for ci, cls in enumerate(clf.classes_):
            p_full[:, cls] = proba[:, ci]
            l_full[:, cls] = logits[:, ci]
        P[va] = p_full
        L[va] = l_full
    return (P, L) if return_logits else P


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
    print("H1: Geometric-mean vs arithmetic-mean ensemble (TTA)")
    print("=" * 72)

    zd = np.load(CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz", allow_pickle=True)
    zb = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)

    Xd = zd["X_scan"]
    yd = zd["scan_y"]
    gd = zd["scan_groups"]
    pd_ = [str(Path(p).resolve()) for p in zd["scan_paths"]]

    Xb = zb["X_scan"]
    yb = zb["scan_y"]
    gb = zb["scan_groups"]
    pb_ = [str(Path(p).resolve()) for p in zb["scan_paths"]]

    # Align b to d order
    Xb, yb, gb = align_by_path(pd_, pb_, Xb, yb, gb)
    assert np.array_equal(yd, yb)
    assert np.array_equal(gd, gb)
    y = yd
    groups = gd

    t0 = time.time()
    print("\nComputing OOF proba for DINOv2-B (TTA)...")
    Pd, Ld = lopo_predict(Xd, y, groups, return_logits=True)
    print(f"  DINOv2-B alone: W-F1 = {f1_of(Pd, y)[0]:.4f}")

    print("Computing OOF proba for BiomedCLIP (TTA)...")
    Pb, Lb = lopo_predict(Xb, y, groups, return_logits=True)
    print(f"  BiomedCLIP alone: W-F1 = {f1_of(Pb, y)[0]:.4f}")

    print(f"  elapsed: {time.time() - t0:.1f}s")

    print("\n--- Combiners (TTA DINOv2-B + TTA BiomedCLIP) ---")
    rows = []

    # A. Arithmetic mean of probabilities (current champion)
    P_arith = 0.5 * (Pd + Pb)
    rows.append(("Arithmetic mean of softmaxes (CHAMP 0.6458)", *f1_of(P_arith, y)))

    # B. Geometric mean of probabilities
    eps = 1e-12
    log_mean = 0.5 * (np.log(Pd + eps) + np.log(Pb + eps))
    P_geo = np.exp(log_mean)
    P_geo /= P_geo.sum(axis=1, keepdims=True)
    rows.append(("Geometric mean of softmaxes", *f1_of(P_geo, y)))

    # C. Arithmetic mean of logits -> softmax
    L_mean = 0.5 * (Ld + Lb)
    # softmax
    L_shift = L_mean - L_mean.max(axis=1, keepdims=True)
    P_logit_arith = np.exp(L_shift); P_logit_arith /= P_logit_arith.sum(axis=1, keepdims=True)
    rows.append(("Arithmetic mean of logits -> softmax", *f1_of(P_logit_arith, y)))

    # D. Sum of logits (same ranking as arith-mean-of-logits up to scale)
    # equivalent to argmax(L_d + L_b)
    P_sum_logit = (Ld + Lb)
    # For F1 only argmax matters
    rows.append(("Sum of logits (argmax)", *f1_of(P_sum_logit, y)))

    # E. Max-of-softmax (which encoder is more confident per scan)
    P_max = np.where(Pd.max(axis=1, keepdims=True) > Pb.max(axis=1, keepdims=True), Pd, Pb)
    rows.append(("Max-confidence switch per scan", *f1_of(P_max, y)))

    # F. Weighted arithmetic by inverse-entropy (higher weight to more confident model)
    Hd = -np.sum(Pd * np.log(Pd + eps), axis=1, keepdims=True)
    Hb = -np.sum(Pb * np.log(Pb + eps), axis=1, keepdims=True)
    wd = 1.0 / (Hd + 1e-6)
    wb = 1.0 / (Hb + 1e-6)
    P_entw = (wd * Pd + wb * Pb) / (wd + wb)
    rows.append(("Entropy-weighted arithmetic", *f1_of(P_entw, y)))

    # Report
    print(f"\n{'Combiner':50s} {'W-F1':>8s} {'M-F1':>8s}")
    for name, wf1, mf1 in rows:
        print(f"  {name:50s} {wf1:>8.4f} {mf1:>8.4f}")

    champion = 0.6458
    print(f"\nChampion (TTA arith-mean): {champion:.4f}")
    best = max(rows, key=lambda r: r[1])
    print(f"Best: {best[0]} -> W-F1 = {best[1]:.4f} (delta vs champ = {best[1] - champion:+.4f})")

    # Save results
    import json
    outf = ROOT / "reports" / "autoresearch_h1_geomean.json"
    json.dump({
        "champion_baseline": champion,
        "results": [{"combiner": n, "wf1": w, "mf1": m} for (n, w, m) in rows],
    }, outf.open("w"), indent=2)
    print(f"[saved] {outf}")


if __name__ == "__main__":
    main()
