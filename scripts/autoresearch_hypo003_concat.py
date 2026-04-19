"""H4 — Early-fusion (concatenate TTA DINOv2-B + TTA BiomedCLIP) + LR w/ nested C.

Rationale: instead of averaging two separate LR outputs (late fusion), train
one LR on the concatenated [DINOv2-B || BiomedCLIP] feature vector. The shared
LR can learn cross-encoder interactions. Nested CV to pick `C` honestly.

Pipeline choices (from H2):
  - L2-normalize each encoder separately, concat, then StandardScaler (recipes-separately-then-concat
    normalizes each to unit norm before joining — prevents one encoder dominating scale)
  - Also test plain concat

Nested LOPO for C in [0.1, 0.3, 1.0, 3.0, 10.0].

Cost: ~60s (6 folds inner * 35 outer * 5 C values ~ 1050 LR fits, each on 800-row 1280-dim — fast).
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
C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]


def transform_pipe(Xt, Xv, pipeline="l2_then_std"):
    if pipeline == "std":
        sc = StandardScaler()
        Xt = sc.fit_transform(Xt); Xv = sc.transform(Xv)
    elif pipeline == "l2_then_std":
        Xt = normalize(Xt, axis=1); Xv = normalize(Xv, axis=1)
        sc = StandardScaler()
        Xt = sc.fit_transform(Xt); Xv = sc.transform(Xv)
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
    return Xt, Xv


def lopo_predict(X, y, groups, pipeline="l2_then_std", C=1.0):
    n = len(y)
    P = np.zeros((n, N_CLASSES), dtype=np.float64)
    var = X.var(axis=0)
    keep = var > 1e-12
    X_k = X[:, keep]
    for tr, va in leave_one_patient_out(groups):
        Xt, Xv = transform_pipe(X_k[tr], X_k[va], pipeline=pipeline)
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


def lopo_predict_nested_C(X, y, groups, pipeline="l2_then_std", inner_Cs=C_GRID):
    """Outer LOPO; for each outer fold, pick best C on inner-LOPO over the 34 persons."""
    n = len(y)
    P = np.zeros((n, N_CLASSES), dtype=np.float64)
    var = X.var(axis=0)
    keep = var > 1e-12
    X_k = X[:, keep]
    outer_best_Cs = []
    for tr_outer, va_outer in leave_one_patient_out(groups):
        # Inner LOPO for picking C: over the 34 training persons.
        inner_groups = groups[tr_outer]
        inner_X = X_k[tr_outer]
        inner_y = y[tr_outer]

        # Compute OOF proba for each C via inner LOPO
        best_C = None; best_wf1 = -1
        for C in inner_Cs:
            P_inner = np.zeros((len(inner_y), N_CLASSES), dtype=np.float64)
            for tr_in, va_in in leave_one_patient_out(inner_groups):
                Xt, Xv = transform_pipe(inner_X[tr_in], inner_X[va_in], pipeline=pipeline)
                clf = LogisticRegression(
                    class_weight="balanced", max_iter=3000, C=C, solver="lbfgs",
                )
                clf.fit(Xt, inner_y[tr_in])
                proba = clf.predict_proba(Xv)
                p_full = np.zeros((len(va_in), N_CLASSES), dtype=np.float64)
                for ci, cls in enumerate(clf.classes_):
                    p_full[:, cls] = proba[:, ci]
                P_inner[va_in] = p_full
            wf1 = f1_score(inner_y, P_inner.argmax(axis=1), average="weighted", zero_division=0)
            if wf1 > best_wf1:
                best_wf1 = wf1; best_C = C
        outer_best_Cs.append(best_C)

        # Train on full outer training set with best_C, predict outer val
        Xt, Xv = transform_pipe(X_k[tr_outer], X_k[va_outer], pipeline=pipeline)
        clf = LogisticRegression(
            class_weight="balanced", max_iter=3000, C=best_C, solver="lbfgs",
        )
        clf.fit(Xt, y[tr_outer])
        proba = clf.predict_proba(Xv)
        p_full = np.zeros((len(va_outer), N_CLASSES), dtype=np.float64)
        for ci, cls in enumerate(clf.classes_):
            p_full[:, cls] = proba[:, ci]
        P[va_outer] = p_full

    return P, outer_best_Cs


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
    print("H4: Early-fusion concat [TTA D + TTA B] + LR")
    print("=" * 72)

    zd = np.load(CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz", allow_pickle=True)
    zb = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)

    Xd = zd["X_scan"]; y = zd["scan_y"]; groups = zd["scan_groups"]
    pd_ = [str(Path(p).resolve()) for p in zd["scan_paths"]]
    Xb = zb["X_scan"]; yb = zb["scan_y"]; gb = zb["scan_groups"]
    pb_ = [str(Path(p).resolve()) for p in zb["scan_paths"]]
    Xb, yb, gb = align_by_path(pd_, pb_, Xb, yb, gb)

    # L2-normalize each encoder separately, then concat
    Xd_n = normalize(Xd, axis=1)
    Xb_n = normalize(Xb, axis=1)
    X_concat_l2 = np.concatenate([Xd_n, Xb_n], axis=1)
    X_concat_raw = np.concatenate([Xd, Xb], axis=1)

    print(f"Concat shapes: l2={X_concat_l2.shape}  raw={X_concat_raw.shape}")

    t0 = time.time()
    print("\n--- Fixed C=1.0, pipeline=std ---")
    for name, X, pipe in [
        ("concat_raw + std", X_concat_raw, "std"),
        ("concat_l2 + std", X_concat_l2, "std"),
        ("concat_raw + l2_then_std", X_concat_raw, "l2_then_std"),
    ]:
        P = lopo_predict(X, y, groups, pipeline=pipe, C=1.0)
        w, m = f1_of(P, y)
        print(f"  {name:30s} W-F1={w:.4f} M-F1={m:.4f}")

    print("\n--- C sweep (LEAKY upper bound; reporting all for reference) ---")
    for C in C_GRID:
        P = lopo_predict(X_concat_l2, y, groups, pipeline="std", C=C)
        w, m = f1_of(P, y)
        print(f"  concat_l2 + std + C={C:<5}: W-F1={w:.4f} M-F1={m:.4f}")

    print("\n--- Nested LOPO (honest): pick C via inner LOPO ---")
    P_nested_concat_l2, best_Cs_1 = lopo_predict_nested_C(X_concat_l2, y, groups, pipeline="std")
    w, m = f1_of(P_nested_concat_l2, y)
    from collections import Counter
    print(f"  concat_l2 + std + nested C: W-F1={w:.4f} M-F1={m:.4f}")
    print(f"  inner-best Cs: {Counter(best_Cs_1)}")

    # Also: per-encoder OOF at best nested C, then geometric ensemble
    # (compare with late-fusion geom-mean 0.6562 champion)
    print("\n--- Reference: late-fusion (geom-mean of 2 separately-L2-normed LRs) ---")
    def lopo_per_encoder_l2(X):
        return lopo_predict(X, y, groups, pipeline="l2_then_std", C=1.0)
    Pd = lopo_per_encoder_l2(Xd)
    Pb = lopo_per_encoder_l2(Xb)
    eps = 1e-12
    P_geom = np.exp(0.5 * (np.log(Pd + eps) + np.log(Pb + eps)))
    P_geom /= P_geom.sum(axis=1, keepdims=True)
    w, m = f1_of(P_geom, y)
    print(f"  late-fusion geom-mean (H1/H2 winner): W-F1={w:.4f} M-F1={m:.4f}")

    print(f"\nelapsed: {time.time() - t0:.1f}s")

    import json
    (ROOT / "reports" / "autoresearch_h4_concat.json").write_text(
        json.dumps({
            "concat_l2_fixed_C1": f1_of(lopo_predict(X_concat_l2, y, groups, "std", 1.0), y),
            "concat_l2_nested_C": [w, m],
            "best_Cs": list(map(float, best_Cs_1)),
            "late_fusion_geom_mean": f1_of(P_geom, y),
        }, indent=2)
    )
    print(f"[saved] reports/autoresearch_h4_concat.json")


if __name__ == "__main__":
    main()
