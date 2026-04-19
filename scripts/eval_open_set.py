"""Open-set abstention simulation.

We use cached v4 embeddings (90 nm DINOv2-B, 45 nm DINOv2-B, BiomedCLIP-TTA)
to simulate the scenario: "the hidden test set contains a class we never
trained on".  We hold out SucheOko entirely from training, train the v4-recipe
LR heads on the remaining 4 classes, then predict on ALL 240 scans (including
the held-out SucheOko).  The max-softmax score is the open-set detector.

We report:
    * AUROC of max-softmax as known-vs-unknown discriminator
    * At threshold T=p10(correct max-probs on known OOF):
        - TPR on unknown class (SucheOko flagged as UNKNOWN)
        - FPR on known classes (correct-class scan wrongly marked UNKNOWN)
    * Honest OOF analysis for threshold choice (using the current 5-class
      production OOF, what fraction of correct vs wrong predictions exceed T?)

Runtime: uses only cached embeddings, so seconds, not minutes.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import CLASSES, person_id  # noqa: E402
from teardrop.open_set import (  # noqa: E402
    max_softmax_auroc, pick_threshold_from_oof,
)

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
EPS = 1e-12
N_CLASSES_FULL = len(CLASSES)
HELDOUT_CLS = "SucheOko"
HELDOUT_IDX = CLASSES.index(HELDOUT_CLS)


def mean_pool(X_tiles, t2s, n_scans):
    d = X_tiles.shape[1]
    out = np.zeros((n_scans, d), dtype=np.float32)
    for si in range(n_scans):
        m = t2s == si
        if m.any():
            out[si] = X_tiles[m].mean(axis=0)
    return out


def align(paths_ref, paths_src, X_src):
    src_idx = {p: i for i, p in enumerate(paths_src)}
    order = np.array([src_idx[p] for p in paths_ref])
    return X_src[order]


def load_feats():
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz",
                  allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz",
                  allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz",
                  allow_pickle=True)

    paths_90 = [str(p) for p in z90["scan_paths"]]
    y = np.asarray(z90["scan_y"], dtype=np.int64)
    groups = np.array([person_id(Path(p)) for p in paths_90])

    X90 = mean_pool(z90["X"], z90["tile_to_scan"], len(paths_90))
    X45 = align(paths_90, [str(p) for p in z45["scan_paths"]],
                mean_pool(z45["X"], z45["tile_to_scan"],
                          len(z45["scan_paths"])))
    Xbc = align(paths_90, [str(p) for p in zbc["scan_paths"]],
                zbc["X_scan"].astype(np.float32))
    return paths_90, y, groups, X90, X45, Xbc


def lopo_softmax_known_only(X: np.ndarray, y: np.ndarray,
                            groups: np.ndarray,
                            heldout_idx: int,
                            n_remap_classes: int) -> np.ndarray:
    """LOPO on KNOWN-only scans — OOF softmax over (remapped) known classes.

    Returns (n_known, n_remap_classes) matrix of OOF softmaxes.
    Rows correspond to positions where y != heldout_idx, in original order.
    """
    known_mask = y != heldout_idx
    known_idx = np.where(known_mask)[0]
    y_known = y[known_idx]
    groups_known = groups[known_idx]
    # Remap labels to contiguous 0..n-1 by dropping heldout class
    remap = {c: i for i, c in enumerate(sorted(set(y_known.tolist())))}
    y_remap = np.array([remap[c] for c in y_known])

    P = np.zeros((len(known_idx), n_remap_classes), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups_known):
        Xt = normalize(X[known_idx[tr]], norm="l2", axis=1)
        Xv = normalize(X[known_idx[va]], norm="l2", axis=1)
        sc = StandardScaler().fit(Xt)
        Xt = np.nan_to_num(sc.transform(Xt), nan=0.0)
        Xv = np.nan_to_num(sc.transform(Xv), nan=0.0)
        clf = LogisticRegression(class_weight="balanced", max_iter=3000,
                                 C=1.0, solver="lbfgs", n_jobs=4,
                                 random_state=42)
        clf.fit(Xt, y_remap[tr])
        proba = clf.predict_proba(Xv)
        p_full = np.zeros((len(va), n_remap_classes), dtype=np.float64)
        for ci, cls in enumerate(clf.classes_):
            p_full[:, cls] = proba[:, ci]
        P[va] = p_full
    return P, y_remap, remap


def predict_all_with_models_trained_on_known(
        X: np.ndarray, y: np.ndarray, groups: np.ndarray,
        heldout_idx: int, n_remap_classes: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train on all KNOWN scans (no heldout); predict softmax on ALL 240 scans.

    This mimics the production setup where we ship a model trained on the 4
    classes we happened to have, and then the organizer sends a test set
    containing a 5th class we never saw.

    Returns (proba_all, is_unknown_all, remap_dict).
    """
    known_mask = y != heldout_idx
    y_known = y[known_mask]
    remap = {c: i for i, c in enumerate(sorted(set(y_known.tolist())))}
    y_remap = np.array([remap[c] for c in y_known])

    X_known = X[known_mask]
    X_known_norm = normalize(X_known, norm="l2", axis=1)
    sc = StandardScaler().fit(X_known_norm)
    X_fit = np.nan_to_num(sc.transform(X_known_norm), nan=0.0)
    clf = LogisticRegression(class_weight="balanced", max_iter=3000,
                             C=1.0, solver="lbfgs", n_jobs=4,
                             random_state=42)
    clf.fit(X_fit, y_remap)

    X_all_norm = normalize(X, norm="l2", axis=1)
    X_pred = np.nan_to_num(sc.transform(X_all_norm), nan=0.0)
    proba_raw = clf.predict_proba(X_pred)

    # make sure proba is in order of remap values
    proba = np.zeros((len(y), n_remap_classes), dtype=np.float64)
    for ci, cls in enumerate(clf.classes_):
        proba[:, cls] = proba_raw[:, ci]
    is_unknown = (y == heldout_idx)
    return proba, is_unknown, remap


def geom_mean(probs_list: list[np.ndarray]) -> np.ndarray:
    log_sum = np.zeros_like(probs_list[0])
    for P in probs_list:
        log_sum += np.log(P + EPS)
    G = np.exp(log_sum / len(probs_list))
    G /= G.sum(axis=1, keepdims=True)
    return G


def main():
    print("=" * 78)
    print(f"Open-set simulation: hold out '{HELDOUT_CLS}' entirely")
    print("=" * 78)

    paths, y, groups, X90, X45, Xbc = load_feats()
    n = len(y)
    n_heldout = int((y == HELDOUT_IDX).sum())
    n_known = n - n_heldout
    print(f"  total={n}  held-out({HELDOUT_CLS})={n_heldout}  known={n_known}")

    n_known_classes = N_CLASSES_FULL - 1  # 4

    # --- 1. Build KNOWN-only OOF predictions for threshold selection ----
    print("\n[1] build KNOWN-only OOF softmax (v4 recipe)")
    P90_oof, y_remap, remap = lopo_softmax_known_only(
        X90, y, groups, HELDOUT_IDX, n_known_classes)
    P45_oof, _, _ = lopo_softmax_known_only(
        X45, y, groups, HELDOUT_IDX, n_known_classes)
    Pbc_oof, _, _ = lopo_softmax_known_only(
        Xbc, y, groups, HELDOUT_IDX, n_known_classes)
    P_oof = geom_mean([P90_oof, P45_oof, Pbc_oof])
    pred_oof = P_oof.argmax(axis=1)
    oof_f1_w = f1_score(y_remap, pred_oof, average="weighted", zero_division=0)
    oof_f1_m = f1_score(y_remap, pred_oof, average="macro", zero_division=0)
    print(f"  4-class OOF weighted_f1={oof_f1_w:.4f}  macro_f1={oof_f1_m:.4f}")

    # Threshold = 10th percentile of correct-prediction max-probs
    T10 = pick_threshold_from_oof(P_oof, y_remap, correct_floor_pct=10.0)
    T25 = pick_threshold_from_oof(P_oof, y_remap, correct_floor_pct=25.0)
    T50 = pick_threshold_from_oof(P_oof, y_remap, correct_floor_pct=50.0)
    print(f"  threshold candidates (from 4-class known OOF):"
          f"  T10={T10:.4f}  T25={T25:.4f}  T50={T50:.4f}")

    # --- 2. Train on KNOWN only, predict on ALL scans -------------------
    print("\n[2] train on KNOWN-only, predict on ALL 240 scans")
    P90, is_unk, _ = predict_all_with_models_trained_on_known(
        X90, y, groups, HELDOUT_IDX, n_known_classes)
    P45, _, _ = predict_all_with_models_trained_on_known(
        X45, y, groups, HELDOUT_IDX, n_known_classes)
    Pbc, _, _ = predict_all_with_models_trained_on_known(
        Xbc, y, groups, HELDOUT_IDX, n_known_classes)
    P_all = geom_mean([P90, P45, Pbc])

    # NOTE: these predictions on KNOWN-class scans are in-sample and will be
    # optimistically confident; they nevertheless reflect what the SHIPPED
    # model (trained on all-known) would produce on new data from the same
    # 4 known classes at deployment.  Honest separation for AUROC uses the
    # KNOWN-only OOF above (out-of-fold known) vs unknown (in-sample wrt the
    # training set but is a NEW CLASS, so the model has never seen it).

    # Separate known vs unknown scans
    max_p_all = P_all.max(axis=1)
    max_p_known = max_p_all[~is_unk]       # known-class scans, in-sample
    max_p_unknown = max_p_all[is_unk]      # heldout SucheOko, OOD

    # For honest AUROC use the OOF KNOWN softmax vs in-distribution-unknown:
    # we combine the out-of-fold KNOWN confidences (properly disjoint) with
    # the unknown scans (which are genuinely unseen class regardless of CV).
    max_p_known_oof = P_oof.max(axis=1)

    auroc_in_sample = max_softmax_auroc(P_all[~is_unk], P_all[is_unk])
    auroc_honest = max_softmax_auroc(P_oof, P_all[is_unk])
    print(f"  AUROC (in-sample known vs OOD unknown): {auroc_in_sample:.4f}")
    print(f"  AUROC (OOF    known vs OOD unknown):    {auroc_honest:.4f}"
          " <- honest number")

    # --- 3. Threshold scan: T -> TPR on unknown, FPR on known -----------
    print("\n[3] threshold scan")
    rows = []
    for T, lbl in [(T10, "T=p10-correct"),
                   (T25, "T=p25-correct"),
                   (T50, "T=p50-correct"),
                   (0.40, "T=0.40"),
                   (0.50, "T=0.50"),
                   (0.60, "T=0.60"),
                   (0.70, "T=0.70")]:
        tpr_unknown_is = float((max_p_unknown < T).mean())
        fpr_known_is = float((max_p_known < T).mean())
        # Using honest OOF for known-side:
        tpr_unknown_oof = float((max_p_unknown < T).mean())
        fpr_known_oof = float((max_p_known_oof < T).mean())

        # Abstention F1 on held-out fraction: if we abstain correctly on
        # unknown but keep argmax on known, how does the 4-class F1 change?
        # (sanity-check that we don't destroy known-class accuracy).
        pred_oof_abst = pred_oof.copy()
        # when max-prob < T, we'd normally emit UNKNOWN; for 4-class F1 we
        # just note how many known scans we abstain from.
        n_abst_known_oof = int((max_p_known_oof < T).sum())
        rows.append({
            "label": lbl,
            "T": T,
            "TPR_unknown_flagged": tpr_unknown_is,
            "FPR_known_in_sample": fpr_known_is,
            "FPR_known_oof": fpr_known_oof,
            "n_abst_known_oof": n_abst_known_oof,
            "n_unknown_flagged": int((max_p_unknown < T).sum()),
        })
        print(f"  {lbl:18s} T={T:.3f}"
              f"  TPR_unk={tpr_unknown_is:.3f}"
              f"  FPR_known(oof)={fpr_known_oof:.3f}"
              f"  n_unk_flagged={int((max_p_unknown < T).sum())}/{n_heldout}"
              f"  n_known_abst(oof)={n_abst_known_oof}/{n_known}")

    # --- 4. Error breakdown on unknowns --------------------------------
    print("\n[4] breakdown of unknown-class predictions (if NOT abstaining)")
    # Map unknown scans' argmax to original 4-class labels for interpretability
    inv_remap = {v: k for k, v in remap.items()}
    arg_unknown = P_all[is_unk].argmax(axis=1)
    print("  SucheOko scans, confidently-wrongly classified as:")
    from collections import Counter
    cnt = Counter([CLASSES[inv_remap[a]] for a in arg_unknown])
    for k, v in cnt.most_common():
        print(f"    {k:22s} n={v}/{n_heldout}")

    # --- 5. Also: max-softmax ECE-ish (how well correlated with correctness?)
    # For the full 5-class production OOF (best_ensemble_predictions.npz)
    print("\n[5] threshold reliability on PRODUCTION 5-class OOF")
    prod_path = CACHE / "best_ensemble_predictions.npz"
    if prod_path.exists():
        zp = np.load(prod_path, allow_pickle=True)
        proba_prod = zp["proba"]
        y_prod = zp["true_label"]
        pred_prod = proba_prod.argmax(axis=1)
        correct_prod = (pred_prod == y_prod)
        max_p_prod = proba_prod.max(axis=1)
        print(f"  n={len(y_prod)} correct={correct_prod.sum()}"
              f" wrong={(~correct_prod).sum()}")
        print("  At thresholds from 4-class OOF:")
        for T, lbl in [(T10, "T=p10"), (T25, "T=p25"), (T50, "T=p50"),
                       (0.50, "T=0.50"), (0.60, "T=0.60")]:
            pass_correct = (max_p_prod[correct_prod] >= T).mean()
            pass_wrong = (max_p_prod[~correct_prod] >= T).mean()
            print(f"    {lbl:8s} T={T:.3f}"
                  f"  pass_correct={pass_correct:.3f}"
                  f"  pass_wrong={pass_wrong:.3f}"
                  f"  (lower pass_wrong = better)")

    # --- 6. Persist artifacts ------------------------------------------
    out = CACHE / "open_set_sim_sucheoko.npz"
    np.savez(out,
             P_oof_known=P_oof,
             P_all_known_model=P_all,
             y=y, groups=groups, is_unknown=is_unk,
             y_remap_known_oof=y_remap,
             heldout_class=HELDOUT_CLS,
             scan_paths=np.array(paths),
             T10=T10, T25=T25, T50=T50,
             auroc_in_sample=auroc_in_sample,
             auroc_honest=auroc_honest)
    print(f"\n[saved] {out}")

    # --- 7. Report -----------------------------------------------------
    REPORTS.mkdir(exist_ok=True)
    out_md = REPORTS / "OPEN_SET_EVAL.md"
    with open(out_md, "w") as f:
        f.write("# Open-set evaluation — honest heldout of `SucheOko`\n\n")
        f.write(f"Heldout class: **{HELDOUT_CLS}** "
                f"(n={n_heldout} scans, 2 persons)\n\n")
        f.write(f"Known classes trained: "
                f"{[c for c in CLASSES if c != HELDOUT_CLS]}\n\n")
        f.write("## Threshold reliability\n\n")
        f.write(f"- AUROC (in-sample known vs OOD unknown):"
                f" **{auroc_in_sample:.4f}**\n")
        f.write(f"- AUROC (OOF    known vs OOD unknown):"
                f" **{auroc_honest:.4f}** (honest number)\n")
        f.write(f"- 4-class OOF weighted F1 (no heldout):"
                f" {oof_f1_w:.4f} (macro {oof_f1_m:.4f})\n\n")
        f.write("## Threshold scan\n\n")
        f.write("| label | T | TPR unknown | FPR known (in-sample) |"
                " FPR known (OOF) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['label']} | {r['T']:.3f} |"
                    f" {r['TPR_unknown_flagged']:.3f} |"
                    f" {r['FPR_known_in_sample']:.3f} |"
                    f" {r['FPR_known_oof']:.3f} |\n")
        f.write("\n## Unknown-class confidently-misclassified breakdown\n\n")
        for k, v in cnt.most_common():
            f.write(f"- {k}: {v}/{n_heldout}\n")
    print(f"[report] {out_md}")


if __name__ == "__main__":
    main()
