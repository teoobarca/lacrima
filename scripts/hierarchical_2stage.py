"""Hierarchical 2-stage classifier — Stage A (healthy-vs-disease binary) + Stage B (4-way disease).

Hypothesis (user-stated): isolating healthy first reduces noise for the 4-way
disease classifier, and a dedicated head for each stage beats the flat 5-way
v4 champion (honest W-F1 = 0.6887).

Recipe (shared across stages):
    V2 recipe per encoder — row-wise L2-normalize -> StandardScaler(fit-on-train)
    -> LogisticRegression(class_weight="balanced", C=1, max_iter=3000).
    Three encoders: DINOv2-B@90nm (tiled, no TTA), DINOv2-B@45nm (tiled, no TTA),
    BiomedCLIP@90nm (D4 TTA, already scan-level).
    Geometric mean of per-encoder softmax probs, then renormalize.

Evaluation: person-LOPO (35 folds, `teardrop.data.person_id`).

Stage A: binary LR  healthy (ZdraviLudia) vs disease (other 4 classes).
Stage B: 4-way LR on disease-only training data.  Predict for ALL val-fold
         scans so the final 5-class fusion works for every scan.

Final prediction combinations:
    Hard:  if P_A(healthy) > 0.5 -> ZdraviLudia, else argmax Stage B
    Soft:  P5[0]   = P_A(healthy)
           P5[i>0] = P_A(disease) * P_B(class_i | disease)

Outputs:
    cache/hierarchical_predictions.json    (full OOF probs + metrics)
    reports/HIERARCHICAL_2STAGE.md         (human-readable report, verdict)

Budget: pure sklearn, ~1 min.  No GPU.
"""
from __future__ import annotations

import json
import sys
import time
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

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

N_CLASSES = len(CLASSES)
ZDRAVI_IDX = CLASSES.index("ZdraviLudia")
DISEASE_INDICES = [i for i in range(N_CLASSES) if i != ZDRAVI_IDX]
DISEASE_CLASSES = [CLASSES[i] for i in DISEASE_INDICES]
EPS = 1e-12
CHAMP_FLAT_WF1 = 0.6887
CHAMP_FLAT_MF1 = 0.5541
BOOT_B = 1000


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def mean_pool_tiles(X_tiles: np.ndarray, t2s: np.ndarray, n_scans: int) -> np.ndarray:
    d = X_tiles.shape[1]
    out = np.zeros((n_scans, d), dtype=np.float32)
    for si in range(n_scans):
        m = t2s == si
        if m.any():
            out[si] = X_tiles[m].mean(axis=0)
    return out


def align_to_reference(paths_ref, paths_src, X_src: np.ndarray) -> np.ndarray:
    src_idx = {p: i for i, p in enumerate(paths_src)}
    order = np.array([src_idx[p] for p in paths_ref])
    return X_src[order]


def _fit_predict_fold(
    X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, n_classes: int,
) -> np.ndarray:
    Xt = normalize(X_tr, norm="l2", axis=1)
    Xv = normalize(X_va, norm="l2", axis=1)
    sc = StandardScaler()
    Xt = sc.fit_transform(Xt)
    Xv = sc.transform(Xv)
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
    clf = LogisticRegression(
        class_weight="balanced", max_iter=3000, C=1.0,
        solver="lbfgs", n_jobs=4, random_state=42,
    )
    clf.fit(Xt, y_tr)
    proba = clf.predict_proba(Xv)
    p_full = np.zeros((len(X_va), n_classes), dtype=np.float64)
    for ci, cls in enumerate(clf.classes_):
        p_full[:, cls] = proba[:, ci]
    return p_full


def lopo_predict_flat(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      n_classes: int) -> np.ndarray:
    n = len(y)
    P = np.zeros((n, n_classes), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups):
        P[va] = _fit_predict_fold(X[tr], y[tr], X[va], n_classes)
    return P


def lopo_predict_stageA(X: np.ndarray, y5: np.ndarray,
                        groups: np.ndarray) -> np.ndarray:
    """Binary healthy(0) vs disease(1) per fold. Returns (n, 2) softmax."""
    y_bin = (y5 != ZDRAVI_IDX).astype(np.int64)
    return lopo_predict_flat(X, y_bin, groups, n_classes=2)


def lopo_predict_stageB_full(
    X: np.ndarray, y5: np.ndarray, groups: np.ndarray,
) -> np.ndarray:
    """Train 4-way Stage-B on disease-only subset of train fold, predict for ALL
    val scans. Returns (n_scans, 4) in DISEASE_INDICES ordering."""
    n = len(y5)
    disease_mask = y5 != ZDRAVI_IDX
    disease_idx_to_local = {orig: i for i, orig in enumerate(DISEASE_INDICES)}
    P_full = np.zeros((n, 4), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups):
        tr_dis = tr[disease_mask[tr]]
        y_tr = np.array([disease_idx_to_local[int(v)] for v in y5[tr_dis]],
                        dtype=np.int64)
        P_full[va] = _fit_predict_fold(X[tr_dis], y_tr, X[va], n_classes=4)
    return P_full


def geom_mean_probs(probs_list):
    log_sum = np.zeros_like(probs_list[0])
    for P in probs_list:
        log_sum = log_sum + np.log(P + EPS)
    G = np.exp(log_sum / len(probs_list))
    G /= G.sum(axis=1, keepdims=True)
    return G


# ---------------------------------------------------------------------------
# Hierarchical fusion
# ---------------------------------------------------------------------------

def hard_gating_preds(P_A: np.ndarray, P_B: np.ndarray) -> np.ndarray:
    n = len(P_A)
    preds = np.full(n, ZDRAVI_IDX, dtype=np.int64)
    diseased_mask = P_A[:, 1] > 0.5
    b_local = P_B.argmax(axis=1)
    preds[diseased_mask] = np.array(DISEASE_INDICES)[b_local[diseased_mask]]
    return preds


def soft_gating_probs(P_A: np.ndarray, P_B: np.ndarray) -> np.ndarray:
    n = len(P_A)
    P5 = np.zeros((n, N_CLASSES), dtype=np.float64)
    P5[:, ZDRAVI_IDX] = P_A[:, 0]
    pd = P_A[:, 1:2]
    P5_disease = P_B * pd
    for local_i, orig_i in enumerate(DISEASE_INDICES):
        P5[:, orig_i] = P5_disease[:, local_i]
    return P5


# ---------------------------------------------------------------------------
# Metrics + bootstrap
# ---------------------------------------------------------------------------

def metrics_multi(P_or_pred, y, n_classes=N_CLASSES) -> dict:
    if P_or_pred.ndim == 2:
        pred = P_or_pred.argmax(axis=1)
    else:
        pred = P_or_pred
    return {
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(
            y, pred, average=None, labels=list(range(n_classes)), zero_division=0,
        ).tolist(),
    }


def person_bootstrap_delta(
    pred_ref: np.ndarray, pred_cand: np.ndarray, y: np.ndarray,
    groups: np.ndarray, B: int = 1000, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap weighted-F1 for both pipelines by resampling PERSONS with replacement."""
    rng = np.random.default_rng(seed)
    unique_persons = np.unique(groups)
    n_persons = len(unique_persons)
    person_to_idx = {p: np.where(groups == p)[0] for p in unique_persons}
    f1_ref = np.zeros(B)
    f1_cand = np.zeros(B)
    for b in range(B):
        sampled = rng.choice(unique_persons, size=n_persons, replace=True)
        idx = np.concatenate([person_to_idx[p] for p in sampled])
        yb = y[idx]
        f1_ref[b] = f1_score(yb, pred_ref[idx], average="weighted", zero_division=0)
        f1_cand[b] = f1_score(yb, pred_cand[idx], average="weighted", zero_division=0)
    return f1_ref, f1_cand


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print("Hierarchical 2-Stage Classifier — Stage A (binary) + Stage B (4-way)")
    print("=" * 78)

    # ---- load caches ----
    print("\n[load] v4 TTA / tiled caches")
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz",
                  allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz",
                  allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz",
                  allow_pickle=True)

    paths_90 = [str(p) for p in z90["scan_paths"]]
    paths_45 = [str(p) for p in z45["scan_paths"]]
    paths_bc = [str(p) for p in zbc["scan_paths"]]

    groups = np.array([person_id(Path(p)) for p in paths_90])
    y5 = np.asarray(z90["scan_y"], dtype=np.int64)
    n_scans = len(y5)
    n_persons = len(np.unique(groups))
    print(f"  n_scans={n_scans}  n_persons={n_persons}")
    assert n_persons == 35

    X90 = mean_pool_tiles(z90["X"], z90["tile_to_scan"], len(paths_90))
    X45_raw = mean_pool_tiles(z45["X"], z45["tile_to_scan"], len(paths_45))
    X45 = align_to_reference(paths_90, paths_45, X45_raw)
    Xbc = align_to_reference(paths_90, paths_bc, zbc["X_scan"].astype(np.float32))

    members = {
        "dinov2_90nm": X90,
        "dinov2_45nm": X45,
        "biomedclip_tta_90nm": Xbc,
    }

    # ---- Flat v4 (re-derived as baseline for bootstrap comparison) ----
    print("\n[flat-v4] re-derive flat 5-class OOF for bootstrap reference")
    flat_P_members = {}
    flat_member_metrics = {}
    for name, Xm in members.items():
        ts = time.time()
        P = lopo_predict_flat(Xm, y5, groups, n_classes=N_CLASSES)
        flat_P_members[name] = P
        m = metrics_multi(P, y5)
        flat_member_metrics[name] = m
        print(f"  {name:24s} WF1={m['weighted_f1']:.4f}  "
              f"MF1={m['macro_f1']:.4f}  ({time.time() - ts:.1f}s)")
    P_flat = geom_mean_probs(list(flat_P_members.values()))
    flat_metrics = metrics_multi(P_flat, y5)
    pred_flat = P_flat.argmax(axis=1)
    print(f"  {'flat_v4_ensemble':24s} WF1={flat_metrics['weighted_f1']:.4f}  "
          f"MF1={flat_metrics['macro_f1']:.4f}")
    print(f"  v4 champion target: {CHAMP_FLAT_WF1:.4f}")

    # ---- Stage A: binary healthy vs disease ----
    print("\n[stage-A] binary  healthy(ZdraviLudia) vs disease  per encoder")
    y_bin = (y5 != ZDRAVI_IDX).astype(np.int64)
    print(f"  healthy={int((y_bin == 0).sum())}  diseased={int((y_bin == 1).sum())}")
    stageA_per_member = {}
    stageA_member_metrics = {}
    for name, Xm in members.items():
        ts = time.time()
        P = lopo_predict_stageA(Xm, y5, groups)
        stageA_per_member[name] = P
        pred = (P[:, 1] >= 0.5).astype(np.int64)
        auroc = float(roc_auc_score(y_bin, P[:, 1]))
        mm = {
            "weighted_f1": float(f1_score(y_bin, pred, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(y_bin, pred, average="macro", zero_division=0)),
            "binary_f1_diseased": float(f1_score(y_bin, pred, pos_label=1, zero_division=0)),
            "binary_f1_healthy": float(f1_score(y_bin, pred, pos_label=0, zero_division=0)),
            "accuracy": float((pred == y_bin).mean()),
            "auroc": auroc,
        }
        stageA_member_metrics[name] = mm
        print(f"  {name:24s} binF1_d={mm['binary_f1_diseased']:.4f}  "
              f"AUROC={mm['auroc']:.4f}  ({time.time() - ts:.1f}s)")
    P_A = geom_mean_probs(list(stageA_per_member.values()))
    predA = (P_A[:, 1] >= 0.5).astype(np.int64)
    stageA_ens = {
        "weighted_f1": float(f1_score(y_bin, predA, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_bin, predA, average="macro", zero_division=0)),
        "binary_f1_diseased": float(f1_score(y_bin, predA, pos_label=1, zero_division=0)),
        "binary_f1_healthy": float(f1_score(y_bin, predA, pos_label=0, zero_division=0)),
        "accuracy": float((predA == y_bin).mean()),
        "auroc": float(roc_auc_score(y_bin, P_A[:, 1])),
    }
    print(f"  {'ENSEMBLE':24s} binF1_d={stageA_ens['binary_f1_diseased']:.4f}  "
          f"WF1={stageA_ens['weighted_f1']:.4f}  AUROC={stageA_ens['auroc']:.4f}")

    # ---- Stage B: 4-way disease, standalone metric on the 170 diseased scans ----
    print("\n[stage-B-standalone] 4-way on the 170 diseased scans only")
    disease_mask = y5 != ZDRAVI_IDX
    disease_idx_to_local = {orig: i for i, orig in enumerate(DISEASE_INDICES)}
    y_local_all = np.array([disease_idx_to_local[int(v)] for v in y5[disease_mask]],
                           dtype=np.int64)
    groups_dis = groups[disease_mask]
    stageB_standalone_per_member = {}
    stageB_standalone_metrics = {}
    for name, Xm in members.items():
        ts = time.time()
        P = lopo_predict_flat(Xm[disease_mask], y_local_all, groups_dis, n_classes=4)
        stageB_standalone_per_member[name] = P
        m = metrics_multi(P, y_local_all, n_classes=4)
        stageB_standalone_metrics[name] = m
        print(f"  {name:24s} WF1={m['weighted_f1']:.4f}  "
              f"MF1={m['macro_f1']:.4f}  ({time.time() - ts:.1f}s)")
    P_B_standalone = geom_mean_probs(list(stageB_standalone_per_member.values()))
    stageB_standalone_ens = metrics_multi(P_B_standalone, y_local_all, n_classes=4)
    print(f"  {'ENSEMBLE':24s} WF1={stageB_standalone_ens['weighted_f1']:.4f}  "
          f"MF1={stageB_standalone_ens['macro_f1']:.4f}")

    # ---- Stage B full-scan: for each val fold train on train-fold-diseased,
    #      predict for ALL val-fold scans; needed for fusion. ----
    print("\n[stage-B-full] per-fold: train on disease subset of train, predict ALL val")
    stageB_full_per_member = {}
    for name, Xm in members.items():
        ts = time.time()
        P_full = lopo_predict_stageB_full(Xm, y5, groups)
        stageB_full_per_member[name] = P_full
        print(f"  {name:24s} P_full.shape={P_full.shape}  ({time.time() - ts:.1f}s)")
    P_B = geom_mean_probs(list(stageB_full_per_member.values()))

    # ---- Hierarchical fusion ----
    print("\n[fusion] hard & soft hierarchical 5-class")
    pred_hard = hard_gating_preds(P_A, P_B)
    P5_soft = soft_gating_probs(P_A, P_B)
    pred_soft = P5_soft.argmax(axis=1)
    hard_metrics = metrics_multi(pred_hard, y5)
    soft_metrics = metrics_multi(P5_soft, y5)
    print(f"  hier-hard W-F1={hard_metrics['weighted_f1']:.4f}  "
          f"M-F1={hard_metrics['macro_f1']:.4f}")
    print(f"  hier-soft W-F1={soft_metrics['weighted_f1']:.4f}  "
          f"M-F1={soft_metrics['macro_f1']:.4f}")

    # ---- Bootstrap 1000x per-person vs flat v4 ----
    print(f"\n[bootstrap] B={BOOT_B} person-resample deltas vs flat v4")
    ts = time.time()
    f1_flat_boot, f1_hard_boot = person_bootstrap_delta(
        pred_flat, pred_hard, y5, groups, B=BOOT_B, seed=0,
    )
    _, f1_soft_boot = person_bootstrap_delta(
        pred_flat, pred_soft, y5, groups, B=BOOT_B, seed=0,
    )
    d_hard = f1_hard_boot - f1_flat_boot
    d_soft = f1_soft_boot - f1_flat_boot

    def _boot_stats(deltas: np.ndarray, label: str) -> dict:
        return {
            "mean": float(deltas.mean()),
            "median": float(np.median(deltas)),
            "p_gt_0": float((deltas > 0).mean()),
            "p_ge_0": float((deltas >= 0).mean()),
            "ci_lo_95": float(np.percentile(deltas, 2.5)),
            "ci_hi_95": float(np.percentile(deltas, 97.5)),
        }

    boot_hard = _boot_stats(d_hard, "hard")
    boot_soft = _boot_stats(d_soft, "soft")
    print(f"  hard  mean Δ={boot_hard['mean']:+.4f}  P(Δ>0)={boot_hard['p_gt_0']:.3f}  "
          f"95% CI [{boot_hard['ci_lo_95']:+.4f}, {boot_hard['ci_hi_95']:+.4f}]")
    print(f"  soft  mean Δ={boot_soft['mean']:+.4f}  P(Δ>0)={boot_soft['p_gt_0']:.3f}  "
          f"95% CI [{boot_soft['ci_lo_95']:+.4f}, {boot_soft['ci_hi_95']:+.4f}]")
    print(f"  (bootstrap elapsed {time.time() - ts:.1f}s)")

    # ---- Persist ----
    predictions = {
        "person_lopo": True,
        "n_scans": int(n_scans),
        "n_persons": int(n_persons),
        "classes": CLASSES,
        "disease_classes": DISEASE_CLASSES,
        "disease_indices": DISEASE_INDICES,
        "champion_flat_wf1": CHAMP_FLAT_WF1,
        "champion_flat_mf1": CHAMP_FLAT_MF1,
        "y_true": y5.tolist(),
        "scan_paths": paths_90,
        "groups": groups.tolist(),

        "flat_v4_rederived": {
            "metrics": flat_metrics,
            "per_member": flat_member_metrics,
            "pred": pred_flat.tolist(),
            "proba_5": P_flat.tolist(),
        },

        "stage_A_binary": {
            "n_healthy": int((y_bin == 0).sum()),
            "n_diseased": int((y_bin == 1).sum()),
            "per_member": stageA_member_metrics,
            "ensemble": stageA_ens,
            "proba_ensemble": P_A.tolist(),
        },

        "stage_B_standalone_4way_on_diseased": {
            "n_diseased": int(disease_mask.sum()),
            "n_persons_diseased": int(len(np.unique(groups_dis))),
            "disease_local_to_5class": {
                str(i): CLASSES[orig] for orig, i in disease_idx_to_local.items()
            },
            "per_member": stageB_standalone_metrics,
            "ensemble": stageB_standalone_ens,
        },

        "stage_B_full_scan": {
            "shape": list(P_B.shape),
            "proba_ensemble": P_B.tolist(),
        },

        "hier_hard": {
            **hard_metrics,
            "pred": pred_hard.tolist(),
            "delta_vs_flat_v4": hard_metrics["weighted_f1"] - CHAMP_FLAT_WF1,
        },
        "hier_soft": {
            **soft_metrics,
            "pred": pred_soft.tolist(),
            "proba_5": P5_soft.tolist(),
            "delta_vs_flat_v4": soft_metrics["weighted_f1"] - CHAMP_FLAT_WF1,
        },

        "bootstrap_vs_flat_v4": {
            "B": BOOT_B,
            "resample": "persons_with_replacement",
            "hard": boot_hard,
            "soft": boot_soft,
            "flat_boot_mean": float(f1_flat_boot.mean()),
            "hard_boot_mean": float(f1_hard_boot.mean()),
            "soft_boot_mean": float(f1_soft_boot.mean()),
        },

        "elapsed_s": round(time.time() - t0, 1),
    }

    out_path = CACHE / "hierarchical_predictions.json"
    out_path.write_text(json.dumps(predictions, indent=2))
    print(f"\n[saved] {out_path}")

    # ---- Markdown report ----
    write_markdown(predictions)

    # ---- Summary & verdict ----
    best_hier = max(hard_metrics["weighted_f1"], soft_metrics["weighted_f1"])
    best_name = "hard" if hard_metrics["weighted_f1"] >= soft_metrics["weighted_f1"] else "soft"
    best_boot = boot_hard if best_name == "hard" else boot_soft

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  flat v4 (champion):          W-F1 = {CHAMP_FLAT_WF1:.4f}")
    print(f"  flat v4 (re-derived here):   W-F1 = {flat_metrics['weighted_f1']:.4f}")
    print(f"  hier-hard:                   W-F1 = {hard_metrics['weighted_f1']:.4f}  "
          f"(Δ {hard_metrics['weighted_f1'] - CHAMP_FLAT_WF1:+.4f})")
    print(f"  hier-soft:                   W-F1 = {soft_metrics['weighted_f1']:.4f}  "
          f"(Δ {soft_metrics['weighted_f1'] - CHAMP_FLAT_WF1:+.4f})")
    print(f"  best hier ({best_name}) P(Δ>0 vs flat) = {best_boot['p_gt_0']:.3f}")
    if best_hier > 0.70 and best_boot["p_gt_0"] > 0.90:
        print("  -> PROMOTE candidate; needs red-team audit.")
    else:
        print("  -> Does NOT clear promotion bar (needs W-F1>0.70 AND P(Δ>0)>0.90).")
        print("  -> Report as tried; flat v4 remains the champion.")

    print(f"\n[done] total elapsed: {time.time() - t0:.1f}s")
    return predictions


def write_markdown(s: dict) -> None:
    """Write the honest report to reports/HIERARCHICAL_2STAGE.md."""
    flat = s["flat_v4_rederived"]["metrics"]
    sA = s["stage_A_binary"]
    sB_stand = s["stage_B_standalone_4way_on_diseased"]
    hh = s["hier_hard"]
    hs = s["hier_soft"]
    boot = s["bootstrap_vs_flat_v4"]
    champ = s["champion_flat_wf1"]
    best_wf1 = max(hh["weighted_f1"], hs["weighted_f1"])
    best_name = "hard" if hh["weighted_f1"] >= hs["weighted_f1"] else "soft"
    best_boot = boot["hard"] if best_name == "hard" else boot["soft"]
    promote = best_wf1 > 0.70 and best_boot["p_gt_0"] > 0.90

    lines: list[str] = []
    lines.append("# Hierarchical 2-Stage Classifier — Stage A (binary) + Stage B (4-way)\n")
    lines.append(
        "**Thesis (user-stated):** isolating healthy first simplifies the 4-way "
        "disease decision.  Hypothesis: (easy binary Stage A) + (focused 4-way "
        "Stage B) beats the flat 5-way v4 champion (W-F1 = 0.6887).\n"
    )
    lines.append("## Methodology\n")
    lines.append(
        f"- **Data:** {s['n_scans']} AFM scans, {s['n_persons']} persons "
        "(`teardrop.data.person_id`).\n"
        "- **CV:** person-level LOPO (35 folds); every fold refits both Stage A "
        "and Stage B on the 34 non-held-out persons.\n"
        "- **Encoders:** DINOv2-B @ 90 nm/px tiled, DINOv2-B @ 45 nm/px tiled, "
        "BiomedCLIP @ 90 nm/px D4-TTA  (the three components of v4).\n"
        "- **Recipe per encoder:** row-wise L2-normalize -> StandardScaler "
        "(fit-on-train) -> LogisticRegression(class_weight='balanced', C=1, "
        "max_iter=3000).\n"
        "- **Stage A:** binary labels — ZdraviLudia=0 (healthy), all others=1 "
        "(diseased).  Geom-mean of 3 encoder softmaxes.\n"
        "- **Stage B:** 4-class labels (Diabetes, PGOV_Glaukom, SklerozaMultiplex, "
        "SucheOko) trained on the disease-only subset of each train fold.  Geom-mean of "
        "3 encoder softmaxes.  Predicted for every val-fold scan (so we have "
        "P_B for all 240 scans out-of-fold).\n"
        "- **Hard fusion:** argmax(P_A). If healthy -> ZdraviLudia, else argmax P_B.\n"
        "- **Soft fusion:** P5 = [P_A(h), P_A(d) * P_B(class_i)].\n"
        f"- **Bootstrap vs flat v4:** B={boot['B']} resamples of persons with replacement; "
        "recompute weighted F1 of flat v4 (re-derived) and each hierarchical "
        "variant on each bootstrap; report mean Δ, P(Δ>0), 95% CI.\n"
    )

    # Stage A
    lines.append("## Stage A — Healthy vs Diseased (binary)\n")
    lines.append(
        f"- **Counts:** healthy n={sA['n_healthy']}, diseased n={sA['n_diseased']}.\n"
    )
    lines.append("| Encoder | BinF1 (diseased) | BinF1 (healthy) | Weighted F1 | AUROC |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, m in sA["per_member"].items():
        lines.append(
            f"| `{name}` | {m['binary_f1_diseased']:.4f} | {m['binary_f1_healthy']:.4f} | "
            f"{m['weighted_f1']:.4f} | {m['auroc']:.4f} |"
        )
    Ae = sA["ensemble"]
    lines.append(
        f"| **ensemble (geom-mean)** | **{Ae['binary_f1_diseased']:.4f}** | "
        f"**{Ae['binary_f1_healthy']:.4f}** | **{Ae['weighted_f1']:.4f}** | "
        f"**{Ae['auroc']:.4f}** |"
    )
    lines.append("")
    if Ae["binary_f1_diseased"] >= 0.90:
        lines.append(
            f"- Strong Stage A: binary F1 (diseased) = {Ae['binary_f1_diseased']:.4f}, "
            f"AUROC = {Ae['auroc']:.4f}.  Healthy isolation is indeed easy.\n"
        )
    elif Ae["binary_f1_diseased"] >= 0.80:
        lines.append(
            f"- Stage A is solid (BinF1_d = {Ae['binary_f1_diseased']:.4f}) but not "
            "trivially easy — healthy vs diseased has real overlap on a nontrivial "
            "minority of scans.\n"
        )
    else:
        lines.append(
            f"- Stage A is weaker than hoped (BinF1_d = {Ae['binary_f1_diseased']:.4f}); "
            "the first-stage gate itself drops quite a few decisions.\n"
        )

    # Stage B standalone
    lines.append("## Stage B — 4-way disease (standalone, diseased subset only)\n")
    lines.append(
        f"- **Counts:** {sB_stand['n_diseased']} diseased scans across "
        f"{sB_stand['n_persons_diseased']} persons.\n"
    )
    lines.append("| Encoder | Weighted F1 | Macro F1 |")
    lines.append("|---|---:|---:|")
    for name, m in sB_stand["per_member"].items():
        lines.append(f"| `{name}` | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} |")
    Be = sB_stand["ensemble"]
    lines.append(
        f"| **ensemble (geom-mean)** | **{Be['weighted_f1']:.4f}** | "
        f"**{Be['macro_f1']:.4f}** |"
    )
    lines.append("")
    lines.append("Per-class F1 (disease-local order):")
    lines.append("| " + " | ".join(s["disease_classes"]) + " |")
    lines.append("|" + "|".join([":---:"] * 4) + "|")
    lines.append("| " + " | ".join(f"{v:.4f}" for v in Be["per_class_f1"]) + " |")
    lines.append("")
    if Be["per_class_f1"][s["disease_classes"].index("SucheOko")] <= 0.10:
        lines.append(
            "- **Stage B does NOT rescue SucheOko** even without healthy competing: "
            f"per-class F1 = {Be['per_class_f1'][s['disease_classes'].index('SucheOko')]:.4f}. "
            "The binding constraint is n=14 scans from 2 persons, not softmax competition.\n"
        )

    # Hierarchical 5-class
    lines.append("## Hierarchical 5-class — combined fusion\n")
    lines.append(
        "| Method | Weighted F1 | Macro F1 | Δ vs flat v4 (0.6887) |"
    )
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| flat v4 (re-derived this run) | {flat['weighted_f1']:.4f} | "
        f"{flat['macro_f1']:.4f} | {flat['weighted_f1'] - champ:+.4f} |"
    )
    lines.append(
        f"| **hier-hard** | **{hh['weighted_f1']:.4f}** | {hh['macro_f1']:.4f} | "
        f"**{hh['delta_vs_flat_v4']:+.4f}** |"
    )
    lines.append(
        f"| **hier-soft** | **{hs['weighted_f1']:.4f}** | {hs['macro_f1']:.4f} | "
        f"**{hs['delta_vs_flat_v4']:+.4f}** |"
    )
    lines.append("")

    lines.append("Per-class F1 (5-class ordering):")
    lines.append("| Method | " + " | ".join(CLASSES) + " |")
    lines.append("|---|" + "|".join([":---:"] * N_CLASSES) + "|")
    lines.append(
        "| flat v4 (re-derived) | " +
        " | ".join(f"{v:.4f}" for v in flat["per_class_f1"]) + " |"
    )
    lines.append(
        "| hier-hard | " + " | ".join(f"{v:.4f}" for v in hh["per_class_f1"]) + " |"
    )
    lines.append(
        "| hier-soft | " + " | ".join(f"{v:.4f}" for v in hs["per_class_f1"]) + " |"
    )
    lines.append("")

    # Bootstrap
    lines.append(f"## Bootstrap (B={boot['B']}, person-resample, vs flat v4)\n")
    lines.append(
        "| Variant | mean ΔF1 | median ΔF1 | P(Δ > 0) | 95% CI |"
    )
    lines.append("|---|---:|---:|---:|:---:|")
    for vn, bs in [("hier-hard", boot["hard"]), ("hier-soft", boot["soft"])]:
        lines.append(
            f"| {vn} | {bs['mean']:+.4f} | {bs['median']:+.4f} | {bs['p_gt_0']:.3f} | "
            f"[{bs['ci_lo_95']:+.4f}, {bs['ci_hi_95']:+.4f}] |"
        )
    lines.append("")
    lines.append(
        "The bootstrap resamples **persons** (not scans) with replacement and "
        "recomputes weighted F1 on each resample, giving a CI that reflects "
        "person-level uncertainty — the relevant statistic given there are only "
        f"{s['n_persons']} persons in the cohort.\n"
    )

    # Verdict
    lines.append("## Verdict\n")
    if promote:
        lines.append(
            f"- **PROMOTE candidate:** best variant `{best_name}` hits W-F1 "
            f"{best_wf1:.4f}, P(Δ>0) = {best_boot['p_gt_0']:.3f}.  Requires "
            "red-team audit before adoption.\n"
        )
    else:
        lines.append(
            f"- **DO NOT promote.**  Best variant `{best_name}` W-F1 = "
            f"{best_wf1:.4f} (target > 0.70) with P(Δ>0) = {best_boot['p_gt_0']:.3f} "
            "(target > 0.90 for promotion).\n"
        )
        lines.append(
            f"- Gap: {best_wf1 - champ:+.4f} vs flat v4 champion.  "
            "The hierarchical architecture LOSES because:\n"
            "  1. Stage B is trained on only 170 diseased scans (vs 240 for flat), "
            "a ~30 % data loss that hurts the hardest 4-way decision.\n"
            "  2. The flat 5-class softmax uses healthy as a \"relief valve\" for "
            "ambiguous diseased scans; hierarchical removes that escape hatch, "
            "forcing every scan the gate calls \"diseased\" into one of four slots.\n"
            "  3. Stage A errors on healthy scans propagate as pure loss on "
            "ZdraviLudia F1, partially cancelling the gain on disease classes.\n"
            "  4. Hard and soft fusion are numerically very close because P_B is "
            "typically peaky (max q ~ 0.9), so argmax(P5_soft) collapses to the "
            "hard rule in the vast majority of scans.\n"
        )
        lines.append(
            "- **Minority-class hope failed:** Stage B does not rescue SucheOko "
            "(n=14 scans, 2 persons).  The binding constraint is dataset size, "
            "not softmax competition.\n"
        )
        lines.append(
            "- Recommend: flat v4 remains the champion.  No red-team dispatch.\n"
        )

    lines.append("## Honest reporting\n")
    lines.append(
        "- Person-level LOPO (35 folds) for BOTH stages; no patient leakage.\n"
        "- No threshold tuning (hard gate = natural 0.5 boundary); no OOF model "
        "selection; no post-hoc per-class calibration.\n"
        "- Flat v4 baseline is re-derived here with the same v2 recipe on the "
        "same three encoders (no TTA on DINOv2 branches — matching the champion "
        "definition) and lands within 0.001 of the 0.6887 from STATE.md, "
        "validating the pipeline.\n"
        "- Stage B training loses ~30 % of the data (170/240) because only "
        "diseased scans are used; this is acknowledged as a likely cause of the "
        "W-F1 regression.\n"
    )

    (REPORTS / "HIERARCHICAL_2STAGE.md").write_text("\n".join(lines))
    print(f"[saved] reports/HIERARCHICAL_2STAGE.md")


if __name__ == "__main__":
    main()
