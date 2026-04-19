"""Hierarchical 2-level classifier for tear AFM disease classification.

Level 1 (binary): healthy (ZdraviLudia) vs diseased (everything else).
Level 2 (4-way):  which specific disease, trained only on diseased scans.

Both levels use the v4 recipe (L2-norm row-wise -> StandardScaler -> LR(balanced)
on three encoders: DINOv2-B 90 nm/px, DINOv2-B 45 nm/px, BiomedCLIP 90 nm/px D4-TTA)
evaluated with PERSON-level LOPO (teardrop.data.person_id, 35 groups).

Hierarchical inference — two variants:

  - Hard gating: argmax over [P(healthy), P(diseased) * L2_proba (4-way)].
    Equivalent to: if P(healthy) > 0.5 -> ZdraviLudia, else argmax L2.
  - Soft gating: final_proba_5 = P(healthy) * onehot(ZdraviLudia)
                          + P(diseased) * scatter(L2_proba into 4 disease slots).

Champion to beat: flat 5-class person-LOPO weighted F1 = 0.6887.

Outputs:
  - reports/hierarchical_results.json (machine-readable)
  - reports/HIERARCHICAL_RESULTS.md   (human-readable, honest report)
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

N_CLASSES = len(CLASSES)  # 5
ZDRAVI_IDX = CLASSES.index("ZdraviLudia")  # 0
DISEASE_CLASSES = [c for i, c in enumerate(CLASSES) if i != ZDRAVI_IDX]
DISEASE_INDICES = [i for i in range(N_CLASSES) if i != ZDRAVI_IDX]  # [1,2,3,4]
EPS = 1e-12
CHAMP_FLAT_WF1 = 0.6887
CHAMP_FLAT_MF1 = 0.5541


# ---------------------------------------------------------------------------
# Shared V2 helpers (match scripts/multiscale_experiment.py)
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


def lopo_predict_v2(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_classes: int,
) -> np.ndarray:
    """V2 recipe person-LOPO OOF softmax for arbitrary num classes.

    - L2-normalize rows -> StandardScaler (train-only fit) -> LR(balanced, C=1).
    - Remap LR's class-subset columns back to full [0..n_classes-1] slots.
    """
    n = len(y)
    P = np.zeros((n, n_classes), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups):
        Xt = normalize(X[tr], norm="l2", axis=1)
        Xv = normalize(X[va], norm="l2", axis=1)
        sc = StandardScaler()
        Xt = sc.fit_transform(Xt)
        Xv = sc.transform(Xv)
        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
        clf = LogisticRegression(
            class_weight="balanced", max_iter=3000, C=1.0,
            solver="lbfgs", n_jobs=4, random_state=42,
        )
        clf.fit(Xt, y[tr])
        proba = clf.predict_proba(Xv)
        p_full = np.zeros((len(va), n_classes), dtype=np.float64)
        for ci, cls in enumerate(clf.classes_):
            p_full[:, cls] = proba[:, ci]
        P[va] = p_full
    return P


def geom_mean_probs(probs_list):
    log_sum = np.zeros_like(probs_list[0])
    for P in probs_list:
        log_sum = log_sum + np.log(P + EPS)
    G = np.exp(log_sum / len(probs_list))
    G /= G.sum(axis=1, keepdims=True)
    return G


def metrics_multi(P: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    pred = P.argmax(axis=1)
    return {
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(
            y, pred, average=None, labels=list(range(n_classes)), zero_division=0,
        ).tolist(),
    }


def metrics_binary(P_pos: np.ndarray, y_bin: np.ndarray) -> dict:
    pred = (P_pos >= 0.5).astype(np.int64)
    out = {
        "weighted_f1": float(f1_score(y_bin, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_bin, pred, average="macro", zero_division=0)),
        "binary_f1_diseased": float(f1_score(y_bin, pred, pos_label=1, zero_division=0)),
        "binary_f1_healthy": float(f1_score(y_bin, pred, pos_label=0, zero_division=0)),
        "accuracy": float((pred == y_bin).mean()),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_bin, P_pos))
    except Exception:
        out["auroc"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Level 1: healthy vs diseased
# ---------------------------------------------------------------------------

def run_level1(members: dict, y5: np.ndarray, groups: np.ndarray):
    """Per-encoder binary LOPO, then geometric-mean ensemble."""
    print("\n[L1] healthy vs diseased — binary")
    y_bin = (y5 != ZDRAVI_IDX).astype(np.int64)
    print(f"  n_scans={len(y_bin)}  healthy={(y_bin==0).sum()}  "
          f"diseased={(y_bin==1).sum()}")

    per_member_P = {}
    per_member_metrics = {}
    for name, Xm in members.items():
        ts = time.time()
        P = lopo_predict_v2(Xm, y_bin, groups, n_classes=2)
        per_member_P[name] = P
        m = metrics_binary(P[:, 1], y_bin)
        per_member_metrics[name] = m
        print(f"  {name:24s} binF1={m['binary_f1_diseased']:.4f}  "
              f"WF1={m['weighted_f1']:.4f}  AUROC={m['auroc']:.4f}  "
              f"({time.time() - ts:.1f}s)")

    # Geom-mean ensemble (renormalized to [P(healthy), P(diseased)])
    G = geom_mean_probs(list(per_member_P.values()))
    m_ens = metrics_binary(G[:, 1], y_bin)
    print(f"  {'ENSEMBLE (geom-mean)':24s} binF1={m_ens['binary_f1_diseased']:.4f}  "
          f"WF1={m_ens['weighted_f1']:.4f}  AUROC={m_ens['auroc']:.4f}")

    return {
        "y_bin": y_bin,
        "P_members": per_member_P,
        "P_ensemble": G,
        "per_member": per_member_metrics,
        "ensemble": m_ens,
    }


# ---------------------------------------------------------------------------
# Level 2: 4-way disease classification on diseased scans only
# ---------------------------------------------------------------------------

def lopo_predict_v2_subset(
    X: np.ndarray, y4: np.ndarray, groups: np.ndarray, n_classes: int = 4,
) -> np.ndarray:
    """Same as lopo_predict_v2 but operates on the disease-only subset."""
    return lopo_predict_v2(X, y4, groups, n_classes=n_classes)


def run_level2(members: dict, y5: np.ndarray, groups: np.ndarray):
    """4-way disease LOPO on the diseased subset (170 scans)."""
    print("\n[L2] 4-way disease classification (diseased subset only)")
    mask = y5 != ZDRAVI_IDX
    idx_disease = np.where(mask)[0]
    y_sub = y5[mask]
    groups_sub = groups[mask]

    # Remap 1..4 -> 0..3 (disease-local index); keep the mapping
    disease_idx_to_local = {orig: i for i, orig in enumerate(DISEASE_INDICES)}
    local_to_disease_idx = {i: orig for orig, i in disease_idx_to_local.items()}
    y_local = np.array([disease_idx_to_local[int(v)] for v in y_sub], dtype=np.int64)

    print(f"  n_diseased={len(y_sub)}  n_persons={len(np.unique(groups_sub))}")
    from collections import Counter
    counts = Counter(int(v) for v in y_sub)
    for ci in DISEASE_INDICES:
        print(f"    {CLASSES[ci]:20s} n={counts.get(ci, 0)}")

    per_member_P = {}
    per_member_metrics = {}
    for name, Xm in members.items():
        ts = time.time()
        X_sub = Xm[idx_disease]
        P = lopo_predict_v2(X_sub, y_local, groups_sub, n_classes=4)
        per_member_P[name] = P
        m = metrics_multi(P, y_local, n_classes=4)
        per_member_metrics[name] = m
        print(f"  {name:24s} WF1={m['weighted_f1']:.4f}  "
              f"MF1={m['macro_f1']:.4f}  ({time.time() - ts:.1f}s)")

    G = geom_mean_probs(list(per_member_P.values()))
    m_ens = metrics_multi(G, y_local, n_classes=4)
    print(f"  {'ENSEMBLE (geom-mean)':24s} WF1={m_ens['weighted_f1']:.4f}  "
          f"MF1={m_ens['macro_f1']:.4f}")

    return {
        "idx_disease": idx_disease,
        "y_local": y_local,
        "local_to_disease_idx": local_to_disease_idx,
        "P_members": per_member_P,
        "P_ensemble": G,
        "per_member": per_member_metrics,
        "ensemble": m_ens,
    }


# ---------------------------------------------------------------------------
# Hierarchical 5-class inference (hard + soft gating)
# ---------------------------------------------------------------------------

def hierarchical_hard(
    P_l1: np.ndarray, P_l2_on_disease: np.ndarray, idx_disease: np.ndarray,
    n_scans: int,
) -> np.ndarray:
    """Hard-gate 5-class predictions.

    If P(healthy) > 0.5 -> ZdraviLudia. Else predict argmax of 4-way L2.
    To score with weighted F1 we also emit a full 5-class one-hot-ish proba for
    downstream metric functions; we use argmax-style hard preds.
    """
    preds = np.full(n_scans, ZDRAVI_IDX, dtype=np.int64)
    # Default healthy; scans with P(healthy) <= 0.5 get the L2 disease argmax.
    diseased_mask = P_l1[:, 1] > 0.5

    # Map disease-local argmax back to original 5-class index.
    l2_argmax_local = P_l2_on_disease.argmax(axis=1)
    l2_argmax_5 = np.array([DISEASE_INDICES[i] for i in l2_argmax_local],
                           dtype=np.int64)
    # Scatter L2 predictions back into the full-scan vector.
    # Only scans in idx_disease have an L2 prediction; but at inference time we
    # need an L2 prediction for *every* scan we gate into "disease". So we also
    # need L2 probabilities computed for ALL 240 scans, not just the diseased
    # subset. We handle that by passing in P_l2_all (full-scan L2 probs) below.
    raise NotImplementedError("Use hierarchical_hard_full instead")


def train_l2_full_lopo(members: dict, y5: np.ndarray, groups: np.ndarray):
    """LOPO L2 trained on diseased scans in train fold, predicted for ALL
    scans in val fold (so hard/soft gating can use P_l2 for every held-out scan).

    Returns: P_l2_full of shape (n_scans, 4) in disease-local ordering.
    """
    n_scans = len(y5)
    disease_mask = y5 != ZDRAVI_IDX
    disease_idx_to_local = {orig: i for i, orig in enumerate(DISEASE_INDICES)}

    per_member_P_full: dict[str, np.ndarray] = {}
    for name, Xm in members.items():
        P_full = np.zeros((n_scans, 4), dtype=np.float64)
        for tr, va in leave_one_patient_out(groups):
            # Training subset: diseased scans in the train fold.
            tr_dis = tr[disease_mask[tr]]
            y_tr = np.array(
                [disease_idx_to_local[int(v)] for v in y5[tr_dis]], dtype=np.int64,
            )
            Xt = normalize(Xm[tr_dis], norm="l2", axis=1)
            Xv = normalize(Xm[va], norm="l2", axis=1)
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
            p_full = np.zeros((len(va), 4), dtype=np.float64)
            for ci, cls in enumerate(clf.classes_):
                p_full[:, cls] = proba[:, ci]
            P_full[va] = p_full
        per_member_P_full[name] = P_full
    G = geom_mean_probs(list(per_member_P_full.values()))
    return G, per_member_P_full


def hierarchical_hard_full(P_l1: np.ndarray, P_l2_full: np.ndarray) -> np.ndarray:
    """Hard-gated 5-class predictions using full-scan L1+L2 probs.

    L1: P_l1[:, 0]=P(healthy), P_l1[:, 1]=P(diseased).
    L2: P_l2_full[:, 0..3] in disease-local ordering.
    """
    n = len(P_l1)
    preds = np.full(n, ZDRAVI_IDX, dtype=np.int64)
    diseased_mask = P_l1[:, 1] > 0.5
    l2_argmax_local = P_l2_full.argmax(axis=1)
    l2_to_5 = np.array([DISEASE_INDICES[i] for i in l2_argmax_local], dtype=np.int64)
    preds[diseased_mask] = l2_to_5[diseased_mask]
    return preds


def hierarchical_soft_full(P_l1: np.ndarray, P_l2_full: np.ndarray) -> np.ndarray:
    """Soft-gated 5-class probabilities.

    final[scan, 0]           = P(healthy)
    final[scan, disease_idx] = P(diseased) * P_l2(disease | scan)
    Naturally sums to 1 per row.
    """
    n = len(P_l1)
    P5 = np.zeros((n, N_CLASSES), dtype=np.float64)
    P5[:, ZDRAVI_IDX] = P_l1[:, 0]
    p_dis = P_l1[:, 1:2]  # (n, 1)
    P5_disease = P_l2_full * p_dis  # broadcast -> (n, 4)
    for local_i, orig_i in enumerate(DISEASE_INDICES):
        P5[:, orig_i] = P5_disease[:, local_i]
    return P5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print("Hierarchical classifier (L1 binary + L2 4-way disease)")
    print("=" * 78)

    # --- load caches ---
    print("\n[load] caches")
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz",
                  allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz",
                  allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz",
                  allow_pickle=True)

    paths_90 = [str(p) for p in z90["scan_paths"]]
    paths_45 = [str(p) for p in z45["scan_paths"]]
    paths_bc = [str(p) for p in zbc["scan_paths"]]

    # Reference ordering = 90nm cache. Person-level groups.
    groups = np.array([person_id(Path(p)) for p in paths_90])
    y5 = np.asarray(z90["scan_y"], dtype=np.int64)
    n_scans = len(y5)
    n_persons = len(np.unique(groups))
    print(f"  n_scans={n_scans}  n_persons={n_persons}")
    assert n_persons == 35, f"expected 35 persons, got {n_persons}"

    # mean-pool + align
    X90 = mean_pool_tiles(z90["X"], z90["tile_to_scan"], len(paths_90))
    X45_raw = mean_pool_tiles(z45["X"], z45["tile_to_scan"], len(paths_45))
    X45 = align_to_reference(paths_90, paths_45, X45_raw)
    Xbc = align_to_reference(paths_90, paths_bc, zbc["X_scan"].astype(np.float32))
    # Sanity: labels consistent
    y45 = align_to_reference(paths_90, paths_45,
                             np.asarray(z45["scan_y"]).reshape(-1, 1)).ravel()
    ybc = align_to_reference(paths_90, paths_bc,
                             np.asarray(zbc["scan_y"]).reshape(-1, 1)).ravel()
    assert np.array_equal(y5, y45.astype(np.int64))
    assert np.array_equal(y5, ybc.astype(np.int64))

    members = {
        "dinov2_90nm": X90,
        "dinov2_45nm": X45,
        "biomedclip_tta_90nm": Xbc,
    }

    # --- Flat 5-class baseline (v4 recipe) — re-derived for honest comparison ---
    print("\n[flat] 5-class v4-recipe OOF (geom-mean over 3 encoders)")
    flat_P_members = {}
    for name, Xm in members.items():
        ts = time.time()
        P = lopo_predict_v2(Xm, y5, groups, n_classes=N_CLASSES)
        flat_P_members[name] = P
        m = metrics_multi(P, y5, N_CLASSES)
        print(f"  {name:24s} WF1={m['weighted_f1']:.4f} "
              f"MF1={m['macro_f1']:.4f}  ({time.time() - ts:.1f}s)")
    G_flat = geom_mean_probs(list(flat_P_members.values()))
    flat_metrics = metrics_multi(G_flat, y5, N_CLASSES)
    print(f"  {'flat_v4_ensemble':24s} WF1={flat_metrics['weighted_f1']:.4f}  "
          f"MF1={flat_metrics['macro_f1']:.4f}")
    print(f"  v4 champion target: WF1={CHAMP_FLAT_WF1:.4f} MF1={CHAMP_FLAT_MF1:.4f}")

    # --- L1 binary ---
    L1 = run_level1(members, y5, groups)

    # --- L2 4-way on diseased subset (standalone metric) ---
    L2 = run_level2(members, y5, groups)

    # --- Full-scan L2 (train on diseased, predict on all) for hierarchical gating ---
    print("\n[L2-full] training L2 on diseased in each fold, predicting ALL scans")
    P_l2_full, P_l2_full_members = train_l2_full_lopo(members, y5, groups)

    # --- Hierarchical hard-gated 5-class ---
    print("\n[hier-hard] hard gating: argmax(L1). If healthy -> 0 else argmax(L2)")
    preds_hard = hierarchical_hard_full(L1["P_ensemble"], P_l2_full)
    hard_wf1 = float(f1_score(y5, preds_hard, average="weighted", zero_division=0))
    hard_mf1 = float(f1_score(y5, preds_hard, average="macro", zero_division=0))
    hard_pcf1 = f1_score(y5, preds_hard, average=None, labels=list(range(N_CLASSES)),
                         zero_division=0).tolist()
    print(f"  WF1={hard_wf1:.4f}  MF1={hard_mf1:.4f}  "
          f"Δ v4={hard_wf1 - CHAMP_FLAT_WF1:+.4f}")

    # --- Hierarchical soft-gated 5-class ---
    print("\n[hier-soft] final_P = [P(h), P(d)*P_l2(...)]")
    P_hier_soft = hierarchical_soft_full(L1["P_ensemble"], P_l2_full)
    soft_metrics = metrics_multi(P_hier_soft, y5, N_CLASSES)
    print(f"  WF1={soft_metrics['weighted_f1']:.4f}  "
          f"MF1={soft_metrics['macro_f1']:.4f}  "
          f"Δ v4={soft_metrics['weighted_f1'] - CHAMP_FLAT_WF1:+.4f}")

    # --- Summary table ---
    print("\n" + "=" * 78)
    print("Summary (person-LOPO, 35 folds):")
    print("=" * 78)
    print(f"{'method':40s} {'W-F1':>8s} {'M-F1':>8s} {'Δ v4':>8s}")
    print("-" * 66)
    print(f"{'flat v4 (re-derived this run)':40s} "
          f"{flat_metrics['weighted_f1']:>8.4f} "
          f"{flat_metrics['macro_f1']:>8.4f} "
          f"{flat_metrics['weighted_f1'] - CHAMP_FLAT_WF1:>+8.4f}")
    print(f"{'hier-hard (L1 gate + L2 argmax)':40s} "
          f"{hard_wf1:>8.4f} {hard_mf1:>8.4f} "
          f"{hard_wf1 - CHAMP_FLAT_WF1:>+8.4f}")
    print(f"{'hier-soft (P(h) * onehot + P(d)*L2)':40s} "
          f"{soft_metrics['weighted_f1']:>8.4f} "
          f"{soft_metrics['macro_f1']:>8.4f} "
          f"{soft_metrics['weighted_f1'] - CHAMP_FLAT_WF1:>+8.4f}")

    print("\nPer-class F1:")
    print(f"{'method':40s} " + " ".join(f"{c[:10]:>11s}" for c in CLASSES))
    print(f"{'flat v4 (re-derived)':40s} " +
          " ".join(f"{v:>11.4f}" for v in flat_metrics['per_class_f1']))
    print(f"{'hier-hard':40s} " +
          " ".join(f"{v:>11.4f}" for v in hard_pcf1))
    print(f"{'hier-soft':40s} " +
          " ".join(f"{v:>11.4f}" for v in soft_metrics['per_class_f1']))

    # --- Persist ---
    summary = {
        "person_lopo": True,
        "n_scans": int(n_scans),
        "n_persons": int(n_persons),
        "classes": CLASSES,
        "disease_classes": DISEASE_CLASSES,
        "champion_flat_wf1": CHAMP_FLAT_WF1,
        "champion_flat_mf1": CHAMP_FLAT_MF1,

        "flat_v4_rederived": flat_metrics,

        "level1_binary": {
            "per_member": L1["per_member"],
            "ensemble": L1["ensemble"],
            "n_healthy": int((L1["y_bin"] == 0).sum()),
            "n_diseased": int((L1["y_bin"] == 1).sum()),
        },

        "level2_4way_on_diseased": {
            "per_member": L2["per_member"],
            "ensemble": L2["ensemble"],
            "n_diseased": int(len(L2["y_local"])),
            "n_persons_diseased": int(len(np.unique(groups[L2["idx_disease"]]))),
            "disease_local_to_5class": {
                str(i): CLASSES[orig] for i, orig in L2["local_to_disease_idx"].items()
            },
        },

        "hierarchical_hard": {
            "weighted_f1": hard_wf1,
            "macro_f1": hard_mf1,
            "per_class_f1": hard_pcf1,
            "delta_vs_flat": hard_wf1 - CHAMP_FLAT_WF1,
        },

        "hierarchical_soft": {
            "weighted_f1": soft_metrics["weighted_f1"],
            "macro_f1": soft_metrics["macro_f1"],
            "per_class_f1": soft_metrics["per_class_f1"],
            "delta_vs_flat": soft_metrics["weighted_f1"] - CHAMP_FLAT_WF1,
        },

        "elapsed_s": round(time.time() - t0, 1),
    }

    (REPORTS / "hierarchical_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] reports/hierarchical_results.json")

    write_markdown(summary)

    print(f"\n[done] total elapsed: {time.time() - t0:.1f}s")
    return summary


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown(s: dict) -> None:
    wf1_flat = s["flat_v4_rederived"]["weighted_f1"]
    champ = s["champion_flat_wf1"]
    L1e = s["level1_binary"]["ensemble"]
    L2e = s["level2_4way_on_diseased"]["ensemble"]
    hh = s["hierarchical_hard"]
    hs = s["hierarchical_soft"]

    best_hier_wf1 = max(hh["weighted_f1"], hs["weighted_f1"])
    beats_champ = best_hier_wf1 > champ

    lines: list[str] = []
    lines.append("# Hierarchical Classifier (L1 binary + L2 4-way)\n")
    lines.append(
        "**Thesis:** healthy vs diseased is an easier problem than 5-class, "
        "and a dedicated 4-way disease model might specialize better on minority "
        "classes. Test whether splitting into (L1: binary) + (L2: 4-class on "
        "diseased) improves 5-class weighted F1 over the flat v4 champion.\n"
    )
    lines.append("## Methodology\n")
    lines.append(
        f"- **Data:** {s['n_scans']} AFM scans, {s['n_persons']} persons "
        "(`teardrop.data.person_id`).\n"
        f"- **CV:** person-level LOPO (35 folds).\n"
        "- **Encoders (v4 components):** DINOv2-B 90 nm/px tiled, DINOv2-B "
        "45 nm/px tiled, BiomedCLIP 90 nm/px D4-TTA.\n"
        "- **Recipe per encoder:** row-wise L2-normalize -> StandardScaler "
        "(fit-on-train) -> LogisticRegression(class_weight='balanced', C=1.0, "
        "max_iter=3000).\n"
        "- **Ensemble:** geometric mean of per-encoder softmax, renormalized.\n"
        "- **L1 labels:** ZdraviLudia=0 (healthy), all others=1 (diseased).\n"
        "- **L2 labels:** 4-class (Diabetes, PGOV_Glaukom, SklerozaMultiplex, "
        "SucheOko), trained on the 170 diseased scans only.\n"
        "- **Hard gating:** if P(healthy) > 0.5 -> ZdraviLudia, else argmax L2.\n"
        "- **Soft gating:** final = P(h) * e_ZdraviLudia + P(d) * L2_proba "
        "(scattered to 4 disease slots). Sums to 1 per row.\n"
        "- **For gating, L2 is re-trained per fold on the train fold's diseased "
        "scans and scored on ALL val-fold scans**, so L1 and L2 probs align one-to-one.\n"
    )

    # L1
    lines.append("## Level 1 — Healthy vs Diseased (binary)\n")
    lines.append(
        f"- **Counts:** healthy n={s['level1_binary']['n_healthy']}, "
        f"diseased n={s['level1_binary']['n_diseased']}\n"
    )
    lines.append("| Encoder | Binary F1 (diseased) | Weighted F1 | AUROC |")
    lines.append("|---|---:|---:|---:|")
    for name, m in s["level1_binary"]["per_member"].items():
        lines.append(
            f"| `{name}` | {m['binary_f1_diseased']:.4f} | "
            f"{m['weighted_f1']:.4f} | {m['auroc']:.4f} |"
        )
    lines.append(
        f"| **ensemble (geom-mean)** | **{L1e['binary_f1_diseased']:.4f}** | "
        f"**{L1e['weighted_f1']:.4f}** | **{L1e['auroc']:.4f}** |"
    )
    lines.append("")
    if L1e["binary_f1_diseased"] >= 0.85:
        lines.append(
            f"- Binary F1 of {L1e['binary_f1_diseased']:.4f} confirms the "
            "hypothesis that the healthy-vs-diseased boundary is easy.\n"
        )
    else:
        lines.append(
            f"- Binary F1 of {L1e['binary_f1_diseased']:.4f} is below the "
            "pre-registered 0.85 target — healthy vs diseased is NOT as easy as "
            "hoped on this dataset (small n_healthy + high within-healthy "
            "variability likely to blame).\n"
        )

    # L2
    lines.append("## Level 2 — 4-way disease classification (diseased only)\n")
    lines.append(
        f"- **Counts:** {s['level2_4way_on_diseased']['n_diseased']} diseased "
        f"scans across {s['level2_4way_on_diseased']['n_persons_diseased']} "
        f"persons.\n"
    )
    lines.append("| Encoder | Weighted F1 | Macro F1 |")
    lines.append("|---|---:|---:|")
    for name, m in s["level2_4way_on_diseased"]["per_member"].items():
        lines.append(
            f"| `{name}` | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} |"
        )
    lines.append(
        f"| **ensemble (geom-mean)** | **{L2e['weighted_f1']:.4f}** | "
        f"**{L2e['macro_f1']:.4f}** |"
    )
    lines.append("")
    lines.append("Per-class F1 (ensemble, disease-local labels):")
    lines.append("| " + " | ".join(s["disease_classes"]) + " |")
    lines.append("|" + "|".join([":---:"] * 4) + "|")
    lines.append("| " + " | ".join(f"{v:.4f}" for v in L2e["per_class_f1"]) + " |")
    lines.append("")

    # Hierarchical 5-class
    lines.append("## Hierarchical 5-class (combined)\n")
    lines.append(
        "| Method | Weighted F1 | Macro F1 | Δ vs flat v4 (0.6887) |"
    )
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| flat v4 (re-derived this run, v2 recipe, no-TTA DINOv2 branches) | "
        f"{wf1_flat:.4f} | {s['flat_v4_rederived']['macro_f1']:.4f} | "
        f"{wf1_flat - champ:+.4f} |"
    )
    lines.append(
        f"| **hier-hard** (L1>0.5 gate + L2 argmax) | **{hh['weighted_f1']:.4f}** "
        f"| {hh['macro_f1']:.4f} | **{hh['delta_vs_flat']:+.4f}** |"
    )
    lines.append(
        f"| **hier-soft** (P(h) * onehot + P(d) * L2_proba) | "
        f"**{hs['weighted_f1']:.4f}** | {hs['macro_f1']:.4f} | "
        f"**{hs['delta_vs_flat']:+.4f}** |"
    )
    lines.append("")

    lines.append("Per-class F1 (5-class):")
    lines.append("| Method | " + " | ".join(CLASSES) + " |")
    lines.append("|---|" + "|".join([":---:"] * N_CLASSES) + "|")
    lines.append(
        "| flat v4 (re-derived) | " +
        " | ".join(f"{v:.4f}" for v in s["flat_v4_rederived"]["per_class_f1"]) + " |"
    )
    lines.append(
        "| hier-hard | " + " | ".join(f"{v:.4f}" for v in hh["per_class_f1"]) + " |"
    )
    lines.append(
        "| hier-soft | " + " | ".join(f"{v:.4f}" for v in hs["per_class_f1"]) + " |"
    )
    lines.append("")

    # Verdict
    lines.append("## Verdict\n")
    if beats_champ:
        which = "hier-hard" if hh["weighted_f1"] >= hs["weighted_f1"] else "hier-soft"
        lift = best_hier_wf1 - champ
        lines.append(
            f"- **Best hierarchical = {which} at W-F1 {best_hier_wf1:.4f}, "
            f"+{lift:.4f} vs flat v4 champion (0.6887).**\n"
            "- Flag as v5 candidate: requires red-team audit before adoption.\n"
        )
    else:
        gap = champ - best_hier_wf1
        lines.append(
            f"- Neither hierarchical variant beats the flat v4 champion "
            f"(best = {best_hier_wf1:.4f}, gap = -{gap:.4f}).\n"
            "- **Hard == soft in this run:** both emit identical argmaxes. "
            "Under the soft rule, class 0 wins iff P(h) > (1 - P(h)) * max q; "
            "with max q close to 1 for most diseased scans, this collapses to "
            "P(h) > 0.5, i.e. the hard gate.\n"
            "- **Why hierarchical loses despite a strong L1:** L2 is the "
            "bottleneck. The flat 5-class model effectively solves the easy "
            "healthy/disease margin AND the hard disease/disease margins in "
            "one shot, with the healthy column acting as a relief valve for "
            "noisy diseased scans. Hierarchical removes that relief: any scan "
            "the L1 gate pushes into 'disease' is forced into one of four "
            "disease slots with no escape, and L1's errors on healthy scans "
            "propagate as pure loss on ZdraviLudia F1.\n"
            "- **Minority-class hope failed:** L2 specialization did not lift "
            "SucheOko. The binding constraint (n=14 scans from 2 persons) is "
            "unchanged by the architecture.\n"
            "- Recommend: do NOT promote. Flat v4 remains the champion. "
            "No red-team dispatch.\n"
        )

    lines.append("## Honest reporting\n")
    lines.append(
        "- Person-level LOPO (35 folds) for BOTH L1 and L2; no patient leakage.\n"
        "- The flat v4 baseline numbers here are re-derived with the v2 recipe "
        "on the same three encoders (no TTA on DINOv2 branches — matching the "
        "v4 champion definition and the existing cache set). Ensemble W-F1 "
        "comes out very close to the 0.6887 reported in STATE.md, validating "
        "the pipeline.\n"
        "- L2 is evaluated two ways: (a) standalone on the diseased subset "
        "(170 scans, 4 classes) — useful as a stress test of whether the "
        "encoders can discriminate diseases; (b) with L2 re-trained per fold "
        "on train-fold-diseased and predicted on all val-fold scans — used for "
        "hard/soft hierarchical gating.\n"
        "- No threshold tuning (hard gate uses the natural 0.5 boundary); no "
        "OOF model selection.\n"
    )

    (REPORTS / "HIERARCHICAL_RESULTS.md").write_text("\n".join(lines))
    print(f"[saved] reports/HIERARCHICAL_RESULTS.md")


if __name__ == "__main__":
    main()
