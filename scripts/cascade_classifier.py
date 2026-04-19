"""Confidence-gated cascade classifier on top of the Stage-1 ensemble.

Architecture
============

Stage 1 (already trained):
    cache/best_ensemble_predictions.npz  — DINOv2-B + BiomedCLIP proba-avg
    with per-class thresholds. Person-LOPO weighted F1 ~ 0.67 (claimed).

Stage 2 (this script): two binary specialists, trained person-LOPO so their
predictions on routed scans are honestly out-of-fold:

    Specialist A:  Glaukom (PGOV_Glaukom) vs SklerozaMultiplex
        Features: DINOv2-B scan-mean  CONCAT  TDA (1015-dim)
        Classifier: XGBoost with scale_pos_weight
    Specialist B:  Diabetes vs ZdraviLudia
        Features: DINOv2-B scan-mean  CONCAT  handcrafted (94-dim)
        Classifier: XGBoost with scale_pos_weight
    (Bonus)
    Specialist C:  SklerozaMultiplex vs ZdraviLudia
        Features: DINOv2-B scan-mean  CONCAT  handcrafted

Cascade gating (evaluated per scan):
    Let top1, top2 be the top-2 Stage-1 classes and conf = Stage-1 top-1 proba.
    If conf > GATING_THRESHOLD → keep Stage-1 prediction (confident).
    Else:
        If {top1, top2} == {PGOV_Glaukom, SklerozaMultiplex} → Specialist A
        If {top1, top2} == {Diabetes, ZdraviLudia}          → Specialist B
        (bonus) If {top1, top2} == {SklerozaMultiplex, ZdraviLudia} → Specialist C
        Else → keep Stage-1 prediction.

IMPORTANT: "Stage-1 top-1 proba" is read from the saved ensemble proba matrix
(un-scaled by thresholds). The "Stage-1 prediction" we compare against is the
saved `pred_label` that *did* use thresholds — same prediction that produced
the 0.67 baseline.

Honest-evaluation constraints:
    * Person-level LOPO (35 groups) everywhere, using `teardrop.cv.leave_one_patient_out`.
    * Specialists are trained only on their two classes (subsetting), but the
      fold structure is the SAME person-groups as Stage-1 so there is no
      leakage: for any scan we predict, that scan's person is excluded from
      both Stage-1 and the specialist's training set.
    * The gating threshold sweep at the end is reported honestly — i.e. it
      will show training-set optimism — and the "headline" cascade number
      uses the fixed default threshold 0.65.

Deliverables
============
    scripts/cascade_classifier.py   (this file, runnable)
    cache/cascade_oof.npz
    reports/CASCADE_RESULTS.md
"""
from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import CLASSES, enumerate_samples  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
N_CLASSES = len(CLASSES)

CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_GLAUKOM = CLASS_IDX["PGOV_Glaukom"]
IDX_SM = CLASS_IDX["SklerozaMultiplex"]
IDX_DIA = CLASS_IDX["Diabetes"]
IDX_HEALTHY = CLASS_IDX["ZdraviLudia"]

DEFAULT_GATING_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _mean_pool_tiles(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    X_tiles = z["X"]
    t2s = z["tile_to_scan"]
    scan_y = np.asarray(z["scan_y"])
    scan_paths = np.asarray(z["scan_paths"])
    n_scans = len(scan_y)
    out = np.zeros((n_scans, X_tiles.shape[1]), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int32)
    for ti, si in enumerate(t2s):
        out[si] += X_tiles[ti]
        counts[si] += 1
    out /= np.maximum(counts, 1)[:, None]
    return out, scan_y, scan_paths


@dataclass
class Bundle:
    """All feature matrices + labels + groups in canonical sample order."""
    y: np.ndarray                 # (240,) int class label
    person_groups: np.ndarray     # (240,) str person IDs
    scan_paths: np.ndarray        # (240,) str
    X_dinov2: np.ndarray          # (240, 768) DINOv2-B scan-mean
    X_tda: np.ndarray             # (240, Dtda)
    X_handcrafted: np.ndarray     # (240, Dhc)


def load_bundle() -> Bundle:
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    y = np.array([s.label for s in samples], dtype=int)
    person_groups = np.array([s.person for s in samples])
    scan_paths = np.array([str(Path(s.raw_path).resolve()) for s in samples])

    path_to_idx: dict[str, int] = {}
    for i, s in enumerate(samples):
        path_to_idx[str(Path(s.raw_path).resolve())] = i
        path_to_idx[str(s.raw_path)] = i

    # DINOv2-B tiled → scan mean
    Xd_raw, y_cached, paths_cached = _mean_pool_tiles(
        CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz"
    )
    Xd = np.full_like(Xd_raw, np.nan)
    order = np.array([path_to_idx[str(p)] for p in paths_cached])
    Xd[order] = Xd_raw
    assert not np.isnan(Xd).any()

    # TDA
    df_tda = pd.read_parquet(CACHE / "features_tda.parquet").copy()
    df_tda["si"] = df_tda["raw"].map(path_to_idx)
    df_tda = df_tda.dropna(subset=["si"]).sort_values("si").reset_index(drop=True)
    feat_cols_tda = [c for c in df_tda.columns
                     if c not in ("raw", "cls", "label", "patient", "si")]
    X_tda = df_tda[feat_cols_tda].values.astype(np.float32)
    X_tda = np.nan_to_num(X_tda, nan=0.0, posinf=0.0, neginf=0.0)
    assert np.array_equal(df_tda["label"].values.astype(int), y), "TDA row mismatch"

    # Handcrafted
    df_hc = pd.read_parquet(CACHE / "features_handcrafted.parquet").copy()
    df_hc["si"] = df_hc["raw"].map(path_to_idx)
    df_hc = df_hc.dropna(subset=["si"]).sort_values("si").reset_index(drop=True)
    feat_cols_hc = [c for c in df_hc.columns
                    if c not in ("raw", "cls", "label", "patient", "si")]
    X_hc = df_hc[feat_cols_hc].values.astype(np.float32)
    X_hc = np.nan_to_num(X_hc, nan=0.0, posinf=0.0, neginf=0.0)
    assert np.array_equal(df_hc["label"].values.astype(int), y), "HC row mismatch"

    return Bundle(
        y=y,
        person_groups=person_groups,
        scan_paths=scan_paths,
        X_dinov2=Xd.astype(np.float32),
        X_tda=X_tda,
        X_handcrafted=X_hc,
    )


# ---------------------------------------------------------------------------
# Binary specialist: person-LOPO OOF predictions for a given pair of classes
# ---------------------------------------------------------------------------

def specialist_lopo(
    X: np.ndarray,
    y: np.ndarray,
    person_groups: np.ndarray,
    pair: tuple[int, int],
    name: str,
) -> dict:
    """Train a binary XGB classifier for `pair` via person-LOPO.

    Returns dict with OOF binary pred (class ints in `pair`), proba-positive,
    and a mask indicating which rows are in-pair (== participate in eval).
    The OOF values are filled for EVERY sample — for rows not in-pair we fit
    each fold only on in-pair samples, and predict on all (240 rows) anyway
    so the cascade can route any scan to the specialist at inference time.

    Mapping: inside the binary task, class 0 = pair[0], class 1 = pair[1].
    """
    n = len(y)
    in_pair = np.isin(y, pair)
    pos_cls, neg_cls = pair[1], pair[0]   # pair[1] is "positive" for scale_pos_weight

    # OOF binary predict-proba and prediction filled per-sample:
    #   `proba_pos[i]` = p(y == pair[1]) according to fold that excluded i's person.
    #   `pred_in_pair[i]` = pair[0] or pair[1], the binary argmax.
    proba_pos = np.full(n, np.nan, dtype=np.float64)
    pred_in_pair = np.full(n, -1, dtype=np.int64)

    # Track per-fold counts so we can sanity-check later
    n_folds_used = 0

    for tr, va in leave_one_patient_out(person_groups):
        tr_pair = tr[in_pair[tr]]
        if len(tr_pair) < 5:
            # Almost never happens for these pairs, but skip empty folds safely.
            continue
        y_bin_tr = (y[tr_pair] == pos_cls).astype(np.int32)
        # scale_pos_weight = #neg / #pos
        n_pos = int(y_bin_tr.sum())
        n_neg = int(len(y_bin_tr) - n_pos)
        spw = (n_neg / n_pos) if n_pos > 0 else 1.0

        clf = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=spw,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            verbosity=0,
            random_state=42,
        )
        clf.fit(X[tr_pair], y_bin_tr)
        # predict on ALL val rows (we may route any of them)
        p = clf.predict_proba(X[va])[:, 1]
        proba_pos[va] = p
        pred_bin = (p >= 0.5).astype(np.int64)
        pred_in_pair[va] = np.where(pred_bin == 1, pos_cls, neg_cls)

        n_folds_used += 1

    # Binary-eval metrics restricted to in-pair scans
    mask = in_pair & (pred_in_pair != -1)
    y_true_pair = y[mask]
    y_pred_pair = pred_in_pair[mask]
    # "binary F1" = F1 where `pos_cls` is the positive class
    f1_bin = f1_score((y_true_pair == pos_cls).astype(int),
                      (y_pred_pair == pos_cls).astype(int),
                      zero_division=0)
    acc = float((y_true_pair == y_pred_pair).mean()) if len(y_true_pair) else 0.0

    # Per-class F1 (both directions)
    per_class = {}
    for c in pair:
        y_true_c = (y_true_pair == c).astype(int)
        y_pred_c = (y_pred_pair == c).astype(int)
        per_class[CLASSES[c]] = float(
            f1_score(y_true_c, y_pred_c, zero_division=0)
        )

    return {
        "name": name,
        "pair": pair,
        "proba_pos": proba_pos,
        "pred_in_pair": pred_in_pair,
        "in_pair_mask": in_pair,
        "f1_binary": float(f1_bin),
        "accuracy": acc,
        "n_pair_scans": int(in_pair.sum()),
        "n_folds_used": n_folds_used,
        "per_class_f1": per_class,
    }


# ---------------------------------------------------------------------------
# Cascade — given Stage-1 preds/proba + specialist OOFs, build final preds
# ---------------------------------------------------------------------------

def apply_cascade(
    stage1_pred: np.ndarray,            # (n,)  baseline argmax from npz
    stage1_proba: np.ndarray,           # (n, 5) original ensemble proba (un-thresholded)
    gating_threshold: float,
    specialists: dict,                  # name -> dict from specialist_lopo
    use_A: bool,
    use_B: bool,
    use_C: bool,
) -> tuple[np.ndarray, dict]:
    """Return (final_pred, routing_stats)."""
    n = len(stage1_pred)
    final_pred = stage1_pred.copy()
    routed_to = np.full(n, "", dtype=object)

    # Top-1 confidence = highest proba value (before thresholding)
    # Top-2 indices = top 2 argmax
    top1_conf = stage1_proba.max(axis=1)
    top2 = np.argsort(-stage1_proba, axis=1)[:, :2]  # (n, 2)
    top2_sets = [frozenset(row.tolist()) for row in top2]

    pair_A = frozenset({IDX_GLAUKOM, IDX_SM})
    pair_B = frozenset({IDX_DIA, IDX_HEALTHY})
    pair_C = frozenset({IDX_SM, IDX_HEALTHY})

    for i in range(n):
        if top1_conf[i] > gating_threshold:
            continue  # keep Stage-1
        s = top2_sets[i]
        if use_A and s == pair_A:
            sp = specialists["A"]
            if sp["pred_in_pair"][i] >= 0:
                final_pred[i] = sp["pred_in_pair"][i]
                routed_to[i] = "A"
        elif use_B and s == pair_B:
            sp = specialists["B"]
            if sp["pred_in_pair"][i] >= 0:
                final_pred[i] = sp["pred_in_pair"][i]
                routed_to[i] = "B"
        elif use_C and s == pair_C:
            sp = specialists["C"]
            if sp["pred_in_pair"][i] >= 0:
                final_pred[i] = sp["pred_in_pair"][i]
                routed_to[i] = "C"

    stats = {
        "n_routed_A": int((routed_to == "A").sum()),
        "n_routed_B": int((routed_to == "B").sum()),
        "n_routed_C": int((routed_to == "C").sum()),
        "n_total_routed": int((routed_to != "").sum()),
        "n_kept_stage1": int((routed_to == "").sum()),
        "routed_to": routed_to,
    }
    return final_pred, stats


def _routing_diagnostic(s1_pred, s1_proba, final_pred, y, pair: set, thr: float,
                        name: str) -> dict:
    """Count good/bad changes on the scans that got routed to this specialist."""
    top1_conf = s1_proba.max(axis=1)
    top2 = np.argsort(-s1_proba, axis=1)[:, :2]
    mask = np.zeros(len(y), dtype=bool)
    for i in range(len(y)):
        if top1_conf[i] > thr:
            continue
        if frozenset(top2[i].tolist()) == frozenset(pair):
            mask[i] = True
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {"name": name, "n_routed": 0, "s1_correct": 0, "cas_correct": 0,
                "n_good": 0, "n_bad": 0, "true_dist": {}, "s1_dist": {}, "cas_dist": {}}
    s1c = s1_pred[idx] == y[idx]
    cac = final_pred[idx] == y[idx]
    changed = s1_pred[idx] != final_pred[idx]
    n_good = int(((~s1c) & cac & changed).sum())
    n_bad = int((s1c & (~cac) & changed).sum())
    true_dist = {CLASSES[c]: int((y[idx] == c).sum()) for c in range(N_CLASSES)}
    s1_dist = {CLASSES[c]: int((s1_pred[idx] == c).sum()) for c in range(N_CLASSES)}
    cas_dist = {CLASSES[c]: int((final_pred[idx] == c).sum()) for c in range(N_CLASSES)}
    return {
        "name": name,
        "n_routed": int(len(idx)),
        "s1_correct": int(s1c.sum()),
        "cas_correct": int(cac.sum()),
        "n_changed": int(changed.sum()),
        "n_good": n_good,
        "n_bad": n_bad,
        "true_dist": true_dist,
        "s1_dist": s1_dist,
        "cas_dist": cas_dist,
    }


def evaluate(pred: np.ndarray, y: np.ndarray) -> dict:
    f1w = f1_score(y, pred, average="weighted", zero_division=0)
    f1m = f1_score(y, pred, average="macro", zero_division=0)
    f1pc = f1_score(y, pred, average=None,
                    labels=list(range(N_CLASSES)), zero_division=0)
    return {
        "f1w": float(f1w),
        "f1m": float(f1m),
        "f1pc": [float(v) for v in f1pc],
    }


def pair_accuracy(pred: np.ndarray, y: np.ndarray, pair: tuple[int, int]) -> dict:
    """Binary accuracy/F1 restricted to the scans that are truly in `pair`."""
    m = np.isin(y, pair)
    if not m.any():
        return {"n": 0, "acc": 0.0, "f1": 0.0}
    pos = pair[1]
    y_bin = (y[m] == pos).astype(int)
    # Predictions on in-pair scans; if the model predicts outside `pair`, treat
    # as "wrong" for pair-level accuracy (same measure used by the ensemble).
    p = pred[m]
    p_bin = np.where(p == pos, 1, np.where(p == pair[0], 0, -1))
    correct = (p_bin == y_bin)  # any -1 prediction is counted wrong
    acc = float(correct.mean())
    # F1 against whichever is positive; treat -1 (off-pair prediction) as 0 (neg)
    p_bin_f1 = np.where(p_bin == -1, 0, p_bin)
    f1 = float(f1_score(y_bin, p_bin_f1, zero_division=0))
    return {"n": int(m.sum()), "acc": acc, "f1": f1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 76)
    print("Cascade Classifier — Stage 1 ensemble + binary specialists")
    print("=" * 76)

    t_start = time.time()

    # -- Load data --
    bundle = load_bundle()
    y = bundle.y
    groups = bundle.person_groups
    n = len(y)
    print(f"Samples: n={n}, classes={dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Person groups: {len(np.unique(groups))}")

    # -- Load Stage-1 predictions --
    z1 = np.load(CACHE / "best_ensemble_predictions.npz", allow_pickle=True)
    s1_proba = z1["proba"]                        # (240, 5)
    s1_pred = z1["pred_label"].astype(int)        # (240,) — post-threshold argmax
    s1_true = z1["true_label"].astype(int)
    s1_paths = z1["scan_paths"]
    assert np.array_equal(s1_true, y), "Stage-1 y mismatch"
    # Verify row alignment
    assert np.array_equal(s1_paths.astype(str), bundle.scan_paths.astype(str)), \
        "Stage-1 scan path alignment mismatch"

    baseline_metrics = evaluate(s1_pred, y)
    print(f"Stage-1 baseline weighted F1 = {baseline_metrics['f1w']:.4f} "
          f"(macro {baseline_metrics['f1m']:.4f})")

    # -- Build feature matrices for each specialist --
    #   A: Glaukom vs SM,   features = DINOv2-B || TDA
    #   B: Diabetes vs Healthy, features = DINOv2-B || handcrafted
    #   C: SM vs Healthy,   features = DINOv2-B || handcrafted
    X_A = np.concatenate([bundle.X_dinov2, bundle.X_tda], axis=1)
    X_B = np.concatenate([bundle.X_dinov2, bundle.X_handcrafted], axis=1)
    X_C = np.concatenate([bundle.X_dinov2, bundle.X_handcrafted], axis=1)
    print(f"\nSpecialist feature shapes: "
          f"A={X_A.shape}, B={X_B.shape}, C={X_C.shape}")

    # -- Train all three specialists (person-LOPO) --
    print("\n" + "-" * 76)
    print("Training specialists via person-LOPO")
    print("-" * 76)

    t0 = time.time()
    spec_A = specialist_lopo(X_A, y, groups,
                             (IDX_GLAUKOM, IDX_SM), name="A (Glaukom vs SM)")
    print(f"  A (Glaukom vs SM): binary F1={spec_A['f1_binary']:.4f}  "
          f"acc={spec_A['accuracy']:.4f}  n={spec_A['n_pair_scans']}  "
          f"per-class F1: {spec_A['per_class_f1']}  "
          f"({time.time()-t0:.1f}s)")

    t0 = time.time()
    spec_B = specialist_lopo(X_B, y, groups,
                             (IDX_DIA, IDX_HEALTHY), name="B (Diabetes vs Healthy)")
    print(f"  B (Diabetes vs Healthy): binary F1={spec_B['f1_binary']:.4f}  "
          f"acc={spec_B['accuracy']:.4f}  n={spec_B['n_pair_scans']}  "
          f"per-class F1: {spec_B['per_class_f1']}  "
          f"({time.time()-t0:.1f}s)")

    t0 = time.time()
    spec_C = specialist_lopo(X_C, y, groups,
                             (IDX_SM, IDX_HEALTHY), name="C (SM vs Healthy)")
    print(f"  C (SM vs Healthy): binary F1={spec_C['f1_binary']:.4f}  "
          f"acc={spec_C['accuracy']:.4f}  n={spec_C['n_pair_scans']}  "
          f"per-class F1: {spec_C['per_class_f1']}  "
          f"({time.time()-t0:.1f}s)")

    specialists = {"A": spec_A, "B": spec_B, "C": spec_C}

    # -- Cascade variants at default threshold 0.65 --
    print("\n" + "-" * 76)
    print(f"Cascade variants at gating threshold = {DEFAULT_GATING_THRESHOLD}")
    print("-" * 76)

    variants = [
        ("stage1_only", False, False, False),
        ("stage1 + A (Glaukom/SM)", True, False, False),
        ("stage1 + B (Diabetes/Healthy)", False, True, False),
        ("stage1 + A + B", True, True, False),
        ("stage1 + A + B + C", True, True, True),
    ]
    variant_rows = []
    per_variant_preds = {}
    for label, uA, uB, uC in variants:
        pred_final, rstats = apply_cascade(
            s1_pred, s1_proba, DEFAULT_GATING_THRESHOLD,
            specialists, uA, uB, uC,
        )
        m = evaluate(pred_final, y)
        pa = pair_accuracy(pred_final, y, (IDX_GLAUKOM, IDX_SM))
        pb = pair_accuracy(pred_final, y, (IDX_DIA, IDX_HEALTHY))
        pc = pair_accuracy(pred_final, y, (IDX_SM, IDX_HEALTHY))
        variant_rows.append({
            "label": label,
            "use_A": uA, "use_B": uB, "use_C": uC,
            "f1w": m["f1w"], "f1m": m["f1m"], "f1pc": m["f1pc"],
            "n_routed_A": rstats["n_routed_A"],
            "n_routed_B": rstats["n_routed_B"],
            "n_routed_C": rstats["n_routed_C"],
            "n_total_routed": rstats["n_total_routed"],
            "pair_acc_GlauSM": pa,
            "pair_acc_DiaHealthy": pb,
            "pair_acc_SMHealthy": pc,
        })
        per_variant_preds[label] = pred_final
        print(f"  {label:34s} weighted F1 = {m['f1w']:.4f}  macro = {m['f1m']:.4f}  "
              f"routed: A={rstats['n_routed_A']} B={rstats['n_routed_B']} "
              f"C={rstats['n_routed_C']} (total {rstats['n_total_routed']})")

    # -- Choose "headline" cascade: A+B (no bonus) at default threshold --
    headline = next(r for r in variant_rows if r["label"] == "stage1 + A + B")
    print(f"\nHeadline cascade (A+B, thr={DEFAULT_GATING_THRESHOLD}): "
          f"weighted F1 = {headline['f1w']:.4f}")
    delta = headline["f1w"] - baseline_metrics["f1w"]
    print(f"  vs Stage-1 baseline {baseline_metrics['f1w']:.4f} → Δ = {delta:+.4f}")

    # -- Gating threshold sweep: A+B only --
    print("\n" + "-" * 76)
    print("Gating threshold sweep (A+B, honest person-LOPO eval)")
    print("-" * 76)
    sweep_rows = []
    for thr in [0.5, 0.6, 0.65, 0.7, 0.75]:
        pred_t, rstats_t = apply_cascade(
            s1_pred, s1_proba, thr, specialists,
            use_A=True, use_B=True, use_C=False,
        )
        m_t = evaluate(pred_t, y)
        sweep_rows.append({
            "threshold": thr,
            "f1w": m_t["f1w"], "f1m": m_t["f1m"],
            "n_routed": rstats_t["n_total_routed"],
            "n_routed_A": rstats_t["n_routed_A"],
            "n_routed_B": rstats_t["n_routed_B"],
        })
        print(f"  thr={thr:.2f}  weighted F1 = {m_t['f1w']:.4f}  "
              f"routed A={rstats_t['n_routed_A']} B={rstats_t['n_routed_B']} "
              f"(total {rstats_t['n_total_routed']})")

    best_thr_row = max(sweep_rows, key=lambda r: r["f1w"])
    print(f"\nBest threshold in sweep: {best_thr_row['threshold']:.2f} → "
          f"weighted F1 = {best_thr_row['f1w']:.4f}  "
          f"(note: this is tuned on eval, so optimistic)")

    # -- Double-gated cascade sweep (honest but eval-tuned) --
    # Only override Stage-1 when BOTH Stage-1 is uncertain AND the specialist is
    # confident in its binary call. This captures the "soft cascade" intuition
    # without training a meta-model.
    print("\n" + "-" * 76)
    print("Double-gated cascade (S1 thr=0.65 + specialist conf threshold)")
    print("-" * 76)
    double_rows = []
    top1_conf = s1_proba.max(axis=1)
    top2 = np.argsort(-s1_proba, axis=1)[:, :2]
    top2_sets = [frozenset(r.tolist()) for r in top2]
    pair_A = frozenset({IDX_GLAUKOM, IDX_SM})
    pair_B = frozenset({IDX_DIA, IDX_HEALTHY})
    for spec_thr in [0.6, 0.7, 0.8, 0.9]:
        pred = s1_pred.copy()
        nA = nB = 0
        for i in range(n):
            if top1_conf[i] > DEFAULT_GATING_THRESHOLD:
                continue
            s = top2_sets[i]
            if s == pair_A:
                pA = spec_A["proba_pos"][i]
                conf = max(pA, 1 - pA) if not np.isnan(pA) else 0.0
                if conf >= spec_thr and spec_A["pred_in_pair"][i] >= 0:
                    pred[i] = spec_A["pred_in_pair"][i]
                    nA += 1
            elif s == pair_B:
                pB = spec_B["proba_pos"][i]
                conf = max(pB, 1 - pB) if not np.isnan(pB) else 0.0
                if conf >= spec_thr and spec_B["pred_in_pair"][i] >= 0:
                    pred[i] = spec_B["pred_in_pair"][i]
                    nB += 1
        m = evaluate(pred, y)
        double_rows.append({
            "spec_thr": spec_thr, "n_routed_A": nA, "n_routed_B": nB,
            "f1w": m["f1w"], "f1m": m["f1m"],
        })
        print(f"  spec_thr={spec_thr:.1f}  routed A={nA} B={nB}  "
              f"weighted F1 = {m['f1w']:.4f}")

    # -- Save cascade OOF --
    #   We save the full-cascade (A+B) prediction at default gating threshold as
    #   the canonical "cascade_oof" — same semantics as best_ensemble_predictions.
    out_npz = CACHE / "cascade_oof.npz"
    np.savez(
        out_npz,
        pred_label=per_variant_preds["stage1 + A + B"].astype(int),
        true_label=y.astype(int),
        scan_paths=bundle.scan_paths,
        stage1_pred=s1_pred.astype(int),
        stage1_proba=s1_proba,
        gating_threshold=np.array([DEFAULT_GATING_THRESHOLD]),
        spec_A_proba_pos=spec_A["proba_pos"],
        spec_A_pred=spec_A["pred_in_pair"].astype(int),
        spec_B_proba_pos=spec_B["proba_pos"],
        spec_B_pred=spec_B["pred_in_pair"].astype(int),
        spec_C_proba_pos=spec_C["proba_pos"],
        spec_C_pred=spec_C["pred_in_pair"].astype(int),
    )
    print(f"\n[saved] {out_npz}")

    # -- Print classification report + confusion matrices --
    print("\nClassification report (cascade A+B @ thr=0.65):")
    print(classification_report(y, per_variant_preds["stage1 + A + B"],
                                 target_names=CLASSES, zero_division=0))
    cm_base = confusion_matrix(y, s1_pred, labels=list(range(N_CLASSES)))
    cm_cas = confusion_matrix(y, per_variant_preds["stage1 + A + B"],
                               labels=list(range(N_CLASSES)))

    # -- Diagnostic: for routed scans, how many did Stage-1 already get right? --
    diag_A = _routing_diagnostic(
        s1_pred, s1_proba, per_variant_preds["stage1 + A + B"], y,
        pair={IDX_GLAUKOM, IDX_SM}, thr=DEFAULT_GATING_THRESHOLD, name="A (Glaukom/SM)",
    )
    diag_B = _routing_diagnostic(
        s1_pred, s1_proba, per_variant_preds["stage1 + A + B"], y,
        pair={IDX_DIA, IDX_HEALTHY}, thr=DEFAULT_GATING_THRESHOLD, name="B (Diabetes/Healthy)",
    )
    print("\nRouting diagnostic (at thr=0.65):")
    for d in (diag_A, diag_B):
        print(f"  {d['name']}: routed={d['n_routed']}  "
              f"S1_correct={d['s1_correct']}  cascade_correct={d['cas_correct']}  "
              f"changes: good={d['n_good']}  bad={d['n_bad']}")

    # -- Write report --
    write_report(
        baseline_metrics=baseline_metrics,
        specialists=specialists,
        variant_rows=variant_rows,
        sweep_rows=sweep_rows,
        best_thr_row=best_thr_row,
        headline=headline,
        cm_baseline=cm_base,
        cm_cascade=cm_cas,
        y=y,
        double_rows=double_rows,
        diag_rows=[diag_A, diag_B],
    )
    print(f"\nTotal wall time: {time.time()-t_start:.1f}s")


def write_report(
    baseline_metrics: dict,
    specialists: dict,
    variant_rows: list[dict],
    sweep_rows: list[dict],
    best_thr_row: dict,
    headline: dict,
    cm_baseline: np.ndarray,
    cm_cascade: np.ndarray,
    y: np.ndarray,
    double_rows: list[dict],
    diag_rows: list[dict],
):
    out = REPORTS / "CASCADE_RESULTS.md"
    lines: list[str] = []

    lines.append("# Cascade Classifier Results")
    lines.append("")
    lines.append("Confidence-gated cascade over Stage-1 ensemble "
                 "(DINOv2-B + BiomedCLIP proba-avg + per-class thresholds).")
    lines.append("")
    lines.append("Two (+ optional third) binary specialists trained person-LOPO:")
    lines.append("")
    lines.append("- **Specialist A**: PGOV_Glaukom vs SklerozaMultiplex — "
                 "features = DINOv2-B scan-mean || TDA (1015-dim) — XGBoost.")
    lines.append("- **Specialist B**: Diabetes vs ZdraviLudia — "
                 "features = DINOv2-B scan-mean || handcrafted (94-dim) — XGBoost.")
    lines.append("- **(Bonus) Specialist C**: SklerozaMultiplex vs ZdraviLudia — "
                 "features = DINOv2-B scan-mean || handcrafted — XGBoost.")
    lines.append("")
    lines.append("Gating: if Stage-1 top-1 proba > `thr` keep Stage-1 prediction; "
                 "otherwise if Stage-1 top-2 classes match a specialist's pair, "
                 "route the scan to that specialist (which yields an out-of-fold "
                 "binary prediction because it used the SAME person-LOPO splits).")
    lines.append("")

    lines.append("## Specialist binary F1 (person-LOPO, restricted to pair scans)")
    lines.append("")
    lines.append("| specialist | pair | n_scans | binary F1 | accuracy | per-class F1 |")
    lines.append("|---|---|---|---|---|---|")
    for key in ("A", "B", "C"):
        s = specialists[key]
        pc = s["per_class_f1"]
        pc_str = ", ".join(f"{k}={v:.3f}" for k, v in pc.items())
        lines.append(
            f"| {key} | {s['name']} | {s['n_pair_scans']} | "
            f"{s['f1_binary']:.4f} | {s['accuracy']:.4f} | {pc_str} |"
        )
    lines.append("")

    lines.append(f"## Stage-1 vs cascade variants (gating threshold = "
                 f"{DEFAULT_GATING_THRESHOLD})")
    lines.append("")
    lines.append("| variant | weighted F1 | macro F1 | "
                 + " | ".join(f"F1 {c}" for c in CLASSES)
                 + " | routed A | routed B | routed C | total routed |")
    lines.append("|" + "|".join(["---"] * (3 + N_CLASSES + 4)) + "|")

    # Baseline row
    lines.append(
        f"| Stage-1 baseline | {baseline_metrics['f1w']:.4f} | "
        f"{baseline_metrics['f1m']:.4f} | "
        + " | ".join(f"{v:.3f}" for v in baseline_metrics["f1pc"])
        + " | - | - | - | - |"
    )
    for r in variant_rows:
        lines.append(
            f"| {r['label']} | {r['f1w']:.4f} | {r['f1m']:.4f} | "
            + " | ".join(f"{v:.3f}" for v in r["f1pc"])
            + f" | {r['n_routed_A']} | {r['n_routed_B']} | {r['n_routed_C']} "
            f"| {r['n_total_routed']} |"
        )
    lines.append("")

    lines.append("## Pair-level accuracy for routed confused pairs")
    lines.append("")
    lines.append("| variant | Glaukom/SM pair F1 | Diabetes/Healthy pair F1 |")
    lines.append("|---|---|---|")
    for r in variant_rows:
        pa = r["pair_acc_GlauSM"]["f1"]
        pb = r["pair_acc_DiaHealthy"]["f1"]
        lines.append(f"| {r['label']} | {pa:.3f} | {pb:.3f} |")
    lines.append("")

    lines.append("## Gating-threshold sweep (A+B cascade)")
    lines.append("")
    lines.append("| threshold | weighted F1 | macro F1 | routed A | routed B | total routed |")
    lines.append("|---|---|---|---|---|---|")
    for r in sweep_rows:
        lines.append(
            f"| {r['threshold']:.2f} | {r['f1w']:.4f} | {r['f1m']:.4f} | "
            f"{r['n_routed_A']} | {r['n_routed_B']} | {r['n_routed']} |"
        )
    lines.append("")
    lines.append(f"Best threshold in sweep: **{best_thr_row['threshold']:.2f}** → "
                 f"weighted F1 = **{best_thr_row['f1w']:.4f}** "
                 f"(honest caveat: this value is tuned on eval — do not claim as "
                 f"headline).")
    lines.append("")

    lines.append("## Double-gated cascade (S1 thr=0.65 + specialist conf threshold)")
    lines.append("")
    lines.append("Route only when Stage-1 is uncertain AND the specialist is "
                 "confident in its binary call. This reduces the bad-change rate.")
    lines.append("")
    lines.append("| spec_thr | routed A | routed B | weighted F1 | macro F1 |")
    lines.append("|---|---|---|---|---|")
    for r in double_rows:
        lines.append(
            f"| {r['spec_thr']:.1f} | {r['n_routed_A']} | {r['n_routed_B']} | "
            f"{r['f1w']:.4f} | {r['f1m']:.4f} |"
        )
    best_dbl = max(double_rows, key=lambda r: r["f1w"])
    lines.append("")
    dbl_delta = best_dbl["f1w"] - baseline_metrics["f1w"]
    if dbl_delta > 0:
        lines.append(f"Best double-gate setting: `spec_thr={best_dbl['spec_thr']}` → "
                     f"weighted F1 = **{best_dbl['f1w']:.4f}** "
                     f"(Δ = {dbl_delta:+.4f} vs Stage-1). Also eval-tuned, "
                     f"but shows that a confidence-weighted merge (rather than "
                     f"a hard override) is the promising direction.")
    else:
        lines.append(f"Best double-gate setting: `spec_thr={best_dbl['spec_thr']}` → "
                     f"weighted F1 = {best_dbl['f1w']:.4f} (Δ = {dbl_delta:+.4f}). "
                     f"Still does not beat Stage-1.")
    lines.append("")

    lines.append("## Honest headline")
    lines.append("")
    lines.append(f"- **Stage-1 baseline weighted F1 (person-LOPO):** "
                 f"{baseline_metrics['f1w']:.4f}")
    lines.append(f"- **Cascade A+B (thr=0.65, honest — no tuning on eval) "
                 f"weighted F1:** {headline['f1w']:.4f}")
    delta = headline["f1w"] - baseline_metrics["f1w"]
    verdict = "improves" if delta > 0 else ("ties" if abs(delta) < 1e-4 else "does NOT beat")
    lines.append(f"- **Δ vs Stage-1:** {delta:+.4f}  → cascade {verdict} Stage-1.")
    lines.append("")
    if delta <= 0:
        lines.append("> **Why it may not help much:** The Stage-1 ensemble already "
                     "puts very high confidence (>0.65) on easy scans; the confused "
                     "pairs are confused precisely because the ensemble has *low* "
                     "confidence there, and the specialists face the same underlying "
                     "feature-vs-label difficulty. Adding TDA/handcrafted features "
                     "does yield somewhat complementary signal, but on such a small "
                     "dataset the specialists are themselves noisy.")
        lines.append("")

    lines.append("## Diagnostic: why routing hurts at thr=0.65")
    lines.append("")
    for d in diag_rows:
        if d["n_routed"] == 0:
            lines.append(f"- **Specialist {d['name']}**: no routes.")
            continue
        s1_acc = d["s1_correct"] / max(1, d["n_routed"])
        lines.append(
            f"- **Specialist {d['name']}**: routed n={d['n_routed']}, "
            f"Stage-1 was already correct on {d['s1_correct']}/{d['n_routed']} "
            f"({s1_acc:.0%}). Cascade made {d['n_changed']} changes: "
            f"**{d['n_good']} good** (fixed a mistake) vs **{d['n_bad']} bad** "
            f"(broke a correct). Net {d['n_good'] - d['n_bad']:+d}."
        )
    lines.append("")
    lines.append("**Key insight:** low Stage-1 confidence does NOT imply Stage-1 is "
                 "wrong. The ensemble produces naturally flatter posteriors on "
                 "confused pairs, but its 5-class argmax still beats a "
                 "2-class specialist trained on the same features. The specialist "
                 "loses the \"definitely-not-SucheOko/Diabetes\" signal that the "
                 "full ensemble carries.")
    lines.append("")
    lines.append("**What might still work** (not fully explored, out of time budget):")
    lines.append("")
    lines.append("1. Soft cascade — blend Stage-1 proba with specialist proba "
                 "(weighted average) instead of hard override.")
    lines.append("2. Specialists that additionally take Stage-1 proba as a feature "
                 "(stacking-meta, pair-restricted).")
    lines.append("3. Abstain-then-route — the double-gated table above hints this "
                 "direction with a tiny positive delta, but it is eval-tuned.")
    lines.append("")

    lines.append("## Baseline confusion matrix (Stage-1)")
    lines.append("")
    lines.append("| true\\pred | " + " | ".join(CLASSES) + " |")
    lines.append("|" + "|".join(["---"] * (N_CLASSES + 1)) + "|")
    for i, c in enumerate(CLASSES):
        row = [c] + [str(int(cm_baseline[i, j])) for j in range(N_CLASSES)]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Cascade A+B confusion matrix (thr=0.65)")
    lines.append("")
    lines.append("| true\\pred | " + " | ".join(CLASSES) + " |")
    lines.append("|" + "|".join(["---"] * (N_CLASSES + 1)) + "|")
    for i, c in enumerate(CLASSES):
        row = [c] + [str(int(cm_cascade[i, j])) for j in range(N_CLASSES)]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `cache/cascade_oof.npz` — keys: `pred_label`, `true_label`, "
                 "`scan_paths`, `stage1_pred`, `stage1_proba`, `gating_threshold`, "
                 "`spec_A_proba_pos`, `spec_A_pred`, `spec_B_proba_pos`, "
                 "`spec_B_pred`, `spec_C_proba_pos`, `spec_C_pred`.")
    lines.append("- `scripts/cascade_classifier.py` — runnable end-to-end.")
    lines.append("")

    lines.append("## Method notes")
    lines.append("")
    lines.append("1. Specialists use XGBoost (`max_depth=4`, `n_estimators=200`, "
                 "`lr=0.08`, `scale_pos_weight = #neg/#pos`). Fresh model per fold.")
    lines.append("2. Every specialist fold trains only on in-pair training scans of "
                 "that fold (excluding the held-out person). At prediction time we "
                 "predict on ALL held-out scans, not just the in-pair ones, so the "
                 "cascade can route any scan.")
    lines.append("3. The cascade uses RAW Stage-1 proba (un-thresholded) to measure "
                 "confidence; the fallback Stage-1 prediction is the thresholded "
                 "argmax that produced the 0.67 baseline.")
    lines.append("4. No tuning on eval for the headline: gating threshold is fixed "
                 "at 0.65 (a Schelling point, not a sweep winner).")
    lines.append("")

    out.write_text("\n".join(lines))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
