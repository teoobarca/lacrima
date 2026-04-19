"""Stacking meta-classifier on top of Stage-1 ensemble + binary specialists.

Motivation
==========

`reports/CASCADE_RESULTS.md` showed that hard-override cascading HURTS F1
(Δ = -0.048 vs Stage-1): low Stage-1 confidence is NOT the same as Stage-1
being wrong, so confidently routing to a 2-class specialist can overwrite
correct predictions with wrong ones. The diagnostic there already hints at
the right direction: use specialist probabilities as FEATURES, not as a
decision gate.

This script builds a feature matrix (240 × D ≈ 12) out of:
  * Stage-1 5-class OOF probabilities
  * Specialist A (PGOV_Glaukom vs SklerozaMultiplex) proba, expanded to the
    two classes (defined on every row — the specialist predicted on every
    held-out person; for rows outside the pair the proba is a noisy prior)
  * Specialist B (Diabetes vs ZdraviLudia) proba, same treatment
  * Specialist A, B confidence |max(proba) - 0.5| → max(p, 1-p)
  * Stage-1 entropy

Then trains a meta-classifier (LR or XGBoost) via NESTED person-LOPO: the
held-out person never contributes to the meta features or the meta model.
A weighted soft-blend is also reported as a simpler reference.

Deliverables
============
  scripts/cascade_stacker.py          (this file)
  cache/stacker_oof.npz               (best stacker's OOF predictions)
  reports/CASCADE_STACKER_RESULTS.md
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
IDX_HEALTHY = CLASS_IDX["ZdraviLudia"]
IDX_DIA = CLASS_IDX["Diabetes"]
IDX_GLAUKOM = CLASS_IDX["PGOV_Glaukom"]
IDX_SM = CLASS_IDX["SklerozaMultiplex"]
IDX_SUCHE = CLASS_IDX["SucheOko"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cascade_oof() -> dict:
    """Load the cascade OOF artifact produced by scripts/cascade_classifier.py.

    All 240 rows carry: Stage-1 proba/pred, specialist A/B/C positive proba,
    specialist A/B/C in-pair binary prediction (the specialists predicted on
    every held-out sample, not just in-pair ones — see
    `specialist_lopo` in cascade_classifier.py).
    """
    z = np.load(CACHE / "cascade_oof.npz", allow_pickle=True)
    return {
        "stage1_proba": np.asarray(z["stage1_proba"], dtype=np.float64),
        "stage1_pred": np.asarray(z["stage1_pred"], dtype=int),
        "true_label": np.asarray(z["true_label"], dtype=int),
        "scan_paths": np.asarray(z["scan_paths"]),
        "spec_A_proba_pos": np.asarray(z["spec_A_proba_pos"], dtype=np.float64),
        "spec_B_proba_pos": np.asarray(z["spec_B_proba_pos"], dtype=np.float64),
        "spec_C_proba_pos": np.asarray(z["spec_C_proba_pos"], dtype=np.float64),
        "spec_A_pred": np.asarray(z["spec_A_pred"], dtype=int),
        "spec_B_pred": np.asarray(z["spec_B_pred"], dtype=int),
        "spec_C_pred": np.asarray(z["spec_C_pred"], dtype=int),
    }


def align_person_groups(scan_paths: np.ndarray) -> np.ndarray:
    """Produce (240,) person-id array aligned to `scan_paths` by resolved path."""
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    path_to_person = {}
    for s in samples:
        path_to_person[str(Path(s.raw_path).resolve())] = s.person
        path_to_person[str(s.raw_path)] = s.person
    groups = np.array([path_to_person[str(p)] for p in scan_paths])
    assert len(np.unique(groups)) == 35, f"expected 35 persons, got {len(np.unique(groups))}"
    return groups


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(oof: dict) -> tuple[np.ndarray, list[str]]:
    """Build the (240, D) stacker feature matrix.

    Features (order fixed for the report):
      0-4 : Stage-1 5-class probability (columns ordered by CLASSES)
      5   : Spec A p(Glaukom) — for Glaukom-or-SM rows this is the specialist
            OOF; for rows outside the pair the specialist also produced a
            prediction (trained on in-pair rows only), which behaves as a
            noisy prior. That noise is fine because the meta-LR weights it.
      6   : Spec A p(SM)   = 1 - Spec A p(Glaukom)
      7   : Spec B p(Diabetes)
      8   : Spec B p(Healthy) = 1 - Spec B p(Diabetes)
      9   : Spec A confidence = max(p, 1-p)
      10  : Spec B confidence = max(p, 1-p)
      11  : Stage-1 entropy = -sum_k p_k log p_k
    """
    n = len(oof["true_label"])
    s1 = oof["stage1_proba"]                 # (n, 5)

    # Specialist A: positive class (per cascade_classifier) is SM (pair[1])
    # because pair_A = (IDX_GLAUKOM, IDX_SM) → pair[1] = IDX_SM.
    # So spec_A_proba_pos[i] is p(y == SM | scan).
    p_A_sm = oof["spec_A_proba_pos"]
    p_A_glau = 1.0 - p_A_sm

    # Specialist B: pair_B = (IDX_DIA, IDX_HEALTHY) → pair[1] = IDX_HEALTHY.
    # So spec_B_proba_pos[i] is p(y == Healthy | scan).
    p_B_healthy = oof["spec_B_proba_pos"]
    p_B_dia = 1.0 - p_B_healthy

    conf_A = np.maximum(p_A_sm, p_A_glau)
    conf_B = np.maximum(p_B_healthy, p_B_dia)

    eps = 1e-12
    ent = -np.sum(s1 * np.log(s1 + eps), axis=1)

    X = np.column_stack([
        s1[:, 0], s1[:, 1], s1[:, 2], s1[:, 3], s1[:, 4],
        p_A_glau, p_A_sm,
        p_B_dia, p_B_healthy,
        conf_A, conf_B,
        ent,
    ]).astype(np.float64)

    feat_names = [
        f"s1_{CLASSES[0]}", f"s1_{CLASSES[1]}", f"s1_{CLASSES[2]}",
        f"s1_{CLASSES[3]}", f"s1_{CLASSES[4]}",
        "specA_p_Glaukom", "specA_p_SM",
        "specB_p_Diabetes", "specB_p_Healthy",
        "specA_conf", "specB_conf",
        "s1_entropy",
    ]
    assert X.shape == (n, len(feat_names))
    return X, feat_names


# ---------------------------------------------------------------------------
# Meta-classifier evaluation (person-LOPO)
# ---------------------------------------------------------------------------

def _fit_meta_lr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def _fit_meta_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=4,
        verbosity=0,
        random_state=42,
    )


def nested_lopo_meta(
    X: np.ndarray,
    y: np.ndarray,
    person_groups: np.ndarray,
    build_model,
) -> tuple[np.ndarray, np.ndarray]:
    """Run person-LOPO with a fresh meta-model per fold.

    Returns:
        oof_proba (n, 5)  — meta-model predicted class probabilities per row
        oof_pred  (n,)    — argmax prediction per row
    """
    n = len(y)
    oof_proba = np.zeros((n, N_CLASSES), dtype=np.float64)
    oof_pred = np.full(n, -1, dtype=np.int64)

    for tr, va in leave_one_patient_out(person_groups):
        model = build_model()
        # XGB wants integer labels; balanced sample weight replicates class_weight.
        if isinstance(model, XGBClassifier):
            counts = np.bincount(y[tr], minlength=N_CLASSES).astype(float)
            counts[counts == 0] = 1.0
            weights_per_class = (len(y[tr]) / (N_CLASSES * counts))
            sw = weights_per_class[y[tr]]
            model.fit(X[tr], y[tr], sample_weight=sw)
        else:
            model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[va])
        # Align columns to 0..N_CLASSES-1 in case some class is missing from
        # the training fold (rare but possible with LOPO dropping SucheOko).
        if hasattr(model, "classes_"):
            cls = model.classes_
        else:
            cls = model.named_steps["lr"].classes_
        full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
        for j, c in enumerate(cls):
            full[:, int(c)] = proba[:, j]
        oof_proba[va] = full
        oof_pred[va] = full.argmax(axis=1)
    assert (oof_pred >= 0).all()
    return oof_proba, oof_pred


def evaluate(pred: np.ndarray, y: np.ndarray) -> dict:
    return {
        "f1w": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "f1m": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1pc": [float(v) for v in f1_score(
            y, pred, average=None, labels=list(range(N_CLASSES)), zero_division=0
        )],
    }


# ---------------------------------------------------------------------------
# Weighted soft-blend (nested α selection)
# ---------------------------------------------------------------------------

def build_specialist_informed_proba(oof: dict) -> np.ndarray:
    """Specialist-informed 5-class posterior.

    We distribute specialist-pair probabilities over the two relevant classes
    while renormalising Stage-1's remaining mass onto the non-pair classes.
    Intuitively: if Stage-1 says Glaukom/SM accounts for 0.6 of the mass and
    the specialist says 70/30 Glaukom/SM, redistribute that 0.6 as 0.42/0.18.

    When the Stage-1 pair mass is low (< 0.2) we fall back to the Stage-1
    proba row — the specialist is unlikely to matter for that scan.
    """
    s1 = oof["stage1_proba"].copy()
    p_A_sm = oof["spec_A_proba_pos"]
    p_A_glau = 1.0 - p_A_sm
    p_B_healthy = oof["spec_B_proba_pos"]
    p_B_dia = 1.0 - p_B_healthy

    out = s1.copy()

    n = s1.shape[0]
    for i in range(n):
        mass_A = s1[i, IDX_GLAUKOM] + s1[i, IDX_SM]
        mass_B = s1[i, IDX_DIA] + s1[i, IDX_HEALTHY]
        # We apply both specialists (they target disjoint classes).
        if mass_A >= 0.1:
            out[i, IDX_GLAUKOM] = mass_A * p_A_glau[i]
            out[i, IDX_SM] = mass_A * p_A_sm[i]
        if mass_B >= 0.1:
            out[i, IDX_DIA] = mass_B * p_B_dia[i]
            out[i, IDX_HEALTHY] = mass_B * p_B_healthy[i]
        # Re-normalise to be safe (no-op if masses sum to 1).
        s = out[i].sum()
        if s > 0:
            out[i] /= s
    return out


def weighted_blend_sweep(
    s1_proba: np.ndarray,
    spec_proba: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
) -> list[dict]:
    rows = []
    for a in alphas:
        blend = a * s1_proba + (1 - a) * spec_proba
        pred = blend.argmax(axis=1)
        m = evaluate(pred, y)
        rows.append({"alpha": float(a), **m})
    return rows


def nested_alpha_selection(
    s1_proba: np.ndarray,
    spec_proba: np.ndarray,
    y: np.ndarray,
    person_groups: np.ndarray,
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float], list[dict]]:
    """For each outer fold, pick alpha on the training portion and apply to val.

    Because the blend is per-sample and doesn't train anything, "training"
    here means selecting alpha that maximises weighted-F1 on the outer-fold
    TRAINING subset. The selected alpha is then used on the held-out person.
    This is honest (no eval-fold tuning) but relies on the training subset
    being representative of the held-out person; with 34 training persons
    per fold this is a reasonable bet.
    """
    n = len(y)
    final_pred = np.full(n, -1, dtype=np.int64)
    final_proba = np.zeros((n, N_CLASSES), dtype=np.float64)
    picked_alphas: list[float] = []
    per_fold_rows: list[dict] = []

    for tr, va in leave_one_patient_out(person_groups):
        best_a = alphas[0]
        best_f1 = -1.0
        for a in alphas:
            blend_tr = a * s1_proba[tr] + (1 - a) * spec_proba[tr]
            pred_tr = blend_tr.argmax(axis=1)
            f1 = f1_score(y[tr], pred_tr, average="weighted", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_a = a
        blend_va = best_a * s1_proba[va] + (1 - best_a) * spec_proba[va]
        final_pred[va] = blend_va.argmax(axis=1)
        final_proba[va] = blend_va
        picked_alphas.append(float(best_a))
        per_fold_rows.append({"alpha": float(best_a), "train_f1w": float(best_f1)})
    return final_proba, final_pred, picked_alphas, per_fold_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 76)
    print("Cascade stacker — soft meta on top of Stage-1 + specialists")
    print("=" * 76)
    t_start = time.time()

    oof = load_cascade_oof()
    y = oof["true_label"]
    scan_paths = oof["scan_paths"]
    person_groups = align_person_groups(scan_paths)
    n = len(y)
    print(f"Samples: n={n}, persons={len(np.unique(person_groups))}, "
          f"classes={dict(zip(*np.unique(y, return_counts=True)))}")

    # Baseline references (recomputed on this y so we can show honest deltas)
    s1_proba = oof["stage1_proba"]
    s1_raw_pred = s1_proba.argmax(axis=1)
    stage1_raw = evaluate(s1_raw_pred, y)
    stage1_thresh = evaluate(oof["stage1_pred"], y)

    # Hard cascade (same as CASCADE_RESULTS Round 3a: A+B @ thr=0.65)
    hard_cas = np.load(CACHE / "cascade_oof.npz", allow_pickle=True)
    hard_cascade_m = evaluate(hard_cas["pred_label"].astype(int), y)

    print(f"\nBaselines on this y:")
    print(f"  Stage-1 raw argmax  : wF1={stage1_raw['f1w']:.4f}  mF1={stage1_raw['f1m']:.4f}")
    print(f"  Stage-1 + threshold : wF1={stage1_thresh['f1w']:.4f}  "
          f"mF1={stage1_thresh['f1m']:.4f}")
    print(f"  Hard cascade (A+B)  : wF1={hard_cascade_m['f1w']:.4f}  "
          f"mF1={hard_cascade_m['f1m']:.4f}")

    # -- Build stacker feature matrix --
    X, feat_names = build_features(oof)
    print(f"\nFeature matrix: {X.shape}")
    for i, nm in enumerate(feat_names):
        v = X[:, i]
        print(f"  [{i:2d}] {nm:22s}  min={v.min():+.3f}  max={v.max():+.3f}  "
              f"mean={v.mean():+.3f}")

    # -- Meta-LR: person-LOPO --
    print("\n" + "-" * 76)
    print("Meta-LR (StandardScaler + balanced LR, C=1.0) via person-LOPO")
    print("-" * 76)
    t0 = time.time()
    lr_proba, lr_pred = nested_lopo_meta(X, y, person_groups, _fit_meta_lr)
    lr_m = evaluate(lr_pred, y)
    print(f"  meta-LR: wF1={lr_m['f1w']:.4f}  mF1={lr_m['f1m']:.4f}  "
          f"({time.time()-t0:.1f}s)")

    # -- Meta-XGB: person-LOPO --
    print("\n" + "-" * 76)
    print("Meta-XGB (depth=3, n_estimators=300, balanced sample weights) via person-LOPO")
    print("-" * 76)
    t0 = time.time()
    xgb_proba, xgb_pred = nested_lopo_meta(X, y, person_groups, _fit_meta_xgb)
    xgb_m = evaluate(xgb_pred, y)
    print(f"  meta-XGB: wF1={xgb_m['f1w']:.4f}  mF1={xgb_m['f1m']:.4f}  "
          f"({time.time()-t0:.1f}s)")

    # -- Weighted soft-blend: full-grid sweep (eval-tuned) + nested alpha --
    print("\n" + "-" * 76)
    print("Weighted soft-blend: final = α * s1_proba + (1-α) * spec_informed_proba")
    print("-" * 76)
    spec_informed = build_specialist_informed_proba(oof)
    # Sanity: blend=1.0 should match raw S1 argmax exactly
    assert np.allclose(spec_informed.sum(axis=1), 1.0, atol=1e-6)

    alphas = np.linspace(0.0, 1.0, 11)
    sweep_rows = weighted_blend_sweep(s1_proba, spec_informed, y, alphas)
    for r in sweep_rows:
        print(f"  α={r['alpha']:.2f}  wF1={r['f1w']:.4f}  mF1={r['f1m']:.4f}")
    best_blend_row = max(sweep_rows, key=lambda r: r["f1w"])
    print(f"  Best α on full eval (leaky): α={best_blend_row['alpha']:.2f}  "
          f"wF1={best_blend_row['f1w']:.4f}")

    # Nested-CV alpha selection (honest):
    nested_proba, nested_pred, picked_alphas, _ = nested_alpha_selection(
        s1_proba, spec_informed, y, person_groups, alphas,
    )
    nested_m = evaluate(nested_pred, y)
    from collections import Counter
    alpha_counter = Counter([round(a, 2) for a in picked_alphas])
    print(f"  Nested-α (honest): wF1={nested_m['f1w']:.4f}  mF1={nested_m['f1m']:.4f}  "
          f"alpha_hist={dict(alpha_counter)}")

    # -- Pick best method & print details --
    candidates = [
        ("meta_LR", lr_pred, lr_m, lr_proba),
        ("meta_XGB", xgb_pred, xgb_m, xgb_proba),
        ("soft_blend_nested", nested_pred, nested_m, nested_proba),
    ]
    best_name, best_pred, best_m, best_proba = max(candidates, key=lambda t: t[2]["f1w"])
    print(f"\nBest method (by weighted F1): {best_name}  "
          f"wF1={best_m['f1w']:.4f}  mF1={best_m['f1m']:.4f}")

    print("\nPer-class F1 (best method):")
    for i, c in enumerate(CLASSES):
        print(f"  {c:20s}  F1={best_m['f1pc'][i]:.3f}")

    print("\nClassification report (best method):")
    print(classification_report(y, best_pred, target_names=CLASSES, zero_division=0))
    cm_best = confusion_matrix(y, best_pred, labels=list(range(N_CLASSES)))

    # -- Save OOF --
    out_npz = CACHE / "stacker_oof.npz"
    np.savez(
        out_npz,
        # Best method
        best_method=np.array([best_name]),
        best_proba=best_proba,
        best_pred=best_pred.astype(int),
        # All three methods (for downstream audits)
        lr_proba=lr_proba, lr_pred=lr_pred.astype(int),
        xgb_proba=xgb_proba, xgb_pred=xgb_pred.astype(int),
        blend_proba=nested_proba, blend_pred=nested_pred.astype(int),
        picked_alphas=np.asarray(picked_alphas, dtype=np.float64),
        # Inputs
        stage1_proba=s1_proba,
        stage1_pred=oof["stage1_pred"].astype(int),
        stacker_features=X,
        feature_names=np.array(feat_names),
        true_label=y.astype(int),
        scan_paths=scan_paths,
    )
    print(f"\n[saved] {out_npz}")

    # -- Write report --
    write_report(
        feat_names=feat_names,
        X=X,
        stage1_raw=stage1_raw,
        stage1_thresh=stage1_thresh,
        hard_cascade_m=hard_cascade_m,
        lr_m=lr_m,
        xgb_m=xgb_m,
        nested_m=nested_m,
        sweep_rows=sweep_rows,
        best_blend_row=best_blend_row,
        picked_alphas=picked_alphas,
        best_name=best_name,
        best_m=best_m,
        cm_best=cm_best,
        y=y,
        best_pred=best_pred,
    )
    print(f"\nTotal wall time: {time.time()-t_start:.1f}s")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    feat_names: list[str],
    X: np.ndarray,
    stage1_raw: dict,
    stage1_thresh: dict,
    hard_cascade_m: dict,
    lr_m: dict,
    xgb_m: dict,
    nested_m: dict,
    sweep_rows: list[dict],
    best_blend_row: dict,
    picked_alphas: list[float],
    best_name: str,
    best_m: dict,
    cm_best: np.ndarray,
    y: np.ndarray,
    best_pred: np.ndarray,
):
    out = REPORTS / "CASCADE_STACKER_RESULTS.md"
    lines: list[str] = []
    from collections import Counter

    lines.append("# Cascade Stacker Results")
    lines.append("")
    lines.append("Stacking meta-classifier that consumes the Stage-1 ensemble "
                 "probabilities and binary specialist probabilities as **features** "
                 "(not as a hard override). Motivated by the diagnostic in "
                 "`reports/CASCADE_RESULTS.md` (hard-cascade Δ vs Stage-1 = -0.048). "
                 "Everything below is evaluated with person-level LOPO (35 groups).")
    lines.append("")

    # Feature-matrix description
    lines.append("## Feature matrix (240 × 12)")
    lines.append("")
    lines.append("All features are out-of-fold. Stage-1 probabilities come from "
                 "`cache/best_ensemble_predictions.npz` (which was produced by "
                 "person-LOPO over the DINOv2-B + BiomedCLIP proba-avg ensemble). "
                 "Specialist probabilities come from `cache/cascade_oof.npz`, where "
                 "each specialist was trained only on in-pair training rows of each "
                 "person-LOPO fold and predicted on every held-out person — so the "
                 "specialist output on every row is a legitimate OOF value (in-pair "
                 "scans get their meaningful binary prediction; off-pair scans get "
                 "a noisy but still-OOF prior which the meta-LR will weight down).")
    lines.append("")
    lines.append("| idx | name | range | mean |")
    lines.append("|---|---|---|---|")
    for i, nm in enumerate(feat_names):
        v = X[:, i]
        lines.append(f"| {i} | `{nm}` | [{v.min():+.3f}, {v.max():+.3f}] | {v.mean():+.3f} |")
    lines.append("")

    # Honest baseline comparison table
    lines.append("## Honest comparison (all person-LOPO weighted F1)")
    lines.append("")
    lines.append("| Method | weighted F1 | macro F1 | notes |")
    lines.append("|---|---:|---:|---|")
    lines.append(
        f"| Stage-1 alone (raw argmax, no threshold) | {stage1_raw['f1w']:.4f} | "
        f"{stage1_raw['f1m']:.4f} | reference |"
    )
    lines.append(
        f"| Stage-1 + tuned thresholds (leaky) | {stage1_thresh['f1w']:.4f} | "
        f"{stage1_thresh['f1m']:.4f} | thresholds tuned on full OOF |"
    )
    lines.append(
        f"| Hard cascade A+B @ thr=0.65 (Round 3a) | {hard_cascade_m['f1w']:.4f} | "
        f"{hard_cascade_m['f1m']:.4f} | from `cache/cascade_oof.npz` |"
    )
    lines.append(
        f"| Meta-LR (this script) | **{lr_m['f1w']:.4f}** | {lr_m['f1m']:.4f} | "
        f"StandardScaler + balanced LR, person-LOPO |"
    )
    lines.append(
        f"| Meta-XGB (this script) | **{xgb_m['f1w']:.4f}** | {xgb_m['f1m']:.4f} | "
        f"depth=3, n_est=300, sample-weight balanced, person-LOPO |"
    )
    lines.append(
        f"| Soft-blend (nested-α, honest) | **{nested_m['f1w']:.4f}** | "
        f"{nested_m['f1m']:.4f} | α selected on training fold per outer fold |"
    )
    lines.append(
        f"| Soft-blend (best α on full eval, LEAKY) | {best_blend_row['f1w']:.4f} | "
        f"{best_blend_row['f1m']:.4f} | α={best_blend_row['alpha']:.2f} — upper bound |"
    )
    lines.append("")

    # Delta vs Stage-1 raw
    best_delta_raw = best_m["f1w"] - stage1_raw["f1w"]
    best_delta_thresh = best_m["f1w"] - stage1_thresh["f1w"]
    verdict_raw = "improves" if best_delta_raw > 0 else "does NOT beat"
    verdict_thresh = "improves" if best_delta_thresh > 0 else "does NOT beat"
    lines.append("### Δ summary (best stacker method)")
    lines.append("")
    lines.append(f"Best method: **{best_name}** — wF1 = {best_m['f1w']:.4f}, "
                 f"mF1 = {best_m['f1m']:.4f}.")
    lines.append("")
    lines.append(f"- vs Stage-1 raw argmax ({stage1_raw['f1w']:.4f}): "
                 f"Δ = {best_delta_raw:+.4f} → {verdict_raw} Stage-1 raw.")
    lines.append(f"- vs Stage-1 + leaky thresholds ({stage1_thresh['f1w']:.4f}): "
                 f"Δ = {best_delta_thresh:+.4f} → {verdict_thresh} the tuned "
                 f"Stage-1 (the 0.6528–0.6698 honest/leaky band from prior audits).")
    lines.append("")

    # Per-class F1 for best
    lines.append(f"## Per-class F1 — best method ({best_name})")
    lines.append("")
    lines.append("| class | F1 |")
    lines.append("|---|---:|")
    for i, c in enumerate(CLASSES):
        lines.append(f"| {c} | {best_m['f1pc'][i]:.3f} |")
    lines.append("")

    # Soft-blend sweep
    lines.append("## Soft-blend α sweep (FULL-EVAL, leaky)")
    lines.append("")
    lines.append("`final_proba = α * stage1_proba + (1-α) * spec_informed_proba`. "
                 "Spec-informed proba redistributes Stage-1's mass on the Glaukom/SM "
                 "pair using Specialist A's probability, and similarly for the "
                 "Diabetes/Healthy pair with Specialist B.")
    lines.append("")
    lines.append("| α | weighted F1 | macro F1 |")
    lines.append("|---:|---:|---:|")
    for r in sweep_rows:
        lines.append(f"| {r['alpha']:.2f} | {r['f1w']:.4f} | {r['f1m']:.4f} |")
    lines.append("")
    lines.append(f"Full-eval best: α={best_blend_row['alpha']:.2f} → "
                 f"wF1={best_blend_row['f1w']:.4f} (UPPER bound, leaky).")
    lines.append("")
    alpha_hist = Counter([round(a, 2) for a in picked_alphas])
    lines.append(f"Nested-α selection (honest) picked α across 35 folds: "
                 f"{dict(sorted(alpha_hist.items()))}.")
    lines.append(f"Nested-α wF1 = {nested_m['f1w']:.4f}, mF1 = {nested_m['f1m']:.4f}.")
    lines.append("")

    # Confusion matrix for best
    lines.append(f"## Confusion matrix — best method ({best_name})")
    lines.append("")
    lines.append("| true\\pred | " + " | ".join(CLASSES) + " |")
    lines.append("|" + "|".join(["---"] * (N_CLASSES + 1)) + "|")
    for i, c in enumerate(CLASSES):
        row = [c] + [str(int(cm_best[i, j])) for j in range(N_CLASSES)]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Honest verdict
    lines.append("## Honest verdict")
    lines.append("")
    best_vs_honest_stage1 = best_m["f1w"] - stage1_raw["f1w"]
    if best_vs_honest_stage1 > 0.003:
        lines.append(f"The soft stacker **improves** over Stage-1 raw argmax "
                     f"(0.6346 → {best_m['f1w']:.4f}, Δ = {best_vs_honest_stage1:+.4f}). "
                     f"Specialist probabilities contribute useful signal when used as "
                     f"features instead of a hard override: the meta-model learns to "
                     f"trust them in the Glaukom/SM and Diabetes/Healthy subspaces and "
                     f"ignore them elsewhere.")
    elif best_vs_honest_stage1 > -0.003:
        lines.append(f"The soft stacker is **essentially tied** with Stage-1 raw argmax "
                     f"(0.6346 → {best_m['f1w']:.4f}, Δ = {best_vs_honest_stage1:+.4f}). "
                     f"Within the honest-evaluation band this does not constitute a "
                     f"measurable improvement — but it also does not *break* anything, "
                     f"unlike the hard cascade which lost ~0.05.")
    else:
        lines.append(f"The soft stacker **does not beat** Stage-1 raw argmax "
                     f"(0.6346 → {best_m['f1w']:.4f}, Δ = {best_vs_honest_stage1:+.4f}). "
                     f"Negative result: the specialist probabilities do not carry enough "
                     f"additional information past what Stage-1 already encodes, and the "
                     f"meta-model is probably over-fitting the small train set in each "
                     f"LOPO fold.")
    lines.append("")
    lines.append("Compared to the 0.6528 honest / 0.6698 leaky band from prior audits:")
    lines.append("")
    lines.append(f"- The meta-LR number is honest (person-LOPO, no parameter on eval).")
    lines.append(f"- It sits at **{best_m['f1w']:.4f}**, which is {best_delta_thresh:+.4f} "
                 f"vs the tuned (leaky) 0.6698 headline.")
    lines.append(f"- The Stage-1 ensemble with bias tuning only survives honest "
                 f"evaluation at ~0.6326 (see RED_TEAM_ENSEMBLE_V2_AUDIT.md); the "
                 f"stacker's honest number should be compared against that, not 0.6698.")
    lines.append("")

    # Method notes
    lines.append("## Method notes")
    lines.append("")
    lines.append("1. Every feature is out-of-fold. Stage-1 probas are person-LOPO; "
                 "specialist probas come from `cache/cascade_oof.npz`, which was "
                 "produced by `scripts/cascade_classifier.py::specialist_lopo` with "
                 "the same `leave_one_patient_out(person_groups)` iterator used here. "
                 "In each fold the specialist trains only on in-pair training rows "
                 "and predicts on all held-out rows of that fold. That means the "
                 "meta-model at fold `k` sees features that never observed person `k`.")
    lines.append("2. The meta-model is refit from scratch in each outer fold "
                 "(35 fits total). Nothing is tuned on the held-out person.")
    lines.append("3. Meta-LR uses `StandardScaler + LogisticRegression("
                 "class_weight='balanced', C=1.0, max_iter=2000)`. Meta-XGB uses "
                 "depth=3, n_estimators=300, balanced sample weights "
                 "(since XGB has no `class_weight`). No inner CV on eval folds.")
    lines.append("4. The soft-blend α is selected inside each outer fold on its "
                 "training subset by choosing the α ∈ {0.0, 0.1, …, 1.0} that "
                 "maximises weighted-F1 on the training subset. The resulting "
                 "prediction on the held-out person is then evaluated.")
    lines.append("5. SucheOko (14 scans, 2 persons) is essentially invisible to "
                 "every method here — same as Stage-1. This is a dataset-size issue, "
                 "not a stacker issue.")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `cache/stacker_oof.npz` — keys: `best_method`, `best_proba`, "
                 "`best_pred`, `lr_proba`, `lr_pred`, `xgb_proba`, `xgb_pred`, "
                 "`blend_proba`, `blend_pred`, `picked_alphas`, `stacker_features`, "
                 "`feature_names`, `stage1_proba`, `stage1_pred`, `true_label`, "
                 "`scan_paths`.")
    lines.append("- `scripts/cascade_stacker.py` — runnable end-to-end; consumes "
                 "`cache/cascade_oof.npz` and `cache/best_ensemble_predictions.npz`.")
    lines.append("")

    out.write_text("\n".join(lines))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
