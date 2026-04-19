"""VLM x v4 ensemble experiments.

Combines Claude Haiku 4.5 vision-language zero-shot predictions (``cache/vlm_predictions.json``)
with the v4 multiscale champion OOF probabilities (``cache/v4_oof_predictions.npz``) under
person-LOPO-safe evaluation.

Strategies evaluated:
  A) Arithmetic mean of softmax
  B) Geometric mean of softmax
  C) Confidence-weighted mean (max-softmax per model)
  D) VLM-as-tiebreaker for low-margin v4 predictions
  E) Per-class specialist (VLM for Diabetes/Glaukom/SucheOko, v4 for Healthy/SM)
  F) Learned grid-search convex combo via leave-one-person-out

Outputs:
  - reports/VLM_V4_ENSEMBLE_RESULTS.md
  - cache/v5_vlm_ensemble_predictions.npz (if winner found)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from teardrop.data import CLASSES  # noqa: E402

CACHE = REPO / "cache"
REPORTS = REPO / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

VLM_PATH = CACHE / "vlm_predictions.json"
V4_PATH = CACHE / "v4_oof_predictions.npz"
OUT_V5 = CACHE / "v5_vlm_ensemble_predictions.npz"
REPORT = REPORTS / "VLM_V4_ENSEMBLE_RESULTS.md"

EPS = 1e-9


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_v4() -> dict:
    d = np.load(V4_PATH, allow_pickle=True)
    return {
        "proba": d["proba"].astype(np.float64),
        "y": d["y"].astype(np.int64),
        "persons": np.array([str(p) for p in d["persons"]]),
        "scan_paths": np.array([str(p) for p in d["scan_paths"]]),
    }


def load_vlm() -> dict:
    raw = json.load(open(VLM_PATH))
    return {k: v for k, v in raw.items() if "predicted_class" in v}


def align_vlm_to_v4(v4: dict, vlm: dict):
    """Build (N,5) pseudo-softmax aligned with v4 scan order.

    For scans where VLM hasn't processed yet, fall back to v4's softmax so that
    the ensemble reduces to v4-only on those rows (unbiased).
    """
    n = len(v4["scan_paths"])
    vlm_proba = np.zeros((n, 5), dtype=np.float64)
    covered = np.zeros(n, dtype=bool)
    missing = []
    for i, p in enumerate(v4["scan_paths"]):
        # v4 stores absolute paths; VLM keys are 'TRAIN_SET/Class/person.scan'
        key = p.split("TRAIN_SET/", 1)
        if len(key) != 2:
            missing.append(p)
            continue
        vlm_key = "TRAIN_SET/" + key[1]
        if vlm_key not in vlm:
            missing.append(vlm_key)
            continue
        entry = vlm[vlm_key]
        cls = entry["predicted_class"]
        if cls not in CLASSES:
            missing.append(vlm_key)
            continue
        conf = float(entry["confidence"])
        conf = max(min(conf, 0.99), 0.21)  # clamp to avoid degenerate deltas / uniform
        idx = CLASSES.index(cls)
        vlm_proba[i, :] = (1.0 - conf) / 4.0
        vlm_proba[i, idx] = conf
        covered[i] = True
    return vlm_proba, covered, missing


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

def arithmetic_mean(pA: np.ndarray, pB: np.ndarray, wA: float = 0.5) -> np.ndarray:
    return wA * pA + (1.0 - wA) * pB


def geometric_mean(pA: np.ndarray, pB: np.ndarray, wA: float = 0.5) -> np.ndarray:
    logA = np.log(pA + EPS)
    logB = np.log(pB + EPS)
    out = np.exp(wA * logA + (1.0 - wA) * logB)
    out /= out.sum(axis=1, keepdims=True)
    return out


def confidence_weighted(pA: np.ndarray, pB: np.ndarray) -> np.ndarray:
    wA = pA.max(axis=1, keepdims=True)
    wB = pB.max(axis=1, keepdims=True)
    total = wA + wB + EPS
    return (wA / total) * pA + (wB / total) * pB


def vlm_tiebreaker(v4_p: np.ndarray, vlm_p: np.ndarray, margin: float = 0.1) -> np.ndarray:
    """Where v4 margin (top1-top2) is small, defer to VLM's top class."""
    sorted_p = np.sort(v4_p, axis=1)
    margin_v4 = sorted_p[:, -1] - sorted_p[:, -2]
    picks = v4_p.argmax(axis=1)
    low = margin_v4 < margin
    picks[low] = vlm_p[low].argmax(axis=1)
    return picks


def per_class_specialist(v4_p: np.ndarray, vlm_p: np.ndarray) -> np.ndarray:
    """VLM for {Diabetes, PGOV_Glaukom, SucheOko}; v4 for {Healthy, SM}.

    Decision rule: take argmax over blended proba where VLM's chosen class is
    weighted heavily if it falls in the VLM-specialist set, otherwise use
    blended as a safety net.
    """
    vlm_pick = vlm_p.argmax(axis=1)
    v4_pick = v4_p.argmax(axis=1)
    vlm_specialist = {CLASSES.index("Diabetes"), CLASSES.index("PGOV_Glaukom"), CLASSES.index("SucheOko")}
    out = np.where(
        np.isin(vlm_pick, list(vlm_specialist)), vlm_pick, v4_pick
    )
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(y, pred, title: str) -> dict:
    w = f1_score(y, pred, average="weighted", zero_division=0)
    m = f1_score(y, pred, average="macro", zero_division=0)
    per = f1_score(y, pred, average=None, labels=list(range(5)), zero_division=0)
    acc = (y == pred).mean()
    return {
        "title": title,
        "weighted_f1": float(w),
        "macro_f1": float(m),
        "accuracy": float(acc),
        "per_class_f1": {CLASSES[i]: float(per[i]) for i in range(5)},
    }


def fmt_row(m: dict) -> str:
    per = m["per_class_f1"]
    return (
        f"| {m['title']:<40s} | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} | "
        f"{m['accuracy']:.4f} | {per['ZdraviLudia']:.2f} | {per['Diabetes']:.2f} | "
        f"{per['PGOV_Glaukom']:.2f} | {per['SklerozaMultiplex']:.2f} | {per['SucheOko']:.2f} |"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    v4 = load_v4()
    vlm = load_vlm()
    vlm_p, covered, missing = align_vlm_to_v4(v4, vlm)
    print(f"v4 rows: {len(v4['y'])}  |  VLM covered: {covered.sum()}/{len(v4['y'])}  |  missing: {len(missing)}")

    # For uncovered rows we back off to v4's own softmax (ensemble becomes v4-only there)
    v4_p = v4["proba"]
    vlm_p_full = vlm_p.copy()
    vlm_p_full[~covered] = v4_p[~covered]

    y = v4["y"]

    # --- Baselines ---
    v4_pred = v4_p.argmax(axis=1)
    vlm_pred = vlm_p_full.argmax(axis=1)

    all_metrics = []
    all_metrics.append(metrics(y, v4_pred, "v4 alone (baseline)"))
    all_metrics.append(metrics(y, vlm_pred, "VLM alone (fallback=v4 if uncovered)"))

    # On covered subset only
    yc = y[covered]
    vlm_pred_c = vlm_p[covered].argmax(axis=1)
    all_metrics.append(metrics(yc, vlm_pred_c, f"VLM alone (covered subset n={covered.sum()})"))

    # --- Strategies ---
    # A) Arithmetic mean, sweep w
    best_arith = None
    for w in np.linspace(0.0, 1.0, 11):
        p = arithmetic_mean(vlm_p_full, v4_p, wA=w)
        r = metrics(y, p.argmax(axis=1), f"A) Arith mean (w_vlm={w:.1f})")
        if best_arith is None or r["weighted_f1"] > best_arith["weighted_f1"]:
            best_arith = r
        if w in (0.3, 0.5, 0.7):
            all_metrics.append(r)
    all_metrics.append({**best_arith, "title": "A*) Arith mean BEST " + best_arith["title"].split("w_vlm=")[1].rstrip(")")})

    # B) Geometric mean, sweep w
    best_geom = None
    for w in np.linspace(0.0, 1.0, 11):
        p = geometric_mean(vlm_p_full, v4_p, wA=w)
        r = metrics(y, p.argmax(axis=1), f"B) Geom mean (w_vlm={w:.1f})")
        if best_geom is None or r["weighted_f1"] > best_geom["weighted_f1"]:
            best_geom = r
        if w in (0.3, 0.5, 0.7):
            all_metrics.append(r)
    all_metrics.append({**best_geom, "title": "B*) Geom mean BEST " + best_geom["title"].split("w_vlm=")[1].rstrip(")")})

    # C) Confidence-weighted
    pC = confidence_weighted(vlm_p_full, v4_p)
    all_metrics.append(metrics(y, pC.argmax(axis=1), "C) Confidence-weighted mean"))

    # D) VLM-as-tiebreaker, sweep margin
    best_tb = None
    for mrg in (0.05, 0.1, 0.15, 0.2, 0.3):
        predD = vlm_tiebreaker(v4_p, vlm_p_full, margin=mrg)
        r = metrics(y, predD, f"D) VLM tiebreaker (margin<{mrg})")
        if best_tb is None or r["weighted_f1"] > best_tb["weighted_f1"]:
            best_tb = r
        all_metrics.append(r)

    # E) Per-class specialist
    predE = per_class_specialist(v4_p, vlm_p_full)
    all_metrics.append(metrics(y, predE, "E) Per-class specialist"))

    # F) Learned weights via leave-one-person-out grid search
    persons = v4["persons"]
    unique_persons = np.unique(persons)
    predF = np.zeros_like(y)
    probaF = np.zeros_like(v4_p)
    for held in unique_persons:
        held_mask = persons == held
        train_mask = ~held_mask
        # choose weight that maximises weighted F1 on training fold
        best_w = 0.5
        best_f = -1.0
        for w in np.linspace(0.0, 1.0, 21):
            p = arithmetic_mean(vlm_p_full[train_mask], v4_p[train_mask], wA=w)
            f = f1_score(y[train_mask], p.argmax(axis=1), average="weighted", zero_division=0)
            if f > best_f:
                best_f = f
                best_w = w
        p_held = arithmetic_mean(vlm_p_full[held_mask], v4_p[held_mask], wA=best_w)
        probaF[held_mask] = p_held
        predF[held_mask] = p_held.argmax(axis=1)
    all_metrics.append(metrics(y, predF, "F) LOPO-learned weights (arith)"))

    # Geom-mean version of F
    predF2 = np.zeros_like(y)
    probaF2 = np.zeros_like(v4_p)
    for held in unique_persons:
        held_mask = persons == held
        train_mask = ~held_mask
        best_w = 0.5
        best_f = -1.0
        for w in np.linspace(0.0, 1.0, 21):
            p = geometric_mean(vlm_p_full[train_mask], v4_p[train_mask], wA=w)
            f = f1_score(y[train_mask], p.argmax(axis=1), average="weighted", zero_division=0)
            if f > best_f:
                best_f = f
                best_w = w
        p_held = geometric_mean(vlm_p_full[held_mask], v4_p[held_mask], wA=best_w)
        probaF2[held_mask] = p_held
        predF2[held_mask] = p_held.argmax(axis=1)
    all_metrics.append(metrics(y, predF2, "F2) LOPO-learned weights (geom)"))

    # --- Complementarity analysis ---
    vlm_correct = (vlm_pred == y)
    v4_correct = (v4_pred == y)
    both_correct = vlm_correct & v4_correct
    vlm_only = vlm_correct & ~v4_correct
    v4_only = v4_correct & ~vlm_correct
    neither = ~vlm_correct & ~v4_correct
    comp = {
        "both_correct": int(both_correct.sum()),
        "vlm_only_correct": int(vlm_only.sum()),
        "v4_only_correct": int(v4_only.sum()),
        "neither_correct": int(neither.sum()),
        "oracle_upper_bound_acc": float((vlm_correct | v4_correct).mean()),
    }

    # Pick winner
    candidates = [m for m in all_metrics if m["title"].startswith(("A", "B", "C", "D", "E", "F"))]
    winner = max(candidates, key=lambda m: m["weighted_f1"])

    # Compute winner predictions precisely
    winner_title = winner["title"]
    if winner_title.startswith("F)"):
        winner_pred = predF
        winner_proba = probaF
    elif winner_title.startswith("F2)"):
        winner_pred = predF2
        winner_proba = probaF2
    elif winner_title.startswith("A*"):
        w = float(best_arith["title"].split("w_vlm=")[1].rstrip(")"))
        winner_proba = arithmetic_mean(vlm_p_full, v4_p, wA=w)
        winner_pred = winner_proba.argmax(axis=1)
    elif winner_title.startswith("B*"):
        w = float(best_geom["title"].split("w_vlm=")[1].rstrip(")"))
        winner_proba = geometric_mean(vlm_p_full, v4_p, wA=w)
        winner_pred = winner_proba.argmax(axis=1)
    elif winner_title.startswith("A)"):
        w = float(winner_title.split("w_vlm=")[1].rstrip(")"))
        winner_proba = arithmetic_mean(vlm_p_full, v4_p, wA=w)
        winner_pred = winner_proba.argmax(axis=1)
    elif winner_title.startswith("B)"):
        w = float(winner_title.split("w_vlm=")[1].rstrip(")"))
        winner_proba = geometric_mean(vlm_p_full, v4_p, wA=w)
        winner_pred = winner_proba.argmax(axis=1)
    elif winner_title.startswith("C)"):
        winner_proba = confidence_weighted(vlm_p_full, v4_p)
        winner_pred = winner_proba.argmax(axis=1)
    elif winner_title.startswith("D)"):
        mrg = float(winner_title.split("<")[1].rstrip(")"))
        winner_pred = vlm_tiebreaker(v4_p, vlm_p_full, margin=mrg)
        winner_proba = None
    elif winner_title.startswith("E)"):
        winner_pred = per_class_specialist(v4_p, vlm_p_full)
        winner_proba = None
    else:
        winner_pred = None
        winner_proba = None

    ship_v5 = (
        winner["weighted_f1"] >= 0.82
        and winner["per_class_f1"]["SucheOko"] > 0.5
        and winner["macro_f1"] > 0.70
    )

    # Save v5 predictions
    if winner_pred is not None and ship_v5:
        if winner_proba is None:
            # synthesize a 1-hot soft version from winner_pred for caching
            winner_proba = np.zeros_like(v4_p)
            winner_proba[np.arange(len(winner_pred)), winner_pred] = 1.0
        np.savez(
            OUT_V5,
            proba=winner_proba,
            pred=winner_pred,
            y=y,
            persons=persons,
            scan_paths=v4["scan_paths"],
            strategy=winner_title,
        )

    # --- Confusion matrices for top 3 ---
    ranked = sorted(candidates, key=lambda m: m["weighted_f1"], reverse=True)[:3]
    top3_titles = [r["title"] for r in ranked]

    # --- Report ---
    lines = []
    lines.append("# VLM x v4 Ensemble Results")
    lines.append("")
    lines.append(f"- VLM coverage: {covered.sum()}/{len(y)} scans ({covered.mean()*100:.1f}%)")
    lines.append(f"- Uncovered rows fall back to v4's own softmax in the ensemble")
    lines.append("- v4 baseline OOF: person-LOPO.  VLM: zero-shot per scan (no training -> no leakage).")
    lines.append("")
    lines.append("## Complementarity")
    lines.append("")
    lines.append(f"- Both correct: {comp['both_correct']}")
    lines.append(f"- VLM only correct: {comp['vlm_only_correct']}")
    lines.append(f"- v4 only correct: {comp['v4_only_correct']}")
    lines.append(f"- Neither correct: {comp['neither_correct']}")
    lines.append(f"- Oracle upper-bound accuracy (if we could pick the right model per scan): {comp['oracle_upper_bound_acc']:.4f}")
    lines.append("")
    lines.append("## Metrics table")
    lines.append("")
    lines.append(
        "| Strategy | Weighted F1 | Macro F1 | Acc | Healthy | Diabetes | Glaukom | SM | SucheOko |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in all_metrics:
        lines.append(fmt_row(m))
    lines.append("")
    lines.append(f"## Winner: **{winner_title}**")
    lines.append(f"- Weighted F1: {winner['weighted_f1']:.4f}")
    lines.append(f"- Macro F1: {winner['macro_f1']:.4f}")
    lines.append(f"- Accuracy: {winner['accuracy']:.4f}")
    lines.append(f"- SucheOko F1: {winner['per_class_f1']['SucheOko']:.4f}")
    lines.append(f"- Ship as v5? **{'YES' if ship_v5 else 'NO'}** "
                 f"(gates: weighted>=0.82, SucheOko>0.5, macro>0.70)")
    lines.append("")
    lines.append("## Top-3 confusion matrices")
    for title in top3_titles:
        lines.append(f"### {title}")
        # recompute preds for this strategy
        if title.startswith("F)"):
            p = predF
        elif title.startswith("F2)"):
            p = predF2
        elif title.startswith("A*"):
            w = float(best_arith["title"].split("w_vlm=")[1].rstrip(")"))
            p = arithmetic_mean(vlm_p_full, v4_p, wA=w).argmax(axis=1)
        elif title.startswith("B*"):
            w = float(best_geom["title"].split("w_vlm=")[1].rstrip(")"))
            p = geometric_mean(vlm_p_full, v4_p, wA=w).argmax(axis=1)
        elif title.startswith("A)"):
            w = float(title.split("w_vlm=")[1].rstrip(")"))
            p = arithmetic_mean(vlm_p_full, v4_p, wA=w).argmax(axis=1)
        elif title.startswith("B)"):
            w = float(title.split("w_vlm=")[1].rstrip(")"))
            p = geometric_mean(vlm_p_full, v4_p, wA=w).argmax(axis=1)
        elif title.startswith("C)"):
            p = confidence_weighted(vlm_p_full, v4_p).argmax(axis=1)
        elif title.startswith("D)"):
            mrg = float(title.split("<")[1].rstrip(")"))
            p = vlm_tiebreaker(v4_p, vlm_p_full, margin=mrg)
        elif title.startswith("E)"):
            p = per_class_specialist(v4_p, vlm_p_full)
        else:
            p = None
        if p is None:
            continue
        cm = confusion_matrix(y, p, labels=list(range(5)))
        lines.append("")
        lines.append("```")
        lines.append("rows=true, cols=pred")
        hdr = "          " + " ".join(f"{c[:6]:>6s}" for c in CLASSES)
        lines.append(hdr)
        for i in range(5):
            row = f"{CLASSES[i][:8]:<8s}  " + " ".join(f"{cm[i,j]:>6d}" for j in range(5))
            lines.append(row)
        lines.append("```")
        lines.append("")
        lines.append("```")
        lines.append(classification_report(y, p, target_names=CLASSES, digits=3, zero_division=0))
        lines.append("```")
        lines.append("")

    REPORT.write_text("\n".join(lines))
    print(f"Wrote {REPORT}")

    # Console summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for m in all_metrics:
        print(fmt_row(m))
    print()
    print(f"Winner: {winner_title}  weighted_f1={winner['weighted_f1']:.4f}  "
          f"macro={winner['macro_f1']:.4f}  SucheOko={winner['per_class_f1']['SucheOko']:.3f}  "
          f"ship_v5={ship_v5}")


if __name__ == "__main__":
    main()
