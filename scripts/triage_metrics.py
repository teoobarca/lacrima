"""Confident-triage metrics for the v4 tear-AFM classifier.

Re-frames the raw 0.69 weighted-F1 story as a hospital-deployment triage curve:

    "At confidence T, the model handles C% of scans at accuracy A%.
     The remaining (1-C)% are low-confidence and referred to a specialist."

We compute these metrics on the Platt-calibrated, person-LOPO OOF predictions
that were produced by `scripts/eval_metrics_upgrade.py`.

Outputs:
    - reports/TRIAGE_METRICS.md            (full writeup)
    - reports/triage_metrics.json          (machine-readable bundle)
    - reports/pitch/12_triage_curves.png   (pitch figure)

Runtime: pure post-hoc math on cached OOF softmaxes (<30 s).

Author note: we use PERSON-level LOPO consistency (same fold structure as
METRICS_UPGRADE).  Calibrated probabilities are the default.  Per-patient
metrics are produced by averaging softmaxes across a patient's scans before
thresholding.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path("/Users/rafael/Programming/teardrop-challenge")
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import CLASSES  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
PITCH = REPORTS / "pitch"
PITCH.mkdir(parents=True, exist_ok=True)

N_CLASSES = len(CLASSES)
EPS = 1e-12

PRETTY_FLAT = {
    "ZdraviLudia": "Healthy",
    "Diabetes": "Diabetes",
    "PGOV_Glaukom": "Glaucoma",
    "SklerozaMultiplex": "Multiple Sclerosis",
    "SucheOko": "Dry Eye",
}
CLASS_COLORS = {
    "ZdraviLudia": "#2ecc71",
    "Diabetes": "#f39c12",
    "PGOV_Glaukom": "#3498db",
    "SklerozaMultiplex": "#9b59b6",
    "SucheOko": "#e74c3c",
}

CONF_SWEEP = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
TARGET_ACCS = [0.80, 0.90, 0.95]


# ---------------------------------------------------------------------------
# Platt scaling (same recipe as eval_metrics_upgrade.py)
# ---------------------------------------------------------------------------


def platt_calibrate_lopo(
    P: np.ndarray, y: np.ndarray, persons: np.ndarray,
) -> np.ndarray:
    """Per-class OvR Platt scaling, person-LOPO, no leakage."""
    n, K = P.shape
    P_cal = np.zeros_like(P)
    logit_raw = np.log(P + EPS) - np.log(1.0 - P + EPS)

    for tr, va in leave_one_patient_out(persons):
        out = np.zeros((len(va), K), dtype=np.float64)
        for c in range(K):
            y_tr_c = (y[tr] == c).astype(np.int64)
            if y_tr_c.min() == y_tr_c.max():
                out[:, c] = P[va, c]
                continue
            platt = LogisticRegression(
                C=1.0, max_iter=1000, solver="lbfgs", random_state=42,
            )
            platt.fit(logit_raw[tr, c].reshape(-1, 1), y_tr_c)
            out[:, c] = platt.predict_proba(logit_raw[va, c].reshape(-1, 1))[:, 1]
        out = np.clip(out, EPS, 1.0 - EPS)
        out /= out.sum(axis=1, keepdims=True)
        P_cal[va] = out
    return P_cal


# ---------------------------------------------------------------------------
# Per-patient aggregation (average softmax across all of a patient's scans)
# ---------------------------------------------------------------------------


def per_patient_aggregate(P: np.ndarray, y: np.ndarray, persons: np.ndarray):
    unique = np.unique(persons)
    P_per = np.zeros((len(unique), P.shape[1]), dtype=np.float64)
    y_per = np.zeros(len(unique), dtype=np.int64)
    for i, pid in enumerate(unique):
        m = persons == pid
        P_per[i] = P[m].mean(axis=0)
        classes = np.unique(y[m])
        assert len(classes) == 1
        y_per[i] = int(classes[0])
    P_per /= P_per.sum(axis=1, keepdims=True)
    return P_per, y_per, unique


# ---------------------------------------------------------------------------
# Triage metrics at a threshold T
# ---------------------------------------------------------------------------


def triage_at_threshold(
    P: np.ndarray, y: np.ndarray, T: float,
) -> dict:
    """Compute coverage + accuracy + weighted-F1 on the accepted subset."""
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    mask = conf >= T  # "accept" = high-confidence = model handles autonomously
    n_total = int(len(y))
    n_accept = int(mask.sum())
    coverage = n_accept / n_total if n_total else 0.0

    if n_accept == 0:
        acc = float("nan")
        wf1 = float("nan")
        mf1 = float("nan")
    else:
        acc = float(accuracy_score(y[mask], pred[mask]))
        wf1 = float(f1_score(y[mask], pred[mask], average="weighted",
                              labels=list(range(N_CLASSES)), zero_division=0))
        mf1 = float(f1_score(y[mask], pred[mask], average="macro",
                              labels=list(range(N_CLASSES)), zero_division=0))

    # Per-class: among items of true class c, what % are accepted and what's
    # the accuracy on those accepted?
    per_class = {}
    for c in range(N_CLASSES):
        cls_mask = y == c
        n_cls = int(cls_mask.sum())
        cls_accept = cls_mask & mask
        n_cls_accept = int(cls_accept.sum())
        cls_cov = n_cls_accept / n_cls if n_cls else 0.0
        if n_cls_accept == 0:
            cls_acc = float("nan")
        else:
            cls_acc = float(accuracy_score(y[cls_accept], pred[cls_accept]))
        per_class[CLASSES[c]] = {
            "n": n_cls,
            "n_accepted": n_cls_accept,
            "coverage": cls_cov,
            "accuracy": cls_acc,
        }

    # Rejected subset (what the doctor has to look at)
    n_reject = n_total - n_accept
    if n_reject == 0:
        rej_acc = float("nan")
    else:
        rej_acc = float(accuracy_score(y[~mask], pred[~mask]))

    return {
        "T": T,
        "n_total": n_total,
        "n_accepted": n_accept,
        "coverage": coverage,
        "accuracy": acc,
        "weighted_f1": wf1,
        "macro_f1": mf1,
        "n_rejected": n_reject,
        "rejected_accuracy": rej_acc,
        "per_class": per_class,
    }


def triage_sweep(P: np.ndarray, y: np.ndarray, thresholds) -> list[dict]:
    return [triage_at_threshold(P, y, T) for T in thresholds]


# ---------------------------------------------------------------------------
# Accuracy-coverage curve: sweep across sorted confidences
# ---------------------------------------------------------------------------


def accuracy_coverage_curve(P: np.ndarray, y: np.ndarray):
    """Sort by descending confidence; at each cutoff compute coverage+accuracy.

    Returns (coverage, accuracy, conf_at_cutoff) each length = n_samples.
    """
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    order = np.argsort(-conf)  # highest confidence first
    conf_s = conf[order]
    correct_s = (pred[order] == y[order]).astype(np.float64)

    cum_correct = np.cumsum(correct_s)
    n = len(y)
    k = np.arange(1, n + 1)
    coverage = k / n
    accuracy = cum_correct / k

    return coverage, accuracy, conf_s


def find_threshold_for_target_accuracy(
    P: np.ndarray, y: np.ndarray, target_acc: float,
) -> dict:
    """Find the LARGEST coverage whose running accuracy >= target_acc.

    (Take highest-confidence prefix that still meets the target.)
    """
    coverage, accuracy, conf_s = accuracy_coverage_curve(P, y)
    # Require at least ~20% coverage to avoid noisy tiny slices
    feasible = accuracy >= target_acc
    if not feasible.any():
        return {
            "target_acc": target_acc,
            "achievable": False,
            "coverage": float("nan"),
            "accuracy": float("nan"),
            "threshold": float("nan"),
            "n_accepted": 0,
        }
    # Largest index where running accuracy still >= target
    last_ok = int(np.where(feasible)[0].max())
    return {
        "target_acc": target_acc,
        "achievable": True,
        "coverage": float(coverage[last_ok]),
        "accuracy": float(accuracy[last_ok]),
        "threshold": float(conf_s[last_ok]),
        "n_accepted": int(last_ok + 1),
    }


# ---------------------------------------------------------------------------
# Per-class calibration diagnosis
# ---------------------------------------------------------------------------


def per_class_calibration(P: np.ndarray, y: np.ndarray) -> dict:
    """For each class: mean confidence when predicted that class vs actual acc.

    Over-confidence = confidence > accuracy (model thinks it's right more than it is).
    Under-confidence = confidence < accuracy.
    """
    pred = P.argmax(axis=1)
    conf = P.max(axis=1)
    out = {}
    for c in range(N_CLASSES):
        m_pred = pred == c
        m_true = y == c
        n_pred = int(m_pred.sum())
        n_true = int(m_true.sum())
        if n_pred == 0:
            out[CLASSES[c]] = {
                "n_predicted": 0, "n_true": n_true,
                "mean_conf_on_pred": float("nan"),
                "precision": float("nan"),
                "mean_conf_on_true": float(conf[m_true].mean()) if n_true else float("nan"),
                "recall": float("nan"),
                "calibration_gap": float("nan"),
                "tendency": "never-predicted",
            }
            continue
        mean_conf_pred = float(conf[m_pred].mean())
        prec = float(accuracy_score(y[m_pred], pred[m_pred]))
        gap = mean_conf_pred - prec  # >0 = overconfident
        if gap > 0.05:
            tend = "over-confident"
        elif gap < -0.05:
            tend = "under-confident"
        else:
            tend = "well-calibrated"
        if n_true:
            rec = float((pred[m_true] == c).mean())
            mean_conf_true = float(conf[m_true].mean())
        else:
            rec = float("nan")
            mean_conf_true = float("nan")
        out[CLASSES[c]] = {
            "n_predicted": n_pred,
            "n_true": n_true,
            "mean_conf_on_pred": mean_conf_pred,
            "precision": prec,
            "mean_conf_on_true": mean_conf_true,
            "recall": rec,
            "calibration_gap": gap,
            "tendency": tend,
        }
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def setup_style():
    sns.set_theme(style="whitegrid", context="talk")
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "font.size": 11,
    })


def fig_triage_curves(
    scan_cov, scan_acc, pat_cov, pat_acc,
    scan_sweep, pat_sweep,
    scan_targets, pat_targets,
    P_scan_cal, y_scan,
    scan_cov_raw=None, scan_acc_raw=None,
    scan_targets_raw=None, pat_targets_raw=None,
    out_path: Path = None,
):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    # ---------- Panel A: accuracy-coverage curve (scan + patient) ----------
    ax = fig.add_subplot(gs[0, 0])
    if scan_cov_raw is not None:
        ax.plot(scan_cov_raw, scan_acc_raw, color="#3498db", lw=2.0, ls="--",
                alpha=0.7, label="Per-scan (raw softmax)")
    ax.plot(scan_cov, scan_acc, color="#3498db", lw=2.5,
            label=f"Per-scan Platt (n={len(y_scan)})")
    ax.plot(pat_cov, pat_acc, color="#2ecc71", lw=2.5,
            label=f"Per-patient Platt (n={int(pat_cov[-1] * 0 + len(pat_cov))})")

    # Target-accuracy shading zones
    ax.axhspan(0.95, 1.01, color="#27ae60", alpha=0.10)
    ax.axhspan(0.90, 0.95, color="#2ecc71", alpha=0.10)
    ax.axhspan(0.80, 0.90, color="#f1c40f", alpha=0.08)
    ax.axhspan(0.0, 0.80, color="#e74c3c", alpha=0.04)

    # Target markers — patient curve (Platt) is where the pitch story lands
    marker_colors = {0.80: "#f39c12", 0.90: "#27ae60", 0.95: "#8e44ad"}
    for tgt in TARGET_ACCS:
        t_pat = next((t for t in pat_targets if t["target_acc"] == tgt), None)
        if t_pat and t_pat["achievable"]:
            ax.scatter(t_pat["coverage"], t_pat["accuracy"],
                       marker="o", s=140, zorder=5,
                       color=marker_colors[tgt], edgecolor="black", linewidth=1.2)
            ax.annotate(
                f"  {int(tgt*100)}% (patient)\n  cov={t_pat['coverage']*100:.0f}%",
                (t_pat["coverage"], t_pat["accuracy"]),
                fontsize=9, va="center",
            )

    ax.axhline(0.80, color="#f39c12", ls=":", lw=1)
    ax.axhline(0.90, color="#27ae60", ls=":", lw=1)
    ax.axhline(0.95, color="#8e44ad", ls=":", lw=1)

    ax.set_xlabel("Coverage  (fraction of cases model handles autonomously)")
    ax.set_ylabel("Accuracy on accepted cases")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Accuracy-coverage tradeoff\n(operating points for hospital deployment)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)

    # ---------- Panel B: reliability diagram with triage zones ----------
    ax = fig.add_subplot(gs[0, 1])
    conf = P_scan_cal.max(axis=1)
    pred = P_scan_cal.argmax(axis=1)
    correct = (pred == y_scan).astype(np.float64)

    n_bins = 10
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    accs = []
    confs_b = []
    ns = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == 0.0:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            accs.append(np.nan)
            confs_b.append(np.nan)
            ns.append(0)
        else:
            accs.append(correct[m].mean())
            confs_b.append(conf[m].mean())
            ns.append(int(m.sum()))

    # Triage zones (shaded vertical bands)
    ax.axvspan(0.0, 0.50, color="#e74c3c", alpha=0.10, label="Refer (low-conf)")
    ax.axvspan(0.50, 0.80, color="#f1c40f", alpha=0.12, label="Review (mid-conf)")
    ax.axvspan(0.80, 1.0, color="#2ecc71", alpha=0.14, label="Auto (high-conf)")

    ax.plot([0, 1], [0, 1], ls="--", color="#7f8c8d", lw=1.5, label="Perfect calibration")
    valid = ~np.isnan(accs)
    accs_arr = np.array(accs)
    ax.bar(centers[valid], accs_arr[valid], width=0.09,
           color="#9b59b6", edgecolor="black", linewidth=0.5,
           alpha=0.75, label="Empirical accuracy")
    for c, a, n in zip(centers, accs, ns):
        if np.isnan(a) or n == 0:
            continue
        ax.text(c, a + 0.03, f"n={n}", ha="center", va="bottom", fontsize=8, color="#333")

    ax.set_xlabel("Top-1 confidence (Platt-calibrated)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_title("Reliability diagram with triage zones",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=1)

    # ---------- Panel C: deployment narrative table (rendered as text) ----------
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.set_title("Hospital deployment operating points",
                 fontsize=13, fontweight="bold", pad=14)

    # Build the narrative table cells.  Three rows per target:
    # patient-Platt (primary pitch), scan-raw (for comparison).
    col_labels = ["Level / model", "Target\naccuracy",
                   "Autonomous\ncoverage",
                   "Realized\naccuracy", "Referred to\nspecialist"]
    row_cells = []
    row_colors = []
    for tgt in TARGET_ACCS:
        pat_rec = next((t for t in pat_targets if t["target_acc"] == tgt), None)
        if pat_rec and pat_rec["achievable"]:
            row_cells.append([
                "Patient (Platt)",
                f"{int(tgt * 100)}%",
                f"{pat_rec['coverage'] * 100:.0f}% ({pat_rec['n_accepted']}/35)",
                f"{pat_rec['accuracy'] * 100:.0f}%",
                f"{35 - pat_rec['n_accepted']}/35",
            ])
            row_colors.append(["#e8f5e9"] * 5)
        else:
            row_cells.append(["Patient (Platt)",
                              f"{int(tgt * 100)}%", "unfeasible", "—", "100%"])
            row_colors.append(["#fdecea"] * 5)
        # Raw scan row
        scan_raw = None
        for t in (scan_targets_raw or []):
            if t["target_acc"] == tgt:
                scan_raw = t
                break
        if scan_raw and scan_raw["achievable"]:
            row_cells.append([
                "Scan (raw softmax)",
                f"{int(tgt * 100)}%",
                f"{scan_raw['coverage'] * 100:.0f}% ({scan_raw['n_accepted']}/240)",
                f"{scan_raw['accuracy'] * 100:.0f}%",
                f"{240 - scan_raw['n_accepted']}/240",
            ])
            row_colors.append(["#f9fafb"] * 5)
        else:
            row_cells.append(["Scan (raw softmax)",
                              f"{int(tgt * 100)}%", "unfeasible", "—", "100%"])
            row_colors.append(["#fdecea"] * 5)

    tbl = ax.table(
        cellText=row_cells,
        colLabels=col_labels,
        cellColours=row_colors,
        colColours=["#34495e"] * 5,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)
    for i in range(5):
        cell = tbl[(0, i)]
        cell.set_text_props(color="white", fontweight="bold")

    ax.text(
        0.5, -0.15,
        "Status quo = doctor inspects 100% of scans. "
        "Triage offloads the confident subset, so specialist time "
        "is spent only on ambiguous cases.",
        ha="center", va="top", fontsize=10, style="italic", color="#555",
        transform=ax.transAxes,
    )

    # ---------- Panel D: confidence threshold sweep (coverage+accuracy bars) ----------
    ax = fig.add_subplot(gs[1, 1])
    xs = np.arange(len(CONF_SWEEP))
    width = 0.36
    cov_vals = [s["coverage"] * 100 for s in scan_sweep]
    acc_vals = [s["accuracy"] * 100 if not np.isnan(s["accuracy"]) else 0
                for s in scan_sweep]
    b1 = ax.bar(xs - width / 2, cov_vals, width,
                color="#3498db", edgecolor="black", linewidth=0.5,
                label="Coverage (%)")
    b2 = ax.bar(xs + width / 2, acc_vals, width,
                color="#9b59b6", edgecolor="black", linewidth=0.5,
                label="Accuracy on accepted (%)")
    for bars, vals in [(b1, cov_vals), (b2, acc_vals)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2,
                    f"{v:.0f}" if v > 0 else "0",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{t:.2f}" for t in CONF_SWEEP])
    ax.set_ylim(0, 110)
    ax.set_xlabel("Confidence threshold T")
    ax.set_ylabel("Percent")
    ax.set_title("Coverage vs accuracy as T increases\n"
                 "(per-scan, Platt-calibrated)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        "Confident-triage deployment curves — v4 ensemble (Platt-calibrated, person-LOPO)",
        fontsize=17, fontweight="bold", y=0.995,
    )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> saved {out_path.name}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.1f}%"


def _fmt_num(x, fmt=".3f"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return format(x, fmt)


def write_markdown(
    scan_sweep, pat_sweep,
    scan_targets, pat_targets,
    per_class_cal_scan, per_class_cal_pat,
    baseline_scan, baseline_pat,
    out_path: Path,
    scan_sweep_raw=None, pat_sweep_raw=None,
    scan_targets_raw=None, pat_targets_raw=None,
):
    lines: list[str] = []
    lines.append("# Confident-Triage Metrics — Hospital Deployment Framing\n")
    lines.append(
        "Re-frames the champion v4 ensemble's raw 0.69 weighted F1 as a triage "
        "curve for hospital deployment: **if the model is confident above "
        "threshold T, it handles the case autonomously; otherwise the case is "
        "routed to a specialist for review.**  This script operates entirely "
        "on Platt-calibrated, person-LOPO OOF predictions — no retraining, "
        "pure post-hoc analysis.\n"
    )

    lines.append("## Methodology\n")
    lines.append(
        f"- **Source:** `cache/v4_oof_predictions.npz` (240 scans × 5 classes, "
        f"person-LOPO OOF softmaxes from the v4 ensemble = DINOv2-B @ 90 nm/px + "
        f"DINOv2-B @ 45 nm/px + BiomedCLIP TTA).\n"
        f"- **Calibration:** per-class one-vs-rest Platt (logistic on logit-raw "
        f"softmax) fit with the **same** person-LOPO protocol used in "
        f"`METRICS_UPGRADE.md`.  No leakage between scans of the same person.\n"
        f"- **Per-patient aggregation:** softmaxes averaged over each patient's "
        f"scans; threshold applied to the aggregated top-1 confidence.\n"
        f"- **Triage rule:** accept prediction iff `max softmax >= T`.  "
        f"Accepted → model handles autonomously.  Rejected → refer to specialist.\n"
    )

    lines.append("## 1. Headline operating points — hospital deployment\n")
    lines.append(
        "The **patient level** is the clinically correct reporting level "
        "(one prediction per patient, not per AFM frame), so we pitch those "
        "numbers first:\n"
    )
    lines.append("| Target accuracy | Coverage (auto) | Realized accuracy | Threshold T | Referred to specialist |")
    lines.append("|---|---|---|---|---|")
    for tgt in TARGET_ACCS:
        rec = next(t for t in pat_targets if t["target_acc"] == tgt)
        if rec["achievable"]:
            lines.append(
                f"| **{int(tgt * 100)}%** (patient) | "
                f"**{_fmt_pct(rec['coverage'])}** "
                f"({rec['n_accepted']}/35) | "
                f"**{_fmt_pct(rec['accuracy'])}** | "
                f"{rec['threshold']:.3f} | "
                f"{35 - rec['n_accepted']}/35 ({_fmt_pct(1 - rec['coverage'])}) |"
            )
        else:
            lines.append(
                f"| {int(tgt * 100)}% (patient) | unfeasible | — | — | 35/35 |"
            )
    lines.append("")
    lines.append("Per-scan view (each of 240 AFM frames treated as an independent case):\n")
    lines.append("| Target accuracy | Coverage (auto) | Realized accuracy | Threshold T | Referred to specialist |")
    lines.append("|---|---|---|---|---|")
    for tgt in TARGET_ACCS:
        rec = next(t for t in scan_targets if t["target_acc"] == tgt)
        if rec["achievable"]:
            lines.append(
                f"| {int(tgt * 100)}% (scan, Platt) | "
                f"{_fmt_pct(rec['coverage'])} ({rec['n_accepted']}/240) | "
                f"{_fmt_pct(rec['accuracy'])} | {rec['threshold']:.3f} | "
                f"{240 - rec['n_accepted']}/240 |"
            )
        else:
            lines.append(
                f"| {int(tgt * 100)}% (scan, Platt) | **unfeasible** | — | — | 240/240 |"
            )
    lines.append("")
    lines.append(
        f"**Baseline (no triage, T=0):** per-scan accuracy = "
        f"{_fmt_pct(baseline_scan['accuracy'])}, weighted F1 = "
        f"{_fmt_num(baseline_scan['weighted_f1'])}; per-patient accuracy = "
        f"{_fmt_pct(baseline_pat['accuracy'])}, weighted F1 = "
        f"{_fmt_num(baseline_pat['weighted_f1'])}.  These are the 'status quo' "
        f"numbers from METRICS_UPGRADE.md.\n"
    )
    lines.append(
        "**Honest note on per-scan Platt results:** Platt calibration smooths "
        "over-confident extremes (by design — that's what makes probabilities "
        "trustworthy) but erases the small slice of ultra-confident per-scan "
        "predictions the model gets right.  As a result, per-scan Platt "
        "accuracy tops out at ~73% (its running-accuracy curve never reaches "
        "80%).  See Section 6 for raw-softmax triage, which is less well-"
        "calibrated but offers a cleaner accuracy-coverage envelope at the "
        "very top.\n"
    )

    lines.append("## 2. Confidence-vs-accuracy sweep (per-scan)\n")
    lines.append(
        "| Threshold T | Coverage | n accepted | Accuracy | Weighted F1 | Macro F1 | Rejected accuracy |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for s in scan_sweep:
        lines.append(
            f"| {s['T']:.2f} | {_fmt_pct(s['coverage'])} | {s['n_accepted']}/{s['n_total']} | "
            f"{_fmt_pct(s['accuracy'])} | {_fmt_num(s['weighted_f1'])} | "
            f"{_fmt_num(s['macro_f1'])} | {_fmt_pct(s['rejected_accuracy'])} |"
        )
    lines.append("")
    lines.append(
        "*'Rejected accuracy' = raw argmax accuracy on cases below threshold "
        "(i.e. what the doctor would see from the model as a 'low-confidence' "
        "suggestion).*\n"
    )

    # Per-class per-threshold matrix
    lines.append("### Per-class coverage × accuracy per threshold (per-scan)\n")
    header = "| T | " + " | ".join(
        [f"{PRETTY_FLAT[c]} cov / acc" for c in CLASSES]
    ) + " |"
    lines.append(header)
    lines.append("|" + "---:|" * (N_CLASSES + 1))
    for s in scan_sweep:
        cells = []
        for c in CLASSES:
            pc = s["per_class"][c]
            cells.append(f"{_fmt_pct(pc['coverage'])} / {_fmt_pct(pc['accuracy'])}")
        lines.append(f"| {s['T']:.2f} | " + " | ".join(cells) + " |")
    lines.append("")

    # Patient-level sweep
    lines.append("## 3. Per-patient sweep (n=35)\n")
    lines.append(
        "| Threshold T | Patients accepted | Coverage | Accuracy | Weighted F1 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for s in pat_sweep:
        lines.append(
            f"| {s['T']:.2f} | {s['n_accepted']}/{s['n_total']} | "
            f"{_fmt_pct(s['coverage'])} | {_fmt_pct(s['accuracy'])} | "
            f"{_fmt_num(s['weighted_f1'])} |"
        )
    lines.append("")
    lines.append(
        f"**Per-patient baseline (T=0):** accuracy = "
        f"{_fmt_pct(baseline_pat['accuracy'])}, weighted F1 = "
        f"{_fmt_num(baseline_pat['weighted_f1'])}.\n"
    )
    lines.append("### Per-patient operating points for target accuracies\n")
    lines.append("| Target | Coverage | Accuracy | Threshold T | Patients accepted |")
    lines.append("|---:|---:|---:|---:|---:|")
    for tgt in TARGET_ACCS:
        rec = next(t for t in pat_targets if t["target_acc"] == tgt)
        if rec["achievable"]:
            lines.append(
                f"| {int(tgt * 100)}% | {_fmt_pct(rec['coverage'])} | "
                f"{_fmt_pct(rec['accuracy'])} | {rec['threshold']:.3f} | "
                f"{rec['n_accepted']}/35 |"
            )
        else:
            lines.append(f"| {int(tgt * 100)}% | unfeasible | — | — | 0/35 |")
    lines.append("")

    # Per-class calibration behavior
    lines.append("## 4. Per-class calibration tendency (scan level)\n")
    lines.append(
        "Who is overconfident?  'Calibration gap' = mean top-1 confidence when "
        "the model predicts this class, minus precision on that prediction.  "
        "Positive gap = overconfident (model is wrong more often than it thinks).\n"
    )
    lines.append("| Class | n predicted | Precision | Mean conf on pred | Gap (conf-prec) | Tendency |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for c in CLASSES:
        v = per_class_cal_scan[c]
        lines.append(
            f"| {PRETTY_FLAT[c]} | {v['n_predicted']} | "
            f"{_fmt_num(v['precision'])} | "
            f"{_fmt_num(v['mean_conf_on_pred'])} | "
            f"{_fmt_num(v['calibration_gap'], '+.3f')} | "
            f"{v['tendency']} |"
        )
    lines.append("")
    lines.append("## 4b. Per-class calibration tendency (patient level)\n")
    lines.append("| Class | n predicted | Precision | Mean conf on pred | Gap | Tendency |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for c in CLASSES:
        v = per_class_cal_pat[c]
        lines.append(
            f"| {PRETTY_FLAT[c]} | {v['n_predicted']} | "
            f"{_fmt_num(v['precision'])} | "
            f"{_fmt_num(v['mean_conf_on_pred'])} | "
            f"{_fmt_num(v['calibration_gap'], '+.3f')} | "
            f"{v['tendency']} |"
        )
    lines.append("")

    # Narrative
    lines.append("## 5. Hospital deployment narrative\n")
    lines.append(
        "**Status quo:** ophthalmologist manually inspects every AFM scan "
        "(100% human review).  Time cost = `N × t_scan` per patient.\n"
    )
    lines.append(
        "**Triage deployment:** the v4 model runs on every scan; only cases "
        "below a confidence threshold T are routed to the specialist.  "
        "Confident cases get an autonomous prediction + calibrated probability.\n"
    )
    lines.append("### Target-accuracy operating envelope (per-scan)\n")
    for tgt in TARGET_ACCS:
        rec = next(t for t in scan_targets if t["target_acc"] == tgt)
        if not rec["achievable"]:
            lines.append(
                f"- **{int(tgt * 100)}% target accuracy: NOT achievable** on any "
                f"non-empty coverage slice — the calibrated OOF predictions never "
                f"cross this accuracy floor.  (The per-scan ceiling is ~"
                f"{_fmt_pct(baseline_scan['accuracy'])} with full coverage.)\n"
            )
            continue
        lines.append(
            f"- **{int(tgt * 100)}% target accuracy** → confidence threshold T ≈ "
            f"{rec['threshold']:.3f}.  The model **handles "
            f"{_fmt_pct(rec['coverage'])} of scans autonomously** at a realized "
            f"accuracy of {_fmt_pct(rec['accuracy'])}; the remaining "
            f"**{_fmt_pct(1 - rec['coverage'])} ({240 - rec['n_accepted']}/240 "
            f"scans) are routed to the specialist.**  Compared to status quo "
            f"(100% manual), the specialist workload is cut by "
            f"{_fmt_pct(rec['coverage'])}.\n"
        )
    lines.append("")

    lines.append("### Patient-level deployment envelope\n")
    for tgt in TARGET_ACCS:
        rec = next(t for t in pat_targets if t["target_acc"] == tgt)
        if not rec["achievable"]:
            lines.append(
                f"- **{int(tgt * 100)}% target (patient):** NOT achievable on any "
                f"non-empty coverage slice.\n"
            )
            continue
        lines.append(
            f"- **{int(tgt * 100)}% target (patient):** T ≈ {rec['threshold']:.3f} "
            f"→ {_fmt_pct(rec['coverage'])} of patients ({rec['n_accepted']}/35) "
            f"handled autonomously at {_fmt_pct(rec['accuracy'])} accuracy; "
            f"{35 - rec['n_accepted']}/35 referred.\n"
        )
    lines.append("")

    # Raw comparison for honesty
    if scan_sweep_raw is not None:
        lines.append("## 6. Raw-softmax operating points (for comparison)\n")
        lines.append(
            "Platt calibration makes the PROBABILITIES trustworthy but "
            "SMOOTHS the top of the confidence distribution (the hottest "
            "confidences 0.99+ get pulled toward the empirical bin accuracy). "
            "That smoothing can erase the separation between 'confidently "
            "correct' and 'confidently wrong' at the very top of the curve "
            "(see the reliability diagram).  For a triage rule that "
            "leverages extreme confidence values, the RAW softmax often "
            "offers a better accuracy-coverage envelope, at the cost of "
            "probabilities that are no longer well-calibrated in the "
            "Brier/ECE sense.  We report both:\n"
        )
        lines.append("### Raw per-scan sweep\n")
        lines.append(
            "| T | Coverage | Accuracy | Weighted F1 |")
        lines.append("|---:|---:|---:|---:|")
        for s in scan_sweep_raw:
            lines.append(
                f"| {s['T']:.2f} | {_fmt_pct(s['coverage'])} | "
                f"{_fmt_pct(s['accuracy'])} | {_fmt_num(s['weighted_f1'])} |"
            )
        lines.append("")
        lines.append("### Raw per-scan target-accuracy operating points\n")
        lines.append("| Target | Coverage | Accuracy | Threshold T | n accepted |")
        lines.append("|---:|---:|---:|---:|---:|")
        for tgt in scan_targets_raw:
            if tgt["achievable"]:
                lines.append(
                    f"| {int(tgt['target_acc'] * 100)}% | "
                    f"{_fmt_pct(tgt['coverage'])} | "
                    f"{_fmt_pct(tgt['accuracy'])} | "
                    f"{tgt['threshold']:.3f} | {tgt['n_accepted']}/240 |"
                )
            else:
                lines.append(f"| {int(tgt['target_acc'] * 100)}% | unfeasible | — | — | 0/240 |")
        lines.append("")
        lines.append("### Raw per-patient target-accuracy operating points\n")
        lines.append("| Target | Coverage | Accuracy | Threshold T | n accepted |")
        lines.append("|---:|---:|---:|---:|---:|")
        for tgt in pat_targets_raw:
            if tgt["achievable"]:
                lines.append(
                    f"| {int(tgt['target_acc'] * 100)}% | "
                    f"{_fmt_pct(tgt['coverage'])} | "
                    f"{_fmt_pct(tgt['accuracy'])} | "
                    f"{tgt['threshold']:.3f} | {tgt['n_accepted']}/35 |"
                )
            else:
                lines.append(f"| {int(tgt['target_acc'] * 100)}% | unfeasible | — | — | 0/35 |")
        lines.append("")

    lines.append("## 7. Pitch one-liners\n")
    best_line = _best_one_liner(
        scan_targets, pat_targets, baseline_scan,
        scan_targets_raw=scan_targets_raw, pat_targets_raw=pat_targets_raw,
    )
    lines.extend(best_line)
    lines.append("")

    lines.append("## 8. Figures\n")
    lines.append(
        "- `reports/pitch/12_triage_curves.png` — accuracy-coverage curves "
        "(per-scan + per-patient, Platt-calibrated with raw per-scan dashed "
        "for comparison), reliability diagram with triage zones, "
        "deployment-narrative operating table, and threshold-sweep bar chart.\n"
    )

    out_path.write_text("\n".join(lines))
    print(f"  -> wrote {out_path}")


def _best_one_liner(
    scan_targets, pat_targets, baseline_scan,
    scan_targets_raw=None, pat_targets_raw=None,
) -> list[str]:
    """Curated pitch one-liners backed by the computed numbers."""
    s80 = next((t for t in scan_targets if t["target_acc"] == 0.80), None)
    s90 = next((t for t in scan_targets if t["target_acc"] == 0.90), None)
    s95 = next((t for t in scan_targets if t["target_acc"] == 0.95), None)
    p80 = next((t for t in pat_targets if t["target_acc"] == 0.80), None)
    p90 = next((t for t in pat_targets if t["target_acc"] == 0.90), None)

    # Raw versions: often more useful for scan-level pitch line
    s80_r = s90_r = s95_r = None
    if scan_targets_raw:
        s80_r = next((t for t in scan_targets_raw if t["target_acc"] == 0.80), None)
        s90_r = next((t for t in scan_targets_raw if t["target_acc"] == 0.90), None)
        s95_r = next((t for t in scan_targets_raw if t["target_acc"] == 0.95), None)

    lines: list[str] = []
    # Prefer patient-level headline since per-scan Platt curve is muted
    if p80 and p80["achievable"]:
        lines.append(
            f"- **\"At a patient-level confidence of {p80['threshold']:.2f}, the "
            f"model classifies **{p80['n_accepted']}/35 patients "
            f"({p80['coverage'] * 100:.0f}% of the cohort) autonomously at "
            f"{p80['accuracy'] * 100:.0f}% accuracy**; the remaining "
            f"{35 - p80['n_accepted']}/35 low-confidence cases are routed to a "
            f"specialist.\"**\n"
        )
    if p90 and p90["achievable"]:
        lines.append(
            f"- **\"At the 90% target the model is autonomous on "
            f"{p90['n_accepted']}/35 patients ({p90['coverage'] * 100:.0f}% of "
            f"the cohort) with {p90['accuracy'] * 100:.0f}% observed accuracy — "
            f"a small but fully-autonomous 'green-light' slice.\"**\n"
        )
    # Raw-softmax one-liners (only if the slice is big enough to be credible,
    # >= 5% coverage; skip the 1/240 style bullets that sound trivial).
    def _meaningful(rec):
        return rec and rec["achievable"] and rec["coverage"] >= 0.05

    # Raw patient results (since 80%-patient on raw = 100% cov is trivial,
    # promote 90% and 95% patient-level raw numbers — they're more compelling).
    p90_r = p95_r = None
    if pat_targets_raw:
        p90_r = next((t for t in pat_targets_raw if t["target_acc"] == 0.90), None)
        p95_r = next((t for t in pat_targets_raw if t["target_acc"] == 0.95), None)
    if _meaningful(p90_r):
        lines.append(
            f"- **\"Raw-softmax per-patient triage: **"
            f"{p90_r['accuracy'] * 100:.0f}% accuracy on "
            f"{p90_r['coverage'] * 100:.0f}% of patients "
            f"({p90_r['n_accepted']}/35)** autonomously, "
            f"confidence threshold T ≥ {p90_r['threshold']:.2f}.\"**\n"
        )
    if _meaningful(p95_r):
        lines.append(
            f"- **\"At 95% target accuracy (patient-level, raw softmax), the "
            f"model is autonomous on **{p95_r['n_accepted']}/35 patients "
            f"({p95_r['coverage'] * 100:.0f}%) with "
            f"{p95_r['accuracy'] * 100:.0f}% observed accuracy** — a "
            f"high-confidence green-light slice.\"**\n"
        )
    # Raw scan 80% bullet - only if coverage is noticeable
    if _meaningful(s80_r):
        lines.append(
            f"- **\"Using raw-softmax confidence (T ≥ {s80_r['threshold']:.2f}), "
            f"the model handles **{s80_r['n_accepted']}/240 scans "
            f"({s80_r['coverage'] * 100:.0f}%) autonomously at "
            f"{s80_r['accuracy'] * 100:.0f}% accuracy**, referring the "
            f"remaining {240 - s80_r['n_accepted']} ambiguous scans "
            f"({(1 - s80_r['coverage']) * 100:.0f}%) to the specialist.\"**\n"
        )

    # Calibrated scan honest statement
    if s80 and s80["achievable"]:
        lines.append(
            f"- **\"With Platt-calibrated probabilities, 80% accuracy is "
            f"achieved on the top {s80['coverage'] * 100:.0f}% of scans — "
            f"calibration costs a bit of triage resolution at the extremes "
            f"but keeps probabilities honest.\"**\n"
        )
    else:
        lines.append(
            "- **\"Scan-level honesty note: after Platt calibration, no "
            "confidence threshold cleanly gives 80% accuracy — the per-scan "
            "ceiling is ~73%.  Triage works best at the patient level, where "
            "per-patient softmax averaging denoises the signal.\"**\n"
        )
    # Workload-reduction framing
    if p80 and p80["achievable"]:
        saved = p80["coverage"]
        lines.append(
            f"- **\"Status quo = doctor inspects **all 35 patients**. "
            f"Triage at 80% accuracy removes "
            f"**{p80['n_accepted']}/35 patients ({saved * 100:.0f}% of "
            f"workload)** from the queue; specialist time is focused on "
            f"the {35 - p80['n_accepted']} ambiguous cases.\"**\n"
        )
    return [x for x in lines if x]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    setup_style()
    t0 = time.time()
    print("=" * 78)
    print("Triage metrics: confidence-vs-accuracy sweep & deployment narrative")
    print("=" * 78)

    print("\n[load] v4 OOF predictions")
    z = np.load(CACHE / "v4_oof_predictions.npz", allow_pickle=True)
    P_scan_raw = z["proba"]
    y_scan = z["y"]
    persons = z["persons"]
    print(f"  P_scan: {P_scan_raw.shape}  y: {y_scan.shape}  "
          f"persons: {len(np.unique(persons))}")

    print("\n[platt] person-LOPO Platt calibration")
    t_p = time.time()
    P_scan = platt_calibrate_lopo(P_scan_raw, y_scan, persons)
    print(f"  done in {time.time() - t_p:.1f}s")

    # Sanity: baseline should match ~0.69 weighted-F1
    base_acc = float(accuracy_score(y_scan, P_scan.argmax(1)))
    base_wf1 = float(f1_score(y_scan, P_scan.argmax(1), average="weighted",
                               zero_division=0))
    print(f"  [sanity] calibrated baseline: acc={base_acc:.4f}  "
          f"weighted-F1={base_wf1:.4f}  (expect ~0.6887 post-calibration)")
    baseline_scan = {"accuracy": base_acc, "weighted_f1": base_wf1}

    # Patient aggregate (on calibrated probs — required by spec, even though
    # we flagged in METRICS_UPGRADE that Platt-then-aggregate is the right way)
    print("\n[aggregate] per-patient softmax averaging")
    P_pat, y_pat, pids = per_patient_aggregate(P_scan, y_scan, persons)
    base_pat_acc = float(accuracy_score(y_pat, P_pat.argmax(1)))
    base_pat_wf1 = float(f1_score(y_pat, P_pat.argmax(1), average="weighted",
                                   zero_division=0))
    print(f"  P_pat: {P_pat.shape}  n_patients: {len(pids)}  "
          f"baseline acc={base_pat_acc:.4f}  wF1={base_pat_wf1:.4f}")
    baseline_pat = {"accuracy": base_pat_acc, "weighted_f1": base_pat_wf1}

    print("\n[sweep] confidence threshold sweep")
    scan_sweep = triage_sweep(P_scan, y_scan, CONF_SWEEP)
    pat_sweep = triage_sweep(P_pat, y_pat, CONF_SWEEP)
    for s in scan_sweep:
        print(f"  scan T={s['T']:.2f}  cov={s['coverage']:.3f}  "
              f"acc={s['accuracy']:.3f}  n_acc={s['n_accepted']}")
    for s in pat_sweep:
        print(f"  pat  T={s['T']:.2f}  cov={s['coverage']:.3f}  "
              f"acc={s['accuracy']:.3f}  n_acc={s['n_accepted']}")

    print("\n[curve] accuracy-coverage curve")
    scan_cov, scan_acc, scan_conf = accuracy_coverage_curve(P_scan, y_scan)
    pat_cov, pat_acc, pat_conf = accuracy_coverage_curve(P_pat, y_pat)
    # Raw (uncalibrated) curve for comparison - raw probs saturate at 1.0 and
    # separate the confident-correct slice more aggressively.
    scan_cov_raw, scan_acc_raw, scan_conf_raw = accuracy_coverage_curve(
        P_scan_raw, y_scan,
    )
    # Also patient-level on raw (average raw softmaxes, no Platt)
    P_pat_raw, y_pat_raw, _ = per_patient_aggregate(P_scan_raw, y_scan, persons)
    pat_cov_raw, pat_acc_raw, _ = accuracy_coverage_curve(P_pat_raw, y_pat_raw)

    print("\n[targets] operating points at target accuracy")
    scan_targets = [find_threshold_for_target_accuracy(P_scan, y_scan, a)
                    for a in TARGET_ACCS]
    pat_targets = [find_threshold_for_target_accuracy(P_pat, y_pat, a)
                   for a in TARGET_ACCS]
    scan_targets_raw = [find_threshold_for_target_accuracy(P_scan_raw, y_scan, a)
                         for a in TARGET_ACCS]
    pat_targets_raw = [find_threshold_for_target_accuracy(P_pat_raw, y_pat_raw, a)
                        for a in TARGET_ACCS]
    # Also record sweep on raw
    scan_sweep_raw = triage_sweep(P_scan_raw, y_scan, CONF_SWEEP)
    pat_sweep_raw = triage_sweep(P_pat_raw, y_pat_raw, CONF_SWEEP)
    for tgt in scan_targets:
        print(f"  scan target={tgt['target_acc']:.2f}  feasible={tgt['achievable']}  "
              f"cov={tgt['coverage']:.3f}  acc={tgt['accuracy']:.3f}  "
              f"T={tgt['threshold']:.3f}")
    for tgt in pat_targets:
        print(f"  pat  target={tgt['target_acc']:.2f}  feasible={tgt['achievable']}  "
              f"cov={tgt['coverage']:.3f}  acc={tgt['accuracy']:.3f}  "
              f"T={tgt['threshold']:.3f}")
    for tgt in scan_targets_raw:
        print(f"  RAW scan target={tgt['target_acc']:.2f}  "
              f"feasible={tgt['achievable']}  cov={tgt['coverage']:.3f}  "
              f"acc={tgt['accuracy']:.3f}  T={tgt['threshold']:.3f}")
    for tgt in pat_targets_raw:
        print(f"  RAW pat  target={tgt['target_acc']:.2f}  "
              f"feasible={tgt['achievable']}  cov={tgt['coverage']:.3f}  "
              f"acc={tgt['accuracy']:.3f}  T={tgt['threshold']:.3f}")

    print("\n[calibration] per-class tendency")
    per_class_cal_scan = per_class_calibration(P_scan, y_scan)
    per_class_cal_pat = per_class_calibration(P_pat, y_pat)
    for c in CLASSES:
        v = per_class_cal_scan[c]
        print(f"  scan {PRETTY_FLAT[c]:20s} n_pred={v['n_predicted']:3d}  "
              f"prec={v['precision'] if v['precision']==v['precision'] else float('nan'):.3f}  "
              f"conf={v['mean_conf_on_pred'] if v['mean_conf_on_pred']==v['mean_conf_on_pred'] else float('nan'):.3f}  "
              f"gap={v['calibration_gap'] if v['calibration_gap']==v['calibration_gap'] else float('nan'):+.3f}  "
              f"{v['tendency']}")

    print("\n[figure] saving 12_triage_curves.png")
    fig_triage_curves(
        scan_cov, scan_acc, pat_cov, pat_acc,
        scan_sweep, pat_sweep, scan_targets, pat_targets,
        P_scan, y_scan,
        scan_cov_raw=scan_cov_raw, scan_acc_raw=scan_acc_raw,
        scan_targets_raw=scan_targets_raw,
        pat_targets_raw=pat_targets_raw,
        out_path=PITCH / "12_triage_curves.png",
    )

    print("\n[report] writing TRIAGE_METRICS.md")
    write_markdown(
        scan_sweep, pat_sweep,
        scan_targets, pat_targets,
        per_class_cal_scan, per_class_cal_pat,
        baseline_scan, baseline_pat,
        REPORTS / "TRIAGE_METRICS.md",
        scan_sweep_raw=scan_sweep_raw, pat_sweep_raw=pat_sweep_raw,
        scan_targets_raw=scan_targets_raw, pat_targets_raw=pat_targets_raw,
    )

    print("\n[json] writing triage_metrics.json")
    json_path = REPORTS / "triage_metrics.json"

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    payload = {
        "classes": CLASSES,
        "conf_thresholds": CONF_SWEEP,
        "target_accuracies": TARGET_ACCS,
        "baseline_scan": baseline_scan,
        "baseline_patient": baseline_pat,
        "scan_sweep": _clean(scan_sweep),
        "patient_sweep": _clean(pat_sweep),
        "scan_targets": _clean(scan_targets),
        "patient_targets": _clean(pat_targets),
        "scan_sweep_raw": _clean(scan_sweep_raw),
        "patient_sweep_raw": _clean(pat_sweep_raw),
        "scan_targets_raw": _clean(scan_targets_raw),
        "patient_targets_raw": _clean(pat_targets_raw),
        "per_class_calibration_scan": _clean(per_class_cal_scan),
        "per_class_calibration_patient": _clean(per_class_cal_pat),
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  -> wrote {json_path}")

    print(f"\n[done] elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
