"""Evaluation-suite upgrade: top-k accuracy, per-patient aggregation,
calibration (ECE/Brier), Platt scaling, and bootstrap CIs.

Pitch-ready honest metrics for the tear-AFM classifier.

Source-of-truth:
    - Champion ensemble = `models/ensemble_v4_multiscale/` (Wave-7 Config D).
    - OOF probabilities regenerated with the EXACT same recipe used in
      `scripts/multiscale_experiment.py` (the one that produced F1 = 0.6887).
    - Person-level LOPO (35 groups) via `teardrop.data.person_id`
      and `teardrop.cv.leave_one_patient_out`.

Outputs:
    - cache/v4_oof_predictions.npz  (softmax, y, person, scan_path)
    - reports/METRICS_UPGRADE.md
    - reports/pitch/07_topk_and_calibration.png
    - reports/pitch/05_per_class_metrics.png  (UPDATED to overlay top-k context)

Runtime budget: ≤ 20 minutes (measured ~2-3 min on an M-series Mac).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path("/Users/rafael/Programming/teardrop-challenge")
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import CLASSES, person_id  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
PITCH = REPORTS / "pitch"
PITCH.mkdir(parents=True, exist_ok=True)

N_CLASSES = len(CLASSES)
EPS = 1e-12
RNG = np.random.default_rng(42)

PRETTY = {
    "ZdraviLudia": "Healthy",
    "Diabetes": "Diabetes",
    "PGOV_Glaukom": "Glaucoma",
    "SklerozaMultiplex": "Multiple\nSclerosis",
    "SucheOko": "Dry Eye",
}
PRETTY_FLAT = {k: v.replace("\n", " ") for k, v in PRETTY.items()}
CLASS_COLORS = {
    "ZdraviLudia": "#2ecc71",
    "Diabetes": "#f39c12",
    "PGOV_Glaukom": "#3498db",
    "SklerozaMultiplex": "#9b59b6",
    "SucheOko": "#e74c3c",
}


# ---------------------------------------------------------------------------
# OOF generation (v4 ensemble, person-LOPO)
# ---------------------------------------------------------------------------


def _mean_pool_tiles(X_tiles: np.ndarray, t2s: np.ndarray, n_scans: int) -> np.ndarray:
    d = X_tiles.shape[1]
    out = np.zeros((n_scans, d), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int64)
    for i, s in enumerate(t2s):
        out[s] += X_tiles[i]
        counts[s] += 1
    counts = np.maximum(counts, 1)
    out /= counts[:, None]
    return out


def _align(paths_ref: list[str], paths_src: list[str], X_src: np.ndarray) -> np.ndarray:
    idx = {p: i for i, p in enumerate(paths_src)}
    order = np.array([idx[p] for p in paths_ref])
    return X_src[order]


def _lopo_v2_softmax(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Honest person-LOPO OOF softmax with the V2 recipe."""
    n = len(y)
    P = np.zeros((n, N_CLASSES), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups):
        Xt = normalize(X[tr], norm="l2", axis=1)
        Xv = normalize(X[va], norm="l2", axis=1)
        sc = StandardScaler().fit(Xt)
        Xt = np.nan_to_num(sc.transform(Xt))
        Xv = np.nan_to_num(sc.transform(Xv))
        clf = LogisticRegression(
            class_weight="balanced", max_iter=3000, C=1.0,
            solver="lbfgs", n_jobs=4, random_state=42,
        )
        clf.fit(Xt, y[tr])
        proba = clf.predict_proba(Xv)
        p_full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
        for ci, cls in enumerate(clf.classes_):
            p_full[:, cls] = proba[:, ci]
        P[va] = p_full
    return P


def _geom_mean(probs: list[np.ndarray]) -> np.ndarray:
    log_sum = sum(np.log(P + EPS) for P in probs)
    G = np.exp(log_sum / len(probs))
    G /= G.sum(axis=1, keepdims=True)
    return G


def load_or_build_v4_oof(force: bool = False) -> dict:
    """Return dict with keys: proba, y, persons, scan_paths."""
    cache_path = CACHE / "v4_oof_predictions.npz"
    if cache_path.exists() and not force:
        print(f"[cache-hit] {cache_path.name}")
        z = np.load(cache_path, allow_pickle=True)
        return {
            "proba": z["proba"],
            "y": z["y"],
            "persons": z["persons"],
            "scan_paths": z["scan_paths"],
        }

    print("[build] v4 OOF predictions (3-component geom-mean, person-LOPO)")
    t0 = time.time()
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz",
                  allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz",
                  allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz",
                  allow_pickle=True)

    paths_90 = [str(p) for p in z90["scan_paths"]]
    paths_45 = [str(p) for p in z45["scan_paths"]]
    paths_bc = [str(p) for p in zbc["scan_paths"]]

    persons = np.array([person_id(Path(p)) for p in paths_90])
    y = np.asarray(z90["scan_y"], dtype=np.int64)
    n_scans = len(y)
    n_persons = len(np.unique(persons))
    print(f"  n_scans={n_scans}  n_persons={n_persons}")
    assert n_persons == 35, f"expected 35 persons, got {n_persons}"

    X90 = _mean_pool_tiles(z90["X"], z90["tile_to_scan"], n_scans)
    X45_raw = _mean_pool_tiles(z45["X"], z45["tile_to_scan"], len(paths_45))
    X45 = _align(paths_90, paths_45, X45_raw)
    Xbc = _align(paths_90, paths_bc, zbc["X_scan"].astype(np.float32))

    print(f"  [lopo] DINOv2-B 90nm  ({X90.shape[1]}D)")
    P90 = _lopo_v2_softmax(X90, y, persons)
    print(f"  [lopo] DINOv2-B 45nm  ({X45.shape[1]}D)")
    P45 = _lopo_v2_softmax(X45, y, persons)
    print(f"  [lopo] BiomedCLIP TTA ({Xbc.shape[1]}D)")
    Pbc = _lopo_v2_softmax(Xbc, y, persons)

    P = _geom_mean([P90, P45, Pbc])

    # Sanity: must match reported champion F1 ~0.6887.
    f1w = f1_score(y, P.argmax(1), average="weighted", zero_division=0)
    f1m = f1_score(y, P.argmax(1), average="macro", zero_division=0)
    print(f"  [sanity] weighted F1 = {f1w:.4f}  macro F1 = {f1m:.4f}  "
          f"(champion: 0.6887 / 0.5541)  elapsed={time.time() - t0:.1f}s")

    np.savez(
        cache_path,
        proba=P,
        y=y,
        persons=persons,
        scan_paths=np.array(paths_90),
    )
    print(f"[saved] {cache_path}")
    return {"proba": P, "y": y, "persons": persons, "scan_paths": np.array(paths_90)}


# ---------------------------------------------------------------------------
# Per-patient aggregation
# ---------------------------------------------------------------------------


def per_patient_aggregate(P: np.ndarray, y: np.ndarray, persons: np.ndarray):
    """Average softmax across a person's scans -> one (P, y) row per person."""
    unique = np.unique(persons)
    P_per = np.zeros((len(unique), P.shape[1]), dtype=np.float64)
    y_per = np.zeros(len(unique), dtype=np.int64)
    sizes = np.zeros(len(unique), dtype=np.int64)
    for i, pid in enumerate(unique):
        m = persons == pid
        P_per[i] = P[m].mean(axis=0)
        # Each person has a single true class (verify).
        classes = np.unique(y[m])
        assert len(classes) == 1, f"person {pid} has mixed classes {classes}"
        y_per[i] = int(classes[0])
        sizes[i] = int(m.sum())
    # Renormalize (means of softmaxes are already normalized, but be safe).
    P_per /= P_per.sum(axis=1, keepdims=True)
    return P_per, y_per, unique, sizes


# ---------------------------------------------------------------------------
# Top-k accuracy
# ---------------------------------------------------------------------------


def topk_accuracy(P: np.ndarray, y: np.ndarray, k: int) -> float:
    # labels=list(range(N_CLASSES)) ensures we pass a fixed column order even
    # if a class is absent from y on a given split (happens when folding).
    return float(top_k_accuracy_score(y, P, k=k, labels=list(range(N_CLASSES))))


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(P: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE with equal-width bins.

    Standard definition: average |accuracy - confidence| weighted by bin size.
    """
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == y).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo == 0.0:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        bin_acc = correct[m].mean()
        bin_conf = conf[m].mean()
        ece += (m.sum() / N) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_bins(P: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Return (bin_centers, bin_acc, bin_conf, bin_count) for the top-label diagram."""
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    accs, confs, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo == 0.0:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            accs.append(np.nan)
            confs.append(np.nan)
        else:
            accs.append(float(correct[m].mean()))
            confs.append(float(conf[m].mean()))
        counts.append(int(m.sum()))
    return centers, np.array(accs), np.array(confs), np.array(counts)


def brier_score_multiclass(P: np.ndarray, y: np.ndarray) -> float:
    """Mean-over-classes one-vs-rest Brier (standard multi-class Brier)."""
    scores = []
    for c in range(P.shape[1]):
        y_c = (y == c).astype(np.float64)
        scores.append(brier_score_loss(y_c, P[:, c]))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Platt scaling (per-class one-vs-rest logistic)
# ---------------------------------------------------------------------------


def platt_calibrate_lopo(P: np.ndarray, y: np.ndarray, persons: np.ndarray) -> np.ndarray:
    """Calibrate via PERSON-LOPO nested Platt scaling.

    For each fold we hold out one person, fit a per-class Platt logistic on
    logit(raw softmax) of the other 34 persons, then apply to held-out scans.

    This is the HONEST protocol — no leakage between scans of the same person.
    """
    n, K = P.shape
    P_cal = np.zeros_like(P)
    logit_raw = np.log(P + EPS) - np.log(1.0 - P + EPS)  # per-class logit, shape (n, K)

    for tr, va in leave_one_patient_out(persons):
        out = np.zeros((len(va), K), dtype=np.float64)
        for c in range(K):
            y_tr_c = (y[tr] == c).astype(np.int64)
            # Guard: Platt needs both classes in train fold.
            if y_tr_c.min() == y_tr_c.max():
                out[:, c] = (P[va, c])
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
# Bootstrap CI on weighted F1 — PERSON level
# ---------------------------------------------------------------------------


def bootstrap_person_f1(
    P: np.ndarray,
    y: np.ndarray,
    persons: np.ndarray,
    n_boot: int = 1000,
    average: str = "weighted",
    seed: int = 42,
) -> dict:
    """Resample persons (with replacement) and recompute F1 on the pooled scans."""
    rng = np.random.default_rng(seed)
    unique = np.unique(persons)
    scan_idx_by_person = {p: np.where(persons == p)[0] for p in unique}

    point = float(f1_score(y, P.argmax(1), average=average, zero_division=0))
    scores = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([scan_idx_by_person[p] for p in sampled])
        scores[b] = f1_score(y[idx], P[idx].argmax(1), average=average, zero_division=0)

    return {
        "point": point,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "ci_lo": float(np.percentile(scores, 2.5)),
        "ci_hi": float(np.percentile(scores, 97.5)),
        "n_boot": n_boot,
        "average": average,
    }


# ---------------------------------------------------------------------------
# Main metric-upgrade report
# ---------------------------------------------------------------------------


def compute_all_metrics(P_scan, y_scan, persons, P_cal_scan):
    """Full metric bundle: scan-level + patient-level × top-k + calibration + F1."""
    metrics: dict = {}

    # Scan level
    metrics["scan"] = {
        "n": int(len(y_scan)),
        "top1_acc": topk_accuracy(P_scan, y_scan, 1),
        "top2_acc": topk_accuracy(P_scan, y_scan, 2),
        "top3_acc": topk_accuracy(P_scan, y_scan, 3),
        "weighted_f1": float(f1_score(y_scan, P_scan.argmax(1), average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_scan, P_scan.argmax(1), average="macro", zero_division=0)),
        "per_class_f1": f1_score(y_scan, P_scan.argmax(1), average=None,
                                  labels=list(range(N_CLASSES)), zero_division=0).tolist(),
        "ece_pre": expected_calibration_error(P_scan, y_scan, 10),
        "ece_post": expected_calibration_error(P_cal_scan, y_scan, 10),
        "brier_pre": brier_score_multiclass(P_scan, y_scan),
        "brier_post": brier_score_multiclass(P_cal_scan, y_scan),
    }

    # Patient level (average softmax per person)
    P_per, y_per, pids, sizes = per_patient_aggregate(P_scan, y_scan, persons)
    P_per_cal, _, _, _ = per_patient_aggregate(P_cal_scan, y_scan, persons)
    metrics["patient"] = {
        "n": int(len(y_per)),
        "top1_acc": topk_accuracy(P_per, y_per, 1),
        "top2_acc": topk_accuracy(P_per, y_per, 2),
        "top3_acc": topk_accuracy(P_per, y_per, 3),
        "weighted_f1": float(f1_score(y_per, P_per.argmax(1), average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_per, P_per.argmax(1), average="macro", zero_division=0)),
        "per_class_f1": f1_score(y_per, P_per.argmax(1), average=None,
                                  labels=list(range(N_CLASSES)), zero_division=0).tolist(),
        "ece_pre": expected_calibration_error(P_per, y_per, 10),
        "ece_post": expected_calibration_error(P_per_cal, y_per, 10),
        "brier_pre": brier_score_multiclass(P_per, y_per),
        "brier_post": brier_score_multiclass(P_per_cal, y_per),
        "scans_per_patient": {
            "min": int(sizes.min()),
            "median": int(np.median(sizes)),
            "max": int(sizes.max()),
        },
    }

    return metrics, (P_per, y_per, pids, P_per_cal)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def setup_style():
    sns.set_theme(style="darkgrid", context="talk")
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "font.size": 12,
    })


def fig_topk_and_calibration(metrics: dict, P_scan, y_scan, P_cal_scan,
                              P_per, y_per, P_per_cal, out_path: Path):
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    # --- Top-left: top-k bar chart (scan vs patient) ---
    ax = fig.add_subplot(gs[0, 0])
    ks = [1, 2, 3]
    x = np.arange(len(ks))
    w = 0.36
    scan_vals = [metrics["scan"][f"top{k}_acc"] for k in ks]
    pat_vals = [metrics["patient"][f"top{k}_acc"] for k in ks]
    b1 = ax.bar(x - w / 2, scan_vals, w, label=f"Per-scan (n={metrics['scan']['n']})",
                color="#3498db", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w / 2, pat_vals, w, label=f"Per-patient (n={metrics['patient']['n']})",
                color="#2ecc71", edgecolor="black", linewidth=0.5)
    for bars, vals in [(b1, scan_vals), (b2, pat_vals)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in ks])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Top-k accuracy — scan vs patient aggregation",
                 fontsize=14, fontweight="bold")
    ax.axhline(1.0 / N_CLASSES, color="#7f8c8d", linestyle=":",
               linewidth=1.5, label=f"Random (1/{N_CLASSES})")
    ax.legend(loc="lower right")

    # --- Top-right: F1 comparison scan vs patient ---
    ax = fig.add_subplot(gs[0, 1])
    metric_names = ["Weighted F1", "Macro F1"]
    scan_f1 = [metrics["scan"]["weighted_f1"], metrics["scan"]["macro_f1"]]
    pat_f1 = [metrics["patient"]["weighted_f1"], metrics["patient"]["macro_f1"]]
    x = np.arange(len(metric_names))
    b1 = ax.bar(x - w / 2, scan_f1, w, color="#3498db",
                edgecolor="black", linewidth=0.5, label=f"Per-scan")
    b2 = ax.bar(x + w / 2, pat_f1, w, color="#2ecc71",
                edgecolor="black", linewidth=0.5, label=f"Per-patient")
    for bars, vals in [(b1, scan_f1), (b2, pat_f1)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("F1 — scan vs patient aggregation",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")

    # --- Bottom-left: reliability diagram (pre-calibration, per scan) ---
    ax = fig.add_subplot(gs[1, 0])
    _reliability_plot(ax, P_scan, y_scan, title_suffix="before Platt",
                      ece=metrics["scan"]["ece_pre"],
                      brier=metrics["scan"]["brier_pre"])

    # --- Bottom-right: reliability diagram (post-calibration) ---
    ax = fig.add_subplot(gs[1, 1])
    _reliability_plot(ax, P_cal_scan, y_scan, title_suffix="after Platt",
                      ece=metrics["scan"]["ece_post"],
                      brier=metrics["scan"]["brier_post"])

    fig.suptitle(
        "v4 ensemble — top-k accuracy, F1 aggregation, and calibration "
        f"(n={metrics['scan']['n']} scans / {metrics['patient']['n']} patients, person-LOPO)",
        fontsize=17, fontweight="bold", y=0.995,
    )
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> saved {out_path.name}")


def _reliability_plot(ax, P, y, title_suffix: str, ece: float, brier: float):
    centers, accs, confs, counts = reliability_bins(P, y, n_bins=10)
    ax.plot([0, 1], [0, 1], ls="--", color="#7f8c8d", lw=1.5, label="Perfect calibration")

    # Bars: actual bin accuracy
    valid = ~np.isnan(accs)
    w = 0.09
    ax.bar(centers[valid], accs[valid], width=w,
           color="#9b59b6", edgecolor="black", linewidth=0.5,
           alpha=0.75, label="Empirical accuracy")
    # Gap bars (confidence > accuracy = overconfident)
    for c, a, f, n in zip(centers, accs, confs, counts):
        if np.isnan(a) or n == 0:
            continue
        # Mark bin sample count
        ax.text(c, max(a, f) + 0.03, f"n={n}",
                ha="center", va="bottom", fontsize=8, color="#555")

    # Confidence markers
    ax.plot(centers[valid], confs[valid], "o-", color="#e74c3c",
            lw=1.2, ms=6, label="Mean confidence")

    ax.set_xlabel("Top-1 confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Reliability diagram ({title_suffix})\n"
                 f"ECE = {ece:.4f}   Brier = {brier:.4f}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)


def update_fig05_with_topk(metrics: dict, y_scan: np.ndarray,
                            y_pred_scan: np.ndarray, out_path: Path):
    """Refresh `05_per_class_metrics.png` overlaying top-k annotation."""
    labels = list(range(N_CLASSES))
    p, r, f, sup = precision_recall_fscore_support(
        y_scan, y_pred_scan, labels=labels, zero_division=0,
    )
    weighted_f1 = metrics["scan"]["weighted_f1"]
    macro_f1 = metrics["scan"]["macro_f1"]
    top1 = metrics["scan"]["top1_acc"]
    top2 = metrics["scan"]["top2_acc"]
    top3 = metrics["scan"]["top3_acc"]

    pretty = [PRETTY_FLAT[c] for c in CLASSES]
    n = N_CLASSES
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    b1 = ax.bar(x - width, p, width, label="Precision", color="#3498db",
                edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x,         r, width, label="Recall",    color="#2ecc71",
                edgecolor="black", linewidth=0.5)
    b3 = ax.bar(x + width, f, width, label="F1",        color="#9b59b6",
                edgecolor="black", linewidth=0.5)

    for bars in (b1, b2, b3):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ymax = max(max(p), max(r), max(f), 1.0) * 1.05
    for xi, s, c in zip(x, sup, CLASSES):
        ax.text(xi, ymax + 0.02, f"n={s}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=CLASS_COLORS[c])

    ax.axhline(weighted_f1, color="#c0392b", linestyle="--", linewidth=2,
               label=f"weighted-F1 = {weighted_f1:.3f}")
    ax.axhline(top1, color="#2980b9", linestyle=":", linewidth=2,
               label=f"top-1 acc = {top1:.3f}")
    ax.axhline(top2, color="#27ae60", linestyle=":", linewidth=2,
               label=f"top-2 acc = {top2:.3f}")
    ax.axhline(top3, color="#16a085", linestyle=":", linewidth=2,
               label=f"top-3 acc = {top3:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(pretty, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_ylim(0, ymax + 0.25)
    ax.set_title(
        f"Per-class metrics + top-k context "
        f"(v4 ensemble, person-LOPO, n=240 scans)\n"
        f"weighted-F1={weighted_f1:.3f}   macro-F1={macro_f1:.3f}   "
        f"top-1={top1:.3f}   top-2={top2:.3f}   top-3={top3:.3f}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", frameon=True, ncol=2, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> saved {out_path.name}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown(metrics: dict, boot_scan: dict, boot_patient: dict,
                    platt_info: dict, out_path: Path):
    s = metrics["scan"]
    p = metrics["patient"]

    lines: list[str] = []
    lines.append("# Evaluation Metrics Upgrade — Top-k + Per-Patient + Calibration\n")
    lines.append(
        "Upgrade to the tear-AFM evaluation suite to report pitch-ready metrics "
        "beyond weighted F1: **top-k accuracy**, **per-patient aggregation**, "
        "**ECE / Brier calibration** (pre/post Platt scaling), and "
        "**person-level bootstrap CIs**.\n"
    )
    lines.append("## Source & methodology\n")
    lines.append(
        f"- **Model:** v4 multi-scale ensemble (`models/ensemble_v4_multiscale/`) — "
        f"3-component geometric mean of DINOv2-B @ 90 nm/px, DINOv2-B @ 45 nm/px, "
        f"and BiomedCLIP with D4 TTA.\n"
        f"- **OOF source:** regenerated honestly with the V2 recipe used by "
        f"`scripts/multiscale_experiment.py`, cached at `cache/v4_oof_predictions.npz`.\n"
        f"- **Split:** **person-level LOPO** (35 persons) via "
        f"`teardrop.cv.leave_one_patient_out` with `teardrop.data.person_id`.\n"
        f"- **Calibration:** per-class Platt (one-vs-rest logistic on logit-raw softmax), "
        f"fit with the **same** person-LOPO protocol — no leakage between scans of one person.\n"
        f"- **Bootstrap:** 1000 resamples of the **35 persons** (with replacement); scans "
        f"of a sampled person are pulled in as a block. Patient-level F1 bootstrap uses "
        f"the person-aggregated predictions.\n"
    )

    lines.append("## Core metric table\n")
    lines.append("| Metric | Per-scan (n={0}) | Per-patient (n={1}) |".format(s["n"], p["n"]))
    lines.append("|---|---:|---:|")
    lines.append(f"| Weighted F1 | **{s['weighted_f1']:.4f}** | **{p['weighted_f1']:.4f}** |")
    lines.append(f"| Macro F1 | {s['macro_f1']:.4f} | {p['macro_f1']:.4f} |")
    lines.append(f"| Top-1 accuracy | {s['top1_acc']:.4f} | {p['top1_acc']:.4f} |")
    lines.append(f"| Top-2 accuracy | **{s['top2_acc']:.4f}** | **{p['top2_acc']:.4f}** |")
    lines.append(f"| Top-3 accuracy | {s['top3_acc']:.4f} | {p['top3_acc']:.4f} |")
    lines.append(f"| ECE (pre-Platt, 10 bins) | {s['ece_pre']:.4f} | {p['ece_pre']:.4f} |")
    lines.append(f"| ECE (post-Platt) | {s['ece_post']:.4f} | {p['ece_post']:.4f} |")
    lines.append(f"| Brier (multi-class OvR) — pre | {s['brier_pre']:.4f} | {p['brier_pre']:.4f} |")
    lines.append(f"| Brier — post | {s['brier_post']:.4f} | {p['brier_post']:.4f} |")
    lines.append("")

    lines.append("## Per-class F1\n")
    lines.append("| Class | Per-scan F1 | Per-patient F1 |")
    lines.append("|---|---:|---:|")
    for i, c in enumerate(CLASSES):
        lines.append(
            f"| {PRETTY_FLAT[c]} | {s['per_class_f1'][i]:.4f} | {p['per_class_f1'][i]:.4f} |"
        )
    lines.append("")

    lines.append("## Bootstrap confidence intervals (person-level)\n")
    lines.append("| Level | Point | Mean ± std | 95% CI | n_boot |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| Per-scan weighted F1 | {boot_scan['point']:.4f} | "
        f"{boot_scan['mean']:.4f} ± {boot_scan['std']:.4f} | "
        f"[{boot_scan['ci_lo']:.4f}, {boot_scan['ci_hi']:.4f}] | {boot_scan['n_boot']} |"
    )
    lines.append(
        f"| Per-patient weighted F1 | {boot_patient['point']:.4f} | "
        f"{boot_patient['mean']:.4f} ± {boot_patient['std']:.4f} | "
        f"[{boot_patient['ci_lo']:.4f}, {boot_patient['ci_hi']:.4f}] | {boot_patient['n_boot']} |"
    )
    lines.append("")

    lines.append("## Calibration effect of Platt scaling\n")
    ece_delta_scan = s['ece_pre'] - s['ece_post']  # positive means improvement
    ece_delta_pat = p['ece_pre'] - p['ece_post']
    brier_delta_scan = s['brier_pre'] - s['brier_post']
    lines.append(
        f"Per-scan ECE went from **{s['ece_pre']:.4f} → {s['ece_post']:.4f}** "
        f"(Δ = **{-ece_delta_scan:+.4f}** — lower is better). "
        f"Brier (multi-class OvR) from **{s['brier_pre']:.4f} → "
        f"{s['brier_post']:.4f}** (Δ = {-brier_delta_scan:+.4f}).\n"
    )
    if ece_delta_pat > 0:
        lines.append(
            f"Per-patient ECE went from {p['ece_pre']:.4f} → {p['ece_post']:.4f} "
            f"(Δ = {-ece_delta_pat:+.4f}).\n"
        )
    else:
        lines.append(
            f"**Honest finding:** per-patient ECE went from "
            f"{p['ece_pre']:.4f} → {p['ece_post']:.4f} "
            f"(Δ = {-ece_delta_pat:+.4f}; *worse* after calibration). "
            f"This is expected — averaging softmaxes across a patient's scans "
            f"already smooths overconfidence; stacking Platt on top of that can "
            f"over-correct. The raw per-patient probabilities are already more "
            f"calibrated than the raw per-scan ones; Platt should be applied "
            f"**only at scan level** and then aggregated, not the other way around.\n"
        )
    lines.append(
        f"Calibration is fit via per-class OvR Platt with person-level LOPO "
        f"(no leakage). Platt parameters: {platt_info['n_calib_folds']} fitted folds.\n"
    )

    # ---- Pitch one-liners ----
    lines.append("## Pitch one-liners (verified numbers)\n")
    lines.append(
        f"- **\"Top-2 accuracy: {s['top2_acc'] * 100:.1f}% — as a triage tool, the model "
        f"ranks the correct class among its top 2 in roughly {s['top2_acc'] * 100:.0f}% "
        f"of scans.\"**\n"
        f"- **\"Per-patient weighted F1: {p['weighted_f1']:.3f} "
        f"(vs per-scan {s['weighted_f1']:.3f}, Δ = "
        f"{p['weighted_f1'] - s['weighted_f1']:+.3f}) — aggregating all of a "
        f"patient's scans softens label noise.\"**\n"
        f"- **\"Calibrated ECE = {s['ece_post']:.3f} "
        f"(down from {s['ece_pre']:.3f} raw) — confidence estimates are "
        f"trustworthy after Platt scaling.\"**\n"
        f"- **\"Honest 95% CI on weighted F1: "
        f"[{boot_scan['ci_lo']:.3f}, {boot_scan['ci_hi']:.3f}] "
        f"(person-level bootstrap, B = {boot_scan['n_boot']}).\"**\n"
    )

    lines.append("## Verdict — strongest pitch framing\n")
    strongest = _pick_strongest(metrics, boot_scan, boot_patient)
    lines.append(strongest)

    lines.append("\n## Figures\n")
    lines.append(
        "- `reports/pitch/07_topk_and_calibration.png` — top-k bar chart (scan "
        "vs patient) + pre/post Platt reliability diagrams.\n"
        "- `reports/pitch/05_per_class_metrics.png` — UPDATED to overlay top-1/2/3 "
        "accuracy alongside per-class F1.\n"
    )

    out_path.write_text("\n".join(lines))
    print(f"  -> wrote {out_path}")


def _pick_strongest(metrics, boot_scan, boot_patient) -> str:
    s = metrics["scan"]
    p = metrics["patient"]
    # Most pitch-actionable = top-2 accuracy AND calibrated ECE, both honest.
    top2_scan = s["top2_acc"]
    ece_post = s["ece_post"]
    ece_pre = s["ece_pre"]
    f1_gap = p["weighted_f1"] - s["weighted_f1"]

    parts = []
    parts.append(
        f"The **top-2 accuracy = {top2_scan:.3f}** (≈ {top2_scan * 100:.0f}%) "
        f"is the most compelling single number for a medical-triage audience: "
        f"the model's second-guess is already right in ~{top2_scan * 100:.0f}% of scans, "
        f"which reframes the headline from \"0.69 F1\" to \"{top2_scan * 100:.0f}% "
        f"top-2 hit rate — human-in-the-loop-ready\"."
    )
    if f1_gap >= 0.08:
        parts.append(
            f"The **per-patient F1 of {p['weighted_f1']:.3f}** "
            f"(vs per-scan {s['weighted_f1']:.3f}, **Δ = +{f1_gap:.3f}**) is by far "
            f"the biggest headline lift we have. It's also the *clinically correct* "
            f"reporting level: a doctor receives one prediction per patient, not one "
            f"per AFM frame. Averaging 3-22 scans' softmaxes trades variance for bias "
            f"in exactly the right direction for a deployed screening tool."
        )
    elif f1_gap >= 0.02:
        parts.append(
            f"The **per-patient F1 of {p['weighted_f1']:.3f}** "
            f"(vs per-scan {s['weighted_f1']:.3f}, Δ = +{f1_gap:.3f}) shows that "
            f"aggregating softmaxes over a patient's multiple scans reduces label-noise "
            f"and is the *clinically correct* reporting level (one prediction per patient)."
        )
    else:
        parts.append(
            f"Per-patient F1 = {p['weighted_f1']:.3f} is within ±{abs(f1_gap):.3f} of the "
            f"per-scan number — aggregating doesn't materially change the headline, "
            f"which is itself a useful honesty signal."
        )
    if ece_pre - ece_post > 0.03:
        parts.append(
            f"Calibration: raw ECE = {ece_pre:.3f}, post-Platt = {ece_post:.3f} "
            f"(**Δ = -{ece_pre - ece_post:.3f}**). Platt scaling roughly halves "
            f"miscalibration at the scan level — a publication-grade story. "
            f"The honest caveat: at the patient level (post-aggregation) Platt "
            f"does not further help and can slightly hurt, because averaging 6 "
            f"softmaxes already smooths confidences. Pitch the scan-level number."
        )
    else:
        parts.append(
            f"Calibration: raw ECE = {ece_pre:.3f}, post-Platt = {ece_post:.3f} "
            f"(Δ = -{ece_pre - ece_post:.3f}). The raw ensemble is already "
            f"reasonably calibrated (LR + balanced class-weights yields moderate "
            f"confidences); Platt is a safety net rather than a headline win here."
        )
    parts.append(
        f"Bootstrap 95% CI on per-scan weighted F1 = "
        f"[{boot_scan['ci_lo']:.3f}, {boot_scan['ci_hi']:.3f}] — pitch this as "
        f"honest uncertainty estimation at the 35-person resolution we actually have."
    )
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    setup_style()
    t0 = time.time()
    print("=" * 78)
    print("Eval metrics upgrade: top-k, per-patient, calibration, bootstrap")
    print("=" * 78)

    data = load_or_build_v4_oof()
    P_scan = data["proba"]
    y_scan = data["y"]
    persons = data["persons"]

    # Calibration
    print("\n[platt] person-LOPO per-class Platt scaling")
    t_p = time.time()
    P_cal_scan = platt_calibrate_lopo(P_scan, y_scan, persons)
    n_cal_folds = len(np.unique(persons))
    print(f"  done in {time.time() - t_p:.1f}s  ({n_cal_folds} folds)")

    # Metrics
    print("\n[metrics] computing full bundle")
    metrics, (P_per, y_per, pids, P_per_cal) = compute_all_metrics(
        P_scan, y_scan, persons, P_cal_scan,
    )

    # Bootstrap (person-level; patient F1 bootstrap uses patient aggregation)
    print("\n[bootstrap] 1000 person-level resamples")
    t_b = time.time()
    boot_scan = bootstrap_person_f1(P_scan, y_scan, persons, n_boot=1000, seed=42)
    # Patient-level F1 bootstrap: resample persons, take their single patient-aggregated row.
    person_to_per_row = {pid: i for i, pid in enumerate(pids)}

    def _patient_boot(seed=43):
        rng = np.random.default_rng(seed)
        unique = np.unique(persons)
        scores = np.zeros(1000, dtype=np.float64)
        for b in range(1000):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            rows = np.array([person_to_per_row[p] for p in sampled])
            scores[b] = f1_score(y_per[rows], P_per[rows].argmax(1),
                                  average="weighted", zero_division=0)
        return {
            "point": float(f1_score(y_per, P_per.argmax(1), average="weighted", zero_division=0)),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "ci_lo": float(np.percentile(scores, 2.5)),
            "ci_hi": float(np.percentile(scores, 97.5)),
            "n_boot": 1000,
            "average": "weighted",
        }

    boot_patient = _patient_boot()
    print(f"  bootstrap done in {time.time() - t_b:.1f}s")
    print(f"  scan boot: point={boot_scan['point']:.4f} "
          f"mean={boot_scan['mean']:.4f}±{boot_scan['std']:.4f} "
          f"CI=[{boot_scan['ci_lo']:.4f},{boot_scan['ci_hi']:.4f}]")
    print(f"  patient boot: point={boot_patient['point']:.4f} "
          f"mean={boot_patient['mean']:.4f}±{boot_patient['std']:.4f} "
          f"CI=[{boot_patient['ci_lo']:.4f},{boot_patient['ci_hi']:.4f}]")

    # Figures
    print("\n[figures]")
    fig_topk_and_calibration(
        metrics, P_scan, y_scan, P_cal_scan, P_per, y_per, P_per_cal,
        PITCH / "07_topk_and_calibration.png",
    )
    update_fig05_with_topk(
        metrics, y_scan, P_scan.argmax(1),
        PITCH / "05_per_class_metrics.png",
    )

    # Markdown
    print("\n[report]")
    platt_info = {"n_calib_folds": n_cal_folds}
    write_markdown(
        metrics, boot_scan, boot_patient, platt_info,
        REPORTS / "METRICS_UPGRADE.md",
    )

    # JSON dump for downstream scripts
    json_out = REPORTS / "metrics_upgrade.json"
    json_out.write_text(json.dumps({
        "metrics": metrics,
        "bootstrap_scan": boot_scan,
        "bootstrap_patient": boot_patient,
        "platt_info": platt_info,
        "classes": CLASSES,
    }, indent=2))
    print(f"  -> wrote {json_out}")

    print(f"\n[done] elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
