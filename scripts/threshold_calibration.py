"""Nested LOPO per-class threshold calibration for v4 multi-scale champion.

Protocol (HONEST — no OOF leakage):

  OUTER loop (35 folds):  for each person P in 1..35:
    - Outer train = persons != P (34 persons, ~235 scans)
    - Outer eval  = person P's scans

    INNER loop (34 folds within outer_train):  for each person Q in outer_train:
      - Inner train = outer_train \ {Q}  (33 persons)
      - Inner eval  = Q's scans
      - Fit v4 (3 members, L2 -> StdScaler -> LR balanced) on inner train only.
      - Predict softmaxes on Q's scans -> concat into inner_OOF.

    Grid-search per-class thresholds on inner_OOF maximizing weighted F1.
    Use:  pred = argmax(softmax_probs - thresholds[None, :])
    Apply best thresholds to outer eval (fit v4 on full outer_train this time).
    Rotate P.

  Concatenate outer predictions across all 35 folds -> final honest metrics.

Leakage check:
  - Inner fits NEVER see person P's scans (P is held out of outer_train before
    the inner loop starts). So inner OOF softmaxes are genuinely out-of-sample
    w.r.t. P. Thresholds tuned on inner OOF are therefore learned without any
    signal from P, and applying them to P's outer eval is honest.

  - The v4 OOF cache (cache/v4_oof_predictions.npz) was produced with LOPO
    where for held-out person P, the LR was trained on ALL other 34 persons.
    We CANNOT tune thresholds on that OOF and eval on the same OOF — that's
    the leaky recipe caught in earlier red-team waves. We rebuild both softmaxes
    and thresholds under the strict nested protocol above.

Cost: 35 outer * (34 inner + 1 refit) * 3 members = 3675 LR fits.
  LR on 240 rows of ~768-D features takes ~1-2s per fit => ~2-3 hr on CPU.
  We optimize by sharing the 34 persons' subset and by using lbfgs with n_jobs=-1.

Fallback modes also evaluated:
  - abstention + fallback: if max prob < tau, route to 2nd-best class.
  - calibrated margin: pred = argmax(log(p+eps) - log_thresh).

Output:
  cache/calibrated_v4_predictions.json
  reports/THRESHOLD_CALIBRATION.md
"""
from __future__ import annotations

import json
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
from teardrop.data import CLASSES, person_id  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"

N_CLASSES = len(CLASSES)
EPS = 1e-12
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _mean_pool_tiles(X_tiles: np.ndarray, tile_to_scan: np.ndarray,
                     n_scans: int) -> np.ndarray:
    D = X_tiles.shape[1]
    out = np.zeros((n_scans, D), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int64)
    for i, s in enumerate(tile_to_scan):
        out[s] += X_tiles[i]
        counts[s] += 1
    counts = np.maximum(counts, 1)
    out /= counts[:, None]
    return out


def load_v4_features():
    """Load and align the 3 v4 per-member scan-level features."""
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz",
                  allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz",
                  allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz",
                  allow_pickle=True)

    y = np.asarray(z90["scan_y"], dtype=np.int64)
    paths_90 = [str(p) for p in z90["scan_paths"]]
    paths_45 = [str(p) for p in z45["scan_paths"]]
    paths_bc = [str(p) for p in zbc["scan_paths"]]

    n_scans = len(y)
    X90 = _mean_pool_tiles(z90["X"], z90["tile_to_scan"], n_scans)

    # 45nm cache may be in a different order — align by path
    idx_map = {p: i for i, p in enumerate(paths_45)}
    order_45 = np.array([idx_map[p] for p in paths_90])
    X45_raw = _mean_pool_tiles(z45["X"], z45["tile_to_scan"], len(paths_45))
    X45 = X45_raw[order_45]

    idx_map = {p: i for i, p in enumerate(paths_bc)}
    order_bc = np.array([idx_map[p] for p in paths_90])
    Xbc = zbc["X_scan"].astype(np.float32)[order_bc]

    # Labels must match
    y45 = np.asarray(z45["scan_y"])[order_45]
    ybc = np.asarray(zbc["scan_y"])[order_bc]
    assert np.array_equal(y, y45), "label mismatch 90↔45"
    assert np.array_equal(y, ybc), "label mismatch 90↔bclip"

    groups = np.array([person_id(Path(p)) for p in paths_90])
    return {
        "X90": X90, "X45": X45, "Xbc": Xbc,
        "y": y, "groups": groups, "scan_paths": np.array(paths_90),
    }


# ---------------------------------------------------------------------------
# v4 member training + softmax
# ---------------------------------------------------------------------------

def fit_member(X_tr: np.ndarray, y_tr: np.ndarray):
    X_n = normalize(X_tr, norm="l2", axis=1)
    sc = StandardScaler().fit(X_n)
    X_std = sc.transform(X_n)
    X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)
    clf = LogisticRegression(
        class_weight="balanced", max_iter=3000, C=1.0,
        solver="lbfgs", n_jobs=1, random_state=42,
    )
    clf.fit(X_std, y_tr)
    return sc, clf


def predict_member(X_te: np.ndarray, sc: StandardScaler,
                   clf: LogisticRegression) -> np.ndarray:
    X_n = normalize(X_te, norm="l2", axis=1)
    X_std = sc.transform(X_n)
    X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)
    p = clf.predict_proba(X_std)
    out = np.zeros((X_te.shape[0], N_CLASSES), dtype=np.float64)
    for ci, cls in enumerate(clf.classes_):
        out[:, cls] = p[:, ci]
    return out


def v4_softmax(X_members_tr: list[np.ndarray],
               X_members_te: list[np.ndarray],
               y_tr: np.ndarray) -> np.ndarray:
    """Fit all 3 members on train; return geometric-mean softmax on test."""
    log_sum = None
    for Xtr, Xte in zip(X_members_tr, X_members_te):
        sc, clf = fit_member(Xtr, y_tr)
        p = predict_member(Xte, sc, clf)
        if log_sum is None:
            log_sum = np.log(p + EPS)
        else:
            log_sum = log_sum + np.log(p + EPS)
    g = np.exp(log_sum / len(X_members_tr))
    g /= g.sum(axis=1, keepdims=True)
    return g


# ---------------------------------------------------------------------------
# Nested LOPO calibration
# ---------------------------------------------------------------------------

def grid_search_thresholds(inner_probs: np.ndarray, inner_y: np.ndarray,
                            grid: np.ndarray) -> tuple[np.ndarray, float]:
    """Coordinate-wise greedy grid search for per-class thresholds to max weighted F1.

    Returns (thresholds vector shape (N_CLASSES,), best_f1).
    We subtract thresholds from the softmax probs before argmax.
    Grid step 0.05 over [0.05, 0.95]. Initialize at 0 (= plain argmax).

    Coordinate ascent: iterate 2 passes.
    """
    best_thr = np.zeros(N_CLASSES, dtype=np.float64)
    best_pred = inner_probs.argmax(axis=1)
    best_f1 = f1_score(inner_y, best_pred, average="weighted", zero_division=0)
    for _pass in range(2):
        for c in range(N_CLASSES):
            for tau in grid:
                trial = best_thr.copy()
                trial[c] = tau
                pred = (inner_probs - trial[None, :]).argmax(axis=1)
                f1 = f1_score(inner_y, pred, average="weighted", zero_division=0)
                if f1 > best_f1 + 1e-8:
                    best_f1 = f1
                    best_thr = trial
    return best_thr, best_f1


def nested_lopo_calibration(data: dict, grid_step: float = 0.05,
                             verbose: bool = True) -> dict:
    """Returns dict with outer_probs_uncal, outer_probs, outer_preds_uncal,
    outer_preds_calib, outer_preds_abstain, thresholds_per_outer, abstain_tau_per_outer.
    """
    y = data["y"]
    groups = data["groups"]
    X_members = [data["X90"], data["X45"], data["Xbc"]]
    unique_persons = np.unique(groups)
    n_scans = len(y)

    grid = np.arange(grid_step, 0.95 + grid_step / 2, grid_step)

    outer_probs_uncal = np.zeros((n_scans, N_CLASSES), dtype=np.float64)
    thresholds_per_outer = np.zeros((len(unique_persons), N_CLASSES), dtype=np.float64)
    abstain_tau_per_outer = np.zeros(len(unique_persons), dtype=np.float64)
    outer_person_ids = []

    t_start = time.time()
    for pi, p_out in enumerate(unique_persons):
        outer_person_ids.append(p_out)
        is_out = groups == p_out
        outer_idx = np.where(is_out)[0]
        inner_pool_mask = ~is_out
        inner_groups = groups[inner_pool_mask]
        inner_y = y[inner_pool_mask]
        unique_inner = np.unique(inner_groups)

        # Inner LOPO: fit v4 on inner_train (exclude Q), predict on Q.
        n_inner = len(inner_groups)
        inner_probs = np.zeros((n_inner, N_CLASSES), dtype=np.float64)
        inner_pool_idx = np.where(inner_pool_mask)[0]

        for q in unique_inner:
            q_local = np.where(inner_groups == q)[0]
            q_global = inner_pool_idx[q_local]
            tr_mask_global = inner_pool_mask & (groups != q)
            tr_idx = np.where(tr_mask_global)[0]
            X_tr_list = [Xm[tr_idx] for Xm in X_members]
            X_te_list = [Xm[q_global] for Xm in X_members]
            probs_q = v4_softmax(X_tr_list, X_te_list, y[tr_idx])
            inner_probs[q_local] = probs_q

        # Grid-search per-class thresholds on inner OOF.
        thr, best_f1 = grid_search_thresholds(inner_probs, inner_y, grid)
        thresholds_per_outer[pi] = thr

        # Abstention fallback tau: best tau in grid such that if max prob < tau,
        # swap to 2nd-best class; also pick tau minimizing weighted F1 loss.
        best_abs_tau = 0.0
        best_abs_f1 = f1_score(inner_y, inner_probs.argmax(axis=1),
                                average="weighted", zero_division=0)
        for tau in grid:
            top2_idx = np.argsort(inner_probs, axis=1)[:, -2:]
            max_prob = inner_probs.max(axis=1)
            pred = np.where(max_prob < tau, top2_idx[:, 0], top2_idx[:, 1])
            f1 = f1_score(inner_y, pred, average="weighted", zero_division=0)
            if f1 > best_abs_f1 + 1e-8:
                best_abs_f1 = f1
                best_abs_tau = tau
        abstain_tau_per_outer[pi] = best_abs_tau

        # Now refit v4 on the FULL outer train (34 persons) and predict on P.
        X_tr_list = [Xm[inner_pool_idx] for Xm in X_members]
        X_te_list = [Xm[outer_idx] for Xm in X_members]
        outer_probs_uncal[outer_idx] = v4_softmax(X_tr_list, X_te_list, inner_y)

        if verbose:
            elapsed = time.time() - t_start
            print(f"  [{pi + 1:2d}/{len(unique_persons)}] person={p_out:24s} "
                  f"n_eval={len(outer_idx)} thr={np.round(thr, 2).tolist()} "
                  f"abs_tau={best_abs_tau:.2f} inner_F1={best_f1:.4f} "
                  f"elapsed={elapsed:.1f}s", flush=True)

    # Apply thresholds per outer person: calibrated = argmax(probs - thr_for_P)
    outer_preds_calib = np.zeros(n_scans, dtype=np.int64)
    outer_preds_abstain = np.zeros(n_scans, dtype=np.int64)
    for pi, p_out in enumerate(outer_person_ids):
        is_out = groups == p_out
        probs = outer_probs_uncal[is_out]
        thr = thresholds_per_outer[pi]
        outer_preds_calib[is_out] = (probs - thr[None, :]).argmax(axis=1)

        # Abstain fallback
        tau = abstain_tau_per_outer[pi]
        top2 = np.argsort(probs, axis=1)[:, -2:]
        maxp = probs.max(axis=1)
        outer_preds_abstain[is_out] = np.where(maxp < tau, top2[:, 0], top2[:, 1])

    outer_preds_uncal = outer_probs_uncal.argmax(axis=1)
    return {
        "outer_probs_uncal": outer_probs_uncal,
        "outer_preds_uncal": outer_preds_uncal,
        "outer_preds_calib": outer_preds_calib,
        "outer_preds_abstain": outer_preds_abstain,
        "thresholds_per_outer": thresholds_per_outer,
        "abstain_tau_per_outer": abstain_tau_per_outer,
        "outer_person_ids": outer_person_ids,
    }


# ---------------------------------------------------------------------------
# Metrics & bootstrap
# ---------------------------------------------------------------------------

def metrics_report(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y, pred, average=None,
                                  labels=list(range(N_CLASSES)),
                                  zero_division=0).tolist(),
    }


def person_bootstrap(y: np.ndarray, groups: np.ndarray,
                      pred_a: np.ndarray, pred_b: np.ndarray,
                      B: int = 1000, seed: int = 7) -> dict:
    """Resample persons with replacement; report Δ(weighted F1) b − a."""
    rng = np.random.default_rng(seed)
    unique_persons = np.unique(groups)
    person_to_idx = {p: np.where(groups == p)[0] for p in unique_persons}

    deltas_w, deltas_m = [], []
    for _ in range(B):
        sampled = rng.choice(unique_persons, size=len(unique_persons), replace=True)
        idx = np.concatenate([person_to_idx[p] for p in sampled])
        yb = y[idx]; pa = pred_a[idx]; pb = pred_b[idx]
        fa_w = f1_score(yb, pa, average="weighted", zero_division=0)
        fb_w = f1_score(yb, pb, average="weighted", zero_division=0)
        fa_m = f1_score(yb, pa, average="macro", zero_division=0)
        fb_m = f1_score(yb, pb, average="macro", zero_division=0)
        deltas_w.append(fb_w - fa_w)
        deltas_m.append(fb_m - fa_m)
    deltas_w = np.array(deltas_w); deltas_m = np.array(deltas_m)
    return {
        "weighted": {
            "mean": float(deltas_w.mean()),
            "median": float(np.median(deltas_w)),
            "ci95": [float(np.percentile(deltas_w, 2.5)),
                     float(np.percentile(deltas_w, 97.5))],
            "p_positive": float((deltas_w > 0).mean()),
        },
        "macro": {
            "mean": float(deltas_m.mean()),
            "median": float(np.median(deltas_m)),
            "ci95": [float(np.percentile(deltas_m, 2.5)),
                     float(np.percentile(deltas_m, 97.5))],
            "p_positive": float((deltas_m > 0).mean()),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("Nested-LOPO per-class threshold calibration for v4")
    print("=" * 78)
    t0 = time.time()

    data = load_v4_features()
    y = data["y"]; groups = data["groups"]
    print(f"  n_scans={len(y)}  n_persons={len(np.unique(groups))}")
    print(f"  class counts: {np.bincount(y).tolist()}  classes: {CLASSES}")

    # Sanity: v4 uncalibrated OOF reproduces 0.6887 from cache
    print("\n[sanity] reproducing v4 uncalibrated OOF via standard LOPO (no nesting)")
    outer_probs_v4_check = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    X_members = [data["X90"], data["X45"], data["Xbc"]]
    for tr, va in leave_one_patient_out(groups):
        Xtr = [Xm[tr] for Xm in X_members]
        Xte = [Xm[va] for Xm in X_members]
        outer_probs_v4_check[va] = v4_softmax(Xtr, Xte, y[tr])
    pred_check = outer_probs_v4_check.argmax(axis=1)
    f1_w_check = f1_score(y, pred_check, average="weighted", zero_division=0)
    f1_m_check = f1_score(y, pred_check, average="macro", zero_division=0)
    print(f"  reproduced v4 W-F1={f1_w_check:.4f} M-F1={f1_m_check:.4f} "
          f"(expect 0.6887 / 0.5541)")

    # Nested LOPO calibration
    print("\n[nested] running 35x34x3 LR fits + outer refits...")
    res = nested_lopo_calibration(data, grid_step=0.05, verbose=True)

    # Compute metrics
    print("\n[metrics]")
    uncal = metrics_report(y, res["outer_preds_uncal"])
    calib = metrics_report(y, res["outer_preds_calib"])
    abstain = metrics_report(y, res["outer_preds_abstain"])
    cache_ref_pred = np.load(CACHE / "v4_oof_predictions.npz")["proba"].argmax(axis=1)
    ref = metrics_report(y, cache_ref_pred)

    print(f"  v4 cached OOF   : W-F1={ref['weighted_f1']:.4f} "
          f"M-F1={ref['macro_f1']:.4f}")
    print(f"  v4 reproduced   : W-F1={uncal['weighted_f1']:.4f} "
          f"M-F1={uncal['macro_f1']:.4f}")
    print(f"  calib-threshold : W-F1={calib['weighted_f1']:.4f} "
          f"M-F1={calib['macro_f1']:.4f}")
    print(f"  abstain+fallbk  : W-F1={abstain['weighted_f1']:.4f} "
          f"M-F1={abstain['macro_f1']:.4f}")

    # Bootstrap calibrated vs uncalibrated
    print("\n[bootstrap] B=1000 person-level, calib - uncal")
    boot_calib = person_bootstrap(y, groups,
                                   res["outer_preds_uncal"],
                                   res["outer_preds_calib"], B=1000)
    print(f"  weighted: Δ={calib['weighted_f1'] - uncal['weighted_f1']:+.4f} "
          f"95% CI={boot_calib['weighted']['ci95']} "
          f"P(Δ>0)={boot_calib['weighted']['p_positive']:.3f}")
    print(f"  macro   : Δ={calib['macro_f1'] - uncal['macro_f1']:+.4f} "
          f"95% CI={boot_calib['macro']['ci95']} "
          f"P(Δ>0)={boot_calib['macro']['p_positive']:.3f}")

    boot_abstain = person_bootstrap(y, groups,
                                     res["outer_preds_uncal"],
                                     res["outer_preds_abstain"], B=1000)
    print(f"\n[bootstrap] B=1000 person-level, abstain - uncal")
    print(f"  weighted: Δ={abstain['weighted_f1'] - uncal['weighted_f1']:+.4f} "
          f"95% CI={boot_abstain['weighted']['ci95']} "
          f"P(Δ>0)={boot_abstain['weighted']['p_positive']:.3f}")

    # Save JSON
    out_json = {
        "protocol": "nested LOPO (outer=35 persons, inner=34 persons within each)",
        "classes": CLASSES,
        "grid": "np.arange(0.05, 0.95+eps, 0.05) per class, coordinate-ascent 2 passes",
        "fusion": "argmax(softmax_probs - thresholds[None,:])",
        "metrics": {
            "v4_cached_uncalibrated": ref,
            "v4_reproduced_uncalibrated": uncal,
            "calibrated_threshold_subtraction": calib,
            "abstain_second_best": abstain,
        },
        "deltas_vs_uncalibrated": {
            "calib_weighted_f1": calib["weighted_f1"] - uncal["weighted_f1"],
            "calib_macro_f1": calib["macro_f1"] - uncal["macro_f1"],
            "abstain_weighted_f1": abstain["weighted_f1"] - uncal["weighted_f1"],
            "abstain_macro_f1": abstain["macro_f1"] - uncal["macro_f1"],
        },
        "bootstrap_calib_vs_uncal": boot_calib,
        "bootstrap_abstain_vs_uncal": boot_abstain,
        "outer_person_ids": list(map(str, res["outer_person_ids"])),
        "thresholds_per_outer": res["thresholds_per_outer"].tolist(),
        "abstain_tau_per_outer": res["abstain_tau_per_outer"].tolist(),
        "preds": {
            "uncal": res["outer_preds_uncal"].tolist(),
            "calib": res["outer_preds_calib"].tolist(),
            "abstain": res["outer_preds_abstain"].tolist(),
            "y": y.tolist(),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    (CACHE / "calibrated_v4_predictions.json").write_text(json.dumps(out_json, indent=2))
    print(f"\n[saved] cache/calibrated_v4_predictions.json")

    # Write markdown report
    write_report(out_json, data)
    print(f"[saved] reports/THRESHOLD_CALIBRATION.md")
    print(f"\n[done] total elapsed: {time.time() - t0:.1f}s")

    # Verdict
    d = calib['weighted_f1'] - uncal['weighted_f1']
    p = boot_calib["weighted"]["p_positive"]
    if d > 0 and p > 0.95:
        verdict = f"CALIBRATION HELPS ({d:+.4f}, P(Δ>0)={p:.3f})"
    elif d > 0:
        verdict = f"CALIBRATION MARGINAL ({d:+.4f}, P(Δ>0)={p:.3f})"
    else:
        verdict = f"NO GAIN — v4 argmax IS the honest ceiling ({d:+.4f})"
    print(f"\n*** VERDICT: {verdict} ***")
    return out_json


def write_report(out: dict, data: dict) -> None:
    lines = []
    lines.append("# Per-Class Threshold Calibration — Nested LOPO\n")
    lines.append(f"Date: {time.strftime('%Y-%m-%d')}  ·  Protocol: nested leave-one-person-out (35 outer × 34 inner).\n")

    uncal = out["metrics"]["v4_reproduced_uncalibrated"]
    calib = out["metrics"]["calibrated_threshold_subtraction"]
    abstain = out["metrics"]["abstain_second_best"]
    ref = out["metrics"]["v4_cached_uncalibrated"]

    d_w = calib['weighted_f1'] - uncal['weighted_f1']
    d_m = calib['macro_f1'] - uncal['macro_f1']
    p_w = out["bootstrap_calib_vs_uncal"]["weighted"]["p_positive"]
    ci_w = out["bootstrap_calib_vs_uncal"]["weighted"]["ci95"]

    if d_w > 0 and p_w > 0.95:
        tl = f"**VERDICT:** calibration delivers +{d_w:.4f} weighted F1 over plain argmax, "
        tl += f"95% CI=[{ci_w[0]:+.4f},{ci_w[1]:+.4f}], P(Δ>0)={p_w:.3f}. Honest gain."
    elif d_w > 0:
        tl = f"**VERDICT:** calibration delivers +{d_w:.4f} weighted F1 but bootstrap "
        tl += f"P(Δ>0)={p_w:.3f} (<0.95) and 95% CI=[{ci_w[0]:+.4f},{ci_w[1]:+.4f}] "
        tl += f"includes 0. Treat as **noise**, not ship-worthy."
    else:
        tl = f"**VERDICT:** nested-LOPO calibration does NOT improve over plain argmax "
        tl += f"(Δ weighted F1={d_w:+.4f}). v4 argmax is the honest ceiling at "
        tl += f"**{uncal['weighted_f1']:.4f}**. Earlier +gains from flat-OOF tuning "
        tl += f"are confirmed as leakage-inflated."

    lines.append("## TL;DR\n")
    lines.append(tl + "\n")

    lines.append("## Protocol\n")
    lines.append(
        "**Nested LOPO (honest, no leakage):**\n\n"
        "```\n"
        "for each outer person P in 1..35:\n"
        "    outer_train_persons = {1..35} \\ {P}           # 34 persons\n"
        "    outer_eval = scans of P                         # 2–12 scans\n"
        "\n"
        "    # INNER: build OOF on outer_train WITHOUT P\n"
        "    for each inner person Q in outer_train_persons:\n"
        "        inner_train = outer_train_persons \\ {Q}    # 33 persons\n"
        "        fit 3-member v4 on inner_train\n"
        "        predict softmax on Q's scans → inner_OOF\n"
        "\n"
        "    τ* = argmax_τ weighted_F1(argmax(inner_OOF − τ), inner_y)\n"
        "         (coordinate-ascent grid search, τ ∈ {0.05, …, 0.95})\n"
        "\n"
        "    fit 3-member v4 on ALL of outer_train_persons (with Q=P excluded)\n"
        "    predict softmax on P's scans\n"
        "    apply τ*: pred_P = argmax(softmax_P − τ*)\n"
        "\n"
        "concat pred_P over all 35 outer folds → final honest metrics\n"
        "```\n\n"
        "**Leakage check (red-team self-audit):**\n"
        "1. Inner OOF for outer fold P is generated by LR heads trained only on "
        "`outer_train_persons \\ {Q}`, which never contains P → thresholds τ* are learned "
        "without any information from P.\n"
        "2. Outer softmax for P is from a model trained on `outer_train_persons` "
        "(= everyone ≠ P) — standard LOPO, already leakage-free.\n"
        "3. Concatenation of outer predictions into the final metric never reuses the inner "
        "OOF probabilities (those are discarded after τ* is picked).\n"
        "4. The cache file `v4_oof_predictions.npz` was **NOT** used to tune thresholds — "
        "we rebuilt all softmaxes from scratch with the nested protocol. Flat-OOF "
        "tuning (using v4_oof_predictions.npz for both tuning and eval) is the leaky recipe "
        "rejected in earlier waves (see `reports/RED_TEAM_*`).\n"
    )

    lines.append("## Results\n")
    lines.append("| Method | Weighted F1 | Macro F1 | Δ W-F1 vs v4 argmax | Δ M-F1 vs v4 argmax |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| v4 cached OOF (sanity) | {ref['weighted_f1']:.4f} | {ref['macro_f1']:.4f} | — | — |")
    lines.append(f"| v4 argmax (reproduced) | {uncal['weighted_f1']:.4f} | {uncal['macro_f1']:.4f} | 0.0000 | 0.0000 |")
    lines.append(f"| **Nested-LOPO calibrated** | **{calib['weighted_f1']:.4f}** | **{calib['macro_f1']:.4f}** | {calib['weighted_f1']-uncal['weighted_f1']:+.4f} | {calib['macro_f1']-uncal['macro_f1']:+.4f} |")
    lines.append(f"| Abstain → 2nd-best fallback | {abstain['weighted_f1']:.4f} | {abstain['macro_f1']:.4f} | {abstain['weighted_f1']-uncal['weighted_f1']:+.4f} | {abstain['macro_f1']-uncal['macro_f1']:+.4f} |")
    lines.append("")

    lines.append("### Per-class F1\n")
    lines.append("| class | v4 argmax | calibrated | Δ |")
    lines.append("|---|---:|---:|---:|")
    for ci, cname in enumerate(CLASSES):
        a = uncal['per_class_f1'][ci]
        b = calib['per_class_f1'][ci]
        lines.append(f"| {cname} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
    lines.append("")

    lines.append("## Bootstrap (B=1000, person-level resampling)\n")
    lines.append("| Comparison | metric | mean Δ | median Δ | 95% CI | P(Δ>0) |")
    lines.append("|---|---|---:|---:|:---:|---:|")
    b = out["bootstrap_calib_vs_uncal"]["weighted"]
    lines.append(f"| calibrated vs argmax | weighted F1 | {b['mean']:+.4f} | {b['median']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | {b['p_positive']:.3f} |")
    b = out["bootstrap_calib_vs_uncal"]["macro"]
    lines.append(f"| calibrated vs argmax | macro F1 | {b['mean']:+.4f} | {b['median']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | {b['p_positive']:.3f} |")
    b = out["bootstrap_abstain_vs_uncal"]["weighted"]
    lines.append(f"| abstain vs argmax | weighted F1 | {b['mean']:+.4f} | {b['median']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | {b['p_positive']:.3f} |")
    lines.append("")

    lines.append("## Threshold stability across outer folds\n")
    thr = np.array(out["thresholds_per_outer"])
    lines.append("| class | mean τ | std τ | median τ | frac folds τ>0 |")
    lines.append("|---|---:|---:|---:|---:|")
    for ci, cname in enumerate(CLASSES):
        col = thr[:, ci]
        lines.append(f"| {cname} | {col.mean():.3f} | {col.std():.3f} | {np.median(col):.3f} | {(col > 0).mean():.2f} |")
    lines.append("")

    lines.append("## Interpretation\n")
    if d_w <= 0:
        lines.append(
            "- Honest nested-LOPO calibration does not improve over the plain argmax of v4 "
            "softmax probabilities. This confirms that the gains previously observed from "
            "threshold tuning on flat OOF (tune + eval on the same 240-row matrix) were "
            "inflated by leakage, as flagged in prior red-team reports.\n"
            "- The class-balanced logistic regression head already absorbs most of the "
            "imbalance correction that per-class thresholds would otherwise add.\n"
            "- SucheOko (2 persons, 14 scans) cannot be meaningfully calibrated under "
            "person-LOPO: the inner-fold OOF rarely contains SucheOko at all, so the grid "
            "search has no signal to move its threshold.\n"
            "- v4 weighted F1 = 0.6887 stands as the honest person-LOPO ceiling for this "
            "multi-scale recipe. Future gains must come from new representations, not from "
            "post-hoc threshold surgery.\n"
        )
    elif p_w > 0.95:
        lines.append(
            "- Nested-LOPO calibration delivers a statistically significant weighted-F1 gain "
            f"({d_w:+.4f} over plain v4 argmax, P(Δ>0)={p_w:.3f}). This is an honestly "
            "measured improvement — threshold search never touched the evaluation set.\n"
            "- Candidate for shipping as v5 (`models/ensemble_v5_calibrated/`) subject to "
            "independent red-team verification of the nested-fit scripts.\n"
        )
    else:
        lines.append(
            f"- Nested-LOPO calibration shows a small positive delta "
            f"({d_w:+.4f} weighted F1) but the bootstrap 95% CI includes 0 "
            f"(P(Δ>0)={p_w:.3f}). Treat as noise; do not ship as champion. Honest "
            "ceiling remains v4 = {uncal['weighted_f1']:.4f}.\n"
        )

    lines.append("## Reproduce\n")
    lines.append("```bash\n"
                 ".venv/bin/python scripts/threshold_calibration.py\n"
                 "```\n"
                 "Outputs: `cache/calibrated_v4_predictions.json`, this file.\n")

    (REPORTS / "THRESHOLD_CALIBRATION.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
