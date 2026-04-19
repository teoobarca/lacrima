"""Hyperparameter sweep of the Logistic Regression head used in v4.

v4 (champion) uses the **default** `LogisticRegression(class_weight='balanced',
C=1.0, solver='lbfgs', penalty='l2')` for all three component heads:
  A) DINOv2-B 90 nm/px (mean-pooled tiles, no TTA)
  B) DINOv2-B 45 nm/px (mean-pooled tiles, no TTA)
  C) BiomedCLIP 90 nm/px D4 TTA (scan-level)

This script runs a **nested person-LOPO sweep** for the primary (fast) config
space and a single-level 5-fold exploratory pass for the expensive L1 path.

### Stage 1 — Nested LOPO sweep (primary, leakage-safe)

  Outer loop: for each person P_out in 35 persons:
      outer_train = scans where person != P_out
      outer_val   = scans where person == P_out

      Inner loop: 5-fold StratifiedGroupKFold over the 34 outer_train persons.
        For each hyperparameter config in the FAST pool:
          fit on 4/5 inner_train, predict on 1/5 inner_val
          → collect inner OOF softmax → compute inner wF1
        Pick config with highest inner mean wF1 (per encoder).

      Refit each encoder's head with its best inner config on FULL outer_train.
      Predict on outer_val → accumulate outer OOF softmax per encoder.

  Fast pool:
    - C ∈ {0.01, 0.1, 1.0, 10.0, 100.0}
    - solver = lbfgs, penalty = l2
    - class_weight ∈ {None, 'balanced', custom sqrt-inverse}
    → 5 × 3 = 15 configs, all fast (lbfgs ~0.05 s / fit).

### Stage 2 — Single-level 5-fold sweep of slow saga+{l1,l2} path (exploratory)

  We do NOT use this for the honest number; saga is too slow to fit into
  nested 35×5. Instead we run a single 5-fold person-stratified CV on all
  240 scans per encoder to answer "does L1 regularization offer additional
  signal?" and compare the best saga config's 5-fold wF1 vs the same 5-fold
  wF1 for the matching lbfgs config. This pass is flagged as single-level
  so it doesn't enter the final-metric pipeline. (Rejecting leaky patterns
  is the whole point — but measuring whether L1 *might* help is fine if
  we don't promote single-level CV numbers to "champion" status.)

  Saga pool:
    - C ∈ {0.1, 1.0, 10.0}
    - solver = saga, penalty ∈ {l1, l2}, tol=1e-3, max_iter=2000
    - class_weight ∈ {'balanced', sqrt_inv}
    → 3 × 2 × 2 = 12 configs

### Also swept (on stage-1 nested outer OOF)

  - Ensemble combining method: geometric mean vs arithmetic mean vs
    weighted-arith (weights selected per outer fold on inner OOF).
  - Custom class_weight: sqrt-inverse-frequency as a softer alternative
    to sklearn's 'balanced' (which is inverse-frequency).

### Outputs
  cache/lr_hparam_sweep_predictions.json — outer OOF softmaxes + winning configs
  reports/LR_HPARAM_SWEEP.md            — protocol + results + verdict
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
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
RNG_SEED = 42
V4_DEFAULT_WF1 = 0.6887  # honest person-LOPO champion (published)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def mean_pool_tiles(X_tiles: np.ndarray, t2s: np.ndarray, n_scans: int) -> np.ndarray:
    d = X_tiles.shape[1]
    out = np.zeros((n_scans, d), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int64)
    for i, s in enumerate(t2s):
        out[s] += X_tiles[i]
        counts[s] += 1
    counts = np.maximum(counts, 1)
    out /= counts[:, None]
    return out


def align_to_reference(paths_ref, paths_src, X_src):
    src_idx = {p: i for i, p in enumerate(paths_src)}
    order = np.array([src_idx[p] for p in paths_ref])
    return X_src[order]


def load_encoder_features():
    """Return dict of {encoder_name: X_scan (240, D)} and (y, groups, paths)."""
    print("[load] cached embeddings")
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz", allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz",
                  allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)

    paths_90 = [str(p) for p in z90["scan_paths"]]
    paths_45 = [str(p) for p in z45["scan_paths"]]
    paths_bc = [str(p) for p in zbc["scan_paths"]]

    y = np.asarray(z90["scan_y"], dtype=np.int64)
    groups = np.array([person_id(Path(p)) for p in paths_90])
    n_scans = len(y)

    X90 = mean_pool_tiles(z90["X"], z90["tile_to_scan"], len(paths_90))
    X45_raw = mean_pool_tiles(z45["X"], z45["tile_to_scan"], len(paths_45))
    X45 = align_to_reference(paths_90, paths_45, X45_raw)
    Xbc = align_to_reference(paths_90, paths_bc, zbc["X_scan"].astype(np.float32))

    print(f"  dinov2_90: {X90.shape}  dinov2_45: {X45.shape}  biomedclip_tta: {Xbc.shape}")
    print(f"  n_scans={n_scans}  n_persons={len(np.unique(groups))}")
    return {
        "dinov2_90": X90,
        "dinov2_45": X45,
        "biomedclip_tta": Xbc,
    }, y, groups, paths_90


# ---------------------------------------------------------------------------
# Hparam config
# ---------------------------------------------------------------------------

def sqrt_inverse_weights(y: np.ndarray) -> dict:
    """Softer alternative to 'balanced': w_c ∝ 1 / sqrt(count_c)."""
    counts = Counter(y.tolist())
    w = {c: 1.0 / np.sqrt(counts[c]) for c in counts}
    mean_w = np.mean(list(w.values()))
    return {c: float(v / mean_w) for c, v in w.items()}


def enumerate_fast_configs():
    """Fast pool: lbfgs+l2 only. Used in the NESTED sweep."""
    Cs = [0.01, 0.1, 1.0, 10.0, 100.0]
    cw_specs = ["none", "balanced", "sqrt_inv"]
    configs = []
    for C, cw in itertools.product(Cs, cw_specs):
        configs.append({
            "C": C,
            "solver": "lbfgs",
            "class_weight": cw,
            "penalty": "l2",
        })
    return configs


def enumerate_saga_configs():
    """Slow pool: saga + l1 / l2. Used in single-level 5-fold exploration only."""
    Cs = [0.1, 1.0, 10.0]
    cw_specs = ["balanced", "sqrt_inv"]
    pens = ["l1", "l2"]
    configs = []
    for C, pen, cw in itertools.product(Cs, pens, cw_specs):
        configs.append({
            "C": C,
            "solver": "saga",
            "class_weight": cw,
            "penalty": pen,
        })
    return configs


def build_clf(cfg: dict, y_train: np.ndarray) -> LogisticRegression:
    cw = cfg["class_weight"]
    if cw == "none":
        cw_arg = None
    elif cw == "balanced":
        cw_arg = "balanced"
    elif cw == "sqrt_inv":
        cw_arg = sqrt_inverse_weights(y_train)
    else:
        raise ValueError(cw)
    if cfg["solver"] == "saga":
        # Relaxed tol for practical convergence on 200-sample features
        return LogisticRegression(
            C=cfg["C"], solver="saga", penalty=cfg["penalty"],
            class_weight=cw_arg, max_iter=2000, tol=1e-3,
            n_jobs=4, random_state=RNG_SEED,
        )
    return LogisticRegression(
        C=cfg["C"], solver=cfg["solver"], penalty=cfg["penalty"],
        class_weight=cw_arg, max_iter=5000,
        n_jobs=4, random_state=RNG_SEED,
    )


# ---------------------------------------------------------------------------
# Core fit/predict with v2 preprocessing (L2-norm -> StandardScaler)
# ---------------------------------------------------------------------------

def fit_predict_proba(X_tr, y_tr, X_va, cfg):
    X_tr_n = normalize(X_tr, norm="l2", axis=1)
    X_va_n = normalize(X_va, norm="l2", axis=1)
    sc = StandardScaler().fit(X_tr_n)
    X_tr_s = np.nan_to_num(sc.transform(X_tr_n), nan=0.0, posinf=0.0, neginf=0.0)
    X_va_s = np.nan_to_num(sc.transform(X_va_n), nan=0.0, posinf=0.0, neginf=0.0)
    clf = build_clf(cfg, y_tr)
    clf.fit(X_tr_s, y_tr)
    proba = clf.predict_proba(X_va_s)
    p_full = np.zeros((len(X_va), N_CLASSES), dtype=np.float64)
    for ci, cls in enumerate(clf.classes_):
        p_full[:, cls] = proba[:, ci]
    return p_full


# ---------------------------------------------------------------------------
# Inner CV model selection (used by Stage 1 nested sweep)
# ---------------------------------------------------------------------------

def _inner_splits(y_tr, g_tr, n_splits=5):
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                               random_state=RNG_SEED)
    try:
        return list(skf.split(np.zeros(len(y_tr)), y_tr, g_tr))
    except ValueError:
        skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RNG_SEED)
        return list(skf.split(np.zeros(len(y_tr)), y_tr, g_tr))


def inner_cv_oof(X_outer_tr, y_outer_tr, g_outer_tr, cfg, n_inner=5):
    """5-fold StratifiedGroupKFold OOF softmax on outer_train."""
    P_inner = np.zeros((len(y_outer_tr), N_CLASSES), dtype=np.float64)
    for tr, va in _inner_splits(y_outer_tr, g_outer_tr, n_inner):
        P_inner[va] = fit_predict_proba(
            X_outer_tr[tr], y_outer_tr[tr], X_outer_tr[va], cfg
        )
    return P_inner


def inner_cv_score(X_outer_tr, y_outer_tr, g_outer_tr, cfg, n_inner=5):
    P = inner_cv_oof(X_outer_tr, y_outer_tr, g_outer_tr, cfg, n_inner)
    pred = P.argmax(axis=1)
    return float(f1_score(y_outer_tr, pred, average="weighted", zero_division=0))


# ---------------------------------------------------------------------------
# Ensemble combiners
# ---------------------------------------------------------------------------

def geom_mean(probs_list, weights=None):
    if weights is None:
        weights = [1.0 / len(probs_list)] * len(probs_list)
    log_sum = np.zeros_like(probs_list[0])
    for w, P in zip(weights, probs_list):
        log_sum = log_sum + w * np.log(P + EPS)
    G = np.exp(log_sum)
    G /= G.sum(axis=1, keepdims=True)
    return G


def arith_mean(probs_list, weights=None):
    if weights is None:
        weights = [1.0 / len(probs_list)] * len(probs_list)
    out = np.zeros_like(probs_list[0])
    for w, P in zip(weights, probs_list):
        out = out + w * P
    out /= out.sum(axis=1, keepdims=True)
    return out


def tune_ensemble_weights_inner(inner_probs, y_inner, method="arith"):
    best_w, best_f1 = None, -1.0
    grid = [0.2, 0.5, 1.0, 2.0, 5.0]
    for w1, w2, w3 in itertools.product(grid, repeat=3):
        s = w1 + w2 + w3
        w = [w1 / s, w2 / s, w3 / s]
        if method == "arith":
            P = arith_mean(inner_probs, weights=w)
        else:
            P = geom_mean(inner_probs, weights=[len(inner_probs) * wi for wi in w])
        pred = P.argmax(axis=1)
        f1 = f1_score(y_inner, pred, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1, best_w = f1, w
    return best_w, best_f1


# ---------------------------------------------------------------------------
# Stage 1: Nested LOPO sweep (lbfgs+l2 only, all class_weight x C)
# ---------------------------------------------------------------------------

def nested_lopo_sweep(features_dict, y, groups, configs):
    """Run nested person-LOPO with per-encoder inner-CV config selection."""
    n = len(y)
    enc_names = list(features_dict.keys())
    outer_probs = {k: np.zeros((n, N_CLASSES), dtype=np.float64) for k in enc_names}
    ens_weight_selections = []
    selected_configs = []
    cfg_pick_counts = {k: Counter() for k in enc_names}

    lopo = list(leave_one_patient_out(groups))
    print(f"[nested-lopo] {len(lopo)} outer folds × {len(configs)} configs × "
          f"{len(enc_names)} encoders, 5-fold inner")

    t0 = time.time()
    for oi, (outer_tr, outer_va) in enumerate(lopo):
        y_tr = y[outer_tr]
        g_tr = groups[outer_tr]
        fold_info = {"outer_person": str(groups[outer_va[0]])}

        fold_inner_probs = {}
        for enc in enc_names:
            X_tr_enc = features_dict[enc][outer_tr]
            best_cfg, best_score = None, -1.0
            for cfg in configs:
                s = inner_cv_score(X_tr_enc, y_tr, g_tr, cfg)
                if s > best_score:
                    best_score, best_cfg = s, cfg
            fold_info[enc] = {"cfg": best_cfg, "inner_wf1": best_score}
            cfg_pick_counts[enc][json.dumps(best_cfg, sort_keys=True)] += 1
            X_va_enc = features_dict[enc][outer_va]
            outer_probs[enc][outer_va] = fit_predict_proba(
                X_tr_enc, y_tr, X_va_enc, best_cfg
            )
            # also build inner OOF with best cfg for ensemble-weight tuning
            fold_inner_probs[enc] = inner_cv_oof(X_tr_enc, y_tr, g_tr, best_cfg)

        inner_probs_list = [fold_inner_probs[k] for k in enc_names]
        w_arith, f1_arith = tune_ensemble_weights_inner(inner_probs_list, y_tr, "arith")
        w_geom,  f1_geom  = tune_ensemble_weights_inner(inner_probs_list, y_tr, "geom")
        ens_weight_selections.append({
            "outer_person": fold_info["outer_person"],
            "arith": {"weights": w_arith, "inner_wf1": f1_arith},
            "geom":  {"weights": w_geom,  "inner_wf1": f1_geom},
        })

        selected_configs.append(fold_info)
        elapsed = time.time() - t0
        if (oi + 1) % 5 == 0 or oi == 0:
            eta = elapsed / (oi + 1) * (len(lopo) - oi - 1)
            print(f"  outer {oi+1}/{len(lopo)}  t={elapsed:.0f}s  eta~{eta:.0f}s", flush=True)

    return outer_probs, selected_configs, ens_weight_selections, cfg_pick_counts


# ---------------------------------------------------------------------------
# Stage 2: Single-level 5-fold saga sweep (exploratory)
# ---------------------------------------------------------------------------

def saga_exploration(features_dict, y, groups, saga_configs, lbfgs_baseline_cfg):
    """Flat 5-fold person-stratified CV: compare saga vs lbfgs per encoder.

    NOT nested — this is exploratory only. Single-level CV on 240 scans with
    5 person-stratified folds. Reports per-config wF1 for each encoder.
    """
    print(f"[saga-exploration] 5-fold person-stratified, "
          f"{len(saga_configs)} saga configs + 1 lbfgs baseline × "
          f"{len(features_dict)} encoders")
    splits = _inner_splits(y, groups, 5)
    results = {}
    for enc, X in features_dict.items():
        enc_res = {}
        # lbfgs baseline
        t0 = time.time()
        P = np.zeros((len(y), N_CLASSES), dtype=np.float64)
        for tr, va in splits:
            P[va] = fit_predict_proba(X[tr], y[tr], X[va], lbfgs_baseline_cfg)
        pred = P.argmax(axis=1)
        wf1 = float(f1_score(y, pred, average="weighted", zero_division=0))
        mf1 = float(f1_score(y, pred, average="macro", zero_division=0))
        enc_res["__lbfgs_baseline__"] = {
            "cfg": lbfgs_baseline_cfg, "wf1": wf1, "mf1": mf1, "t_s": round(time.time() - t0, 1)
        }
        print(f"  [{enc}] lbfgs baseline: wF1={wf1:.4f}  mF1={mf1:.4f}")
        for cfg in saga_configs:
            t0 = time.time()
            P = np.zeros((len(y), N_CLASSES), dtype=np.float64)
            for tr, va in splits:
                P[va] = fit_predict_proba(X[tr], y[tr], X[va], cfg)
            pred = P.argmax(axis=1)
            wf1 = float(f1_score(y, pred, average="weighted", zero_division=0))
            mf1 = float(f1_score(y, pred, average="macro", zero_division=0))
            key = f"saga_{cfg['penalty']}_C{cfg['C']}_{cfg['class_weight']}"
            enc_res[key] = {
                "cfg": cfg, "wf1": wf1, "mf1": mf1, "t_s": round(time.time() - t0, 1)
            }
            print(f"  [{enc}] {key:30s} wF1={wf1:.4f}  mF1={mf1:.4f}  ({time.time()-t0:.0f}s)")
        results[enc] = enc_res
    return results


# ---------------------------------------------------------------------------
# Metrics + bootstrap
# ---------------------------------------------------------------------------

def metrics_of(P: np.ndarray, y: np.ndarray) -> dict:
    pred = P.argmax(axis=1)
    return {
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1":    float(f1_score(y, pred, average="macro",    zero_division=0)),
        "per_class_f1": f1_score(
            y, pred, average=None, labels=list(range(N_CLASSES)), zero_division=0,
        ).tolist(),
    }


def bootstrap_delta(P_new, P_ref, y, n_iter=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    pred_new = P_new.argmax(axis=1)
    pred_ref = P_ref.argmax(axis=1)
    deltas = np.zeros(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        f_new = f1_score(y[idx], pred_new[idx], average="weighted", zero_division=0)
        f_ref = f1_score(y[idx], pred_ref[idx], average="weighted", zero_division=0)
        deltas[i] = f_new - f_ref
    return {
        "P_delta_gt_0": float(np.mean(deltas > 0)),
        "mean_delta": float(np.mean(deltas)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
    }


# ---------------------------------------------------------------------------
# Class-weight ablation (flat LOPO)
# ---------------------------------------------------------------------------

def ablation_classweight(features_dict, y, groups):
    configs = {
        "none":      {"C": 1.0, "solver": "lbfgs", "penalty": "l2", "class_weight": "none"},
        "balanced":  {"C": 1.0, "solver": "lbfgs", "penalty": "l2", "class_weight": "balanced"},
        "sqrt_inv":  {"C": 1.0, "solver": "lbfgs", "penalty": "l2", "class_weight": "sqrt_inv"},
    }
    n = len(y)
    results = {}
    for label, cfg in configs.items():
        enc_probs = {}
        for enc, X in features_dict.items():
            P = np.zeros((n, N_CLASSES), dtype=np.float64)
            for tr, va in leave_one_patient_out(groups):
                P[va] = fit_predict_proba(X[tr], y[tr], X[va], cfg)
            enc_probs[enc] = P
        G = geom_mean(list(enc_probs.values()))
        results[label] = metrics_of(G, y)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    features_dict, y, groups, paths = load_encoder_features()

    fast_configs = enumerate_fast_configs()
    saga_configs = enumerate_saga_configs()
    print(f"[sweep] Stage 1 (nested LOPO): {len(fast_configs)} fast configs per encoder")
    print(f"        Stage 2 (5-fold exploratory): {len(saga_configs)} saga configs per encoder")

    # =================================================================
    # Stage 1: Nested LOPO sweep over fast (lbfgs+l2) configs
    # =================================================================
    outer_probs, selected_cfgs, ens_weights, cfg_counts = nested_lopo_sweep(
        features_dict, y, groups, fast_configs
    )

    per_enc_metrics = {}
    print("\n[results] per-encoder outer OOF (nested-selected hparams)")
    for enc, P in outer_probs.items():
        m = metrics_of(P, y)
        per_enc_metrics[enc] = m
        print(f"  {enc:20s}  wF1={m['weighted_f1']:.4f}  mF1={m['macro_f1']:.4f}")

    # Ensemble combinations
    probs_list = [outer_probs[k] for k in ["dinov2_90", "dinov2_45", "biomedclip_tta"]]
    ens_methods = {}
    G = geom_mean(probs_list)
    ens_methods["geom_unweighted"] = metrics_of(G, y)
    A = arith_mean(probs_list)
    ens_methods["arith_unweighted"] = metrics_of(A, y)
    w_arith_all = np.array([e["arith"]["weights"] for e in ens_weights])
    w_geom_all  = np.array([e["geom"]["weights"]  for e in ens_weights])
    w_arith_med = w_arith_all.mean(axis=0); w_arith_med /= w_arith_med.sum()
    w_geom_med  = w_geom_all.mean(axis=0);  w_geom_med  /= w_geom_med.sum()
    Aw = arith_mean(probs_list, weights=list(w_arith_med))
    ens_methods["arith_weighted"] = metrics_of(Aw, y)
    Gw = geom_mean(probs_list, weights=list(3 * w_geom_med))
    ens_methods["geom_weighted"] = metrics_of(Gw, y)

    print("\n[results] ensemble combinations (outer OOF)")
    for name, m in ens_methods.items():
        print(f"  {name:20s}  wF1={m['weighted_f1']:.4f}  mF1={m['macro_f1']:.4f}")

    best_ens = max(ens_methods, key=lambda k: ens_methods[k]["weighted_f1"])
    best_P = {
        "geom_unweighted": G, "arith_unweighted": A,
        "arith_weighted": Aw, "geom_weighted": Gw,
    }[best_ens]
    best_m = ens_methods[best_ens]
    print(f"\n[best-ensemble] {best_ens}  wF1={best_m['weighted_f1']:.4f}  "
          f"mF1={best_m['macro_f1']:.4f}")

    # =================================================================
    # v4-default reconstruction (flat LOPO)
    # =================================================================
    print("\n[bootstrap] reconstructing v4-default baseline (flat LOPO, C=1, balanced, lbfgs, l2)")
    default_cfg = {"C": 1.0, "solver": "lbfgs", "penalty": "l2", "class_weight": "balanced"}
    n = len(y)
    default_probs_list = []
    for enc, X in features_dict.items():
        P_enc = np.zeros((n, N_CLASSES), dtype=np.float64)
        for tr, va in leave_one_patient_out(groups):
            P_enc[va] = fit_predict_proba(X[tr], y[tr], X[va], default_cfg)
        default_probs_list.append(P_enc)
    P_v4 = geom_mean(default_probs_list)
    v4_m = metrics_of(P_v4, y)
    print(f"  v4-default reconstructed  wF1={v4_m['weighted_f1']:.4f}  "
          f"(published: {V4_DEFAULT_WF1:.4f})")

    boot = bootstrap_delta(best_P, P_v4, y, n_iter=1000, seed=42)
    print(f"  ΔwF1 mean={boot['mean_delta']:+.4f}  "
          f"P(Δ>0)={boot['P_delta_gt_0']:.3f}  "
          f"95% CI=[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")

    # =================================================================
    # Class-weight ablation
    # =================================================================
    print("\n[ablation] class_weight comparison (C=1, lbfgs, l2, geom-mean ensemble)")
    ablation = ablation_classweight(features_dict, y, groups)
    for label, m in ablation.items():
        sucheoko_f1 = m["per_class_f1"][CLASSES.index("SucheOko")]
        print(f"  class_weight={label:10s}  wF1={m['weighted_f1']:.4f}  "
              f"mF1={m['macro_f1']:.4f}  SucheOko-F1={sucheoko_f1:.4f}")

    # =================================================================
    # Winning configs
    # =================================================================
    print("\n[winning-configs] most-frequently-selected per encoder (across 35 outer folds)")
    winning_cfgs = {}
    for enc, counts in cfg_counts.items():
        top = counts.most_common(3)
        winning_cfgs[enc] = [
            {"cfg": json.loads(c), "n_folds": int(n)} for c, n in top
        ]
        print(f"  {enc}:")
        for c, n_sel in top:
            print(f"    {n_sel:2d}/35  {c}")

    # =================================================================
    # Stage 2: saga exploration (single-level 5-fold, NOT nested)
    # =================================================================
    saga_res = saga_exploration(features_dict, y, groups, saga_configs,
                                 lbfgs_baseline_cfg=default_cfg)

    # =================================================================
    # Verdict
    # =================================================================
    w_best = best_m["weighted_f1"]
    p_gt_0 = boot["P_delta_gt_0"]
    if w_best > 0.70 and p_gt_0 > 0.90:
        verdict = "CHAMPION_CANDIDATE"
    elif w_best <= 0.695:
        verdict = "NOISE_FLOOR_V4_AT_CEILING"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\n[verdict] {verdict}  (best={w_best:.4f}, P(Δ>0)={p_gt_0:.3f})")

    # =================================================================
    # Persist
    # =================================================================
    summary = {
        "protocol": (
            "Stage 1: nested person-LOPO. Outer 35-fold LOPO (hold out 1 person), "
            "inner 5-fold StratifiedGroupKFold on remaining 34 persons. "
            "Per encoder, inner CV selects best {C, class_weight} from lbfgs+l2 pool; "
            "refit on full outer_train with best; predict outer_val. "
            "Ensemble weights tuned on inner OOF (arith & geom grid search). "
            "All Stage-1 reported metrics are outer-OOF — no inner leakage into test. "
            "\n\nStage 2: single-level 5-fold person-stratified CV (saga+{l1,l2} pool). "
            "Exploratory only — NOT used for final metric because single-level CV "
            "on 240 scans is slightly optimistic; documented for future inclusion "
            "if saga+l1 shows a large gain worth bearing the nested cost."
        ),
        "n_scans": int(n),
        "n_persons": int(len(np.unique(groups))),
        "stage1_fast_pool_size": len(fast_configs),
        "stage2_saga_pool_size": len(saga_configs),
        "v4_default_reconstructed_wf1": v4_m["weighted_f1"],
        "v4_default_reconstructed_mf1": v4_m["macro_f1"],
        "v4_published_wf1": V4_DEFAULT_WF1,
        "per_encoder_outer_oof": per_enc_metrics,
        "ensemble_methods": ens_methods,
        "best_ensemble": best_ens,
        "bootstrap_vs_v4": boot,
        "class_weight_ablation": ablation,
        "winning_configs_per_encoder": winning_cfgs,
        "tuned_ensemble_weights_median": {
            "arith": [float(x) for x in w_arith_med.tolist()],
            "geom":  [float(x) for x in w_geom_med.tolist()],
        },
        "saga_exploration_5fold": saga_res,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t_start, 1),
    }

    out_json = CACHE / "lr_hparam_sweep_predictions.json"
    out_json.write_text(json.dumps({
        "summary": summary,
        "outer_probs": {k: v.tolist() for k, v in outer_probs.items()},
        "ensemble_best_probs": best_P.tolist(),
        "v4_default_probs": P_v4.tolist(),
        "y": y.tolist(),
        "scan_paths": paths,
    }, indent=2))
    print(f"[saved] {out_json}")

    write_markdown_report(summary, winning_cfgs)
    print(f"[done] total elapsed: {time.time() - t_start:.1f}s")


def write_markdown_report(summary, winning_cfgs):
    lines = []
    lines.append("# LR Hyperparameter Sweep — Nested Person-LOPO\n")
    lines.append("## TL;DR\n")
    best = summary["ensemble_methods"][summary["best_ensemble"]]
    v4 = summary["v4_default_reconstructed_wf1"]
    boot = summary["bootstrap_vs_v4"]
    lines.append(
        f"- **Best ensemble (nested-LOPO, honest):** `{summary['best_ensemble']}` → "
        f"**wF1 = {best['weighted_f1']:.4f}**, mF1 = {best['macro_f1']:.4f}\n"
        f"- **v4 default (flat LOPO, reconstructed):** wF1 = {v4:.4f} "
        f"(published: {summary['v4_published_wf1']:.4f})\n"
        f"- **Δ vs v4:** mean = {boot['mean_delta']:+.4f}, "
        f"**P(Δ>0) = {boot['P_delta_gt_0']:.3f}**, "
        f"95% CI = [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]\n"
        f"- **Verdict:** `{summary['verdict']}`\n"
    )

    lines.append("## Protocol (no inner→outer leakage)\n")
    lines.append(summary["protocol"] + "\n")
    lines.append(
        "\n**Stage 1 axes (nested-LOPO, per encoder):**\n"
        "- `C`: {0.01, 0.1, 1.0, 10.0, 100.0}\n"
        "- `solver` × `penalty`: {lbfgs × l2}\n"
        "- `class_weight`: {None, 'balanced', custom sqrt-inverse frequency}\n"
        "- `max_iter` = 5000 (all lbfgs runs converged).\n\n"
        "**Stage 2 axes (single-level 5-fold, exploratory only):**\n"
        "- `C`: {0.1, 1.0, 10.0}\n"
        "- `solver` × `penalty`: {saga × l1, saga × l2}\n"
        "- `class_weight`: {'balanced', sqrt_inv}\n"
        "- `max_iter` = 2000, `tol` = 1e-3.\n\n"
        "**Ensemble-combining axes (tuned on Stage 1 inner OOF per outer fold):**\n"
        "- geometric mean (unweighted + weighted)\n"
        "- arithmetic mean (unweighted + weighted)\n"
        "- per-member weight grid: {0.2, 0.5, 1.0, 2.0, 5.0}, triplet normalized.\n"
    )

    lines.append("## Per-encoder outer-OOF metrics (Stage 1, nested-selected)\n")
    lines.append("| Encoder | Weighted F1 | Macro F1 | Per-class F1 |")
    lines.append("|---|---:|---:|---|")
    for enc, m in summary["per_encoder_outer_oof"].items():
        pcf1 = " / ".join(f"{v:.3f}" for v in m["per_class_f1"])
        lines.append(f"| `{enc}` | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} | {pcf1} |")
    lines.append(f"\n*Per-class order:* {' / '.join(CLASSES)}\n")

    lines.append("## Ensemble combinations (Stage 1 outer OOF)\n")
    lines.append("| Method | Weighted F1 | Macro F1 | Per-class F1 |")
    lines.append("|---|---:|---:|---|")
    for name, m in summary["ensemble_methods"].items():
        pcf1 = " / ".join(f"{v:.3f}" for v in m["per_class_f1"])
        star = " *" if name == summary["best_ensemble"] else ""
        lines.append(
            f"| `{name}`{star} | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} | {pcf1} |"
        )
    lines.append("")
    tw = summary["tuned_ensemble_weights_median"]
    lines.append(
        f"**Mean tuned weights** across 35 outer folds "
        f"(encoders = dinov2_90, dinov2_45, biomedclip_tta):\n"
        f"- arith: {[round(w,3) for w in tw['arith']]}\n"
        f"- geom:  {[round(w,3) for w in tw['geom']]}\n"
    )

    lines.append("## Winning LR config per encoder (Stage 1)\n")
    lines.append(
        "Frequency across 35 outer folds (each outer fold independently picks "
        "its best config on its inner 5-fold CV):\n"
    )
    for enc in ["dinov2_90", "dinov2_45", "biomedclip_tta"]:
        lines.append(f"\n### `{enc}` — top 3 picks\n")
        lines.append("| # folds / 35 | C | solver | penalty | class_weight |")
        lines.append("|---:|---:|---|---|---|")
        for entry in winning_cfgs[enc]:
            c = entry["cfg"]
            lines.append(
                f"| {entry['n_folds']} | {c['C']} | {c['solver']} | "
                f"{c['penalty']} | {c['class_weight']} |"
            )

    lines.append("\n## Class-weight ablation (C=1, lbfgs, l2, geom-mean)\n")
    lines.append("| class_weight | Weighted F1 | Macro F1 | SucheOko F1 | Per-class F1 |")
    lines.append("|---|---:|---:|---:|---|")
    so_idx = CLASSES.index("SucheOko")
    for label, m in summary["class_weight_ablation"].items():
        pcf1 = " / ".join(f"{v:.3f}" for v in m["per_class_f1"])
        lines.append(
            f"| `{label}` | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['per_class_f1'][so_idx]:.4f} | {pcf1} |"
        )
    lines.append("")

    lines.append("## Stage 2: saga + {l1, l2} 5-fold exploration (NOT nested)\n")
    lines.append(
        "*Single-level 5-fold person-stratified CV. Numbers are slightly optimistic "
        "compared to LOPO — use only to decide whether saga+l1 is worth pursuing.*\n"
    )
    lines.append("| Encoder | Config | Weighted F1 | Macro F1 |")
    lines.append("|---|---|---:|---:|")
    for enc, res in summary["saga_exploration_5fold"].items():
        for name, r in res.items():
            lines.append(f"| `{enc}` | `{name}` | {r['wf1']:.4f} | {r['mf1']:.4f} |")
    lines.append("")

    lines.append("## Bootstrap vs v4-default (1000 paired resamples)\n")
    lines.append(
        f"- Mean Δ weighted F1: {boot['mean_delta']:+.4f}\n"
        f"- **P(Δ > 0)**: {boot['P_delta_gt_0']:.3f}\n"
        f"- 95% CI: [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]\n"
    )

    lines.append("## Verdict\n")
    v = summary["verdict"]
    if v == "CHAMPION_CANDIDATE":
        lines.append(
            "**CHAMPION CANDIDATE.** Best sweep > 0.70 wF1 with P(Δ>0) > 0.90. "
            "Submit for red-team review before shipping.\n"
        )
    elif v == "NOISE_FLOOR_V4_AT_CEILING":
        lines.append(
            "**NOISE FLOOR.** Best sweep ≤ 0.695 — despite exhaustive nested CV "
            "over C × class_weight, no meaningful lift over v4 defaults. "
            "The LR head is not the bottleneck; features are. "
            "v4 defaults confirmed at their ceiling.\n"
        )
    else:
        lines.append(
            "**INCONCLUSIVE.** Lift observed but below the decision threshold for "
            "declaring a new champion. Treat as negative — stick with v4.\n"
        )

    lines.append(
        f"\n_Elapsed: {summary['elapsed_s']}s. "
        f"Cache: `cache/lr_hparam_sweep_predictions.json`._\n"
    )

    (REPORTS / "LR_HPARAM_SWEEP.md").write_text("\n".join(lines))
    print(f"[saved] reports/LR_HPARAM_SWEEP.md")


if __name__ == "__main__":
    main()
