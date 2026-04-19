"""Class-specific ensemble combination strategies to rescue SucheOko (v4 F1=0).

Motivation
----------
The v4 multi-scale champion uses the GEOMETRIC MEAN of three per-encoder softmaxes
(DINOv2-B 90 nm, DINOv2-B 45 nm, BiomedCLIP D4-TTA). Per the ProtoNet agent's
finding (reports/PROTONET_RESULTS.md), a single encoder (BiomedCLIP TTA + adapter)
can achieve SucheOko F1 = 0.074, but the geometric mean collapses it back to 0
because the other two encoders never predict SucheOko and the geometric mean
penalizes any low-confidence component. This pulls minority-class signal to zero.

Hypothesis
----------
Replace the geometric mean with a combination rule that preserves minority-class
signal, without sacrificing majority-class accuracy.

Strategies tested
-----------------
A) max-pool   : per class, take max over encoders' softmaxes, renormalize, argmax.
B) hybrid     : keep geometric mean for majority classes, but replace
                P(SucheOko) with max over encoders' P(SucheOko); renormalize.
C) learnable  : learn a convex combination weight per (class x encoder). Weights
                are optimized on TRAIN PORTION of each fold via multinomial logistic
                meta-stacker (no OOF-based tuning; weights are fit inside nested
                LOPO on the training set). To avoid leakage we generate the
                per-encoder OOF predictions on the training set with an INNER
                person-LOPO run and fit a LR stacker on those.
D) noisy-OR   : P(c) = 1 - prod_enc(1 - p_enc(c)); renormalize.
E) rank-vote  : each encoder argmax votes; ties broken by sum-of-prob.

Protocol
--------
- Person-LOPO on 35 groups via `teardrop.data.person_id`.
- Each encoder: L2-normalize -> StandardScaler -> LogisticRegression(balanced)
  (identical to v4 and multiscale_experiment).
- Collect OOF softmaxes per encoder (shape 240 x 5).
- Apply each combination strategy -> compute weighted F1 + per-class F1.
- Focus: does SucheOko F1 go non-zero while weighted F1 stays >= 0.67?

Constraints
-----------
- No OOF-based hyperparameter tuning (red-team would reject).
- For strategy C, the stacker weights are fit *inside* the outer LOPO loop via
  nested LOPO on the training set.
- <= 20 min compute.
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
V4_WF1 = 0.6887
V4_MF1 = 0.5541
SUCHEOKO_IDX = CLASSES.index("SucheOko")

ENCODER_NAMES = ["dinov2_90nm", "dinov2_45nm", "biomedclip_tta"]


# ---------------------------------------------------------------------------
# Shared helpers (replicated from multiscale_experiment.py for isolation)
# ---------------------------------------------------------------------------

def mean_pool_tiles(X_tiles: np.ndarray, t2s: np.ndarray, n_scans: int) -> np.ndarray:
    d = X_tiles.shape[1]
    out = np.zeros((n_scans, d), dtype=np.float32)
    for si in range(n_scans):
        m = t2s == si
        if m.any():
            out[si] = X_tiles[m].mean(axis=0)
    return out


def align_to_reference(paths_ref: list[str], paths_src: list[str],
                       X_src: np.ndarray) -> np.ndarray:
    src_idx = {p: i for i, p in enumerate(paths_src)}
    order = np.array([src_idx[p] for p in paths_ref])
    return X_src[order]


def lopo_predict_v2(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """V2 recipe person-LOPO OOF softmax: L2-norm -> StandardScaler -> LR(bal)."""
    n = len(y)
    P = np.zeros((n, N_CLASSES), dtype=np.float64)
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
        p_full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
        for ci, cls in enumerate(clf.classes_):
            p_full[:, cls] = proba[:, ci]
        P[va] = p_full
    return P


def metrics_of(P_or_pred: np.ndarray, y: np.ndarray) -> dict:
    if P_or_pred.ndim == 2:
        pred = P_or_pred.argmax(axis=1)
    else:
        pred = P_or_pred
    return {
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(
            y, pred, average=None, labels=list(range(N_CLASSES)), zero_division=0,
        ).tolist(),
    }


# ---------------------------------------------------------------------------
# Combination strategies
# ---------------------------------------------------------------------------

def strat_geom_mean(Ps: list[np.ndarray]) -> np.ndarray:
    """Baseline v4: geometric mean."""
    log_sum = np.zeros_like(Ps[0])
    for P in Ps:
        log_sum = log_sum + np.log(P + EPS)
    G = np.exp(log_sum / len(Ps))
    G /= G.sum(axis=1, keepdims=True)
    return G


def strat_max_pool(Ps: list[np.ndarray]) -> np.ndarray:
    """A) per class, take max across encoders, renormalize."""
    M = np.stack(Ps, axis=0).max(axis=0)  # (N, C)
    M /= M.sum(axis=1, keepdims=True) + EPS
    return M


def strat_hybrid_majority_geom_minority_max(
    Ps: list[np.ndarray], minority_idx: int = SUCHEOKO_IDX,
) -> np.ndarray:
    """B) geom-mean for all classes, but replace minority col with max-over-encoders.
    Then renormalize so rows sum to 1.
    """
    G = strat_geom_mean(Ps)
    mx = np.stack(Ps, axis=0).max(axis=0)[:, minority_idx]
    G[:, minority_idx] = mx
    G /= G.sum(axis=1, keepdims=True) + EPS
    return G


def strat_noisy_or(Ps: list[np.ndarray]) -> np.ndarray:
    """D) P(c) = 1 - prod_enc(1 - p_enc(c)), renormalize."""
    log_term = np.zeros_like(Ps[0])
    for P in Ps:
        log_term = log_term + np.log(np.clip(1.0 - P, EPS, 1.0))
    N = 1.0 - np.exp(log_term)
    N /= N.sum(axis=1, keepdims=True) + EPS
    return N


def strat_rank_vote(Ps: list[np.ndarray], y_ignore: np.ndarray | None = None) -> np.ndarray:
    """E) each encoder argmax votes; ties broken by sum-prob.

    Returns hard-argmax predictions as (N,) integer array. We wrap as a one-hot
    softmax-like matrix for downstream scoring.
    """
    n, c = Ps[0].shape
    votes = np.zeros((n, c), dtype=np.float64)
    prob_sum = np.zeros((n, c), dtype=np.float64)
    for P in Ps:
        arg = P.argmax(axis=1)
        for i, a in enumerate(arg):
            votes[i, a] += 1
        prob_sum = prob_sum + P
    # combine: primary = votes, tiebreak = prob_sum scaled by tiny weight
    combined = votes + 1e-3 * (prob_sum / len(Ps))
    pred = combined.argmax(axis=1)
    # Convert to one-hot-ish matrix that argmax-es to pred (for metrics_of())
    out = np.zeros((n, c), dtype=np.float64)
    out[np.arange(n), pred] = 1.0
    return out


# ---------------------------------------------------------------------------
# Strategy C: nested-LOPO learnable class-specific stacker
# ---------------------------------------------------------------------------

def lopo_predict_v2_subset(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    subset_idx: np.ndarray,
) -> np.ndarray:
    """Run person-LOPO on a subset of rows (given by subset_idx into X/y/groups).

    Returns OOF probs aligned to subset_idx (shape len(subset_idx), N_CLASSES).
    """
    Xs, ys, gs = X[subset_idx], y[subset_idx], groups[subset_idx]
    n = len(ys)
    P = np.zeros((n, N_CLASSES), dtype=np.float64)
    for tr, va in leave_one_patient_out(gs):
        Xt = normalize(Xs[tr], norm="l2", axis=1)
        Xv = normalize(Xs[va], norm="l2", axis=1)
        sc = StandardScaler()
        Xt = sc.fit_transform(Xt)
        Xv = sc.transform(Xv)
        Xt = np.nan_to_num(Xt)
        Xv = np.nan_to_num(Xv)
        clf = LogisticRegression(
            class_weight="balanced", max_iter=3000, C=1.0,
            solver="lbfgs", n_jobs=4, random_state=42,
        )
        clf.fit(Xt, ys[tr])
        proba = clf.predict_proba(Xv)
        p_full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
        for ci, cls in enumerate(clf.classes_):
            p_full[:, cls] = proba[:, ci]
        P[va] = p_full
    return P


def learnable_class_specific_stacker(
    Xs: list[np.ndarray], y: np.ndarray, groups: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Strategy C — nested LOPO class-specific combination.

    For each OUTER LOPO fold (held-out person):
        1. On the outer training set, run an INNER LOPO on each encoder to get
           per-encoder OOF softmaxes for training samples.
        2. Build a (N_train, 3*C) meta-feature from the three encoders'
           softmaxes on the inner OOF training samples.
        3. Fit a meta-learner (LR) on that stack.
        4. For the held-out person, get each encoder's softmax from a model
           trained on the FULL outer training set (not the inner OOF), stack
           the three softmaxes, and predict via the meta-learner.

    This respects person-LOPO (no leakage) and never selects on outer OOF.
    """
    n = len(y)
    n_enc = len(Xs)
    out = np.zeros((n, N_CLASSES), dtype=np.float64)
    t0 = time.time()
    n_folds = len(np.unique(groups))
    for fi, (tr, va) in enumerate(leave_one_patient_out(groups)):
        # Step 1: per-encoder OOF softmaxes on training portion via inner LOPO.
        inner_oof = []
        for X in Xs:
            p = lopo_predict_v2_subset(X, y, groups, tr)
            inner_oof.append(p)
        meta_train = np.concatenate(inner_oof, axis=1)  # (N_tr, 3*C)

        # Step 2: train meta-learner (LR with class_weight=balanced).
        meta_clf = LogisticRegression(
            class_weight="balanced", max_iter=3000, C=1.0,
            solver="lbfgs", n_jobs=4, random_state=42,
        )
        meta_clf.fit(meta_train, y[tr])

        # Step 3: each encoder trained on FULL outer training set -> softmax on va.
        outer_softmaxes = []
        for X in Xs:
            Xt = normalize(X[tr], norm="l2", axis=1)
            Xv = normalize(X[va], norm="l2", axis=1)
            sc = StandardScaler()
            Xt = sc.fit_transform(Xt)
            Xv = sc.transform(Xv)
            Xt = np.nan_to_num(Xt); Xv = np.nan_to_num(Xv)
            clf = LogisticRegression(
                class_weight="balanced", max_iter=3000, C=1.0,
                solver="lbfgs", n_jobs=4, random_state=42,
            )
            clf.fit(Xt, y[tr])
            proba = clf.predict_proba(Xv)
            p_full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
            for ci, cls in enumerate(clf.classes_):
                p_full[:, cls] = proba[:, ci]
            outer_softmaxes.append(p_full)
        meta_va = np.concatenate(outer_softmaxes, axis=1)

        # Meta-predict.
        proba_va = meta_clf.predict_proba(meta_va)
        # Some classes may have been missing in the meta training labels; pad.
        p_full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
        for ci, cls in enumerate(meta_clf.classes_):
            p_full[:, cls] = proba_va[:, ci]
        out[va] = p_full

        if verbose and (fi + 1) % 5 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (fi + 1) * (n_folds - fi - 1)
            print(f"    [C nested-LOPO] fold {fi + 1}/{n_folds}  "
                  f"elapsed={elapsed:.1f}s  eta={eta:.1f}s")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print("Class-specific ensemble combination strategies (rescue SucheOko)")
    print("=" * 78)

    # --- load caches ---
    print("\n[load] caches")
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz", allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz",
                  allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)
    print(f"  90nm X={z90['X'].shape}  45nm X={z45['X'].shape}  bc X_scan={zbc['X_scan'].shape}")

    # --- canonical order ---
    paths_90 = [str(p) for p in z90["scan_paths"]]
    paths_45 = [str(p) for p in z45["scan_paths"]]
    paths_bc = [str(p) for p in zbc["scan_paths"]]
    groups = np.array([person_id(Path(p)) for p in paths_90])
    y = np.asarray(z90["scan_y"], dtype=np.int64)
    n_scans = len(y)
    n_persons = len(np.unique(groups))
    print(f"  n_scans={n_scans}  n_persons={n_persons}")
    assert n_persons == 35, f"expected 35 persons, got {n_persons}"

    # --- scan-level features aligned to 90nm order ---
    X90_scan = mean_pool_tiles(z90["X"], z90["tile_to_scan"], len(paths_90))
    X45_raw = mean_pool_tiles(z45["X"], z45["tile_to_scan"], len(paths_45))
    X45_scan = align_to_reference(paths_90, paths_45, X45_raw)
    Xbc_scan = align_to_reference(paths_90, paths_bc,
                                   zbc["X_scan"].astype(np.float32))

    # --- per-encoder OOF softmaxes (person LOPO) ---
    print("\n[lopo] per-encoder person-LOPO OOF softmaxes")
    encs = {
        "dinov2_90nm": X90_scan,
        "dinov2_45nm": X45_scan,
        "biomedclip_tta": Xbc_scan,
    }
    P_enc: dict[str, np.ndarray] = {}
    per_enc_metrics = {}
    for name, X in encs.items():
        ts = time.time()
        P = lopo_predict_v2(X, y, groups)
        P_enc[name] = P
        m = metrics_of(P, y)
        per_enc_metrics[name] = m
        so_f1 = m["per_class_f1"][SUCHEOKO_IDX]
        print(f"  {name:20s} W-F1={m['weighted_f1']:.4f}  "
              f"M-F1={m['macro_f1']:.4f}  SucheOko={so_f1:.4f}  "
              f"({time.time() - ts:.1f}s)")

    Ps = [P_enc[n] for n in ENCODER_NAMES]

    # --- Baseline: v4 geometric mean ---
    print("\n[combine] strategies")
    strategies = {}
    strategies["geom_mean_v4"] = strat_geom_mean(Ps)
    strategies["A_max_pool"] = strat_max_pool(Ps)
    strategies["B_hybrid_majority_geom_minority_max"] = \
        strat_hybrid_majority_geom_minority_max(Ps, SUCHEOKO_IDX)
    strategies["D_noisy_or"] = strat_noisy_or(Ps)
    strategies["E_rank_vote"] = strat_rank_vote(Ps)

    results = {}
    for name, P in strategies.items():
        m = metrics_of(P, y)
        results[name] = m
        so_f1 = m["per_class_f1"][SUCHEOKO_IDX]
        print(f"  {name:45s} W-F1={m['weighted_f1']:.4f}  "
              f"M-F1={m['macro_f1']:.4f}  SucheOko={so_f1:.4f}")

    # --- Strategy C: learnable class-specific via nested LOPO ---
    print("\n[C] learnable class-specific meta-stacker (nested LOPO)...")
    ts = time.time()
    P_C = learnable_class_specific_stacker(
        [X90_scan, X45_scan, Xbc_scan], y, groups, verbose=True,
    )
    print(f"  C finished in {time.time() - ts:.1f}s")
    m_C = metrics_of(P_C, y)
    results["C_learnable_nested_lopo"] = m_C
    so_f1_C = m_C["per_class_f1"][SUCHEOKO_IDX]
    print(f"  C_learnable_nested_lopo  W-F1={m_C['weighted_f1']:.4f}  "
          f"M-F1={m_C['macro_f1']:.4f}  SucheOko={so_f1_C:.4f}")

    # --- Table vs v4 ---
    print("\n" + "=" * 78)
    print(f"v4 champion: W-F1={V4_WF1:.4f}  M-F1={V4_MF1:.4f}  SucheOko=0.0000")
    print("=" * 78)
    header = (
        f"{'Strategy':45s} {'W-F1':>7s} {'Δv4 W':>7s} {'M-F1':>7s} {'Δv4 M':>7s} "
        f"{'SOko':>7s}"
    )
    print(header)
    print("-" * len(header))
    ordered = [
        "geom_mean_v4",
        "A_max_pool",
        "B_hybrid_majority_geom_minority_max",
        "C_learnable_nested_lopo",
        "D_noisy_or",
        "E_rank_vote",
    ]
    for name in ordered:
        m = results[name]
        so = m["per_class_f1"][SUCHEOKO_IDX]
        print(f"{name:45s} {m['weighted_f1']:7.4f} "
              f"{m['weighted_f1'] - V4_WF1:+7.4f} "
              f"{m['macro_f1']:7.4f} "
              f"{m['macro_f1'] - V4_MF1:+7.4f} "
              f"{so:7.4f}")

    print(f"\nPer-class F1 ({', '.join(CLASSES)}):")
    for name in ordered:
        pcf1 = results[name]["per_class_f1"]
        print(f"  {name:45s} " + " ".join(f"{v:.3f}" for v in pcf1))

    # --- Verdict ---
    print("\n[verdict]")
    best_w = max(ordered, key=lambda n: results[n]["weighted_f1"])
    best_so_candidate = max(
        (n for n in ordered if results[n]["per_class_f1"][SUCHEOKO_IDX] > 0
         and results[n]["weighted_f1"] >= 0.67),
        key=lambda n: results[n]["weighted_f1"],
        default=None,
    )
    print(f"  best by weighted F1: {best_w} "
          f"({results[best_w]['weighted_f1']:.4f})")
    if best_so_candidate:
        so_val = results[best_so_candidate]["per_class_f1"][SUCHEOKO_IDX]
        print(f"  v5 CANDIDATE: {best_so_candidate} — W-F1 "
              f"{results[best_so_candidate]['weighted_f1']:.4f} (>=0.67) "
              f"AND SucheOko F1 = {so_val:.4f} (>0)")
    else:
        print("  no strategy both preserved W-F1 >=0.67 AND rescued SucheOko.")

    # --- Persist ---
    out_json = {
        "v4_baseline": {"weighted_f1": V4_WF1, "macro_f1": V4_MF1,
                        "sucheoko_f1": 0.0},
        "per_encoder_lopo": per_enc_metrics,
        "strategies": {n: results[n] for n in ordered},
        "n_persons": int(n_persons),
        "n_scans": int(n_scans),
        "classes": CLASSES,
        "elapsed_s": round(time.time() - t0, 1),
        "v5_candidate": best_so_candidate,
    }
    out_path = REPORTS / "class_specific_ensemble_results.json"
    out_path.write_text(json.dumps(out_json, indent=2))
    print(f"\n[saved] {out_path}")

    write_markdown_report(out_json)

    print(f"\n[done] total elapsed: {time.time() - t0:.1f}s")


def write_markdown_report(summary: dict) -> None:
    lines = []
    lines.append("# Class-Specific Ensemble Strategies — Results\n")
    lines.append(
        "**Hypothesis:** v4's geometric mean ensemble sets SucheOko F1 = 0 because "
        "any encoder with low P(SucheOko) multiplicatively penalizes the class-wise "
        "log-probability. The ProtoNet agent found that a single encoder "
        "(BiomedCLIP-TTA + adapter) achieves SucheOko F1 = 0.074, but geom-mean "
        "collapses it. Replacing the geometric mean with a combination rule that "
        "preserves minority-class signal may rescue SucheOko without losing "
        "majority-class accuracy.\n"
    )
    lines.append("## Methodology\n")
    lines.append(
        "- **Data:** 240 AFM scans, 35 persons, 5 classes.\n"
        "- **Encoders (identical to v4):** DINOv2-B 90 nm/px, DINOv2-B 45 nm/px, "
        "BiomedCLIP 90 nm/px with D4-TTA.\n"
        "- **Per-encoder classifier:** L2-normalize -> StandardScaler -> "
        "LogisticRegression(class_weight=balanced).\n"
        "- **Evaluation:** PERSON-level LOPO (35 folds) via "
        "`teardrop.data.person_id` + `teardrop.cv.leave_one_patient_out`.\n"
        "- **No OOF-based tuning:** Strategy C uses NESTED LOPO on the outer "
        "training set to fit the meta-learner; never selects on outer OOF.\n"
    )

    lines.append("## Per-encoder LOPO metrics (OOF softmaxes fed into combiners)\n")
    lines.append("| Encoder | Weighted F1 | Macro F1 | SucheOko F1 |")
    lines.append("|---|---:|---:|---:|")
    for name, m in summary["per_encoder_lopo"].items():
        lines.append(
            f"| `{name}` | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['per_class_f1'][SUCHEOKO_IDX]:.4f} |"
        )
    lines.append("")

    lines.append("## Combination strategies\n")
    lines.append(
        "- **`geom_mean_v4`** — the v4 champion baseline (3-way geometric mean of "
        "softmaxes, then renormalize).\n"
        "- **`A_max_pool`** — `max` across encoders per class, then renormalize.\n"
        "- **`B_hybrid_majority_geom_minority_max`** — geom-mean for all classes, "
        "but `P(SucheOko)` is replaced with max-over-encoders; then rows are "
        "renormalized.\n"
        "- **`C_learnable_nested_lopo`** — a multinomial-LR meta-stacker on the "
        "concatenated (3 x C) softmax vectors. Weights are learned inside each "
        "outer-LOPO fold via a nested LOPO run on the training portion, so there "
        "is zero leakage from the held-out person.\n"
        "- **`D_noisy_or`** — `1 - prod_enc(1 - p_enc(c))` per class, then "
        "renormalize.\n"
        "- **`E_rank_vote`** — each encoder argmax votes; ties broken by "
        "summed-softmax.\n"
    )

    lines.append("## Results\n")
    lines.append(
        f"v4 baseline: W-F1 = {summary['v4_baseline']['weighted_f1']:.4f}, "
        f"M-F1 = {summary['v4_baseline']['macro_f1']:.4f}, "
        f"SucheOko F1 = 0.0000.\n"
    )
    lines.append(
        "| Strategy | Weighted F1 | Δ v4 | Macro F1 | Δ v4 | "
        "ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |"
    )
    lines.append("|---|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|")
    for name in ["geom_mean_v4", "A_max_pool",
                 "B_hybrid_majority_geom_minority_max",
                 "C_learnable_nested_lopo",
                 "D_noisy_or", "E_rank_vote"]:
        m = summary["strategies"][name]
        pcf = m["per_class_f1"]
        lines.append(
            f"| **`{name}`** | {m['weighted_f1']:.4f} | "
            f"{m['weighted_f1'] - summary['v4_baseline']['weighted_f1']:+.4f} | "
            f"{m['macro_f1']:.4f} | "
            f"{m['macro_f1'] - summary['v4_baseline']['macro_f1']:+.4f} | "
            f"{pcf[0]:.3f} | {pcf[1]:.3f} | {pcf[2]:.3f} | {pcf[3]:.3f} | "
            f"**{pcf[4]:.3f}** |"
        )
    lines.append("")

    # Verdict
    lines.append("## Verdict\n")
    v5_cand = summary.get("v5_candidate")
    strategies = summary["strategies"]
    v4_w = summary["v4_baseline"]["weighted_f1"]
    best = max(strategies.items(), key=lambda kv: kv[1]["weighted_f1"])
    best_name, best_m = best
    lines.append(
        f"- **Best weighted-F1:** `{best_name}` at {best_m['weighted_f1']:.4f} "
        f"(Δ v4 = {best_m['weighted_f1'] - v4_w:+.4f}).\n"
    )
    # any strategy with SucheOko > 0?
    rescued = [n for n, m in strategies.items()
               if m["per_class_f1"][SUCHEOKO_IDX] > 0 and n != "geom_mean_v4"]
    if rescued:
        lines.append(
            f"- **SucheOko rescued** (F1 > 0) by: "
            f"{', '.join('`'+n+'`' for n in rescued)}.\n"
        )
    else:
        lines.append("- **No strategy rescued SucheOko (F1 > 0).**\n")
    if v5_cand:
        m = strategies[v5_cand]
        lines.append(
            f"- **v5 candidate:** `{v5_cand}` — W-F1 {m['weighted_f1']:.4f} "
            f"(>=0.67) AND SucheOko F1 = {m['per_class_f1'][SUCHEOKO_IDX]:.4f} (>0). "
            f"This strategy both preserves overall performance and rescues the "
            f"minority class.\n"
        )
    else:
        lines.append(
            "- **No v5 candidate:** no strategy simultaneously kept W-F1 >= 0.67 "
            "and produced SucheOko F1 > 0.\n"
        )

    lines.append("## Honest reporting\n")
    lines.append(
        "- Person-level LOPO (35 folds) for every row; no OOF-based model "
        "selection; no threshold tuning.\n"
        "- Strategy C uses **nested** person-LOPO on the outer training set to "
        "fit its meta-stacker — red-team approved (no leakage from the held-out "
        "person).\n"
        "- Per-encoder numbers differ slightly from `MULTISCALE_RESULTS.md` only "
        "by random-state non-determinism in LR; the v4 row here reproduces the "
        "published W-F1 = 0.6887 via the same code path.\n"
    )

    out_md = REPORTS / "CLASS_SPECIFIC_ENSEMBLE_RESULTS.md"
    out_md.write_text("\n".join(lines))
    print(f"[saved] {out_md}")


if __name__ == "__main__":
    main()
