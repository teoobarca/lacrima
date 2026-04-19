"""Prototypical Network rescue experiment for SucheOko (2-person class).

Singular focus vs the existing multi-encoder `prototypical_networks.py`:

  - Uses ONE cached embedding only: `cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`
    (DINOv2-B TTA-D4, 240 x 768, already scan-level).
  - Person-level LOPO (mandatory) -- prototypes are computed strictly from
    training persons; the query person never contributes to any prototype.
  - Four prototype variants:
      (a) Standard (L2-normed cosine mean)
      (b) Weighted (sqrt-inverse class-size reweighted -- boosts SucheOko)
      (c) Person-averaged (mean-within-person first, then mean across persons)
      (d) K-NN-weighted (query-dependent support weighting, soft k-NN prototype)

  - Bootstrap (1000x, stratified-by-person) vs v4 champion (wF1 = 0.6887).
  - Ensemble tests with v4 softmax (cache/v4_oof_predictions.npz):
      * flat geometric mean
      * class-gated mean (ProtoNet gets SucheOko channel, v4 gets the rest)

Outputs:
  - cache/protonet_predictions.json   (all variant softmaxes + ensembles)
  - reports/PROTOTYPICAL_NETWORKS.md  (human-readable summary)

Budget: pure numpy/sklearn, a few seconds.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import CLASSES, person_id  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"

N_CLASSES = len(CLASSES)
SUCHEOKO = CLASSES.index("SucheOko")
EPS = 1e-12

CHAMP_V4_WF1 = 0.6887
CHAMP_V4_MF1 = 0.5541


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def l2norm(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def softmax_from_distances(D: np.ndarray, T: float = 0.1) -> np.ndarray:
    """softmax(-D / T), row-wise."""
    logits = -D / max(T, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / np.maximum(ex.sum(axis=1, keepdims=True), EPS)


def cosine_distance(Q: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Cosine distance (1 - dot) assuming Q and P rows are L2-normalized.

    Returns (n_q, n_p).
    """
    return 1.0 - Q @ P.T


def weighted_f1(y, pred):
    return float(f1_score(y, pred, average="weighted", zero_division=0))


def macro_f1(y, pred):
    return float(f1_score(y, pred, average="macro", zero_division=0))


def per_class_f1(y, pred):
    return f1_score(y, pred, labels=list(range(N_CLASSES)),
                    average=None, zero_division=0).tolist()


# ---------------------------------------------------------------------------
# Prototype builders
# ---------------------------------------------------------------------------

def proto_standard(X_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """Class prototype = L2-normed mean of (L2-normed) training embeddings."""
    Xn = l2norm(X_tr.astype(np.float32))
    P = np.zeros((N_CLASSES, Xn.shape[1]), dtype=np.float32)
    for c in range(N_CLASSES):
        m = y_tr == c
        if m.any():
            P[c] = Xn[m].mean(axis=0)
            P[c] = P[c] / max(np.linalg.norm(P[c]), EPS)
    return P


def proto_person_averaged(X_tr: np.ndarray, y_tr: np.ndarray,
                          g_tr: np.ndarray) -> np.ndarray:
    """Average embeddings within person first, then mean across persons.

    Reduces intra-person bias: a person with 15 scans no longer dominates over
    a person with 3 scans.
    """
    Xn = l2norm(X_tr.astype(np.float32))
    P = np.zeros((N_CLASSES, Xn.shape[1]), dtype=np.float32)
    for c in range(N_CLASSES):
        m = y_tr == c
        if not m.any():
            continue
        persons = np.unique(g_tr[m])
        person_protos = []
        for pid in persons:
            pm = m & (g_tr == pid)
            if pm.any():
                v = Xn[pm].mean(axis=0)
                v = v / max(np.linalg.norm(v), EPS)
                person_protos.append(v)
        if person_protos:
            P[c] = np.stack(person_protos).mean(axis=0)
            P[c] = P[c] / max(np.linalg.norm(P[c]), EPS)
    return P


def predict_proto(P_protos: np.ndarray, X_va: np.ndarray,
                  T: float = 0.1) -> np.ndarray:
    """Standard: softmax over negative cosine distances."""
    Qn = l2norm(X_va.astype(np.float32))
    D = cosine_distance(Qn, P_protos)
    return softmax_from_distances(D, T=T)


def predict_weighted_by_classsize(P_protos: np.ndarray, X_va: np.ndarray,
                                  y_tr: np.ndarray, T: float = 0.1,
                                  boost: str = "sqrt_inv") -> np.ndarray:
    """Boost minority classes via logit bias = log(weight).

    weight_c ~ 1/sqrt(N_c)  (or 1/N_c). Adds a per-class bias term so the
    decision boundary is shifted toward rare classes -- cheap equivalent of
    class-weighted softmax.
    """
    Qn = l2norm(X_va.astype(np.float32))
    D = cosine_distance(Qn, P_protos)
    counts = np.array([(y_tr == c).sum() for c in range(N_CLASSES)],
                      dtype=np.float32)
    counts = np.maximum(counts, 1.0)
    if boost == "inv":
        w = 1.0 / counts
    else:  # sqrt_inv
        w = 1.0 / np.sqrt(counts)
    w = w / w.sum()
    bias = np.log(np.maximum(w, EPS))  # (C,)
    logits = -D / max(T, 1e-6) + bias[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / np.maximum(ex.sum(axis=1, keepdims=True), EPS)


def predict_knn_weighted(X_tr: np.ndarray, y_tr: np.ndarray,
                         X_va: np.ndarray, T: float = 0.1,
                         tau: float = 0.15) -> np.ndarray:
    """Query-dependent soft-kNN prototype.

    For each query q and class c:
        score(q, c) = sum over support s in c of exp(-d(q, s)/tau)
    Then probs = softmax(score / T).  Equivalent to a kernel-density style
    distance-weighted voting: close anchors count more. This naturally
    handles 2-person SucheOko because the 14 SucheOko anchors create a
    coherent kernel blob even if the class centroid is suboptimal.
    """
    Xn_tr = l2norm(X_tr.astype(np.float32))
    Xn_va = l2norm(X_va.astype(np.float32))
    D = cosine_distance(Xn_va, Xn_tr)     # (n_q, n_s)
    K = np.exp(-D / max(tau, 1e-6))         # similarities
    # per-class normalization: divide by class size so that SucheOko isn't
    # drowned by majority classes having more anchors. This is a mild
    # class-frequency correction (equivalent to mean rather than sum).
    scores = np.zeros((Xn_va.shape[0], N_CLASSES), dtype=np.float64)
    for c in range(N_CLASSES):
        m = y_tr == c
        n_c = int(m.sum())
        if n_c == 0:
            continue
        scores[:, c] = K[:, m].sum(axis=1) / n_c
    logits = np.log(np.maximum(scores, EPS)) / max(T, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / np.maximum(ex.sum(axis=1, keepdims=True), EPS)


# ---------------------------------------------------------------------------
# LOPO runners
# ---------------------------------------------------------------------------

def run_variant(variant: str, X: np.ndarray, y: np.ndarray,
                groups: np.ndarray, T: float = 0.1) -> np.ndarray:
    """Returns OOF softmax (N, C) for the chosen variant."""
    N = len(y)
    P = np.zeros((N, N_CLASSES), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups):
        X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
        X_va = X[va]
        if variant == "standard":
            protos = proto_standard(X_tr, y_tr)
            P[va] = predict_proto(protos, X_va, T=T)
        elif variant == "weighted":
            protos = proto_standard(X_tr, y_tr)
            P[va] = predict_weighted_by_classsize(
                protos, X_va, y_tr, T=T, boost="sqrt_inv")
        elif variant == "person_avg":
            protos = proto_person_averaged(X_tr, y_tr, g_tr)
            P[va] = predict_proto(protos, X_va, T=T)
        elif variant == "knn_weighted":
            P[va] = predict_knn_weighted(X_tr, y_tr, X_va, T=T, tau=0.15)
        else:
            raise ValueError(variant)
    return P


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------

def geomean(probs_list: list[np.ndarray]) -> np.ndarray:
    L = np.zeros_like(probs_list[0])
    for Pk in probs_list:
        L = L + np.log(np.maximum(Pk, EPS))
    G = np.exp(L / len(probs_list))
    G /= np.maximum(G.sum(axis=1, keepdims=True), EPS)
    return G


def class_gated_ensemble(v4: np.ndarray, proto: np.ndarray,
                         boost_classes=(SUCHEOKO,),
                         alpha_boost: float = 0.6) -> np.ndarray:
    """Per-class weighted geometric mean: favor ProtoNet on boost_classes,
    v4 on the rest.

    For class c in boost_classes:   log P = alpha_boost * log(proto)
                                          + (1 - alpha_boost) * log(v4)
    For the rest:                   log P = 0.2 * log(proto) + 0.8 * log(v4)
    """
    alpha = np.full(N_CLASSES, 0.2, dtype=np.float64)
    for c in boost_classes:
        alpha[c] = alpha_boost
    # Compute per-class weighted logs, then renormalize into a proper softmax.
    L_p = np.log(np.maximum(proto, EPS))
    L_v = np.log(np.maximum(v4, EPS))
    L = alpha[None, :] * L_p + (1.0 - alpha[None, :]) * L_v
    L -= L.max(axis=1, keepdims=True)
    ex = np.exp(L)
    return ex / np.maximum(ex.sum(axis=1, keepdims=True), EPS)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_pval(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                   groups: np.ndarray, n_boot: int = 1000,
                   seed: int = 42) -> dict:
    """Person-level bootstrap: Delta = wF1(a) - wF1(b).

    Returns mean/std Delta, 95% CI, and P(Delta > 0).
    Sampling is person-with-replacement; ties broken by scan-concatenation.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_person = {p: np.where(groups == p)[0] for p in uniq}
    deltas = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_person[p] for p in sampled])
        wa = weighted_f1(y[sel], pred_a[sel])
        wb = weighted_f1(y[sel], pred_b[sel])
        deltas.append(wa - wb)
    deltas = np.array(deltas)
    return {
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std()),
        "ci_lo": float(np.quantile(deltas, 0.025)),
        "ci_hi": float(np.quantile(deltas, 0.975)),
        "p_gt_0": float((deltas > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_cache():
    z = np.load(CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz",
                allow_pickle=True)
    X = z["X_scan"].astype(np.float32)
    y = z["scan_y"].astype(np.int64)
    paths = [str(p) for p in z["scan_paths"]]
    groups = np.array([person_id(Path(p)) for p in paths])
    return X, y, groups, paths


def load_v4():
    z = np.load(CACHE / "v4_oof_predictions.npz", allow_pickle=True)
    return z["proba"].astype(np.float64), z["y"].astype(np.int64), \
        [str(p) for p in z["scan_paths"]]


def metrics_row(y, P):
    pred = P.argmax(axis=1)
    return {
        "weighted_f1": weighted_f1(y, pred),
        "macro_f1": macro_f1(y, pred),
        "per_class_f1": per_class_f1(y, pred),
        "sucheoko_f1": per_class_f1(y, pred)[SUCHEOKO],
    }


def pretty_pc(v):
    return "  ".join(f"{CLASSES[i][:6]}={p:.3f}" for i, p in enumerate(v))


def main():
    t0 = time.time()
    print("=" * 78)
    print("ProtoNet SucheOko rescue — single-encoder DINOv2-B TTA")
    print("=" * 78)

    X, y, groups, paths = load_cache()
    v4_proba, y_v4, paths_v4 = load_v4()
    assert paths == paths_v4, "Path ordering mismatch between caches."
    assert np.array_equal(y, y_v4), "Label mismatch between caches."

    print(f"[data] n_scans={len(y)}  n_persons={len(np.unique(groups))}  "
          f"n_classes={N_CLASSES}")
    for c in range(N_CLASSES):
        m = y == c
        print(f"  {CLASSES[c]:20s} scans={m.sum():3d}  "
              f"persons={len(np.unique(groups[m])):2d}")
    print(f"[v4 champion] wF1={weighted_f1(y, v4_proba.argmax(axis=1)):.4f}  "
          f"mF1={macro_f1(y, v4_proba.argmax(axis=1)):.4f}  "
          f"SucheOko={per_class_f1(y, v4_proba.argmax(axis=1))[SUCHEOKO]:.4f}")

    # --- Run 4 ProtoNet variants ---------------------------------------------
    variants = ["standard", "weighted", "person_avg", "knn_weighted"]
    # Pick T = 0.1 (typical ProtoNet softmax temperature for cosine on unit sphere)
    T = 0.1
    all_probs = {}
    all_metrics = {}
    for v in variants:
        t_s = time.time()
        P = run_variant(v, X, y, groups, T=T)
        m = metrics_row(y, P)
        all_probs[v] = P
        all_metrics[v] = m
        dt = time.time() - t_s
        print(f"\n[variant] {v:16s} wF1={m['weighted_f1']:.4f}  "
              f"mF1={m['macro_f1']:.4f}  SucheOko={m['sucheoko_f1']:.4f}  "
              f"({dt:.1f}s)\n           {pretty_pc(m['per_class_f1'])}")

    # --- Ensembles with v4 ---------------------------------------------------
    # Best ProtoNet by SucheOko F1 for the "minority-class" ensemble; best by
    # weighted F1 for the "neutral" ensemble.
    best_by_w = max(variants, key=lambda v: all_metrics[v]["weighted_f1"])
    best_by_suo = max(variants, key=lambda v: all_metrics[v]["sucheoko_f1"])
    print(f"\n[ensemble] best ProtoNet by wF1: {best_by_w}; "
          f"best by SucheOko: {best_by_suo}")

    ens = {}
    # (1) Flat geomean of v4 and best-w proto
    flat = geomean([v4_proba, all_probs[best_by_w]])
    ens["v4_x_protoflat"] = {
        "components": ["v4", best_by_w],
        "strategy": "geometric_mean",
        **metrics_row(y, flat),
    }
    # (2) Class-gated: boost SucheOko with best-suo proto
    gated = class_gated_ensemble(v4_proba, all_probs[best_by_suo],
                                 boost_classes=(SUCHEOKO,),
                                 alpha_boost=0.6)
    ens["v4_x_protogated"] = {
        "components": ["v4", best_by_suo],
        "strategy": "class_gated_sucheoko_alpha=0.6",
        **metrics_row(y, gated),
    }
    # (3) Sweep alpha_boost to find best gated variant (honest: pick by wF1
    # and report which alpha won -- no nested CV leak since alpha is a
    # post-hoc hyperparam on the same data; we report as "oracle alpha").
    alpha_grid = np.arange(0.1, 0.91, 0.1)
    alpha_results = []
    for a in alpha_grid:
        G = class_gated_ensemble(v4_proba, all_probs[best_by_suo],
                                 boost_classes=(SUCHEOKO,), alpha_boost=float(a))
        mm = metrics_row(y, G)
        alpha_results.append({"alpha": float(a), **mm})
    best_alpha = max(alpha_results, key=lambda d: d["weighted_f1"])
    ens["v4_x_protogated_oracle_alpha"] = {
        **best_alpha,
        "components": ["v4", best_by_suo],
        "strategy": f"class_gated_sucheoko_alpha={best_alpha['alpha']:.2f}_ORACLE",
        "note": "alpha picked on OOF set; reported as oracle ceiling only",
    }

    print("\n[ensemble] v4 x ProtoNet (flat geomean): "
          f"wF1={ens['v4_x_protoflat']['weighted_f1']:.4f}  "
          f"mF1={ens['v4_x_protoflat']['macro_f1']:.4f}  "
          f"SucheOko={ens['v4_x_protoflat']['sucheoko_f1']:.4f}")
    print("[ensemble] v4 x ProtoNet (class-gated a=0.6): "
          f"wF1={ens['v4_x_protogated']['weighted_f1']:.4f}  "
          f"mF1={ens['v4_x_protogated']['macro_f1']:.4f}  "
          f"SucheOko={ens['v4_x_protogated']['sucheoko_f1']:.4f}")
    print(f"[ensemble] class-gated oracle alpha={best_alpha['alpha']:.2f}: "
          f"wF1={best_alpha['weighted_f1']:.4f}  "
          f"mF1={best_alpha['macro_f1']:.4f}  "
          f"SucheOko={best_alpha['sucheoko_f1']:.4f}")

    # --- Bootstrap vs v4 -----------------------------------------------------
    print("\n[bootstrap] 1000x person-level vs v4...")
    v4_pred = v4_proba.argmax(axis=1)
    boots = {}
    boots["protonet_best_by_w"] = bootstrap_pval(
        y, all_probs[best_by_w].argmax(axis=1), v4_pred, groups,
        n_boot=1000, seed=42)
    boots["v4_x_protoflat"] = bootstrap_pval(
        y, flat.argmax(axis=1), v4_pred, groups, n_boot=1000, seed=42)
    boots["v4_x_protogated"] = bootstrap_pval(
        y, gated.argmax(axis=1), v4_pred, groups, n_boot=1000, seed=42)
    G_best = class_gated_ensemble(v4_proba, all_probs[best_by_suo],
                                  boost_classes=(SUCHEOKO,),
                                  alpha_boost=best_alpha["alpha"])
    boots["v4_x_protogated_oracle_alpha"] = bootstrap_pval(
        y, G_best.argmax(axis=1), v4_pred, groups, n_boot=1000, seed=42)

    for name, b in boots.items():
        print(f"  {name:35s} dW={b['mean_delta']:+.4f}  "
              f"95%CI=[{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]  "
              f"P(d>0)={b['p_gt_0']:.3f}")

    # --- Persist -------------------------------------------------------------
    predictions_out = {
        "n_scans": int(len(y)),
        "n_persons": int(len(np.unique(groups))),
        "classes": CLASSES,
        "v4_champion": {"weighted_f1": CHAMP_V4_WF1,
                        "macro_f1": CHAMP_V4_MF1},
        "variants": {v: all_metrics[v] for v in variants},
        "ensembles": ens,
        "bootstraps": boots,
        "best_by_weighted_f1": best_by_w,
        "best_by_sucheoko_f1": best_by_suo,
        "oracle_alpha_boost": best_alpha["alpha"],
        # Compact softmaxes (as lists)
        "softmax": {
            "v4": v4_proba.tolist(),
            **{f"protonet_{v}": all_probs[v].tolist() for v in variants},
            "ensemble_flat": flat.tolist(),
            "ensemble_gated_a0.6": gated.tolist(),
            "ensemble_gated_oracle": G_best.tolist(),
        },
        "y": y.tolist(),
        "persons": groups.tolist(),
        "paths": paths,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = CACHE / "protonet_predictions.json"
    out_path.write_text(json.dumps(predictions_out, indent=2))
    print(f"\n[saved] {out_path}")

    write_report(predictions_out, best_alpha, boots, all_metrics, ens)
    print(f"[saved] reports/PROTOTYPICAL_NETWORKS.md")
    print(f"[done] elapsed {time.time() - t0:.1f}s")


def write_report(summary, best_alpha, boots, all_metrics, ens):
    classes = summary["classes"]
    y = np.array(summary["y"])
    v4_w = summary["v4_champion"]["weighted_f1"]
    v4_m = summary["v4_champion"]["macro_f1"]
    suo = classes.index("SucheOko")

    L = []
    L.append("# Prototypical Network — SucheOko Rescue Experiment\n")
    L.append("Single-encoder ProtoNet rescue attempt for the 2-person "
             "SucheOko class, which the v4 champion predicts F1 = 0.000.\n")
    L.append("## Setup\n")
    L.append(
        f"- **Data:** {summary['n_scans']} scans, {summary['n_persons']} "
        "persons, 5 classes.\n"
        "- **Embeddings:** `cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz` "
        "(DINOv2-B, TTA D4, 240 x 768).\n"
        "- **CV:** person-level LOPO (35 folds), groups = "
        "`teardrop.data.person_id`. Prototypes computed strictly from training "
        "persons; query person never in prototype set.\n"
        "- **Distance:** cosine on L2-normalized embeddings. Softmax temperature T=0.1.\n"
        f"- **Baseline:** v4 multiscale LR champion, wF1={v4_w:.4f}, "
        f"mF1={v4_m:.4f}, SucheOko F1=0.000.\n"
    )

    L.append("\n## Variants\n")
    L.append("1. **Standard** — L2-normed mean per class, cosine -> softmax.\n"
             "2. **Weighted** — per-class logit bias = log(1/sqrt(N_c)); "
             "shifts decision boundary toward rare classes.\n"
             "3. **Person-averaged** — mean within person first, then mean "
             "across persons -- reduces dominant-person bias inside class.\n"
             "4. **K-NN-weighted** — query-dependent soft kNN voting, kernel "
             "bandwidth tau=0.15; per-class normalized.\n")

    L.append("\n## Per-variant results (person-LOPO)\n")
    L.append("| Variant | wF1 | mF1 | " + " | ".join(classes) + " |")
    L.append("|---|---:|---:|" + "|".join([":---:"] * len(classes)) + "|")
    for v, m in all_metrics.items():
        pc = " | ".join(f"{p:.3f}" for p in m["per_class_f1"])
        L.append(f"| **{v}** | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} | {pc} |")
    L.append("")

    L.append("\n## Ensembles with v4\n")
    L.append("| Ensemble | wF1 | mF1 | Δ vs v4 | SucheOko F1 |")
    L.append("|---|---:|---:|---:|---:|")
    for name, e in ens.items():
        dw = e["weighted_f1"] - v4_w
        L.append(f"| `{name}` | {e['weighted_f1']:.4f} | "
                 f"{e['macro_f1']:.4f} | {dw:+.4f} | {e['sucheoko_f1']:.4f} |")
    L.append("")

    L.append("\n## Bootstrap (1000x person-level, vs v4)\n")
    L.append("| Candidate | mean ΔwF1 | 95% CI | P(Δ>0) |")
    L.append("|---|---:|---:|---:|")
    for name, b in boots.items():
        L.append(f"| {name} | {b['mean_delta']:+.4f} | "
                 f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {b['p_gt_0']:.3f} |")
    L.append("")

    L.append("\n## Verdict\n")
    # Is SucheOko rescued in any variant?
    best_suo_var = summary["best_by_sucheoko_f1"]
    suo_f1 = all_metrics[best_suo_var]["sucheoko_f1"]
    if suo_f1 >= 0.30:
        L.append(f"- **SucheOko rescued:** variant `{best_suo_var}` achieves "
                 f"F1 = {suo_f1:.3f} (vs v4 = 0.000). This is a pitch "
                 "asset -- we can claim ProtoNet gives us a minority-class "
                 "rescue channel.\n")
    elif suo_f1 > 0:
        L.append(f"- **Partial SucheOko rescue:** best variant "
                 f"`{best_suo_var}` reaches F1 = {suo_f1:.3f} (vs v4 = 0.000). "
                 "Below the 0.30 threshold; useful signal but not pitch-ready.\n")
    else:
        L.append("- **SucheOko NOT rescued** under any ProtoNet variant.\n")

    # Best ensemble
    best_ens_name = max(ens, key=lambda k: ens[k]["weighted_f1"])
    best_ens = ens[best_ens_name]
    L.append(f"- **Best ensemble:** `{best_ens_name}` -> wF1 = "
             f"{best_ens['weighted_f1']:.4f} (Δ v4 = "
             f"{best_ens['weighted_f1'] - v4_w:+.4f}), SucheOko = "
             f"{best_ens['sucheoko_f1']:.4f}.\n")
    p_gt = boots[best_ens_name]["p_gt_0"]
    if best_ens["weighted_f1"] > 0.70 and p_gt > 0.90:
        L.append("- **NEW CHAMPION CANDIDATE:** wF1 > 0.70 with "
                 f"P(Δ>0) = {p_gt:.3f} > 0.90. Proceed to red-team.\n")
    elif best_ens["weighted_f1"] > v4_w and p_gt > 0.50:
        L.append("- **Marginal improvement** over v4: useful but not a new "
                 "champion on its own.\n")
    else:
        L.append("- **No reliable ensemble gain** over v4. Ensemble with "
                 "v4 does not beat the champion.\n")

    L.append("\n## Honest reporting\n")
    L.append(
        "- Prototypes built strictly from training persons (LOPO guarantee).\n"
        "- Softmax T fixed at 0.1 (no per-fold tuning). Oracle alpha boost for "
        f"the gated ensemble was chosen on the OOF set "
        f"(alpha={best_alpha['alpha']:.2f}); this number is an OPTIMISTIC "
        "ceiling and is labeled as such.\n"
        "- Bootstrap resamples persons with replacement (not scans) to respect "
        "person-level evaluation.\n"
    )

    (REPORTS / "PROTOTYPICAL_NETWORKS.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
