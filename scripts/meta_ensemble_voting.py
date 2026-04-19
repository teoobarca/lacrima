"""Meta-ensemble voting over all cached OOF softmax predictions.

Goal: see whether ANY honest combination of previously-evaluated models beats
the v4 multi-scale champion (weighted F1 = 0.6887, person-LOPO).

Inputs (all aligned on the SAME 240-scan order, person_id groups):
    - cache/v4_oof.npz                               (v4 geom-mean)
    - cache/best_ensemble_predictions.npz            (v2-style 2-component)
    - cache/cascade_oof.npz        stage1_proba
    - cache/stacker_oof.npz        best_proba (soft_blend_nested)
    - cache/best_multichannel_v2_predictions.npz
         P_dinov2_height_pool / P_dinov2_rgb_pool / P_biomedclip_tta_height
    - cache/llm_gated_refined.npz  proba (stage1) — the LLM refinement itself
      is categorical, we use the proba underlying it as a reference.

Strategies:
    S1 Uniform mean           : mean of per-model softmaxes
    S2 Geometric mean         : log-space mean, renormalized
    S3 F1-weighted mean       : weight each softmax by its standalone
                                 weighted-F1 (normalized to sum to 1)
    S4 Rank vote              : per-scan majority vote of argmaxes; break ties
                                 by summed softmax
    S5 Meta-LR (per-model probs): LogisticRegression on concat of softmaxes
                                    (N_models * 5 features), NESTED person-LOPO
    S6 Meta-LR (argmax votes)   : LogisticRegression on one-hot argmax per model
                                    (N_models * 5 features), NESTED person-LOPO

Red-team rules:
    - Any strategy that TUNES parameters (S5, S6) is evaluated via NESTED
      person-level LOPO: the meta-model is fit ONLY on inner folds (persons
      != held-out person), never looks at the held-out person's base OOFs.
    - Simple aggregations (S1..S4) are leakage-free by construction; we just
      apply and score.

Outputs:
    - reports/META_ENSEMBLE_RESULTS.md
    - prints summary to stdout
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import CLASSES, person_id  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

N_CLASSES = len(CLASSES)
EPS = 1e-12

# Champion reference (v4 multi-scale, person-LOPO, honest)
V4_W_F1 = 0.6887
V4_M_F1 = 0.5541


# ---------------------------------------------------------------------------
# Load aligned OOF softmaxes
# ---------------------------------------------------------------------------
def load_oofs() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Returns (models, y, groups, scan_paths). All softmaxes shape (240, 5)."""
    z_ref = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz",
                    allow_pickle=True)
    scan_paths = z_ref["scan_paths"]
    y = np.asarray(z_ref["scan_y"], dtype=np.int64)
    groups = np.array([person_id(Path(str(p))) for p in scan_paths])

    models: dict[str, np.ndarray] = {}

    # v4 multi-scale champion (target)
    z = np.load(CACHE / "v4_oof.npz", allow_pickle=True)
    assert np.array_equal(z["scan_paths"], scan_paths), "v4 paths misaligned"
    models["v4_multiscale"] = z["proba"].astype(np.float64)

    # v2-style 2-component ensemble (best_ensemble_predictions)
    z = np.load(CACHE / "best_ensemble_predictions.npz", allow_pickle=True)
    assert np.array_equal(z["scan_paths"], scan_paths), "best_ens paths misaligned"
    models["v2_2comp"] = z["proba"].astype(np.float64)

    # cascade stage-1 proba
    z = np.load(CACHE / "cascade_oof.npz", allow_pickle=True)
    assert np.array_equal(z["scan_paths"], scan_paths), "cascade paths misaligned"
    models["cascade_stage1"] = z["stage1_proba"].astype(np.float64)

    # cascade-stacker best (soft_blend_nested)
    z = np.load(CACHE / "stacker_oof.npz", allow_pickle=True)
    assert np.array_equal(z["scan_paths"], scan_paths), "stacker paths misaligned"
    models["cascade_stacker_blend"] = z["best_proba"].astype(np.float64)

    # multichannel v2 3 members
    z = np.load(CACHE / "best_multichannel_v2_predictions.npz", allow_pickle=True)
    # stored under tta_paths here
    assert np.array_equal(z["tta_paths"], scan_paths), "mcv2 paths misaligned"
    models["mcv2_dinov2_height"] = z["P_dinov2_height_pool"].astype(np.float64)
    models["mcv2_dinov2_rgb"] = z["P_dinov2_rgb_pool"].astype(np.float64)
    models["mcv2_biomedclip_tta"] = z["P_biomedclip_tta_height"].astype(np.float64)

    # Renormalize softmax rows just in case of numerical drift
    for k, P in models.items():
        s = P.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        models[k] = P / s

    return models, y, groups, scan_paths


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics_of(P: np.ndarray, y: np.ndarray) -> dict:
    """Weighted / macro / per-class F1 from a softmax (N, K)."""
    pred = P.argmax(axis=1)
    return {
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y, pred, average=None,
                                 labels=list(range(N_CLASSES)),
                                 zero_division=0).tolist(),
    }


def metrics_of_pred(pred: np.ndarray, y: np.ndarray) -> dict:
    return {
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y, pred, average=None,
                                 labels=list(range(N_CLASSES)),
                                 zero_division=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Simple aggregators
# ---------------------------------------------------------------------------
def uniform_mean(probs_list: list[np.ndarray]) -> np.ndarray:
    M = np.mean(np.stack(probs_list, axis=0), axis=0)
    M /= M.sum(axis=1, keepdims=True)
    return M


def geometric_mean(probs_list: list[np.ndarray]) -> np.ndarray:
    log_sum = np.zeros_like(probs_list[0])
    for P in probs_list:
        log_sum = log_sum + np.log(P + EPS)
    G = np.exp(log_sum / len(probs_list))
    G /= G.sum(axis=1, keepdims=True)
    return G


def weighted_mean(probs_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    M = np.zeros_like(probs_list[0])
    for wi, P in zip(w, probs_list):
        M = M + wi * P
    M /= M.sum(axis=1, keepdims=True)
    return M


def rank_vote(probs_list: list[np.ndarray]) -> np.ndarray:
    """Majority vote of argmaxes. Ties broken by summed soft probas."""
    votes = np.stack([P.argmax(axis=1) for P in probs_list], axis=1)  # (N, M)
    sum_probs = np.sum(np.stack(probs_list, axis=0), axis=0)          # (N, K)
    n = votes.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        counts = np.bincount(votes[i], minlength=N_CLASSES)
        top = counts.max()
        cand = np.where(counts == top)[0]
        if len(cand) == 1:
            out[i] = cand[0]
        else:
            # break tie by summed softmax among candidates
            out[i] = cand[np.argmax(sum_probs[i][cand])]
    return out


# ---------------------------------------------------------------------------
# NESTED meta-LR stackers
# ---------------------------------------------------------------------------
def meta_lr_nested(features_by_person: np.ndarray, y: np.ndarray,
                   groups: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Nested person-LOPO: for each held-out person, fit LR on the other
    persons' OOF features only, predict on held-out.

    Because the INPUT FEATURES are themselves OOF predictions (each row's
    probability vector was produced by a model that did NOT see that person
    in training), using the other persons' OOF vectors as meta-training data
    is a legitimate stacked-generalization setup — no target leakage.
    """
    n = len(y)
    P_out = np.zeros((n, N_CLASSES), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups):
        X_tr = features_by_person[tr]
        X_va = features_by_person[va]
        y_tr = y[tr]
        clf = LogisticRegression(
            class_weight="balanced", max_iter=3000, C=C,
            solver="lbfgs", n_jobs=2, random_state=42,
        )
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_va)
        p_full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
        for ci, cls in enumerate(clf.classes_):
            p_full[:, cls] = proba[:, ci]
        P_out[va] = p_full
    return P_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("Meta-ensemble voting over cached OOFs (person-LOPO)")
    print("=" * 78)

    models, y, groups, scan_paths = load_oofs()
    n_persons = len(np.unique(groups))
    n = len(y)
    print(f"n_scans={n}  n_persons={n_persons}  n_models={len(models)}")
    print(f"class dist: {np.bincount(y)}  ({CLASSES})")

    # Per-model standalone metrics
    print("\n[per-model standalone F1 (person-LOPO, honest via OOF)]")
    per_model = {}
    for name, P in models.items():
        m = metrics_of(P, y)
        per_model[name] = m
        print(f"  {name:28s}  W-F1={m['weighted_f1']:.4f}  "
              f"M-F1={m['macro_f1']:.4f}")

    names = list(models.keys())
    probs_list = [models[k] for k in names]
    f1_weights = [per_model[k]["weighted_f1"] for k in names]

    results: dict[str, dict] = {}

    # ---- S1 Uniform mean ----
    P1 = uniform_mean(probs_list)
    results["S1_uniform_mean"] = metrics_of(P1, y)

    # ---- S2 Geometric mean ----
    P2 = geometric_mean(probs_list)
    results["S2_geometric_mean"] = metrics_of(P2, y)

    # ---- S3 F1-weighted mean ----
    P3 = weighted_mean(probs_list, f1_weights)
    results["S3_f1_weighted_mean"] = metrics_of(P3, y)

    # ---- S4 Rank / majority vote ----
    pred4 = rank_vote(probs_list)
    results["S4_rank_vote"] = metrics_of_pred(pred4, y)

    # ---- S5 Meta-LR on concatenated probas (NESTED) ----
    X_cat = np.concatenate(probs_list, axis=1)  # (240, N*5)
    P5 = meta_lr_nested(X_cat, y, groups, C=1.0)
    results["S5_meta_lr_nested_probs"] = metrics_of(P5, y)

    # Also report an IN-SAMPLE (leaky) fit of the same features to show
    # the optimism gap we always see with this setup.
    clf_leaky = LogisticRegression(class_weight="balanced", max_iter=3000,
                                   C=1.0, solver="lbfgs",
                                   random_state=42).fit(X_cat, y)
    P5_leaky = clf_leaky.predict_proba(X_cat)
    results["S5_meta_lr_INSAMPLE_leaky"] = metrics_of(P5_leaky, y)

    # ---- S6 Meta-LR on one-hot argmax votes (NESTED) ----
    one_hot = []
    for P in probs_list:
        oh = np.zeros_like(P)
        oh[np.arange(n), P.argmax(axis=1)] = 1.0
        one_hot.append(oh)
    X_votes = np.concatenate(one_hot, axis=1)
    P6 = meta_lr_nested(X_votes, y, groups, C=1.0)
    results["S6_meta_lr_nested_onehot"] = metrics_of(P6, y)

    # ---- Print summary ----
    print("\n[strategy results, person-LOPO]")
    print(f"{'Strategy':34s} {'W-F1':>7s} {'M-F1':>7s} {'ΔW vs v4':>10s} "
          f"{'leakage-risk':>14s}")
    print("-" * 80)
    risk = {
        "S1_uniform_mean": "none",
        "S2_geometric_mean": "none",
        "S3_f1_weighted_mean": "low*",
        "S4_rank_vote": "none",
        "S5_meta_lr_nested_probs": "controlled (nested)",
        "S5_meta_lr_INSAMPLE_leaky": "HIGH (leaky ref)",
        "S6_meta_lr_nested_onehot": "controlled (nested)",
    }
    for sname, m in results.items():
        delta = m["weighted_f1"] - V4_W_F1
        flag = f"{delta:+.4f}"
        print(f"{sname:34s} {m['weighted_f1']:7.4f} {m['macro_f1']:7.4f} "
              f"{flag:>10s} {risk[sname]:>14s}")
    print("\n* S3 weights come from OOF F1 used as global scalar; they do NOT")
    print("  depend on a held-out person's y, but they do depend on the")
    print("  FULL OOF set — treated as a mild leak-vs-noise gray zone.")

    # Per-class F1 table
    print("\n[per-class F1]")
    header = f"{'Strategy':34s} " + " ".join(f"{c[:10]:>11s}" for c in CLASSES)
    print(header)
    for sname, m in results.items():
        pcf1 = m["per_class_f1"]
        print(f"{sname:34s} " + " ".join(f"{v:>11.4f}" for v in pcf1))

    # ---- Write markdown report ----
    write_report(models, per_model, results, y, groups, f1_weights)

    # Persist full JSON too
    out = {
        "v4_reference": {"weighted_f1": V4_W_F1, "macro_f1": V4_M_F1},
        "n_scans": int(n),
        "n_persons": int(n_persons),
        "per_model": per_model,
        "strategies": results,
    }
    (REPORTS / "meta_ensemble_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n[saved] reports/meta_ensemble_results.json")
    print(f"[saved] reports/META_ENSEMBLE_RESULTS.md")


def write_report(models, per_model, results, y, groups, f1_weights):
    best_sname = max(results, key=lambda k: results[k]["weighted_f1"])
    best = results[best_sname]
    delta = best["weighted_f1"] - V4_W_F1

    lines = []
    lines.append("# Meta-Ensemble Voting — Results\n")
    lines.append(
        "**Question:** does ANY combination of previously-evaluated OOF "
        "predictions beat the v4 multi-scale champion "
        f"(person-LOPO weighted F1 = **{V4_W_F1:.4f}**)?\n"
    )
    lines.append("## Inputs (aligned on 240 scans, 35 persons)\n")
    lines.append("| Source | Model key | Standalone W-F1 | Standalone M-F1 |")
    lines.append("|---|---|---:|---:|")
    for k, m in per_model.items():
        lines.append(
            f"| `cache/...` | `{k}` | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} |"
        )
    lines.append("")

    lines.append("## Strategies\n")
    lines.append("| # | Strategy | W-F1 | M-F1 | Δ vs v4 (0.6887) | Leakage risk |")
    lines.append("|---|---|---:|---:|---:|---|")
    risk = {
        "S1_uniform_mean": "none — simple arithmetic mean of softmaxes",
        "S2_geometric_mean": "none — log-space mean of softmaxes",
        "S3_f1_weighted_mean": "mild — weights use full-OOF F1 (global scalar per model)",
        "S4_rank_vote": "none — argmax majority vote",
        "S5_meta_lr_nested_probs": "controlled — nested person-LOPO meta-LR on concat probas (N*5 feats)",
        "S5_meta_lr_INSAMPLE_leaky": "HIGH — same LR fit and scored in-sample (shown ONLY to expose optimism gap)",
        "S6_meta_lr_nested_onehot": "controlled — nested person-LOPO meta-LR on one-hot argmax (N*5 feats)",
    }
    for sname, m in results.items():
        d = m["weighted_f1"] - V4_W_F1
        lines.append(
            f"| {sname.split('_', 1)[0]} | **{sname}** | "
            f"{m['weighted_f1']:.4f} | {m['macro_f1']:.4f} | {d:+.4f} | "
            f"{risk[sname]} |"
        )
    lines.append("")

    lines.append("## Per-class F1\n")
    lines.append("| Strategy | " + " | ".join(CLASSES) + " |")
    lines.append("|---|" + "|".join([":---:"] * len(CLASSES)) + "|")
    for sname, m in results.items():
        pcf1 = m["per_class_f1"]
        lines.append(f"| **{sname}** | "
                     + " | ".join(f"{v:.4f}" for v in pcf1) + " |")
    lines.append("")

    lines.append("## Red-team analysis\n")
    lines.append(
        "- **S1, S2, S4** have no tuning: they combine cached OOFs with a "
        "deterministic rule. Their weighted-F1 is a direct read of the "
        "underlying predictions' agreement structure, not an inflated "
        "in-sample fit.\n"
        "- **S3** uses each model's full-OOF weighted-F1 as a global scalar "
        "weight. Those weights are scalars computed from the full 240-row "
        "OOF set; they do NOT depend on any held-out person's labels in a "
        "sample-specific way, but the same full OOFs appear in both the "
        "weight estimate and the combination. Treat the number as mildly "
        "optimistic (< 0.002 typical gap vs a nested-weight variant; prior "
        "red-team in `reports/RED_TEAM_ENSEMBLE_AUDIT.md` shows this).\n"
        "- **S5, S6** would be MAJOR leak offenders if naively fit on the "
        "full OOF and then scored on it. We instead refit the meta-LR for "
        "each held-out person on the other persons' OOF rows only — a true "
        "nested person-LOPO. The leaky ref row (`S5_..._INSAMPLE_leaky`) is "
        "included to show the size of the optimism gap when the nesting "
        "is skipped.\n"
    )

    lines.append("## Verdict\n")
    # Only consider honest strategies
    honest = {k: v for k, v in results.items()
              if k != "S5_meta_lr_INSAMPLE_leaky"}
    best_honest_name = max(honest, key=lambda k: honest[k]["weighted_f1"])
    best_honest = honest[best_honest_name]
    d_honest = best_honest["weighted_f1"] - V4_W_F1
    if d_honest > 0:
        lines.append(
            f"- Best HONEST strategy: **{best_honest_name}** with "
            f"W-F1 = {best_honest['weighted_f1']:.4f} "
            f"(Δ vs v4 = {d_honest:+.4f}).\n"
        )
        if d_honest >= 0.005:
            lines.append(
                "- This exceeds v4 by ≥ 0.005 — **flag as v5 candidate**. "
                "Run a paired red-team bootstrap (per-scan F1 differences, "
                "10k resamples at person granularity) before declaring a "
                "new champion.\n"
            )
        else:
            lines.append(
                f"- The delta is only {d_honest:+.4f} — within the noise "
                "floor of a 240-scan / 35-person OOF (typical sigma 0.01-0.02 "
                "under bootstrap). **Do NOT promote** to v5; fold the finding "
                "into the 'explored alternatives' ledger.\n"
            )
    else:
        lines.append(
            f"- Best HONEST strategy: **{best_honest_name}** with "
            f"W-F1 = {best_honest['weighted_f1']:.4f} "
            f"(Δ vs v4 = {d_honest:+.4f}).\n"
            f"- **No honest strategy beats v4.** The meta-ensemble approach "
            f"over these cached OOFs does not surface a new champion. v4 "
            f"(multi-scale DINOv2-B 90+45 nm + BiomedCLIP-TTA, geom-mean) "
            f"remains the best honest model at {V4_W_F1:.4f} W-F1 / "
            f"{V4_M_F1:.4f} M-F1.\n"
        )

    # Optimism gap signal
    if "S5_meta_lr_INSAMPLE_leaky" in results:
        leaky = results["S5_meta_lr_INSAMPLE_leaky"]["weighted_f1"]
        honest_s5 = results["S5_meta_lr_nested_probs"]["weighted_f1"]
        gap = leaky - honest_s5
        lines.append(
            f"- **Optimism gap for S5 (leaky vs nested):** "
            f"{leaky:.4f} − {honest_s5:.4f} = **{gap:+.4f}**. "
            "Any stacker score reported without nesting would be inflated by "
            "roughly this much — a textbook leakage trap on a 240-row OOF.\n"
        )

    (REPORTS / "META_ENSEMBLE_RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
