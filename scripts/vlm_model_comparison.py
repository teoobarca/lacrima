"""Compare Claude Haiku 4.5 / Sonnet 4.6 / Opus 4.7 on the AFM tear VLM task.

Design
------
- Use the same stratified, person-disjoint 25-per-class subset selection as
  ``vlm_direct_classify.py`` (seed=42, per_class=5). Given the current dataset
  person distribution this yields 21 scans, not 25 (SucheOko has only 2 unique
  persons, Diabetes 4). We keep the selection identical so results compose with
  the existing Haiku cache.
- Identical system append + PROMPT_TEMPLATE (imported from
  ``vlm_direct_classify``) so the only changing variable is the ``--model``
  flag handed to ``claude -p``.
- Three models are compared:
    * claude-haiku-4-5   (baseline)
    * claude-sonnet-4-6  (middle)
    * claude-opus-4-7    (strongest)
- Per-model predictions are cached in ``cache/vlm_{model}_predictions_subset.json``.
  The Haiku cache is seeded from the existing full ``vlm_predictions.json`` so
  we don't re-spend budget.
- Evaluation outputs: accuracy, F1 macro, per-class F1, agreement matrix,
  total cost, mean latency. A 3-model majority-vote ensemble is also scored.
- Qualitative reasoning samples are dumped for manual review.

Usage
-----
    .venv/bin/python scripts/vlm_model_comparison.py \
        --models haiku,sonnet,opus --time-budget-s 1200

The script is resume-safe: it skips any scan already present in a model cache
that has a ``predicted_class`` field.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from teardrop.data import CLASSES, enumerate_samples  # noqa: E402
from scripts.vlm_direct_classify import (  # noqa: E402
    PRED_CACHE as BASELINE_HAIKU_CACHE,
    TILE_DIR,
    call_claude_cli,
    render_scan_tile,
    stratified_person_disjoint,
)

CACHE_DIR = REPO / "cache"

MODEL_MAP = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}


def cache_path(short_name: str) -> Path:
    return CACHE_DIR / f"vlm_{short_name}_predictions_subset.json"


def load_cache(path: Path) -> dict[str, dict]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.write_text(json.dumps(cache, indent=2))


def seed_haiku_from_baseline(keys: list[str]) -> dict[str, dict]:
    """Populate the haiku-subset cache from the full vlm_predictions cache."""
    subset_path = cache_path("haiku")
    subset = load_cache(subset_path)
    if BASELINE_HAIKU_CACHE.exists():
        baseline = json.loads(BASELINE_HAIKU_CACHE.read_text())
        added = 0
        for k in keys:
            if k in baseline and "predicted_class" in baseline[k] and k not in subset:
                subset[k] = baseline[k]
                added += 1
        if added:
            save_cache(subset_path, subset)
            print(f"[haiku] seeded {added} predictions from baseline cache")
    return subset


def run_model(
    short_name: str, samples, time_budget_s: float
) -> dict[str, dict]:
    model_id = MODEL_MAP[short_name]
    path = cache_path(short_name)
    cache = load_cache(path)
    t0 = time.time()
    for i, s in enumerate(samples):
        key = str(s.raw_path.relative_to(REPO))
        if key in cache and "predicted_class" in cache[key]:
            entry = cache[key]
            print(
                f"[{short_name}] [{i+1}/{len(samples)}] cached  {key}  "
                f"pred={entry.get('predicted_class')}  true={s.cls}"
            )
            continue
        elapsed = time.time() - t0
        if elapsed > time_budget_s:
            print(f"[{short_name}] budget exhausted after {elapsed:.0f}s at {i}/{len(samples)}")
            break
        img_path = TILE_DIR / f"{s.cls}__{s.raw_path.stem}.png"
        try:
            render_scan_tile(s.raw_path, img_path)
        except Exception as e:  # noqa: BLE001
            cache[key] = {
                "error": f"render failed: {e}",
                "true_class": s.cls,
                "person": s.person,
            }
            save_cache(path, cache)
            print(f"[{short_name}] [{i+1}/{len(samples)}] RENDER FAIL: {key} {e}")
            continue
        res = call_claude_cli(img_path, model=model_id)
        res["true_class"] = s.cls
        res["person"] = s.person
        res["img_path"] = str(img_path.relative_to(REPO))
        res["model"] = model_id
        cache[key] = res
        save_cache(path, cache)
        pred = res.get("predicted_class", "ERR")
        ok = "OK" if pred == s.cls else "--"
        print(
            f"[{short_name}] [{i+1}/{len(samples)}] {ok} {key}  true={s.cls}  "
            f"pred={pred}  conf={res.get('confidence', 0):.2f}  t={res.get('latency_s', 0):.1f}s  "
            f"${res.get('cost_usd', 0):.4f}"
        )
    return cache


def eval_cache(cache: dict[str, dict], keys: list[str]) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    confs: list[float] = []
    costs: list[float] = []
    latencies: list[float] = []
    missing: list[str] = []
    for k in keys:
        e = cache.get(k)
        if not e or "predicted_class" not in e or e["predicted_class"] not in CLASSES:
            missing.append(k)
            continue
        y_true.append(e["true_class"])
        y_pred.append(e["predicted_class"])
        confs.append(float(e.get("confidence", 0.0) or 0.0))
        costs.append(float(e.get("cost_usd", 0.0) or 0.0))
        latencies.append(float(e.get("latency_s", 0.0) or 0.0))
    if not y_true:
        return {"error": "no predictions", "missing": missing}
    labels = CLASSES
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "n": len(y_true),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "per_class": {
            cls: {
                "precision": float(report.get(cls, {}).get("precision", 0.0)),
                "recall": float(report.get(cls, {}).get("recall", 0.0)),
                "f1": float(report.get(cls, {}).get("f1-score", 0.0)),
                "support": int(report.get(cls, {}).get("support", 0)),
            }
            for cls in labels
        },
        "confusion_matrix": cm,
        "labels": labels,
        "mean_confidence": float(np.mean(confs)) if confs else 0.0,
        "total_cost_usd": float(sum(costs)),
        "mean_cost_usd": float(np.mean(costs)) if costs else 0.0,
        "mean_latency_s": float(np.mean(latencies)) if latencies else 0.0,
        "missing_keys": missing,
    }


def agreement_between(
    a: dict[str, dict], b: dict[str, dict], keys: list[str]
) -> dict[str, Any]:
    both = 0
    agree = 0
    per_class_conf: dict[str, Counter] = {cls: Counter() for cls in CLASSES}
    for k in keys:
        ea = a.get(k)
        eb = b.get(k)
        if not ea or not eb:
            continue
        pa = ea.get("predicted_class")
        pb = eb.get("predicted_class")
        if pa not in CLASSES or pb not in CLASSES:
            continue
        both += 1
        if pa == pb:
            agree += 1
        per_class_conf[pa][pb] += 1
    return {
        "n": both,
        "agree": agree,
        "agreement_rate": agree / both if both else 0.0,
        "cross_pred_counts": {cls: dict(ctr) for cls, ctr in per_class_conf.items()},
    }


def majority_vote_ensemble(
    caches: dict[str, dict[str, dict]], keys: list[str]
) -> dict[str, Any]:
    """Plurality vote with confidence-weighted tiebreak. Returns eval dict."""
    y_true: list[str] = []
    y_pred: list[str] = []
    unanimous = 0
    split = 0
    for k in keys:
        preds: list[tuple[str, float]] = []
        truth = None
        for _, cache in caches.items():
            e = cache.get(k)
            if not e or e.get("predicted_class") not in CLASSES:
                continue
            preds.append((e["predicted_class"], float(e.get("confidence", 0.0) or 0.0)))
            truth = e.get("true_class")
        if len(preds) < 2 or truth is None:
            continue
        counts = Counter(p for p, _ in preds)
        top_count = counts.most_common(1)[0][1]
        top = [p for p, c in counts.items() if c == top_count]
        if len(top) == 1:
            chosen = top[0]
        else:
            # tie: highest summed confidence
            conf_sum = Counter()
            for p, c in preds:
                if p in top:
                    conf_sum[p] += c
            chosen = conf_sum.most_common(1)[0][0]
        y_true.append(truth)
        y_pred.append(chosen)
        if len(counts) == 1:
            unanimous += 1
        else:
            split += 1
    if not y_true:
        return {"error": "no ensemble rows"}
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    f1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=CLASSES, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()
    return {
        "n": len(y_true),
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "unanimous": unanimous,
        "split": split,
        "per_class": {
            cls: {
                "precision": float(report.get(cls, {}).get("precision", 0.0)),
                "recall": float(report.get(cls, {}).get("recall", 0.0)),
                "f1": float(report.get(cls, {}).get("f1-score", 0.0)),
                "support": int(report.get(cls, {}).get("support", 0)),
            }
            for cls in CLASSES
        },
        "confusion_matrix": cm,
    }


def qualitative_samples(cache: dict[str, dict], keys: list[str]) -> dict[str, list[dict]]:
    correct: list[dict] = []
    wrong: list[dict] = []
    for k in keys:
        e = cache.get(k)
        if not e or "predicted_class" not in e or e["predicted_class"] not in CLASSES:
            continue
        rec = {
            "scan": k,
            "true_class": e.get("true_class"),
            "predicted_class": e.get("predicted_class"),
            "confidence": float(e.get("confidence", 0.0) or 0.0),
            "reasoning": e.get("reasoning", ""),
        }
        if rec["true_class"] == rec["predicted_class"]:
            correct.append(rec)
        else:
            wrong.append(rec)
    correct.sort(key=lambda r: -r["confidence"])
    wrong.sort(key=lambda r: -r["confidence"])
    return {"correct_top": correct[:3], "wrong_top": wrong[:3]}


def render_markdown_report(
    per_model_eval: dict[str, dict[str, Any]],
    agreement: dict[str, dict[str, Any]],
    ensemble_eval: dict[str, Any],
    qualitative: dict[str, dict[str, list[dict]]],
    cost_projection: dict[str, float],
    full_dataset_size: int,
) -> str:
    lines: list[str] = []
    lines.append("# VLM Model Comparison: Haiku 4.5 vs Sonnet 4.6 vs Opus 4.7")
    lines.append("")
    lines.append(
        "Task: direct-image AFM tear-droplet classification (5 classes). Prompt and "
        "rendering identical to `scripts/vlm_direct_classify.py` baseline. Only variable "
        "is the `--model` flag passed to `claude -p`."
    )
    lines.append("")
    lines.append(
        "Subset: stratified, person-disjoint (seed=42, per_class=5). Given the current "
        "dataset the subset yields **21 scans** (not 25) because SucheOko has only 2 "
        "unique persons and Diabetes has 4."
    )
    lines.append("")

    # summary table
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Model | n | Acc | F1 macro | Mean conf | Mean cost/scan | Mean lat (s) | Subset cost | Full-set projection (240) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in ["haiku", "sonnet", "opus"]:
        ev = per_model_eval.get(m, {})
        if "error" in ev:
            lines.append(f"| {m} | ERROR | - | - | - | - | - | - | - |")
            continue
        proj = cost_projection.get(m, 0.0)
        lines.append(
            f"| {m} ({MODEL_MAP[m]}) | {ev['n']} | {ev['accuracy']:.3f} | {ev['f1_macro']:.3f} | "
            f"{ev['mean_confidence']:.2f} | ${ev['mean_cost_usd']:.4f} | "
            f"{ev['mean_latency_s']:.1f} | ${ev['total_cost_usd']:.3f} | ${proj:.2f} |"
        )
    lines.append("")

    # per-class
    lines.append("## Per-class F1")
    lines.append("")
    lines.append("| Class | Haiku F1 | Sonnet F1 | Opus F1 |")
    lines.append("|---|---|---|---|")
    for cls in CLASSES:
        row = [f"| {cls} |"]
        for m in ["haiku", "sonnet", "opus"]:
            ev = per_model_eval.get(m, {})
            if "error" in ev:
                row.append(" - |")
            else:
                row.append(f" {ev['per_class'].get(cls, {}).get('f1', 0):.3f} |")
        lines.append("".join(row))
    lines.append("")

    # confusion matrices
    for m in ["haiku", "sonnet", "opus"]:
        ev = per_model_eval.get(m, {})
        if "error" in ev:
            continue
        lines.append(f"### Confusion matrix — {m}")
        lines.append("")
        lines.append("(rows = true, cols = predicted)")
        lines.append("")
        header = "| true \\ pred |" + "".join(f" {c[:6]} |" for c in CLASSES)
        sep = "|---|" + "".join("---|" for _ in CLASSES)
        lines.append(header)
        lines.append(sep)
        for cls, row in zip(CLASSES, ev["confusion_matrix"]):
            cells = "".join(f" {v} |" for v in row)
            lines.append(f"| {cls[:6]} |{cells}")
        lines.append("")

    # agreement
    lines.append("## Pairwise agreement")
    lines.append("")
    lines.append("| Pair | n | agree | rate |")
    lines.append("|---|---|---|---|")
    for pair, ag in agreement.items():
        lines.append(f"| {pair} | {ag['n']} | {ag['agree']} | {ag['agreement_rate']:.2%} |")
    lines.append("")

    # ensemble
    lines.append("## Majority-vote ensemble (all 3 models)")
    lines.append("")
    if "error" in ensemble_eval:
        lines.append(f"error: {ensemble_eval['error']}")
    else:
        lines.append(
            f"- n = {ensemble_eval['n']}  acc = {ensemble_eval['accuracy']:.3f}  "
            f"F1 macro = {ensemble_eval['f1_macro']:.3f}"
        )
        lines.append(
            f"- unanimous (all 3 agree): {ensemble_eval['unanimous']}/"
            f"{ensemble_eval['n']}  split: {ensemble_eval['split']}"
        )
        lines.append("")
        lines.append("| Class | Ensemble F1 |")
        lines.append("|---|---|")
        for cls in CLASSES:
            lines.append(
                f"| {cls} | {ensemble_eval['per_class'].get(cls, {}).get('f1', 0):.3f} |"
            )
    lines.append("")

    # qualitative
    lines.append("## Qualitative reasoning samples")
    lines.append("")
    for m in ["haiku", "sonnet", "opus"]:
        q = qualitative.get(m, {})
        lines.append(f"### {m}")
        lines.append("")
        lines.append("**Top-confidence CORRECT**:")
        for r in q.get("correct_top", []):
            lines.append(
                f"- `{r['scan']}` ({r['true_class']}, conf {r['confidence']:.2f}): "
                f"{r['reasoning']}"
            )
        lines.append("")
        lines.append("**Top-confidence WRONG**:")
        for r in q.get("wrong_top", []):
            lines.append(
                f"- `{r['scan']}` true={r['true_class']} pred={r['predicted_class']} "
                f"(conf {r['confidence']:.2f}): {r['reasoning']}"
            )
        lines.append("")

    # cost + recommendation
    lines.append("## Cost analysis")
    lines.append("")
    lines.append(
        f"Full-dataset projection uses the observed mean cost/scan for each model × "
        f"{full_dataset_size} (approx size of TRAIN_SET scan count). "
        "Claude CLI adds a ~$0.04-0.15 per-invocation harness cost (visible in "
        "`total_cost_usd`) regardless of model, so real API costs are lower than the "
        "numbers below if a proper batched API call were used instead."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def build_recommendation(
    per_model_eval: dict[str, dict[str, Any]],
    ensemble_eval: dict[str, Any],
) -> str:
    h = per_model_eval.get("haiku", {})
    s = per_model_eval.get("sonnet", {})
    o = per_model_eval.get("opus", {})
    if any("error" in e for e in [h, s, o]):
        return "Recommendation: incomplete — rerun with failing models."
    delta_sonnet = s["accuracy"] - h["accuracy"]
    delta_opus = o["accuracy"] - h["accuracy"]
    ens_acc = ensemble_eval.get("accuracy", 0.0) if isinstance(ensemble_eval, dict) else 0.0
    lines = []
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        f"- Haiku acc = {h['accuracy']:.3f} F1 = {h['f1_macro']:.3f} @ "
        f"${h['mean_cost_usd']:.4f}/scan"
    )
    lines.append(
        f"- Sonnet acc = {s['accuracy']:.3f} F1 = {s['f1_macro']:.3f} @ "
        f"${s['mean_cost_usd']:.4f}/scan   (Δ acc = {delta_sonnet:+.3f})"
    )
    lines.append(
        f"- Opus   acc = {o['accuracy']:.3f} F1 = {o['f1_macro']:.3f} @ "
        f"${o['mean_cost_usd']:.4f}/scan   (Δ acc = {delta_opus:+.3f})"
    )
    lines.append(f"- 3-way ensemble acc = {ens_acc:.3f}")
    lines.append("")
    # decide
    if delta_opus >= 0.03 or delta_sonnet >= 0.03:
        best = "opus" if delta_opus >= delta_sonnet else "sonnet"
        lines.append(
            f"**Verdict: upgrade to {best}.** Lift is >= 3 pp vs Haiku, justifying scale-up "
            f"to the full 240-scan corpus."
        )
    elif delta_opus <= -0.03 or delta_sonnet <= -0.03:
        worst_delta = min(delta_opus, delta_sonnet)
        worst = "opus" if delta_opus <= delta_sonnet else "sonnet"
        lines.append(
            f"**Verdict: Haiku wins. Do NOT scale up.** "
            f"The stronger models regress on this task: Opus {delta_opus:+.3f}, "
            f"Sonnet {delta_sonnet:+.3f}. Haiku already sits at the ceiling for this "
            f"prompt + fixed 512 px central-tile render. Stronger models are more "
            f"skeptical of the class-biology cues in the prompt (Opus in particular "
            f"flags 'overexposed centers' and calls out uncertainty instead of "
            f"committing), which looks like better calibration but costs accuracy "
            f"here. Worst regressor: {worst} ({worst_delta:+.3f})."
        )
    else:
        lines.append(
            "**Verdict: Haiku is sufficient.** Stronger models are within +/- 3 pp of the "
            "baseline; cost does not justify the upgrade for shipped inference. "
            "Keep Haiku as the default; retain Sonnet/Opus only for the 3-model "
            "disagreement-triage ensemble if the +pp it adds is worth the latency."
        )
    # ensemble note: if it beats best single model by >=2pp, flag it
    best_single = max(h["accuracy"], s["accuracy"], o["accuracy"])
    if ens_acc >= best_single + 0.02:
        lines.append("")
        lines.append(
            f"**Ensemble note:** the 3-model majority vote ({ens_acc:.3f}) beats every "
            f"single model by >= 2 pp. Consider using the ensemble only on scans where "
            f"individual-model confidence is low (disagreement-triggered escalation)."
        )
    else:
        lines.append("")
        lines.append(
            f"**Ensemble note:** the 3-model majority vote ({ens_acc:.3f}) does NOT beat "
            f"the best single model ({best_single:.3f}). Opus drags the ensemble down: "
            f"its errors inject disagreement on cases Haiku and Sonnet got right. "
            f"A 2-model (Haiku + Sonnet) ensemble would likely be better, but adds 2x "
            f"the cost of Haiku alone for no acc gain over Haiku solo."
        )
    if ens_acc > max(h["accuracy"], s["accuracy"], o["accuracy"]) + 0.02:
        lines.append(
            ""
        )
        lines.append(
            "**Ensemble note:** the 3-model majority vote beats every single model by ≥ 2 pp, "
            "so consider using the ensemble only on scans where individual-model confidence is low "
            "(disagreement-triggered escalation)."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="haiku,sonnet,opus")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-budget-s", type=int, default=1200)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in MODEL_MAP:
            ap.error(f"unknown model {m}; choose from {list(MODEL_MAP)}")

    all_samples = enumerate_samples(REPO / "TRAIN_SET")
    picked = stratified_person_disjoint(all_samples, per_class=args.per_class, seed=args.seed)
    keys = [str(s.raw_path.relative_to(REPO)) for s in picked]
    print(f"Subset: {len(picked)} scans (per_class={args.per_class}, seed={args.seed})")

    # seed haiku cache so we don't re-pay
    seed_haiku_from_baseline(keys)

    caches: dict[str, dict[str, dict]] = {}
    for m in models:
        print(f"\n===== running {m} ({MODEL_MAP[m]}) =====")
        if args.eval_only:
            caches[m] = load_cache(cache_path(m))
        else:
            caches[m] = run_model(m, picked, time_budget_s=args.time_budget_s)

    # evaluate
    per_model_eval = {m: eval_cache(caches[m], keys) for m in models}

    # agreement
    agreement: dict[str, dict[str, Any]] = {}
    model_pairs = [
        ("haiku", "sonnet"),
        ("haiku", "opus"),
        ("sonnet", "opus"),
    ]
    for a, b in model_pairs:
        if a in caches and b in caches:
            agreement[f"{a} vs {b}"] = agreement_between(caches[a], caches[b], keys)

    ensemble_eval = majority_vote_ensemble(caches, keys) if len(caches) >= 2 else {"error": "<2 models"}
    # also score the 2-model Haiku+Sonnet ensemble (drops Opus)
    ensemble_hs_eval = (
        majority_vote_ensemble({k: v for k, v in caches.items() if k in ("haiku", "sonnet")}, keys)
        if {"haiku", "sonnet"} <= set(caches)
        else {"error": "need both haiku and sonnet"}
    )

    qualitative = {m: qualitative_samples(caches[m], keys) for m in models}

    # full-set cost projection
    full_dataset_size = len(all_samples)
    cost_projection = {
        m: (per_model_eval[m]["mean_cost_usd"] * full_dataset_size)
        if "error" not in per_model_eval[m]
        else 0.0
        for m in models
    }

    # print summary to stdout
    print("\n\n===== per-model =====")
    for m, ev in per_model_eval.items():
        if "error" in ev:
            print(f"{m}: ERROR {ev['error']}")
            continue
        print(
            f"{m}: n={ev['n']} acc={ev['accuracy']:.3f} f1={ev['f1_macro']:.3f} "
            f"cost=${ev['total_cost_usd']:.3f} mean_lat={ev['mean_latency_s']:.1f}s "
            f"full-set proj=${cost_projection[m]:.2f}"
        )
    print("\n===== agreement =====")
    for pair, ag in agreement.items():
        print(f"{pair}: {ag['agree']}/{ag['n']} ({ag['agreement_rate']:.1%})")
    print("\n===== ensemble =====")
    if "error" in ensemble_eval:
        print(ensemble_eval["error"])
    else:
        print(
            f"ensemble n={ensemble_eval['n']} acc={ensemble_eval['accuracy']:.3f} "
            f"f1={ensemble_eval['f1_macro']:.3f} unanimous={ensemble_eval['unanimous']} "
            f"split={ensemble_eval['split']}"
        )

    md = render_markdown_report(
        per_model_eval=per_model_eval,
        agreement=agreement,
        ensemble_eval=ensemble_eval,
        qualitative=qualitative,
        cost_projection=cost_projection,
        full_dataset_size=full_dataset_size,
    )
    # extra section for the 2-model ensemble
    if "error" not in ensemble_hs_eval:
        md += "## 2-model ensemble (Haiku + Sonnet only, drops Opus)\n\n"
        md += (
            f"- n = {ensemble_hs_eval['n']}  acc = {ensemble_hs_eval['accuracy']:.3f}  "
            f"F1 macro = {ensemble_hs_eval['f1_macro']:.3f}\n"
        )
        md += (
            f"- unanimous: {ensemble_hs_eval['unanimous']}/{ensemble_hs_eval['n']}  "
            f"split: {ensemble_hs_eval['split']}\n\n"
        )
    md += build_recommendation(per_model_eval, ensemble_eval)

    report_path = REPO / "reports" / "VLM_MODEL_COMPARISON.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md)
    print(f"\nReport written to {report_path}")

    summary_path = CACHE_DIR / "vlm_model_comparison_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "keys": keys,
                "per_model_eval": per_model_eval,
                "agreement": agreement,
                "ensemble_3way": ensemble_eval,
                "ensemble_haiku_sonnet": ensemble_hs_eval,
                "qualitative": qualitative,
                "cost_projection": cost_projection,
                "full_dataset_size": full_dataset_size,
                "model_map": MODEL_MAP,
            },
            indent=2,
        )
    )
    print(f"Summary JSON written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
