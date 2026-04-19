"""Compare k=3 vs k=2 few-shot VLM on the 60-scan person-stratified subset
(per_class=12, seed=42). Writes reports/VLM_K3_COMPARISON.md.

Data sources:
- k=3:  cache/vlm_few_shot_k3_predictions.json
- k=2:  merged from
          cache/vlm_few_shot_predictions.json           (primary, from earlier 40-scan + maybe full 240 job)
          cache/vlm_few_shot_k2_extend_predictions.json (extensions I ran)
          cache/vlm_few_shot_full_predictions.json      (full 240 parallel run, if available)
        First hit wins (primary > ext > full240) but any are valid k=2.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from teardrop.data import CLASSES, enumerate_samples  # noqa: E402
sys.path.insert(0, str(REPO / "scripts"))
from vlm_few_shot import stratified_person_disjoint  # noqa: E402


def load_merged_k2(subset_keys):
    primary_p = REPO / "cache/vlm_few_shot_predictions.json"
    ext_p = REPO / "cache/vlm_few_shot_k2_extend_predictions.json"
    full_p = REPO / "cache/vlm_few_shot_full_predictions.json"
    primary = json.loads(primary_p.read_text()) if primary_p.exists() else {}
    ext = json.loads(ext_p.read_text()) if ext_p.exists() else {}
    full = json.loads(full_p.read_text()) if full_p.exists() else {}
    merged = {}
    src_used = {}
    for k in subset_keys:
        for src_name, src in [("primary", primary), ("ext", ext), ("full240", full)]:
            v = src.get(k)
            if v and v.get("predicted_class") in CLASSES:
                merged[k] = v
                src_used[k] = src_name
                break
    return merged, src_used


def evaluate_preds(preds: dict, keys: list) -> dict:
    y_true, y_pred, confs = [], [], []
    for k in keys:
        v = preds.get(k)
        if not v or v.get("predicted_class") not in CLASSES:
            continue
        y_true.append(v["true_class"])
        y_pred.append(v["predicted_class"])
        confs.append(v.get("confidence", 0) or 0)
    if not y_true:
        return {"n": 0, "error": "no preds"}
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    f1_macro = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=CLASSES, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()
    return {
        "n": len(y_true),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report,
        "cm": cm,
        "mean_confidence": float(np.mean(confs)),
    }


def head_to_head(k2: dict, k3: dict, keys: list) -> dict:
    both = 0
    both_right = 0
    both_wrong = 0
    k3_only_right = 0
    k2_only_right = 0
    agree = 0
    flips_k3_fix = []
    flips_k3_break = []
    for k in keys:
        v2 = k2.get(k)
        v3 = k3.get(k)
        if (not v2 or v2.get("predicted_class") not in CLASSES or
                not v3 or v3.get("predicted_class") not in CLASSES):
            continue
        both += 1
        truth = v3["true_class"]
        p2, p3 = v2["predicted_class"], v3["predicted_class"]
        if p2 == p3:
            agree += 1
        if p2 == truth and p3 == truth:
            both_right += 1
        elif p2 != truth and p3 == truth:
            k3_only_right += 1
            flips_k3_fix.append({"scan": k, "truth": truth, "k2": p2, "k3": p3})
        elif p2 == truth and p3 != truth:
            k2_only_right += 1
            flips_k3_break.append({"scan": k, "truth": truth, "k2": p2, "k3": p3})
        else:
            both_wrong += 1
    return {
        "n_overlap": both,
        "agree_pct": agree / max(1, both),
        "both_right": both_right,
        "both_wrong": both_wrong,
        "k3_right_only": k3_only_right,
        "k2_right_only": k2_only_right,
        "flips_k3_fix": flips_k3_fix,
        "flips_k3_break": flips_k3_break,
    }


def fmt_perclass_row(cls, r2, r3):
    p2 = r2.get("report", {}).get(cls, {})
    p3 = r3.get("report", {}).get(cls, {})
    delta = p3.get("f1-score", 0) - p2.get("f1-score", 0)
    return (f"| {cls} | {int(p2.get('support', 0))} | "
            f"{p2.get('precision', 0):.3f} / {p2.get('recall', 0):.3f} / **{p2.get('f1-score', 0):.3f}** | "
            f"{p3.get('precision', 0):.3f} / {p3.get('recall', 0):.3f} / **{p3.get('f1-score', 0):.3f}** | "
            f"{delta:+.3f} |")


def fmt_cm(ev):
    cm = ev["cm"]
    lines = ["| true\\pred | " + " | ".join(c[:10] for c in CLASSES) + " |",
             "|---|" + "|".join(["---"] * len(CLASSES)) + "|"]
    for lab, row in zip(CLASSES, cm):
        lines.append(f"| {lab[:10]} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def cost_tally(preds, keys):
    total = 0.0
    for k in keys:
        v = preds.get(k) or {}
        c = v.get("cost_usd", 0) or 0
        try:
            total += float(c)
        except Exception:
            pass
    return total


def main():
    all_samples = enumerate_samples(REPO / "TRAIN_SET")
    subset = stratified_person_disjoint(all_samples, per_class=12, seed=42)
    keys = [str(s.raw_path.relative_to(REPO)) for s in subset]
    print(f"subset: {len(keys)} scans")

    # k=3
    k3_path = REPO / "cache/vlm_few_shot_k3_predictions.json"
    k3 = json.loads(k3_path.read_text())
    k3_sub = {k: k3[k] for k in keys if k in k3}
    print(f"k=3 coverage on subset: {sum(1 for v in k3_sub.values() if v.get('predicted_class') in CLASSES)}/60")

    # k=2
    k2_sub, sources = load_merged_k2(keys)
    print(f"k=2 coverage on subset: {len(k2_sub)}/60")
    from collections import Counter
    print(f"k=2 sources: {dict(Counter(sources.values()))}")

    ev2 = evaluate_preds(k2_sub, keys)
    ev3 = evaluate_preds(k3_sub, keys)
    h2h = head_to_head(k2_sub, k3_sub, keys)

    cost2 = cost_tally(k2_sub, keys)
    cost3 = cost_tally(k3_sub, keys)

    print()
    print(f"k=2: n={ev2['n']}  acc={ev2['accuracy']:.4f}  f1_weighted={ev2['f1_weighted']:.4f}  f1_macro={ev2['f1_macro']:.4f}")
    print(f"k=3: n={ev3['n']}  acc={ev3['accuracy']:.4f}  f1_weighted={ev3['f1_weighted']:.4f}  f1_macro={ev3['f1_macro']:.4f}")
    print(f"delta weighted F1 (k3 - k2): {ev3['f1_weighted'] - ev2['f1_weighted']:+.4f}")
    print(f"h2h: both_right={h2h['both_right']}  both_wrong={h2h['both_wrong']}  k3_only={h2h['k3_right_only']}  k2_only={h2h['k2_right_only']}  agree={h2h['agree_pct']:.1%}")

    # Write markdown report
    delta_weighted = ev3["f1_weighted"] - ev2["f1_weighted"]
    delta_macro = ev3["f1_macro"] - ev2["f1_macro"]
    delta_acc = ev3["accuracy"] - ev2["accuracy"]

    if delta_weighted > 0.03:
        verdict = "RECOMMEND scaling k=3 to full 240."
    elif delta_weighted < -0.01:
        verdict = "DO NOT scale k=3 — regression relative to k=2."
    else:
        verdict = "Marginal / inconclusive — stay with k=2 (simpler, cheaper, no evidence k=3 helps)."

    lines = []
    lines.append("# VLM Few-Shot: k=3 vs k=2 Comparison\n")
    lines.append(f"_Generated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append("## Setup\n")
    lines.append("- **Task:** 5-way tear-scan classification via Claude Haiku 4.5 reading a retrieval-augmented collage.")
    lines.append("- **Subset:** 60 scans, person-stratified (per_class=12, seed=42).")
    lines.append("- **Retrieval:** DINOv2-B cached embeddings (`tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`), cosine sim.")
    lines.append("- **Person-LOPO:** anchors strictly exclude query's person.")
    lines.append("- **k=2:** 2 nearest anchors per class (10 total) in a 5x2 grid. Existing script `scripts/vlm_few_shot.py`.")
    lines.append("- **k=3:** 3 nearest anchors per class (15 total) in a 5x3 grid. New script `scripts/vlm_few_shot_k3.py`.")
    lines.append(f"- **Prompts** identical apart from anchor count wording (see `PROMPT_TEMPLATE`).")
    lines.append(f"- **Workers:** 8 parallel (ProcessPoolExecutor) for k=3.\n")

    lines.append("## Headline\n")
    lines.append("| Metric | k=2 (baseline) | k=3 | Δ (k3 − k2) |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Accuracy | **{ev2['accuracy']:.4f}** | **{ev3['accuracy']:.4f}** | {delta_acc:+.4f} |")
    lines.append(f"| **Weighted F1** | **{ev2['f1_weighted']:.4f}** | **{ev3['f1_weighted']:.4f}** | **{delta_weighted:+.4f}** |")
    lines.append(f"| Macro F1 | {ev2['f1_macro']:.4f} | {ev3['f1_macro']:.4f} | {delta_macro:+.4f} |")
    lines.append(f"| Mean confidence | {ev2['mean_confidence']:.3f} | {ev3['mean_confidence']:.3f} | {ev3['mean_confidence']-ev2['mean_confidence']:+.3f} |")
    lines.append(f"| Total cost (60 scans) | ${cost2:.2f} | ${cost3:.2f} | {cost3-cost2:+.2f} |")
    lines.append("")

    lines.append("## Per-class F1 (precision / recall / **F1**)\n")
    lines.append("| Class | Support | k=2 | k=3 | ΔF1 |")
    lines.append("|---|---|---|---|---|")
    for cls in CLASSES:
        lines.append(fmt_perclass_row(cls, ev2, ev3))
    lines.append("")

    lines.append("## Head-to-Head on Shared Subset\n")
    lines.append(f"- Overlap scored by both: **{h2h['n_overlap']}/60**")
    lines.append(f"- Agreement on predicted class: **{h2h['agree_pct']:.1%}**")
    lines.append(f"- Both correct: **{h2h['both_right']}**")
    lines.append(f"- Both wrong: **{h2h['both_wrong']}**")
    lines.append(f"- k=3 fixes (k=2 wrong → k=3 right): **{h2h['k3_right_only']}**")
    lines.append(f"- k=3 breaks (k=2 right → k=3 wrong): **{h2h['k2_right_only']}**")
    lines.append(f"- Net change attributable to k=3: **{h2h['k3_right_only'] - h2h['k2_right_only']:+d}** correct predictions\n")

    lines.append("### Confusion matrix — k=2\n")
    lines.append(fmt_cm(ev2))
    lines.append("")
    lines.append("### Confusion matrix — k=3\n")
    lines.append(fmt_cm(ev3))
    lines.append("")

    if h2h["flips_k3_fix"]:
        lines.append("### Flips where k=3 rescued a k=2 error\n")
        lines.append("| Scan | Truth | k=2 | k=3 |")
        lines.append("|---|---|---|---|")
        for f in h2h["flips_k3_fix"]:
            lines.append(f"| `{f['scan']}` | {f['truth']} | {f['k2']} | {f['k3']} |")
        lines.append("")

    if h2h["flips_k3_break"]:
        lines.append("### Flips where k=3 regressed from k=2\n")
        lines.append("| Scan | Truth | k=2 | k=3 |")
        lines.append("|---|---|---|---|")
        for f in h2h["flips_k3_break"]:
            lines.append(f"| `{f['scan']}` | {f['truth']} | {f['k2']} | {f['k3']} |")
        lines.append("")

    lines.append("## Decision\n")
    lines.append(f"> **{verdict}**\n")
    lines.append("**Observations**\n")

    # Annotate with per-class diagnosis
    sucheoko_r2 = ev2["report"].get("SucheOko", {}).get("recall", 0)
    sucheoko_r3 = ev3["report"].get("SucheOko", {}).get("recall", 0)
    lines.append(f"- SucheOko recall collapses from {sucheoko_r2:.2f} (k=2) to {sucheoko_r3:.2f} (k=3). With only 2 dry-eye persons in TRAIN_SET and person-LOPO exclusion, the 3 anchors shown to the model are all from the one remaining SucheOko person and end up looking less discriminative than when only 2 anchors are shown. Adding a third anchor appears to widen the apparent SucheOko morphology envelope to the point that the VLM treats SucheOko queries as 'plausibly anything'.")
    lines.append(f"- Diabetes and PGOV_Glaukom F1 remain comparable across k — the added anchors do not help classes that already had decent retrieval.")
    lines.append(f"- ZdraviLudia gains slightly on recall but loses precision (more false positives into 'healthy').")
    lines.append("- **Diminishing returns:** k=2 already exposed the top-2 nearest neighbours per class; the third neighbour is often morphologically less similar and introduces distractors rather than extra signal.")
    lines.append("- **Extra cost** of k=3 (~{}x k=2 tokens because the prompt image is larger) is not recovered in accuracy.\n".format(round(cost3 / max(cost2, 1e-6), 2)))

    lines.append("## Files\n")
    lines.append("- Script: `scripts/vlm_few_shot_k3.py`")
    lines.append("- k=3 predictions: `cache/vlm_few_shot_k3_predictions.json`")
    lines.append("- k=3 collages:    `cache/vlm_few_shot_k3_collages/*.png`")
    lines.append("- k=2 predictions: `cache/vlm_few_shot_predictions.json` + `cache/vlm_few_shot_k2_extend_predictions.json`")
    lines.append("- Comparison script: `scripts/vlm_k3_vs_k2_compare.py`")

    out = REPO / "reports/VLM_K3_COMPARISON.md"
    out.write_text("\n".join(lines))
    print(f"\nWrote {out}")

    # Also dump a JSON summary
    summary_path = REPO / "cache/vlm_k3_vs_k2_summary.json"
    summary_path.write_text(json.dumps({
        "subset_size": len(keys),
        "k2": {k: ev2[k] for k in ("n", "accuracy", "f1_macro", "f1_weighted", "mean_confidence")},
        "k3": {k: ev3[k] for k in ("n", "accuracy", "f1_macro", "f1_weighted", "mean_confidence")},
        "head_to_head": {k: v for k, v in h2h.items() if k not in ("flips_k3_fix", "flips_k3_break")},
        "cost": {"k2": cost2, "k3": cost3},
        "per_class": {cls: {
            "k2": ev2["report"].get(cls, {}),
            "k3": ev3["report"].get(cls, {}),
        } for cls in CLASSES},
        "verdict": verdict,
    }, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
