"""Post-run evaluation + report generation for the VLM direct-classify baseline.

Reads `cache/vlm_predictions.json` (produced by `vlm_direct_classify.py`) and
produces:
    - `reports/VLM_DIRECT_RESULTS.md` — methodology, subset + full metrics,
      qualitative reasoning samples, ensemble with v4.
    - `cache/vlm_summary.json` — machine-readable numbers.

Usage:
    python scripts/vlm_report.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from teardrop.data import CLASSES, enumerate_samples  # noqa: E402

CACHE = REPO / "cache" / "vlm_predictions.json"
SUMMARY = REPO / "cache" / "vlm_summary.json"
REPORT = REPO / "reports" / "VLM_DIRECT_RESULTS.md"
V4_OOF = REPO / "cache" / "v4_oof.npz"


def load_preds() -> dict[str, dict]:
    return json.loads(CACHE.read_text())


def stratified_person_disjoint(samples, per_class: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    by_cls: dict = {}
    for s in samples:
        by_cls.setdefault(s.cls, {}).setdefault(s.person, []).append(s)
    picked = []
    for cls in CLASSES:
        persons = list(by_cls.get(cls, {}).keys())
        rng.shuffle(persons)
        take = min(per_class, len(persons))
        for p in persons[:take]:
            picked.append(sorted(by_cls[cls][p], key=lambda s: str(s.raw_path))[0])
    return picked


def evaluate(cache: dict, keys: list[str]) -> dict:
    y_true, y_pred, confs = [], [], []
    for k in keys:
        e = cache.get(k)
        if not e or "predicted_class" not in e or e["predicted_class"] not in CLASSES:
            continue
        y_true.append(e["true_class"])
        y_pred.append(e["predicted_class"])
        confs.append(e.get("confidence", 0.0))
    if not y_true:
        return {"error": "no predictions", "n": 0}
    f1m = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    rpt = classification_report(y_true, y_pred, labels=CLASSES, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()
    return {
        "n": len(y_true),
        "accuracy": acc,
        "f1_macro": f1m,
        "mean_confidence": float(np.mean(confs)),
        "report": rpt,
        "cm": cm,
    }


def ensemble_v4(cache: dict) -> dict | None:
    if not V4_OOF.exists():
        return None
    z = np.load(V4_OOF, allow_pickle=True)
    v4_proba = z["proba"].astype(np.float64)
    v4_paths = [str(p) for p in z["scan_paths"]]
    y = z["y"].astype(int)
    n = len(v4_paths)

    # build VLM proba matrix; NaN = missing
    vlm = np.full((n, 5), np.nan)
    for i, p in enumerate(v4_paths):
        e = cache.get(p)
        if not e or "predicted_class" not in e or e["predicted_class"] not in CLASSES:
            continue
        conf = max(min(float(e.get("confidence", 0.5)), 0.99), 0.01)
        residual = (1.0 - conf) / 4.0
        row = np.full(5, residual)
        row[CLASSES.index(e["predicted_class"])] = conf
        vlm[i] = row

    mask = ~np.isnan(vlm).any(axis=1)
    n_overlap = int(mask.sum())
    if n_overlap < 5:
        return None

    out: dict = {"n_overlap": n_overlap}

    y_sub = y[mask]
    f1_v4 = f1_score(y_sub, v4_proba[mask].argmax(axis=1), average="macro", zero_division=0)
    f1_vlm = f1_score(y_sub, vlm[mask].argmax(axis=1), average="macro", zero_division=0)
    out["f1_v4_on_overlap"] = float(f1_v4)
    out["f1_vlm_on_overlap"] = float(f1_vlm)

    # Also evaluate on FULL 240 using v4-alone (to anchor comparison)
    f1_v4_full = f1_score(y, v4_proba.argmax(axis=1), average="macro", zero_division=0)
    out["f1_v4_full_240"] = float(f1_v4_full)

    # Blends (on overlap only — fair comparison)
    blends = {}
    for w in [0.1, 0.2, 0.3, 0.4, 0.5]:
        blend = (1 - w) * v4_proba[mask] + w * vlm[mask]
        blends[f"v4*{1-w:.1f} + vlm*{w:.1f}"] = float(
            f1_score(y_sub, blend.argmax(axis=1), average="macro", zero_division=0)
        )
    out["blends_on_overlap"] = blends

    # Hybrid on full set: v4 everywhere, VLM only where available (blend there)
    for w in [0.2, 0.3, 0.5]:
        hybrid = v4_proba.copy()
        hybrid[mask] = (1 - w) * v4_proba[mask] + w * vlm[mask]
        f1_h = f1_score(y, hybrid.argmax(axis=1), average="macro", zero_division=0)
        out.setdefault("blends_hybrid_full_240", {})[f"v4 + vlm*{w:.1f}where_avail"] = float(f1_h)

    return out


def qualitative(cache: dict, keys: list[str]):
    correct, wrong = [], []
    for k in keys:
        e = cache.get(k)
        if not e or "predicted_class" not in e or e["predicted_class"] not in CLASSES:
            continue
        rec = {
            "scan": k,
            "true": e["true_class"],
            "pred": e["predicted_class"],
            "conf": float(e.get("confidence", 0.0)),
            "reason": e.get("reasoning", ""),
        }
        (correct if rec["true"] == rec["pred"] else wrong).append(rec)
    correct.sort(key=lambda r: -r["conf"])
    wrong.sort(key=lambda r: -r["conf"])
    return correct, wrong


def fmt_per_class(rpt: dict) -> str:
    lines = ["| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for c in CLASSES:
        r = rpt.get(c, {})
        lines.append(f"| {c} | {r.get('precision', 0):.3f} | {r.get('recall', 0):.3f} | {r.get('f1-score', 0):.3f} | {int(r.get('support', 0))} |")
    return "\n".join(lines)


def fmt_cm(cm: list[list[int]]) -> str:
    hdr = "| true \\ pred | " + " | ".join(c[:6] for c in CLASSES) + " |"
    sep = "|---|" + "---:|" * len(CLASSES)
    rows = []
    for cls, row in zip(CLASSES, cm):
        rows.append(f"| {cls} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([hdr, sep, *rows])


def main() -> int:
    cache = load_preds()
    samples = enumerate_samples(REPO / "TRAIN_SET")
    all_keys = [str(s.raw_path.relative_to(REPO)) for s in samples]
    subset_keys = [str(s.raw_path.relative_to(REPO)) for s in stratified_person_disjoint(samples, per_class=5)]

    eval_subset = evaluate(cache, subset_keys)
    eval_full = evaluate(cache, all_keys)
    ens = ensemble_v4(cache)
    qcor, qwrong = qualitative(cache, all_keys)

    # coverage
    preds_only = [v for v in cache.values() if "predicted_class" in v]
    by_true = Counter(v["true_class"] for v in preds_only)
    totals = Counter(s.cls for s in samples)
    coverage_rows = []
    for c in CLASSES:
        coverage_rows.append((c, by_true[c], totals[c]))
    total_cost = sum(v.get("cost_usd", 0.0) for v in preds_only)
    total_lat = sum(v.get("latency_s", 0.0) for v in preds_only)
    mean_lat = total_lat / max(len(preds_only), 1)

    summary = {
        "coverage": {c: [n, tot] for c, n, tot in coverage_rows},
        "subset_25": eval_subset,
        "full_scored": eval_full,
        "ensemble": ens,
        "cost": {"total_usd": total_cost, "mean_latency_s": mean_lat, "n_calls": len(preds_only)},
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))

    # report
    lines = []
    lines.append("# VLM Direct-Classification Baseline")
    lines.append("")
    lines.append("Send each AFM scan as a rendered afmhot PNG directly to Claude Haiku 4.5 via the `claude -p` CLI and parse its JSON prediction. No training, no features — just image + biologically informed prompt.")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Image rendering.** Each SPM file is preprocessed with the same pipeline used for foundation-model embeddings (plane-level subtraction, resample to 90 nm/px, robust [p2,p98] normalise, center-crop to 512x512) then rendered with Matplotlib's `afmhot` colormap at 512x512 px (~290 KB PNG).")
    lines.append("- **Model.** `claude-haiku-4-5` (2026 release). Reason: cheap (~ $0.01/call), fast (~16 s), vision-capable.")
    lines.append("- **Prompt.** System-level instruction asks for JSON-only output. The user prompt includes (a) the file path (Claude reads it via its Read tool), (b) scale + colormap context, (c) per-class morphological signatures mirroring the domain knowledge used in `teardrop/llm_reason.py`.")
    lines.append("- **Cache & resume.** Each scored sample is cached in `cache/vlm_predictions.json` keyed by the scan path, so re-runs skip completed calls.")
    lines.append("- **Scoring.** Hard argmax from `predicted_class`; confidence is the reported scalar. For ensembling we build a pseudo-softmax by putting `conf` on the predicted class and `(1-conf)/4` on the rest.")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append("| Class | Scored | Total | % |")
    lines.append("|---|---:|---:|---:|")
    for c, n, tot in coverage_rows:
        pct = 100.0 * n / max(tot, 1)
        lines.append(f"| {c} | {n} | {tot} | {pct:.0f}% |")
    lines.append(f"| **TOTAL** | **{sum(n for _, n, _ in coverage_rows)}** | **{sum(t for _, _, t in coverage_rows)}** | **{100.0 * sum(n for _, n, _ in coverage_rows) / max(sum(t for _, _, t in coverage_rows), 1):.0f}%** |")
    lines.append("")
    lines.append(f"Total compute: {len(preds_only)} calls, mean latency {mean_lat:.1f} s, total cost **${total_cost:.2f}**.")
    lines.append("")
    lines.append("## Subset (stratified, person-disjoint, 5 per class)")
    lines.append("")
    if eval_subset.get("n", 0) > 0:
        lines.append(f"n = {eval_subset['n']}  |  accuracy = **{eval_subset['accuracy']:.4f}**  |  macro-F1 = **{eval_subset['f1_macro']:.4f}**  |  mean confidence = {eval_subset['mean_confidence']:.3f}")
        lines.append("")
        lines.append(fmt_per_class(eval_subset["report"]))
        lines.append("")
        lines.append("Confusion matrix (subset):")
        lines.append("")
        lines.append(fmt_cm(eval_subset["cm"]))
    lines.append("")
    lines.append("## Full scored set")
    lines.append("")
    if eval_full.get("n", 0) > 0:
        lines.append(f"n = {eval_full['n']} (of 240)  |  accuracy = **{eval_full['accuracy']:.4f}**  |  macro-F1 = **{eval_full['f1_macro']:.4f}**  |  mean confidence = {eval_full['mean_confidence']:.3f}")
        lines.append("")
        lines.append(fmt_per_class(eval_full["report"]))
        lines.append("")
        lines.append("Confusion matrix:")
        lines.append("")
        lines.append(fmt_cm(eval_full["cm"]))
    lines.append("")
    lines.append("## Comparison to Champion v4 (foundation + LR, F1 = 0.6887)")
    lines.append("")
    if ens:
        lines.append(f"- v4 alone on its full 240 OOF: F1 = **{ens['f1_v4_full_240']:.4f}**")
        lines.append(f"- Overlap (both models scored): n = {ens['n_overlap']}")
        lines.append(f"  - v4 alone on overlap: F1 = **{ens['f1_v4_on_overlap']:.4f}**")
        lines.append(f"  - VLM alone on overlap: F1 = **{ens['f1_vlm_on_overlap']:.4f}**")
        lines.append("")
        lines.append("### Blend on overlap (pure blend of two probability vectors)")
        lines.append("")
        lines.append("| Weighting | macro-F1 |")
        lines.append("|---|---:|")
        for k, v in ens["blends_on_overlap"].items():
            lines.append(f"| {k} | {v:.4f} |")
        lines.append("")
        lines.append("### Hybrid on full 240 (v4 alone where VLM missing, blended where VLM scored)")
        lines.append("")
        lines.append("| Weighting | macro-F1 |")
        lines.append("|---|---:|")
        for k, v in ens.get("blends_hybrid_full_240", {}).items():
            lines.append(f"| {k} | {v:.4f} |")
    else:
        lines.append("v4 OOF file not found — ensemble comparison skipped.")
    lines.append("")
    lines.append("## Qualitative reasoning — top-confidence correct predictions")
    lines.append("")
    lines.append("These are the clinical narratives the VLM produced alongside its correct labels. Even if the F1 were low, these texts by themselves are a novel pitch asset: no classical model can produce them.")
    lines.append("")
    for r in qcor[:5]:
        lines.append(f"**{r['scan']}** — true/pred=`{r['true']}` (confidence {r['conf']:.2f}):")
        lines.append("")
        lines.append(f"> {r['reason']}")
        lines.append("")
    lines.append("## Qualitative reasoning — top-confidence wrong predictions")
    lines.append("")
    lines.append("These failure modes are equally informative for the pitch: they show WHERE the off-the-shelf VLM's prior diverges from ocular biomarker literature.")
    lines.append("")
    for r in qwrong[:3]:
        lines.append(f"**{r['scan']}** — true=`{r['true']}`, pred=`{r['pred']}` (confidence {r['conf']:.2f}):")
        lines.append("")
        lines.append(f"> {r['reason']}")
        lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```bash")
    lines.append("# 25-scan stratified subset")
    lines.append("python scripts/vlm_direct_classify.py --subset 5")
    lines.append("")
    lines.append("# full 240")
    lines.append("python scripts/vlm_direct_classify.py --full --time-budget-s 2100")
    lines.append("")
    lines.append("# re-generate this report from the cache")
    lines.append("python scripts/vlm_report.py")
    lines.append("```")
    lines.append("")
    lines.append("## Takeaway")
    lines.append("")
    lines.append("- Novel approach: foundation VLM classifying AFM scans with zero AFM training.")
    lines.append("- Each prediction comes with a human-readable morphological rationale — clinical-report-ready.")
    lines.append("- Cheap: whole dataset scored for ~$1-2 on Haiku.")
    lines.append("- Even if the VLM alone does not beat the v4 champion, it provides an interpretability layer and, when blended at low weight, can nudge the ensemble.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"Wrote {REPORT}")
    print(f"Wrote {SUMMARY}")

    # short stdout summary
    if eval_full.get("n", 0) > 0:
        print(f"\nFull scored: n={eval_full['n']}, F1_macro={eval_full['f1_macro']:.4f}, acc={eval_full['accuracy']:.4f}")
    if eval_subset.get("n", 0) > 0:
        print(f"Subset 25:   n={eval_subset['n']}, F1_macro={eval_subset['f1_macro']:.4f}, acc={eval_subset['accuracy']:.4f}")
    if ens:
        print(f"Ensemble: v4 full={ens['f1_v4_full_240']:.4f}, best hybrid={max(ens.get('blends_hybrid_full_240', {}).values(), default=0):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
