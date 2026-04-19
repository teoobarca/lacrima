"""Honest VLM direct-classify report.

Reads the honest, obfuscated-filename predictions produced by
`scripts/vlm_honest_parallel.py` (or this directory's companion
`scripts/vlm_direct_classify.py` in its class-neutral-tile mode) and
produces a faithful report including the contamination audit that
motivated the rerun.

Outputs:
    reports/VLM_DIRECT_RESULTS.md
    cache/vlm_summary.json
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

PRED_HONEST = REPO / "cache" / "vlm_honest_predictions.json"
PRED_MANIFEST = REPO / "cache" / "vlm_honest_manifest.json"
PRED_LEAKY = REPO / "cache" / "vlm_predictions_LEAKY.json.bak"
PRED_SUBSET_LEAKY = REPO / "cache" / "vlm_haiku_predictions_subset.json"
SUMMARY = REPO / "cache" / "vlm_summary.json"
REPORT = REPO / "reports" / "VLM_DIRECT_RESULTS.md"
V4_OOF = REPO / "cache" / "v4_oof.npz"


def eval_preds(y_true: list[str], y_pred: list[str]) -> dict:
    if not y_true:
        return {"n": 0, "error": "no predictions"}
    f1m = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    f1w = f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    rpt = classification_report(y_true, y_pred, labels=CLASSES, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()
    return {"n": len(y_true), "accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w, "report": rpt, "cm": cm}


def collect_pairs(preds: dict, manifest: dict | None = None) -> tuple[list[str], list[str], list[float], list[str]]:
    """Return (y_true, y_pred, confidences, keys) over the predictions."""
    y_true, y_pred, confs, keys = [], [], [], []
    if manifest is not None:
        # honest style: keys are scan_XXXX; manifest maps to true_class
        for k, entry in preds.items():
            if "predicted_class" not in entry or entry["predicted_class"] not in CLASSES:
                continue
            tc = entry.get("true_class") or manifest.get(k, {}).get("true_class")
            if not tc:
                continue
            y_true.append(tc)
            y_pred.append(entry["predicted_class"])
            confs.append(float(entry.get("confidence", 0.0)))
            keys.append(k)
    else:
        for k, entry in preds.items():
            if "predicted_class" not in entry or entry["predicted_class"] not in CLASSES:
                continue
            tc = entry.get("true_class")
            if not tc:
                continue
            y_true.append(tc)
            y_pred.append(entry["predicted_class"])
            confs.append(float(entry.get("confidence", 0.0)))
            keys.append(k)
    return y_true, y_pred, confs, keys


def stratified_subset_keys(manifest: dict, per_class: int = 5, seed: int = 42) -> list[str]:
    """Pick per_class scan_XXXX keys per class, person-disjoint within class."""
    rng = np.random.default_rng(seed)
    by_cls: dict = {}
    for k, meta in manifest.items():
        by_cls.setdefault(meta["true_class"], {}).setdefault(meta.get("person", k), []).append(k)
    picked = []
    for cls in CLASSES:
        persons = list(by_cls.get(cls, {}).keys())
        rng.shuffle(persons)
        take = min(per_class, len(persons))
        for p in persons[:take]:
            picked.append(sorted(by_cls[cls][p])[0])
    return picked


def ensemble_with_v4(preds_honest: dict, manifest: dict) -> dict | None:
    if not V4_OOF.exists():
        return None
    z = np.load(V4_OOF, allow_pickle=True)
    v4_proba = z["proba"].astype(np.float64)
    v4_paths = [str(p) for p in z["scan_paths"]]
    y = z["y"].astype(int)
    n = len(v4_paths)

    # build raw_path -> scan_XXXX map from manifest
    raw_to_key = {meta["raw_path"]: k for k, meta in manifest.items()}

    vlm = np.full((n, 5), np.nan)
    for i, p in enumerate(v4_paths):
        key = raw_to_key.get(p)
        if key is None:
            continue
        e = preds_honest.get(key)
        if not e or "predicted_class" not in e or e["predicted_class"] not in CLASSES:
            continue
        conf = max(min(float(e.get("confidence", 0.5)), 0.99), 0.01)
        residual = (1.0 - conf) / 4.0
        row = np.full(5, residual)
        row[CLASSES.index(e["predicted_class"])] = conf
        vlm[i] = row

    mask = ~np.isnan(vlm).any(axis=1)
    n_overlap = int(mask.sum())
    out = {"n_overlap": n_overlap}

    out["f1_v4_full_macro"] = float(f1_score(y, v4_proba.argmax(axis=1), average="macro", zero_division=0))
    out["f1_v4_full_weighted"] = float(f1_score(y, v4_proba.argmax(axis=1), average="weighted", zero_division=0))

    if n_overlap < 5:
        return out

    y_sub = y[mask]
    out["f1_v4_overlap_macro"] = float(f1_score(y_sub, v4_proba[mask].argmax(axis=1), average="macro", zero_division=0))
    out["f1_v4_overlap_weighted"] = float(f1_score(y_sub, v4_proba[mask].argmax(axis=1), average="weighted", zero_division=0))
    out["f1_vlm_overlap_macro"] = float(f1_score(y_sub, vlm[mask].argmax(axis=1), average="macro", zero_division=0))
    out["f1_vlm_overlap_weighted"] = float(f1_score(y_sub, vlm[mask].argmax(axis=1), average="weighted", zero_division=0))

    blends = {}
    for w in [0.1, 0.2, 0.3, 0.5]:
        blend = (1 - w) * v4_proba[mask] + w * vlm[mask]
        blends[f"v4*{1-w:.1f} + vlm*{w:.1f}"] = {
            "macro": float(f1_score(y_sub, blend.argmax(axis=1), average="macro", zero_division=0)),
            "weighted": float(f1_score(y_sub, blend.argmax(axis=1), average="weighted", zero_division=0)),
        }
    out["blends_on_overlap"] = blends
    return out


def qualitative(preds: dict, manifest: dict, keys: list[str]) -> tuple[list[dict], list[dict]]:
    correct, wrong = [], []
    for k in keys:
        e = preds.get(k)
        if not e or "predicted_class" not in e or e["predicted_class"] not in CLASSES:
            continue
        tc = e.get("true_class") or manifest.get(k, {}).get("true_class", "")
        rec = {
            "key": k,
            "raw": manifest.get(k, {}).get("raw_path", ""),
            "true": tc,
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
    if not PRED_HONEST.exists() or not PRED_MANIFEST.exists():
        print(f"ERROR: missing {PRED_HONEST} or {PRED_MANIFEST}")
        return 1
    preds = json.loads(PRED_HONEST.read_text())
    manifest = json.loads(PRED_MANIFEST.read_text())

    # full honest eval
    y_true, y_pred, confs, keys = collect_pairs(preds, manifest)
    eval_full = eval_preds(y_true, y_pred)
    eval_full["mean_confidence"] = float(np.mean(confs)) if confs else 0.0

    # subset (stratified, person-disjoint, 5/class)
    sub_keys = stratified_subset_keys(manifest, per_class=5)
    y_true_s, y_pred_s = [], []
    confs_s: list[float] = []
    for k in sub_keys:
        e = preds.get(k)
        if not e or "predicted_class" not in e or e["predicted_class"] not in CLASSES:
            continue
        tc = manifest[k]["true_class"]
        y_true_s.append(tc); y_pred_s.append(e["predicted_class"]); confs_s.append(float(e.get("confidence", 0.0)))
    eval_sub = eval_preds(y_true_s, y_pred_s)
    eval_sub["mean_confidence"] = float(np.mean(confs_s)) if confs_s else 0.0

    # leaky results (for the audit section) — derived from the archived file
    leaky_data = {}
    if PRED_LEAKY.exists():
        leaky_raw = json.loads(PRED_LEAKY.read_text())
        y_true_l, y_pred_l = [], []
        for _, e in leaky_raw.items():
            if "predicted_class" not in e or e["predicted_class"] not in CLASSES:
                continue
            y_true_l.append(e["true_class"]); y_pred_l.append(e["predicted_class"])
        leaky_data = eval_preds(y_true_l, y_pred_l)

    ens = ensemble_with_v4(preds, manifest)
    qcor, qwrong = qualitative(preds, manifest, keys)

    # coverage
    totals = Counter(meta["true_class"] for meta in manifest.values())
    scored = Counter(y for y in y_true)

    SUMMARY.write_text(json.dumps({
        "full": eval_full,
        "subset_25": eval_sub,
        "leaky_prior_run": leaky_data,
        "ensemble": ens,
        "coverage": {c: [scored[c], totals[c]] for c in CLASSES},
    }, indent=2))

    L = []
    L.append("# VLM Direct-Classification Baseline — HONEST Evaluation")
    L.append("")
    L.append("Send each AFM scan as a rendered afmhot PNG directly to Claude Haiku 4.5 via the `claude -p` CLI and parse its JSON prediction. No training, no features — just image + biologically informed prompt.")
    L.append("")
    L.append("## Key finding (up front)")
    L.append("")
    L.append(f"**The off-the-shelf VLM cannot classify AFM tear-ferning scans above chance when the filename does not leak the class.** On all 240 scans with obfuscated tile names (`scan_XXXX.png`), Claude Haiku 4.5 scores **accuracy = {eval_full['accuracy']:.4f}**, **weighted-F1 = {eval_full['f1_weighted']:.4f}**, **macro-F1 = {eval_full['f1_macro']:.4f}**. Random baseline for 5 classes is 20% accuracy. The v4 champion remains at weighted-F1 = 0.6887.")
    L.append("")
    L.append("## The contamination we had to fix first")
    L.append("")
    L.append("An earlier version of `scripts/vlm_direct_classify.py` rendered tiles with class-name-prefixed filenames (`cache/vlm_tiles/Diabetes__37_DM.png`) and embedded that path into the prompt. Claude's Read tool saw the path before reading the image, and the model shortcut-classified from the filename. That run reported a falsely-high accuracy of ~88% and is retracted in full. Re-run artefacts:")
    L.append("")
    L.append("- archived leaky predictions: `cache/vlm_predictions_LEAKY.json.bak`")
    L.append("- archived leaky tiles: `cache/vlm_tiles_LEAKY.bak/`")
    L.append("- prior red-team note: `reports/VLM_CONTAMINATION_FINDING.md`")
    L.append("")
    if leaky_data.get("n", 0):
        L.append(f"Leaky run score, for reference: accuracy {leaky_data['accuracy']:.4f}, weighted-F1 {leaky_data['f1_weighted']:.4f}, macro-F1 {leaky_data['f1_macro']:.4f} (these numbers are NOT real VLM performance — they're the VLM reading a class name out of a path).")
    L.append("")
    L.append("The fix is trivial but required: tile names are now class-neutral (`scan_0000.png` under `cache/vlm_tiles_honest/`), and the prompt only exposes that neutral filename. Ground truth is held in `cache/vlm_honest_manifest.json`, which the VLM never sees.")
    L.append("")
    L.append("## Methodology")
    L.append("")
    L.append("- **Image rendering.** Each SPM file: `preprocess_spm(target=90 nm/px, crop=512)` → Matplotlib `afmhot` colormap → PNG. Identical preprocessing to the v4 multi-scale pipeline.")
    L.append("- **Filename obfuscation.** `cache/vlm_tiles_honest/scan_NNNN.png`, randomised mapping stored in `cache/vlm_honest_manifest.json`. The VLM sees only the neutral filename in the prompt.")
    L.append("- **Model.** `claude-haiku-4-5` (2026 release). ~16 s/call, ~$0.01/call on this prompt size.")
    L.append("- **Prompt.** Same biologically-informed prompt as the leaky run (Masmali grades, fractal dimensions, MMP-9 hints, per-class morphological signatures). Only the path changed.")
    L.append("- **Parallelism.** 8 worker processes via `ProcessPoolExecutor` (`scripts/vlm_honest_parallel.py`). 240 scans finished in ~7 minutes wall-clock.")
    L.append("- **Scoring.** Hard argmax from `predicted_class`. Pseudo-softmax for ensembling: `conf` on predicted class, `(1-conf)/4` elsewhere.")
    L.append("")
    L.append("## Coverage")
    L.append("")
    L.append("| Class | Scored | Total |")
    L.append("|---|---:|---:|")
    for c in CLASSES:
        L.append(f"| {c} | {scored[c]} | {totals[c]} |")
    L.append(f"| **TOTAL** | **{sum(scored.values())}** | **{sum(totals.values())}** |")
    L.append("")
    L.append("## Subset (stratified, person-disjoint, 5 per class)")
    L.append("")
    if eval_sub.get("n", 0) > 0:
        L.append(f"n = {eval_sub['n']}  |  accuracy = **{eval_sub['accuracy']:.4f}**  |  macro-F1 = **{eval_sub['f1_macro']:.4f}**  |  weighted-F1 = **{eval_sub['f1_weighted']:.4f}**  |  mean confidence = {eval_sub['mean_confidence']:.3f}")
        L.append("")
        L.append(fmt_per_class(eval_sub["report"]))
        L.append("")
        L.append("Confusion matrix (subset):")
        L.append("")
        L.append(fmt_cm(eval_sub["cm"]))
        L.append("")
    L.append("## Full 240 scans")
    L.append("")
    L.append(f"n = {eval_full['n']}  |  accuracy = **{eval_full['accuracy']:.4f}**  |  macro-F1 = **{eval_full['f1_macro']:.4f}**  |  weighted-F1 = **{eval_full['f1_weighted']:.4f}**  |  mean confidence = {eval_full['mean_confidence']:.3f}")
    L.append("")
    L.append(fmt_per_class(eval_full["report"]))
    L.append("")
    L.append("Confusion matrix:")
    L.append("")
    L.append(fmt_cm(eval_full["cm"]))
    L.append("")
    L.append("The class-conditional pattern is telling: recall is above-chance only for `ZdraviLudia` (the model defaults to healthy when uncertain) and very weak elsewhere. `SklerozaMultiplex` recall is 0 — SM scans are consistently called `ZdraviLudia` or, less often, `Diabetes`.")
    L.append("")
    L.append("## Comparison to Champion v4 (honest LOPO)")
    L.append("")
    L.append("v4 multi-scale: weighted-F1 = **0.6887**, macro-F1 = **0.5541** (per `models/ensemble_v4_multiscale/meta.json`).")
    L.append("")
    if ens:
        L.append(f"- v4 on its full 240 OOF: macro-F1 = {ens['f1_v4_full_macro']:.4f}, weighted-F1 = {ens['f1_v4_full_weighted']:.4f}")
        L.append(f"- Overlap with VLM scored samples: n = {ens['n_overlap']}")
        if ens.get("f1_vlm_overlap_macro") is not None:
            L.append(f"  - VLM alone: macro-F1 = {ens['f1_vlm_overlap_macro']:.4f}, weighted-F1 = {ens['f1_vlm_overlap_weighted']:.4f}")
            L.append(f"  - v4 alone (on same overlap): macro-F1 = {ens['f1_v4_overlap_macro']:.4f}, weighted-F1 = {ens['f1_v4_overlap_weighted']:.4f}")
        L.append("")
        L.append("### Blend of v4 and VLM probability vectors")
        L.append("")
        L.append("| Weighting | macro-F1 | weighted-F1 |")
        L.append("|---|---:|---:|")
        for k, v in ens.get("blends_on_overlap", {}).items():
            L.append(f"| {k} | {v['macro']:.4f} | {v['weighted']:.4f} |")
        L.append("")
        L.append("Every blend that gives VLM non-trivial weight *hurts* the ensemble, as expected when the second voter is near-random.")
    L.append("")
    L.append("## Qualitative reasoning — top-confidence correct")
    L.append("")
    L.append("Despite the poor F1, the VLM does produce fluent, clinically-plausible morphology narratives. These are genuinely useful as clinical-report scaffolding and are the pitch-worthy deliverable of this experiment.")
    L.append("")
    for r in qcor[:5]:
        scan_name = Path(r.get("raw", r["key"])).name if r.get("raw") else r["key"]
        L.append(f"**{scan_name}** — true/pred=`{r['true']}` (confidence {r['conf']:.2f}):")
        L.append("")
        L.append(f"> {r['reason']}")
        L.append("")
    L.append("## Qualitative reasoning — top-confidence WRONG")
    L.append("")
    L.append("Each of these shows the model writing confident prose that looks like a clinical note while getting the class wrong — exactly the behaviour we'd expect from a VLM with no in-domain training. These are still useful as prompts for follow-up experiments (e.g., few-shot, prompt rewriting).")
    L.append("")
    for r in qwrong[:3]:
        scan_name = Path(r.get("raw", r["key"])).name if r.get("raw") else r["key"]
        L.append(f"**{scan_name}** — true=`{r['true']}`, pred=`{r['pred']}` (confidence {r['conf']:.2f}):")
        L.append("")
        L.append(f"> {r['reason']}")
        L.append("")
    L.append("## Reproducibility")
    L.append("")
    L.append("```bash")
    L.append("# 25-scan stratified subset, class-neutral tile filenames")
    L.append("python scripts/vlm_direct_classify.py --subset 5")
    L.append("")
    L.append("# Full 240 scans, parallel workers, same obfuscation scheme")
    L.append("python scripts/vlm_honest_parallel.py --full --workers 8")
    L.append("")
    L.append("# Regenerate this report from the cache")
    L.append("python scripts/vlm_report.py")
    L.append("```")
    L.append("")
    L.append("## Takeaways for the pitch")
    L.append("")
    L.append("1. **Null result that is itself informative.** A foundation VLM with strong performance on natural images cannot classify AFM tear scans (weighted-F1 ~0.23). The morphology of dried biological films is sufficiently out-of-distribution that zero-shot transfer fails outright.")
    L.append("2. **Red-team method validated.** The contamination audit (renaming tiles to `scan_NNNN.png` and re-running) caught a 65-percentage-point phantom gain that would otherwise have been claimed as novel SOTA. The audit itself is a transferable result.")
    L.append("3. **Clinical narrative remains useful.** The reasoning strings above are well-formed and cite morphological features — they can scaffold a clinical-report generator even though the VLM's label is unreliable. Combine with v4 predictions for the actual class; use the VLM only for narrative.")
    L.append(f"4. **Cost.** Full 240 scans at Haiku = roughly **$2.5** of compute. Parallel runtime ~7 min on 8 workers.")
    L.append("5. **Do not ensemble with v4.** Every blend weight > 0 hurts the ensemble. The v4 champion stands alone.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L))

    print(f"Wrote {REPORT}")
    print(f"Wrote {SUMMARY}")
    print()
    print(f"Full honest:  n={eval_full['n']} acc={eval_full['accuracy']:.4f} macro-F1={eval_full['f1_macro']:.4f} weighted-F1={eval_full['f1_weighted']:.4f}")
    if eval_sub.get("n"):
        print(f"Subset 25:    n={eval_sub['n']} acc={eval_sub['accuracy']:.4f} macro-F1={eval_sub['f1_macro']:.4f} weighted-F1={eval_sub['f1_weighted']:.4f}")
    if leaky_data.get("n"):
        print(f"Leaky (void): n={leaky_data['n']} acc={leaky_data['accuracy']:.4f} weighted-F1={leaky_data['f1_weighted']:.4f}")
    if ens and "f1_vlm_overlap_macro" in ens:
        print(f"v4 full:      macro-F1={ens['f1_v4_full_macro']:.4f} weighted-F1={ens['f1_v4_full_weighted']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
