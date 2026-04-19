"""VLM prompt-tuning harness for AFM tear-droplet classification.

Tests 4 prompt variants against the baseline `vlm_direct_classify.py` on a
30-scan stratified subset. Uses the `claude -p` CLI subprocess (same path as
the baseline, different cache files so the two don't conflict).

Variants
--------
1. minimal    - Strip all biology, just ask for JSON class label.
2. fewshot    - Include 2 labeled example images per class (10 anchors) in-context.
3. cot        - Ask the model to describe morphology first, THEN classify.
4. expert     - Optometrist persona + expanded biology (fractal D / Ra,Rq / Masmali).

Each variant writes to `cache/vlm_variant_{N}_predictions.json`.

Usage
-----
    .venv/bin/python scripts/vlm_prompt_variants.py --variants 1 2 3 4
    .venv/bin/python scripts/vlm_prompt_variants.py --eval-only
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from teardrop.data import CLASSES, enumerate_samples, preprocess_spm  # noqa: E402

CACHE_DIR = REPO / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TILE_DIR = CACHE_DIR / "vlm_tiles"
TILE_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "claude-haiku-4-5"

SYSTEM_APPEND = (
    "You are a vision-language classifier. Respond with one JSON object only "
    "and nothing else. No markdown fence, no preamble, no tool calls beyond "
    "reading the referenced image(s)."
)


# ---------------------------------------------------------------------------
# Tile rendering (same as baseline)
# ---------------------------------------------------------------------------


def render_scan_tile(raw_path: Path, out_path: Path, *, crop: int = 512) -> Path:
    if out_path.exists():
        return out_path
    h = preprocess_spm(raw_path, target_nm_per_px=90.0, crop_size=crop)
    fig = plt.figure(figsize=(4, 4), dpi=128)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(h, cmap="afmhot", vmin=0.0, vmax=1.0, interpolation="bilinear")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


def stratified_subset(samples, per_class: int = 6, seed: int = 42):
    """Pick per_class samples per class, maximizing person diversity.

    If the class has fewer unique persons than per_class, we allow multiple
    scans from the same person (picking distinct scans in file order).
    """
    rng = np.random.default_rng(seed)
    by_cls: dict[str, dict[str, list]] = {}
    for s in samples:
        by_cls.setdefault(s.cls, {}).setdefault(s.person, []).append(s)
    picked = []
    for cls in CLASSES:
        persons = list(by_cls.get(cls, {}).keys())
        rng.shuffle(persons)
        chosen: list = []
        person_idx: dict[str, int] = {p: 0 for p in persons}
        while len(chosen) < per_class:
            progressed = False
            for p in persons:
                if len(chosen) >= per_class:
                    break
                scans = sorted(by_cls[cls][p], key=lambda s: str(s.raw_path))
                i = person_idx[p]
                if i < len(scans):
                    chosen.append(scans[i])
                    person_idx[p] += 1
                    progressed = True
            if not progressed:
                break
        picked.extend(chosen)
    return picked


def few_shot_anchors(samples, query_keys: set[str], per_class: int = 2, seed: int = 7):
    """Pick per_class anchor examples per class that are NOT in query_keys,
    prefer person-disjoint from query, then prefer distinct persons among anchors."""
    rng = np.random.default_rng(seed)
    query_persons: set[str] = set()
    for s in samples:
        if str(s.raw_path.relative_to(REPO)) in query_keys:
            query_persons.add(s.person)

    anchors: dict[str, list] = {c: [] for c in CLASSES}
    by_cls: dict[str, list] = {c: [] for c in CLASSES}
    for s in samples:
        by_cls[s.cls].append(s)

    for cls in CLASSES:
        pool = [s for s in by_cls[cls]
                if str(s.raw_path.relative_to(REPO)) not in query_keys]
        disjoint = [s for s in pool if s.person not in query_persons]
        fallback = [s for s in pool if s.person in query_persons]
        rng.shuffle(disjoint)
        rng.shuffle(fallback)
        chosen: list = []
        seen_persons: set[str] = set()
        for s in disjoint + fallback:
            if len(chosen) >= per_class:
                break
            if s.person in seen_persons and len(disjoint) + len(fallback) > per_class:
                continue
            chosen.append(s)
            seen_persons.add(s.person)
        if len(chosen) < per_class:
            for s in disjoint + fallback:
                if s in chosen:
                    continue
                chosen.append(s)
                if len(chosen) >= per_class:
                    break
        anchors[cls] = chosen[:per_class]
    return anchors


# ---------------------------------------------------------------------------
# Prompt builders (one per variant)
# ---------------------------------------------------------------------------


def prompt_v1_minimal(img_path: Path) -> str:
    return (
        f"Read the image at {img_path}. Classify it into one of these 5 classes: "
        + ", ".join(CLASSES) + ".\n\n"
        + "Respond ONLY with a JSON object of exactly this shape (no markdown, no preamble):\n"
        + '{"predicted_class": "<class>", "confidence": <float 0 to 1>, "reasoning": "<1 sentence>"}'
    )


def prompt_v2_fewshot(img_path: Path, anchors: dict[str, list[Path]]) -> str:
    lines = [
        "You are classifying an AFM (atomic force microscopy) scan of a dried tear droplet "
        "into one of 5 classes. Below are 10 labeled example scans (2 per class) "
        "showing typical morphology. After studying the anchors, classify the NEW scan.\n",
        "REFERENCE ANCHORS (read each image):",
    ]
    for cls, paths in anchors.items():
        for i, p in enumerate(paths):
            lines.append(f"  [{cls}] example {i+1}: {p}")
    lines.append(f"\nNEW SCAN to classify (read this image): {img_path}\n")
    lines.append(
        "Compare the new scan's morphology to the anchors. Respond ONLY with a JSON object "
        "of exactly this shape (no markdown, no preamble):\n"
        '{"predicted_class": "<one of the 5 classes>", "confidence": <0-1>, '
        '"reasoning": "<1-2 sentences referencing the most similar anchors>"}'
    )
    return "\n".join(lines)


def prompt_v3_cot(img_path: Path) -> str:
    return (
        "You are a medical expert classifying AFM (atomic force microscopy) scans of dried tear droplets.\n"
        f"Read the image at {img_path}. Surface topography rendered with the afmhot colormap "
        "(bright = high, dark = low). 1 px = 90 nm; field of view ~46 um.\n\n"
        "Possible classes: " + ", ".join(CLASSES) + ".\n\n"
        "STEP 1: Write 2-4 sentences describing what you see — concrete morphology only: "
        "branching pattern (dense dendritic / granular / fragmented / mixed), "
        "crystal thickness, fractal self-similarity, apparent roughness, "
        "presence of loops/rings, intra-scan heterogeneity.\n\n"
        "STEP 2: Based on your description, pick the class.\n\n"
        "Respond ONLY with a JSON object of exactly this shape (no markdown, no preamble):\n"
        '{"description": "<2-4 sentences of observed morphology>", '
        '"predicted_class": "<one of the 5 classes>", "confidence": <0-1>, '
        '"reasoning": "<link the observed morphology to the chosen class>"}'
    )


def prompt_v4_expert(img_path: Path) -> str:
    return (
        "You are a board-certified optometrist specializing in ocular surface disease with 15+ years "
        "of experience interpreting AFM (atomic force microscopy) scans of dried tear crystal "
        "ferning patterns. You routinely grade scans on the Masmali scale (0 = dense uniform ferning, "
        "4 = no ferning) and compute morphometric features (fractal dimension D, RMS roughness Rq, "
        "arithmetic roughness Ra, branching density).\n\n"
        f"Read the image at {img_path}. Surface topography rendered with afmhot "
        "(bright = high z, dark = low z). 1 pixel = 90 nm; field of view ~46 um.\n\n"
        "Class reference (expanded morphometric signatures):\n\n"
        "- ZdraviLudia (healthy control): Dense dendritic ferning with self-similar secondary and "
        "tertiary branching. Masmali grade 0-1. Fractal dimension D = 1.70-1.85 (high, near-space-filling). "
        "Rq typically 20-60 nm. Uniform branch thickness 200-400 nm. Minimal amorphous regions.\n\n"
        "- Diabetes (hyperglycemic tear film): Advanced glycation end-products (AGEs) cross-link tear "
        "proteins, producing THICKER and DENSER crystal lattices. Branches coarser (400-800 nm). "
        "Packing density elevated. Ra/Rq ~1.5-2x healthy. Fractal D somewhat reduced (1.65-1.78) "
        "from loss of fine-scale self-similarity. Masmali grade 1-2.\n\n"
        "- PGOV_Glaukom (glaucoma, often on prostaglandin-analogue therapy): MMP-9 activity degrades "
        "tear mucin -> shorter, stubbier branches; locally granular texture; loops/rings sometimes "
        "visible. Fractal D reduced (1.55-1.72). Ra/Rq variable but rougher at short scales. "
        "Masmali grade 2-3. BAK preservative can add chaotic granularity.\n\n"
        "- SklerozaMultiplex (multiple sclerosis, autoimmune dry-eye pattern): HIGH intra-scan "
        "heterogeneity — one region may show coarse rods while another shows fine granules. "
        "Mixed morphology is the tell. Fractal D 1.55-1.75. Confusable with PGOV but distinguished "
        "by heterogeneity rather than uniform granularity. Masmali grade 2-3.\n\n"
        "- SucheOko (severe dry-eye, keratoconjunctivitis sicca): Fragmented, sparse network. "
        "Large amorphous / empty regions. Masmali grade 3-4. Fractal D < 1.65. Ra can be low "
        "(smooth amorphous zones) or high (isolated crystals). Branches short, disconnected.\n\n"
        "Classify the scan. Identify which morphometric features you rely on, then give the class.\n\n"
        "Respond ONLY with a JSON object of exactly this shape (no markdown, no preamble):\n"
        '{"predicted_class": "<one of the 5 classes>", "confidence": <0-1>, '
        '"reasoning": "<2-3 sentences citing Masmali grade, estimated fractal D, branch morphology, '
        'and why competing classes are ruled out>"}'
    )


# ---------------------------------------------------------------------------
# CLI invocation
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def call_claude_cli(prompt: str, model: str = MODEL, timeout_s: int = 180) -> dict:
    """Invoke `claude -p` as a subprocess; return parsed response + meta."""
    cmd = [
        "claude",
        "-p",
        "--model", model,
        "--output-format", "json",
        "--tools", "Read",
        "--append-system-prompt", SYSTEM_APPEND,
        prompt,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(REPO),
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout_s}s", "latency_s": float(timeout_s)}
    latency = time.time() - t0
    if proc.returncode != 0:
        return {
            "error": f"cli exit {proc.returncode}",
            "stderr": proc.stderr[:500],
            "latency_s": latency,
        }
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {
            "error": f"envelope parse failed: {e}",
            "stdout": proc.stdout[:500],
            "latency_s": latency,
        }
    result_text = envelope.get("result", "")
    try:
        parsed = _extract_json(result_text)
    except (ValueError, json.JSONDecodeError) as e:
        return {
            "error": f"inner JSON parse failed: {e}",
            "result_text": result_text,
            "latency_s": latency,
            "cost_usd": envelope.get("total_cost_usd", 0.0),
        }
    out = {
        "predicted_class": str(parsed.get("predicted_class", "")),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "reasoning": str(parsed.get("reasoning", "")),
        "latency_s": latency,
        "cost_usd": envelope.get("total_cost_usd", 0.0),
        "duration_api_ms": envelope.get("duration_api_ms", 0),
        "result_text": result_text,
    }
    if "description" in parsed:
        out["description"] = str(parsed["description"])
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_variant(
    variant: int,
    samples,
    anchors: dict[str, list] | None,
    cache_path: Path,
    time_budget_s: float,
) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    # Pre-render anchor tiles (for variant 2)
    anchor_paths_by_cls: dict[str, list[Path]] = {}
    if variant == 2 and anchors is not None:
        for cls, anchor_samples in anchors.items():
            anchor_paths_by_cls[cls] = []
            for s in anchor_samples:
                p = TILE_DIR / f"{s.cls}__{s.raw_path.stem}.png"
                render_scan_tile(s.raw_path, p)
                anchor_paths_by_cls[cls].append(p)

    t_start = time.time()
    for i, s in enumerate(samples):
        key = str(s.raw_path.relative_to(REPO))
        if key in cache and "predicted_class" in cache[key]:
            print(f"  [{i+1}/{len(samples)}] cached: {key} -> {cache[key]['predicted_class']}")
            continue
        elapsed = time.time() - t_start
        if elapsed > time_budget_s:
            print(f"  [budget] stopping variant {variant} at {i}/{len(samples)} after {elapsed:.0f}s")
            break
        img_path = TILE_DIR / f"{s.cls}__{s.raw_path.stem}.png"
        try:
            render_scan_tile(s.raw_path, img_path)
        except Exception as e:
            cache[key] = {"error": f"render failed: {e}", "true_class": s.cls, "person": s.person}
            continue

        if variant == 1:
            prompt = prompt_v1_minimal(img_path)
        elif variant == 2:
            prompt = prompt_v2_fewshot(img_path, anchor_paths_by_cls)
        elif variant == 3:
            prompt = prompt_v3_cot(img_path)
        elif variant == 4:
            prompt = prompt_v4_expert(img_path)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        res = call_claude_cli(prompt)
        res["true_class"] = s.cls
        res["person"] = s.person
        res["img_path"] = str(img_path.relative_to(REPO))
        cache[key] = res
        cache_path.write_text(json.dumps(cache, indent=2))
        pred = res.get("predicted_class", "ERR")
        ok = "OK" if pred == s.cls else "--"
        cost = res.get("cost_usd", 0.0)
        lat = res.get("latency_s", 0.0)
        print(f"  [{i+1}/{len(samples)}] {ok} {key}  true={s.cls}  pred={pred}  "
              f"conf={res.get('confidence', 0):.2f}  t={lat:.1f}s  ${cost:.4f}")
    return cache


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(cache: dict[str, dict], keys: list[str]) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    total_cost = 0.0
    total_latency = 0.0
    for k in keys:
        entry = cache.get(k)
        if not entry or "predicted_class" not in entry or entry["predicted_class"] not in CLASSES:
            continue
        y_true.append(entry["true_class"])
        y_pred.append(entry["predicted_class"])
        confidences.append(entry.get("confidence", 0.0))
        total_cost += entry.get("cost_usd", 0.0)
        total_latency += entry.get("latency_s", 0.0)
    if not y_true:
        return {"error": "no valid predictions"}
    labels = CLASSES
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "n": len(y_true),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": labels,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "total_cost_usd": total_cost,
        "total_latency_s": total_latency,
    }


def print_eval(name: str, ev: dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    if "error" in ev:
        print("ERROR:", ev["error"])
        return
    print(f"n={ev['n']}  acc={ev['accuracy']:.4f}  f1_macro={ev['f1_macro']:.4f}  "
          f"mean_conf={ev['mean_confidence']:.3f}  cost=${ev['total_cost_usd']:.4f}  "
          f"total_t={ev['total_latency_s']:.0f}s")
    print("Per-class F1:")
    for cls in ev["labels"]:
        row = ev["classification_report"].get(cls, {})
        print(f"  {cls:22s}  P={row.get('precision', 0):.3f}  "
              f"R={row.get('recall', 0):.3f}  F1={row.get('f1-score', 0):.3f}  "
              f"support={int(row.get('support', 0))}")
    print("Confusion matrix (rows=true, cols=pred):")
    print("  " + "  ".join(f"{c[:6]:>6s}" for c in ev["labels"]))
    for lab, row in zip(ev["labels"], ev["confusion_matrix"]):
        print(f"  {lab[:6]:>6s} " + "  ".join(f"{v:>6d}" for v in row))


def evaluate_baseline_on_subset(query_keys: list[str]) -> dict[str, Any] | None:
    baseline_cache_path = CACHE_DIR / "vlm_predictions.json"
    if not baseline_cache_path.exists():
        return None
    cache = json.loads(baseline_cache_path.read_text())
    # evaluate on whatever overlap exists
    overlap = [k for k in query_keys if k in cache]
    if not overlap:
        return {"error": "no overlap with baseline cache"}
    ev = evaluate(cache, overlap)
    ev["n_overlap"] = len(overlap)
    ev["n_query"] = len(query_keys)
    return ev


def qualitative_samples(cache: dict[str, dict], keys: list[str]) -> dict[str, list[dict]]:
    correct: list[dict] = []
    wrong: list[dict] = []
    for k in keys:
        e = cache.get(k)
        if not e or "predicted_class" not in e:
            continue
        if e["predicted_class"] not in CLASSES:
            continue
        rec = {
            "scan": k,
            "true_class": e["true_class"],
            "predicted_class": e["predicted_class"],
            "confidence": e.get("confidence", 0.0),
            "reasoning": e.get("reasoning", ""),
            "description": e.get("description", ""),
        }
        if rec["true_class"] == rec["predicted_class"]:
            correct.append(rec)
        else:
            wrong.append(rec)
    correct.sort(key=lambda r: -r["confidence"])
    wrong.sort(key=lambda r: -r["confidence"])
    return {"correct": correct[:3], "wrong": wrong[:3]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--time-budget-s", type=int, default=1500,
                    help="per-variant wall clock cap (default 25 min)")
    args = ap.parse_args()

    all_samples = enumerate_samples(REPO / "TRAIN_SET")
    print(f"Loaded {len(all_samples)} total samples")

    query_samples = stratified_subset(all_samples, per_class=args.per_class, seed=args.seed)
    query_keys = [str(s.raw_path.relative_to(REPO)) for s in query_samples]
    print(f"\nQuery subset: {len(query_samples)} scans ({args.per_class}/class)")
    by_cls: dict[str, int] = {}
    for s in query_samples:
        by_cls.setdefault(s.cls, 0)
        by_cls[s.cls] += 1
    print(f"  distribution: {by_cls}")
    query_keys_set = set(query_keys)
    anchors = few_shot_anchors(all_samples, query_keys_set, per_class=2, seed=7)
    print("\nFew-shot anchors (for variant 2):")
    for cls, sa in anchors.items():
        for s in sa:
            print(f"  [{cls}]  {s.raw_path.name}  person={s.person}")

    all_evals: dict[int, dict] = {}
    qual_by_variant: dict[int, dict] = {}
    for v in args.variants:
        cache_path = CACHE_DIR / f"vlm_variant_{v}_predictions.json"
        print(f"\n{'=' * 60}")
        print(f"Variant {v} -> {cache_path.name}")
        print(f"{'=' * 60}")
        if not args.eval_only:
            cache = run_variant(v, query_samples, anchors=anchors,
                                cache_path=cache_path, time_budget_s=args.time_budget_s)
        else:
            cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        ev = evaluate(cache, query_keys)
        all_evals[v] = ev
        print_eval(f"Variant {v}", ev)
        qual_by_variant[v] = qualitative_samples(cache, query_keys)

    print(f"\n{'=' * 60}")
    print("Baseline (biology-rich prompt, from vlm_predictions.json)")
    print(f"{'=' * 60}")
    baseline_ev = evaluate_baseline_on_subset(query_keys)
    if baseline_ev and "error" not in baseline_ev:
        print_eval("Baseline (overlap only)", baseline_ev)
    elif baseline_ev:
        print(f"Baseline eval: {baseline_ev.get('error')}")
    else:
        print("No baseline cache found.")

    print(f"\n{'=' * 60}")
    print("Summary comparison")
    print(f"{'=' * 60}")
    print(f"{'Variant':<20} {'n':>4} {'Acc':>8} {'F1_macro':>10} {'MeanConf':>10} {'Cost':>10}")
    if baseline_ev and "error" not in baseline_ev:
        print(f"{'baseline':<20} {baseline_ev['n']:>4d} "
              f"{baseline_ev['accuracy']:>8.4f} {baseline_ev['f1_macro']:>10.4f} "
              f"{baseline_ev['mean_confidence']:>10.3f} {baseline_ev.get('total_cost_usd', 0):>10.4f}")
    for v, ev in all_evals.items():
        if "error" in ev:
            print(f"{'variant_' + str(v):<20} ERROR: {ev['error']}")
            continue
        label = {1: "v1_minimal", 2: "v2_fewshot", 3: "v3_cot", 4: "v4_expert"}.get(v, f"v{v}")
        print(f"{label:<20} {ev['n']:>4d} "
              f"{ev['accuracy']:>8.4f} {ev['f1_macro']:>10.4f} "
              f"{ev['mean_confidence']:>10.3f} {ev['total_cost_usd']:>10.4f}")

    summary_path = CACHE_DIR / "vlm_variants_summary.json"
    summary_path.write_text(json.dumps({
        "args": vars(args),
        "query_keys": query_keys,
        "anchors": {cls: [str(s.raw_path.relative_to(REPO)) for s in sa] for cls, sa in anchors.items()},
        "variants": {str(v): ev for v, ev in all_evals.items()},
        "baseline": baseline_ev,
        "qualitative": {str(v): q for v, q in qual_by_variant.items()},
    }, indent=2))
    print(f"\nSummary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
