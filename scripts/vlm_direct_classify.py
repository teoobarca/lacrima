"""VLM direct-image baseline for AFM tear-droplet classification.

Approach: render each AFM height map as an afmhot PNG, hand the image
file to a Claude vision model via the `claude -p` CLI subprocess, parse
its JSON response. No training, no features — just the image + prompt.

This is complementary to `teardrop.llm_reason` (which sends numeric
features). The VLM path offers a natural-language clinical narrative
per scan that a classical model cannot produce, regardless of F1.

Usage
-----
Subset (stratified, person-disjoint, 5 per class):
    python scripts/vlm_direct_classify.py --subset 25

Full dataset:
    python scripts/vlm_direct_classify.py --full

The raw VLM outputs are cached under ``cache/vlm_predictions.json`` so
re-runs can skip already-scored scans.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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

# repo root on path so we can import teardrop
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from teardrop.data import CLASSES, enumerate_samples, preprocess_spm  # noqa: E402
from teardrop.safe_paths import SAFE_ROOT, assert_prompt_safe, safe_tile_path  # noqa: E402

CACHE_DIR = REPO / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Obfuscated VLM tile subdir. Labels are stored in the prediction cache
# (``vlm_predictions.json``) and the renamed scan_XXXX.png filename carries
# no class / person / raw-name info.
TILE_SUBDIR = "direct"
PRED_CACHE = CACHE_DIR / "vlm_predictions.json"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_APPEND = (
    "You are a vision-language classifier. Respond with one JSON object only "
    "and nothing else. No markdown fence, no preamble, no tool calls beyond "
    "reading the referenced image."
)

PROMPT_TEMPLATE = """You are a medical expert classifying AFM (atomic force microscopy) scans of dried tear droplets.
The image at {img_path} shows surface topography rendered with the afmhot colormap (bright = high, dark = low). 1 px = 90 nm. Field of view = 46 um across.

Possible classes with morphological signatures:
- ZdraviLudia (healthy): Dense dendritic ferning, uniform branching, Masmali grade 0-1, fractal D 1.70-1.85.
- Diabetes: Thicker crystals, elevated roughness, glycated proteins produce denser lattice with coarser branches and higher packing density.
- PGOV_Glaukom (glaucoma): Granular structure, loops or rings visible, MMP-9 degrades matrix -> shorter / thicker branches, locally chaotic texture, fractal D lower than healthy.
- SklerozaMultiplex (multiple sclerosis): Heterogeneous morphology within one scan; mixed coarse rods and fine granules; high intra-sample variance; often confusable with PGOV.
- SucheOko (dry eye): Fragmented / sparse network, Masmali grade 3-4, amorphous / empty regions, fractal D < 1.65.

Classify this scan into ONE of the 5 classes above. Respond ONLY with a single JSON object, no markdown fence, exactly this shape:

{{"predicted_class": "<one of the 5 class names>", "confidence": <float 0 to 1>, "reasoning": "<1-2 sentences citing morphological evidence>"}}"""


# ---------------------------------------------------------------------------
# Tile rendering
# ---------------------------------------------------------------------------


def render_scan_tile(raw_path: Path, out_path: Path, *, crop: int = 512) -> Path:
    """Render the central 512x512 tile of a scan into an afmhot PNG."""
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
# Claude CLI wrapper
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


def call_claude_cli(img_path: Path, model: str = "claude-haiku-4-5", timeout_s: int = 90) -> dict:
    """Invoke `claude -p` as a subprocess; return parsed response + meta."""
    prompt = PROMPT_TEMPLATE.format(img_path=str(img_path))
    assert_prompt_safe(prompt)
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--output-format",
        "json",
        "--tools",
        "Read",
        "--append-system-prompt",
        SYSTEM_APPEND,
        prompt,
    ]
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(REPO),
    )
    latency = time.time() - t0
    if proc.returncode != 0:
        return {"error": f"cli exit {proc.returncode}", "stderr": proc.stderr[:500], "latency_s": latency}
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"envelope parse failed: {e}", "stdout": proc.stdout[:500], "latency_s": latency}
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
    return {
        "predicted_class": str(parsed.get("predicted_class", "")),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "reasoning": str(parsed.get("reasoning", "")),
        "latency_s": latency,
        "cost_usd": envelope.get("total_cost_usd", 0.0),
        "duration_api_ms": envelope.get("duration_api_ms", 0),
        "result_text": result_text,
    }


# ---------------------------------------------------------------------------
# Subset selection (stratified, person-disjoint)
# ---------------------------------------------------------------------------


def stratified_person_disjoint(samples, per_class: int = 5, seed: int = 42):
    """Pick per_class samples per class, each from a DIFFERENT person.

    Within one class we draw without replacement over unique persons.
    Across classes there is no overlap (classes are disjoint by design
    of this dataset).
    """
    rng = np.random.default_rng(seed)
    picked = []
    by_cls: dict[str, dict[str, list]] = {}
    for s in samples:
        by_cls.setdefault(s.cls, {}).setdefault(s.person, []).append(s)
    for cls in CLASSES:
        persons = list(by_cls.get(cls, {}).keys())
        rng.shuffle(persons)
        take = min(per_class, len(persons))
        for p in persons[:take]:
            # pick the first scan for that person (deterministic)
            picked.append(sorted(by_cls[cls][p], key=lambda s: str(s.raw_path))[0])
    return picked


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def load_cache() -> dict[str, dict]:
    if PRED_CACHE.exists():
        return json.loads(PRED_CACHE.read_text())
    return {}


def save_cache(cache: dict[str, dict]) -> None:
    PRED_CACHE.write_text(json.dumps(cache, indent=2))


def run(samples, model: str, time_budget_s: float, cache: dict[str, dict]) -> dict[str, dict]:
    """Score a list of samples, updating cache in place."""
    t_start = time.time()
    for i, s in enumerate(samples):
        key = str(s.raw_path.relative_to(REPO))
        if key in cache and "predicted_class" in cache[key]:
            print(f"[{i+1}/{len(samples)}] cached: {key} -> {cache[key]['predicted_class']}")
            continue
        elapsed = time.time() - t_start
        if elapsed > time_budget_s:
            print(f"[budget] stopping at {i}/{len(samples)} after {elapsed:.0f}s")
            break
        # Class-neutral, unique tile path. The filename is sha1(rel_path)
        # truncated to 16 hex chars, stored under cache/vlm_safe/direct/
        # (see teardrop.safe_paths). No class/person info anywhere in the
        # path passed to claude -p.
        rel_key = str(s.raw_path.relative_to(REPO))
        tile_id = hashlib.sha1(rel_key.encode("utf-8")).hexdigest()[:16]
        tile_dir = SAFE_ROOT / TILE_SUBDIR
        tile_dir.mkdir(parents=True, exist_ok=True)
        img_path = tile_dir / f"scan_{tile_id}.png"
        try:
            render_scan_tile(s.raw_path, img_path)
        except Exception as e:
            cache[key] = {"error": f"render failed: {e}", "true_class": s.cls, "person": s.person}
            print(f"[{i+1}/{len(samples)}] RENDER FAIL: {key} {e}")
            continue
        res = call_claude_cli(img_path, model=model)
        res["true_class"] = s.cls
        res["person"] = s.person
        res["img_path"] = str(img_path.relative_to(REPO))
        cache[key] = res
        save_cache(cache)
        pred = res.get("predicted_class", "ERR")
        ok = "OK" if pred == s.cls else "--"
        cost = res.get("cost_usd", 0.0)
        lat = res.get("latency_s", 0.0)
        print(f"[{i+1}/{len(samples)}] {ok} {key}  true={s.cls}  pred={pred}  conf={res.get('confidence', 0):.2f}  t={lat:.1f}s  ${cost:.4f}")
    return cache


def evaluate(cache: dict[str, dict], keys: list[str]) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    for k in keys:
        entry = cache.get(k)
        if not entry or "predicted_class" not in entry or entry["predicted_class"] not in CLASSES:
            continue
        y_true.append(entry["true_class"])
        y_pred.append(entry["predicted_class"])
        confidences.append(entry.get("confidence", 0.0))
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
    }


def print_eval(name: str, ev: dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    if "error" in ev:
        print("ERROR:", ev["error"])
        return
    print(f"n = {ev['n']}  acc = {ev['accuracy']:.4f}  f1_macro = {ev['f1_macro']:.4f}  mean_conf = {ev['mean_confidence']:.3f}")
    print("\nPer-class F1:")
    for cls in ev["labels"]:
        row = ev["classification_report"].get(cls, {})
        print(f"  {cls:22s}  P={row.get('precision', 0):.3f}  R={row.get('recall', 0):.3f}  F1={row.get('f1-score', 0):.3f}  support={int(row.get('support', 0))}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("  " + "  ".join(f"{c[:6]:>6s}" for c in ev["labels"]))
    for lab, row in zip(ev["labels"], ev["confusion_matrix"]):
        print(f"  {lab[:6]:>6s} " + "  ".join(f"{v:>6d}" for v in row))


def ensemble_with_v4(cache: dict[str, dict], keys: list[str]) -> dict[str, Any] | None:
    """If VLM confidence + v4 OOF exist, compare pure-v4 vs averaged."""
    v4_path = CACHE_DIR / "v4_oof.npz"
    if not v4_path.exists():
        return None
    z = np.load(v4_path, allow_pickle=True)
    v4_proba = z["proba"]
    v4_paths = z["scan_paths"]
    v4_y = z["y"]
    # map scan_paths in v4 to keys in cache (both are repo-relative)
    path_to_idx = {str(p): i for i, p in enumerate(v4_paths)}

    vlm_proba = np.full((len(v4_paths), 5), np.nan)
    for k, entry in cache.items():
        if k not in path_to_idx:
            continue
        if "predicted_class" not in entry or entry["predicted_class"] not in CLASSES:
            continue
        conf = max(min(float(entry.get("confidence", 0.5)), 0.99), 0.01)
        # spread remaining (1-conf) across 4 other classes uniformly
        residual = (1.0 - conf) / 4.0
        row = np.full(5, residual)
        row[CLASSES.index(entry["predicted_class"])] = conf
        vlm_proba[path_to_idx[k]] = row

    have = ~np.isnan(vlm_proba).any(axis=1)
    if have.sum() < 5:
        return None
    v4_only_pred = v4_proba[have].argmax(axis=1)
    y_sub = v4_y[have]
    f1_v4 = f1_score(y_sub, v4_only_pred, average="macro", zero_division=0)

    results = {"n_overlap": int(have.sum()), "f1_v4_alone": float(f1_v4), "blends": {}}
    for w in [0.1, 0.2, 0.3, 0.5, 0.7]:
        blend = (1 - w) * v4_proba[have] + w * vlm_proba[have]
        blend_pred = blend.argmax(axis=1)
        f1_blend = f1_score(y_sub, blend_pred, average="macro", zero_division=0)
        results["blends"][f"v4*{1-w:.1f} + vlm*{w:.1f}"] = float(f1_blend)
    return results


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
        }
        if rec["true_class"] == rec["predicted_class"]:
            correct.append(rec)
        else:
            wrong.append(rec)
    # sort by confidence desc
    correct.sort(key=lambda r: -r["confidence"])
    wrong.sort(key=lambda r: -r["confidence"])
    return {"correct": correct[:5], "wrong": wrong[:5]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=0, help="per-class stratified picks (0 = use --full)")
    ap.add_argument("--full", action="store_true", help="score every sample")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--time-budget-s", type=int, default=2100, help="hard wall-clock cap (default 35 min)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true", help="skip LLM calls; evaluate cache")
    args = ap.parse_args()

    all_samples = enumerate_samples(REPO / "TRAIN_SET")
    print(f"Loaded {len(all_samples)} total samples")

    if args.subset > 0:
        samples = stratified_person_disjoint(all_samples, per_class=args.subset, seed=args.seed)
        print(f"Subset: {len(samples)} samples (stratified, person-disjoint)")
    elif args.full:
        samples = all_samples
        print(f"Full: {len(samples)} samples")
    else:
        ap.error("pass --subset N or --full")

    cache = load_cache()
    if not args.eval_only:
        cache = run(samples, args.model, args.time_budget_s, cache)
        save_cache(cache)

    scored_keys = [str(s.raw_path.relative_to(REPO)) for s in samples]
    ev = evaluate(cache, scored_keys)
    print_eval("VLM direct-classify", ev)

    qual = qualitative_samples(cache, scored_keys)
    print("\n=== Qualitative: top-confidence correct ===")
    for r in qual["correct"]:
        print(f"- {r['scan']}  (conf {r['confidence']:.2f}): {r['reasoning']}")
    print("\n=== Qualitative: top-confidence WRONG ===")
    for r in qual["wrong"]:
        print(f"- {r['scan']}  true={r['true_class']}  pred={r['predicted_class']}  (conf {r['confidence']:.2f}): {r['reasoning']}")

    ens = ensemble_with_v4(cache, scored_keys)
    if ens:
        print("\n=== Ensemble with v4 OOF ===")
        print(f"overlap n={ens['n_overlap']}")
        print(f"v4 alone (same overlap) F1_macro = {ens['f1_v4_alone']:.4f}")
        for k, v in ens["blends"].items():
            print(f"  {k}  ->  F1_macro = {v:.4f}")

    # dump a JSON summary for the report
    summary_path = CACHE_DIR / "vlm_summary.json"
    summary_path.write_text(json.dumps({
        "args": vars(args),
        "eval": ev,
        "ensemble": ens,
        "qualitative": qual,
    }, indent=2))
    print(f"\nSummary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
