"""Zero-shot Claude Opus 4.7 classification on AFM tear scans (HONEST, obfuscated).

Goal
----
Test whether the strongest available Anthropic vision model (Opus 4.7) can do
the job WITHOUT any retrieval anchors or in-context examples — only textual
class descriptions + one image. Prior honest runs:

  - Zero-shot Haiku 4.5 (obfuscated):  wF1 = 0.226  (near random)
  - Few-shot Haiku 4.5 (with anchors): wF1 = 0.7351 (big jump)

If Opus zero-shot is competitive with few-shot Haiku, the pipeline simplifies
(no anchor retrieval). If it is still near random, anchors are essential
regardless of model strength.

Safeguards against label leakage
--------------------------------
1. Tiles are rendered to ``cache/vlm_zero_shot_opus_tiles/scan_XXXX.png`` —
   the filename carries no class or person information.
2. The mapping scan_XXXX -> (true class, person, raw path) is kept in a
   separate manifest file the model never reads.
3. The prompt mentions ONLY the tile path; everything else (class list,
   morphology signatures) is identical to the honest-parallel Haiku script
   so the only changing variable is the model.

Usage
-----
    .venv/bin/python scripts/vlm_zero_shot_opus.py --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from teardrop.data import CLASSES, enumerate_samples, preprocess_spm  # noqa: E402
from teardrop.safe_paths import SAFE_ROOT, assert_prompt_safe  # noqa: E402

CACHE = REPO / "cache"
TILE_DIR = SAFE_ROOT / "zero_shot_opus"
TILE_DIR.mkdir(parents=True, exist_ok=True)
PRED_FILE = CACHE / "vlm_zero_shot_opus_predictions.json"
MANIFEST = CACHE / "vlm_zero_shot_opus_manifest.json"

MODEL_ID = "claude-opus-4-7"

PROMPT_TEMPLATE = """You are a medical expert classifying AFM (atomic force microscopy) scans of dried tear droplets. The image at {img_path} shows surface topography rendered with the afmhot colormap (bright = high, dark = low). 1 px = 90 nm. Field of view = 46 um across.

Possible classes with morphological signatures:
- ZdraviLudia (healthy): Dense dendritic ferning, uniform branching, Masmali grade 0-1, fractal D 1.70-1.85.
- Diabetes: Thicker crystals, elevated roughness, glycated proteins produce denser lattice with coarser branches and higher packing density.
- PGOV_Glaukom (glaucoma): Granular structure, loops or rings visible, MMP-9 degrades matrix -> shorter / thicker branches, locally chaotic texture, fractal D lower than healthy.
- SklerozaMultiplex (multiple sclerosis): Heterogeneous morphology within one scan; mixed coarse rods and fine granules; high intra-sample variance; often confusable with PGOV.
- SucheOko (dry eye): Fragmented / sparse network, Masmali grade 3-4, amorphous / empty regions, fractal D < 1.65.

Classify this scan into ONE of the 5 classes above. Respond ONLY with a single JSON object, no markdown fence, exactly this shape:

{{"predicted_class": "<one of the 5 class names>", "confidence": <float 0 to 1>, "reasoning": "<1-2 sentences citing morphological evidence>"}}"""

SYSTEM_APPEND = (
    "You are a vision-language classifier. Respond with one JSON object only "
    "and nothing else. No markdown fence, no preamble, no tool calls beyond "
    "reading the referenced image."
)


def render_scan(raw_path: Path, out: Path) -> None:
    if out.exists() and out.stat().st_size > 0:
        return
    h = preprocess_spm(raw_path, target_nm_per_px=90.0, crop_size=512)
    fig = plt.figure(figsize=(4, 4), dpi=128)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(h, cmap="afmhot", vmin=0.0, vmax=1.0, interpolation="bilinear")
    ax.axis("off")
    fig.savefig(out, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def stratified_person_disjoint_subset(samples, per_class: int, seed: int = 42):
    """Pick ``per_class`` samples per class, each from a DIFFERENT person.

    For SucheOko (only 2 unique persons) we get 2; for others up to ``per_class``.
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
            # deterministic scan within the person
            scans = sorted(by_cls[cls][p], key=lambda s: str(s.raw_path))
            picked.append(scans[0])
    return picked


def build_manifest(selected_samples, seed: int = 42) -> dict:
    """Shuffle + obfuscate the selected subset. Save to disk once; reload thereafter."""
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    rng = np.random.default_rng(seed)
    order = list(range(len(selected_samples)))
    rng.shuffle(order)
    manifest = {}
    for dest_i, src_i in enumerate(order):
        s = selected_samples[src_i]
        key = f"scan_{dest_i:04d}"
        manifest[key] = {
            "true_class": s.cls,
            "person": s.person,
            "raw_path": str(s.raw_path),
        }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def classify_one(task: tuple) -> tuple:
    key, img_path, model = task
    prompt = PROMPT_TEMPLATE.format(img_path=str(img_path))
    assert_prompt_safe(prompt)
    try:
        cmd = [
            "claude", "-p",
            "--model", model,
            "--output-format", "json",
            "--tools", "Read",
            "--append-system-prompt", SYSTEM_APPEND,
            prompt,
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=str(REPO))
        latency = time.time() - t0
        if proc.returncode != 0:
            return key, {"error": f"exit {proc.returncode}: {proc.stderr[:200]}", "latency_s": latency}
        env = json.loads(proc.stdout)
        result_text = env.get("result", "")
        # strip markdown fence if present, then match first JSON object
        cleaned = result_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if not m:
            return key, {
                "error": "no json found",
                "result_text": result_text[:500],
                "latency_s": latency,
                "cost_usd": env.get("total_cost_usd", 0.0),
            }
        parsed = json.loads(m.group(0))
        return key, {
            "predicted_class": str(parsed.get("predicted_class", "")),
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "reasoning": str(parsed.get("reasoning", "")),
            "latency_s": latency,
            "cost_usd": env.get("total_cost_usd", 0.0),
            "duration_api_ms": env.get("duration_api_ms", 0),
            "result_text": result_text,
        }
    except subprocess.TimeoutExpired:
        return key, {"error": "timeout"}
    except Exception as e:
        return key, {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def evaluate(manifest: dict, cache: dict) -> dict:
    y_true, y_pred, confs = [], [], []
    for key, meta in manifest.items():
        v = cache.get(key, {})
        if "predicted_class" in v and v["predicted_class"] in CLASSES:
            y_true.append(meta["true_class"])
            y_pred.append(v["predicted_class"])
            confs.append(float(v.get("confidence", 0.0) or 0.0))
    if not y_true:
        return {"error": "no valid predictions"}
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    f1_w = f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=CLASSES, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()
    per_class = {
        cls: {
            "precision": float(report.get(cls, {}).get("precision", 0.0)),
            "recall": float(report.get(cls, {}).get("recall", 0.0)),
            "f1": float(report.get(cls, {}).get("f1-score", 0.0)),
            "support": int(report.get(cls, {}).get("support", 0)),
        }
        for cls in CLASSES
    }
    return {
        "n": len(y_true),
        "accuracy": float(acc),
        "f1_weighted": float(f1_w),
        "f1_macro": float(f1_m),
        "mean_confidence": float(np.mean(confs)) if confs else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": CLASSES,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=6,
                    help="Samples per class (SucheOko capped at 2 unique persons).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    samples = enumerate_samples(REPO / "TRAIN_SET")
    selected = stratified_person_disjoint_subset(samples, per_class=args.per_class, seed=args.seed)
    print(f"[subset] per_class={args.per_class} -> selected N={len(selected)}")
    from collections import Counter
    print(f"[subset] class breakdown: {Counter(s.cls for s in selected)}")

    manifest = build_manifest(selected, seed=args.seed)
    print(f"[manifest] N={len(manifest)} at {MANIFEST}")

    cache = {}
    if PRED_FILE.exists():
        cache = json.loads(PRED_FILE.read_text())

    # Render tiles + build task list for rows not yet scored
    task_items = []
    for key, meta in manifest.items():
        img_path = TILE_DIR / f"{key}.png"
        raw_path = Path(meta["raw_path"])
        try:
            render_scan(raw_path, img_path)
        except Exception as e:  # noqa: BLE001
            cache[key] = {
                "error": f"render failed: {e}",
                "true_class": meta["true_class"],
                "person": meta["person"],
            }
            continue
        if key in cache and "predicted_class" in cache[key]:
            continue
        task_items.append((key, img_path, args.model))

    done_cnt = sum(1 for v in cache.values() if "predicted_class" in v)
    print(f"[run] model={args.model} cached={done_cnt} todo={len(task_items)} workers={args.workers}")

    if task_items:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(classify_one, t) for t in task_items]
            for i, fut in enumerate(as_completed(futures), 1):
                key, result = fut.result()
                result["true_class"] = manifest[key]["true_class"]
                result["person"] = manifest[key]["person"]
                result["img_path"] = f"cache/vlm_zero_shot_opus_tiles/{key}.png"
                result["model"] = args.model
                cache[key] = result
                PRED_FILE.write_text(json.dumps(cache, indent=2))
                pred = result.get("predicted_class", "ERR")
                ok = "OK" if pred == result["true_class"] else "--"
                elapsed = time.time() - t0
                print(
                    f"[{i}/{len(task_items)}] {ok} {key} true={result['true_class']} "
                    f"pred={pred} conf={result.get('confidence', 0):.2f} "
                    f"${result.get('cost_usd', 0):.4f} elapsed={elapsed:.1f}s"
                )

    PRED_FILE.write_text(json.dumps(cache, indent=2))

    metrics = evaluate(manifest, cache)
    print()
    print("=== ZERO-SHOT OPUS 4.7 (obfuscated filenames) ===")
    if "error" in metrics:
        print(metrics["error"])
        return
    print(f"N           = {metrics['n']}")
    print(f"Accuracy    = {metrics['accuracy']:.4f}")
    print(f"Weighted F1 = {metrics['f1_weighted']:.4f}")
    print(f"Macro F1    = {metrics['f1_macro']:.4f}")
    print(f"Mean conf   = {metrics['mean_confidence']:.3f}")
    print()
    y_true = [manifest[k]["true_class"] for k in manifest
              if k in cache and cache[k].get("predicted_class") in CLASSES]
    y_pred = [cache[k]["predicted_class"] for k in manifest
              if k in cache and cache[k].get("predicted_class") in CLASSES]
    print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))

    # Cost accounting
    total_cost = sum(float(v.get("cost_usd", 0) or 0) for v in cache.values())
    mean_latency = np.mean([float(v.get("latency_s", 0) or 0)
                            for v in cache.values()
                            if "predicted_class" in v])
    print(f"Total cost  = ${total_cost:.4f}")
    print(f"Mean latency= {mean_latency:.1f}s")

    # Also persist a summary blob next to the predictions
    summary = {
        "model": args.model,
        "prompt": "zero-shot (textual class descriptions only, no anchors)",
        "subset": {"per_class": args.per_class, "seed": args.seed, "n": len(manifest)},
        "metrics": metrics,
        "total_cost_usd": total_cost,
        "mean_latency_s": float(mean_latency) if not np.isnan(mean_latency) else 0.0,
        "comparisons": {
            "random_baseline_f1": 0.20,  # approx for 5-class uniform
            "zero_shot_haiku_wf1": 0.226,
            "few_shot_haiku_wf1": 0.7351,
        },
    }
    (CACHE / "vlm_zero_shot_opus_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[summary] cache/vlm_zero_shot_opus_summary.json")


if __name__ == "__main__":
    main()
