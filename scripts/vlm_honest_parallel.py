"""HONEST VLM classification with OBFUSCATED filenames + parallel workers.

Fixes the critical leakage in `vlm_direct_classify.py`:
  OLD: `cache/vlm_tiles/Diabetes__37_DM.png`
  NEW: `cache/vlm_tiles_honest/scan_0042.png`

The prompt passed to `claude -p` only contains the obfuscated filename,
so the model cannot read the class name. Label is stored separately.

Runs N parallel workers via ProcessPoolExecutor. Target: finish 240 scans
in 10-15 min (vs sequential 2+ hours).

Usage:
    .venv/bin/python scripts/vlm_honest_parallel.py --full --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import classification_report, f1_score  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from teardrop.data import CLASSES, enumerate_samples, preprocess_spm  # noqa: E402
from teardrop.safe_paths import assert_prompt_safe  # noqa: E402

CACHE = REPO / "cache"
TILE_DIR = CACHE / "vlm_tiles_honest"
TILE_DIR.mkdir(parents=True, exist_ok=True)
PRED_FILE = CACHE / "vlm_honest_predictions.json"
MANIFEST = CACHE / "vlm_honest_manifest.json"


PROMPT_TEMPLATE = """You are a medical expert classifying AFM (atomic force microscopy) scans of dried tear droplets. The image at {img_path} shows surface topography rendered with the afmhot colormap (bright = high, dark = low). 1 px = 90 nm. Field of view = 46 um across.

Possible classes with morphological signatures:
- ZdraviLudia (healthy): Dense dendritic ferning, uniform branching, Masmali grade 0-1, fractal D 1.70-1.85.
- Diabetes: Thicker crystals, elevated roughness, glycated proteins produce denser lattice with coarser branches and higher packing density.
- PGOV_Glaukom (glaucoma): Granular structure, loops or rings visible, MMP-9 degrades matrix -> shorter / thicker branches, locally chaotic texture, fractal D lower than healthy.
- SklerozaMultiplex (multiple sclerosis): Heterogeneous morphology within one scan; mixed coarse rods and fine granules; high intra-sample variance; often confusable with PGOV.
- SucheOko (dry eye): Fragmented / sparse network, Masmali grade 3-4, amorphous / empty regions, fractal D < 1.65.

Classify this scan into ONE of the 5 classes above. Respond ONLY with a single JSON object, no markdown fence, exactly this shape:

{{"predicted_class": "<one of the 5 class names>", "confidence": <float 0 to 1>, "reasoning": "<1-2 sentences citing morphological evidence>"}}"""

SYSTEM_APPEND = "You are a vision-language classifier. Respond with one JSON object only and nothing else. No markdown fence, no preamble, no tool calls beyond reading the referenced image."


def render_scan(raw_path: Path, out: Path) -> None:
    if out.exists() and out.stat().st_size > 0:
        return
    h = preprocess_spm(raw_path, target_nm_per_px=90.0, crop_size=512)
    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
    ax.imshow(h, cmap="afmhot")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def build_manifest(samples: list) -> dict:
    """Map obfuscated scan_XXXX.png → true class + metadata. Saved to disk."""
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    random.seed(42)
    indices = list(range(len(samples)))
    random.shuffle(indices)  # so scan_0 isn't necessarily class 0
    manifest = {}
    for dest_i, src_i in enumerate(indices):
        s = samples[src_i]
        key = f"scan_{dest_i:04d}"
        manifest[key] = {
            "true_class": s.cls,
            "person": s.person,
            "raw_path": str(s.raw_path),
        }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def classify_one(task: tuple) -> tuple:
    """Run claude -p on one rendered tile. Called in worker process."""
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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if proc.returncode != 0:
            return key, {"error": f"exit {proc.returncode}: {proc.stderr[:200]}"}
        # claude CLI returns JSON envelope with 'result' string field
        env = json.loads(proc.stdout)
        result_str = env.get("result", "")
        # Strip any markdown fence and extract JSON object
        m = re.search(r"\{[^{}]*\}", result_str, re.DOTALL)
        if not m:
            return key, {"error": "no json found", "raw": result_str[:300]}
        parsed = json.loads(m.group(0))
        return key, parsed
    except subprocess.TimeoutExpired:
        return key, {"error": "timeout"}
    except Exception as e:
        return key, {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default="claude-haiku-4-5")
    args = ap.parse_args()

    samples = enumerate_samples(REPO / "TRAIN_SET")
    manifest = build_manifest(samples)

    # Cache existing predictions
    cache = {}
    if PRED_FILE.exists():
        cache = json.loads(PRED_FILE.read_text())

    # Build task list
    task_items = []
    for key, meta in manifest.items():
        if key in cache and "predicted_class" in cache[key]:
            continue  # already done
        img_path = TILE_DIR / f"{key}.png"
        raw_path = Path(meta["raw_path"])
        render_scan(raw_path, img_path)
        task_items.append((key, img_path, args.model))

    if args.subset > 0:
        task_items = task_items[:args.subset]

    print(f"[honest] manifest={len(manifest)}, cached={sum(1 for v in cache.values() if 'predicted_class' in v)}, todo={len(task_items)}, workers={args.workers}")

    # Run parallel
    start = __import__("time").time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(classify_one, t) for t in task_items]
        for i, fut in enumerate(as_completed(futures), 1):
            key, result = fut.result()
            result["true_class"] = manifest[key]["true_class"]
            result["person"] = manifest[key]["person"]
            cache[key] = result
            if i % 5 == 0 or i == len(futures):
                PRED_FILE.write_text(json.dumps(cache, indent=2))
                done = sum(1 for v in cache.values() if "predicted_class" in v)
                elapsed = __import__("time").time() - start
                print(f"[{i}/{len(futures)}] done={done}/{len(manifest)} elapsed={elapsed:.1f}s")

    PRED_FILE.write_text(json.dumps(cache, indent=2))

    # Summarize
    y_true, y_pred = [], []
    for key, meta in manifest.items():
        v = cache.get(key, {})
        if "predicted_class" in v and v["predicted_class"] in CLASSES:
            y_true.append(meta["true_class"])
            y_pred.append(v["predicted_class"])

    if y_true:
        print(f"\n=== HONEST VLM (obfuscated filenames) ===")
        print(f"N = {len(y_true)}")
        print(f"Accuracy = {sum(a==b for a,b in zip(y_true, y_pred))/len(y_true):.4f}")
        print(f"Weighted F1 = {f1_score(y_true, y_pred, average='weighted', labels=CLASSES):.4f}")
        print(f"Macro F1    = {f1_score(y_true, y_pred, average='macro', labels=CLASSES):.4f}")
        print()
        print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))


if __name__ == "__main__":
    main()
