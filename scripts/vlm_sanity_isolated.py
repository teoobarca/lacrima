"""Sanity test: run VLM on 15 scans with COMPLETELY ISOLATED paths outside project.

Writes tiles to /tmp/vlm_sanity_test_<timestamp>/scan_XXXX.png — the path has
no project directory name, no class name, no patient identifier. Only the
image content can inform classification.

If VLM still gets >=80% accuracy on this small subset, it's using VISUAL cues
(real signal). If it drops toward random (~20%), it was cheating via filename.
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from teardrop.data import CLASSES, enumerate_samples, preprocess_spm  # noqa: E402

N_SANITY = 15  # 3 per class, stratified


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


def run_one(img_path: Path) -> dict:
    prompt = PROMPT_TEMPLATE.format(img_path=str(img_path))
    try:
        proc = subprocess.run(
            ["claude", "-p", "--model", "claude-haiku-4-5",
             "--output-format", "json", "--tools", "Read",
             "--append-system-prompt", SYSTEM_APPEND, prompt],
            capture_output=True, text=True, timeout=90,
        )
        if proc.returncode != 0:
            return {"error": f"exit {proc.returncode}"}
        env = json.loads(proc.stdout)
        result_str = env.get("result", "")
        m = re.search(r"\{[^{}]*\}", result_str, re.DOTALL)
        if not m:
            return {"error": "no json", "raw": result_str[:200]}
        return json.loads(m.group(0))
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": f"{type(e).__name__}"}


def main():
    # Isolated output dir (mktemp ensures unique name with NO class/project info)
    out_dir = Path(tempfile.mkdtemp(prefix="vlm_sanity_"))
    print(f"Output dir (isolated, no class names anywhere): {out_dir}")

    # Select stratified 3 per class (seed for reproducibility)
    samples = enumerate_samples(REPO / "TRAIN_SET")
    random.seed(123)
    buckets = {c: [] for c in CLASSES}
    for s in samples:
        buckets[s.cls].append(s)
    selected = []
    for c in CLASSES:
        k = min(3, len(buckets[c]))
        selected.extend(random.sample(buckets[c], k))
    random.shuffle(selected)

    # Render with obfuscated names. Manifest stays in-memory only.
    manifest = {}
    for i, s in enumerate(selected):
        obf_name = f"scan_{i:04d}.png"
        img_path = out_dir / obf_name
        h = preprocess_spm(s.raw_path, target_nm_per_px=90.0, crop_size=512)
        fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
        ax.imshow(h, cmap="afmhot")
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(img_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        manifest[obf_name] = {
            "true_class": s.cls,
            "person": s.person,
            "raw_name": s.raw_path.name,  # kept in manifest, NOT sent to VLM
        }

    print(f"Rendered {len(manifest)} tiles to isolated dir.")
    print(f"Example paths that will be sent to VLM:")
    for name in list(manifest.keys())[:3]:
        print(f"  {out_dir / name}")

    # Run VLM
    results = {}
    t0 = time.time()
    for i, (name, meta) in enumerate(manifest.items(), 1):
        img_path = out_dir / name
        r = run_one(img_path)
        r["true_class"] = meta["true_class"]
        r["person"] = meta["person"]
        results[name] = r
        pred = r.get("predicted_class", "ERR")
        mark = "✓" if pred == meta["true_class"] else "✗"
        print(f"  [{i}/{len(manifest)}] {mark} true={meta['true_class']:18s} pred={pred:18s} "
              f"conf={r.get('confidence', '?')}  {time.time()-t0:.0f}s elapsed")

    # Summary
    from collections import Counter
    y_true, y_pred = [], []
    for v in results.values():
        if "predicted_class" in v and v["predicted_class"] in CLASSES:
            y_true.append(v["true_class"])
            y_pred.append(v["predicted_class"])
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    print(f"\n=== ISOLATED VLM SANITY (15 scans, stratified) ===")
    print(f"N = {len(y_true)}, Correct = {correct}, Accuracy = {correct/max(len(y_true),1):.2%}")
    print(f"Per-class correctness:")
    by_cls = Counter()
    right_by_cls = Counter()
    for t, p in zip(y_true, y_pred):
        by_cls[t] += 1
        if t == p:
            right_by_cls[t] += 1
    for c in CLASSES:
        if c in by_cls:
            print(f"  {c:20s}: {right_by_cls[c]}/{by_cls[c]}")

    # Save
    (out_dir / "sanity_results.json").write_text(
        json.dumps({"results": results, "manifest": manifest}, indent=2))
    print(f"\nResults saved to {out_dir}/sanity_results.json")
    print(f"Keep or delete: `rm -rf {out_dir}`")


if __name__ == "__main__":
    main()
