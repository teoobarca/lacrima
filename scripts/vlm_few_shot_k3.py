"""Few-shot / retrieval-augmented VLM classifier — **k=3 variant**.

Same design as `vlm_few_shot.py` but with **3 nearest anchors per class**
(15 anchors total in a 5-col x 3-row grid) plus the query tile below.

Rendering + Claude CLI calls run in a ProcessPoolExecutor (default 8 workers).

Outputs
-------
- cache/vlm_few_shot_k3_predictions.json   (raw per-scan results)
- cache/vlm_few_shot_k3_collages/*.png     (per-query 5x3 collage images)
- reports/VLM_FEW_SHOT_K3_RESULTS.md       (markdown summary — written at end)
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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from teardrop.data import CLASSES, enumerate_samples, preprocess_spm  # noqa: E402
from teardrop.safe_paths import SAFE_ROOT, assert_prompt_safe  # noqa: E402

CACHE_DIR = REPO / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TILE_DIR = SAFE_ROOT / "few_shot_k3" / "tiles"
TILE_DIR.mkdir(parents=True, exist_ok=True)
COLLAGE_DIR = SAFE_ROOT / "few_shot_k3" / "collages"
COLLAGE_DIR.mkdir(parents=True, exist_ok=True)
PRED_CACHE = CACHE_DIR / "vlm_few_shot_k3_predictions.json"
EMB_PATH = CACHE_DIR / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz"

K_PER_CLASS = 3  # <<< the only structural change vs k=2

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_APPEND = (
    "You are a vision-language classifier. Respond with one JSON object only "
    "and nothing else. No markdown fence, no preamble, no tool calls beyond "
    "reading the referenced image."
)

PROMPT_TEMPLATE = """You are a medical expert classifying AFM (atomic force microscopy) scans of dried tear droplets.

Read the image at {img_path}. It is a single composite figure containing:
- A 5-column x 3-row grid of labeled reference anchors (15 total): for EACH of the 5 classes, 3 training examples closest to the query in DINOv2 embedding space. Each anchor column is labeled with its class name at the top.
- One QUERY tile below the anchor grid, labeled "QUERY".

All tiles use the afmhot colormap (bright = high topography, dark = low). Physical scale: 1 px ~ 90 nm; full tile field of view ~ 46 um.

The 5 possible classes are: ZdraviLudia, Diabetes, PGOV_Glaukom, SklerozaMultiplex, SucheOko.

Morphological signatures to keep in mind:
- ZdraviLudia (healthy): dense dendritic ferning, uniform fine branching, high fractal dimension (~1.75).
- Diabetes: thicker crystals, elevated roughness, glycated proteins -> coarser lattice, denser packing.
- PGOV_Glaukom: granular, loops/rings, MMP-9 degradation -> shorter thicker branches, chaotic texture, lower fractal D.
- SklerozaMultiplex: heterogeneous within one scan, mixed coarse rods + fine granules, high intra-sample variance.
- SucheOko (dry eye): fragmented / sparse network, amorphous empty regions, fractal D < 1.65.

Compare the QUERY tile to the 15 anchor tiles. Classify the query into ONE of the 5 classes by finding the anchor(s) it most visually resembles. With 3 anchors per class you can judge intra-class consistency — prefer the class whose 3 anchors collectively look most like the query.

Respond ONLY with a single JSON object, no markdown fence, exactly this shape:

{{"predicted_class": "<one of the 5 class names>", "confidence": <float 0 to 1>, "reasoning": "<1-2 sentences citing which anchor(s) the query most resembles and what morphological features>", "most_similar_anchor_class": "<one of the 5 class names>"}}"""


# ---------------------------------------------------------------------------
# Tile rendering
# ---------------------------------------------------------------------------


def tile_filename(cls: str, raw_path: Path) -> Path:
    """Class-neutral tile path (see teardrop.safe_paths). ``cls`` is
    deliberately unused — filename is ``scan_<sha1>.png``."""
    import hashlib as _hl
    try:
        rel = str(raw_path.resolve().relative_to(REPO))
    except ValueError:
        rel = str(raw_path)
    tid = _hl.sha1(rel.encode("utf-8")).hexdigest()[:16]
    return TILE_DIR / f"scan_{tid}.png"


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
# Collage composer  (5 cols x 3 rows)
# ---------------------------------------------------------------------------


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        if Path(name).exists():
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
    return ImageFont.load_default()


def compose_collage_k3(
    anchor_tiles: list[tuple[str, Path, str]],  # list of (class_name, tile_path, anchor_id)
    query_tile: Path,
    query_id: str,
    out_path: Path,
    *,
    anchor_size: int = 240,
    query_size: int = 384,
) -> Path:
    """Compose a single labeled PNG with 15 anchors (5 cols x 3 rows) + query.

    Layout:
        [ZdraviLudia #1] [Diabetes #1] [Glaukom #1] [SM #1] [SucheOko #1]
        [ZdraviLudia #2] [Diabetes #2] [Glaukom #2] [SM #2] [SucheOko #2]
        [ZdraviLudia #3] [Diabetes #3] [Glaukom #3] [SM #3] [SucheOko #3]
                            [ QUERY (enlarged, red border) ]
    """
    assert len(anchor_tiles) == 5 * K_PER_CLASS, f"expected {5*K_PER_CLASS} tiles, got {len(anchor_tiles)}"
    header_h = 36
    label_h = 22
    tile_pad = 8
    query_pad = 14

    cols = 5
    rows = K_PER_CLASS  # 3

    grid_w = cols * anchor_size + (cols + 1) * tile_pad
    grid_h = header_h + rows * (anchor_size + label_h) + (rows + 1) * tile_pad
    query_h = query_pad + 28 + query_size + query_pad
    total_w = max(grid_w, query_size + 2 * query_pad)
    total_h = grid_h + query_h + 10

    bg = Image.new("RGB", (total_w, total_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(bg)
    header_font = _get_font(20)
    sub_font = _get_font(13)
    query_font = _get_font(22)

    # 1) class headers (one per column)
    for ci, cls in enumerate(CLASSES):
        x0 = tile_pad + ci * (anchor_size + tile_pad)
        bbox = draw.textbbox((0, 0), cls, font=header_font)
        tw = bbox[2] - bbox[0]
        tx = x0 + (anchor_size - tw) // 2
        draw.text((tx, 6), cls, fill=(255, 255, 180), font=header_font)

    # 2) place anchors. Order = CLASSES[0] #1, CLASSES[0] #2, CLASSES[0] #3, CLASSES[1] #1, ...
    for ci, cls in enumerate(CLASSES):
        for ri in range(rows):
            idx = ci * rows + ri
            _cls_name, tile_path, anchor_id = anchor_tiles[idx]
            assert _cls_name == cls
            x0 = tile_pad + ci * (anchor_size + tile_pad)
            y0 = header_h + tile_pad + ri * (anchor_size + label_h + tile_pad)
            try:
                tim = Image.open(tile_path).convert("RGB").resize(
                    (anchor_size, anchor_size), Image.Resampling.LANCZOS
                )
            except Exception as e:
                tim = Image.new("RGB", (anchor_size, anchor_size), (50, 0, 0))
                ImageDraw.Draw(tim).text((8, 8), f"ERR {e}"[:40], fill=(255, 200, 200))
            bg.paste(tim, (x0, y0))
            sub = f"{cls} #{ri + 1}: {anchor_id[:24]}"
            draw.text((x0 + 4, y0 + anchor_size + 3), sub, fill=(210, 210, 210), font=sub_font)

    # 3) query tile (centered below grid)
    qy0 = grid_h + query_pad + 28
    qx0 = (total_w - query_size) // 2
    qhdr = "QUERY (classify this)"
    bbox = draw.textbbox((0, 0), qhdr, font=query_font)
    tw = bbox[2] - bbox[0]
    draw.text(((total_w - tw) // 2, grid_h + query_pad), qhdr, fill=(255, 180, 180), font=query_font)
    try:
        qim = Image.open(query_tile).convert("RGB").resize(
            (query_size, query_size), Image.Resampling.LANCZOS
        )
    except Exception as e:
        qim = Image.new("RGB", (query_size, query_size), (50, 0, 0))
        ImageDraw.Draw(qim).text((8, 8), f"ERR {e}"[:40], fill=(255, 200, 200))
    border = 3
    qbg = Image.new("RGB", (query_size + 2 * border, query_size + 2 * border), (220, 40, 40))
    qbg.paste(qim, (border, border))
    bg.paste(qbg, (qx0 - border, qy0 - border))

    bg.save(out_path, format="PNG", optimize=True)
    return out_path


# ---------------------------------------------------------------------------
# Claude CLI wrapper (module-level so ProcessPoolExecutor can pickle it)
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


def call_claude_cli(img_path: Path, model: str = "claude-haiku-4-5", timeout_s: int = 120) -> dict:
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
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(REPO),
        )
    except subprocess.TimeoutExpired:
        return {"error": "cli timeout", "latency_s": time.time() - t0}
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
        "most_similar_anchor_class": str(parsed.get("most_similar_anchor_class", "")),
        "latency_s": latency,
        "cost_usd": envelope.get("total_cost_usd", 0.0),
        "duration_api_ms": envelope.get("duration_api_ms", 0),
        "result_text": result_text,
    }


def worker_classify(task: tuple) -> tuple:
    """Worker: receives (key, collage_path_str, true_class, person, anchor_meta, query_path, model).
    Returns (key, result_dict).
    """
    key, collage_path_str, true_class, person, anchor_meta, query_path, model = task
    res = call_claude_cli(Path(collage_path_str), model=model)
    res["true_class"] = true_class
    res["person"] = person
    res["query_path"] = query_path
    res["collage_path"] = collage_path_str
    res["anchors"] = anchor_meta
    return key, res


# ---------------------------------------------------------------------------
# Person-disjoint kNN retrieval
# ---------------------------------------------------------------------------


def load_embeddings():
    z = np.load(EMB_PATH, allow_pickle=True)
    X = z["X_scan"].astype(np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return {
        "X": Xn,
        "y": z["scan_y"],
        "groups": np.asarray(z["scan_groups"], dtype=str),
        "paths": np.asarray(z["scan_paths"], dtype=str),
    }


def retrieve_anchors_per_class(
    emb: dict,
    query_idx: int,
    query_person: str,
    k_per_class: int = K_PER_CLASS,
) -> dict[int, list[int]]:
    X = emb["X"]
    y = emb["y"]
    groups = emb["groups"]
    sims = X @ X[query_idx]
    excl = (groups == query_person)
    excl[query_idx] = True

    out: dict[int, list[int]] = {}
    for cls_idx in range(len(CLASSES)):
        mask = (y == cls_idx) & (~excl)
        if mask.sum() == 0:
            raise RuntimeError(
                f"No available anchors for class {CLASSES[cls_idx]} "
                f"excluding person {query_person}"
            )
        idxs = np.where(mask)[0]
        order = np.argsort(-sims[idxs])
        top = idxs[order[:k_per_class]]
        if len(top) < k_per_class:
            # pool smaller than k — duplicate the best available (rare; SucheOko with 2 persons
            # and same-person exclusion can leave us with <3 scans for SucheOko itself, but since
            # other classes always have >>3 scans after exclusion this only matters when the
            # query IS SucheOko).
            pad = np.full(k_per_class - len(top), top[0], dtype=int)
            top = np.concatenate([top, pad])
        out[cls_idx] = top.tolist()
    return out


# ---------------------------------------------------------------------------
# Subset selection (copy from k=2 script for consistency)
# ---------------------------------------------------------------------------


def stratified_person_disjoint(samples, per_class: int = 12, seed: int = 42):
    rng = np.random.default_rng(seed)
    by_cls: dict[str, dict[str, list]] = {}
    for s in samples:
        by_cls.setdefault(s.cls, {}).setdefault(s.person, []).append(s)
    picked = []
    for cls in CLASSES:
        persons = list(by_cls.get(cls, {}).keys())
        rng.shuffle(persons)
        chosen: list = []
        person_scans = {p: sorted(by_cls[cls][p], key=lambda s: str(s.raw_path))
                        for p in persons}
        round_idx = 0
        while len(chosen) < per_class:
            added = 0
            for p in persons:
                if round_idx < len(person_scans[p]):
                    chosen.append(person_scans[p][round_idx])
                    added += 1
                    if len(chosen) >= per_class:
                        break
            if added == 0:
                break
            round_idx += 1
        picked.extend(chosen[:per_class])
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


def prepare_tasks(samples_to_score, all_samples, cache, model):
    """Render all needed tiles + compose collages, return task list for workers."""
    emb = load_embeddings()
    abs_to_emb_idx = {p: i for i, p in enumerate(emb["paths"])}
    path_to_sample = {str(s.raw_path): s for s in all_samples}

    tasks = []
    skipped = 0
    for i, s in enumerate(samples_to_score):
        key = str(s.raw_path.relative_to(REPO))
        if key in cache and "predicted_class" in cache[key] and cache[key]["predicted_class"] in CLASSES:
            skipped += 1
            continue

        abs_path = str(s.raw_path)
        if abs_path not in abs_to_emb_idx:
            cache[key] = {"error": "no embedding for query", "true_class": s.cls, "person": s.person}
            print(f"[prep] NO-EMB: {key}")
            continue
        q_idx = abs_to_emb_idx[abs_path]

        try:
            anchors_per_cls = retrieve_anchors_per_class(emb, q_idx, s.person, k_per_class=K_PER_CLASS)
        except RuntimeError as e:
            cache[key] = {"error": str(e), "true_class": s.cls, "person": s.person}
            print(f"[prep] RETRIEVE FAIL: {key} {e}")
            continue

        q_tile = tile_filename(s.cls, s.raw_path)
        try:
            render_scan_tile(s.raw_path, q_tile)
        except Exception as e:
            cache[key] = {"error": f"query render failed: {e}", "true_class": s.cls, "person": s.person}
            print(f"[prep] RENDER FAIL: {key} {e}")
            continue

        anchor_info: list[tuple[str, Path, str]] = []
        anchor_meta: list[dict] = []
        render_fail = False
        for ci, cls in enumerate(CLASSES):
            for rank, emb_idx in enumerate(anchors_per_cls[ci]):
                anchor_abs = emb["paths"][emb_idx]
                a_sample = path_to_sample.get(anchor_abs)
                if a_sample is None:
                    render_fail = True
                    break
                a_tile = tile_filename(a_sample.cls, a_sample.raw_path)
                try:
                    render_scan_tile(a_sample.raw_path, a_tile)
                except Exception as e:
                    print(f"  anchor render failed {anchor_abs}: {e}")
                    render_fail = True
                    break
                import hashlib as _hl_anchor
                _a_rel = str(a_sample.raw_path.relative_to(REPO))
                _anchor_visible_id = _hl_anchor.sha1(_a_rel.encode("utf-8")).hexdigest()[:10]
                anchor_info.append((cls, a_tile, _anchor_visible_id))
                anchor_meta.append({
                    "class": cls,
                    "rank": rank + 1,
                    "name": a_sample.raw_path.name,
                    "person": a_sample.person,
                    "path": str(a_sample.raw_path.relative_to(REPO)),
                })
            if render_fail:
                break
        if render_fail:
            cache[key] = {"error": "anchor render failed", "true_class": s.cls, "person": s.person}
            continue

        # person-integrity assertion
        for am in anchor_meta:
            assert am["person"] != s.person, f"LEAK: anchor person == query person for {key}"

        import hashlib as _hl_collage
        _rel_q = str(s.raw_path.relative_to(REPO))
        _collage_id = _hl_collage.sha1(_rel_q.encode("utf-8")).hexdigest()[:16]
        collage_path = COLLAGE_DIR / f"scan_{_collage_id}.png"
        compose_collage_k3(anchor_info, q_tile, s.raw_path.stem, collage_path)

        tasks.append((
            key,
            str(collage_path),
            s.cls,
            s.person,
            anchor_meta,
            key,
            model,
        ))

    print(f"[prep] {len(tasks)} tasks to dispatch, {skipped} already cached")
    return tasks


def run_parallel(samples_to_score, all_samples, model: str, time_budget_s: float,
                 cache: dict[str, dict], workers: int = 8) -> dict[str, dict]:
    t_start = time.time()
    tasks = prepare_tasks(samples_to_score, all_samples, cache, model)
    save_cache(cache)  # save any pre-flight error entries

    if not tasks:
        print("[run] nothing to do (all cached)")
        return cache

    print(f"[run] dispatching {len(tasks)} tasks across {workers} workers")
    done_count = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker_classify, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            done_count += 1
            elapsed = time.time() - t_start
            if elapsed > time_budget_s:
                print(f"[budget] hit {elapsed:.0f}s > {time_budget_s}s; waiting for remaining in-flight...")
            try:
                key, res = fut.result(timeout=180)
            except Exception as e:
                key = futures[fut]
                res = {"error": f"worker exception: {type(e).__name__}: {e}"}
            cache[key] = res
            pred = res.get("predicted_class", "ERR")
            truth = res.get("true_class", "?")
            ok = "OK" if pred == truth else "--"
            cost = res.get("cost_usd", 0.0) or 0.0
            lat = res.get("latency_s", 0.0) or 0.0
            sim_anchor = res.get("most_similar_anchor_class", "?")
            conf = res.get("confidence", 0) or 0
            print(f"[{done_count}/{len(tasks)}] {ok} {key}  true={truth}  pred={pred}  sim={sim_anchor}  conf={conf:.2f}  t={lat:.1f}s  ${cost:.4f}")
            if done_count % 5 == 0 or done_count == len(tasks):
                save_cache(cache)
    save_cache(cache)
    return cache


def evaluate(cache: dict[str, dict], keys: list[str]) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    sim_anchor_matches = 0
    anchor_was_true = 0
    for k in keys:
        entry = cache.get(k)
        if not entry or "predicted_class" not in entry or entry["predicted_class"] not in CLASSES:
            continue
        y_true.append(entry["true_class"])
        y_pred.append(entry["predicted_class"])
        confidences.append(entry.get("confidence", 0.0))
        if entry.get("most_similar_anchor_class") == entry["predicted_class"]:
            sim_anchor_matches += 1
        if "anchors" in entry:
            anchor_classes = {a["class"] for a in entry["anchors"]}
            if entry["true_class"] in anchor_classes:
                anchor_was_true += 1
    if not y_true:
        return {"error": "no valid predictions"}
    labels = CLASSES
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "n": len(y_true),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": labels,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "sim_anchor_agrees_with_pred_pct": sim_anchor_matches / len(y_true),
        "true_class_always_in_anchors_pct": anchor_was_true / len(y_true),
    }


def print_eval(name: str, ev: dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    if "error" in ev:
        print("ERROR:", ev["error"])
        return
    print(f"n = {ev['n']}  acc = {ev['accuracy']:.4f}  "
          f"f1_weighted = {ev['f1_weighted']:.4f}  f1_macro = {ev['f1_macro']:.4f}  "
          f"mean_conf = {ev['mean_confidence']:.3f}")
    print(f"  'most_similar_anchor' matches prediction: {ev['sim_anchor_agrees_with_pred_pct']:.1%}")
    print(f"  true class always in anchors (sanity):   {ev['true_class_always_in_anchors_pct']:.1%}")
    print("\nPer-class F1:")
    for cls in ev["labels"]:
        row = ev["classification_report"].get(cls, {})
        print(f"  {cls:22s}  P={row.get('precision', 0):.3f}  R={row.get('recall', 0):.3f}  F1={row.get('f1-score', 0):.3f}  support={int(row.get('support', 0))}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("  " + "  ".join(f"{c[:6]:>6s}" for c in ev["labels"]))
    for lab, row in zip(ev["labels"], ev["confusion_matrix"]):
        print(f"  {lab[:6]:>6s} " + "  ".join(f"{v:>6d}" for v in row))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=12, help="per-class stratified picks")
    ap.add_argument("--full", action="store_true", help="score every sample (overrides subset)")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--time-budget-s", type=int, default=1800, help="hard wall-clock cap (default 30 min)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-only", action="store_true", help="skip LLM calls; evaluate cache")
    args = ap.parse_args()

    all_samples = enumerate_samples(REPO / "TRAIN_SET")
    print(f"Loaded {len(all_samples)} total samples")

    if args.full:
        samples = all_samples
        print(f"Full: {len(samples)} samples")
    else:
        samples = stratified_person_disjoint(all_samples, per_class=args.subset, seed=args.seed)
        print(f"Subset: {len(samples)} samples (stratified, person-disjoint within class where possible)")

    cache = load_cache()
    if not args.eval_only:
        cache = run_parallel(samples, all_samples, args.model, args.time_budget_s,
                             cache, workers=args.workers)
        save_cache(cache)

    scored_keys = [str(s.raw_path.relative_to(REPO)) for s in samples]
    ev = evaluate(cache, scored_keys)
    print_eval(f"VLM few-shot k={K_PER_CLASS} (15-anchor 5x3 collage)", ev)

    total_cost = sum(float(e.get("cost_usd", 0.0) or 0.0) for k, e in cache.items() if k in scored_keys)
    print(f"\nTotal cost on scored subset: ${total_cost:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
