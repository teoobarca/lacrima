"""Few-shot / retrieval-augmented VLM classifier for AFM tear scans.

For each query scan we compose a single image collage containing 10
labeled anchor tiles (2 nearest training exemplars per class, retrieved
via DINOv2-B cached embeddings, person-disjoint from the query) plus the
query tile itself. We then ask Claude (via `claude -p` CLI) to classify
the query using the visual anchors as references.

This is a hybrid: DINOv2 retrieval + Claude VLM classification.

Usage
-----
    # Stratified 40-scan subset (8 per class):
    .venv/bin/python scripts/vlm_few_shot.py --subset 8

    # Full dataset:
    .venv/bin/python scripts/vlm_few_shot.py --full

Outputs
-------
- cache/vlm_few_shot_predictions.json   (raw per-scan results)
- cache/vlm_few_shot_collages/*.png     (per-query collage images)
- reports/VLM_FEW_SHOT_RESULTS.md       (markdown summary — written at end)
"""
from __future__ import annotations

import argparse
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from teardrop.data import CLASSES, enumerate_samples, preprocess_spm  # noqa: E402
from teardrop.safe_paths import SAFE_ROOT, assert_prompt_safe  # noqa: E402

CACHE_DIR = REPO / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TILE_DIR = SAFE_ROOT / "few_shot_sonnet" / "tiles"
TILE_DIR.mkdir(parents=True, exist_ok=True)
COLLAGE_DIR = SAFE_ROOT / "few_shot_sonnet" / "collages"
COLLAGE_DIR.mkdir(parents=True, exist_ok=True)
PRED_CACHE = CACHE_DIR / "vlm_sonnet_predictions.json"
EMB_PATH = CACHE_DIR / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz"


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
- A 5-column x 2-row grid of labeled reference anchors (10 total): for EACH of the 5 classes, 2 training examples closest to the query in DINOv2 embedding space. Each anchor is labeled with its class name at the top.
- One QUERY tile below the anchor grid, labeled "QUERY".

All tiles use the afmhot colormap (bright = high topography, dark = low). Physical scale: 1 px ~ 90 nm; full tile field of view ~ 46 um.

The 5 possible classes are: ZdraviLudia, Diabetes, PGOV_Glaukom, SklerozaMultiplex, SucheOko.

Morphological signatures to keep in mind:
- ZdraviLudia (healthy): dense dendritic ferning, uniform fine branching, high fractal dimension (~1.75).
- Diabetes: thicker crystals, elevated roughness, glycated proteins -> coarser lattice, denser packing.
- PGOV_Glaukom: granular, loops/rings, MMP-9 degradation -> shorter thicker branches, chaotic texture, lower fractal D.
- SklerozaMultiplex: heterogeneous within one scan, mixed coarse rods + fine granules, high intra-sample variance.
- SucheOko (dry eye): fragmented / sparse network, amorphous empty regions, fractal D < 1.65.

Compare the QUERY tile to the 10 anchor tiles. Classify the query into ONE of the 5 classes by finding the anchor(s) it most visually resembles.

Respond ONLY with a single JSON object, no markdown fence, exactly this shape:

{{"predicted_class": "<one of the 5 class names>", "confidence": <float 0 to 1>, "reasoning": "<1-2 sentences citing which anchor(s) the query most resembles and what morphological features>", "most_similar_anchor_class": "<one of the 5 class names>"}}"""


# ---------------------------------------------------------------------------
# Tile rendering
# ---------------------------------------------------------------------------


def tile_filename(cls: str, raw_path: Path) -> Path:
    """Class-neutral tile path (see teardrop.safe_paths). ``cls`` is
    deliberately unused — the filename is ``scan_<sha1>.png``."""
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
# Collage composer
# ---------------------------------------------------------------------------


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Try common macOS fonts; fall back to default bitmap.
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


def compose_collage(
    anchor_tiles: list[tuple[str, Path, str]],  # list of (class_name, tile_path, anchor_id)
    query_tile: Path,
    query_id: str,
    out_path: Path,
    *,
    anchor_size: int = 256,
    query_size: int = 384,
) -> Path:
    """Compose a single labeled PNG with 10 anchors (5 cols x 2 rows) + query.

    Layout:
        [ZdraviLudia #1] [Diabetes #1] [Glaukom #1] [SM #1] [SucheOko #1]
        [ZdraviLudia #2] [Diabetes #2] [Glaukom #2] [SM #2] [SucheOko #2]
                            [ QUERY (enlarged) ]

    Class labels appear above each anchor column (merged across rows).
    """
    assert len(anchor_tiles) == 10
    header_h = 36       # class-header strip
    label_h = 22        # per-tile subtitle strip
    tile_pad = 8
    query_pad = 14

    cols = 5
    rows = 2

    grid_w = cols * anchor_size + (cols + 1) * tile_pad
    grid_h = header_h + rows * (anchor_size + label_h) + (rows + 1) * tile_pad
    query_h = query_pad + 28 + query_size + query_pad  # label + tile
    total_w = max(grid_w, query_size + 2 * query_pad)
    total_h = grid_h + query_h + 10

    bg = Image.new("RGB", (total_w, total_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(bg)
    header_font = _get_font(20)
    sub_font = _get_font(14)
    query_font = _get_font(22)

    # 1) class headers (one per column)
    # class order in collage = CLASSES (same order always)
    for ci, cls in enumerate(CLASSES):
        x0 = tile_pad + ci * (anchor_size + tile_pad)
        # center text above column
        bbox = draw.textbbox((0, 0), cls, font=header_font)
        tw = bbox[2] - bbox[0]
        tx = x0 + (anchor_size - tw) // 2
        draw.text((tx, 6), cls, fill=(255, 255, 180), font=header_font)

    # 2) place anchors — assume anchor_tiles are grouped by class, 2 per class,
    # ordered CLASSES[0] #1, CLASSES[0] #2, CLASSES[1] #1, ...
    # rearrange into column-major 5x2
    for ci, cls in enumerate(CLASSES):
        for ri in range(rows):
            idx = ci * 2 + ri
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
            # subtitle with anchor id (short)
            sub = f"{cls} #{ri + 1}: {anchor_id[:28]}"
            draw.text((x0 + 4, y0 + anchor_size + 3), sub, fill=(210, 210, 210), font=sub_font)

    # 3) query tile (centered below grid)
    qy0 = grid_h + query_pad + 28
    qx0 = (total_w - query_size) // 2
    # "QUERY" header
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
    # red border around query
    border = 3
    qbg = Image.new("RGB", (query_size + 2 * border, query_size + 2 * border), (220, 40, 40))
    qbg.paste(qim, (border, border))
    bg.paste(qbg, (qx0 - border, qy0 - border))

    bg.save(out_path, format="PNG", optimize=True)
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


def call_claude_cli(img_path: Path, model: str = "claude-sonnet-4-6", timeout_s: int = 180) -> dict:
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
        "most_similar_anchor_class": str(parsed.get("most_similar_anchor_class", "")),
        "latency_s": latency,
        "cost_usd": envelope.get("total_cost_usd", 0.0),
        "duration_api_ms": envelope.get("duration_api_ms", 0),
        "result_text": result_text,
    }


# ---------------------------------------------------------------------------
# Person-disjoint kNN retrieval
# ---------------------------------------------------------------------------


def load_embeddings():
    z = np.load(EMB_PATH, allow_pickle=True)
    X = z["X_scan"].astype(np.float32)
    # L2 normalize for cosine similarity
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
    k_per_class: int = 2,
) -> dict[int, list[int]]:
    """For each class, return the indices of the k nearest training scans
    to the query (cosine sim in DINOv2 space) EXCLUDING scans from the
    query's own person.
    """
    X = emb["X"]
    y = emb["y"]
    groups = emb["groups"]
    # cosine similarity
    sims = X @ X[query_idx]
    excl = (groups == query_person)
    # never retrieve self
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
        # sort by similarity desc, take k
        order = np.argsort(-sims[idxs])
        top = idxs[order[:k_per_class]]
        if len(top) < k_per_class:
            # if pool smaller than k, reuse the best available (duplicate)
            pad = np.full(k_per_class - len(top), top[0], dtype=int)
            top = np.concatenate([top, pad])
        out[cls_idx] = top.tolist()
    return out


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


def stratified_person_disjoint(samples, per_class: int = 8, seed: int = 42):
    rng = np.random.default_rng(seed)
    by_cls: dict[str, dict[str, list]] = {}
    for s in samples:
        by_cls.setdefault(s.cls, {}).setdefault(s.person, []).append(s)
    picked = []
    for cls in CLASSES:
        # flatten, diversify by person first, then extra scans from bigger persons
        persons = list(by_cls.get(cls, {}).keys())
        rng.shuffle(persons)
        chosen: list = []
        # round-robin: first scan of each person, then 2nd scan of each, ...
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


def build_samples_index(samples):
    """Map repo-relative path string -> Sample object."""
    return {str(s.raw_path.relative_to(REPO)): s for s in samples}


def run(samples_to_score, all_samples, model: str, time_budget_s: float, cache: dict[str, dict]) -> dict[str, dict]:
    t_start = time.time()
    emb = load_embeddings()
    # map absolute path -> embedding index
    abs_to_emb_idx = {p: i for i, p in enumerate(emb["paths"])}
    # map embedding index -> Sample
    path_to_sample = {str(s.raw_path): s for s in all_samples}

    for i, s in enumerate(samples_to_score):
        key = str(s.raw_path.relative_to(REPO))
        if key in cache and "predicted_class" in cache[key] and cache[key]["predicted_class"] in CLASSES:
            print(f"[{i+1}/{len(samples_to_score)}] cached: {key} -> {cache[key]['predicted_class']}")
            continue
        elapsed = time.time() - t_start
        if elapsed > time_budget_s:
            print(f"[budget] stopping at {i}/{len(samples_to_score)} after {elapsed:.0f}s")
            break

        abs_path = str(s.raw_path)
        if abs_path not in abs_to_emb_idx:
            cache[key] = {"error": "no embedding for query", "true_class": s.cls, "person": s.person}
            save_cache(cache)
            print(f"[{i+1}/{len(samples_to_score)}] NO-EMB: {key}")
            continue
        q_idx = abs_to_emb_idx[abs_path]

        # retrieve 2 anchors per class
        try:
            anchors_per_cls = retrieve_anchors_per_class(emb, q_idx, s.person, k_per_class=2)
        except RuntimeError as e:
            cache[key] = {"error": str(e), "true_class": s.cls, "person": s.person}
            save_cache(cache)
            print(f"[{i+1}/{len(samples_to_score)}] RETRIEVE FAIL: {key} {e}")
            continue

        # render query tile + all 10 anchor tiles
        q_tile = tile_filename(s.cls, s.raw_path)
        try:
            render_scan_tile(s.raw_path, q_tile)
        except Exception as e:
            cache[key] = {"error": f"query render failed: {e}", "true_class": s.cls, "person": s.person}
            save_cache(cache)
            print(f"[{i+1}/{len(samples_to_score)}] RENDER FAIL: {key} {e}")
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
                # Obfuscate anchor id so raw-filename class fragments
                # can't be OCR'd off the rendered collage.
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
            save_cache(cache)
            continue

        # person-integrity assertion
        for am in anchor_meta:
            assert am["person"] != s.person, f"LEAK: anchor person == query person for {key}"

        # compose collage
        import hashlib as _hl_collage
        _rel_q = str(s.raw_path.relative_to(REPO))
        _collage_id = _hl_collage.sha1(_rel_q.encode("utf-8")).hexdigest()[:16]
        collage_path = COLLAGE_DIR / f"scan_{_collage_id}.png"
        compose_collage(anchor_info, q_tile, s.raw_path.stem, collage_path)

        # call Claude
        res = call_claude_cli(collage_path, model=model)
        res["true_class"] = s.cls
        res["person"] = s.person
        res["query_path"] = key
        res["collage_path"] = str(collage_path.relative_to(REPO))
        res["anchors"] = anchor_meta
        cache[key] = res
        save_cache(cache)
        pred = res.get("predicted_class", "ERR")
        ok = "OK" if pred == s.cls else "--"
        cost = res.get("cost_usd", 0.0)
        lat = res.get("latency_s", 0.0)
        sim_anchor = res.get("most_similar_anchor_class", "?")
        print(f"[{i+1}/{len(samples_to_score)}] {ok} {key}  true={s.cls}  pred={pred}  sim_to={sim_anchor}  conf={res.get('confidence', 0):.2f}  t={lat:.1f}s  ${cost:.4f}")
    return cache


def evaluate(cache: dict[str, dict], keys: list[str]) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    sim_anchor_matches = 0  # how often "most_similar_anchor_class" == predicted_class
    anchor_was_true = 0     # how often any retrieved anchor was the true class
    for k in keys:
        entry = cache.get(k)
        if not entry or "predicted_class" not in entry or entry["predicted_class"] not in CLASSES:
            continue
        y_true.append(entry["true_class"])
        y_pred.append(entry["predicted_class"])
        confidences.append(entry.get("confidence", 0.0))
        if entry.get("most_similar_anchor_class") == entry["predicted_class"]:
            sim_anchor_matches += 1
        # retrieval recall: was the true class present in anchors? (it always is — we fetch 2 per class)
        if "anchors" in entry:
            anchor_classes = {a["class"] for a in entry["anchors"]}
            if entry["true_class"] in anchor_classes:
                anchor_was_true += 1
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
        "sim_anchor_agrees_with_pred_pct": sim_anchor_matches / len(y_true),
        "true_class_always_in_anchors_pct": anchor_was_true / len(y_true),
    }


def print_eval(name: str, ev: dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    if "error" in ev:
        print("ERROR:", ev["error"])
        return
    print(f"n = {ev['n']}  acc = {ev['accuracy']:.4f}  f1_macro = {ev['f1_macro']:.4f}  mean_conf = {ev['mean_confidence']:.3f}")
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


def _load_zero_shot_cache(prefer: str | None = None) -> tuple[dict[str, dict], str] | tuple[None, None]:
    """Load the zero-shot VLM predictions keyed by repo-relative path.

    Preference order:
      1. explicit `prefer` path-keyed file (if given)
      2. `vlm_predictions.json` (legacy original ~88% baseline)
      3. `vlm_haiku_predictions_subset.json` (haiku zero-shot subset)
      4. `vlm_honest_predictions.json` (honest baseline, via manifest)
    """
    def _load_path_keyed(name: str):
        p = CACHE_DIR / name
        if p.exists():
            return json.loads(p.read_text()), name
        return None

    candidates = []
    if prefer:
        candidates.append(prefer)
    candidates += [
        "vlm_predictions.json",
        "vlm_haiku_predictions_subset.json",
    ]
    for name in candidates:
        res = _load_path_keyed(name)
        if res is not None:
            return res

    honest = CACHE_DIR / "vlm_honest_predictions.json"
    manifest = CACHE_DIR / "vlm_honest_manifest.json"
    if honest.exists() and manifest.exists():
        preds = json.loads(honest.read_text())
        mani = json.loads(manifest.read_text())
        out: dict[str, dict] = {}
        for scan_id, entry in preds.items():
            meta = mani.get(scan_id)
            if not meta:
                continue
            raw = meta.get("raw_path", "")
            try:
                rel = str(Path(raw).resolve().relative_to(REPO))
            except ValueError:
                continue
            out[rel] = entry
        return out, "vlm_honest_predictions.json (via manifest)"
    return None, None


def compare_to_zeroshot(cache: dict[str, dict], keys: list[str], prefer: str | None = None) -> dict[str, Any]:
    """Compare the few-shot predictions to the zero-shot VLM predictions
    on the same scans (when both exist)."""
    zs, zs_name = _load_zero_shot_cache(prefer=prefer)
    if zs is None:
        return {"error": "no zero-shot cache"}
    agree = 0
    both_right = 0
    fs_right_only = 0
    zs_right_only = 0
    both_wrong = 0
    details: list[dict] = []
    overlap_keys = [k for k in keys if k in zs and k in cache
                    and zs[k].get("predicted_class") in CLASSES
                    and cache[k].get("predicted_class") in CLASSES]
    for k in overlap_keys:
        z_entry = zs[k]
        f_entry = cache[k]
        truth = f_entry["true_class"]
        zs_pred = z_entry["predicted_class"]
        fs_pred = f_entry["predicted_class"]
        if zs_pred == fs_pred:
            agree += 1
        if zs_pred == truth and fs_pred == truth:
            both_right += 1
        elif zs_pred != truth and fs_pred == truth:
            fs_right_only += 1
            details.append({"scan": k, "truth": truth, "zs": zs_pred, "fs": fs_pred, "flip": "ZS_wrong->FS_right"})
        elif zs_pred == truth and fs_pred != truth:
            zs_right_only += 1
            details.append({"scan": k, "truth": truth, "zs": zs_pred, "fs": fs_pred, "flip": "ZS_right->FS_wrong"})
        else:
            both_wrong += 1
    zs_acc = (both_right + zs_right_only) / max(1, len(overlap_keys))
    fs_acc = (both_right + fs_right_only) / max(1, len(overlap_keys))
    return {
        "n_overlap": len(overlap_keys),
        "zero_shot_source": zs_name,
        "zero_shot_acc": zs_acc,
        "few_shot_acc": fs_acc,
        "agreement_pct": agree / max(1, len(overlap_keys)),
        "both_right": both_right,
        "both_wrong": both_wrong,
        "few_shot_right_only": fs_right_only,
        "zero_shot_right_only": zs_right_only,
        "flips": details[:20],
    }


def write_markdown_report(args, ev, cmp_zs, total_cost, out_path: Path, *, all_cmps: dict | None = None):
    import datetime as dt
    lines: list[str] = []
    lines.append("# VLM Few-Shot Classification Results\n")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n")
    lines.append("## Setup\n")
    lines.append("- Retrieval: DINOv2-B (`cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`), cosine sim.")
    lines.append("- k=2 anchors per class (10 total), nearest-neighbor, **person-disjoint**.")
    lines.append("- Prompt: one composite image (5x2 labeled anchor grid + query tile below).")
    lines.append(f"- Model: {args.model}")
    lines.append(f"- Samples scored: {ev.get('n', 0)}")
    lines.append(f"- Total API cost (this run + cached): ${total_cost:.3f}\n")

    if "error" not in ev:
        lines.append("## Accuracy\n")
        lines.append(f"- Accuracy: **{ev['accuracy']:.4f}**")
        lines.append(f"- Macro-F1: **{ev['f1_macro']:.4f}**")
        lines.append(f"- Mean confidence: {ev['mean_confidence']:.3f}")
        lines.append(f"- Reasoning quality: VLM's `most_similar_anchor_class` agrees with `predicted_class` **{ev['sim_anchor_agrees_with_pred_pct']:.1%}** of the time.")
        lines.append(f"- Sanity: true class in anchor set {ev['true_class_always_in_anchors_pct']:.1%} (should be 100%).\n")

        lines.append("## Per-class F1\n")
        lines.append("| Class | P | R | F1 | Support |")
        lines.append("|---|---|---|---|---|")
        for cls in ev["labels"]:
            row = ev["classification_report"].get(cls, {})
            lines.append(f"| {cls} | {row.get('precision', 0):.3f} | {row.get('recall', 0):.3f} | {row.get('f1-score', 0):.3f} | {int(row.get('support', 0))} |")
        lines.append("")

        lines.append("## Confusion Matrix\n")
        lines.append("Rows=true, cols=pred.\n")
        header = "| true\\pred | " + " | ".join(c[:10] for c in ev["labels"]) + " |"
        sep = "|---|" + "|".join(["---"] * len(ev["labels"])) + "|"
        lines.append(header)
        lines.append(sep)
        for lab, row in zip(ev["labels"], ev["confusion_matrix"]):
            lines.append(f"| {lab[:10]} | " + " | ".join(str(v) for v in row) + " |")
        lines.append("")

    if all_cmps:
        lines.append("## All available zero-shot baselines (for context)\n")
        lines.append("| Baseline | N overlap | Zero-shot acc | Few-shot acc | Delta | FS fixes | FS regresses |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, c in sorted(all_cmps.items(), key=lambda kv: -kv[1]["n_overlap"]):
            lines.append(
                f"| `{key}` | {c['n_overlap']} | {c['zero_shot_acc']:.3f} | "
                f"{c['few_shot_acc']:.3f} | {c['few_shot_acc']-c['zero_shot_acc']:+.3f} | "
                f"{c['few_shot_right_only']} | {c['zero_shot_right_only']} |"
            )
        lines.append("")

    if cmp_zs and "error" not in cmp_zs:
        lines.append("## Few-Shot vs Zero-Shot Head-to-Head (largest overlap)\n")
        lines.append(f"- Zero-shot source: `{cmp_zs.get('zero_shot_source', 'unknown')}`")
        lines.append(f"- Overlap: {cmp_zs['n_overlap']} scans scored by both.")
        lines.append(f"- **Zero-shot accuracy**: {cmp_zs['zero_shot_acc']:.4f}")
        lines.append(f"- **Few-shot accuracy**:  {cmp_zs['few_shot_acc']:.4f}")
        delta = cmp_zs['few_shot_acc'] - cmp_zs['zero_shot_acc']
        lines.append(f"- **Delta**: {delta:+.4f}")
        lines.append(f"- Agreement: {cmp_zs['agreement_pct']:.1%}")
        lines.append(f"- Both right: {cmp_zs['both_right']}  |  Both wrong: {cmp_zs['both_wrong']}")
        lines.append(f"- Few-shot fixes (ZS-wrong -> FS-right): {cmp_zs['few_shot_right_only']}")
        lines.append(f"- Few-shot regresses (ZS-right -> FS-wrong): {cmp_zs['zero_shot_right_only']}\n")
        if cmp_zs.get("flips"):
            lines.append("### Flips (first 20)\n")
            lines.append("| Scan | Truth | Zero-shot | Few-shot | Flip |")
            lines.append("|---|---|---|---|---|")
            for f in cmp_zs["flips"]:
                lines.append(f"| `{f['scan']}` | {f['truth']} | {f['zs']} | {f['fs']} | {f['flip']} |")
            lines.append("")

    lines.append("## Baselines to compare against\n")
    lines.append("- Zero-shot VLM (honest, 40 overlap): 0.225 accuracy")
    lines.append("- Zero-shot VLM (curated first-scan-per-person subsets): 0.77 - 1.00 on 30-scan overlaps")
    lines.append("- v4 multiscale ensemble: 0.6887 macro-F1 (full 240)\n")

    # reasoning quality showcase
    try:
        cache_raw = json.loads(PRED_CACHE.read_text())
        scored = [v for v in cache_raw.values() if v.get("predicted_class") in CLASSES]
        correct = [v for v in scored if v.get("predicted_class") == v.get("true_class")]
        wrong = [v for v in scored if v.get("predicted_class") != v.get("true_class")]
        lines.append("## Reasoning quality (sample VLM explanations)\n")
        lines.append("### Correct predictions (cite anchors explicitly)\n")
        for v in correct[:3]:
            lines.append(f"- **true={v['true_class']}** pred={v['predicted_class']} (conf {v.get('confidence', 0):.2f}): {v['reasoning']}")
        lines.append("")
        if wrong:
            lines.append("### Wrong predictions (failure modes)\n")
            for v in wrong[:4]:
                lines.append(f"- **true={v['true_class']}** pred={v['predicted_class']} (conf {v.get('confidence', 0):.2f}): {v['reasoning']}")
            lines.append("")
    except Exception:
        pass

    out_path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=0, help="per-class stratified picks (0 = use --full)")
    ap.add_argument("--full", action="store_true", help="score every sample")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--time-budget-s", type=int, default=1800, help="hard wall-clock cap (default 30 min)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true", help="skip LLM calls; evaluate cache")
    args = ap.parse_args()

    all_samples = enumerate_samples(REPO / "TRAIN_SET")
    print(f"Loaded {len(all_samples)} total samples")

    if args.subset > 0:
        samples = stratified_person_disjoint(all_samples, per_class=args.subset, seed=args.seed)
        print(f"Subset: {len(samples)} samples (stratified, person-disjoint within class where possible)")
    elif args.full:
        samples = all_samples
        print(f"Full: {len(samples)} samples")
    else:
        ap.error("pass --subset N or --full")

    cache = load_cache()
    if not args.eval_only:
        cache = run(samples, all_samples, args.model, args.time_budget_s, cache)
        save_cache(cache)

    scored_keys = [str(s.raw_path.relative_to(REPO)) for s in samples]
    ev = evaluate(cache, scored_keys)
    print_eval("VLM few-shot (10-anchor collage)", ev)

    # Compare against every zero-shot baseline we can find
    baseline_candidates = [
        "vlm_predictions.json",
        "vlm_haiku_predictions_subset.json",
        "vlm_variant_1_predictions.json",
        "vlm_variant_2_predictions.json",
        "vlm_variant_3_predictions.json",
        "vlm_variant_4_predictions.json",
        "vlm_sonnet_predictions_subset.json",
        "vlm_opus_predictions_subset.json",
        "vlm_honest_predictions.json",  # via manifest
    ]
    all_cmps: dict[str, dict] = {}
    for b in baseline_candidates:
        if b == "vlm_honest_predictions.json":
            # load honest explicitly (path-mapped via manifest)
            honest_p = CACHE_DIR / "vlm_honest_predictions.json"
            mani_p = CACHE_DIR / "vlm_honest_manifest.json"
            if not (honest_p.exists() and mani_p.exists()):
                continue
            hp = json.loads(honest_p.read_text())
            mani = json.loads(mani_p.read_text())
            honest_pk: dict[str, dict] = {}
            for sid, e in hp.items():
                m = mani.get(sid)
                if not m:
                    continue
                try:
                    rel = str(Path(m["raw_path"]).resolve().relative_to(REPO))
                except ValueError:
                    continue
                honest_pk[rel] = e
            # write a transient temp cache file so compare_to_zeroshot can pick it up
            # (simpler: just inline the diff logic)
            overlap_keys = [k for k in scored_keys if k in honest_pk and k in cache
                            and honest_pk[k].get("predicted_class") in CLASSES
                            and cache[k].get("predicted_class") in CLASSES]
            agree = 0; both_right = 0; fs_right_only = 0; zs_right_only = 0; both_wrong = 0
            for k in overlap_keys:
                truth = cache[k]["true_class"]
                zs_pred = honest_pk[k]["predicted_class"]
                fs_pred = cache[k]["predicted_class"]
                if zs_pred == fs_pred: agree += 1
                if zs_pred == truth and fs_pred == truth: both_right += 1
                elif zs_pred != truth and fs_pred == truth: fs_right_only += 1
                elif zs_pred == truth and fs_pred != truth: zs_right_only += 1
                else: both_wrong += 1
            if overlap_keys:
                cmp_b = {
                    "n_overlap": len(overlap_keys),
                    "zero_shot_source": "vlm_honest_predictions.json (via manifest)",
                    "zero_shot_acc": (both_right + zs_right_only) / len(overlap_keys),
                    "few_shot_acc": (both_right + fs_right_only) / len(overlap_keys),
                    "agreement_pct": agree / len(overlap_keys),
                    "both_right": both_right,
                    "both_wrong": both_wrong,
                    "few_shot_right_only": fs_right_only,
                    "zero_shot_right_only": zs_right_only,
                    "flips": [],
                }
                key = "honest"
            else:
                continue
        else:
            if not (CACHE_DIR / b).exists():
                continue
            cmp_b = compare_to_zeroshot(cache, scored_keys, prefer=b)
            key = b
        if cmp_b and "error" not in cmp_b and cmp_b.get("n_overlap", 0) > 0:
            all_cmps[key] = cmp_b
            print(f"\n=== vs {key} ===")
            print(f"  source={cmp_b['zero_shot_source']}")
            print(f"  n_overlap={cmp_b['n_overlap']}  zs_acc={cmp_b['zero_shot_acc']:.4f}  fs_acc={cmp_b['few_shot_acc']:.4f}  delta={cmp_b['few_shot_acc']-cmp_b['zero_shot_acc']:+.4f}")
            print(f"  agreement={cmp_b['agreement_pct']:.1%}  fs_fixes={cmp_b['few_shot_right_only']}  fs_regresses={cmp_b['zero_shot_right_only']}")

    # primary comparison used in the report: the one with largest overlap
    cmp_zs = max(all_cmps.values(), key=lambda c: c["n_overlap"]) if all_cmps else None

    # total cost
    total_cost = sum(float(e.get("cost_usd", 0.0) or 0.0) for k, e in cache.items() if k in scored_keys)
    print(f"\nTotal cost on scored subset: ${total_cost:.4f}")

    report_path = REPO / "reports" / "VLM_FEW_SHOT_RESULTS.md"
    write_markdown_report(args, ev, cmp_zs, total_cost, report_path, all_cmps=all_cmps)
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
