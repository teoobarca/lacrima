"""VLM binary re-ranker for v4's uncertain predictions.

Motivation
----------
Full 5-way VLM classification was proven DEAD in red-team evaluation
(honest Sonnet 4.6 weighted F1 = 0.3424 on all 240 scans; see
`reports/VLM_SONNET_HONEST.md`). But a 5-class decision is
information-theoretically ~10x harder than a 2-class decision. The
hypothesis tested here: on v4's UNCERTAIN predictions (small top-1 vs
top-2 margin), the VLM might still be useful if we restrict the task
to choosing between the two most-probable classes.

Protocol
--------
1. Load v4 OOF softmax predictions (240 scans, 5 classes).
2. Compute margin = p[top1] - p[top2]. Define "abstain set" as scans
   with margin < threshold (target ~20 scans; raise threshold if fewer).
3. For each abstain scan:
     - Retrieve 3 nearest anchors of class top1 and 3 nearest of class
       top2 using DINOv2-B embeddings, person-disjoint from query.
     - Compose an OBFUSCATED collage with column headers "Class A" and
       "Class B" ONLY (no actual class names on the image!) — left
       column = top1 class anchors, right column = top2 class anchors.
     - Save collage via `safe_tile_path()` (scan_XXXX.png) under
       cache/vlm_safe/binary_reranker/.
     - Ask Sonnet 4.6: "Does the query match Class A or Class B?"
     - Internal map: {A: top1_class, B: top2_class}.
4. Replace v4 argmax with VLM's choice ONLY for abstain cases.
5. Compute wF1 / mF1 / per-class F1. Bootstrap vs v4-only baseline.
6. Report VLM accuracy on abstain subset vs v4 accuracy on abstain
   subset. If VLM < v4 on abstain, abstain. If VLM > v4 on abstain,
   adopt.

Safety
------
Every prompt is gated by `assert_prompt_safe()`. Collage paths are
obfuscated (scan_XXXX.png). Column headers on the image are "Class A"
and "Class B" only — no class names are visible anywhere the VLM can
see (neither in filename nor in pixels).

Usage
-----
    .venv/bin/python scripts/vlm_binary_reranker.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from teardrop.data import CLASSES, enumerate_samples, preprocess_spm  # noqa: E402
from teardrop.safe_paths import (  # noqa: E402
    SAFE_ROOT,
    assert_prompt_safe,
    safe_tile_path,
    safe_manifest_path,
)

CACHE_DIR = REPO / "cache"
V4_OOF_PATH = CACHE_DIR / "v4_oof.npz"
EMB_PATH = CACHE_DIR / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz"
PRED_CACHE = CACHE_DIR / "vlm_binary_reranker_predictions.json"
SUBDIR = "binary_reranker"

# Neutral tile dir (class-agnostic filenames)
NEUTRAL_TILE_DIR = SAFE_ROOT / SUBDIR / "tiles"
NEUTRAL_TILE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Prompt (no class names mentioned — pure visual match task)
# ---------------------------------------------------------------------------

SYSTEM_APPEND = (
    "You are a vision-language classifier. Respond with one JSON object only "
    "and nothing else. No markdown fence, no preamble, no tool calls beyond "
    "reading the referenced image."
)

PROMPT_TEMPLATE = """You are comparing microscopy images by visual pattern.

Read the image at {img_path}. It shows:
- LEFT column, labeled "Class A": 3 reference tiles.
- RIGHT column, labeled "Class B": 3 reference tiles.
- BOTTOM (red border), labeled "QUERY": 1 query tile.

All tiles use the afmhot colormap (bright = high topography, dark = low).

Your task: decide whether the QUERY tile more closely matches the visual pattern of the Class A reference tiles or the Class B reference tiles. Look at branching density, crystal thickness, fractal texture, presence of rings/loops, roughness, and sparsity.

Respond ONLY with a single JSON object, no markdown fence, exactly this shape:

{{"choice": "A" or "B", "confidence": <float 0 to 1>, "reasoning": "<1-2 sentences citing visual features that drove the choice>"}}"""


# ---------------------------------------------------------------------------
# Tile rendering + collage composition
# ---------------------------------------------------------------------------


def _neutral_tile_path(raw_path: Path) -> Path:
    """Deterministic class-neutral tile path (sha1 of repo-relative)."""
    import hashlib
    try:
        rel = str(raw_path.resolve().relative_to(REPO))
    except ValueError:
        rel = str(raw_path)
    tid = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
    return NEUTRAL_TILE_DIR / f"tile_{tid}.png"


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


def compose_binary_collage(
    class_a_tiles: list[Path],
    class_b_tiles: list[Path],
    query_tile: Path,
    out_path: Path,
    *,
    anchor_size: int = 260,
    query_size: int = 400,
) -> Path:
    """Compose a 2-column (A/B) x 3-row anchor grid + query tile below.

    Headers: "Class A" (left), "Class B" (right) ONLY. No real class names.
    Query tile gets a red border.
    """
    assert len(class_a_tiles) == 3
    assert len(class_b_tiles) == 3

    header_h = 40
    tile_pad = 10
    query_pad = 18
    cols = 2
    rows = 3

    grid_w = cols * anchor_size + (cols + 1) * tile_pad
    grid_h = header_h + rows * anchor_size + (rows + 1) * tile_pad
    query_h = query_pad + 32 + query_size + query_pad
    total_w = max(grid_w, query_size + 2 * query_pad + 20)
    total_h = grid_h + query_h + 10

    bg = Image.new("RGB", (total_w, total_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(bg)
    header_font = _get_font(24)
    query_font = _get_font(24)

    # Center the grid horizontally
    grid_x_off = (total_w - grid_w) // 2

    # Headers
    col_labels = ["Class A", "Class B"]
    header_colors = [(180, 220, 255), (255, 220, 180)]
    for ci, label in enumerate(col_labels):
        x0 = grid_x_off + tile_pad + ci * (anchor_size + tile_pad)
        bbox = draw.textbbox((0, 0), label, font=header_font)
        tw = bbox[2] - bbox[0]
        tx = x0 + (anchor_size - tw) // 2
        draw.text((tx, 8), label, fill=header_colors[ci], font=header_font)

    # Place anchors — column-major
    all_cols = [class_a_tiles, class_b_tiles]
    for ci, tiles in enumerate(all_cols):
        for ri, tile_path in enumerate(tiles):
            x0 = grid_x_off + tile_pad + ci * (anchor_size + tile_pad)
            y0 = header_h + tile_pad + ri * (anchor_size + tile_pad)
            try:
                tim = Image.open(tile_path).convert("RGB").resize(
                    (anchor_size, anchor_size), Image.Resampling.LANCZOS
                )
            except Exception as e:
                tim = Image.new("RGB", (anchor_size, anchor_size), (50, 0, 0))
                ImageDraw.Draw(tim).text((8, 8), f"ERR {str(e)[:40]}", fill=(255, 200, 200))
            bg.paste(tim, (x0, y0))

    # QUERY tile
    qy0 = grid_h + query_pad + 32
    qx0 = (total_w - query_size) // 2
    qhdr = "QUERY"
    bbox = draw.textbbox((0, 0), qhdr, font=query_font)
    tw = bbox[2] - bbox[0]
    draw.text(((total_w - tw) // 2, grid_h + query_pad), qhdr, fill=(255, 180, 180), font=query_font)
    try:
        qim = Image.open(query_tile).convert("RGB").resize(
            (query_size, query_size), Image.Resampling.LANCZOS
        )
    except Exception as e:
        qim = Image.new("RGB", (query_size, query_size), (50, 0, 0))
        ImageDraw.Draw(qim).text((8, 8), f"ERR {str(e)[:40]}", fill=(255, 200, 200))
    # red border
    border = 4
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


def call_claude_cli(img_path: Path, model: str = "claude-sonnet-4-6",
                    timeout_s: int = 120) -> dict:
    prompt = PROMPT_TEMPLATE.format(img_path=str(img_path))
    # CRITICAL: verify no leakage. Also forbid the literal class names in
    # the prompt (belt-and-braces: our prompt is already class-neutral, but
    # assert anyway to catch future template changes).
    assert_prompt_safe(prompt, extra_forbidden=CLASSES)
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
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s, cwd=str(REPO)
    )
    latency = time.time() - t0
    if proc.returncode != 0:
        return {"error": f"cli exit {proc.returncode}",
                "stderr": proc.stderr[:500], "latency_s": latency}
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"envelope parse failed: {e}",
                "stdout": proc.stdout[:500], "latency_s": latency}
    result_text = envelope.get("result", "")
    try:
        parsed = _extract_json(result_text)
    except (ValueError, json.JSONDecodeError) as e:
        return {"error": f"inner JSON parse failed: {e}",
                "result_text": result_text, "latency_s": latency,
                "cost_usd": envelope.get("total_cost_usd", 0.0)}
    choice = str(parsed.get("choice", "")).strip().upper()
    if choice not in ("A", "B"):
        return {"error": f"invalid choice {choice!r}",
                "result_text": result_text, "latency_s": latency,
                "cost_usd": envelope.get("total_cost_usd", 0.0)}
    return {
        "choice": choice,
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "reasoning": str(parsed.get("reasoning", "")),
        "latency_s": latency,
        "cost_usd": envelope.get("total_cost_usd", 0.0),
        "duration_api_ms": envelope.get("duration_api_ms", 0),
        "result_text": result_text,
    }


# ---------------------------------------------------------------------------
# Retrieval
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


def retrieve_anchors(emb, query_idx: int, query_person: str,
                     cls_idx: int, k: int = 3) -> list[int]:
    """Top-k nearest anchors (cosine) for a given class, excluding query's own person."""
    X = emb["X"]
    y = emb["y"]
    groups = emb["groups"]
    sims = X @ X[query_idx]
    excl = (groups == query_person)
    excl[query_idx] = True
    mask = (y == cls_idx) & (~excl)
    if mask.sum() == 0:
        raise RuntimeError(f"no anchors for class {CLASSES[cls_idx]} "
                           f"excluding person {query_person}")
    idxs = np.where(mask)[0]
    order = np.argsort(-sims[idxs])
    top = idxs[order[:k]]
    if len(top) < k:
        pad = np.full(k - len(top), top[0], dtype=int)
        top = np.concatenate([top, pad])
    return top.tolist()


# ---------------------------------------------------------------------------
# Job prep / dispatch
# ---------------------------------------------------------------------------


def prepare_job(obf_key: str, q_emb_idx: int, q_path: str, q_person: str,
                q_true: str, top1_cls: str, top2_cls: str,
                emb, path_to_sample) -> dict:
    """Render tiles + compose obfuscated binary collage."""
    # Query tile
    q_sample = path_to_sample[q_path]
    q_tile = _neutral_tile_path(q_sample.raw_path)
    try:
        render_scan_tile(q_sample.raw_path, q_tile)
    except Exception as e:
        return {"key": obf_key, "error": f"query render: {e}",
                "true_class": q_true, "top1": top1_cls, "top2": top2_cls}

    def _anchor_tiles(cls_name: str) -> list[Path]:
        cls_idx = CLASSES.index(cls_name)
        anchor_idxs = retrieve_anchors(emb, q_emb_idx, q_person, cls_idx, k=3)
        tiles = []
        for emb_idx in anchor_idxs:
            anchor_abs = emb["paths"][emb_idx]
            a_sample = path_to_sample.get(anchor_abs)
            if a_sample is None:
                raise RuntimeError(f"no sample for path {anchor_abs}")
            assert a_sample.person != q_person, (
                f"LEAK: anchor person {a_sample.person} == query person {q_person}"
            )
            t = _neutral_tile_path(a_sample.raw_path)
            render_scan_tile(a_sample.raw_path, t)
            tiles.append(t)
        return tiles

    try:
        a_tiles = _anchor_tiles(top1_cls)
        b_tiles = _anchor_tiles(top2_cls)
    except Exception as e:
        return {"key": obf_key, "error": f"anchor retrieve/render: {e}",
                "true_class": q_true, "top1": top1_cls, "top2": top2_cls}

    # Numerical index for safe_tile_path: derive from obf_key ("scan_0042")
    idx = int(obf_key.split("_")[-1])
    collage_path = safe_tile_path(idx, subdir=SUBDIR)
    try:
        compose_binary_collage(a_tiles, b_tiles, q_tile, collage_path)
    except Exception as e:
        return {"key": obf_key, "error": f"compose: {e}",
                "true_class": q_true, "top1": top1_cls, "top2": top2_cls}

    return {
        "key": obf_key,
        "collage_path": str(collage_path),
        "true_class": q_true,
        "top1_class": top1_cls,
        "top2_class": top2_cls,
        "person": q_person,
        "query_path": q_path,
    }


def dispatch_vlm(job: dict, model: str) -> dict:
    if "error" in job:
        return {job["key"]: {
            "error": job["error"],
            "true_class": job.get("true_class", ""),
            "top1_class": job.get("top1", job.get("top1_class", "")),
            "top2_class": job.get("top2", job.get("top2_class", "")),
        }}
    res = call_claude_cli(Path(job["collage_path"]), model=model)
    res["true_class"] = job["true_class"]
    res["top1_class"] = job["top1_class"]
    res["top2_class"] = job["top2_class"]
    res["person"] = job["person"]
    res["query_path"] = job["query_path"]
    res["obf_key"] = job["key"]
    # Decode choice -> class
    if "choice" in res:
        chosen_class = job["top1_class"] if res["choice"] == "A" else job["top2_class"]
        res["predicted_class"] = chosen_class
    return {job["key"]: res}


# ---------------------------------------------------------------------------
# Metrics + bootstrap
# ---------------------------------------------------------------------------


def compute_metrics(y_true, y_pred, label_order=CLASSES):
    w_f1 = f1_score(y_true, y_pred, labels=label_order, average="weighted", zero_division=0)
    m_f1 = f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    report = classification_report(y_true, y_pred, labels=label_order,
                                   zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=label_order).tolist()
    return {
        "n": len(y_true),
        "accuracy": acc,
        "f1_weighted": w_f1,
        "f1_macro": m_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": label_order,
    }


def bootstrap_paired(y_true, y_pred_a, y_pred_b, *, n_boot: int = 1000,
                     seed: int = 42) -> dict:
    """Paired bootstrap of wF1(pred_a) - wF1(pred_b)."""
    yt = np.array(y_true)
    pa = np.array(y_pred_a)
    pb = np.array(y_pred_b)
    n = len(yt)
    rng = np.random.default_rng(seed)
    boot_a = np.empty(n_boot)
    boot_b = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_a[i] = f1_score(yt[idx], pa[idx], labels=CLASSES,
                             average="weighted", zero_division=0)
        boot_b[i] = f1_score(yt[idx], pb[idx], labels=CLASSES,
                             average="weighted", zero_division=0)
    deltas = boot_a - boot_b
    return {
        "n_boot": n_boot,
        "a_mean": float(boot_a.mean()),
        "a_ci_lo": float(np.quantile(boot_a, 0.025)),
        "a_ci_hi": float(np.quantile(boot_a, 0.975)),
        "b_mean": float(boot_b.mean()),
        "b_ci_lo": float(np.quantile(boot_b, 0.025)),
        "b_ci_hi": float(np.quantile(boot_b, 0.975)),
        "delta_mean": float(deltas.mean()),
        "delta_ci_lo": float(np.quantile(deltas, 0.025)),
        "delta_ci_hi": float(np.quantile(deltas, 0.975)),
        "p_a_gt_b": float((deltas > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_cache() -> dict:
    if PRED_CACHE.exists():
        return json.loads(PRED_CACHE.read_text())
    return {}


def save_cache(cache: dict) -> None:
    tmp = PRED_CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    tmp.replace(PRED_CACHE)


def select_abstain_set(proba: np.ndarray, *, target_min: int = 20,
                       start_threshold: float = 0.20,
                       max_threshold: float = 0.45) -> tuple[np.ndarray, float]:
    """Find a margin threshold giving at least `target_min` abstain scans.

    Start at `start_threshold`; if too few, raise to 0.25, 0.30, ... up to
    `max_threshold`. Returns (mask, used_threshold).
    """
    sorted_proba = np.sort(proba, axis=1)[:, ::-1]
    margins = sorted_proba[:, 0] - sorted_proba[:, 1]
    thr = start_threshold
    while thr <= max_threshold + 1e-9:
        mask = margins < thr
        if int(mask.sum()) >= target_min:
            return mask, thr
        thr += 0.05
    # fallback: use max_threshold regardless
    mask = margins < max_threshold
    return mask, max_threshold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--threshold", type=float, default=0.20,
                    help="margin threshold for abstain set")
    ap.add_argument("--target-min", type=int, default=20,
                    help="min abstain count; raise threshold until met")
    ap.add_argument("--v4-baseline-wf1", type=float, default=0.6887)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    # --- Load v4 OOF ---
    z = np.load(V4_OOF_PATH, allow_pickle=True)
    proba = z["proba"].astype(np.float64)
    y = z["y"].astype(int)
    persons = np.asarray(z["persons"], dtype=str)
    paths = np.asarray(z["scan_paths"], dtype=str)
    n_total = len(y)
    y_labels = np.array([CLASSES[i] for i in y])

    y_pred_v4 = proba.argmax(axis=1)
    y_pred_v4_labels = np.array([CLASSES[i] for i in y_pred_v4])
    v4_wf1 = f1_score(y_labels, y_pred_v4_labels, labels=CLASSES,
                      average="weighted", zero_division=0)
    v4_mf1 = f1_score(y_labels, y_pred_v4_labels, labels=CLASSES,
                      average="macro", zero_division=0)
    print(f"v4 baseline on {n_total} scans: wF1={v4_wf1:.4f}, mF1={v4_mf1:.4f}")

    # --- Abstain set selection ---
    abstain_mask, used_thr = select_abstain_set(
        proba, target_min=args.target_min, start_threshold=args.threshold
    )
    n_abstain = int(abstain_mask.sum())
    n_confident = n_total - n_abstain
    print(f"Abstain set (margin<{used_thr:.2f}): {n_abstain} scans  "
          f"(confident: {n_confident})")
    sorted_proba = np.sort(proba, axis=1)[:, ::-1]
    margins = sorted_proba[:, 0] - sorted_proba[:, 1]

    # v4 accuracy on abstain subset (baseline to beat)
    v4_acc_abst = float((y_pred_v4[abstain_mask] == y[abstain_mask]).mean())
    v4_acc_conf = float((y_pred_v4[~abstain_mask] == y[~abstain_mask]).mean())
    print(f"v4 accuracy on abstain: {v4_acc_abst:.3f}   on confident: {v4_acc_conf:.3f}")

    # --- Build abstain jobs ---
    abstain_idxs = np.where(abstain_mask)[0]

    # Build obfuscated key per abstain scan (keep deterministic index
    # within the abstain set)
    abstain_entries = []
    for rank, global_idx in enumerate(abstain_idxs):
        top2 = np.argsort(-proba[global_idx])[:2]
        top1_cls = CLASSES[int(top2[0])]
        top2_cls = CLASSES[int(top2[1])]
        obf_key = f"scan_{rank:04d}"
        abstain_entries.append({
            "obf_key": obf_key,
            "global_idx": int(global_idx),
            "path": str(paths[global_idx]),
            "person": str(persons[global_idx]),
            "true_class": CLASSES[int(y[global_idx])],
            "top1_class": top1_cls,
            "top2_class": top2_cls,
            "top1_prob": float(proba[global_idx, int(top2[0])]),
            "top2_prob": float(proba[global_idx, int(top2[1])]),
            "margin": float(margins[global_idx]),
        })

    # Persist manifest (never shown to VLM)
    manifest_path = safe_manifest_path(SUBDIR)
    manifest_path.write_text(json.dumps(
        {e["obf_key"]: {k: v for k, v in e.items() if k != "obf_key"}
         for e in abstain_entries}, indent=2
    ))
    print(f"Manifest written to {manifest_path.relative_to(REPO)}")

    cache = load_cache()

    if not args.eval_only and abstain_entries:
        # --- Prepare collages ---
        all_samples = enumerate_samples(REPO / "TRAIN_SET")
        path_to_sample = {str(s.raw_path): s for s in all_samples}
        emb = load_embeddings()
        abs_to_emb_idx = {p: i for i, p in enumerate(emb["paths"])}

        # Self-test: render one collage, inspect the prompt.
        probe = abstain_entries[0]
        probe_collage = safe_tile_path(int(probe["obf_key"].split("_")[-1]),
                                        subdir=SUBDIR)
        probe_prompt = PROMPT_TEMPLATE.format(img_path=str(probe_collage))
        try:
            assert_prompt_safe(probe_prompt, extra_forbidden=CLASSES)
            print(f"[sanity] prompt passes assert_prompt_safe()")
            print(f"[sanity] probe path: {probe_collage}")
        except Exception as e:
            print(f"FATAL: probe prompt unsafe: {e}")
            return 2

        print(f"\n[1/2] preparing {len(abstain_entries)} collages ...")
        jobs = []
        for entry in abstain_entries:
            if entry["obf_key"] in cache and cache[entry["obf_key"]].get("choice") in ("A", "B"):
                continue
            q_emb_idx = abs_to_emb_idx.get(entry["path"])
            if q_emb_idx is None:
                cache[entry["obf_key"]] = {
                    "error": "no embedding for query",
                    "true_class": entry["true_class"],
                    "top1_class": entry["top1_class"],
                    "top2_class": entry["top2_class"],
                }
                continue
            job = prepare_job(
                entry["obf_key"], q_emb_idx, entry["path"],
                entry["person"], entry["true_class"],
                entry["top1_class"], entry["top2_class"],
                emb, path_to_sample,
            )
            jobs.append(job)
        print(f"  {len(jobs)} new jobs; {len(abstain_entries) - len(jobs)} cached/errored")

        if jobs:
            print(f"[2/2] dispatching {len(jobs)} VLM calls "
                  f"(model={args.model}, workers={args.workers}) ...")
            t0 = time.time()
            done = 0
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(dispatch_vlm, job, args.model): job for job in jobs}
                for fut in as_completed(futs):
                    try:
                        result = fut.result()
                    except Exception as e:
                        job = futs[fut]
                        result = {job["key"]: {
                            "error": f"future: {e}",
                            "true_class": job.get("true_class", ""),
                            "top1_class": job.get("top1_class", ""),
                            "top2_class": job.get("top2_class", ""),
                        }}
                    cache.update(result)
                    done += 1
                    for k, v in result.items():
                        if v.get("choice") in ("A", "B"):
                            ok = "OK" if v.get("predicted_class") == v.get("true_class") else "--"
                            print(f"[{done}/{len(jobs)}] {ok} {k}  true={v['true_class']}  "
                                  f"choice={v['choice']}({v.get('predicted_class')})  "
                                  f"A={v['top1_class']} B={v['top2_class']}  "
                                  f"conf={v.get('confidence',0):.2f}  ${v.get('cost_usd',0):.4f}")
                        else:
                            print(f"[{done}/{len(jobs)}] ERR {k}: {v.get('error', v)}")
                    if done % 5 == 0:
                        save_cache(cache)
            save_cache(cache)
            print(f"All VLM calls done in {time.time()-t0:.1f}s")
        save_cache(cache)

    # --- Evaluate final predictions ---
    # Replace v4 prediction with VLM pick for abstain cases
    y_pred_fused = y_pred_v4_labels.copy()
    vlm_pred_on_abstain = []
    vlm_true_on_abstain = []
    vlm_errors = 0
    for entry in abstain_entries:
        gi = entry["global_idx"]
        ce = cache.get(entry["obf_key"])
        if not ce or ce.get("choice") not in ("A", "B"):
            vlm_errors += 1
            # Keep v4 prediction as fallback
            continue
        y_pred_fused[gi] = ce["predicted_class"]
        vlm_pred_on_abstain.append(ce["predicted_class"])
        vlm_true_on_abstain.append(entry["true_class"])
    vlm_acc_abst = (float(sum(a == b for a, b in zip(vlm_true_on_abstain, vlm_pred_on_abstain)))
                    / max(1, len(vlm_pred_on_abstain)))

    fused_metrics = compute_metrics(y_labels.tolist(), y_pred_fused.tolist())
    v4_metrics = compute_metrics(y_labels.tolist(), y_pred_v4_labels.tolist())

    # --- Bootstrap fused vs v4 ---
    boot = bootstrap_paired(
        y_labels.tolist(), y_pred_fused.tolist(), y_pred_v4_labels.tolist(),
        n_boot=1000, seed=42,
    )

    print("\n=== v4-only (baseline) ===")
    print(f"wF1={v4_metrics['f1_weighted']:.4f}  mF1={v4_metrics['f1_macro']:.4f}  acc={v4_metrics['accuracy']:.4f}")
    print("\n=== v4 + VLM-binary-rerank on abstain ===")
    print(f"wF1={fused_metrics['f1_weighted']:.4f}  mF1={fused_metrics['f1_macro']:.4f}  acc={fused_metrics['accuracy']:.4f}")
    print(f"\n=== VLM vs v4 on abstain subset ({len(vlm_pred_on_abstain)} evaluated, {vlm_errors} errored/fallback) ===")
    print(f"v4 acc on abstain:  {v4_acc_abst:.3f}")
    print(f"VLM acc on abstain: {vlm_acc_abst:.3f}")
    verdict_abst = ("VLM HURTS — retract" if vlm_acc_abst < v4_acc_abst
                    else ("VLM HELPS" if vlm_acc_abst > v4_acc_abst else "tie"))
    print(f"Verdict on abstain: {verdict_abst}")

    print("\n=== Paired bootstrap 1000x (fused vs v4-only) ===")
    print(f"Fused wF1 95% CI: [{boot['a_ci_lo']:.4f}, {boot['a_ci_hi']:.4f}]")
    print(f"v4    wF1 95% CI: [{boot['b_ci_lo']:.4f}, {boot['b_ci_hi']:.4f}]")
    print(f"Delta mean: {boot['delta_mean']:+.4f}  95% CI [{boot['delta_ci_lo']:+.4f}, {boot['delta_ci_hi']:+.4f}]")
    print(f"P(Delta > 0) = {boot['p_a_gt_b']:.3f}")

    # --- Per-class comparison ---
    print("\nPer-class F1 (v4 -> fused):")
    for cls in CLASSES:
        v4_f1 = v4_metrics["classification_report"].get(cls, {}).get("f1-score", 0.0)
        f_f1 = fused_metrics["classification_report"].get(cls, {}).get("f1-score", 0.0)
        sup = int(v4_metrics["classification_report"].get(cls, {}).get("support", 0))
        print(f"  {cls:22s}  v4={v4_f1:.3f}  fused={f_f1:.3f}  Δ={f_f1-v4_f1:+.3f}  support={sup}")

    # --- Cost ---
    total_cost = sum(float(cache[e["obf_key"]].get("cost_usd", 0.0) or 0.0)
                     for e in abstain_entries if e["obf_key"] in cache)
    print(f"\nTotal VLM cost: ${total_cost:.4f}")

    # --- Dump per-scan final prediction json ---
    out = {
        "config": {
            "threshold_used": used_thr,
            "n_abstain": n_abstain,
            "n_confident": n_confident,
            "n_total": n_total,
            "model": args.model,
            "v4_baseline_wf1": v4_wf1,
        },
        "abstain_manifest": {
            e["obf_key"]: {
                "path": e["path"],
                "person": e["person"],
                "true_class": e["true_class"],
                "top1_class": e["top1_class"],
                "top2_class": e["top2_class"],
                "margin": e["margin"],
            } for e in abstain_entries
        },
        "per_abstain_scan": [
            {
                "obf_key": e["obf_key"],
                "global_idx": e["global_idx"],
                "true_class": e["true_class"],
                "top1_class": e["top1_class"],
                "top2_class": e["top2_class"],
                "margin": e["margin"],
                "v4_pred": CLASSES[int(y_pred_v4[e["global_idx"]])],
                "vlm_choice": cache.get(e["obf_key"], {}).get("choice"),
                "vlm_pred": cache.get(e["obf_key"], {}).get("predicted_class"),
                "vlm_confidence": cache.get(e["obf_key"], {}).get("confidence"),
                "vlm_reasoning": cache.get(e["obf_key"], {}).get("reasoning"),
                "vlm_error": cache.get(e["obf_key"], {}).get("error"),
            } for e in abstain_entries
        ],
        "metrics": {
            "v4_only": {k: v for k, v in v4_metrics.items() if k != "classification_report"},
            "v4_plus_vlm_rerank": {k: v for k, v in fused_metrics.items() if k != "classification_report"},
            "v4_per_class_f1": {
                cls: v4_metrics["classification_report"].get(cls, {}).get("f1-score", 0.0)
                for cls in CLASSES
            },
            "fused_per_class_f1": {
                cls: fused_metrics["classification_report"].get(cls, {}).get("f1-score", 0.0)
                for cls in CLASSES
            },
            "v4_acc_on_abstain": v4_acc_abst,
            "vlm_acc_on_abstain": vlm_acc_abst,
            "n_vlm_evaluated": len(vlm_pred_on_abstain),
            "n_vlm_errors": vlm_errors,
        },
        "bootstrap": boot,
        "cost_usd": total_cost,
    }
    out_path = CACHE_DIR / "vlm_binary_reranker_predictions.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Predictions dumped to {out_path.relative_to(REPO)}")

    # --- Write report ---
    write_report(
        REPO / "reports" / "VLM_BINARY_RERANKER.md",
        args, used_thr, n_abstain, n_confident, n_total,
        v4_metrics, fused_metrics, v4_acc_abst, vlm_acc_abst,
        len(vlm_pred_on_abstain), vlm_errors, boot, total_cost,
        abstain_entries, cache, y_pred_v4, y,
    )
    return 0


def write_report(out_path: Path, args, used_thr: float,
                 n_abstain: int, n_confident: int, n_total: int,
                 v4_metrics: dict, fused_metrics: dict,
                 v4_acc_abst: float, vlm_acc_abst: float,
                 n_vlm_eval: int, vlm_errors: int, boot: dict,
                 total_cost: float, abstain_entries: list, cache: dict,
                 y_pred_v4: np.ndarray, y: np.ndarray):
    import datetime as dt
    lines: list[str] = []
    lines.append("# VLM Binary Re-ranker on v4 uncertain predictions\n")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n")

    wf1_fused = fused_metrics["f1_weighted"]
    wf1_v4 = v4_metrics["f1_weighted"]
    delta = wf1_fused - wf1_v4
    p_up = boot["p_a_gt_b"]

    if vlm_acc_abst < v4_acc_abst:
        verdict = (f"**RETRACT — VLM hurts on abstain.** "
                   f"VLM acc={vlm_acc_abst:.3f} < v4 acc={v4_acc_abst:.3f} on the same subset.")
    elif wf1_fused > wf1_v4 and p_up > 0.80:
        verdict = (f"**PROMISING.** Fused wF1 ({wf1_fused:.4f}) > v4 ({wf1_v4:.4f}) "
                   f"with P(Δ>0)={p_up:.2f}. Consider red-teaming before promotion.")
    elif wf1_fused > wf1_v4:
        verdict = (f"**MARGINAL.** Fused wF1 ({wf1_fused:.4f}) > v4 ({wf1_v4:.4f}) "
                   f"but P(Δ>0)={p_up:.2f} < 0.80 — not statistically convincing.")
    else:
        verdict = (f"**NO BENEFIT.** Fused wF1 ({wf1_fused:.4f}) ≤ v4 ({wf1_v4:.4f}). "
                   f"VLM binary rerank does not improve the champion.")

    lines.append("## TL;DR\n")
    lines.append(f"- Margin threshold: **{used_thr:.2f}**  (abstain set = {n_abstain}/{n_total} scans, confident = {n_confident})")
    lines.append(f"- v4 accuracy on abstain: **{v4_acc_abst:.3f}**")
    lines.append(f"- VLM accuracy on abstain: **{vlm_acc_abst:.3f}**  ({n_vlm_eval} evaluated, {vlm_errors} errored)")
    lines.append(f"- Overall wF1: v4-only = {wf1_v4:.4f}  →  v4+VLM-rerank = **{wf1_fused:.4f}**  (Δ = {delta:+.4f})")
    lines.append(f"- Overall mF1: v4-only = {v4_metrics['f1_macro']:.4f}  →  v4+VLM-rerank = **{fused_metrics['f1_macro']:.4f}**")
    lines.append(f"- Paired bootstrap P(Δ > 0) = **{p_up:.3f}**  (95% CI [{boot['delta_ci_lo']:+.4f}, {boot['delta_ci_hi']:+.4f}])")
    lines.append(f"- Cost: ${total_cost:.4f}  ({n_vlm_eval} calls, model={args.model})")
    lines.append(f"- **Verdict:** {verdict}\n")

    lines.append("## Protocol\n")
    lines.append("1. Load v4 multiscale OOF softmax (240 scans × 5 classes) from `cache/v4_oof.npz`.")
    lines.append(f"2. Abstain set: scans where top-1 prob − top-2 prob < **{used_thr:.2f}** (auto-raised from 0.20 to reach target_min={args.target_min}).")
    lines.append("3. For each abstain scan, retrieve 3 nearest DINOv2-B anchors for top-1 class and 3 nearest for top-2 class (cosine sim; person-disjoint from query).")
    lines.append("4. Compose an obfuscated collage: 2 columns (headers only \"Class A\" / \"Class B\") × 3 rows + query at bottom with red border. **No actual class names appear anywhere in the collage or the filename.**")
    lines.append("5. Save via `teardrop.safe_paths.safe_tile_path(idx, subdir='binary_reranker')`. Call `assert_prompt_safe(prompt, extra_forbidden=CLASSES)` immediately before every `claude -p`.")
    lines.append("6. VLM returns `{choice: 'A' | 'B', confidence, reasoning}`. Internal map: A = top-1 class, B = top-2 class.")
    lines.append("7. Final fused prediction = VLM's choice on abstain scans, v4 argmax on confident scans.\n")

    lines.append("## Abstain subset analysis\n")
    lines.append("| Scan idx | True | v4 top-1 | v4 top-2 | margin | VLM choice | VLM pred | VLM correct? | v4 correct? |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for e in abstain_entries:
        ce = cache.get(e["obf_key"], {})
        vlm_choice = ce.get("choice", "ERR")
        vlm_pred = ce.get("predicted_class", "-")
        vlm_ok = "✓" if vlm_pred == e["true_class"] else "✗"
        v4_pred = CLASSES[int(y_pred_v4[e["global_idx"]])]
        v4_ok = "✓" if v4_pred == e["true_class"] else "✗"
        lines.append(f"| {e['global_idx']} | {e['true_class']} | {e['top1_class']} | {e['top2_class']} | "
                     f"{e['margin']:.3f} | {vlm_choice} | {vlm_pred} | {vlm_ok} | {v4_ok} |")
    lines.append("")

    lines.append("## Per-class F1 (v4 → fused)\n")
    lines.append("| Class | v4 F1 | fused F1 | Δ | Support |")
    lines.append("|---|---|---|---|---|")
    for cls in CLASSES:
        v4_f1 = v4_metrics["classification_report"].get(cls, {}).get("f1-score", 0.0)
        f_f1 = fused_metrics["classification_report"].get(cls, {}).get("f1-score", 0.0)
        sup = int(v4_metrics["classification_report"].get(cls, {}).get("support", 0))
        lines.append(f"| {cls} | {v4_f1:.3f} | {f_f1:.3f} | {f_f1 - v4_f1:+.3f} | {sup} |")
    lines.append("")

    lines.append("## Confusion matrices\n")
    lines.append("### v4-only\n")
    lines.append("| true\\pred | " + " | ".join(c[:10] for c in CLASSES) + " |")
    lines.append("|---|" + "|".join(["---"] * len(CLASSES)) + "|")
    for lab, row in zip(CLASSES, v4_metrics["confusion_matrix"]):
        lines.append(f"| {lab[:10]} | " + " | ".join(str(v) for v in row) + " |")
    lines.append("\n### v4 + VLM-rerank\n")
    lines.append("| true\\pred | " + " | ".join(c[:10] for c in CLASSES) + " |")
    lines.append("|---|" + "|".join(["---"] * len(CLASSES)) + "|")
    for lab, row in zip(CLASSES, fused_metrics["confusion_matrix"]):
        lines.append(f"| {lab[:10]} | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")

    lines.append("## Paired bootstrap 1000x (fused − v4)\n")
    lines.append(f"- Fused wF1 bootstrap mean: {boot['a_mean']:.4f}  (95% CI [{boot['a_ci_lo']:.4f}, {boot['a_ci_hi']:.4f}])")
    lines.append(f"- v4    wF1 bootstrap mean: {boot['b_mean']:.4f}  (95% CI [{boot['b_ci_lo']:.4f}, {boot['b_ci_hi']:.4f}])")
    lines.append(f"- Delta (fused − v4) mean: **{boot['delta_mean']:+.4f}**  (95% CI [{boot['delta_ci_lo']:+.4f}, {boot['delta_ci_hi']:+.4f}])")
    lines.append(f"- **P(Δ > 0) = {boot['p_a_gt_b']:.3f}**\n")

    lines.append("## Safety verification\n")
    lines.append("- Collage filename pattern: `cache/vlm_safe/binary_reranker/scan_XXXX.png` (no class, no person, no raw name).")
    lines.append("- Column headers rendered in the collage are strictly `\"Class A\"` and `\"Class B\"` — actual class names never appear in pixels.")
    lines.append("- `assert_prompt_safe(prompt, extra_forbidden=CLASSES)` called before every `claude -p`. This catches both path-context leaks AND any literal class-name string.")
    lines.append("- Manifest (mapping `scan_XXXX` → true_class + top1/top2 + person) persisted at `cache/vlm_safe/binary_reranker/manifest.json` — never passed to the VLM.\n")

    lines.append("## Decision\n")
    if vlm_acc_abst < v4_acc_abst:
        lines.append("- **VLM DOES NOT HELP even on the narrow binary task.** On the abstain subset — where v4 is already unsure — the VLM picks the *wrong* class between top-1 and top-2 more often than v4 does.")
        lines.append("- This closes the 'maybe VLM still helps on uncertain cases' hypothesis. Full-5-way VLM is dead (F1 0.34) AND binary-VLM is dead on the narrow subset.")
        lines.append("- Recommendation: **do not ensemble VLM into the final pipeline**. v4 multiscale remains the champion.")
    elif wf1_fused > wf1_v4 and p_up > 0.80:
        lines.append("- Fused pipeline shows a promising lift. Next steps: (a) red-team by re-running with a permuted abstain set; (b) test on held-out scans if available; (c) inspect per-class balance.")
    else:
        lines.append("- No statistically convincing benefit. Keep v4 as champion.")

    lines.append("")
    lines.append("## Reproducibility\n")
    lines.append("- Script: `scripts/vlm_binary_reranker.py`")
    lines.append("- Predictions JSON: `cache/vlm_binary_reranker_predictions.json`")
    lines.append("- Raw VLM cache: `cache/vlm_binary_reranker_predictions.json` (keyed by obf_key)")
    lines.append("- Collages: `cache/vlm_safe/binary_reranker/scan_XXXX.png`")
    lines.append("- Manifest: `cache/vlm_safe/binary_reranker/manifest.json`")
    lines.append(f"- Model: `{args.model}`   Threshold used: {used_thr:.2f}")

    out_path.write_text("\n".join(lines))
    print(f"Report written to {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    sys.exit(main())
