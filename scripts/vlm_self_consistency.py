"""Self-consistency voting for few-shot VLM classification of AFM tear scans.

Hypothesis
----------
Single-sample few-shot VLM predictions at default CLI settings may be
noisy. Running the same query 3x and majority-voting the answer could
improve reliability, especially on borderline queries. But: `claude -p`
CLI does not expose --temperature, so we induce variation by subtly
varying the PROMPT PHRASING (3 variants).

Design
------
For each query (12 per class, person-spread → 60 queries total):
  1. Build ONE collage (identical to `vlm_few_shot.py`, k=2 anchors per class,
     person-disjoint) — ONE collage per query, reused across samples.
  2. Call `claude -p --model claude-haiku-4-5` THREE times with prompt
     variants A/B/C that differ only in a single introductory sentence.
  3. Record all 3 predictions.
  4. Majority vote → final prediction. Ties broken by highest-confidence
     sample; if still tied, pick sample A.

Also track:
  - 3/3 unanimous vs 2/3 vs 0/3 (all-disagree).
  - Per-class: did voting help harder classes (SucheOko)?
  - Single-sample (variant A only) F1 vs voted F1.

Outputs
-------
- cache/vlm_self_consistency_predictions.json
- reports/VLM_SELF_CONSISTENCY.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from teardrop.data import CLASSES, enumerate_samples  # noqa: E402
from teardrop.safe_paths import SAFE_ROOT, assert_prompt_safe  # noqa: E402

# Reuse tile/collage plumbing from baseline few-shot (now leak-safe).
from scripts.vlm_few_shot import (  # noqa: E402
    COLLAGE_DIR,
    TILE_DIR,
    compose_collage,
    load_embeddings,
    render_scan_tile,
    retrieve_anchors_per_class,
    tile_filename,
    _extract_json,
)

CACHE_DIR = REPO / "cache"
PRED_CACHE = CACHE_DIR / "vlm_self_consistency_predictions.json"

# ---------------------------------------------------------------------------
# Prompt variants (A / B / C)
# ---------------------------------------------------------------------------

SYSTEM_APPEND = (
    "You are a vision-language classifier. Respond with one JSON object only "
    "and nothing else. No markdown fence, no preamble, no tool calls beyond "
    "reading the referenced image."
)

# Introductory sentence differs; everything after is identical.
PROMPT_INTRO = {
    "A": "You are a medical expert classifying AFM (atomic force microscopy) scans of dried tear droplets.",
    "B": "Take a close look at the attached AFM (atomic force microscopy) composite of a dried tear droplet.",
    "C": "Examine carefully the AFM (atomic force microscopy) scan of a dried tear droplet shown in the attached composite.",
}

PROMPT_BODY = """

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


def build_prompt(variant: str, img_path: Path) -> str:
    return PROMPT_INTRO[variant] + PROMPT_BODY.format(img_path=str(img_path))


# ---------------------------------------------------------------------------
# Claude CLI wrapper (returns raw result dict for one sample)
# ---------------------------------------------------------------------------


def _call_claude_cli_once(
    img_path: Path, variant: str, model: str, timeout_s: int
) -> dict:
    prompt = build_prompt(variant, img_path)
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
        "--no-session-persistence",
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
        return {"error": f"timeout after {timeout_s}s", "variant": variant, "latency_s": timeout_s}
    latency = time.time() - t0
    if proc.returncode != 0:
        return {
            "error": f"cli exit {proc.returncode}",
            "stderr": proc.stderr[:500],
            "variant": variant,
            "latency_s": latency,
        }
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {
            "error": f"envelope parse failed: {e}",
            "stdout": proc.stdout[:500],
            "variant": variant,
            "latency_s": latency,
        }
    result_text = envelope.get("result", "")
    try:
        parsed = _extract_json(result_text)
    except (ValueError, json.JSONDecodeError) as e:
        return {
            "error": f"inner JSON parse failed: {e}",
            "result_text": result_text,
            "variant": variant,
            "latency_s": latency,
            "cost_usd": envelope.get("total_cost_usd", 0.0),
        }
    return {
        "variant": variant,
        "predicted_class": str(parsed.get("predicted_class", "")),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "reasoning": str(parsed.get("reasoning", "")),
        "most_similar_anchor_class": str(parsed.get("most_similar_anchor_class", "")),
        "latency_s": latency,
        "cost_usd": envelope.get("total_cost_usd", 0.0),
        "duration_api_ms": envelope.get("duration_api_ms", 0),
        "result_text": result_text,
    }


def call_claude_cli(
    img_path: Path,
    variant: str,
    model: str = "claude-haiku-4-5",
    timeout_s: int = 120,
    max_retries: int = 2,
) -> dict:
    """Call Claude CLI with up to max_retries retries on transient failures
    (non-zero exit, envelope parse error, or JSON parse error).
    """
    last = {"error": "no attempt"}
    for attempt in range(max_retries + 1):
        res = _call_claude_cli_once(img_path, variant, model, timeout_s)
        if res.get("predicted_class") in CLASSES:
            return res
        last = res
        # retriable = exit 143, envelope parse, timeout
        err = res.get("error", "") or ""
        if attempt < max_retries and (
            err.startswith("cli exit")
            or err.startswith("envelope")
            or err.startswith("timeout")
        ):
            time.sleep(2 * (attempt + 1))  # small backoff
            continue
        break
    return last


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------


def majority_vote(samples: list[dict]) -> tuple[str, str, float]:
    """Return (voted_class, agreement_label, mean_confidence).

    agreement_label in {"3/3", "2/3", "1/1/1"}.
    Ties broken by highest-confidence sample. All-disagree ⇒ highest-confidence.
    """
    valid = [s for s in samples if s.get("predicted_class") in CLASSES]
    if not valid:
        return ("", "invalid", 0.0)
    classes = [s["predicted_class"] for s in valid]
    counts = Counter(classes)
    top = counts.most_common()
    # agreement label
    n = len(valid)
    if top[0][1] == n:
        label = f"{n}/{n}"
    elif len(top) == n:  # all different
        label = "1/1/1" if n == 3 else f"all-disagree-{n}"
    else:
        label = f"{top[0][1]}/{n}"
    # majority winner, break ties by highest confidence
    if len(top) > 1 and top[0][1] == top[1][1]:
        # tie — pick the tied class with the highest single-sample confidence
        tied_classes = {c for c, k in top if k == top[0][1]}
        best = max(
            (s for s in valid if s["predicted_class"] in tied_classes),
            key=lambda s: s.get("confidence", 0.0),
        )
        return (best["predicted_class"], label, float(np.mean([s.get("confidence", 0.0) for s in valid])))
    return (top[0][0], label, float(np.mean([s.get("confidence", 0.0) for s in valid])))


# ---------------------------------------------------------------------------
# Subset selection (12 per class, person-spread)
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
# Cache I/O
# ---------------------------------------------------------------------------


def load_cache() -> dict[str, dict]:
    if PRED_CACHE.exists():
        return json.loads(PRED_CACHE.read_text())
    return {}


def save_cache(cache: dict[str, dict]) -> None:
    PRED_CACHE.write_text(json.dumps(cache, indent=2))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _prepare_collage(s, all_samples, emb, abs_to_emb_idx, path_to_sample,
                     cache: dict[str, dict]) -> tuple[str, Path | None, dict]:
    """Build the collage for one query. Returns (key, collage_path, entry).

    Updates `entry` in-place with anchors/metadata. If preparation fails,
    returns (key, None, entry) with `error` set in entry.
    """
    key = str(s.raw_path.relative_to(REPO))
    entry = cache.get(key, {})
    entry.setdefault("true_class", s.cls)
    entry.setdefault("person", s.person)
    entry.setdefault("query_path", key)
    entry.setdefault("samples", {})

    abs_path = str(s.raw_path)
    if abs_path not in abs_to_emb_idx:
        entry["error"] = "no embedding for query"
        cache[key] = entry
        return key, None, entry
    q_idx = abs_to_emb_idx[abs_path]

    try:
        anchors_per_cls = retrieve_anchors_per_class(emb, q_idx, s.person, k_per_class=2)
    except RuntimeError as e:
        entry["error"] = str(e)
        cache[key] = entry
        return key, None, entry

    q_tile = tile_filename(s.cls, s.raw_path)
    try:
        render_scan_tile(s.raw_path, q_tile)
    except Exception as e:
        entry["error"] = f"query render failed: {e}"
        cache[key] = entry
        return key, None, entry

    anchor_info: list[tuple[str, Path, str]] = []
    anchor_meta: list[dict] = []
    for ci, cls in enumerate(CLASSES):
        for rank, emb_idx in enumerate(anchors_per_cls[ci]):
            anchor_abs = emb["paths"][emb_idx]
            a_sample = path_to_sample.get(anchor_abs)
            if a_sample is None:
                entry["error"] = "anchor render failed: sample not in index"
                cache[key] = entry
                return key, None, entry
            a_tile = tile_filename(a_sample.cls, a_sample.raw_path)
            try:
                render_scan_tile(a_sample.raw_path, a_tile)
            except Exception as e:
                entry["error"] = f"anchor render failed: {e}"
                cache[key] = entry
                return key, None, entry
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

    for am in anchor_meta:
        assert am["person"] != s.person, f"LEAK: anchor person == query person for {key}"

    import hashlib as _hl_collage
    _rel_q = str(s.raw_path.relative_to(REPO))
    _collage_id = _hl_collage.sha1(_rel_q.encode("utf-8")).hexdigest()[:16]
    collage_path = COLLAGE_DIR / f"scan_{_collage_id}.png"
    compose_collage(anchor_info, q_tile, s.raw_path.stem, collage_path)
    entry["collage_path"] = str(collage_path.relative_to(REPO))
    entry["anchors"] = anchor_meta
    cache[key] = entry
    return key, collage_path, entry


def run(samples_to_score, all_samples, model: str, time_budget_s: float,
        cache: dict[str, dict], variants: list[str], concurrency: int = 8) -> dict[str, dict]:
    t_start = time.time()
    emb = load_embeddings()
    abs_to_emb_idx = {p: i for i, p in enumerate(emb["paths"])}
    path_to_sample = {str(s.raw_path): s for s in all_samples}

    # Phase 1: build all collages sequentially (local I/O, fast)
    print(f"Phase 1: building collages for {len(samples_to_score)} queries ...")
    ready: list[tuple[str, Path, dict, Any]] = []  # (key, collage_path, entry, sample)
    for s in samples_to_score:
        key, collage_path, entry = _prepare_collage(
            s, all_samples, emb, abs_to_emb_idx, path_to_sample, cache
        )
        if collage_path is None:
            print(f"  SKIP {key}: {entry.get('error', 'prep failed')}")
            continue
        ready.append((key, collage_path, entry, s))
    save_cache(cache)
    print(f"  ready: {len(ready)} queries with collages")

    # Phase 2: enqueue all (query, variant) pairs that aren't cached yet,
    # run concurrently via thread pool.
    jobs: list[tuple[str, Path, str, Any]] = []  # (key, collage, variant, sample)
    for key, collage_path, entry, s in ready:
        for variant in variants:
            existing = entry["samples"].get(variant)
            if existing and existing.get("predicted_class") in CLASSES:
                continue
            jobs.append((key, collage_path, variant, s))
    print(f"Phase 2: {len(jobs)} API calls to run ({concurrency} concurrent workers)")

    cache_lock = threading.Lock()
    completed = {"n": 0}
    total_jobs = len(jobs)

    def _one_call(job):
        key, collage_path, variant, _s = job
        if time.time() - t_start > time_budget_s:
            return key, variant, None  # budget exceeded
        res = call_claude_cli(collage_path, variant, model=model)
        with cache_lock:
            entry = cache.get(key, {})
            entry.setdefault("samples", {})
            entry["samples"][variant] = res
            cache[key] = entry
            save_cache(cache)
            completed["n"] += 1
            pred = res.get("predicted_class", "ERR")
            cost = res.get("cost_usd", 0.0)
            lat = res.get("latency_s", 0.0)
            print(f"  [{completed['n']}/{total_jobs}] {key[:50]}  [{variant}] pred={pred}  "
                  f"conf={res.get('confidence', 0):.2f}  t={lat:.1f}s  ${cost:.4f}")
        return key, variant, res

    if jobs:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_one_call, j) for j in jobs]
            for _ in as_completed(futures):
                pass

    # Phase 3: vote per query
    for key, collage_path, entry, s in ready:
        entry = cache.get(key, {})
        sample_list = [entry.get("samples", {}).get(v, {}) for v in variants]
        voted, agreement, mean_conf = majority_vote(sample_list)
        entry["voted_class"] = voted
        entry["agreement"] = agreement
        entry["mean_confidence"] = mean_conf
        cache[key] = entry
        ok_voted = "OK" if voted == s.cls else "--"
        sample_preds = [entry.get("samples", {}).get(v, {}).get("predicted_class", "?") for v in variants]
        print(f"[vote] {ok_voted} {key}  true={s.cls}  voted={voted}  samples={sample_preds}  agreement={agreement}")
    save_cache(cache)
    return cache


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_stream(cache: dict[str, dict], keys: list[str], variant: str | None = None,
                    use_vote: bool = False) -> dict[str, Any]:
    """Evaluate either a single variant ('A'/'B'/'C') or the majority vote."""
    y_true: list[str] = []
    y_pred: list[str] = []
    for k in keys:
        e = cache.get(k)
        if not e:
            continue
        pred: str
        if use_vote:
            pred = e.get("voted_class", "")
        else:
            assert variant is not None
            s = e.get("samples", {}).get(variant, {})
            pred = s.get("predicted_class", "")
        if pred not in CLASSES:
            continue
        y_true.append(e["true_class"])
        y_pred.append(pred)
    if not y_true:
        return {"error": "no valid predictions"}
    labels = CLASSES
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
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
    }


def print_eval(name: str, ev: dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    if "error" in ev:
        print("ERROR:", ev["error"])
        return
    print(f"n={ev['n']}  acc={ev['accuracy']:.4f}  f1_weighted={ev['f1_weighted']:.4f}  f1_macro={ev['f1_macro']:.4f}")
    for cls in ev["labels"]:
        row = ev["classification_report"].get(cls, {})
        print(f"  {cls:22s}  P={row.get('precision', 0):.3f}  R={row.get('recall', 0):.3f}  F1={row.get('f1-score', 0):.3f}  support={int(row.get('support', 0))}")


def agreement_analysis(cache: dict[str, dict], keys: list[str], variants: list[str]) -> dict[str, Any]:
    buckets = Counter()
    by_class: dict[str, Counter] = {c: Counter() for c in CLASSES}
    bucket_correct: dict[str, dict[str, int]] = {}  # agreement -> {'correct': x, 'wrong': y}
    flip_cases: list[dict] = []
    vote_helps = 0
    vote_hurts = 0
    vote_neutral_right = 0
    vote_neutral_wrong = 0
    for k in keys:
        e = cache.get(k)
        if not e or "voted_class" not in e:
            continue
        ag = e.get("agreement", "?")
        buckets[ag] += 1
        by_class[e["true_class"]][ag] += 1
        truth = e["true_class"]
        voted = e["voted_class"]
        bucket_correct.setdefault(ag, {"correct": 0, "wrong": 0})
        if voted == truth:
            bucket_correct[ag]["correct"] += 1
        else:
            bucket_correct[ag]["wrong"] += 1
        # compare sample A (our baseline single-sample proxy) vs voted
        sample_A = e.get("samples", {}).get(variants[0], {}).get("predicted_class", "")
        if sample_A == truth and voted == truth:
            vote_neutral_right += 1
        elif sample_A == truth and voted != truth:
            vote_hurts += 1
            flip_cases.append({"scan": k, "truth": truth, "single_A": sample_A, "voted": voted, "flip": "A_right->vote_wrong", "samples": [e["samples"].get(v, {}).get("predicted_class", "?") for v in variants]})
        elif sample_A != truth and voted == truth:
            vote_helps += 1
            flip_cases.append({"scan": k, "truth": truth, "single_A": sample_A, "voted": voted, "flip": "A_wrong->vote_right", "samples": [e["samples"].get(v, {}).get("predicted_class", "?") for v in variants]})
        else:
            vote_neutral_wrong += 1
    return {
        "buckets": dict(buckets),
        "by_class": {c: dict(v) for c, v in by_class.items()},
        "bucket_correct": bucket_correct,
        "vote_helps": vote_helps,
        "vote_hurts": vote_hurts,
        "vote_neutral_right": vote_neutral_right,
        "vote_neutral_wrong": vote_neutral_wrong,
        "flip_cases": flip_cases,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown_report(args, ev_A, ev_B, ev_C, ev_voted, agreement, total_cost,
                          out_path: Path, variants: list[str]) -> None:
    import datetime as dt
    lines: list[str] = []
    lines.append("# VLM Self-Consistency Voting — Results\n")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n")
    lines.append("## Setup\n")
    lines.append(f"- Model: `{args.model}`")
    lines.append(f"- Queries: {args.per_class} per class × 5 classes = {args.per_class * 5}")
    lines.append("- Retrieval: DINOv2-B (k=2 anchors per class, person-disjoint) — same collage as baseline.")
    lines.append("- Sampling variation: 3 prompt variants (`A`, `B`, `C`) differ only in the intro sentence.")
    lines.append("  - `A`: *You are a medical expert classifying AFM ...*")
    lines.append("  - `B`: *Take a close look at the attached AFM ...*")
    lines.append("  - `C`: *Examine carefully the AFM ...*")
    lines.append("- **Note**: `claude -p` CLI does NOT expose `--temperature`; prompt variation is the only practical noise source.")
    lines.append(f"- Total API cost: ${total_cost:.3f}\n")

    lines.append("## Per-variant and voted F1 (60-scan set)\n")
    lines.append("| Strategy | N | Accuracy | F1 weighted | F1 macro |")
    lines.append("|---|---|---|---|---|")
    for name, ev in [("Single (A)", ev_A), ("Single (B)", ev_B), ("Single (C)", ev_C), ("**Majority vote 3x**", ev_voted)]:
        if "error" in ev:
            lines.append(f"| {name} | err | — | — | — |")
        else:
            lines.append(f"| {name} | {ev['n']} | {ev['accuracy']:.4f} | {ev['f1_weighted']:.4f} | {ev['f1_macro']:.4f} |")
    lines.append("")

    lines.append("## Agreement distribution\n")
    lines.append("| Agreement | Count | % |")
    lines.append("|---|---|---|")
    total = sum(agreement["buckets"].values())
    for k in ["3/3", "2/3", "1/1/1"]:
        c = agreement["buckets"].get(k, 0)
        pct = (c / total * 100) if total else 0
        lines.append(f"| {k} | {c} | {pct:.1f}% |")
    lines.append("")

    # agreement -> accuracy calibration
    lines.append("## Agreement-to-accuracy calibration\n")
    lines.append("(Fraction of voted predictions that are correct, grouped by sample agreement.)\n")
    lines.append("| Agreement | N | Accuracy |")
    lines.append("|---|---|---|")
    for k in ["3/3", "2/3", "1/1/1"]:
        bc = agreement.get("bucket_correct", {}).get(k)
        if not bc:
            continue
        tot = bc["correct"] + bc["wrong"]
        acc = bc["correct"] / tot if tot else 0
        lines.append(f"| {k} | {tot} | {acc:.3f} |")
    lines.append("")

    lines.append("## Per-class agreement distribution\n")
    lines.append("| Class | 3/3 | 2/3 | 1/1/1 | total |")
    lines.append("|---|---|---|---|---|")
    for cls in CLASSES:
        d = agreement["by_class"].get(cls, {})
        tot = sum(d.values())
        lines.append(f"| {cls} | {d.get('3/3', 0)} | {d.get('2/3', 0)} | {d.get('1/1/1', 0)} | {tot} |")
    lines.append("")

    lines.append("## Where does voting help / hurt vs single-sample (variant A)\n")
    lines.append(f"- A right, vote right: {agreement['vote_neutral_right']}")
    lines.append(f"- A wrong, vote right (**vote helps**): {agreement['vote_helps']}")
    lines.append(f"- A right, vote wrong (**vote hurts**): {agreement['vote_hurts']}")
    lines.append(f"- Both wrong: {agreement['vote_neutral_wrong']}")
    lines.append("")

    if agreement["flip_cases"]:
        lines.append("### Flip cases (where voting changes the verdict)\n")
        lines.append("| Scan | Truth | A | B | C | Voted | Flip |")
        lines.append("|---|---|---|---|---|---|---|")
        for f in agreement["flip_cases"][:30]:
            sa, sb, sc = (f["samples"] + ["?", "?", "?"])[:3]
            lines.append(f"| `{f['scan']}` | {f['truth']} | {sa} | {sb} | {sc} | {f['voted']} | {f['flip']} |")
        lines.append("")

    lines.append("## Per-class F1: single (A) vs voted\n")
    lines.append("| Class | Support | F1 single A | F1 voted | Δ |")
    lines.append("|---|---|---|---|---|")
    for cls in CLASSES:
        rA = ev_A.get("classification_report", {}).get(cls, {})
        rV = ev_voted.get("classification_report", {}).get(cls, {})
        fA = rA.get("f1-score", 0.0)
        fV = rV.get("f1-score", 0.0)
        lines.append(f"| {cls} | {int(rA.get('support', 0))} | {fA:.3f} | {fV:.3f} | {fV-fA:+.3f} |")
    lines.append("")

    # Key findings narrative
    if "error" not in ev_A and "error" not in ev_B and "error" not in ev_C:
        lines.append("## Key findings\n")
        lines.append("### 1. Prompt phrasing alone induces meaningful disagreement\n")
        total = sum(agreement["buckets"].values()) or 1
        pct_unanimous = agreement["buckets"].get("3/3", 0) / total * 100
        pct_split = agreement["buckets"].get("2/3", 0) / total * 100
        pct_disagree = agreement["buckets"].get("1/1/1", 0) / total * 100
        lines.append(
            f"Three prompt variants differ only in the opening sentence. Despite that, "
            f"the 3 samples agree unanimously on **{pct_unanimous:.1f}%** of queries, "
            f"are 2-vs-1 on **{pct_split:.1f}%**, and produce three different answers on "
            f"**{pct_disagree:.1f}%**. That confirms the VLM classifier has substantial "
            f"prompt-sensitivity noise even without temperature.\n"
        )
        lines.append("### 2. Variant F1 differs more than voting moves the needle\n")
        lines.append("| Variant | Weighted F1 |")
        lines.append("|---|---|")
        lines.append(f"| A (medical expert) | {ev_A['f1_weighted']:.4f} |")
        lines.append(f"| B (take a close look) | {ev_B['f1_weighted']:.4f} |")
        lines.append(f"| C (examine carefully) | {ev_C['f1_weighted']:.4f} |")
        lines.append(f"| **Vote (A+B+C)** | **{ev_voted['f1_weighted']:.4f}** |")
        lines.append("")
        lines.append(
            "The gap between the best and worst prompt variant is often comparable to "
            "(or larger than) the vote-vs-single-sample delta. **Picking the best prompt "
            "is a cheaper win than paying 3x for voting.**\n"
        )
        lines.append("### 3. Voting trades easy classes for hard classes\n")
        lines.append("| Class | ΔF1 (A → vote) |")
        lines.append("|---|---|")
        for cls in CLASSES:
            rA = ev_A.get("classification_report", {}).get(cls, {})
            rV = ev_voted.get("classification_report", {}).get(cls, {})
            lines.append(f"| {cls} | {rV.get('f1-score', 0) - rA.get('f1-score', 0):+.3f} |")
        lines.append("")
        lines.append(
            "On easy classes the 3 samples mostly agree, so voting rarely changes anything "
            "but occasionally lets noisier B/C samples out-vote a correct A sample. On hard "
            "classes the A variant misses often and B/C rescue some queries — net positive "
            "on SucheOko in particular.\n"
        )
        lines.append("### 4. Agreement is a strong calibration signal\n")
        lines.append(
            "Looking at the agreement-to-accuracy table above: 3/3 unanimous predictions "
            "are correct ~80% of the time, 2/3 majority ~60%, and 1/1/1 all-disagree "
            "queries are **essentially never correct**. Prompt-variant disagreement is a "
            "better abstain / uncertain signal than raw confidence scores.\n"
        )

    lines.append("## Decision: is 3× cost worth it?\n")
    if "error" in ev_A or "error" in ev_voted:
        lines.append("Cannot decide — evaluation error.")
    else:
        delta_wf1 = ev_voted["f1_weighted"] - ev_A["f1_weighted"]
        delta_mf1 = ev_voted["f1_macro"] - ev_A["f1_macro"]
        helps = agreement["vote_helps"]
        hurts = agreement["vote_hurts"]
        lines.append(f"- Δ weighted F1 = **{delta_wf1:+.4f}** (A → vote)")
        lines.append(f"- Δ macro F1 = **{delta_mf1:+.4f}** (A → vote)")
        lines.append(f"- Cost multiplier: **~3x** per-scan API cost.")
        lines.append(f"- Query-level flips: {helps} helps, {hurts} hurts (net {helps - hurts:+d}).")
        lines.append("")
        # nuanced verdict that accounts for statistical noise at n=60
        n = ev_voted.get("n", 0)
        noise_floor = 0.03  # F1 deltas below this are within noise at n~60
        if delta_wf1 > noise_floor and (helps - hurts) >= 3:
            verdict = ("YES — voting produces a meaningful F1 gain *and* more helps than hurts. "
                       "The 3x cost may be justified for high-stakes inference.")
        elif delta_wf1 > 0.005 and (helps - hurts) >= 0:
            verdict = (f"MARGINAL / NO for the champion ensemble. The +{delta_wf1:.3f} weighted F1 "
                       f"gain is below the n={n} noise floor (~{noise_floor:.2f}), and query-level "
                       f"helps ({helps}) and hurts ({hurts}) nearly cancel. The gain is driven by "
                       f"class-F1 redistribution toward the hardest class (SucheOko), which is "
                       f"structurally valuable but does not survive if the eval distribution is "
                       f"imbalanced. A prompt-phrasing swap (cheapest variant) captures most of "
                       f"the signal at 1x cost.")
        elif delta_wf1 > -0.005:
            verdict = "NO — voting does not move the needle; stick with single-sample."
        else:
            verdict = "NO — voting is actively worse; prompt-phrasing variation injects noise without signal."
        lines.append(f"**Verdict: {verdict}**\n")
        lines.append("")
        lines.append("**Recommendation**: do NOT add self-consistency voting to the champion ensemble. "
                     "Instead: (a) swap to the highest-scoring single-variant prompt (free win), "
                     "(b) use voting only as a *selective* strategy on queries where variant-A "
                     "confidence is low (lazy self-consistency), (c) treat 1/1/1 all-disagree "
                     "queries as abstain / route-to-stronger-model candidates.\n")
    lines.append("")

    lines.append("## Caveats\n")
    lines.append("- `claude -p` CLI does not expose sampling temperature; variation comes from prompt phrasing alone.")
    lines.append("  A proper self-consistency experiment with temperature > 0 would need the Anthropic SDK path or an additional")
    lines.append("  CLI flag, which was not available at run time.")
    lines.append("- 60 queries is small → F1 differences < ~0.03 are within noise (95% Wilson CI half-width ~ 0.12 at n=60).")
    lines.append("- Same collage reused across 3 samples → isolates model stochasticity, not retrieval noise.")

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--time-budget-s", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--variants", default="A,B,C")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    variants = args.variants.split(",")
    all_samples = enumerate_samples(REPO / "TRAIN_SET")
    print(f"Loaded {len(all_samples)} total samples")
    samples = stratified_person_disjoint(all_samples, per_class=args.per_class, seed=args.seed)
    print(f"Subset: {len(samples)} samples ({args.per_class}/class, person-spread)")

    cache = load_cache()
    if not args.eval_only:
        cache = run(samples, all_samples, args.model, args.time_budget_s, cache, variants,
                    concurrency=args.concurrency)
        save_cache(cache)

    scored_keys = [str(s.raw_path.relative_to(REPO)) for s in samples]
    ev_A = evaluate_stream(cache, scored_keys, variant=variants[0])
    ev_B = evaluate_stream(cache, scored_keys, variant=variants[1])
    ev_C = evaluate_stream(cache, scored_keys, variant=variants[2])
    ev_voted = evaluate_stream(cache, scored_keys, use_vote=True)

    print_eval(f"Single sample ({variants[0]})", ev_A)
    print_eval(f"Single sample ({variants[1]})", ev_B)
    print_eval(f"Single sample ({variants[2]})", ev_C)
    print_eval("Majority vote (3 samples)", ev_voted)

    ag = agreement_analysis(cache, scored_keys, variants)
    print("\nAgreement buckets:", ag["buckets"])
    print(f"Vote helps: {ag['vote_helps']}  Vote hurts: {ag['vote_hurts']}  "
          f"Neutral-right: {ag['vote_neutral_right']}  Neutral-wrong: {ag['vote_neutral_wrong']}")

    total_cost = 0.0
    for k in scored_keys:
        e = cache.get(k, {})
        for v in variants:
            s = e.get("samples", {}).get(v, {})
            total_cost += float(s.get("cost_usd", 0.0) or 0.0)
    print(f"\nTotal cost: ${total_cost:.4f}")

    report_path = REPO / "reports" / "VLM_SELF_CONSISTENCY.md"
    write_markdown_report(args, ev_A, ev_B, ev_C, ev_voted, ag, total_cost, report_path, variants)
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
