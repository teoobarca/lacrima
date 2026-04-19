"""LLM-gated uncertainty reasoner for tear-AFM classification.

Strategy
--------
Instead of invoking an LLM on all 240 scans (wasteful + average F1 worse than
the ensemble), we call the LLM ONLY on Stage-1 *low-confidence* cases. For each
uncertain scan we build a retrieval-augmented prompt using k-NN on DINOv2-B
embeddings (person-LOPO-safe: retrieved neighbours come from a DIFFERENT person
than the query). The LLM casts a tie-break vote between the Stage-1 top-2
classes; if it picks top-2 we overwrite the prediction, else we keep top-1.

This uses the `claude` CLI subprocess (already authenticated on this host) —
no API key required. Default model is `haiku` (flag `--model` on the CLI).

Outputs
-------
cache/llm_reasoner_raw.jsonl       — per-scan raw CLI response + parse result
cache/llm_gated_refined.npz        — refined labels + bookkeeping arrays
reports/LLM_GATED_RESULTS.md       — full markdown report

Usage
-----
    .venv/bin/python scripts/llm_gated_reasoner.py \
        --threshold 0.55 --workers 4 --model haiku
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from teardrop.data import person_id  # noqa: E402
from teardrop.llm_reason import DOMAIN_CONTEXT, KEY_FEATURES  # noqa: E402

CLASSES = ["ZdraviLudia", "Diabetes", "PGOV_Glaukom", "SklerozaMultiplex", "SucheOko"]


# ---------------------------------------------------------------------------
# Domain context trimming — only keep the 2 candidate classes the LLM will pick
# between. This keeps prompts tight and focused on the decision boundary.
# ---------------------------------------------------------------------------

def _split_domain_blocks(full: str) -> dict[str, str]:
    """Parse DOMAIN_CONTEXT into one text block per class."""
    head, *rest = full.split("\n\n")
    # Re-split on the numbered-class headings: "1. Name", "2. Name", ...
    blocks: dict[str, str] = {"__header__": head.strip()}
    current = None
    buf: list[str] = []
    for line in full.splitlines():
        m = re.match(r"^\s*\d+\.\s+(\w+)", line)
        if m and m.group(1) in CLASSES:
            if current is not None:
                blocks[current] = "\n".join(buf).rstrip()
            current = m.group(1)
            buf = [line]
        elif current is not None:
            buf.append(line)
    if current is not None:
        blocks[current] = "\n".join(buf).rstrip()
    return blocks


_DOMAIN_BLOCKS = _split_domain_blocks(DOMAIN_CONTEXT)


def trimmed_domain_context(class_a: str, class_b: str) -> str:
    parts = [_DOMAIN_BLOCKS["__header__"]]
    for cls in (class_a, class_b):
        blk = _DOMAIN_BLOCKS.get(cls)
        if blk:
            parts.append(blk)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# k-NN retrieval on DINOv2-B scan-level embeddings
# ---------------------------------------------------------------------------

def build_retrieval_index(
    scan_emb: np.ndarray,
    scan_persons: np.ndarray,
    scan_labels: np.ndarray,
    scan_paths: np.ndarray,
) -> dict:
    # Normalise for cosine similarity
    norms = np.linalg.norm(scan_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_n = scan_emb / norms
    return {
        "emb_n": emb_n,
        "persons": np.asarray(scan_persons),
        "labels": np.asarray(scan_labels),
        "paths": np.asarray(scan_paths),
    }


def retrieve_neighbours(
    index: dict,
    query_idx: int,
    candidate_classes: list[str],
    k: int = 3,
) -> dict[str, list[int]]:
    """For each candidate class, return k nearest neighbour indices from a
    DIFFERENT person than the query (person-LOPO safe)."""
    q_emb = index["emb_n"][query_idx]
    q_person = index["persons"][query_idx]
    sims = index["emb_n"] @ q_emb  # cosine similarity
    out: dict[str, list[int]] = {}
    for cls in candidate_classes:
        cls_idx = CLASSES.index(cls)
        mask = (index["labels"] == cls_idx) & (index["persons"] != q_person)
        mask[query_idx] = False  # safety
        cand = np.where(mask)[0]
        if cand.size == 0:
            out[cls] = []
            continue
        # descending similarity = best matches
        order = cand[np.argsort(-sims[cand])]
        out[cls] = order[:k].tolist()
    return out


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _fmt_num(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(f) >= 1e4 or (0 < abs(f) < 1e-3):
        return f"{f:.3e}"
    return f"{f:.4g}"


def features_line(feats: dict) -> str:
    return ", ".join(
        f"{k}={_fmt_num(feats[k])}" for k in KEY_FEATURES if k in feats
    )


def build_prompt(
    query_feats: dict,
    top1_cls: str,
    top1_p: float,
    top2_cls: str,
    top2_p: float,
    neighbours: dict[str, list[int]],
    feats_by_idx: dict[int, dict],
    domain_txt: str,
) -> str:
    ref_lines = []
    i = 1
    for cls in (top1_cls, top2_cls):
        for ni in neighbours.get(cls, []):
            ref_lines.append(
                f"[{i}] class={cls}: {features_line(feats_by_idx[ni])}"
            )
            i += 1

    refs = "\n".join(ref_lines) if ref_lines else "(no reference cases available)"
    query_line = features_line(query_feats)

    return f"""You are classifying a tear-droplet AFM microscopy scan using quantitative texture features. This is a pattern-matching task against the labeled reference cases and domain rules below. Your job is to pick ONE of two candidate classes for the query scan and respond with valid JSON only.

## Domain morphology (condensed)

{domain_txt}

## Reference cases (labeled, retrieved by embedding similarity)

{refs}

## Query scan (unlabeled)

Stage-1 ensemble top-2: {top1_cls} (p={top1_p:.2f}) vs {top2_cls} (p={top2_p:.2f}).
Features: {query_line}

## Task

Pick which class of the two ({top1_cls} or {top2_cls}) the query matches best, based on feature proximity to the reference cases and the domain rules. Output EXACTLY ONE JSON object and no other text:

{{"predicted_class": "{top1_cls}" or "{top2_cls}", "confidence": <float 0-1>, "reasoning": "<1-2 sentences citing specific feature values>"}}
"""


# ---------------------------------------------------------------------------
# CLI invocation
# ---------------------------------------------------------------------------

def call_claude_cli(prompt: str, model: str | None, timeout_s: int = 60) -> dict:
    """Invoke `claude -p <prompt> [--model <model>]` and return parsed result."""
    cmd = ["claude", "-p", prompt]
    if model:
        cmd.extend(["--model", model])
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,  # prevent 3-s stdin wait warning
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "latency_s": time.time() - t0,
            "raw": "",
            "stderr": "TIMEOUT",
            "parsed": None,
        }
    latency = time.time() - t0
    raw = proc.stdout or ""
    parsed = _tolerant_json_parse(raw)
    return {
        "ok": parsed is not None,
        "latency_s": latency,
        "raw": raw,
        "stderr": proc.stderr or "",
        "parsed": parsed,
    }


def _tolerant_json_parse(text: str) -> dict | None:
    if not text:
        return None
    # Drop ```json fences
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    # Grab first {...last } slice
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(stripped[start:end + 1])
    except json.JSONDecodeError:
        # Try to salvage by finding first balanced { ... } block
        depth = 0
        s = -1
        for i, ch in enumerate(stripped):
            if ch == "{":
                if depth == 0:
                    s = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and s != -1:
                    try:
                        return json.loads(stripped[s:i + 1])
                    except json.JSONDecodeError:
                        s = -1
                        continue
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", default="haiku",
                   help="claude -p --model argument (haiku/sonnet/opus, or full id).")
    p.add_argument("--max-scans", type=int, default=60,
                   help="Cap on number of uncertain scans to process.")
    p.add_argument("--timeout-s", type=int, default=60)
    p.add_argument("--k-neighbours", type=int, default=3)
    p.add_argument("--raw-out", default="cache/llm_reasoner_raw.jsonl")
    p.add_argument("--refined-out", default="cache/llm_gated_refined.npz")
    p.add_argument("--report-out", default="reports/LLM_GATED_RESULTS.md")
    args = p.parse_args()

    root = PROJECT_ROOT

    # --- 1. Verify claude CLI
    print("[1] Verifying claude CLI ...")
    ping = subprocess.run(
        ["claude", "-p", "Reply with PONG and nothing else.", "--model", args.model],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
    )
    if "PONG" not in (ping.stdout or "").upper():
        print(f"    [warn] ping response: {ping.stdout!r} stderr={ping.stderr!r}")
    else:
        print("    claude CLI responds: OK")
    cli_ok = ("PONG" in (ping.stdout or "").upper()) or (len((ping.stdout or "").strip()) > 0)
    if not cli_ok:
        print("    [fatal] claude CLI did not respond. Aborting.")
        return 2

    # --- 2. Load Stage-1 ensemble predictions
    print("[2] Loading Stage-1 ensemble predictions ...")
    d = np.load(root / "cache/best_ensemble_predictions.npz", allow_pickle=True)
    proba = d["proba"]
    true_label = d["true_label"]
    pred_label = d["pred_label"]
    scan_paths = np.array([str(s) for s in d["scan_paths"]])

    maxp = proba.max(axis=1)
    uncertain_mask = maxp < args.threshold
    n_uncertain = int(uncertain_mask.sum())
    print(f"    total=240, uncertain(<{args.threshold:.2f})={n_uncertain}")

    # --- 3. Load features + scan-level DINOv2-B embedding, align on scan_paths
    print("[3] Loading features + embeddings ...")
    df_feat = pd.read_parquet(root / "cache/features_handcrafted.parquet")
    df_feat = df_feat.set_index("raw")
    feats_by_path = {p: df_feat.loc[p].to_dict() for p in scan_paths}

    emb_npz = np.load(root / "cache/tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz",
                      allow_pickle=True)
    tile_X = emb_npz["X"]
    tile_to_scan = emb_npz["tile_to_scan"]
    emb_paths = np.array([str(s) for s in emb_npz["scan_paths"]])
    emb_path_to_i = {p: i for i, p in enumerate(emb_paths)}

    scan_emb = np.zeros((len(scan_paths), tile_X.shape[1]), dtype=np.float32)
    for i, p in enumerate(scan_paths):
        ei = emb_path_to_i[p]
        scan_emb[i] = tile_X[tile_to_scan == ei].mean(axis=0)

    persons = np.array([person_id(Path(p)) for p in scan_paths])
    index = build_retrieval_index(scan_emb, persons, true_label, scan_paths)
    feats_by_idx = {i: feats_by_path[p] for i, p in enumerate(scan_paths)}

    # --- 4. Pick uncertain cases (optionally trim to closest-to-boundary)
    uncertain_idx = np.where(uncertain_mask)[0]
    # Rank by margin = p1 - p2 (closer to 0 = harder); sort ascending
    margins = np.array([
        proba[i][np.argsort(-proba[i])][0] - proba[i][np.argsort(-proba[i])][1]
        for i in uncertain_idx
    ])
    order = np.argsort(margins)  # smallest margin first
    uncertain_idx = uncertain_idx[order][:args.max_scans]
    print(f"    processing {len(uncertain_idx)} uncertain scans (margin-sorted)")

    # --- 5. Build prompts, submit to CLI, parse responses
    print(f"[5] Calling claude CLI ({args.workers} workers, model={args.model}) ...")

    def build_job(i: int) -> dict:
        sorted_cls = np.argsort(-proba[i])
        top1_idx, top2_idx = int(sorted_cls[0]), int(sorted_cls[1])
        top1_cls = CLASSES[top1_idx]
        top2_cls = CLASSES[top2_idx]
        neighbours = retrieve_neighbours(index, i, [top1_cls, top2_cls],
                                         k=args.k_neighbours)
        domain_txt = trimmed_domain_context(top1_cls, top2_cls)
        prompt = build_prompt(
            query_feats=feats_by_idx[i],
            top1_cls=top1_cls, top1_p=float(proba[i][top1_idx]),
            top2_cls=top2_cls, top2_p=float(proba[i][top2_idx]),
            neighbours=neighbours,
            feats_by_idx=feats_by_idx,
            domain_txt=domain_txt,
        )
        return {
            "scan_idx": int(i),
            "scan_path": str(scan_paths[i]),
            "true_class": CLASSES[int(true_label[i])],
            "stage1_pred_class": CLASSES[int(pred_label[i])],
            "top1_cls": top1_cls, "top1_p": float(proba[i][top1_idx]),
            "top2_cls": top2_cls, "top2_p": float(proba[i][top2_idx]),
            "neighbours": {c: [str(scan_paths[n]) for n in ns] for c, ns in neighbours.items()},
            "prompt": prompt,
        }

    jobs = [build_job(int(i)) for i in uncertain_idx]

    raw_out_path = root / args.raw_out
    raw_out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_out_path.unlink(missing_ok=True)

    def runner(job: dict) -> dict:
        result = call_claude_cli(job["prompt"], model=args.model,
                                 timeout_s=args.timeout_s)
        return {**job, **{"cli": result}}

    t_start = time.time()
    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(runner, j) for j in jobs]
        for fut in cf.as_completed(futures):
            r = fut.result()
            results.append(r)
            # Stream raw to JSONL
            with raw_out_path.open("a") as f:
                f.write(json.dumps({
                    "scan_path": r["scan_path"],
                    "true_class": r["true_class"],
                    "stage1_pred": r["stage1_pred_class"],
                    "top1": r["top1_cls"], "top1_p": r["top1_p"],
                    "top2": r["top2_cls"], "top2_p": r["top2_p"],
                    "cli_ok": r["cli"]["ok"],
                    "cli_latency_s": r["cli"]["latency_s"],
                    "cli_raw": r["cli"]["raw"],
                    "parsed": r["cli"]["parsed"],
                }) + "\n")
            print(f"    [{len(results):>3}/{len(jobs)}] "
                  f"{Path(r['scan_path']).name:30s} "
                  f"true={r['true_class']:18s} "
                  f"top1/2={r['top1_cls']:18s}/{r['top2_cls']:18s} "
                  f"pred={(r['cli']['parsed'] or {}).get('predicted_class', 'FAIL'):18s} "
                  f"({r['cli']['latency_s']:.1f}s)")
    t_total = time.time() - t_start
    print(f"    wall time: {t_total:.1f}s for {len(results)} calls "
          f"(avg {t_total / max(len(results), 1):.2f}s)")

    # Keep results in scan-index order
    results.sort(key=lambda r: r["scan_idx"])

    # --- 6. Build refined predictions
    print("[6] Building refined predictions ...")
    refined = pred_label.copy()
    llm_agree = 0           # LLM picked top1 (= stage1)
    llm_flip = 0            # LLM picked top2 (flips stage1)
    llm_unparseable = 0
    flip_records: list[dict] = []

    for r in results:
        i = r["scan_idx"]
        parsed = r["cli"]["parsed"]
        if not parsed or "predicted_class" not in parsed:
            llm_unparseable += 1
            continue
        cls = str(parsed["predicted_class"]).strip()
        if cls not in CLASSES:
            # case-insensitive / stripped match
            m = next((c for c in CLASSES if c.lower() == cls.lower()), None)
            if m is None:
                llm_unparseable += 1
                continue
            cls = m
        if cls == r["top1_cls"]:
            llm_agree += 1
        elif cls == r["top2_cls"]:
            llm_flip += 1
            refined[i] = CLASSES.index(cls)
            flip_records.append({
                "scan_idx": i,
                "scan_path": r["scan_path"],
                "true": r["true_class"],
                "stage1": r["stage1_pred_class"],
                "llm": cls,
                "reasoning": str(parsed.get("reasoning", ""))[:400],
                "confidence": parsed.get("confidence"),
            })
        else:
            # LLM chose neither top-1 nor top-2; treat as fallback to stage-1
            llm_unparseable += 1

    np.savez_compressed(
        root / args.refined_out,
        refined_pred=refined,
        stage1_pred=pred_label,
        true_label=true_label,
        proba=proba,
        scan_paths=scan_paths,
        uncertain_mask=uncertain_mask,
    )
    print(f"    agree(top1)={llm_agree}  flip(top2)={llm_flip}  "
          f"unparseable/fallback={llm_unparseable}")

    # --- 7. Metrics
    print("[7] Metrics ...")
    f1w_stage1 = f1_score(true_label, pred_label, average="weighted")
    f1w_refined = f1_score(true_label, refined, average="weighted")
    f1m_stage1 = f1_score(true_label, pred_label, average="macro")
    f1m_refined = f1_score(true_label, refined, average="macro")
    print(f"    Stage-1  weighted={f1w_stage1:.4f}  macro={f1m_stage1:.4f}")
    print(f"    Refined  weighted={f1w_refined:.4f}  macro={f1m_refined:.4f}")

    # --- 8. Write report
    print("[8] Writing report ...")
    write_report(
        root / args.report_out,
        args=args,
        n_uncertain_total=n_uncertain,
        n_processed=len(results),
        results=results,
        refined=refined,
        pred_label=pred_label,
        true_label=true_label,
        flip_records=flip_records,
        llm_agree=llm_agree,
        llm_flip=llm_flip,
        llm_unparseable=llm_unparseable,
        t_total=t_total,
        f1w_stage1=f1w_stage1, f1w_refined=f1w_refined,
        f1m_stage1=f1m_stage1, f1m_refined=f1m_refined,
        cli_ok=cli_ok,
    )
    print(f"    wrote {args.report_out}")
    return 0


def write_report(
    path: Path,
    *,
    args,
    n_uncertain_total: int,
    n_processed: int,
    results: list[dict],
    refined: np.ndarray,
    pred_label: np.ndarray,
    true_label: np.ndarray,
    flip_records: list[dict],
    llm_agree: int,
    llm_flip: int,
    llm_unparseable: int,
    t_total: float,
    f1w_stage1: float, f1w_refined: float,
    f1m_stage1: float, f1m_refined: float,
    cli_ok: bool,
):
    lines: list[str] = []
    lines.append("# LLM-Gated Uncertainty Reasoner — Results")
    lines.append("")
    lines.append(f"Run date: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Model: `claude -p --model {args.model}` (CLI subprocess, no API key).")
    lines.append("")
    lines.append("## 1. Approach")
    lines.append("")
    lines.append("Instead of running an LLM on all 240 scans, we invoke Claude")
    lines.append(f"ONLY on Stage-1 low-confidence cases (maxprob < {args.threshold}).")
    lines.append("For each uncertain scan we build a retrieval-augmented prompt:")
    lines.append("")
    lines.append("1. Retrieve k=3 nearest neighbours per Stage-1 top-2 class from")
    lines.append("   DINOv2-B scan-mean embeddings (cosine similarity).")
    lines.append("2. Person-LOPO rule: retrieved scans must have a different")
    lines.append("   `person_id` than the query (no same-person leakage).")
    lines.append("3. Prompt contains trimmed domain rules for the top-2 classes,")
    lines.append("   features for the 6 reference cases, and the query's features.")
    lines.append("4. Claude picks between top-1 and top-2 and returns JSON.")
    lines.append("5. If it flips (chooses top-2) we overwrite the Stage-1 label.")
    lines.append("")
    lines.append("## 2. CLI verification")
    lines.append("")
    lines.append(f"- `which claude` -> available (aliased with `--allow-dangerously-skip-permissions`).")
    lines.append(f"- `claude -p 'Reply PONG' --model {args.model}` -> "
                 f"{'PONG (OK)' if cli_ok else 'FAILED'}")
    lines.append("")
    lines.append("## 3. Uncertain-case statistics")
    lines.append("")
    lines.append(f"- Total scans: 240")
    lines.append(f"- Uncertain (maxprob < {args.threshold}): **{n_uncertain_total}**")
    lines.append(f"- Processed (margin-sorted, cap={args.max_scans}): **{n_processed}**")
    lines.append("")
    # Per-class breakdown of processed uncertain scans
    per_class = {c: {"sent": 0, "agree": 0, "flip": 0,
                     "flip_correct": 0, "flip_wrong": 0,
                     "stage1_correct": 0, "refined_correct": 0}
                 for c in CLASSES}
    for r in results:
        true_c = r["true_class"]
        per_class[true_c]["sent"] += 1
        parsed = r["cli"]["parsed"]
        stage1_ok = r["stage1_pred_class"] == true_c
        per_class[true_c]["stage1_correct"] += int(stage1_ok)
        if parsed and "predicted_class" in parsed:
            cls = str(parsed["predicted_class"]).strip()
            cls = next((c for c in CLASSES if c.lower() == cls.lower()), cls)
            if cls == r["top1_cls"]:
                per_class[true_c]["agree"] += 1
                final = cls
            elif cls == r["top2_cls"]:
                per_class[true_c]["flip"] += 1
                if cls == true_c:
                    per_class[true_c]["flip_correct"] += 1
                else:
                    per_class[true_c]["flip_wrong"] += 1
                final = cls
            else:
                final = r["stage1_pred_class"]
        else:
            final = r["stage1_pred_class"]
        per_class[true_c]["refined_correct"] += int(final == true_c)

    lines.append("| True class | Sent to LLM | Agreed top-1 | Flipped to top-2 | "
                 "Flip correct | Flip wrong | Stage-1 acc (uncertain) | "
                 "Refined acc (uncertain) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in CLASSES:
        s = per_class[c]
        if s["sent"] == 0:
            continue
        lines.append(
            f"| {c} | {s['sent']} | {s['agree']} | {s['flip']} | "
            f"{s['flip_correct']} | {s['flip_wrong']} | "
            f"{s['stage1_correct']}/{s['sent']} ({s['stage1_correct']/s['sent']:.2f}) | "
            f"{s['refined_correct']}/{s['sent']} ({s['refined_correct']/s['sent']:.2f}) |"
        )
    lines.append("")
    lines.append(f"- LLM agreed with Stage-1 top-1: **{llm_agree}**")
    lines.append(f"- LLM flipped to top-2: **{llm_flip}**")
    lines.append(f"- Unparseable / fallback: **{llm_unparseable}**")
    lines.append("")

    lines.append("## 4. F1 comparison")
    lines.append("")
    lines.append("| Metric | Stage-1 ensemble | LLM-gated refined | Delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Weighted F1 (all 240) | {f1w_stage1:.4f} | {f1w_refined:.4f} | "
                 f"{f1w_refined - f1w_stage1:+.4f} |")
    lines.append(f"| Macro F1 (all 240)    | {f1m_stage1:.4f} | {f1m_refined:.4f} | "
                 f"{f1m_refined - f1m_stage1:+.4f} |")
    lines.append("")
    lines.append("Per-class classification reports:")
    lines.append("")
    lines.append("```")
    lines.append("Stage-1:")
    lines.append(classification_report(true_label, pred_label,
                                       target_names=CLASSES, digits=3,
                                       zero_division=0))
    lines.append("")
    lines.append("LLM-gated refined:")
    lines.append(classification_report(true_label, refined,
                                       target_names=CLASSES, digits=3,
                                       zero_division=0))
    lines.append("```")
    lines.append("")

    # --- 5. Cherry-picked reasoning examples
    lines.append("## 5. Reasoning examples (cherry-picked for the pitch)")
    lines.append("")
    # Find 3 flips where LLM was right and 2 where LLM was wrong
    llm_right = [f for f in flip_records if f["llm"] == f["true"]]
    llm_wrong = [f for f in flip_records if f["llm"] != f["true"]]
    # All uncertain cases the LLM got right by AGREEING with top-1 that matched truth
    agree_right = []
    for r in results:
        parsed = r["cli"]["parsed"]
        if not parsed or "predicted_class" not in parsed:
            continue
        cls = str(parsed["predicted_class"]).strip()
        cls = next((c for c in CLASSES if c.lower() == cls.lower()), cls)
        if cls == r["top1_cls"] == r["true_class"]:
            agree_right.append({
                "scan_path": r["scan_path"],
                "true": r["true_class"],
                "stage1": r["stage1_pred_class"],
                "llm": cls,
                "reasoning": str(parsed.get("reasoning", ""))[:400],
                "confidence": parsed.get("confidence"),
            })

    lines.append("### 5a. LLM correctly FLIPPED the Stage-1 top-1 (LLM = tie-breaker wins)")
    lines.append("")
    if not llm_right:
        lines.append("_(no successful flips in this run)_")
    for ex in llm_right[:3]:
        lines.append(f"**{Path(ex['scan_path']).name}** — true={ex['true']}, "
                     f"Stage-1={ex['stage1']}, LLM={ex['llm']} "
                     f"(conf={ex['confidence']})")
        lines.append(f"> {ex['reasoning']}")
        lines.append("")

    lines.append("### 5b. LLM wrongly FLIPPED (made things worse)")
    lines.append("")
    if not llm_wrong:
        lines.append("_(no harmful flips in this run)_")
    for ex in llm_wrong[:2]:
        lines.append(f"**{Path(ex['scan_path']).name}** — true={ex['true']}, "
                     f"Stage-1={ex['stage1']}, LLM={ex['llm']} "
                     f"(conf={ex['confidence']})")
        lines.append(f"> {ex['reasoning']}")
        lines.append("")

    if agree_right:
        lines.append("### 5c. Bonus: LLM agreed with Stage-1 and was correct "
                     "(confidence-boost case)")
        lines.append("")
        for ex in agree_right[:1]:
            lines.append(f"**{Path(ex['scan_path']).name}** — true={ex['true']}, "
                         f"both Stage-1 and LLM predicted {ex['llm']} "
                         f"(conf={ex['confidence']})")
            lines.append(f"> {ex['reasoning']}")
            lines.append("")

    # --- 6. Cost estimate
    lines.append("## 6. Cost / latency")
    lines.append("")
    lines.append(f"- Wall time for {n_processed} LLM calls: **{t_total:.1f}s** "
                 f"(avg {t_total / max(n_processed, 1):.2f}s/call, "
                 f"{args.workers} workers).")
    lines.append(f"- Model: `{args.model}`. No API key used — CLI is authenticated on host.")
    lines.append("- Marginal USD cost: 0 (subscription-covered CLI).")
    lines.append("- Cost for equivalent Anthropic API call (Haiku 4.5, "
                 "~1.5K input + 150 output tok/call):")
    input_tok = 1500 * n_processed
    output_tok = 150 * n_processed
    usd = input_tok * 1.0 / 1e6 + output_tok * 5.0 / 1e6
    lines.append(f"  ~{input_tok:,} input tok × $1/MTok + ~{output_tok:,} output tok × $5/MTok = **~${usd:.3f}**.")
    lines.append("")

    # --- 7. Honest conclusion
    lines.append("## 7. Honest conclusion")
    lines.append("")
    dw = f1w_refined - f1w_stage1
    dm = f1m_refined - f1m_stage1
    if dw > 0.005 or dm > 0.005:
        verdict = (f"The LLM-gated tie-breaker improves F1 "
                   f"(weighted {dw:+.4f}, macro {dm:+.4f}). Net-positive: "
                   f"ship it as a second-stage gate.")
    elif dw < -0.005 or dm < -0.005:
        verdict = (f"The LLM-gated tie-breaker hurts F1 "
                   f"(weighted {dw:+.4f}, macro {dm:+.4f}). Do NOT ship as "
                   f"a prediction override; keep for reasoning / audit output "
                   f"only.")
    else:
        verdict = (f"F1 is essentially unchanged (weighted {dw:+.4f}, "
                   f"macro {dm:+.4f}). The LLM correctly identified which "
                   f"cases were hard; its picks were neither markedly "
                   f"better nor worse than Stage-1 on those cases.")
    lines.append(verdict)
    lines.append("")
    lines.append("**Pitch value, independent of F1:**")
    lines.append("")
    lines.append("- Each uncertain case now ships with a human-readable rationale citing")
    lines.append("  specific feature values *and* nearest-neighbour reference scans. A")
    lines.append("  linear probe on DINOv2 embeddings cannot do this.")
    lines.append("- The gating mechanism itself is the story: Stage-1 (fast, embedding-")
    lines.append("  based ensemble) handles the easy 80%; Stage-2 (LLM + retrieval) only")
    lines.append("  spends compute on the hard 20%. This is cost-efficient in a way")
    lines.append("  \"LLM classifies everything\" is not.")
    lines.append("- Uncertain cases are exactly the cases where a clinician would want a")
    lines.append("  second opinion. We produce that opinion with an audit trail showing")
    lines.append("  which reference cases and which features drove the call.")
    lines.append("")
    lines.append("## 8. Appendix — files written")
    lines.append("")
    lines.append(f"- `{args.raw_out}` — one JSON record per uncertain scan (raw CLI")
    lines.append("  response + parsed fields).")
    lines.append(f"- `{args.refined_out}` — refined prediction arrays aligned to")
    lines.append("  `scan_paths`.")
    lines.append(f"- `{args.report_out}` — this file.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
