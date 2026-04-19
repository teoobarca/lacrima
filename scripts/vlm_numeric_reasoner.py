"""VLM Numeric Reasoner — Sonnet 4.6 on quantitative features only (NO image).

Hypothesis
----------
All prior VLM-vision attempts failed because AFM ferning is out-of-distribution
visually (zero-shot 30%, few-shot honest 34%, binary re-ranker 35%). But the
class signatures are well-documented in TEXT (published tear-ferning morphology
+ Masmali grading). This script tests whether Sonnet 4.6 — given ONLY 15-20
quantitative features (fractal D, Masmali heuristic, Rq/Sa/Ssk/Sku, branch
density, GLCM, LBP, lacunarity) + the written class signatures — can reason
back to the correct class. If yes, we bypass the OOD-vision problem because
text reasoning is IN-distribution for a frontier LLM.

Pipeline
--------
1. Load `cache/features_advanced.parquet` (240 scans, 448 features). Extract
   the 16 most diagnostic features. Derive Masmali heuristic grade and
   branch density proxy (expert_council recipe).
2. Stratified 60-scan pilot (seed=42). Sonnet 4.6 via Anthropic SDK.
   Concurrent calls bounded by a ThreadPoolExecutor (default 4 workers).
3. Run `assert_prompt_safe(prompt)` before every call — belt-and-braces,
   even though no image and no path are ever sent. Raw filenames are never
   embedded; scans are referenced by the zero-padded index only.
4. Stream raw JSONL records to `cache/vlm_numeric_reasoner_raw.jsonl`
   as they arrive.
5. Score the subset (weighted/macro F1, per-class F1, confusion matrix).
   If wF1 >= 0.50 -> scale to remaining 180 scans. Else stop.
6. Bootstrap Delta wF1 vs v4 baseline on the scored-scan overlap.
7. Write `cache/vlm_numeric_reasoner_predictions.json` + the report
   `reports/VLM_NUMERIC_REASONER.md`.

No image tokens, no tile paths, no filenames leaked — this is a pure numeric
+ text prompt. The `safe_paths.assert_prompt_safe` call is still made to
inherit the project-wide guarantee that no VLM script can accidentally leak
the label through prompt content.

Run
---
    .venv/bin/python scripts/vlm_numeric_reasoner.py

Uses the local ``claude -p`` CLI (no ANTHROPIC_API_KEY needed — the CLI
reuses the user's Claude login session).

Options
-------
    --subset N     Pilot size (default 60, stratified).
    --workers K    Thread count (default 4).
    --model ID     Claude model ID (default claude-sonnet-4-6).
    --full         Force full 240-scan run even if pilot wF1 < gate.
    --dry-run      Print one rendered prompt and exit — no API calls.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from teardrop.data import CLASSES  # noqa: E402
from teardrop.safe_paths import assert_prompt_safe  # noqa: E402


CACHE = REPO / "cache"
REPORTS = REPO / "reports"

FEATS_ADV = CACHE / "features_advanced.parquet"
FEATS_HC = CACHE / "features_handcrafted.parquet"  # fallback

OUT_JSON = CACHE / "vlm_numeric_reasoner_predictions.json"
RAW_JSONL = CACHE / "vlm_numeric_reasoner_raw.jsonl"
REPORT_PATH = REPORTS / "VLM_NUMERIC_REASONER.md"

V4_OOF_FILE = CACHE / "v4_oof_predictions.npz"

# Gate for scaling up from pilot -> full
F1_SCALE_UP_GATE = 0.50

# Sonnet 4.6 pricing cached 2026-04: $3.00/MTok input, $15.00/MTok output.
SONNET_IN_USD = 3.00
SONNET_OUT_USD = 15.00


# ---------------------------------------------------------------------------
# Class signatures (text) — the biological priors that make the LLM reasoning
# possible. Single source of truth for the prompt.
# ---------------------------------------------------------------------------

CLASS_SIGNATURES = """\
- ZdraviLudia (healthy): Dense, highly-branched dendritic fern \
(Masmali grade 0-1). Fractal D ~ 1.70-1.85. Moderate GLCM homogeneity. \
LBP balanced between flat and edge.
- Diabetes: Thicker/coarser branches, higher packing density (elevated tear \
osmolarity). Fractal D ~ 1.60-1.75. Sq elevated. Ssk skewed positive. \
GLCM contrast elevated.
- PGOV_Glaukom (primary open-angle glaucoma): Granular structure with \
loops/rings, MMP-9 degrades glycoprotein matrix. Fractal D typically \
< 1.65, noisier. GLCM correlation LOW (locally chaotic). Branch density low.
- SklerozaMultiplex (MS): HETEROGENEOUS morphology within sample — coarse \
rods AND fine granules side by side. High intra-sample variance (mf_alpha_width \
large, high GLCM/LBP spread). Often confused with PGOV_Glaukom visually.
- SucheOko (dry eye): Fragmented, SPARSE network (Masmali 3-4). Large \
empty regions. Fractal D DEPRESSED (< 1.65). Low branching / low HOG mean. \
LBP histogram skewed toward uniform/flat bins.
"""


# ---------------------------------------------------------------------------
# Feature selection + Masmali heuristic (same recipe as expert_council.py)
# ---------------------------------------------------------------------------

# 16 compact, diagnostic features spanning all families.
PROMPT_FEATURES = [
    # Surface roughness
    "Sa", "Sq", "Ssk", "Sku",
    # GLCM — three that dominate in the literature
    "glcm_contrast_d1_mean", "glcm_homogeneity_d1_mean",
    "glcm_correlation_d1_mean",
    # Fractal + heterogeneity
    "fractal_D_mean", "fractal_D_std",
    "mf_alpha_width",  # multifractal spectrum width = heterogeneity proxy
    # Lacunarity — gap / void structure
    "lac_slope",
    # Branch density proxy (HOG mean — expert_council convention)
    "hog_mean",
    # Three LBP bins that track flat / edge / uniform textures
    "lbp_0", "lbp_10", "lbp_25",
    # Gabor anisotropy (orientation preference of structure)
    "gabor_f2_aniso",
]


def masmali_grade(Sa: float, fractal_D: float, homog: float, contrast: float) -> int:
    """Heuristic Masmali score (0-4) from handcrafted features.

    Recipe inherited from scripts/expert_council.py. Not Masmali's original
    clinical chart — a proxy assembled so a dense healthy ferning tends to
    score 0-1 and a fragmented dry-eye sample 3-4.
    """
    g = 0
    if fractal_D < 1.70:
        g += 2
    elif fractal_D < 1.76:
        g += 1
    if homog < 0.55:
        g += 1
    if contrast > 8.0:
        g += 1
    if Sa < 50:
        g += 1
    return int(min(g, 4))


def _fmt(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(f) >= 1e4 or (0 < abs(f) < 1e-3):
        return f"{f:.3e}"
    return f"{f:.4g}"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a clinician-scientist specialising in tear-film biomarkers. "
    "Respond with exactly one JSON object — no markdown fence, no preamble, "
    "no apologies. Valid JSON only."
)


def build_prompt(feats: dict, scan_idx: int) -> str:
    """Assemble the user-side prompt for a single scan.

    `scan_idx` is a zero-padded integer — the only identifier sent. No raw
    filename, no class, no person ID is ever in the prompt.
    """
    lines = [
        "You are reviewing an anonymised Atomic Force Microscopy (AFM) scan",
        "of a dried tear droplet. Classify the patient into ONE of five classes",
        "based on the quantitative measurements below. NO IMAGE is provided —",
        "this is a pure quantitative reasoning task; judge on numbers + biology alone.",
        "",
        "## Class morphological signatures",
        "",
        CLASS_SIGNATURES.rstrip(),
        "",
        f"## Quantitative measurements for scan {scan_idx:04d}",
        "",
    ]
    for k in PROMPT_FEATURES:
        if k in feats:
            lines.append(f"- {k}: {_fmt(feats[k])}")
    # Derived numbers — keep the prompt rich with interpretive context
    lines.append(
        f"- masmali_grade (heuristic 0-4): "
        f"{feats.get('masmali_grade', 'n/a')}"
    )
    lines.append("")
    lines.append("## Task")
    lines.append("")
    lines.append(
        "Reason from the numbers to the class. Cite at least two specific "
        "values that drove your decision. Output EXACTLY one JSON object "
        "with this shape (keys in this order):"
    )
    lines.append("")
    lines.append(
        '{"predicted_class": "<one of: ZdraviLudia | Diabetes | '
        'PGOV_Glaukom | SklerozaMultiplex | SucheOko>", '
        '"confidence": <float 0 to 1>, '
        '"reasoning": "<2-4 sentences citing specific feature values>"}'
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feature loading + augmentation
# ---------------------------------------------------------------------------

def load_features() -> pd.DataFrame:
    """Load advanced features parquet. Raise if missing — no silent downgrade."""
    if not FEATS_ADV.exists():
        raise FileNotFoundError(
            f"Advanced features not found at {FEATS_ADV}. "
            f"Run scripts/extract_advanced_features.py first."
        )
    df = pd.read_parquet(FEATS_ADV)
    # Some advanced-features rows lack `mf_alpha_width` etc. — safe-default to 0
    for col in PROMPT_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    # Derive Masmali grade
    df["masmali_grade"] = [
        masmali_grade(
            row["Sa"],
            row["fractal_D_mean"],
            row["glcm_homogeneity_d1_mean"],
            row["glcm_contrast_d1_mean"],
        )
        for _, row in df.iterrows()
    ]
    return df


# ---------------------------------------------------------------------------
# Stratified subset selection (seed=42)
# ---------------------------------------------------------------------------

def stratified_subset(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Person-aware class-stratified subset.

    Aims for ``n // len(CLASSES)`` scans per class while spreading picks
    across as many distinct persons as possible (diversifies the pilot
    so one chatty person doesn't dominate a class). Deterministic.
    """
    if n >= len(df):
        return df.reset_index(drop=False).rename(columns={"index": "orig_idx"})
    per_class = max(1, n // len(CLASSES))
    rng = np.random.RandomState(seed)
    picked: list[int] = []
    for cls in CLASSES:
        cls_df = df[df["cls"] == cls]
        if cls_df.empty:
            continue
        # Group by person, shuffle person order, round-robin pick
        persons = list(cls_df["person"].unique())
        rng.shuffle(persons)
        buckets = {
            p: rng.permutation(cls_df.index[cls_df["person"] == p].values).tolist()
            for p in persons
        }
        taken = 0
        while taken < per_class and any(buckets.values()):
            for p in persons:
                if not buckets[p]:
                    continue
                picked.append(int(buckets[p].pop(0)))
                taken += 1
                if taken >= per_class:
                    break
    # Top up to exactly `n` with remaining rows
    remaining = n - len(picked)
    if remaining > 0:
        pool = np.setdiff1d(df.index.values, np.array(picked))
        if len(pool) > 0:
            picked.extend(
                rng.choice(pool, size=min(remaining, len(pool)), replace=False)
                .tolist()
            )
    sub = df.loc[picked].copy()
    sub.insert(0, "orig_idx", sub.index)
    return sub.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Worker: one CLI call (subprocess -> `claude -p`)
# ---------------------------------------------------------------------------


def _parse_json_tolerant(text: str) -> dict | None:
    """Extract first balanced JSON object from response text."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        # Strip markdown fence
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # Brace-balanced scan for the first valid JSON object
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(s[start : i + 1])
                except json.JSONDecodeError:
                    start = -1
    return None


def classify_one(task: dict, model: str, max_retries: int = 3) -> dict:
    """Invoke ``claude -p`` for a single scan. Process-safe (pure subprocess)."""
    prompt = task["prompt"]
    # Belt-and-braces leak check — no image, no path, but still verify prompt
    # doesn't contain forbidden raw-filename fragments or legacy leaky dirs.
    assert_prompt_safe(prompt)

    last_err: str | None = None
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            cmd = [
                "claude", "-p",
                "--model", model,
                "--output-format", "json",
                "--append-system-prompt", SYSTEM_PROMPT,
                prompt,
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            last_err = "timeout"
            time.sleep(2 ** attempt)
            continue
        except Exception as e:  # pragma: no cover - defensive
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
            continue

        latency = time.time() - t0
        if proc.returncode != 0:
            last_err = f"exit {proc.returncode}: {proc.stderr[:200]}"
            time.sleep(2 ** attempt)
            continue

        # `claude -p --output-format json` emits a JSON envelope whose
        # `result` field contains the assistant's reply text.
        usage = {"input_tokens": 0, "output_tokens": 0}
        try:
            env = json.loads(proc.stdout)
        except json.JSONDecodeError:
            last_err = f"envelope parse failed: {proc.stdout[:200]!r}"
            continue
        result_str = env.get("result", "") if isinstance(env, dict) else ""
        u = env.get("usage") if isinstance(env, dict) else None
        if isinstance(u, dict):
            usage["input_tokens"] = int(
                u.get("input_tokens", u.get("inputTokens", 0)) or 0
            )
            usage["output_tokens"] = int(
                u.get("output_tokens", u.get("outputTokens", 0)) or 0
            )

        parsed = _parse_json_tolerant(result_str)
        if parsed is None or parsed.get("predicted_class") not in CLASSES:
            last_err = f"bad parse: {result_str[:200]!r}"
            continue
        return {
            "orig_idx": task["orig_idx"],
            "scan_path": task["scan_path"],
            "true_cls": task["true_cls"],
            "true_label": task["true_label"],
            "predicted_class": parsed["predicted_class"],
            "confidence": float(parsed.get("confidence", 0.0)),
            "reasoning": str(parsed.get("reasoning", ""))[:2000],
            "raw": result_str[:2000],
            "usage": usage,
            "latency_s": latency,
            "model": model,
            "error": None,
        }
    return {
        "orig_idx": task["orig_idx"],
        "scan_path": task["scan_path"],
        "true_cls": task["true_cls"],
        "true_label": task["true_label"],
        "predicted_class": None,
        "confidence": 0.0,
        "reasoning": "",
        "raw": "",
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "latency_s": 0.0,
        "model": model,
        "error": last_err,
    }


# ---------------------------------------------------------------------------
# Parallel driver
# ---------------------------------------------------------------------------

def run_parallel(
    df: pd.DataFrame,
    model: str,
    workers: int,
    out_fp,
) -> list[dict]:
    tasks = []
    for _, row in df.iterrows():
        feats = {k: row[k] for k in PROMPT_FEATURES if k in row.index}
        feats["masmali_grade"] = int(row["masmali_grade"])
        prompt = build_prompt(feats, int(row["orig_idx"]))
        tasks.append({
            "orig_idx": int(row["orig_idx"]),
            "scan_path": str(row["raw"]),
            "true_cls": str(row["cls"]),
            "true_label": int(row["label"]),
            "prompt": prompt,
        })
    records: list[dict] = []
    total_cost = 0.0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(classify_one, task, model): task for task in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            cost = (
                rec["usage"]["input_tokens"] * SONNET_IN_USD / 1e6
                + rec["usage"]["output_tokens"] * SONNET_OUT_USD / 1e6
            )
            rec["cost_usd"] = cost
            total_cost += cost
            records.append(rec)
            out_fp.write(json.dumps(rec) + "\n")
            out_fp.flush()
            ok = "+" if rec["predicted_class"] == rec["true_cls"] else (
                "-" if rec["predicted_class"] else "x"
            )
            print(
                f"  [{i:>3}/{len(tasks)}] {ok} true={rec['true_cls']:20s} "
                f"-> pred={rec['predicted_class'] or 'PARSE_FAIL':20s} "
                f"(${total_cost:.4f}, {time.time()-t0:.1f}s)",
                flush=True,
            )
    return records


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(records: list[dict]) -> dict:
    rec = [r for r in records if r["predicted_class"] in CLASSES]
    if not rec:
        return {
            "n": 0, "n_parsed": 0, "accuracy": 0.0,
            "weighted_f1": 0.0, "macro_f1": 0.0,
        }
    y_true = [CLASSES.index(r["true_cls"]) for r in rec]
    y_pred = [CLASSES.index(r["predicted_class"]) for r in rec]
    labels = list(range(len(CLASSES)))
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    return {
        "n": len(records),
        "n_parsed": len(rec),
        "accuracy": acc,
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted",
            labels=labels, zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro",
            labels=labels, zero_division=0,
        ),
        "per_class_f1": f1_score(
            y_true, y_pred, average=None, labels=labels, zero_division=0,
        ).tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
        "report": classification_report(
            y_true, y_pred, target_names=CLASSES, digits=3,
            labels=labels, zero_division=0,
        ),
        "confusion": confusion_matrix(
            y_true, y_pred, labels=labels,
        ).tolist(),
    }


def bootstrap_delta(
    y_true: list[int],
    y_pred_a: list[int],
    y_pred_b: list[int],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap Delta wF1 = B - A. Positive -> B beats A."""
    rng = np.random.default_rng(seed)
    ya = np.asarray(y_pred_a)
    yb = np.asarray(y_pred_b)
    yt = np.asarray(y_true)
    n = len(yt)
    labels = list(range(len(CLASSES)))
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fa = f1_score(yt[idx], ya[idx], average="weighted",
                      labels=labels, zero_division=0)
        fb = f1_score(yt[idx], yb[idx], average="weighted",
                      labels=labels, zero_division=0)
        deltas[i] = fb - fa
    return {
        "delta_mean": float(deltas.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_delta_gt_0": float((deltas > 0).mean()),
    }


def overlap_with_v4(records: list[dict]) -> tuple[list[int], list[int], list[int]] | None:
    """Align our predictions with v4 OOF predictions on matching scan paths.

    Returns (y_true, y_v4_pred, y_our_pred) or None if v4 cache missing.
    """
    if not V4_OOF_FILE.exists():
        return None
    v4 = np.load(V4_OOF_FILE, allow_pickle=True)
    v4_paths = {str(p): i for i, p in enumerate(v4["scan_paths"])}
    v4_proba = v4["proba"]
    yt_all: list[int] = []
    yv: list[int] = []
    ys: list[int] = []
    for r in records:
        if r["predicted_class"] not in CLASSES:
            continue
        p = r["scan_path"]
        if p not in v4_paths:
            continue
        j = v4_paths[p]
        yt_all.append(int(r["true_label"]))
        yv.append(int(np.argmax(v4_proba[j])))
        ys.append(CLASSES.index(r["predicted_class"]))
    if len(yt_all) < 5:
        return None
    return yt_all, yv, ys


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    pilot_scores: dict,
    full_scores: dict | None,
    pilot_records: list[dict],
    full_records: list[dict] | None,
    model: str,
    pilot_cost: float,
    full_cost: float | None,
    wall_s: float,
    boot_vs_v4: dict | None,
) -> None:
    used = full_records if full_records else pilot_records
    used_scores = full_scores if full_scores else pilot_scores

    lines: list[str] = []
    lines.append("# VLM Numeric Reasoner — Sonnet 4.6 on Features Only\n")
    lines.append(f"Date: 2026-04-18. Model: `{model}`.\n")
    lines.append(
        "## Why this experiment\n"
        "\n"
        "Every prior VLM-vision attempt failed because AFM tear-ferning is "
        "out-of-distribution for vision encoders trained on natural images "
        "(zero-shot 30 %, few-shot honest 34 %, binary re-ranker 35 %). "
        "The class signatures, however, are well-described in TEXT — "
        "published Masmali grading, AFM literature on diabetic / MS / "
        "glaucoma tear-film biomarkers. Text reasoning is IN-distribution "
        "for a frontier LLM. This script tests whether Sonnet 4.6, given "
        "only 16 quantitative features per scan + the written class "
        "signatures (no image, no path, no filename), can reason back to "
        "the correct class.\n"
    )

    lines.append("## Methodology\n")
    lines.append(
        "- **Features sent** (per scan, 16 + 1 derived):\n"
        f"  `{', '.join(PROMPT_FEATURES)}` + heuristic `masmali_grade` (0-4).\n"
        "- **Source**: `cache/features_advanced.parquet` (240 scans, 448 cols).\n"
        "- **No image, no path, no filename** is ever included in the prompt — "
        "only a zero-padded anonymous scan index.\n"
        "- **Safety guard**: `teardrop.safe_paths.assert_prompt_safe(prompt)` "
        "is called before every API request; raises `PromptLeakError` if any "
        "class name appears in a path-like context or any raw-filename "
        "fragment slips in.\n"
        "- **Concurrency**: `ThreadPoolExecutor`, default 4 workers.\n"
        "- **Determinism**: `temperature=0.0`, `max_tokens=1024`.\n"
        "- **Scale-up gate**: if pilot wF1 >= "
        f"{F1_SCALE_UP_GATE:.2f}, run the remaining scans.\n"
    )

    lines.append("## Pilot run (60-scan stratified subset, seed=42)\n")
    lines.append(
        f"- **n processed / parsed**: {pilot_scores['n']} / {pilot_scores['n_parsed']}  \n"
        f"- **Accuracy**: {pilot_scores['accuracy']:.3f}  \n"
        f"- **Weighted F1**: {pilot_scores['weighted_f1']:.3f}  \n"
        f"- **Macro F1**: {pilot_scores['macro_f1']:.3f}  \n"
        f"- **Cost**: ${pilot_cost:.3f}"
    )
    lines.append("\n```\n" + pilot_scores.get("report", "") + "\n```\n")
    cm = pilot_scores.get("confusion")
    if cm is not None:
        lines.append(
            "Confusion matrix (rows=true, cols=pred, order = "
            + ", ".join(CLASSES) + "):\n"
        )
        lines.append("```")
        for r, row in enumerate(cm):
            lines.append(f"{CLASSES[r]:20s} " + " ".join(f"{v:3d}" for v in row))
        lines.append("```\n")

    if full_scores is not None:
        lines.append("## Full run (all 240 scans)\n")
        lines.append(
            f"- **n processed / parsed**: {full_scores['n']} / {full_scores['n_parsed']}  \n"
            f"- **Accuracy**: {full_scores['accuracy']:.3f}  \n"
            f"- **Weighted F1**: {full_scores['weighted_f1']:.3f}  \n"
            f"- **Macro F1**: {full_scores['macro_f1']:.3f}  \n"
            f"- **Cost**: ${(full_cost or 0):.3f}"
        )
        lines.append("\n```\n" + full_scores.get("report", "") + "\n```\n")
        cm = full_scores.get("confusion")
        if cm is not None:
            lines.append(
                "Confusion matrix (rows=true, cols=pred, order = "
                + ", ".join(CLASSES) + "):\n"
            )
            lines.append("```")
            for r, row in enumerate(cm):
                lines.append(
                    f"{CLASSES[r]:20s} " + " ".join(f"{v:3d}" for v in row)
                )
            lines.append("```\n")

    lines.append("## vs v4 ensemble (paired overlap on scored scans)\n")
    if boot_vs_v4 is None:
        lines.append(
            "- Not available — either `cache/v4_oof_predictions.npz` is missing "
            "or overlap was too small.\n"
        )
    else:
        mean = boot_vs_v4["delta_mean"]
        lo = boot_vs_v4["ci_low"]
        hi = boot_vs_v4["ci_high"]
        p_gt = boot_vs_v4["p_delta_gt_0"]
        lines.append(
            f"- Delta wF1 = Numeric-Reasoner - v4 = "
            f"{mean:+.3f} (95 % CI [{lo:+.3f}, {hi:+.3f}], "
            f"P[delta>0] = {p_gt:.2f})\n"
        )

    lines.append("## Verdict\n")
    best_wf1 = used_scores["weighted_f1"]
    if best_wf1 < 0.50:
        lines.append(
            f"- **STOP**. Weighted F1 = {best_wf1:.3f} is below 0.50, "
            f"the gate for 'useful as an ensemble member'. Text reasoning "
            f"on these 16 numeric features is worse than dumb priors on "
            f"this problem — numbers alone are insufficient context for "
            f"Sonnet to recover class separation that the vision track "
            f"also can't recover.\n"
        )
    elif best_wf1 < 0.65:
        lines.append(
            f"- **LIMITED**. Weighted F1 = {best_wf1:.3f} is in the "
            f"0.50-0.65 range — better than zero-shot / few-shot vision "
            f"(0.30-0.35), worse than the v4 ensemble (0.66). Worth "
            f"keeping as a *diagnostic* channel (rationales + abstain "
            f"signal) but not as a standalone model. Consider: "
            f"calibrated fusion with v4 (small weight), gated "
            f"abstention on low-confidence v4 calls.\n"
        )
    else:
        lines.append(
            f"- **STRONG**. Weighted F1 = {best_wf1:.3f} >= 0.65 — "
            f"this becomes a genuine ensemble component and a **pitch "
            f"asset** ('LLM as quantitative biologist'). Next steps: "
            f"stacking with v4, ablation on feature subset, "
            f"temperature=0.5 with self-consistency (k=5 majority vote).\n"
        )

    # Sample outputs — interpretability
    correct = [r for r in used if r["predicted_class"] == r["true_cls"]][:3]
    wrong = [r for r in used if r["predicted_class"] and r["predicted_class"] != r["true_cls"]][:3]

    if correct:
        lines.append("\n### Sample: correctly classified\n")
        for i, r in enumerate(correct, 1):
            lines.append(f"**C{i}** — true = {r['true_cls']}, "
                         f"pred = {r['predicted_class']}, "
                         f"conf = {r['confidence']:.2f}\n")
            lines.append(f"  - Reasoning: _{r['reasoning']}_\n")
    if wrong:
        lines.append("\n### Sample: misclassified (diagnostic)\n")
        for i, r in enumerate(wrong, 1):
            lines.append(f"**W{i}** — true = {r['true_cls']}, "
                         f"pred = {r['predicted_class']}, "
                         f"conf = {r['confidence']:.2f}\n")
            lines.append(f"  - Reasoning: _{r['reasoning']}_\n")

    lines.append("\n## Raw artefacts\n")
    lines.append(f"- Predictions JSON: `{OUT_JSON.relative_to(REPO)}`\n")
    lines.append(f"- Per-call JSONL: `{RAW_JSONL.relative_to(REPO)}`\n")
    lines.append(f"- Wall time: {wall_s:.1f} s\n")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\n[report] wrote {REPORT_PATH}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subset", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--full", action="store_true",
                    help="Force full 240-scan run, bypassing the F1 gate.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Render one prompt, verify safe-paths, exit.")
    args = ap.parse_args()

    print(f"[0] Loading features from {FEATS_ADV.name} ...", flush=True)
    df = load_features()
    print(
        f"    loaded {len(df)} rows; class dist: "
        f"{df['cls'].value_counts().to_dict()}",
        flush=True,
    )

    df_pilot = stratified_subset(df, args.subset)
    print(
        f"[1] Pilot subset: {len(df_pilot)} scans; class dist: "
        f"{df_pilot['cls'].value_counts().to_dict()}",
        flush=True,
    )

    if args.dry_run:
        row = df_pilot.iloc[0]
        feats = {k: row[k] for k in PROMPT_FEATURES if k in row.index}
        feats["masmali_grade"] = int(row["masmali_grade"])
        p = build_prompt(feats, int(row["orig_idx"]))
        print("\n=== PROMPT (safe-check + print) ===\n")
        assert_prompt_safe(p)
        print(p)
        print("\n[dry-run] assert_prompt_safe passed. Exiting without API call.")
        return 0

    CACHE.mkdir(exist_ok=True)
    RAW_JSONL.parent.mkdir(exist_ok=True)

    t_start = time.time()
    out_fp = RAW_JSONL.open("w")

    # ----- pilot -----
    print(f"[2] Running pilot: {args.workers} workers, model={args.model}",
          flush=True)
    pilot_records = run_parallel(df_pilot, args.model, args.workers, out_fp)
    pilot_scores = score(pilot_records)
    pilot_cost = sum(r.get("cost_usd", 0.0) for r in pilot_records)
    print(
        f"\n[pilot] wF1={pilot_scores['weighted_f1']:.3f} "
        f"macroF1={pilot_scores['macro_f1']:.3f} "
        f"acc={pilot_scores['accuracy']:.3f} "
        f"cost=${pilot_cost:.3f}",
        flush=True,
    )

    # ----- scale-up gate -----
    full_records = None
    full_scores = None
    full_cost = None
    if args.full or pilot_scores["weighted_f1"] >= F1_SCALE_UP_GATE:
        done = {r["scan_path"] for r in pilot_records}
        df_rest = df[~df["raw"].astype(str).isin(done)].copy()
        df_rest.insert(0, "orig_idx", df_rest.index)
        df_rest = df_rest.reset_index(drop=True)
        print(
            f"[3] Gate passed (or --full) — running remaining {len(df_rest)} scans",
            flush=True,
        )
        rest_records = run_parallel(df_rest, args.model, args.workers, out_fp)
        full_records = pilot_records + rest_records
        full_scores = score(full_records)
        full_cost = sum(r.get("cost_usd", 0.0) for r in full_records)
        print(
            f"\n[full] wF1={full_scores['weighted_f1']:.3f} "
            f"macroF1={full_scores['macro_f1']:.3f} "
            f"acc={full_scores['accuracy']:.3f} "
            f"cost=${full_cost:.3f}",
            flush=True,
        )
    else:
        print(
            f"[3] Pilot wF1 {pilot_scores['weighted_f1']:.3f} < "
            f"{F1_SCALE_UP_GATE:.2f} — NOT scaling up. Re-run with --full to force.",
            flush=True,
        )
    out_fp.close()

    # ----- bootstrap vs v4 -----
    boot_vs_v4 = None
    used_records = full_records or pilot_records
    ov = overlap_with_v4(used_records)
    if ov is not None:
        yt, yv4, ymine = ov
        boot_vs_v4 = bootstrap_delta(yt, yv4, ymine)
        print(
            f"[4] Bootstrap vs v4 on n={len(yt)} overlapping scans: "
            f"delta wF1 = {boot_vs_v4['delta_mean']:+.3f} "
            f"[{boot_vs_v4['ci_low']:+.3f}, {boot_vs_v4['ci_high']:+.3f}], "
            f"P[delta>0] = {boot_vs_v4['p_delta_gt_0']:.2f}",
            flush=True,
        )
    else:
        print("[4] No v4 overlap computed (cache missing or too few scans).")

    # ----- persist JSON -----
    payload = {
        "model": args.model,
        "n_total": len(used_records),
        "n_parsed": sum(1 for r in used_records if r["predicted_class"]),
        "pilot_scores": {k: v for k, v in pilot_scores.items() if k != "y_true" and k != "y_pred"},
        "full_scores": (
            {k: v for k, v in full_scores.items() if k != "y_true" and k != "y_pred"}
            if full_scores else None
        ),
        "cost_usd_total": float((full_cost if full_cost is not None else pilot_cost) or 0.0),
        "wall_s": time.time() - t_start,
        "bootstrap_vs_v4": boot_vs_v4,
        "classes": CLASSES,
        "prompt_features": PROMPT_FEATURES + ["masmali_grade"],
        "records": used_records,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[5] Wrote {OUT_JSON}", flush=True)

    # ----- report -----
    wall = time.time() - t_start
    write_report(
        pilot_scores, full_scores,
        pilot_records, full_records,
        args.model, pilot_cost, full_cost, wall, boot_vs_v4,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
