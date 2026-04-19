"""Expert Council — multiple base classifiers + LLM judge → final prediction.

Architecture
------------
For each of 240 AFM tear scans (person-LOPO):
  1. Expert 1 (Vision): Wave-5 v4 multiscale ensemble softmax probabilities
     (pre-computed OOF, loaded from cache/v4_oof_predictions.npz).
  2. Expert 2 (Retrieval): pure k-NN on DINOv2-B scan embeddings. Top-5 nearest
     neighbours with SAME person excluded. Similarity-weighted class vote.
  3. Expert 3 (Morphology): XGBoost on 440 handcrafted features (GLCM multi-scale,
     LBP, fractal D, multifractal, lacunarity, gabor, wavelet packet, HOG,
     Hurst/DFA, surface stats). Person-LOPO OOF predictions computed here.
  4. Quantitative summary: fractal D, Sq (Rq in nm-like units after normalisation),
     Masmali heuristic grade (Daza 2022 surrogate), branch density proxy.
  5. Build a JUDGE prompt including all of the above + class-specific signatures.
     Call Claude Haiku 4.5 via `claude -p` with the obfuscated PNG attached
     (cache/vlm_tiles_honest/scan_XXXX.png) — no class leak in path or text.
  6. Parse JSON response: {predicted_class, confidence, reasoning}.

Parallelism
-----------
ProcessPoolExecutor with N workers (default 8). Each worker spawns one
`claude -p` subprocess.

Outputs
-------
cache/expert_council_predictions.json        — one record per scan
reports/EXPERT_COUNCIL_RESULTS.md            — full analysis

CLI
---
    .venv/bin/python scripts/expert_council.py --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from teardrop.data import CLASSES, person_id  # noqa: E402
from teardrop.safe_paths import assert_prompt_safe  # noqa: E402

CACHE = REPO / "cache"
REPORTS = REPO / "reports"
TILE_DIR = CACHE / "vlm_tiles_honest"
MANIFEST_FILE = CACHE / "vlm_honest_manifest.json"

V4_OOF_FILE = CACHE / "v4_oof_predictions.npz"
DINOV2_EMB_FILE = CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz"
ADVANCED_FEATS_FILE = CACHE / "features_advanced.parquet"

OUT_FILE = CACHE / "expert_council_predictions.json"
REPORT_FILE = REPORTS / "EXPERT_COUNCIL_RESULTS.md"

# ---------------------------------------------------------------------------
# Prompt — emphasise that scan filename is obfuscated (no class leakage).
# ---------------------------------------------------------------------------

CLASS_SIGNATURES = """\
- ZdraviLudia (healthy): Dense, highly-branched dendritic ferning. Fractal D ~ 1.70-1.85. \
Masmali grade 0-1. Moderate GLCM homogeneity and correlation.
- Diabetes: Coarser/thicker branches, higher packing density. Elevated Sq (roughness). \
Fractal D ~ 1.60-1.75. Positive skewness (tall peaks). Higher GLCM contrast.
- PGOV_Glaukom (glaucoma): Granular structure with loops/rings visible — MMP-9 \
degrades the glycoprotein matrix. Shorter/thicker branches, fewer end-points. \
GLCM correlation LOW (locally chaotic texture).
- SklerozaMultiplex (MS): Heterogeneous within-sample morphology. Mixed coarse rods \
and fine granules side by side. High intra-sample variance in GLCM / fractal. \
Often visually confused with PGOV_Glaukom.
- SucheOko (dry eye): Fragmented, SPARSE network. Large empty regions. Fractal D \
DEPRESSED (typically < 1.65). Masmali grade 3-4. Low branch density.
"""

JUDGE_PROMPT_TEMPLATE = """You are an expert AFM tear-ferning pathologist sitting on a consultation panel. An anonymised AFM scan of a dried tear droplet has been reviewed by three independent classifiers. Your job is to weigh their evidence, the quantitative measurements, and your own reading of the image, and output a single final diagnosis.

The image file at {img_path} is the scan under review (afmhot colormap: bright = high, dark = low; 1 px ~= 90 nm, field of view ~= 46 um). The filename is a hash-obfuscated identifier; it carries NO class information.

## Panel members (independent experts)

EXPERT 1 (Vision ensemble, Wave-5 v4 multiscale DINOv2+BiomedCLIP, person-LOPO CV)
{v4_line}
Full softmax: {v4_probs}

EXPERT 2 (k-NN retrieval, top-5 DINOv2 nearest neighbours, SAME person excluded)
Votes: {knn_votes}
Weighted probabilities: {knn_probs}
Top-1 pick: {knn_top1} (weight {knn_top1_w:.2f})

EXPERT 3 (XGBoost on 440 handcrafted morphology features, person-LOPO OOF)
{xgb_line}
Full softmax: {xgb_probs}

## Quantitative morphology of this scan

- Fractal dimension D: {fractal_D:.3f} (std {fractal_D_std:.3f})
- Surface roughness Sq: {Sq:.2f}   Sa: {Sa:.2f}   Skewness Ssk: {Ssk:.2f}
- GLCM contrast d1: {glcm_contrast:.2f}   homogeneity d1: {glcm_homog:.3f}   correlation d1: {glcm_corr:.3f}
- Masmali grade (heuristic surrogate, 0-4): {masmali_grade}
- Branch density proxy (HOG mean): {hog_mean:.3f}
- Lacunarity slope: {lac_slope:.3f}

## Class-specific morphological signatures

{class_signatures}

## Task

Look at the image. Reason step by step through the evidence:
 1. Which experts agree? Where do they disagree?
 2. Do the quantitative measurements support one candidate over another? (fractal D, Masmali, Sq)
 3. Does the image itself show dense ferning, granular loops, heterogeneous mix, sparse fragments, or coarse branches?
 4. Which class best reconciles vision + retrieval + morphology + image inspection?

Then output EXACTLY ONE JSON object (no markdown fence, no extra text):

{{"predicted_class": "<one of: ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko>", "confidence": <float 0 to 1>, "reasoning": "<3-5 sentences citing experts and specific numbers>"}}
"""

SYSTEM_APPEND = (
    "You are a medical AFM-tear classification judge. You MUST read the attached "
    "image file with the Read tool before deciding. Respond with exactly one JSON "
    "object and nothing else — no markdown fence, no preamble."
)

# ---------------------------------------------------------------------------
# Helper: format probability dict compactly
# ---------------------------------------------------------------------------

def _fmt_probs(probs: np.ndarray) -> str:
    parts = []
    for i, c in enumerate(CLASSES):
        parts.append(f"{c}={probs[i]:.3f}")
    return "{" + ", ".join(parts) + "}"


def _top1(probs: np.ndarray) -> tuple[str, float]:
    i = int(np.argmax(probs))
    return CLASSES[i], float(probs[i])


# ---------------------------------------------------------------------------
# Expert 2 — pure k-NN retrieval (person-LOPO safe)
# ---------------------------------------------------------------------------

def knn_expert(
    query_idx: int,
    emb_n: np.ndarray,
    persons: np.ndarray,
    y: np.ndarray,
    k: int = 5,
) -> dict:
    """Return weighted votes + soft probabilities for the query."""
    q = emb_n[query_idx]
    q_person = persons[query_idx]
    sims = emb_n @ q
    # Exclude same person AND self
    mask = (persons != q_person)
    mask[query_idx] = False
    cand = np.where(mask)[0]
    order = cand[np.argsort(-sims[cand])]
    top = order[:k]
    top_sims = sims[top]
    top_y = y[top]
    # Convert cosine sims to non-negative weights (clip to 0+)
    w = np.clip(top_sims, 0.0, None)
    if w.sum() < 1e-8:
        w = np.ones_like(w)
    w = w / w.sum()
    probs = np.zeros(len(CLASSES), dtype=np.float64)
    votes = {c: 0 for c in CLASSES}
    for yi, wi in zip(top_y, w):
        probs[int(yi)] += wi
        votes[CLASSES[int(yi)]] += 1
    return {
        "votes": votes,
        "probs": probs,
        "neighbour_paths_masked": [f"ref_{j+1}" for j in range(len(top))],
        "neighbour_sims": top_sims.tolist(),
        "neighbour_labels": [CLASSES[int(yi)] for yi in top_y],
    }


# ---------------------------------------------------------------------------
# Expert 3 — XGBoost on advanced handcrafted features, person-LOPO OOF
# ---------------------------------------------------------------------------

def xgboost_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    from xgboost import XGBClassifier

    oof = np.zeros((X.shape[0], len(CLASSES)), dtype=np.float64)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=random_state)
    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups=groups)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            objective="multi:softprob",
            num_class=len(CLASSES),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=2,
            verbosity=0,
        )
        # class weights via sample_weight to handle imbalance
        from sklearn.utils.class_weight import compute_sample_weight
        sw = compute_sample_weight("balanced", y[tr])
        model.fit(Xtr, y[tr], sample_weight=sw)
        oof[te] = model.predict_proba(Xte)
        print(f"    XGB fold {fold+1}/{n_splits}: train={len(tr)} val={len(te)}")
    return oof


# ---------------------------------------------------------------------------
# Masmali heuristic
# ---------------------------------------------------------------------------

def masmali_grade(Sa: float, fractal_D: float, homog: float, contrast: float) -> int:
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


# ---------------------------------------------------------------------------
# Worker — call Claude Haiku via `claude -p`, parse JSON
# ---------------------------------------------------------------------------

def _tolerant_json_parse(text: str) -> dict | None:
    if not text:
        return None
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(stripped[start:end + 1])
    except json.JSONDecodeError:
        pass
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


def _classify_one(task: dict) -> dict:
    """Worker entry: call claude CLI, parse response. Pure function — picklable."""
    prompt = task["prompt"]
    model = task["model"]
    img_path = task["img_path"]
    assert_prompt_safe(prompt)
    t0 = time.time()
    cmd = [
        "claude", "-p",
        "--model", model,
        "--output-format", "json",
        "--tools", "Read",
        "--append-system-prompt", SYSTEM_APPEND,
        prompt,
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=task.get("timeout_s", 120),
        )
    except subprocess.TimeoutExpired:
        return {
            **task,
            "cli_ok": False,
            "latency_s": time.time() - t0,
            "cli_raw": "",
            "parsed": None,
            "error": "TIMEOUT",
        }
    latency = time.time() - t0
    if proc.returncode != 0:
        return {
            **task,
            "cli_ok": False,
            "latency_s": latency,
            "cli_raw": proc.stdout or "",
            "parsed": None,
            "error": f"exit {proc.returncode}: {(proc.stderr or '')[:200]}",
        }
    # CLI returns JSON envelope with 'result' string
    raw = proc.stdout or ""
    result_str = raw
    try:
        env = json.loads(raw)
        result_str = env.get("result", raw)
    except json.JSONDecodeError:
        pass
    parsed = _tolerant_json_parse(result_str)
    return {
        **task,
        "cli_ok": parsed is not None,
        "latency_s": latency,
        "cli_raw": result_str,
        "parsed": parsed,
        "error": None if parsed else "PARSE_FAIL",
    }


# ---------------------------------------------------------------------------
# Metrics + bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true, y_pred_a, y_pred_b, n_boot=1000, seed=42):
    """Return (delta_mean, delta_ci_low, delta_ci_high, p_delta_gt_0)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_a = np.asarray(y_pred_a)
    y_b = np.asarray(y_pred_b)
    n = len(y_true)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1a = f1_score(y_true[idx], y_a[idx], average="weighted",
                       labels=list(range(len(CLASSES))), zero_division=0)
        f1b = f1_score(y_true[idx], y_b[idx], average="weighted",
                       labels=list(range(len(CLASSES))), zero_division=0)
        deltas[i] = f1b - f1a
    mean = float(deltas.mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_gt = float((deltas > 0).mean())
    return mean, float(lo), float(hi), p_gt


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--model", default="claude-haiku-4-5")
    p.add_argument("--timeout-s", type=int, default=120)
    p.add_argument("--k-neighbours", type=int, default=5)
    p.add_argument("--limit", type=int, default=0,
                   help="Optional cap on #scans processed (0 = all 240).")
    p.add_argument("--resume", action="store_true",
                   help="Reuse existing predictions in OUT_FILE.")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip LLM calls and just run baseline experts (for testing).")
    args = p.parse_args()

    # ---------------------- 1. Verify CLI + load caches ----------------------
    if not args.no_llm:
        print("[0] Pinging claude CLI ...")
        ping = subprocess.run(
            ["claude", "-p", "Reply with PONG and nothing else.",
             "--model", args.model, "--output-format", "json"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
        )
        if ping.returncode != 0:
            print(f"    [fatal] CLI ping failed: {ping.stderr[:200]}")
            return 2
        print(f"    OK ({ping.stdout[:80]!r})")

    print("[1] Loading v4 OOF ensemble predictions ...")
    v4 = np.load(V4_OOF_FILE, allow_pickle=True)
    v4_proba = v4["proba"]  # (240, 5)
    y = v4["y"].astype(np.int64)
    scan_paths = np.array([str(s) for s in v4["scan_paths"]])
    persons = np.array([person_id(Path(p)) for p in scan_paths])
    N = len(scan_paths)
    print(f"    N={N}, persons={len(np.unique(persons))}")

    print("[2] Loading DINOv2-B TTA scan embeddings ...")
    emb = np.load(DINOV2_EMB_FILE, allow_pickle=True)
    X_emb = emb["X_scan"].astype(np.float64)
    emb_paths = np.array([str(s) for s in emb["scan_paths"]])
    # Re-index to match v4 scan_paths order
    emb_idx = {p: i for i, p in enumerate(emb_paths)}
    X_emb = X_emb[[emb_idx[p] for p in scan_paths]]
    # L2-normalise for cosine similarity
    norms = np.linalg.norm(X_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_n = X_emb / norms
    print(f"    emb shape={X_emb.shape}")

    print("[3] Loading handcrafted features + running XGBoost person-LOPO OOF ...")
    df_feat = pd.read_parquet(ADVANCED_FEATS_FILE).set_index("raw")
    df_feat = df_feat.loc[scan_paths]  # align
    # Pick numeric feature columns (drop label/person/patient columns)
    drop_cols = {"cls", "label", "patient", "person"}
    feat_cols = [c for c in df_feat.columns if c not in drop_cols]
    X_feat = df_feat[feat_cols].to_numpy(dtype=np.float64)
    # Drop any NaN/Inf
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"    features: {X_feat.shape}")
    xgb_oof = xgboost_oof(X_feat, y, persons, n_splits=5, random_state=42)

    print("[4] Loading obfuscated manifest + tile paths ...")
    manifest = json.loads(MANIFEST_FILE.read_text())
    path_to_key = {v["raw_path"]: k for k, v in manifest.items()}
    assert all(p in path_to_key for p in scan_paths), "manifest missing paths"

    # Per-scan quick-lookups
    feats_by_idx = {}
    for i in range(N):
        row = df_feat.iloc[i]
        feats_by_idx[i] = {
            "Sa": float(row.get("Sa", 0.0)),
            "Sq": float(row.get("Sq", 0.0)),
            "Ssk": float(row.get("Ssk", 0.0)),
            "fractal_D_mean": float(row.get("fractal_D_mean", 0.0)),
            "fractal_D_std": float(row.get("fractal_D_std", 0.0)),
            "glcm_contrast_d1_mean": float(row.get("glcm_contrast_d1_mean", 0.0)),
            "glcm_homogeneity_d1_mean": float(row.get("glcm_homogeneity_d1_mean", 0.0)),
            "glcm_correlation_d1_mean": float(row.get("glcm_correlation_d1_mean", 0.0)),
            "hog_mean": float(row.get("hog_mean", 0.0)),
            "lac_slope": float(row.get("lac_slope", 0.0)),
        }

    # ---------------------- 5. Build per-scan records ----------------------
    cache_records: dict[str, dict] = {}
    if args.resume and OUT_FILE.exists():
        cache_records = json.loads(OUT_FILE.read_text())
        print(f"[5] Resume: {len(cache_records)} cached records")

    tasks = []
    records_by_key: dict[str, dict] = {}

    for i in range(N):
        scan_path = scan_paths[i]
        key = path_to_key[scan_path]
        img_path = TILE_DIR / f"{key}.png"
        if not img_path.exists():
            print(f"    [warn] missing tile for {key}; skipping")
            continue

        # Expert 1 — v4 softmax
        v4_top1_cls, v4_top1_p = _top1(v4_proba[i])

        # Expert 2 — k-NN
        knn = knn_expert(i, emb_n, persons, y, k=args.k_neighbours)
        knn_top1_idx = int(np.argmax(knn["probs"]))
        knn_top1_cls = CLASSES[knn_top1_idx]
        knn_top1_w = float(knn["probs"][knn_top1_idx])

        # Expert 3 — XGBoost
        xgb_top1_cls, xgb_top1_p = _top1(xgb_oof[i])

        # Quantitative morphology
        f = feats_by_idx[i]
        masm = masmali_grade(
            f["Sa"], f["fractal_D_mean"],
            f["glcm_homogeneity_d1_mean"], f["glcm_contrast_d1_mean"],
        )

        rec = {
            "scan_key": key,
            "scan_path_hash": hashlib.sha256(scan_path.encode()).hexdigest()[:12],
            "true_class": CLASSES[int(y[i])],
            "person": persons[i],
            "expert1_v4": {
                "top1": v4_top1_cls, "top1_p": v4_top1_p,
                "probs": {c: float(v4_proba[i, j]) for j, c in enumerate(CLASSES)},
            },
            "expert2_knn": {
                "top1": knn_top1_cls, "top1_w": knn_top1_w,
                "votes": knn["votes"],
                "probs": {c: float(knn["probs"][j]) for j, c in enumerate(CLASSES)},
                "neighbour_labels": knn["neighbour_labels"],
                "neighbour_sims": knn["neighbour_sims"],
            },
            "expert3_xgb": {
                "top1": xgb_top1_cls, "top1_p": xgb_top1_p,
                "probs": {c: float(xgb_oof[i, j]) for j, c in enumerate(CLASSES)},
            },
            "morphology": {
                "fractal_D_mean": f["fractal_D_mean"],
                "fractal_D_std": f["fractal_D_std"],
                "Sq": f["Sq"], "Sa": f["Sa"], "Ssk": f["Ssk"],
                "glcm_contrast_d1": f["glcm_contrast_d1_mean"],
                "glcm_homogeneity_d1": f["glcm_homogeneity_d1_mean"],
                "glcm_correlation_d1": f["glcm_correlation_d1_mean"],
                "hog_mean": f["hog_mean"],
                "lac_slope": f["lac_slope"],
                "masmali_grade": masm,
            },
        }

        # Build judge prompt
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            img_path=str(img_path),
            v4_line=f"Top-1 pick: {v4_top1_cls} (p={v4_top1_p:.3f})",
            v4_probs=_fmt_probs(v4_proba[i]),
            knn_votes=json.dumps(knn["votes"]),
            knn_probs=_fmt_probs(knn["probs"]),
            knn_top1=knn_top1_cls, knn_top1_w=knn_top1_w,
            xgb_line=f"Top-1 pick: {xgb_top1_cls} (p={xgb_top1_p:.3f})",
            xgb_probs=_fmt_probs(xgb_oof[i]),
            fractal_D=f["fractal_D_mean"], fractal_D_std=f["fractal_D_std"],
            Sq=f["Sq"], Sa=f["Sa"], Ssk=f["Ssk"],
            glcm_contrast=f["glcm_contrast_d1_mean"],
            glcm_homog=f["glcm_homogeneity_d1_mean"],
            glcm_corr=f["glcm_correlation_d1_mean"],
            masmali_grade=masm,
            hog_mean=f["hog_mean"],
            lac_slope=f["lac_slope"],
            class_signatures=CLASS_SIGNATURES,
        )
        rec["prompt_chars"] = len(prompt)
        records_by_key[key] = rec

        if args.resume and key in cache_records:
            prev = cache_records[key].get("judge", {}) or {}
            # Only reuse REAL LLM predictions (cli_ok=True). v4-fallbacks (NO_LLM) are re-run.
            if prev.get("cli_ok") and prev.get("predicted_class"):
                rec["judge"] = prev
                continue

        tasks.append({
            "scan_key": key,
            "img_path": str(img_path),
            "model": args.model,
            "prompt": prompt,
            "timeout_s": args.timeout_s,
        })

    if args.limit > 0:
        tasks = tasks[:args.limit]
        print(f"    [limit] processing only {len(tasks)} scans")

    # ---------------------- 6. Dispatch LLM calls ----------------------
    print(f"[6] Dispatching {len(tasks)} judge calls ({args.workers} workers, model={args.model}) ...")
    t_start = time.time()
    if not args.no_llm and tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_classify_one, t) for t in tasks]
            for done_count, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                key = r["scan_key"]
                rec = records_by_key[key]
                parsed = r["parsed"] or {}
                # Normalise class name
                cls = parsed.get("predicted_class", "")
                cls_n = next((c for c in CLASSES if c.lower() == str(cls).strip().lower()), None)
                rec["judge"] = {
                    "predicted_class": cls_n,
                    "confidence": parsed.get("confidence"),
                    "reasoning": (parsed.get("reasoning") or "")[:800],
                    "raw_cls_string": cls,
                    "cli_ok": r["cli_ok"],
                    "latency_s": r["latency_s"],
                    "error": r.get("error"),
                }
                # Live write every 10 records
                if done_count % 10 == 0 or done_count == len(futures):
                    OUT_FILE.write_text(json.dumps(records_by_key, indent=2))
                    elapsed = time.time() - t_start
                    print(f"    [{done_count}/{len(futures)}] elapsed={elapsed:.0f}s  "
                          f"key={key} true={rec['true_class']:18s} judge={cls_n}")
    t_total = time.time() - t_start
    print(f"[6] Wall time: {t_total:.1f}s")

    # Ensure every rec has a judge field (fallback to v4 top1 if LLM unavailable)
    for key, rec in records_by_key.items():
        if "judge" not in rec:
            rec["judge"] = {
                "predicted_class": rec["expert1_v4"]["top1"],
                "confidence": rec["expert1_v4"]["top1_p"],
                "reasoning": "[fallback] v4 top-1 used (judge not called)",
                "raw_cls_string": rec["expert1_v4"]["top1"],
                "cli_ok": False,
                "latency_s": 0.0,
                "error": "NO_LLM",
            }
        elif rec["judge"].get("predicted_class") is None:
            # LLM failed to parse — fallback to v4
            rec["judge"]["fallback_used"] = True
            rec["judge"]["predicted_class"] = rec["expert1_v4"]["top1"]

    OUT_FILE.write_text(json.dumps(records_by_key, indent=2))
    print(f"[7] Wrote {OUT_FILE}")

    # ---------------------- 8. Metrics + bootstrap ----------------------
    print("[8] Computing metrics ...")
    y_true = np.array([CLASSES.index(records_by_key[path_to_key[p]]["true_class"])
                       for p in scan_paths], dtype=np.int64)
    judge_pred = np.array([CLASSES.index(records_by_key[path_to_key[p]]["judge"]["predicted_class"])
                           for p in scan_paths], dtype=np.int64)
    v4_pred = np.argmax(v4_proba, axis=1)
    knn_pred_arr = []
    xgb_pred_arr = []
    for i, p in enumerate(scan_paths):
        rec = records_by_key[path_to_key[p]]
        knn_pred_arr.append(CLASSES.index(rec["expert2_knn"]["top1"]))
        xgb_pred_arr.append(CLASSES.index(rec["expert3_xgb"]["top1"]))
    knn_pred = np.array(knn_pred_arr, dtype=np.int64)
    xgb_pred = np.array(xgb_pred_arr, dtype=np.int64)

    def f1_pair(a, b):
        return (f1_score(a, b, average="weighted",
                         labels=list(range(len(CLASSES))), zero_division=0),
                f1_score(a, b, average="macro",
                         labels=list(range(len(CLASSES))), zero_division=0))

    f1w_v4, f1m_v4 = f1_pair(y_true, v4_pred)
    f1w_knn, f1m_knn = f1_pair(y_true, knn_pred)
    f1w_xgb, f1m_xgb = f1_pair(y_true, xgb_pred)
    f1w_jud, f1m_jud = f1_pair(y_true, judge_pred)

    print(f"    Expert 1 (v4): w={f1w_v4:.4f} m={f1m_v4:.4f}")
    print(f"    Expert 2 (knn): w={f1w_knn:.4f} m={f1m_knn:.4f}")
    print(f"    Expert 3 (xgb): w={f1w_xgb:.4f} m={f1m_xgb:.4f}")
    print(f"    Judge        : w={f1w_jud:.4f} m={f1m_jud:.4f}")

    print("[9] Bootstrap 1000x vs v4 ...")
    dmean, dlo, dhi, p_gt = bootstrap_ci(y_true, v4_pred, judge_pred,
                                         n_boot=1000, seed=42)
    print(f"    Delta F1w (judge - v4): mean={dmean:+.4f}, "
          f"95% CI=[{dlo:+.4f}, {dhi:+.4f}], P(Delta>0)={p_gt:.3f}")

    # ---------------------- 10. Error analysis ----------------------
    # Cases where judge disagrees with ALL THREE experts
    disagree_all = []
    # Cases where judge corrected a wrong majority
    correct_flip = []
    # Cases where judge picked wrong class despite majority right
    bad_flip = []

    for i, p in enumerate(scan_paths):
        rec = records_by_key[path_to_key[p]]
        e1 = rec["expert1_v4"]["top1"]
        e2 = rec["expert2_knn"]["top1"]
        e3 = rec["expert3_xgb"]["top1"]
        j = rec["judge"]["predicted_class"]
        truth = rec["true_class"]
        experts = [e1, e2, e3]
        counts = {}
        for c in experts:
            counts[c] = counts.get(c, 0) + 1
        majority = max(counts, key=counts.get) if max(counts.values()) >= 2 else None

        if j not in experts:
            disagree_all.append({
                "key": rec["scan_key"], "true": truth,
                "e1": e1, "e2": e2, "e3": e3, "judge": j,
                "judge_correct": j == truth,
                "reasoning": rec["judge"].get("reasoning", "")[:300],
            })
        if majority is not None and majority != truth and j == truth:
            correct_flip.append({
                "key": rec["scan_key"], "true": truth,
                "majority": majority, "judge": j,
                "reasoning": rec["judge"].get("reasoning", "")[:300],
            })
        if majority is not None and majority == truth and j != truth:
            bad_flip.append({
                "key": rec["scan_key"], "true": truth,
                "majority": majority, "judge": j,
                "reasoning": rec["judge"].get("reasoning", "")[:300],
            })

    print(f"    judge disagrees with all 3 experts: {len(disagree_all)} "
          f"({sum(1 for d in disagree_all if d['judge_correct'])} correct)")
    print(f"    judge corrected wrong majority: {len(correct_flip)}")
    print(f"    judge overrode correct majority (bad): {len(bad_flip)}")

    # Per-person F1
    persons_u = np.unique(persons)
    per_person_f1w = []
    for pn in persons_u:
        mask = persons == pn
        if mask.sum() < 1:
            continue
        f1p = f1_score(y_true[mask], judge_pred[mask], average="weighted",
                       labels=list(range(len(CLASSES))), zero_division=0)
        per_person_f1w.append(f1p)

    # Confusion matrix
    cm_judge = confusion_matrix(y_true, judge_pred,
                                 labels=list(range(len(CLASSES))))

    # Per-class F1
    from sklearn.metrics import f1_score as f1s
    per_class_f1 = f1s(y_true, judge_pred, average=None,
                       labels=list(range(len(CLASSES))), zero_division=0)

    # ---------------------- 11. Write report ----------------------
    write_report(
        REPORT_FILE,
        args=args,
        n_scans=N,
        n_judge_calls=len([r for r in records_by_key.values()
                           if r["judge"].get("cli_ok")]),
        t_total=t_total,
        metrics={
            "f1w_v4": f1w_v4, "f1m_v4": f1m_v4,
            "f1w_knn": f1w_knn, "f1m_knn": f1m_knn,
            "f1w_xgb": f1w_xgb, "f1m_xgb": f1m_xgb,
            "f1w_judge": f1w_jud, "f1m_judge": f1m_jud,
        },
        bootstrap={"dmean": dmean, "dlo": dlo, "dhi": dhi, "p_gt": p_gt},
        y_true=y_true, judge_pred=judge_pred,
        cm_judge=cm_judge,
        per_class_f1=per_class_f1,
        per_person_f1=per_person_f1w,
        disagree_all=disagree_all,
        correct_flip=correct_flip,
        bad_flip=bad_flip,
        records=records_by_key,
    )
    print(f"[11] Wrote {REPORT_FILE}")
    return 0


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    path: Path,
    *,
    args,
    n_scans: int,
    n_judge_calls: int,
    t_total: float,
    metrics: dict,
    bootstrap: dict,
    y_true: np.ndarray,
    judge_pred: np.ndarray,
    cm_judge: np.ndarray,
    per_class_f1: np.ndarray,
    per_person_f1: list,
    disagree_all: list,
    correct_flip: list,
    bad_flip: list,
    records: dict,
):
    L = []
    L.append("# Expert Council — multi-classifier + LLM judge")
    L.append("")
    L.append(f"Run: {time.strftime('%Y-%m-%d %H:%M')}   Model: `{args.model}`   "
             f"Workers: {args.workers}")
    L.append("")
    L.append("## 1. Architecture")
    L.append("")
    L.append("Each of the 240 AFM tear scans is processed person-LOPO by three")
    L.append("independent experts plus a quantitative morphology summary. The LLM")
    L.append("judge receives all of the above along with the anonymised query image")
    L.append("and outputs a single predicted class.")
    L.append("")
    L.append("| Expert | Signal | Training | Output |")
    L.append("|---|---|---|---|")
    L.append("| 1. Vision ensemble (v4) | DINOv2-B TTA @ 90 nm + DINOv2-B @ 45 nm + BiomedCLIP-TTA, L2-normed, geomean of softmaxes | person-LOPO 5-fold CV | 5-way softmax |")
    L.append("| 2. k-NN retrieval | DINOv2-B scan embeddings, cosine | top-5 nearest w/ SAME person excluded | similarity-weighted votes |")
    L.append(f"| 3. XGBoost morphology | 440 handcrafted features (GLCM×4 scales, LBP, fractal D, multifractal, lacunarity, gabor, wavelet packet, HOG, Hurst/DFA, surface stats) | person-LOPO StratifiedGroupKFold(5) | 5-way softmax |")
    L.append("| 4. Morphology panel | fractal D, Sq, Ssk, GLCM d1, Masmali grade heuristic, HOG mean, lacunarity slope | — | quantitative summary in prompt |")
    L.append("| Judge | Claude Haiku 4.5 (CLI, no API key) | — | JSON {predicted_class, confidence, reasoning} |")
    L.append("")
    L.append("Key safeguards:")
    L.append("- Images referenced ONLY by `scan_XXXX.png` (obfuscated); true class")
    L.append("  never appears in the prompt or path.")
    L.append("- k-NN neighbour set explicitly excludes scans from the query's person")
    L.append("  (person-LOPO retrieval).")
    L.append("- XGBoost OOF uses `StratifiedGroupKFold(group=person_id)` — no")
    L.append("  same-person leakage.")
    L.append("- v4 softmaxes are pre-computed OOF; each scan's probability comes from")
    L.append("  the fold where its person was the held-out group.")
    L.append("")
    L.append(f"## 2. Run statistics")
    L.append("")
    L.append(f"- Total scans: **{n_scans}**")
    L.append(f"- Judge calls that returned parseable JSON: **{n_judge_calls}**")
    L.append(f"- Wall time (LLM): **{t_total:.1f}s**")
    L.append(f"- Estimated API cost (Haiku 4.5, ~3.0k input + 200 out / call, "
             f"$1/$5 per MTok): "
             f"**~${n_judge_calls * (3000 * 1.0 + 200 * 5.0) / 1e6:.3f}**")
    L.append("")
    L.append("## 3. Headline F1 comparison")
    L.append("")
    L.append("| Expert | Weighted F1 | Macro F1 |")
    L.append("|---|---:|---:|")
    L.append(f"| Expert 1 — v4 vision ensemble | {metrics['f1w_v4']:.4f} | {metrics['f1m_v4']:.4f} |")
    L.append(f"| Expert 2 — k-NN retrieval | {metrics['f1w_knn']:.4f} | {metrics['f1m_knn']:.4f} |")
    L.append(f"| Expert 3 — XGBoost morphology | {metrics['f1w_xgb']:.4f} | {metrics['f1m_xgb']:.4f} |")
    L.append(f"| **Judge (Haiku 4.5)** | **{metrics['f1w_judge']:.4f}** | **{metrics['f1m_judge']:.4f}** |")
    L.append(f"| Delta (judge - v4) | {metrics['f1w_judge'] - metrics['f1w_v4']:+.4f} | {metrics['f1m_judge'] - metrics['f1m_v4']:+.4f} |")
    L.append("")
    L.append(f"Person-LOPO reported F1 for v4 baseline was 0.6887 (doc). On this")
    L.append(f"run, v4 weighted F1 = {metrics['f1w_v4']:.4f} (sanity check).")
    L.append("")
    L.append("## 4. Bootstrap 1000x vs v4")
    L.append("")
    L.append(f"Paired bootstrap on weighted F1 (resample 240 scans w/ replacement):")
    L.append("")
    L.append(f"- Mean Delta F1w (judge - v4) = **{bootstrap['dmean']:+.4f}**")
    L.append(f"- 95% CI: [{bootstrap['dlo']:+.4f}, {bootstrap['dhi']:+.4f}]")
    L.append(f"- P(Delta > 0) = **{bootstrap['p_gt']:.3f}**")
    L.append("")
    if bootstrap["p_gt"] > 0.95:
        L.append("**Verdict:** judge clearly beats v4 (P > 0.95). Ship it.")
    elif bootstrap["p_gt"] > 0.80:
        L.append("**Verdict:** judge probably beats v4. Worth shipping as secondary head "
                 "/ audit layer.")
    elif bootstrap["p_gt"] < 0.20:
        L.append("**Verdict:** judge probably hurts F1. Keep for reasoning/audit only; "
                 "do NOT override vision ensemble.")
    else:
        L.append("**Verdict:** ambiguous — F1 is roughly unchanged, but reasoning "
                 "adds interpretability value even when accuracy is flat.")
    L.append("")
    L.append("## 5. Per-class F1 (Judge)")
    L.append("")
    L.append("| Class | F1 |")
    L.append("|---|---:|")
    for i, c in enumerate(CLASSES):
        L.append(f"| {c} | {per_class_f1[i]:.4f} |")
    L.append("")
    L.append("```")
    L.append(classification_report(y_true, judge_pred,
                                   target_names=CLASSES, digits=3,
                                   zero_division=0))
    L.append("```")
    L.append("")
    L.append("## 6. Confusion matrix (Judge)")
    L.append("")
    header = "| true \\ pred |" + "".join(f" {c} |" for c in CLASSES)
    L.append(header)
    L.append("|---|" + "---:|" * len(CLASSES))
    for i, c in enumerate(CLASSES):
        L.append(f"| {c} | " + " | ".join(str(int(v)) for v in cm_judge[i]) + " |")
    L.append("")
    L.append("## 7. Per-person F1 (Judge, weighted)")
    L.append("")
    per_person_f1 = list(per_person_f1)
    if per_person_f1:
        arr = np.array(per_person_f1)
        L.append(f"- Mean: {arr.mean():.4f}   Median: {np.median(arr):.4f}   "
                 f"Min: {arr.min():.4f}   Max: {arr.max():.4f}")
        L.append(f"- Fraction of persons with F1 >= 0.8: "
                 f"{(arr >= 0.8).mean():.2%}")
        L.append(f"- Fraction of persons with F1 == 0.0: "
                 f"{(arr == 0.0).mean():.2%}")
    L.append("")
    L.append("## 8. Error analysis")
    L.append("")
    L.append(f"### 8a. Judge disagrees with ALL three experts ({len(disagree_all)} cases)")
    L.append("")
    L.append(f"Of these, **{sum(1 for d in disagree_all if d['judge_correct'])}** "
             f"are judge-correct (rescue) and "
             f"**{sum(1 for d in disagree_all if not d['judge_correct'])}** "
             f"are judge-wrong (overrule experts unsuccessfully).")
    L.append("")
    if disagree_all:
        L.append("| key | true | v4 | kNN | XGB | judge | correct? | reasoning |")
        L.append("|---|---|---|---|---|---|:-:|---|")
        for d in disagree_all[:15]:
            mark = "OK" if d["judge_correct"] else "NO"
            L.append(f"| {d['key']} | {d['true']} | {d['e1']} | {d['e2']} | "
                     f"{d['e3']} | {d['judge']} | {mark} | "
                     f"{d['reasoning'][:180].replace('|', '/')} |")
    L.append("")
    L.append(f"### 8b. Judge CORRECTED a wrong majority ({len(correct_flip)} cases)")
    L.append("")
    if correct_flip:
        L.append("Cases where >=2 of 3 experts agreed on a wrong answer but the judge")
        L.append("pulled the prediction back to the true class.")
        L.append("")
        L.append("| key | true | majority-wrong | judge | reasoning |")
        L.append("|---|---|---|---|---|")
        for d in correct_flip[:10]:
            L.append(f"| {d['key']} | {d['true']} | {d['majority']} | {d['judge']} | "
                     f"{d['reasoning'][:200].replace('|', '/')} |")
    else:
        L.append("_(none)_")
    L.append("")
    L.append(f"### 8c. Judge OVERRULED a correct majority ({len(bad_flip)} cases)")
    L.append("")
    if bad_flip:
        L.append("Cases where >=2 of 3 experts agreed on the correct answer but the")
        L.append("judge flipped to a wrong class — these are the failure mode.")
        L.append("")
        L.append("| key | true | majority-right | judge | reasoning |")
        L.append("|---|---|---|---|---|")
        for d in bad_flip[:10]:
            L.append(f"| {d['key']} | {d['true']} | {d['majority']} | {d['judge']} | "
                     f"{d['reasoning'][:200].replace('|', '/')} |")
    else:
        L.append("_(none)_")
    L.append("")
    L.append("## 9. Files written")
    L.append("")
    L.append(f"- `cache/expert_council_predictions.json` — one record per scan")
    L.append(f"  (all expert outputs, morphology, judge response, reasoning).")
    L.append(f"- `reports/EXPERT_COUNCIL_RESULTS.md` — this file.")
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
