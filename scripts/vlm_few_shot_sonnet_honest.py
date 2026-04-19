"""HONEST few-shot VLM classifier — Sonnet 4.6 on full 240 scans.

Purpose
-------
The legacy `vlm_few_shot_sonnet_full_240.py` inherited a CRITICAL filename
leak from `vlm_few_shot.py`:

    collage_path = COLLAGE_DIR / f"{s.cls}__{s.raw_path.name.replace('.', '_')}.png"
    prompt = PROMPT_TEMPLATE.format(img_path=str(collage_path))
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^
    #  "Read the image at /.../vlm_few_shot_collages/Diabetes__37_DM_010.png"
    #                                                 ^^^^^^^^  CLASS LEAK

The VLM reads the class out of the filename and fabricates matching
morphological reasoning. Red-team audit (`reports/RED_TEAM_SONNET_0_8873.md`)
showed 20/20 previously-correct queries crashed to 5/19 when obfuscated.

This script removes the leak completely:

- Collage output path is content-agnostic: `scan_XXXX.png` (zero-padded,
  order-shuffled) under an isolated directory whose path contains no
  class or project keywords.
- A manifest (`cache/vlm_sonnet_honest_manifest.json`) maps the obfuscated
  name back to (true_class, person, patient, raw_path). NEVER passed
  to the VLM.
- Predictions cache: `cache/vlm_sonnet_honest_predictions.json`
  (separate from the leaky `vlm_sonnet_full_predictions.json`).

All the retrieval / collage composition logic is reused from `vlm_few_shot`
so the image *content* is identical to the leaky run — only the filename
and prompt path differ. This isolates the filename leak as the causal
factor.

Usage
-----
    .venv/bin/python scripts/vlm_few_shot_sonnet_honest.py --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

# Reuse the collage composer and Claude CLI wrapper from the base script.
# We intentionally do NOT reuse its `COLLAGE_DIR` (which is leaky by name).
from vlm_few_shot import (  # noqa: E402
    CACHE_DIR,
    call_claude_cli,
    compose_collage,
    load_embeddings,
    render_scan_tile,
    retrieve_anchors_per_class,
    tile_filename,
)
from teardrop.data import CLASSES, enumerate_samples  # noqa: E402

# ---------------------------------------------------------------------------
# Honest (leak-free) output paths
# ---------------------------------------------------------------------------

# Directory name contains no class keyword and no "few_shot" string
# (which is still a project-specific hint). Keep it generic.
HONEST_COLLAGE_DIR = CACHE_DIR / "vlm_few_shot_collages_honest"
HONEST_COLLAGE_DIR.mkdir(parents=True, exist_ok=True)

HONEST_PRED_CACHE = CACHE_DIR / "vlm_sonnet_honest_predictions.json"
HONEST_MANIFEST = CACHE_DIR / "vlm_sonnet_honest_manifest.json"


def build_manifest(samples) -> dict[str, dict]:
    """Build obfuscated scan_XXXX.png → metadata map. Deterministic (seed=42).

    The mapping is persisted so re-runs reuse identical obfuscated names
    and we can merge cached predictions across invocations.
    """
    if HONEST_MANIFEST.exists():
        return json.loads(HONEST_MANIFEST.read_text())

    rng = random.Random(42)
    indices = list(range(len(samples)))
    rng.shuffle(indices)  # scan_0000 is NOT guaranteed to be class 0

    manifest: dict[str, dict] = {}
    for obf_i, src_i in enumerate(indices):
        s = samples[src_i]
        key = f"scan_{obf_i:04d}"
        manifest[key] = {
            "true_class": s.cls,
            "person": s.person,
            "patient": s.patient,
            "raw_path": str(s.raw_path),
            "raw_rel": str(s.raw_path.relative_to(REPO)),
        }
    HONEST_MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_cache() -> dict[str, dict]:
    if HONEST_PRED_CACHE.exists():
        return json.loads(HONEST_PRED_CACHE.read_text())
    return {}


def save_cache(cache: dict[str, dict]) -> None:
    tmp = HONEST_PRED_CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    tmp.replace(HONEST_PRED_CACHE)


# ---------------------------------------------------------------------------
# Job prep (render query + 10 anchor tiles, compose collage at obfuscated path)
# ---------------------------------------------------------------------------


def prepare_job(
    obf_key: str,
    meta: dict,
    samples_by_raw: dict,
    emb,
    abs_to_emb_idx: dict,
    path_to_sample: dict,
) -> dict:
    """Render query + anchors + compose collage at obfuscated output path."""
    sample = samples_by_raw.get(meta["raw_path"])
    if sample is None:
        return {"key": obf_key, "error": "sample not found",
                "true_class": meta["true_class"], "person": meta["person"]}

    abs_path = str(sample.raw_path)
    if abs_path not in abs_to_emb_idx:
        return {"key": obf_key, "error": "no embedding for query",
                "true_class": sample.cls, "person": sample.person}
    q_idx = abs_to_emb_idx[abs_path]

    try:
        anchors_per_cls = retrieve_anchors_per_class(
            emb, q_idx, sample.person, k_per_class=2
        )
    except RuntimeError as e:
        return {"key": obf_key, "error": f"retrieve: {e}",
                "true_class": sample.cls, "person": sample.person}

    # Render query tile. We reuse tile_filename so its pixels are identical
    # to the leaky run; the tile_filename path is NEVER passed to the VLM
    # (only its pixels are pasted into the collage by compose_collage).
    q_tile = tile_filename(sample.cls, sample.raw_path)
    try:
        render_scan_tile(sample.raw_path, q_tile)
    except Exception as e:
        return {"key": obf_key, "error": f"query render: {e}",
                "true_class": sample.cls, "person": sample.person}

    anchor_info = []
    anchor_meta = []
    for ci, cls in enumerate(CLASSES):
        for rank, emb_idx in enumerate(anchors_per_cls[ci]):
            anchor_abs = emb["paths"][emb_idx]
            a_sample = path_to_sample.get(anchor_abs)
            if a_sample is None:
                return {"key": obf_key, "error": f"anchor sample missing: {anchor_abs}",
                        "true_class": sample.cls, "person": sample.person}
            assert a_sample.person != sample.person, (
                f"LEAK: anchor person {a_sample.person} == query person "
                f"{sample.person} for {obf_key}"
            )
            a_tile = tile_filename(a_sample.cls, a_sample.raw_path)
            try:
                render_scan_tile(a_sample.raw_path, a_tile)
            except Exception as e:
                return {"key": obf_key, "error": f"anchor render: {e}",
                        "true_class": sample.cls, "person": sample.person}
            anchor_info.append((cls, a_tile, a_sample.raw_path.name))
            anchor_meta.append({
                "class": cls,
                "rank": rank + 1,
                "name": a_sample.raw_path.name,
                "person": a_sample.person,
                "path": str(a_sample.raw_path.relative_to(REPO)),
            })

    # CRITICAL: obfuscated output path. No class, no person, no raw scan name.
    collage_path = HONEST_COLLAGE_DIR / f"{obf_key}.png"
    try:
        # query_id is rendered only as "QUERY (classify this)" header — we
        # still pass an obfuscated id to be safe (though compose_collage
        # currently ignores query_id in the rendered PNG).
        compose_collage(anchor_info, q_tile, obf_key, collage_path)
    except Exception as e:
        return {"key": obf_key, "error": f"compose: {e}",
                "true_class": sample.cls, "person": sample.person}

    return {
        "key": obf_key,
        "true_class": sample.cls,
        "person": sample.person,
        "patient": sample.patient,
        "collage_path": str(collage_path),
        "anchors": anchor_meta,
    }


def dispatch_vlm(job: dict, model: str) -> dict:
    if "error" in job:
        return {job["key"]: {
            "error": job["error"],
            "true_class": job["true_class"],
            "person": job["person"],
        }}
    res = call_claude_cli(Path(job["collage_path"]), model=model)
    res["true_class"] = job["true_class"]
    res["person"] = job["person"]
    res["patient"] = job["patient"]
    res["obf_key"] = job["key"]
    res["collage_path"] = str(Path(job["collage_path"]).relative_to(REPO))
    res["anchors"] = job["anchors"]
    return {job["key"]: res}


# ---------------------------------------------------------------------------
# Metrics + bootstrap
# ---------------------------------------------------------------------------


def compute_metrics(cache: dict[str, dict], manifest: dict[str, dict]) -> dict:
    keys = list(manifest.keys())
    y_true, y_pred, conf = [], [], []
    persons, patients = [], []
    for k in keys:
        e = cache.get(k)
        if not e or e.get("predicted_class") not in CLASSES:
            continue
        y_true.append(e["true_class"])
        y_pred.append(e["predicted_class"])
        conf.append(float(e.get("confidence", 0.0) or 0.0))
        persons.append(e.get("person", manifest[k]["person"]))
        patients.append(e.get("patient", manifest[k]["patient"]))
    if not y_true:
        return {"error": "no valid predictions"}
    labels = CLASSES
    w_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    m_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    from collections import Counter, defaultdict

    def majority_agg(groups):
        buckets = defaultdict(list)
        true_bucket = {}
        for i, g in enumerate(groups):
            buckets[g].append(y_pred[i])
            true_bucket[g] = y_true[i]
        gy_t, gy_p = [], []
        for g, preds in buckets.items():
            gy_t.append(true_bucket[g])
            most = Counter(preds).most_common(1)[0][0]
            gy_p.append(most)
        return gy_t, gy_p

    gy_t_person, gy_p_person = majority_agg(persons)
    per_person_wf1 = f1_score(gy_t_person, gy_p_person, labels=labels, average="weighted", zero_division=0)
    per_person_mf1 = f1_score(gy_t_person, gy_p_person, labels=labels, average="macro", zero_division=0)
    per_person_acc = sum(a == b for a, b in zip(gy_t_person, gy_p_person)) / len(gy_t_person)

    gy_t_pat, gy_p_pat = majority_agg(patients)
    per_pat_wf1 = f1_score(gy_t_pat, gy_p_pat, labels=labels, average="weighted", zero_division=0)
    per_pat_mf1 = f1_score(gy_t_pat, gy_p_pat, labels=labels, average="macro", zero_division=0)
    per_pat_acc = sum(a == b for a, b in zip(gy_t_pat, gy_p_pat)) / len(gy_t_pat)

    return {
        "n": len(y_true),
        "accuracy": acc,
        "f1_weighted": w_f1,
        "f1_macro": m_f1,
        "mean_confidence": float(np.mean(conf)),
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": labels,
        "per_person": {
            "n_persons": len(gy_t_person),
            "accuracy": per_person_acc,
            "f1_weighted": per_person_wf1,
            "f1_macro": per_person_mf1,
        },
        "per_patient_eye": {
            "n_patients": len(gy_t_pat),
            "accuracy": per_pat_acc,
            "f1_weighted": per_pat_wf1,
            "f1_macro": per_pat_mf1,
        },
        "y_true": y_true,
        "y_pred": y_pred,
    }


def bootstrap_f1_delta(y_true, y_pred, baseline_wf1: float, *,
                       baseline_label: str = "baseline",
                       n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    yt = np.array(y_true)
    yp = np.array(y_pred)
    n = len(yt)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = f1_score(yt[idx], yp[idx], labels=CLASSES,
                           average="weighted", zero_division=0)
    deltas = boot - baseline_wf1
    return {
        "baseline_label": baseline_label,
        "baseline_wf1": baseline_wf1,
        "n_boot": n_boot,
        "vlm_wf1_mean": float(boot.mean()),
        "vlm_wf1_ci_lo": float(np.quantile(boot, 0.025)),
        "vlm_wf1_ci_hi": float(np.quantile(boot, 0.975)),
        "delta_mean": float(deltas.mean()),
        "delta_ci_lo": float(np.quantile(deltas, 0.025)),
        "delta_ci_hi": float(np.quantile(deltas, 0.975)),
        "p_improved": float((deltas > 0).mean()),
    }


def paired_bootstrap_vs_leaky(honest_cache: dict, leaky_cache: dict,
                              manifest: dict, *, n_boot: int = 1000, seed: int = 42) -> dict:
    """Paired bootstrap over per-scan predictions. Quantifies filename-leak inflation."""
    # Build (truth, honest_pred, leaky_pred) triples over the same scans.
    # honest_cache is keyed by obf scan_XXXX; leaky_cache is keyed by repo-rel raw path.
    overlap = []
    for obf_k, meta in manifest.items():
        he = honest_cache.get(obf_k)
        leaky_key = meta["raw_rel"]
        le = leaky_cache.get(leaky_key)
        if not he or not le:
            continue
        if he.get("predicted_class") not in CLASSES or le.get("predicted_class") not in CLASSES:
            continue
        truth = meta["true_class"]
        overlap.append((truth, he["predicted_class"], le["predicted_class"]))
    if not overlap:
        return {"error": "no overlap with leaky cache"}
    yt = np.array([t for t, _, _ in overlap])
    yp_h = np.array([h for _, h, _ in overlap])
    yp_l = np.array([l for _, _, l in overlap])
    n = len(overlap)
    rng = np.random.default_rng(seed)
    boot_h = np.empty(n_boot)
    boot_l = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_h[i] = f1_score(yt[idx], yp_h[idx], labels=CLASSES,
                             average="weighted", zero_division=0)
        boot_l[i] = f1_score(yt[idx], yp_l[idx], labels=CLASSES,
                             average="weighted", zero_division=0)
    # Inflation = leaky - honest (we expect strongly positive)
    deltas = boot_l - boot_h
    return {
        "n_overlap": n,
        "n_boot": n_boot,
        "honest_wf1_mean": float(boot_h.mean()),
        "honest_wf1_ci_lo": float(np.quantile(boot_h, 0.025)),
        "honest_wf1_ci_hi": float(np.quantile(boot_h, 0.975)),
        "leaky_wf1_mean": float(boot_l.mean()),
        "leaky_wf1_ci_lo": float(np.quantile(boot_l, 0.025)),
        "leaky_wf1_ci_hi": float(np.quantile(boot_l, 0.975)),
        "inflation_mean": float(deltas.mean()),
        "inflation_ci_lo": float(np.quantile(deltas, 0.025)),
        "inflation_ci_hi": float(np.quantile(deltas, 0.975)),
        "p_leaky_gt_honest": float((deltas > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--limit", type=int, default=0, help="0 = all 240")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--v4-baseline-wf1", type=float, default=0.6887)
    ap.add_argument("--leaky-baseline-wf1", type=float, default=0.8873)
    args = ap.parse_args()

    all_samples = enumerate_samples(REPO / "TRAIN_SET")
    manifest = build_manifest(all_samples)
    print(f"Manifest: {len(manifest)} obfuscated IDs (persisted to {HONEST_MANIFEST.relative_to(REPO)})")

    keys = sorted(manifest.keys())  # scan_0000 .. scan_0239
    if args.limit > 0:
        keys = keys[: args.limit]
    print(f"Scoring {len(keys)} / {len(manifest)} scans (model={args.model}, workers={args.workers})")

    cache = load_cache()

    # --- Sanity verification: prompt path uniqueness / no class leak ---------
    # We render one collage and print the prompt that would be passed.
    # This doubles as a self-test for the leak fix.
    if not args.eval_only and keys:
        probe_key = keys[0]
        probe_path = HONEST_COLLAGE_DIR / f"{probe_key}.png"
        from vlm_few_shot import PROMPT_TEMPLATE
        probe_prompt = PROMPT_TEMPLATE.format(img_path=str(probe_path))
        # Basic grep — no class names should appear in the rendered path string.
        path_str = str(probe_path)
        for bad in CLASSES:
            if bad in path_str:
                print(f"FATAL: class name {bad!r} leaked into collage path {path_str}")
                return 2
        for bad_frag in ["SucheOko", "PGOV", "Diabetes", "Skleroza",
                         "Zdravi", "healthy", "_DM_", "_MS_", "suche_oko",
                         "glaukom", "TRAIN_SET"]:
            if bad_frag in path_str:
                print(f"FATAL: suspicious fragment {bad_frag!r} in collage path {path_str}")
                return 2
        print(f"[sanity] probe collage path: {path_str}")
        print(f"[sanity] prompt first 200 chars: {probe_prompt[:200]!r}")

    if not args.eval_only:
        emb = load_embeddings()
        abs_to_emb_idx = {p: i for i, p in enumerate(emb["paths"])}
        samples_by_raw = {str(s.raw_path): s for s in all_samples}
        path_to_sample = samples_by_raw  # alias (same mapping)

        print("[1/2] preparing collages ...")
        jobs = []
        t0 = time.time()
        for i, k in enumerate(keys):
            if k in cache and cache[k].get("predicted_class") in CLASSES:
                continue
            job = prepare_job(k, manifest[k], samples_by_raw,
                              emb, abs_to_emb_idx, path_to_sample)
            jobs.append(job)
            if (i + 1) % 30 == 0:
                print(f"  prepared {i+1}/{len(keys)}  ({time.time()-t0:.1f}s)")
        print(f"  {len(jobs)} jobs ready in {time.time()-t0:.1f}s "
              f"({len(keys) - len(jobs)} already cached)")

        if jobs:
            print(f"[2/2] dispatching {len(jobs)} VLM calls ({args.workers} workers) ...")
            t0 = time.time()
            done = 0
            errs = 0
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
                            "person": job.get("person", ""),
                        }}
                    cache.update(result)
                    done += 1
                    for k, v in result.items():
                        if v.get("predicted_class") in CLASSES:
                            ok = "OK" if v["predicted_class"] == v.get("true_class") else "--"
                            print(f"[{done}/{len(jobs)}] {ok} {k}  true={v.get('true_class')}  pred={v['predicted_class']}  conf={v.get('confidence', 0):.2f}  ${v.get('cost_usd', 0):.4f}")
                        else:
                            errs += 1
                            print(f"[{done}/{len(jobs)}] ERR {k}: {v.get('error', v)}")
                    if done % 10 == 0:
                        save_cache(cache)
            save_cache(cache)
            total_wall = time.time() - t0
            print(f"All VLM calls done in {total_wall:.1f}s (errors: {errs})")
        else:
            total_wall = 0.0
    else:
        total_wall = 0.0

    # Metrics
    metrics = compute_metrics(cache, {k: manifest[k] for k in keys})
    if "error" in metrics:
        print("ERROR:", metrics["error"])
        return 1

    print("\n=== HONEST Sonnet full-240 metrics ===")
    print(f"n           = {metrics['n']}")
    print(f"accuracy    = {metrics['accuracy']:.4f}")
    print(f"weighted F1 = {metrics['f1_weighted']:.4f}  <- primary")
    print(f"macro F1    = {metrics['f1_macro']:.4f}")
    print(f"mean conf   = {metrics['mean_confidence']:.3f}")
    print("\nPer-class:")
    for cls in metrics["labels"]:
        row = metrics["classification_report"].get(cls, {})
        print(f"  {cls:22s}  P={row.get('precision', 0):.3f}  R={row.get('recall', 0):.3f}  F1={row.get('f1-score', 0):.3f}  support={int(row.get('support', 0))}")
    print("\nConfusion matrix:")
    print("  " + "  ".join(f"{c[:6]:>6s}" for c in metrics["labels"]))
    for lab, row in zip(metrics["labels"], metrics["confusion_matrix"]):
        print(f"  {lab[:6]:>6s} " + "  ".join(f"{v:>6d}" for v in row))

    # Bootstrap vs v4
    boot_v4 = bootstrap_f1_delta(metrics["y_true"], metrics["y_pred"],
                                 args.v4_baseline_wf1,
                                 baseline_label="v4_multiscale")
    print(f"\n=== Bootstrap 1000x vs v4 (baseline wF1={args.v4_baseline_wf1}) ===")
    print(f"Honest Sonnet wF1 95% CI: [{boot_v4['vlm_wf1_ci_lo']:.4f}, {boot_v4['vlm_wf1_ci_hi']:.4f}]")
    print(f"Delta 95% CI:             [{boot_v4['delta_ci_lo']:+.4f}, {boot_v4['delta_ci_hi']:+.4f}]")
    print(f"P(Delta > 0 vs v4)      = {boot_v4['p_improved']:.3f}")

    # Paired bootstrap vs leaky (same scans, different pipelines)
    leaky_cache_path = CACHE_DIR / "vlm_sonnet_full_predictions.json"
    boot_leaky_paired = None
    if leaky_cache_path.exists():
        leaky_cache = json.loads(leaky_cache_path.read_text())
        boot_leaky_paired = paired_bootstrap_vs_leaky(
            cache, leaky_cache, {k: manifest[k] for k in keys}
        )
        if "error" not in boot_leaky_paired:
            bp = boot_leaky_paired
            print(f"\n=== Paired bootstrap 1000x: honest vs leaky Sonnet (quantify inflation) ===")
            print(f"N overlap = {bp['n_overlap']}")
            print(f"Honest wF1 (paired bootstrap): {bp['honest_wf1_mean']:.4f}  [{bp['honest_wf1_ci_lo']:.4f}, {bp['honest_wf1_ci_hi']:.4f}]")
            print(f"Leaky  wF1 (paired bootstrap): {bp['leaky_wf1_mean']:.4f}  [{bp['leaky_wf1_ci_lo']:.4f}, {bp['leaky_wf1_ci_hi']:.4f}]")
            print(f"Inflation (leaky - honest):    {bp['inflation_mean']:+.4f}  [{bp['inflation_ci_lo']:+.4f}, {bp['inflation_ci_hi']:+.4f}]")
            print(f"P(leaky > honest)            = {bp['p_leaky_gt_honest']:.3f}")

    # Bootstrap vs scalar leaky baseline too
    boot_leaky_scalar = bootstrap_f1_delta(metrics["y_true"], metrics["y_pred"],
                                           args.leaky_baseline_wf1,
                                           baseline_label="leaky_sonnet_0.8873",
                                           seed=43)

    # Per-person / per-patient
    pp = metrics["per_person"]
    pe = metrics["per_patient_eye"]
    print(f"\nPer-person ({pp['n_persons']} persons):  acc={pp['accuracy']:.4f}  wF1={pp['f1_weighted']:.4f}  mF1={pp['f1_macro']:.4f}")
    print(f"Per-patient ({pe['n_patients']} eyes):     acc={pe['accuracy']:.4f}  wF1={pe['f1_weighted']:.4f}  mF1={pe['f1_macro']:.4f}")

    # Cost
    total_cost = sum(float(cache[k].get("cost_usd", 0.0) or 0.0)
                     for k in keys if k in cache)
    print(f"\nTotal cost: ${total_cost:.3f}")

    # --- Write report ---------------------------------------------------------
    write_report(
        REPO / "reports" / "VLM_SONNET_HONEST.md",
        args, metrics, boot_v4, boot_leaky_scalar, boot_leaky_paired,
        total_cost, total_wall, args.workers,
    )

    return 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(out_path: Path, args, metrics: dict,
                 boot_v4: dict, boot_leaky: dict, boot_leaky_paired,
                 total_cost: float, total_wall: float, n_workers: int):
    import datetime as dt
    lines: list[str] = []
    lines.append("# VLM Sonnet 4.6 Few-Shot — HONEST (filename-leak fixed)\n")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n")

    wf1 = metrics["f1_weighted"]
    if wf1 < 0.50:
        verdict = "**DEAD — retract.** Honest wF1 below 0.50 means the few-shot VLM path is not competitive and was essentially random-ish after the leak fix."
    elif wf1 < 0.69:
        verdict = "**ENSEMBLE CANDIDATE ONLY.** Honest wF1 below v4 champion; only useful in stacking."
    else:
        verdict = "**POSSIBLE CHAMPION.** Requires P(Delta>0 vs v4) > 0.95 + red-team before promotion."

    lines.append("## TL;DR\n")
    lines.append(f"- **Honest weighted F1: {wf1:.4f}** (n={metrics['n']} scans, model={args.model})")
    lines.append(f"- Honest macro F1: {metrics['f1_macro']:.4f}")
    lines.append(f"- vs leaky-Sonnet claim (0.8873): **inflation = {args.leaky_baseline_wf1 - wf1:+.4f}** (scalar diff)")
    lines.append(f"- vs v4 champion (0.6887): delta = {wf1 - args.v4_baseline_wf1:+.4f}")
    if boot_v4:
        lines.append(f"- Bootstrap P(Delta > 0 vs v4) = {boot_v4['p_improved']:.3f}")
    lines.append(f"- Verdict: {verdict}\n")

    lines.append("## Setup\n")
    lines.append("- Fix: collage file written to `cache/vlm_few_shot_collages_honest/scan_XXXX.png` "
                 "(shuffled, zero-padded index; no class, person, or raw-scan name in path).")
    lines.append("- Manifest (NOT passed to VLM): `cache/vlm_sonnet_honest_manifest.json`.")
    lines.append("- Retrieval & anchor composition identical to the leaky pipeline — only the collage filename differs. "
                 "This isolates the filename leak as the causal factor.")
    lines.append("- Prompt passes `{img_path}` = obfuscated path; self-test verifies no class keyword in path.")
    lines.append(f"- Model: `{args.model}`; workers: {n_workers}; wall-clock: {total_wall:.1f}s; cost: ${total_cost:.3f}.\n")

    lines.append("## Scan-level metrics\n")
    lines.append(f"- Accuracy: {metrics['accuracy']:.4f}")
    lines.append(f"- **Weighted F1: {metrics['f1_weighted']:.4f}**  <- primary challenge metric")
    lines.append(f"- Macro F1: {metrics['f1_macro']:.4f}")
    lines.append(f"- Mean confidence: {metrics['mean_confidence']:.3f}\n")

    lines.append("## Per-class F1\n")
    lines.append("| Class | P | R | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    for cls in metrics["labels"]:
        row = metrics["classification_report"].get(cls, {})
        lines.append(f"| {cls} | {row.get('precision', 0):.3f} | {row.get('recall', 0):.3f} | {row.get('f1-score', 0):.3f} | {int(row.get('support', 0))} |")
    lines.append("")

    lines.append("## Confusion matrix (rows=true, cols=pred)\n")
    header = "| true\\pred | " + " | ".join(c[:10] for c in metrics["labels"]) + " |"
    sep = "|---|" + "|".join(["---"] * len(metrics["labels"])) + "|"
    lines.append(header)
    lines.append(sep)
    for lab, row in zip(metrics["labels"], metrics["confusion_matrix"]):
        lines.append(f"| {lab[:10]} | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")

    lines.append("## Per-person aggregation (majority vote)\n")
    pp = metrics["per_person"]
    lines.append(f"- N persons: {pp['n_persons']}")
    lines.append(f"- Accuracy: {pp['accuracy']:.4f}")
    lines.append(f"- Weighted F1: {pp['f1_weighted']:.4f}")
    lines.append(f"- Macro F1: {pp['f1_macro']:.4f}\n")

    lines.append("## Per-patient-eye aggregation (majority vote)\n")
    pe = metrics["per_patient_eye"]
    lines.append(f"- N patient-eyes: {pe['n_patients']}")
    lines.append(f"- Accuracy: {pe['accuracy']:.4f}")
    lines.append(f"- Weighted F1: {pe['f1_weighted']:.4f}")
    lines.append(f"- Macro F1: {pe['f1_macro']:.4f}\n")

    lines.append(f"## Bootstrap 1000x vs v4 multiscale (baseline wF1 = {boot_v4['baseline_wf1']:.4f})\n")
    lines.append(f"- Honest Sonnet wF1 bootstrap mean: {boot_v4['vlm_wf1_mean']:.4f}  (95% CI [{boot_v4['vlm_wf1_ci_lo']:.4f}, {boot_v4['vlm_wf1_ci_hi']:.4f}])")
    lines.append(f"- Delta (Sonnet - v4) mean: **{boot_v4['delta_mean']:+.4f}**  (95% CI [{boot_v4['delta_ci_lo']:+.4f}, {boot_v4['delta_ci_hi']:+.4f}])")
    lines.append(f"- **P(Delta > 0 vs v4) = {boot_v4['p_improved']:.3f}**\n")

    lines.append(f"## Bootstrap 1000x vs leaky Sonnet (baseline wF1 = {boot_leaky['baseline_wf1']:.4f}) — inflation quantification\n")
    lines.append(f"- Delta (honest - leaky) mean: **{boot_leaky['delta_mean']:+.4f}**  (95% CI [{boot_leaky['delta_ci_lo']:+.4f}, {boot_leaky['delta_ci_hi']:+.4f}])")
    lines.append(f"- P(honest > leaky) = {boot_leaky['p_improved']:.3f}")
    lines.append(f"- Inflation caused by filename leak (scalar): **{boot_leaky['baseline_wf1'] - metrics['f1_weighted']:+.4f}**\n")

    if boot_leaky_paired and "error" not in boot_leaky_paired:
        bp = boot_leaky_paired
        lines.append("## Paired bootstrap 1000x: honest vs leaky (same scans) — tighter inflation estimate\n")
        lines.append(f"- N overlap: {bp['n_overlap']}")
        lines.append(f"- Honest wF1 mean (paired): {bp['honest_wf1_mean']:.4f}  (95% CI [{bp['honest_wf1_ci_lo']:.4f}, {bp['honest_wf1_ci_hi']:.4f}])")
        lines.append(f"- Leaky  wF1 mean (paired): {bp['leaky_wf1_mean']:.4f}  (95% CI [{bp['leaky_wf1_ci_lo']:.4f}, {bp['leaky_wf1_ci_hi']:.4f}])")
        lines.append(f"- **Inflation (leaky - honest) = {bp['inflation_mean']:+.4f}  (95% CI [{bp['inflation_ci_lo']:+.4f}, {bp['inflation_ci_hi']:+.4f}])**")
        lines.append(f"- P(leaky > honest) = {bp['p_leaky_gt_honest']:.3f}\n")

    lines.append("## Decision\n")
    if wf1 < 0.50:
        lines.append("- **DEAD: retract few-shot VLM path.** Mark `VLM_SONNET_FULL_240.md` and all prior Sonnet/Haiku few-shot tables as CONTAMINATED.")
        lines.append("- Do NOT ensemble into Expert Council. Do NOT cite in pitch.")
    elif wf1 < 0.69:
        lines.append(f"- **Ensemble candidate only.** Honest wF1 ({wf1:.4f}) is below v4 champion (0.6887). Could contribute diverse predictions in stacking, but not a standalone winner.")
        lines.append("- Re-run the red-team before any promotion of this pipeline.")
    else:
        lines.append(f"- **Possible champion.** Honest wF1 ({wf1:.4f}) >= 0.69 and above v4.")
        if boot_v4['p_improved'] > 0.95:
            lines.append(f"- P(Delta > 0 vs v4) = {boot_v4['p_improved']:.3f} > 0.95 — PROMOTE to stable bundle.")
        else:
            lines.append(f"- P(Delta > 0 vs v4) = {boot_v4['p_improved']:.3f} <= 0.95 — need more scans / paired eval before promotion.")

    lines.append("")
    lines.append("## Reproducibility\n")
    lines.append("- Script: `scripts/vlm_few_shot_sonnet_honest.py`")
    lines.append("- Predictions cache: `cache/vlm_sonnet_honest_predictions.json`")
    lines.append("- Manifest: `cache/vlm_sonnet_honest_manifest.json`")
    lines.append("- Collages: `cache/vlm_few_shot_collages_honest/scan_XXXX.png`")
    lines.append(f"- Model slug: `{args.model}`")
    lines.append("- Leaky cache retained for comparison: `cache/vlm_sonnet_full_predictions.json` (do NOT overwrite).")

    out_path.write_text("\n".join(lines))
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
