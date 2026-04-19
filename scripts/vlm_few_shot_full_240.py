"""Scale the DINOv2-retrieval + VLM (Haiku 4.5) few-shot classifier to
ALL 240 scans with parallel workers.

Writes to a separate cache (`cache/vlm_few_shot_full_predictions.json`)
so the 40-scan result (`cache/vlm_few_shot_predictions.json`) stays
untouched.

Key design:
- Reuses `scripts/vlm_few_shot.py` helpers (rendering, retrieval,
  composition, Claude CLI wrapper).
- Prepares (render + compose) all collages serially (cheap, bounded
  by disk I/O).
- Dispatches the Haiku 4.5 calls in parallel via a ThreadPoolExecutor
  (default 16 workers). Each Haiku call is a subprocess, so threading
  is sufficient.
- Asserts person-disjoint anchors (no same-person leakage).
- After inference, computes:
    - weighted F1, macro F1, per-class F1
    - confusion matrix
    - per-patient / per-person aggregation F1
    - bootstrap 1000x CI on (VLM weighted F1 - v4 0.6887)
- Saves per-query confidences so we can fuse in a later ensemble.

Usage:
    .venv/bin/python scripts/vlm_few_shot_full_240.py --workers 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

# Reuse helpers from the 40-scan script
from vlm_few_shot import (  # noqa: E402
    CACHE_DIR,
    COLLAGE_DIR,
    call_claude_cli,
    compose_collage,
    load_embeddings,
    render_scan_tile,
    retrieve_anchors_per_class,
    tile_filename,
)
from teardrop.data import CLASSES, enumerate_samples  # noqa: E402

# Separate cache file — do NOT clobber the 40-scan one.
FULL_PRED_CACHE = CACHE_DIR / "vlm_few_shot_full_predictions.json"


def load_full_cache() -> dict[str, dict]:
    if FULL_PRED_CACHE.exists():
        return json.loads(FULL_PRED_CACHE.read_text())
    return {}


def save_full_cache(cache: dict[str, dict]) -> None:
    tmp = FULL_PRED_CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    tmp.replace(FULL_PRED_CACHE)


# ---------------------------------------------------------------------------
# Prepare all collages, then fan out Claude calls
# ---------------------------------------------------------------------------


def prepare_job(sample, emb, abs_to_emb_idx, path_to_sample) -> dict:
    """Render query tile, retrieve 10 anchors, render them, compose
    collage. Return a dict with all info needed to call the VLM.
    """
    key = str(sample.raw_path.relative_to(REPO))
    abs_path = str(sample.raw_path)
    if abs_path not in abs_to_emb_idx:
        return {"key": key, "error": "no embedding for query",
                "true_class": sample.cls, "person": sample.person}
    q_idx = abs_to_emb_idx[abs_path]

    try:
        anchors_per_cls = retrieve_anchors_per_class(
            emb, q_idx, sample.person, k_per_class=2
        )
    except RuntimeError as e:
        return {"key": key, "error": f"retrieve: {e}",
                "true_class": sample.cls, "person": sample.person}

    # render query tile
    q_tile = tile_filename(sample.cls, sample.raw_path)
    try:
        render_scan_tile(sample.raw_path, q_tile)
    except Exception as e:
        return {"key": key, "error": f"query render: {e}",
                "true_class": sample.cls, "person": sample.person}

    anchor_info = []
    anchor_meta = []
    for ci, cls in enumerate(CLASSES):
        for rank, emb_idx in enumerate(anchors_per_cls[ci]):
            anchor_abs = emb["paths"][emb_idx]
            a_sample = path_to_sample.get(anchor_abs)
            if a_sample is None:
                return {"key": key, "error": f"anchor sample missing: {anchor_abs}",
                        "true_class": sample.cls, "person": sample.person}
            # HARD ASSERT: no same-person anchors
            assert a_sample.person != sample.person, (
                f"LEAK: anchor person {a_sample.person} == query person "
                f"{sample.person} for {key}"
            )
            a_tile = tile_filename(a_sample.cls, a_sample.raw_path)
            try:
                render_scan_tile(a_sample.raw_path, a_tile)
            except Exception as e:
                return {"key": key, "error": f"anchor render: {e}",
                        "true_class": sample.cls, "person": sample.person}
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

    import hashlib as _hl_collage
    _rel_q = str(sample.raw_path.relative_to(REPO))
    _collage_id = _hl_collage.sha1(_rel_q.encode("utf-8")).hexdigest()[:16]
    collage_path = COLLAGE_DIR / f"scan_{_collage_id}.png"
    try:
        compose_collage(anchor_info, q_tile, _collage_id, collage_path)
    except Exception as e:
        return {"key": key, "error": f"compose: {e}",
                "true_class": sample.cls, "person": sample.person}

    return {
        "key": key,
        "true_class": sample.cls,
        "person": sample.person,
        "patient": sample.patient,
        "collage_path": str(collage_path),
        "anchors": anchor_meta,
    }


def dispatch_vlm(job: dict, model: str) -> dict:
    """Call Claude on a prepared collage. Returns the full result dict
    keyed on `job['key']`.
    """
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
    res["query_path"] = job["key"]
    res["collage_path"] = str(Path(job["collage_path"]).relative_to(REPO))
    res["anchors"] = job["anchors"]
    return {job["key"]: res}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(cache: dict[str, dict], keys: list[str]) -> dict:
    y_true, y_pred, conf = [], [], []
    persons, patients = [], []
    for k in keys:
        e = cache.get(k)
        if not e or e.get("predicted_class") not in CLASSES:
            continue
        y_true.append(e["true_class"])
        y_pred.append(e["predicted_class"])
        conf.append(float(e.get("confidence", 0.0) or 0.0))
        persons.append(e.get("person", ""))
        patients.append(e.get("patient", ""))
    if not y_true:
        return {"error": "no valid predictions"}
    labels = CLASSES
    w_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    m_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # per-person majority vote F1
    def majority_agg(groups):
        from collections import Counter, defaultdict
        buckets = defaultdict(list)
        true_bucket = {}
        for i, g in enumerate(groups):
            buckets[g].append(y_pred[i])
            true_bucket[g] = y_true[i]  # by construction all same per group
        gy_t, gy_p = [], []
        for g, preds in buckets.items():
            gy_t.append(true_bucket[g])
            # mode pred (break ties by first)
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


def bootstrap_f1_delta(y_true, y_pred, baseline_wf1: float, *, n_boot: int = 1000, seed: int = 42) -> dict:
    """Bootstrap the VLM weighted F1 and report CI + P(VLM - baseline > 0)."""
    rng = np.random.default_rng(seed)
    yt = np.array(y_true)
    yp = np.array(y_pred)
    n = len(yt)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = f1_score(yt[idx], yp[idx], labels=CLASSES, average="weighted", zero_division=0)
    deltas = boot - baseline_wf1
    p_improved = float((deltas > 0).mean())
    return {
        "n_boot": n_boot,
        "baseline_wf1": baseline_wf1,
        "vlm_wf1_mean": float(boot.mean()),
        "vlm_wf1_ci_lo": float(np.quantile(boot, 0.025)),
        "vlm_wf1_ci_hi": float(np.quantile(boot, 0.975)),
        "delta_mean": float(deltas.mean()),
        "delta_ci_lo": float(np.quantile(deltas, 0.025)),
        "delta_ci_hi": float(np.quantile(deltas, 0.975)),
        "p_improved": p_improved,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_report(out_path: Path, args, metrics: dict, boot: dict, total_cost: float, total_latency: float, n_workers: int):
    import datetime as dt
    lines = []
    lines.append("# VLM Few-Shot on FULL 240 Scans (Wave 7 candidate)\n")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n")
    lines.append("## Setup\n")
    lines.append("- Retrieval: DINOv2-B (`cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`), cosine sim, **person-disjoint** anchors.")
    lines.append(f"- k=2 anchors per class -> 10-anchor collage + QUERY tile.")
    lines.append(f"- VLM: `{args.model}`, `claude -p --output-format json` via CLI subprocess.")
    lines.append(f"- Parallelism: {n_workers} worker threads.")
    lines.append(f"- Scans scored: {metrics.get('n', 0)} / 240.")
    lines.append(f"- Total API cost: ${total_cost:.3f}")
    lines.append(f"- Wall-clock time: {total_latency:.1f} s")
    lines.append("")

    lines.append("## Scan-level metrics\n")
    lines.append(f"- Accuracy: **{metrics['accuracy']:.4f}**")
    lines.append(f"- **Weighted F1: {metrics['f1_weighted']:.4f}**")
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

    lines.append("## Per-person aggregation (majority vote across scans of same person)\n")
    pp = metrics["per_person"]
    lines.append(f"- N persons: {pp['n_persons']}")
    lines.append(f"- Accuracy: {pp['accuracy']:.4f}")
    lines.append(f"- Weighted F1: {pp['f1_weighted']:.4f}")
    lines.append(f"- Macro F1: {pp['f1_macro']:.4f}\n")

    lines.append("## Per-patient-eye aggregation (majority vote across scans of same eye)\n")
    pe = metrics["per_patient_eye"]
    lines.append(f"- N patient-eyes: {pe['n_patients']}")
    lines.append(f"- Accuracy: {pe['accuracy']:.4f}")
    lines.append(f"- Weighted F1: {pe['f1_weighted']:.4f}")
    lines.append(f"- Macro F1: {pe['f1_macro']:.4f}\n")

    lines.append("## Bootstrap 1000x vs v4 baseline (weighted F1 = 0.6887)\n")
    lines.append(f"- VLM weighted F1 mean: {boot['vlm_wf1_mean']:.4f}  (95% CI [{boot['vlm_wf1_ci_lo']:.4f}, {boot['vlm_wf1_ci_hi']:.4f}])")
    lines.append(f"- Delta (VLM - v4) mean: **{boot['delta_mean']:+.4f}**  (95% CI [{boot['delta_ci_lo']:+.4f}, {boot['delta_ci_hi']:+.4f}])")
    lines.append(f"- **P(Delta > 0) = {boot['p_improved']:.3f}**")
    if boot["p_improved"] > 0.95 and metrics["f1_weighted"] > 0.70:
        lines.append("\n**>>> CHAMPION CANDIDATE: weighted F1 > 0.70 AND P(Delta > 0) > 0.95 <<<**\n")
    lines.append("")

    lines.append("## Comparison vs 40-scan subset (`reports/VLM_FEW_SHOT_RESULTS.md`)\n")
    lines.append("| Run | N | Accuracy | Weighted F1 | Macro F1 |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| 40-scan subset | 40 | 0.8000 | (not reported) | 0.7974 |")
    lines.append(f"| Full 240 | {metrics['n']} | {metrics['accuracy']:.4f} | {metrics['f1_weighted']:.4f} | {metrics['f1_macro']:.4f} |")
    lines.append("")
    lines.append("If macro F1 dropped vs 40-scan subset: the stratified subset picked 8 scans per class, the first per person, which systematically picks \"canonical\" scans and hides the large within-person variance (especially in SklerozaMultiplex, 9 persons x up to ~14 scans). The full 240 eval exposes harder within-person scans and the 7:1 class imbalance now weights SklerozaMultiplex heavily in weighted F1.\n")

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--limit", type=int, default=0, help="0 = all 240")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--v4-baseline-wf1", type=float, default=0.6887)
    args = ap.parse_args()

    all_samples = enumerate_samples(REPO / "TRAIN_SET")
    if args.limit > 0:
        samples = all_samples[: args.limit]
    else:
        samples = all_samples
    print(f"Scoring {len(samples)} / {len(all_samples)} samples")

    cache = load_full_cache()
    scored_keys = [str(s.raw_path.relative_to(REPO)) for s in samples]

    if not args.eval_only:
        emb = load_embeddings()
        abs_to_emb_idx = {p: i for i, p in enumerate(emb["paths"])}
        path_to_sample = {str(s.raw_path): s for s in all_samples}

        # 1) prepare all jobs serially (cheap; renders tiles + collages)
        print("[1/2] preparing collages ...")
        jobs = []
        t0 = time.time()
        for i, s in enumerate(samples):
            key = str(s.raw_path.relative_to(REPO))
            if key in cache and cache[key].get("predicted_class") in CLASSES:
                continue  # already done
            job = prepare_job(s, emb, abs_to_emb_idx, path_to_sample)
            jobs.append(job)
            if (i + 1) % 30 == 0:
                print(f"  prepared {i+1}/{len(samples)}  ({time.time()-t0:.1f}s)")
        print(f"  prepared {len(jobs)} jobs in {time.time()-t0:.1f}s")

        # 2) fan out Claude calls in parallel
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
                    result = {job["key"]: {"error": f"future: {e}", "true_class": job.get("true_class", ""), "person": job.get("person", "")}}
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
                    save_full_cache(cache)
        save_full_cache(cache)
        total_wall = time.time() - t0
        print(f"All VLM calls done in {total_wall:.1f}s (errors: {errs})")
    else:
        total_wall = 0.0

    # metrics
    metrics = compute_metrics(cache, scored_keys)
    if "error" in metrics:
        print("ERROR: ", metrics["error"])
        return 1

    print("\n=== FULL 240 scan-level metrics ===")
    print(f"n = {metrics['n']}")
    print(f"accuracy = {metrics['accuracy']:.4f}")
    print(f"weighted F1 = {metrics['f1_weighted']:.4f}")
    print(f"macro F1 = {metrics['f1_macro']:.4f}")
    print(f"mean conf = {metrics['mean_confidence']:.3f}")
    print("\nPer-class:")
    for cls in metrics["labels"]:
        row = metrics["classification_report"].get(cls, {})
        print(f"  {cls:22s}  F1={row.get('f1-score', 0):.3f}  support={int(row.get('support', 0))}")
    print("\nConfusion matrix:")
    print("  " + "  ".join(f"{c[:6]:>6s}" for c in metrics["labels"]))
    for lab, row in zip(metrics["labels"], metrics["confusion_matrix"]):
        print(f"  {lab[:6]:>6s} " + "  ".join(f"{v:>6d}" for v in row))

    # bootstrap
    boot = bootstrap_f1_delta(metrics["y_true"], metrics["y_pred"], args.v4_baseline_wf1)
    print(f"\n=== Bootstrap 1000x vs v4 (baseline wF1={args.v4_baseline_wf1}) ===")
    print(f"VLM wF1 95% CI: [{boot['vlm_wf1_ci_lo']:.4f}, {boot['vlm_wf1_ci_hi']:.4f}]")
    print(f"Delta 95% CI:   [{boot['delta_ci_lo']:+.4f}, {boot['delta_ci_hi']:+.4f}]")
    print(f"P(delta > 0) = {boot['p_improved']:.3f}")

    # per-person / per-patient
    pp = metrics["per_person"]
    pe = metrics["per_patient_eye"]
    print(f"\nPer-person ({pp['n_persons']} persons):   acc={pp['accuracy']:.4f}  wF1={pp['f1_weighted']:.4f}  mF1={pp['f1_macro']:.4f}")
    print(f"Per-patient ({pe['n_patients']} eyes):     acc={pe['accuracy']:.4f}  wF1={pe['f1_weighted']:.4f}  mF1={pe['f1_macro']:.4f}")

    # costs
    total_cost = sum(float(cache[k].get("cost_usd", 0.0) or 0.0) for k in scored_keys if k in cache)
    print(f"\nTotal cost: ${total_cost:.3f}")

    # write report
    out = REPO / "reports" / "VLM_FEW_SHOT_FULL_240.md"
    write_report(out, args, metrics, boot, total_cost, total_wall, args.workers)
    print(f"Report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
