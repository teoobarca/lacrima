"""Scale the DINOv2-retrieval + VLM (Sonnet 4.6) few-shot classifier to
ALL 240 scans with parallel workers.

Fork of `scripts/vlm_few_shot_full_240.py` (Haiku full 240) with:
- model = claude-sonnet-4-6
- cache  = cache/vlm_sonnet_full_predictions.json   (DO NOT clobber the
                                                     60-subset cache at
                                                     cache/vlm_sonnet_predictions.json)
- report = reports/VLM_SONNET_FULL_240.md
- bootstrap 1000x vs v4 weighted F1 baseline (0.6887)
- bootstrap 1000x vs Haiku-full-240 weighted F1 (0.6755)
- 8 worker threads (subprocess-bound; threads beat processes here)

Usage:
    .venv/bin/python scripts/vlm_few_shot_sonnet_full_240.py --workers 8
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

# Reuse helpers from the base few-shot script
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

# Separate cache file — do NOT clobber the 60-scan Sonnet subset cache
# (cache/vlm_sonnet_predictions.json).
FULL_PRED_CACHE = CACHE_DIR / "vlm_sonnet_full_predictions.json"


def load_full_cache() -> dict[str, dict]:
    if FULL_PRED_CACHE.exists():
        return json.loads(FULL_PRED_CACHE.read_text())
    return {}


def save_full_cache(cache: dict[str, dict]) -> None:
    tmp = FULL_PRED_CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    tmp.replace(FULL_PRED_CACHE)


# ---------------------------------------------------------------------------
# Prepare jobs (render query tile + 10 anchor tiles + compose collage)
# ---------------------------------------------------------------------------


def prepare_job(sample, emb, abs_to_emb_idx, path_to_sample) -> dict:
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
            # HARD ASSERT: person-disjoint anchors (LOPO)
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

    def majority_agg(groups):
        from collections import Counter, defaultdict
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
    """Bootstrap VLM weighted F1, report CI + P(VLM - baseline > 0).

    Note: the baseline is a scalar (reported on the same 240-scan set by
    a different model), so we only resample VLM predictions. This is
    conservative — a paired bootstrap would be tighter but requires
    per-scan baseline predictions.
    """
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


# ---------------------------------------------------------------------------
# Paired bootstrap against Haiku-full-240 (per-scan predictions available)
# ---------------------------------------------------------------------------


def paired_bootstrap_vs_haiku(cache_sonnet: dict[str, dict],
                              cache_haiku: dict[str, dict],
                              keys: list[str],
                              *, n_boot: int = 1000, seed: int = 42) -> dict:
    overlap = []
    for k in keys:
        se = cache_sonnet.get(k)
        he = cache_haiku.get(k)
        if not se or not he:
            continue
        if se.get("predicted_class") not in CLASSES or he.get("predicted_class") not in CLASSES:
            continue
        if se.get("true_class") != he.get("true_class"):
            continue
        overlap.append((se["true_class"], se["predicted_class"], he["predicted_class"]))
    if not overlap:
        return {"error": "no overlap with Haiku cache"}
    yt = np.array([t for t, _, _ in overlap])
    yp_s = np.array([s for _, s, _ in overlap])
    yp_h = np.array([h for _, _, h in overlap])
    n = len(overlap)
    rng = np.random.default_rng(seed)
    boot_s = np.empty(n_boot)
    boot_h = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_s[i] = f1_score(yt[idx], yp_s[idx], labels=CLASSES,
                             average="weighted", zero_division=0)
        boot_h[i] = f1_score(yt[idx], yp_h[idx], labels=CLASSES,
                             average="weighted", zero_division=0)
    deltas = boot_s - boot_h
    return {
        "n_overlap": n,
        "n_boot": n_boot,
        "sonnet_wf1_mean": float(boot_s.mean()),
        "sonnet_wf1_ci_lo": float(np.quantile(boot_s, 0.025)),
        "sonnet_wf1_ci_hi": float(np.quantile(boot_s, 0.975)),
        "haiku_wf1_mean": float(boot_h.mean()),
        "haiku_wf1_ci_lo": float(np.quantile(boot_h, 0.025)),
        "haiku_wf1_ci_hi": float(np.quantile(boot_h, 0.975)),
        "delta_mean": float(deltas.mean()),
        "delta_ci_lo": float(np.quantile(deltas, 0.025)),
        "delta_ci_hi": float(np.quantile(deltas, 0.975)),
        "p_improved": float((deltas > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_report(out_path: Path, args, metrics: dict,
                 boot_v4: dict, boot_haiku: dict,
                 boot_haiku_paired: dict | None,
                 total_cost: float, total_latency: float, n_workers: int,
                 agreement_stats: dict):
    import datetime as dt
    lines: list[str] = []
    lines.append("# VLM Sonnet 4.6 Few-Shot on FULL 240 Scans\n")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n")
    lines.append("## Setup\n")
    lines.append("- Retrieval: DINOv2-B (`cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`), cosine sim, **person-disjoint** anchors.")
    lines.append("- k=2 anchors per class -> 10-anchor collage + QUERY tile.")
    lines.append(f"- VLM: `{args.model}`, `claude -p --output-format json` via CLI subprocess.")
    lines.append(f"- Parallelism: {n_workers} worker threads (subprocess-bound; threads > processes).")
    lines.append(f"- Scans scored: {metrics.get('n', 0)} / 240.")
    lines.append(f"- Total API cost: ${total_cost:.3f}")
    lines.append(f"- Wall-clock time: {total_latency:.1f} s")
    lines.append("")

    lines.append("## Scan-level metrics (primary)\n")
    lines.append(f"- Accuracy: **{metrics['accuracy']:.4f}**")
    lines.append(f"- **Weighted F1: {metrics['f1_weighted']:.4f}**  <- challenge metric")
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

    lines.append(f"## Bootstrap 1000x vs v4 multiscale champion (weighted F1 = {boot_v4['baseline_wf1']:.4f})\n")
    lines.append(f"- Sonnet weighted F1 bootstrap mean: {boot_v4['vlm_wf1_mean']:.4f}  (95% CI [{boot_v4['vlm_wf1_ci_lo']:.4f}, {boot_v4['vlm_wf1_ci_hi']:.4f}])")
    lines.append(f"- Delta (Sonnet - v4) mean: **{boot_v4['delta_mean']:+.4f}**  (95% CI [{boot_v4['delta_ci_lo']:+.4f}, {boot_v4['delta_ci_hi']:+.4f}])")
    lines.append(f"- **P(Delta > 0 vs v4) = {boot_v4['p_improved']:.3f}**\n")

    lines.append(f"## Bootstrap 1000x vs Haiku-full-240 (weighted F1 = {boot_haiku['baseline_wf1']:.4f}) — unpaired\n")
    lines.append(f"- Delta (Sonnet - Haiku) mean: **{boot_haiku['delta_mean']:+.4f}**  (95% CI [{boot_haiku['delta_ci_lo']:+.4f}, {boot_haiku['delta_ci_hi']:+.4f}])")
    lines.append(f"- **P(Delta > 0 vs Haiku) = {boot_haiku['p_improved']:.3f}**\n")

    if boot_haiku_paired and "error" not in boot_haiku_paired:
        lines.append(f"## Paired bootstrap 1000x vs Haiku-full-240 (tighter, same-scan resampling)\n")
        bp = boot_haiku_paired
        lines.append(f"- N overlap: {bp['n_overlap']}")
        lines.append(f"- Sonnet wF1 mean (paired resamples): {bp['sonnet_wf1_mean']:.4f}  (95% CI [{bp['sonnet_wf1_ci_lo']:.4f}, {bp['sonnet_wf1_ci_hi']:.4f}])")
        lines.append(f"- Haiku  wF1 mean (paired resamples): {bp['haiku_wf1_mean']:.4f}  (95% CI [{bp['haiku_wf1_ci_lo']:.4f}, {bp['haiku_wf1_ci_hi']:.4f}])")
        lines.append(f"- Delta (Sonnet - Haiku) mean: **{bp['delta_mean']:+.4f}**  (95% CI [{bp['delta_ci_lo']:+.4f}, {bp['delta_ci_hi']:+.4f}])")
        lines.append(f"- **P(Delta > 0 vs Haiku, paired) = {bp['p_improved']:.3f}**\n")

    if agreement_stats:
        lines.append("## Sonnet vs Haiku agreement on 240 scans\n")
        lines.append(f"- Both correct: {agreement_stats['both_right']}")
        lines.append(f"- Both wrong: {agreement_stats['both_wrong']}")
        lines.append(f"- Sonnet right only (Haiku wrong): {agreement_stats['sonnet_right_only']}")
        lines.append(f"- Haiku right only (Sonnet wrong): {agreement_stats['haiku_right_only']}")
        lines.append(f"- Agreement rate: {agreement_stats['agreement_pct']:.1%}\n")

    # Champion flag
    if (metrics["f1_weighted"] >= 0.75
            and boot_v4["p_improved"] > 0.95):
        lines.append("## >>> NEW CHAMPION CANDIDATE <<<\n")
        lines.append(f"- Weighted F1 = {metrics['f1_weighted']:.4f} >= 0.75")
        lines.append(f"- P(Delta > 0 vs v4) = {boot_v4['p_improved']:.3f} > 0.95")
        lines.append("- **ACTION**: Red-team before promotion (person-LOPO audit, anchor-leak audit, prompt-sensitivity, cost/latency ops review).\n")
    elif metrics["f1_weighted"] >= 0.70:
        lines.append("## Promising (not yet new-champion threshold)\n")
        lines.append(f"- Weighted F1 = {metrics['f1_weighted']:.4f} is strong but either < 0.75 or P(Delta>0 vs v4) <= 0.95.")
        lines.append(f"- Worth ensembling into Expert Council.\n")

    lines.append("## Comparison summary\n")
    lines.append("| Run | N | Weighted F1 | Macro F1 | Notes |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| Sonnet 60-subset (`VLM_FEW_SHOT_RESULTS.md`) | 60 | 0.8454 | 0.8454 | stratified, person-disjoint |")
    lines.append(f"| Haiku full 240 (`VLM_FEW_SHOT_FULL_240.md`) | 240 | 0.6755 | 0.5925 | champion-beating attempt |")
    lines.append(f"| **Sonnet full 240 (this run)** | {metrics['n']} | **{metrics['f1_weighted']:.4f}** | **{metrics['f1_macro']:.4f}** | 10-anchor collage, k=2 |")
    lines.append(f"| v4 multiscale ensemble | 240 | 0.6887 | — | current Wave-5 champion |")
    lines.append("")

    lines.append("## Reproducibility\n")
    lines.append("- Script: `scripts/vlm_few_shot_sonnet_full_240.py`")
    lines.append(f"- Predictions cache: `cache/vlm_sonnet_full_predictions.json`")
    lines.append("- Collages: `cache/vlm_safe/few_shot/collages/`")
    lines.append(f"- Model slug: `{args.model}`")
    lines.append("- Embeddings: `cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz` (L2-normalized)")
    lines.append("")

    out_path.write_text("\n".join(lines))


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
    ap.add_argument("--haiku-baseline-wf1", type=float, default=0.6755)
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

    print("\n=== Sonnet FULL 240 scan-level metrics ===")
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

    # bootstrap vs v4
    boot_v4 = bootstrap_f1_delta(metrics["y_true"], metrics["y_pred"],
                                 args.v4_baseline_wf1, baseline_label="v4_multiscale")
    print(f"\n=== Bootstrap 1000x vs v4 (baseline wF1={args.v4_baseline_wf1}) ===")
    print(f"Sonnet wF1 95% CI: [{boot_v4['vlm_wf1_ci_lo']:.4f}, {boot_v4['vlm_wf1_ci_hi']:.4f}]")
    print(f"Delta 95% CI:      [{boot_v4['delta_ci_lo']:+.4f}, {boot_v4['delta_ci_hi']:+.4f}]")
    print(f"P(delta > 0 vs v4) = {boot_v4['p_improved']:.3f}")

    # bootstrap vs Haiku (unpaired)
    boot_haiku = bootstrap_f1_delta(metrics["y_true"], metrics["y_pred"],
                                    args.haiku_baseline_wf1, baseline_label="haiku_full240", seed=43)
    print(f"\n=== Bootstrap 1000x vs Haiku (baseline wF1={args.haiku_baseline_wf1}) ===")
    print(f"Delta 95% CI:      [{boot_haiku['delta_ci_lo']:+.4f}, {boot_haiku['delta_ci_hi']:+.4f}]")
    print(f"P(delta > 0 vs Haiku, unpaired) = {boot_haiku['p_improved']:.3f}")

    # paired bootstrap vs Haiku (if Haiku cache available)
    haiku_cache_path = CACHE_DIR / "vlm_few_shot_full_predictions.json"
    boot_haiku_paired = None
    agreement_stats = {}
    if haiku_cache_path.exists():
        haiku_cache = json.loads(haiku_cache_path.read_text())
        boot_haiku_paired = paired_bootstrap_vs_haiku(cache, haiku_cache, scored_keys)
        if "error" not in boot_haiku_paired:
            print(f"\n=== Paired bootstrap 1000x vs Haiku ===")
            print(f"N overlap = {boot_haiku_paired['n_overlap']}")
            print(f"Delta 95% CI: [{boot_haiku_paired['delta_ci_lo']:+.4f}, {boot_haiku_paired['delta_ci_hi']:+.4f}]")
            print(f"P(delta > 0 vs Haiku, paired) = {boot_haiku_paired['p_improved']:.3f}")

        # simple agreement count
        both_right = both_wrong = s_only = h_only = 0
        for k in scored_keys:
            se = cache.get(k); he = haiku_cache.get(k)
            if not se or not he:
                continue
            if se.get("predicted_class") not in CLASSES or he.get("predicted_class") not in CLASSES:
                continue
            truth = se.get("true_class")
            s_ok = se["predicted_class"] == truth
            h_ok = he["predicted_class"] == truth
            if s_ok and h_ok: both_right += 1
            elif not s_ok and not h_ok: both_wrong += 1
            elif s_ok and not h_ok: s_only += 1
            elif h_ok and not s_ok: h_only += 1
        total = both_right + both_wrong + s_only + h_only
        if total:
            # agreement on prediction string (not just correctness)
            agree = 0
            for k in scored_keys:
                se = cache.get(k); he = haiku_cache.get(k)
                if not se or not he: continue
                if se.get("predicted_class") not in CLASSES or he.get("predicted_class") not in CLASSES: continue
                if se["predicted_class"] == he["predicted_class"]:
                    agree += 1
            agreement_stats = {
                "both_right": both_right,
                "both_wrong": both_wrong,
                "sonnet_right_only": s_only,
                "haiku_right_only": h_only,
                "agreement_pct": agree / total,
            }
            print(f"\nAgreement: both_right={both_right}  both_wrong={both_wrong}  "
                  f"sonnet_only={s_only}  haiku_only={h_only}  agree_pct={agree/total:.1%}")

    # per-person / per-patient
    pp = metrics["per_person"]
    pe = metrics["per_patient_eye"]
    print(f"\nPer-person ({pp['n_persons']} persons):   acc={pp['accuracy']:.4f}  wF1={pp['f1_weighted']:.4f}  mF1={pp['f1_macro']:.4f}")
    print(f"Per-patient ({pe['n_patients']} eyes):     acc={pe['accuracy']:.4f}  wF1={pe['f1_weighted']:.4f}  mF1={pe['f1_macro']:.4f}")

    # costs
    total_cost = sum(float(cache[k].get("cost_usd", 0.0) or 0.0) for k in scored_keys if k in cache)
    print(f"\nTotal cost: ${total_cost:.3f}")

    # write report
    out = REPO / "reports" / "VLM_SONNET_FULL_240.md"
    write_report(out, args, metrics, boot_v4, boot_haiku, boot_haiku_paired,
                 total_cost, total_wall, args.workers, agreement_stats)
    print(f"Report written to {out}")

    # Champion-gate terminal banner
    if metrics["f1_weighted"] >= 0.75 and boot_v4["p_improved"] > 0.95:
        print("\n" + "=" * 72)
        print(">>> NEW CHAMPION CANDIDATE — red-team before promotion <<<")
        print(f"    weighted F1 = {metrics['f1_weighted']:.4f}  (>= 0.75)")
        print(f"    P(Delta > 0 vs v4) = {boot_v4['p_improved']:.3f}  (> 0.95)")
        print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
