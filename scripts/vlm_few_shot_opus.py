"""Few-shot / retrieval-augmented VLM classifier — Opus 4.7 variant.

Same pipeline as `vlm_few_shot_sonnet.py` (DINOv2 retrieval + 10-anchor
collage + Claude VLM classification), but uses `claude-opus-4-7` for the
VLM call and runs up to N parallel workers since Opus is slow / rate-
limited.

All logic except the model arg, cache path, and concurrency comes from
the Sonnet script — this is an apples-to-apples comparison run on the
SAME 60-scan person-stratified subset (seed=123, 12 per class).

Outputs
-------
- cache/vlm_opus_predictions.json   (raw per-scan results)
- reports/VLM_OPUS_COMPARISON.md    (Haiku vs Sonnet vs Opus table)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Re-use all the machinery from the Sonnet script
from scripts.vlm_few_shot_sonnet import (  # noqa: E402
    CACHE_DIR,
    COLLAGE_DIR,
    call_claude_cli,
    compose_collage,
    load_embeddings,
    render_scan_tile,
    retrieve_anchors_per_class,
    stratified_person_disjoint,
    tile_filename,
)
from teardrop.data import CLASSES, enumerate_samples  # noqa: E402

PRED_CACHE = CACHE_DIR / "vlm_opus_predictions.json"
REPORT_PATH = REPO / "reports" / "VLM_OPUS_COMPARISON.md"


# ---------------------------------------------------------------------------
# Cache IO (local to this script so we don't clobber Sonnet cache)
# ---------------------------------------------------------------------------


def load_cache() -> dict[str, dict]:
    if PRED_CACHE.exists():
        return json.loads(PRED_CACHE.read_text())
    return {}


def save_cache(cache: dict[str, dict], lock: Lock | None = None) -> None:
    if lock is not None:
        with lock:
            PRED_CACHE.write_text(json.dumps(cache, indent=2))
    else:
        PRED_CACHE.write_text(json.dumps(cache, indent=2))


# ---------------------------------------------------------------------------
# Per-sample worker: prepare collage (if needed) + one LLM call
# ---------------------------------------------------------------------------


def prepare_collage_for_sample(
    s,
    emb: dict,
    abs_to_emb_idx: dict[str, int],
    path_to_sample: dict,
) -> tuple[Path, list[dict]] | tuple[None, str]:
    abs_path = str(s.raw_path)
    if abs_path not in abs_to_emb_idx:
        return None, "no embedding for query"
    q_idx = abs_to_emb_idx[abs_path]

    try:
        anchors_per_cls = retrieve_anchors_per_class(emb, q_idx, s.person, k_per_class=2)
    except RuntimeError as e:
        return None, str(e)

    q_tile = tile_filename(s.cls, s.raw_path)
    try:
        render_scan_tile(s.raw_path, q_tile)
    except Exception as e:  # noqa: BLE001
        return None, f"query render failed: {e}"

    anchor_info: list[tuple[str, Path, str]] = []
    anchor_meta: list[dict] = []
    for ci, cls in enumerate(CLASSES):
        for rank, emb_idx in enumerate(anchors_per_cls[ci]):
            anchor_abs = emb["paths"][emb_idx]
            a_sample = path_to_sample.get(anchor_abs)
            if a_sample is None:
                return None, "anchor sample missing"
            a_tile = tile_filename(a_sample.cls, a_sample.raw_path)
            try:
                render_scan_tile(a_sample.raw_path, a_tile)
            except Exception as e:  # noqa: BLE001
                return None, f"anchor render failed: {e}"
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

    # HARD person-LOPO assertion — no scan from same person may appear as anchor
    for am in anchor_meta:
        assert am["person"] != s.person, (
            f"LEAK: anchor person == query person for {s.raw_path.name}"
        )

    import hashlib as _hl_collage
    _rel_q = str(s.raw_path.relative_to(REPO))
    _collage_id = _hl_collage.sha1(_rel_q.encode("utf-8")).hexdigest()[:16]
    collage_path = COLLAGE_DIR / f"scan_{_collage_id}.png"
    compose_collage(anchor_info, q_tile, _collage_id, collage_path)
    return collage_path, anchor_meta


def score_one(
    s,
    collage_path: Path,
    anchor_meta: list[dict],
    model: str,
) -> dict:
    res = call_claude_cli(collage_path, model=model, timeout_s=300)
    res["true_class"] = s.cls
    res["person"] = s.person
    res["query_path"] = str(s.raw_path.relative_to(REPO))
    res["collage_path"] = str(collage_path.relative_to(REPO))
    res["anchors"] = anchor_meta
    res["model"] = model
    return res


# ---------------------------------------------------------------------------
# Parallel driver
# ---------------------------------------------------------------------------


def run_parallel(
    samples_to_score,
    all_samples,
    model: str,
    n_workers: int,
    time_budget_s: float,
    cache: dict[str, dict],
) -> dict[str, dict]:
    t_start = time.time()
    emb = load_embeddings()
    abs_to_emb_idx = {p: i for i, p in enumerate(emb["paths"])}
    path_to_sample = {str(s.raw_path): s for s in all_samples}
    cache_lock = Lock()

    # First pass: prepare every collage we still need (sequential — fast & disk-bound)
    pending: list[tuple[Any, Path, list[dict], str]] = []  # (sample, collage_path, anchor_meta, key)
    for s in samples_to_score:
        key = str(s.raw_path.relative_to(REPO))
        if (
            key in cache
            and "predicted_class" in cache[key]
            and cache[key]["predicted_class"] in CLASSES
        ):
            continue
        result = prepare_collage_for_sample(s, emb, abs_to_emb_idx, path_to_sample)
        if result[0] is None:
            cache[key] = {"error": result[1], "true_class": s.cls, "person": s.person}
            save_cache(cache, cache_lock)
            print(f"PREP FAIL: {key} -- {result[1]}")
            continue
        collage_path, anchor_meta = result
        pending.append((s, collage_path, anchor_meta, key))

    print(f"Prepared {len(pending)} collages, {len(samples_to_score) - len(pending)} already cached or skipped")
    if not pending:
        return cache

    # Second pass: parallel LLM calls
    n_total = len(pending)
    done = 0
    print(f"Dispatching {n_total} calls with {n_workers} parallel workers, model={model}")

    def worker(item):
        s, collage_path, anchor_meta, key = item
        return key, s, score_one(s, collage_path, anchor_meta, model)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(worker, item) for item in pending]
        for fut in as_completed(futures):
            elapsed = time.time() - t_start
            if elapsed > time_budget_s:
                print(f"[budget] hit {elapsed:.0f}s after {done}/{n_total}, waiting for in-flight to finish")
                # still collect in-flight results
            try:
                key, s, res = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"WORKER FAIL: {e}")
                continue
            with cache_lock:
                cache[key] = res
            save_cache(cache, cache_lock)
            done += 1
            pred = res.get("predicted_class", "ERR")
            ok = "OK" if pred == s.cls else "--"
            cost = res.get("cost_usd", 0.0)
            lat = res.get("latency_s", 0.0)
            print(f"[{done}/{n_total}] {ok} {key}  true={s.cls}  pred={pred}  conf={res.get('confidence', 0):.2f}  t={lat:.1f}s  ${cost:.4f}")

    return cache


# ---------------------------------------------------------------------------
# Evaluation + comparison
# ---------------------------------------------------------------------------


def score_cache_on_keys(cache_dict: dict[str, dict], keys: list[str]) -> dict[str, Any]:
    y_true, y_pred = [], []
    for k in keys:
        v = cache_dict.get(k)
        if not v or v.get("predicted_class") not in CLASSES or "true_class" not in v:
            continue
        y_true.append(v["true_class"])
        y_pred.append(v["predicted_class"])
    if not y_true:
        return {"error": "no predictions"}
    labels = list(CLASSES)
    return {
        "n": len(y_true),
        "wf1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "mf1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "accuracy": sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
        "cm": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def bootstrap_delta_ci(
    y_true_a: list[str],
    y_pred_a: list[str],
    y_true_b: list[str],
    y_pred_b: list[str],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap of Δ = wF1(A) - wF1(B) on the SAME sample indices.

    A = Opus, B = Sonnet. If the 95% CI excludes 0, the difference is
    statistically significant at alpha=0.05.
    """
    assert len(y_true_a) == len(y_true_b)
    assert y_true_a == y_true_b  # paired: same samples
    rng = np.random.default_rng(seed)
    n = len(y_true_a)
    labels = list(CLASSES)
    y_true_arr = np.array(y_true_a)
    y_pred_a_arr = np.array(y_pred_a)
    y_pred_b_arr = np.array(y_pred_b)

    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        wf1_a = f1_score(y_true_arr[idx], y_pred_a_arr[idx], labels=labels, average="weighted", zero_division=0)
        wf1_b = f1_score(y_true_arr[idx], y_pred_b_arr[idx], labels=labels, average="weighted", zero_division=0)
        deltas[b] = wf1_a - wf1_b
    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    mean_d = float(np.mean(deltas))
    p_nonpositive = float(np.mean(deltas <= 0.0))
    return {
        "delta_mean": mean_d,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "n_boot": n_boot,
        "p_delta_nonpositive": p_nonpositive,  # one-sided p approx for H0: Opus <= Sonnet
        "significant_at_0.05_two_sided": (ci_lo > 0) or (ci_hi < 0),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
    out: Path,
    subset_keys: list[str],
    ev_opus: dict,
    ev_sonnet: dict | None,
    ev_haiku: dict | None,
    boot: dict | None,
    total_cost: float,
    decision: str,
):
    import datetime as dt
    lines: list[str] = []
    lines.append("# VLM Opus 4.7 vs Sonnet 4.6 vs Haiku 4.5 — Head-to-Head\n")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n")
    lines.append("## Setup\n")
    lines.append("- Subset: 60 scans, person-stratified (seed=123, 12 per class).")
    lines.append("- Pipeline: DINOv2-B kNN retrieval -> 10-anchor (2/class) collage -> VLM classifies query.")
    lines.append("- Person-LOPO hard-assertion: anchor person != query person for every sample.")
    lines.append(f"- Total cost this run (Opus): ${total_cost:.2f}\n")

    lines.append("## Headline comparison\n")
    lines.append("| Model | N | wF1 | mF1 | Accuracy |")
    lines.append("|---|---:|---:|---:|---:|")
    if ev_haiku:
        lines.append(f"| Haiku 4.5 (cached note) | {ev_haiku['n']} | {ev_haiku['wf1']:.4f} | {ev_haiku['mf1']:.4f} | {ev_haiku['accuracy']:.4f} |")
    if ev_sonnet:
        lines.append(f"| Sonnet 4.6 | {ev_sonnet['n']} | {ev_sonnet['wf1']:.4f} | {ev_sonnet['mf1']:.4f} | {ev_sonnet['accuracy']:.4f} |")
    lines.append(f"| **Opus 4.7** | {ev_opus['n']} | **{ev_opus['wf1']:.4f}** | **{ev_opus['mf1']:.4f}** | **{ev_opus['accuracy']:.4f}** |")
    lines.append("")

    if ev_sonnet:
        delta_wf1 = ev_opus["wf1"] - ev_sonnet["wf1"]
        lines.append(f"**Δ wF1 (Opus − Sonnet) on overlapping scans: {delta_wf1:+.4f}**\n")

    if boot:
        lines.append("## Paired bootstrap CI (1000 resamples, Opus − Sonnet)\n")
        lines.append(f"- Δ wF1 mean: {boot['delta_mean']:+.4f}")
        lines.append(f"- 95% CI: [{boot['ci95_lo']:+.4f}, {boot['ci95_hi']:+.4f}]")
        lines.append(f"- Two-sided significant at alpha=0.05: **{boot['significant_at_0.05_two_sided']}**")
        lines.append(f"- Fraction of bootstraps with Δ <= 0: {boot['p_delta_nonpositive']:.3f}\n")

    # per-class F1
    lines.append("## Per-class F1\n")
    header = "| Class | "
    sep = "|---|"
    if ev_haiku:
        header += "Haiku F1 | "
        sep += "---:|"
    if ev_sonnet:
        header += "Sonnet F1 | "
        sep += "---:|"
    header += "Opus F1 | Support |"
    sep += "---:|---:|"
    lines.append(header)
    lines.append(sep)
    for cls in ev_opus["labels"]:
        row = f"| {cls} |"
        if ev_haiku:
            r = ev_haiku["report"].get(cls, {})
            row += f" {r.get('f1-score', 0):.3f} |"
        if ev_sonnet:
            r = ev_sonnet["report"].get(cls, {})
            row += f" {r.get('f1-score', 0):.3f} |"
        r = ev_opus["report"].get(cls, {})
        row += f" {r.get('f1-score', 0):.3f} | {int(r.get('support', 0))} |"
        lines.append(row)
    lines.append("")

    # confusion matrix
    lines.append("## Opus 4.7 Confusion Matrix (rows=true, cols=pred)\n")
    head = "| true\\pred | " + " | ".join(c[:10] for c in ev_opus["labels"]) + " |"
    s = "|---|" + "|".join(["---:"] * len(ev_opus["labels"])) + "|"
    lines.append(head)
    lines.append(s)
    for lab, row in zip(ev_opus["labels"], ev_opus["cm"]):
        lines.append(f"| {lab[:10]} | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")

    lines.append("## Cost comparison (this run only)\n")
    lines.append(f"- Opus 60-scan cost: ${total_cost:.2f}")
    lines.append("- Sonnet reference: ~$0.50 for 60 scans historically (~5x cheaper).")
    lines.append("- Full 240 Opus projected: ~${:.2f}\n".format(total_cost * 240 / max(1, ev_opus["n"])))

    lines.append("## Decision\n")
    lines.append(decision + "\n")

    out.write_text("\n".join(lines))
    print(f"Report written to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=12, help="per-class stratified picks (default 12 -> 60 scans)")
    ap.add_argument("--seed", type=int, default=123, help="stratification seed (123 matches Sonnet baseline)")
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--time-budget-s", type=int, default=1800, help="wall-clock budget (default 30 min)")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    all_samples = enumerate_samples(REPO / "TRAIN_SET")
    print(f"Loaded {len(all_samples)} total samples")

    samples = stratified_person_disjoint(all_samples, per_class=args.subset, seed=args.seed)
    scored_keys = [str(s.raw_path.relative_to(REPO)) for s in samples]
    print(f"Subset: {len(samples)} scans (seed={args.seed}, per_class={args.subset})")

    cache = load_cache()
    if not args.eval_only:
        cache = run_parallel(
            samples, all_samples, args.model, args.workers, args.time_budget_s, cache
        )
        save_cache(cache)

    # --- Opus eval ---
    ev_opus = score_cache_on_keys(cache, scored_keys)
    print(f"\nOpus on {ev_opus.get('n', 0)} scans: wF1={ev_opus.get('wf1', 0):.4f}  mF1={ev_opus.get('mf1', 0):.4f}  acc={ev_opus.get('accuracy', 0):.4f}")

    # --- Sonnet eval on same keys ---
    sonnet_cache_path = CACHE_DIR / "vlm_sonnet_predictions.json"
    ev_sonnet = None
    boot = None
    if sonnet_cache_path.exists():
        sonnet_cache = json.loads(sonnet_cache_path.read_text())
        ev_sonnet = score_cache_on_keys(sonnet_cache, scored_keys)
        print(f"Sonnet on same {ev_sonnet.get('n', 0)} scans: wF1={ev_sonnet.get('wf1', 0):.4f}  mF1={ev_sonnet.get('mf1', 0):.4f}")

        # Bootstrap paired CI — only on the INTERSECTION of scored scans
        overlap = [k for k in scored_keys
                   if (k in cache and cache[k].get("predicted_class") in CLASSES)
                   and (k in sonnet_cache and sonnet_cache[k].get("predicted_class") in CLASSES)
                   and "true_class" in cache[k]]
        print(f"Paired overlap for bootstrap: {len(overlap)} scans")
        if len(overlap) >= 10:
            y_true = [cache[k]["true_class"] for k in overlap]
            y_pred_opus = [cache[k]["predicted_class"] for k in overlap]
            y_pred_sonnet = [sonnet_cache[k]["predicted_class"] for k in overlap]
            boot = bootstrap_delta_ci(y_true, y_pred_opus, y_true, y_pred_sonnet, n_boot=1000, seed=0)
            print(f"Bootstrap Δ wF1: mean={boot['delta_mean']:+.4f}  95% CI=[{boot['ci95_lo']:+.4f}, {boot['ci95_hi']:+.4f}]  sig={boot['significant_at_0.05_two_sided']}")

    # --- Haiku eval on overlap (only 19 scans available historically) ---
    haiku_cache_path = CACHE_DIR / "vlm_haiku_predictions_subset.json"
    ev_haiku = None
    if haiku_cache_path.exists():
        haiku_cache = json.loads(haiku_cache_path.read_text())
        ev_haiku = score_cache_on_keys(haiku_cache, scored_keys)
        print(f"Haiku on overlap ({ev_haiku.get('n', 0)} scans): wF1={ev_haiku.get('wf1', 0):.4f}")

    # --- Cost ---
    total_cost = sum(float(cache[k].get("cost_usd", 0.0) or 0.0)
                     for k in scored_keys if k in cache and cache[k].get("predicted_class") in CLASSES)
    print(f"\nOpus total cost (60 scan): ${total_cost:.2f}")

    # --- Decision ---
    decision_lines: list[str] = []
    if ev_opus and ev_sonnet and ev_opus.get("n") and ev_sonnet.get("n"):
        delta = ev_opus["wf1"] - ev_sonnet["wf1"]
        full_proj = total_cost * 240 / max(1, ev_opus["n"])
        if delta >= 0.03:
            decision_lines.append(f"**PROCEED** with full 240-scan Opus run. Δ wF1 = {delta:+.4f} exceeds +3 pp threshold.")
            decision_lines.append(f"Projected full-run cost: ~${full_proj:.2f}.")
            if boot and boot["significant_at_0.05_two_sided"]:
                decision_lines.append(f"Paired bootstrap 95% CI excludes 0 ([{boot['ci95_lo']:+.4f}, {boot['ci95_hi']:+.4f}]) — improvement is statistically significant.")
            elif boot:
                decision_lines.append(f"Caveat: bootstrap CI = [{boot['ci95_lo']:+.4f}, {boot['ci95_hi']:+.4f}] — lift is large but not robustly significant on n={ev_opus['n']}.")
        else:
            decision_lines.append(f"**STOP** — Opus does NOT clear the +3 pp threshold. Δ wF1 = {delta:+.4f}.")
            decision_lines.append("Sonnet 4.6 is the economic winner at ~1/5 the cost.")
            if boot:
                decision_lines.append(f"Paired bootstrap Δ 95% CI: [{boot['ci95_lo']:+.4f}, {boot['ci95_hi']:+.4f}] — {'excludes' if boot['significant_at_0.05_two_sided'] else 'includes'} 0.")
    else:
        decision_lines.append("Insufficient paired data for decision.")
    decision = "\n".join(f"- {line}" if not line.startswith("**") else line for line in decision_lines)

    print("\n=== DECISION ===")
    print(decision)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_report(REPORT_PATH, scored_keys, ev_opus, ev_sonnet, ev_haiku, boot, total_cost, decision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
