"""One-shot extension: run k=2 few-shot VLM on the 14-20 scans missing
from `cache/vlm_few_shot_predictions.json` so we have k=2 predictions
on the same 60-scan subset (per_class=12, seed=42) that the k=3
comparison uses.

Uses the SAME subset selection, retrieval, rendering, collage composer,
and Claude CLI wrapper as `scripts/vlm_few_shot.py` — just wraps them in
a ProcessPoolExecutor.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from teardrop.data import CLASSES, enumerate_samples  # noqa: E402

from vlm_few_shot import (  # noqa: E402
    CACHE_DIR, COLLAGE_DIR,
    call_claude_cli, compose_collage,
    load_embeddings, render_scan_tile, retrieve_anchors_per_class,
    stratified_person_disjoint, tile_filename,
)

# Separate cache file so we don't race with a parallel run of
# vlm_few_shot.py writing to vlm_few_shot_predictions.json.
EXT_CACHE = CACHE_DIR / "vlm_few_shot_k2_extend_predictions.json"


def load_ext_cache() -> dict:
    if EXT_CACHE.exists():
        return json.loads(EXT_CACHE.read_text())
    return {}


def save_ext_cache(cache: dict) -> None:
    EXT_CACHE.write_text(json.dumps(cache, indent=2))


def load_primary_cache() -> dict:
    p = CACHE_DIR / "vlm_few_shot_predictions.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def worker_classify_k2(task: tuple) -> tuple:
    key, collage_path_str, true_class, person, anchor_meta, query_path, model = task
    res = call_claude_cli(Path(collage_path_str), model=model)
    res["true_class"] = true_class
    res["person"] = person
    res["query_path"] = query_path
    res["collage_path"] = collage_path_str
    res["anchors"] = anchor_meta
    return key, res


def prepare_k2_tasks(samples_to_score, all_samples, primary_cache, ext_cache, model):
    emb = load_embeddings()
    abs_to_emb_idx = {p: i for i, p in enumerate(emb["paths"])}
    path_to_sample = {str(s.raw_path): s for s in all_samples}

    tasks = []
    for s in samples_to_score:
        key = str(s.raw_path.relative_to(REPO))
        # Skip if EITHER cache has a good prediction
        if (key in primary_cache and primary_cache[key].get("predicted_class") in CLASSES):
            continue
        if (key in ext_cache and ext_cache[key].get("predicted_class") in CLASSES):
            continue
        abs_path = str(s.raw_path)
        if abs_path not in abs_to_emb_idx:
            ext_cache[key] = {"error": "no embedding", "true_class": s.cls, "person": s.person}
            continue
        q_idx = abs_to_emb_idx[abs_path]
        try:
            anchors = retrieve_anchors_per_class(emb, q_idx, s.person, k_per_class=2)
        except RuntimeError as e:
            ext_cache[key] = {"error": str(e), "true_class": s.cls, "person": s.person}
            continue

        q_tile = tile_filename(s.cls, s.raw_path)
        render_scan_tile(s.raw_path, q_tile)

        anchor_info = []
        anchor_meta = []
        for ci, cls in enumerate(CLASSES):
            for rank, emb_idx in enumerate(anchors[ci]):
                a_sample = path_to_sample[emb["paths"][emb_idx]]
                a_tile = tile_filename(a_sample.cls, a_sample.raw_path)
                render_scan_tile(a_sample.raw_path, a_tile)
                import hashlib as _hl_anchor
                _a_rel = str(a_sample.raw_path.relative_to(REPO))
                _anchor_visible_id = _hl_anchor.sha1(_a_rel.encode("utf-8")).hexdigest()[:10]
                anchor_info.append((cls, a_tile, _anchor_visible_id))
                anchor_meta.append({
                    "class": cls, "rank": rank + 1,
                    "name": a_sample.raw_path.name,
                    "person": a_sample.person,
                    "path": str(a_sample.raw_path.relative_to(REPO)),
                })

        # person-disjoint sanity
        for am in anchor_meta:
            assert am["person"] != s.person, f"LEAK on {key}"

        import hashlib as _hl_collage
        _rel_q = str(s.raw_path.relative_to(REPO))
        _collage_id = _hl_collage.sha1(_rel_q.encode("utf-8")).hexdigest()[:16]
        collage_path = COLLAGE_DIR / f"scan_{_collage_id}.png"
        compose_collage(anchor_info, q_tile, _collage_id, collage_path)
        tasks.append((key, str(collage_path), s.cls, s.person, anchor_meta, key, model))
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--time-budget-s", type=int, default=900)
    args = ap.parse_args()

    all_samples = enumerate_samples(REPO / "TRAIN_SET")
    subset = stratified_person_disjoint(all_samples, per_class=args.per_class, seed=args.seed)
    print(f"[k2-extend] subset: {len(subset)} samples")

    primary_cache = load_primary_cache()  # read-only access to main k=2 cache
    ext_cache = load_ext_cache()
    tasks = prepare_k2_tasks(subset, all_samples, primary_cache, ext_cache, args.model)
    save_ext_cache(ext_cache)
    print(f"[k2-extend] tasks to dispatch: {len(tasks)}")

    if not tasks:
        print("[k2-extend] nothing to do")
        return 0

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker_classify_k2, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            key = futures[fut]
            try:
                _, res = fut.result(timeout=180)
            except Exception as e:
                res = {"error": f"worker: {type(e).__name__}: {e}"}
            ext_cache[key] = res
            pred = res.get("predicted_class", "ERR")
            truth = res.get("true_class", "?")
            ok = "OK" if pred == truth else "--"
            lat = res.get("latency_s", 0) or 0
            cost = res.get("cost_usd", 0) or 0
            print(f"[{done}/{len(tasks)}] {ok} {key}  true={truth}  pred={pred}  t={lat:.1f}s  ${cost:.4f}")
            if done % 5 == 0 or done == len(tasks):
                save_ext_cache(ext_cache)
            if time.time() - t0 > args.time_budget_s:
                print("[k2-extend] budget exceeded, waiting for inflight to drain")
    save_ext_cache(ext_cache)
    print(f"[k2-extend] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
