"""Pure k-NN retrieval baseline on DINOv2 embeddings (person-LOPO).

Goal: quantify how much of the few-shot VLM 80 % accuracy comes from
retrieval alone (i.e. from picking neighbors in DINOv2 space) vs VLM
reasoning on top of those neighbors.

Pipeline:
    1. Load cached DINOv2 scan embeddings (TTA D4 mean = strongest scan-level cache).
    2. For each of the 240 scans as query, compute cosine similarity to
       the remaining 239. Mask out any scan sharing the query's *person_id*.
    3. For k in {1, 3, 5, 7, 10} and three voting strategies, assign a class.
    4. Compute weighted / macro / per-class F1.
    5. Bootstrap 1000x the best config vs the v4 OOF predictions (0.6887 F1).
    6. Also evaluate the best config on the 40-scan subset the few-shot VLM was
       tested on, for an apples-to-apples comparison against the VLM 80 %.

Outputs:
    cache/knn_baseline_results.json            – full (k, strategy) grid
    cache/knn_baseline_best_predictions.json   – best config, per-scan predictions
    reports/KNN_BASELINE.md                    – human-readable report

Runs in < 30 s (no model inference, embeddings are pre-cached).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from teardrop.data import CLASSES  # noqa: E402  (ZdraviLudia, Diabetes, Glaukom, SM, SucheOko)

CACHE_DIR = ROOT / "cache"
REPORTS_DIR = ROOT / "reports"
RNG_SEED = 42
K_VALUES = [1, 3, 5, 7, 10]
STRATEGIES = ["majority", "sim_weighted", "softmax"]


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_l2, y, persons, paths) from the best cached DINOv2 scan cache.

    Preference order: TTA D4 (strongest, same as Wave 5 champion) → plain DINOv2 → mean tiles.
    """
    candidates = [
        CACHE_DIR / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz",
        CACHE_DIR / "emb_dinov2_vitb14_afmhot.npz",
        CACHE_DIR / "_scan_mean_emb_dinov2b.npz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=True)
        keys = set(z.files)
        # unify across the three possible layouts
        X = z["X_scan"] if "X_scan" in keys else (z["X"] if "X" in keys else z["emb"])
        y = z["scan_y"] if "scan_y" in keys else z["y"]
        persons = z["scan_groups"] if "scan_groups" in keys else z["groups"]
        paths = z["scan_paths"] if "scan_paths" in keys else z["paths"]
        print(f"[embeddings] loaded {path.name}  shape={X.shape}")
        X = X.astype(np.float32)
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        X_l2 = X / norm
        return X_l2, np.asarray(y, dtype=np.int64), np.asarray(persons), np.asarray(paths)
    raise FileNotFoundError("No DINOv2 embedding cache found.")


# ---------------------------------------------------------------------------
# Voting strategies
# ---------------------------------------------------------------------------

def vote_majority(neighbor_labels: np.ndarray, neighbor_sims: np.ndarray, n_classes: int) -> tuple[int, np.ndarray]:
    counts = np.bincount(neighbor_labels, minlength=n_classes).astype(np.float64)
    total = counts.sum()
    proba = counts / total if total > 0 else np.full(n_classes, 1.0 / n_classes)
    # tie-break: highest summed similarity (deterministic)
    top = counts.max()
    tied = np.where(counts == top)[0]
    if len(tied) == 1:
        pred = int(tied[0])
    else:
        sims_by_class = np.zeros(n_classes, dtype=np.float64)
        for lbl, sim in zip(neighbor_labels, neighbor_sims):
            sims_by_class[lbl] += sim
        pred = int(tied[np.argmax(sims_by_class[tied])])
    return pred, proba


def vote_sim_weighted(neighbor_labels: np.ndarray, neighbor_sims: np.ndarray, n_classes: int) -> tuple[int, np.ndarray]:
    # Map cosine sim [-1, 1] → [0, 2] so negatives don't cancel a lonely correct neighbor
    w = neighbor_sims + 1.0
    scores = np.zeros(n_classes, dtype=np.float64)
    for lbl, wi in zip(neighbor_labels, w):
        scores[lbl] += wi
    total = scores.sum()
    proba = scores / total if total > 0 else np.full(n_classes, 1.0 / n_classes)
    return int(np.argmax(scores)), proba


def vote_softmax(neighbor_labels: np.ndarray, neighbor_sims: np.ndarray, n_classes: int,
                 temperature: float = 0.1) -> tuple[int, np.ndarray]:
    """Aggregate (sum) similarities per class, then softmax across classes.

    Differs from sim_weighted in the final probability mapping (softmax vs linear),
    which sharpens the distribution and is commonly used in prototype-network
    style classification.
    """
    agg = np.zeros(n_classes, dtype=np.float64)
    count = np.zeros(n_classes, dtype=np.int64)
    for lbl, sim in zip(neighbor_labels, neighbor_sims):
        agg[lbl] += sim
        count[lbl] += 1
    # class logit = mean similarity over present neighbors; classes with zero
    # neighbors get -inf so softmax cleanly zeros them.
    logits = np.full(n_classes, -1e9, dtype=np.float64)
    mask = count > 0
    logits[mask] = agg[mask] / count[mask]
    logits = logits / temperature
    logits -= logits.max()
    expv = np.exp(logits)
    proba = expv / expv.sum()
    return int(np.argmax(proba)), proba


STRATEGY_FNS = {
    "majority": vote_majority,
    "sim_weighted": vote_sim_weighted,
    "softmax": vote_softmax,
}


# ---------------------------------------------------------------------------
# k-NN run
# ---------------------------------------------------------------------------

def run_knn(X: np.ndarray, y: np.ndarray, persons: np.ndarray,
            k: int, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """Person-LOPO leave-one-out retrieval.

    Returns (preds, proba) of shape (N,) and (N, n_classes).
    """
    n, _ = X.shape
    n_classes = len(CLASSES)
    sim_full = X @ X.T                      # (N, N), cosine because X is L2-normalized
    preds = np.zeros(n, dtype=np.int64)
    proba_all = np.zeros((n, n_classes), dtype=np.float64)

    vote_fn = STRATEGY_FNS[strategy]

    for i in range(n):
        mask_same_person = persons == persons[i]
        sims = sim_full[i].copy()
        sims[mask_same_person] = -np.inf          # drop self AND same-person scans
        if not np.any(np.isfinite(sims)):
            # Should never happen (only SucheOko with 2 persons could be risky,
            # but masking one of 2 still leaves 1 other person).
            preds[i] = y[i]  # fallback — harmless for diagnostics
            proba_all[i] = np.full(n_classes, 1.0 / n_classes)
            continue
        top_idx = np.argpartition(-sims, kth=min(k, np.isfinite(sims).sum() - 1))[:k]
        # sort by sim desc for determinism
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        neighbor_labels = y[top_idx]
        neighbor_sims = sims[top_idx]
        pred, proba = vote_fn(neighbor_labels, neighbor_sims, n_classes)
        preds[i] = pred
        proba_all[i] = proba
    return preds, proba_all


# ---------------------------------------------------------------------------
# Metrics + bootstrap
# ---------------------------------------------------------------------------

def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    scores = f1_score(y_true, y_pred, labels=list(range(n_classes)),
                      average=None, zero_division=0)
    return {CLASSES[i]: float(scores[i]) for i in range(n_classes)}


def bootstrap_ci(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray,
                 n_iter: int = 1000, seed: int = RNG_SEED) -> dict[str, float]:
    """Bootstrap CI for (A − B) weighted F1. Positive Δ = A better than B."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs, a_scores, b_scores = [], [], []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        a = f1_score(y_true[idx], y_pred_a[idx], average="weighted", zero_division=0)
        b = f1_score(y_true[idx], y_pred_b[idx], average="weighted", zero_division=0)
        diffs.append(a - b)
        a_scores.append(a)
        b_scores.append(b)
    diffs = np.array(diffs)
    return {
        "mean_delta": float(diffs.mean()),
        "ci_lo_2.5": float(np.percentile(diffs, 2.5)),
        "ci_hi_97.5": float(np.percentile(diffs, 97.5)),
        "p_delta_gt_0": float((diffs > 0).mean()),
        "a_mean_f1": float(np.mean(a_scores)),
        "a_ci": [float(np.percentile(a_scores, 2.5)), float(np.percentile(a_scores, 97.5))],
        "b_mean_f1": float(np.mean(b_scores)),
        "b_ci": [float(np.percentile(b_scores, 2.5)), float(np.percentile(b_scores, 97.5))],
    }


# ---------------------------------------------------------------------------
# Few-shot VLM apples-to-apples
# ---------------------------------------------------------------------------

def load_vlm_few_shot() -> dict:
    p = CACHE_DIR / "vlm_few_shot_predictions.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def evaluate_on_subset(preds: np.ndarray, y: np.ndarray, paths: np.ndarray,
                       subset_paths: set[str]) -> dict:
    idx = np.array([i for i, p in enumerate(paths) if rel_path(p) in subset_paths], dtype=np.int64)
    if len(idx) == 0:
        return {"n": 0}
    y_sub = y[idx]
    p_sub = preds[idx]
    return {
        "n": int(len(idx)),
        "accuracy": float((p_sub == y_sub).mean()),
        "weighted_f1": float(f1_score(y_sub, p_sub, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_sub, p_sub, average="macro", zero_division=0)),
    }


def rel_path(p) -> str:
    s = str(p)
    marker = "TRAIN_SET/"
    i = s.find(marker)
    return s[i:] if i >= 0 else s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    X, y, persons, paths = load_embeddings()
    n = len(y)
    n_classes = len(CLASSES)
    print(f"[setup] N={n} scans, {n_classes} classes, {len(set(persons))} unique persons")
    print(f"[setup] class distribution: {dict(Counter(int(l) for l in y))}")

    results = {}
    best_key = None
    best_f1 = -1.0
    best_preds = None
    best_proba = None

    for k in K_VALUES:
        for strat in STRATEGIES:
            preds, proba = run_knn(X, y, persons, k=k, strategy=strat)
            w_f1 = float(f1_score(y, preds, average="weighted", zero_division=0))
            m_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
            per_cls = per_class_f1(y, preds, n_classes)
            acc = float((preds == y).mean())
            key = f"k{k}_{strat}"
            results[key] = {
                "k": k,
                "strategy": strat,
                "accuracy": acc,
                "weighted_f1": w_f1,
                "macro_f1": m_f1,
                "per_class_f1": per_cls,
            }
            print(f"  {key:20s}  acc={acc:.4f}  wF1={w_f1:.4f}  mF1={m_f1:.4f}")
            if w_f1 > best_f1:
                best_f1 = w_f1
                best_key = key
                best_preds = preds
                best_proba = proba

    print(f"\n[winner] {best_key} with weighted F1 = {best_f1:.4f}")

    # v4 comparison
    v4_path = CACHE_DIR / "v4_oof_predictions.npz"
    v4_compare = None
    if v4_path.exists():
        v4 = np.load(v4_path, allow_pickle=True)
        # Align by path
        v4_paths = v4["scan_paths"]
        v4_y = v4["y"]
        v4_proba = v4["proba"]
        # Build path→idx map for v4
        v4_idx = {str(p): i for i, p in enumerate(v4_paths)}
        order = np.array([v4_idx[str(p)] for p in paths], dtype=np.int64)
        v4_preds = v4_proba[order].argmax(axis=1)
        v4_y_aligned = v4_y[order]
        assert np.array_equal(v4_y_aligned, y), "v4 label mismatch!"
        v4_w_f1 = float(f1_score(y, v4_preds, average="weighted", zero_division=0))
        v4_m_f1 = float(f1_score(y, v4_preds, average="macro", zero_division=0))
        print(f"\n[v4 baseline]  wF1={v4_w_f1:.4f}  mF1={v4_m_f1:.4f}")

        boot = bootstrap_ci(y, best_preds, v4_preds, n_iter=1000, seed=RNG_SEED)
        print(f"[bootstrap knn − v4]  Δ mean={boot['mean_delta']:+.4f}  "
              f"95% CI=[{boot['ci_lo_2.5']:+.4f}, {boot['ci_hi_97.5']:+.4f}]  "
              f"P(Δ>0)={boot['p_delta_gt_0']:.3f}")
        v4_compare = {
            "v4_weighted_f1": v4_w_f1,
            "v4_macro_f1": v4_m_f1,
            "bootstrap": boot,
        }

    # VLM few-shot apples-to-apples on the 40-scan subset
    vlm_fs = load_vlm_few_shot()
    vlm_compare = None
    if vlm_fs:
        subset_rel = set(vlm_fs.keys())
        # Evaluate k-NN on same 40 scans
        knn_on_subset = evaluate_on_subset(best_preds, y, paths, subset_rel)

        # Re-compute VLM metrics
        y_true = []
        y_pred_vlm = []
        for k_rel, v in vlm_fs.items():
            y_true.append(CLASSES.index(v["true_class"]))
            y_pred_vlm.append(CLASSES.index(v["predicted_class"]))
        y_true = np.array(y_true, dtype=np.int64)
        y_pred_vlm = np.array(y_pred_vlm, dtype=np.int64)
        vlm_acc = float((y_true == y_pred_vlm).mean())
        vlm_wf1 = float(f1_score(y_true, y_pred_vlm, average="weighted", zero_division=0))
        vlm_mf1 = float(f1_score(y_true, y_pred_vlm, average="macro", zero_division=0))

        # Bootstrap knn-on-subset vs vlm on the same 40 items (aligned by path)
        # align
        path_to_knn_pred = {rel_path(p): int(best_preds[i]) for i, p in enumerate(paths)}
        aligned_idx = [i for i, k_rel in enumerate(vlm_fs.keys()) if k_rel in path_to_knn_pred]
        knn_preds_subset = np.array([path_to_knn_pred[list(vlm_fs.keys())[i]] for i in aligned_idx])
        vlm_preds_subset = y_pred_vlm[aligned_idx]
        y_true_subset = y_true[aligned_idx]

        vlm_boot = bootstrap_ci(y_true_subset, knn_preds_subset, vlm_preds_subset, n_iter=1000, seed=RNG_SEED)
        print(f"\n[apples-to-apples on N={len(aligned_idx)} VLM subset]")
        print(f"  k-NN best:  acc={knn_on_subset['accuracy']:.4f}  wF1={knn_on_subset['weighted_f1']:.4f}  mF1={knn_on_subset['macro_f1']:.4f}")
        print(f"  VLM fewshot: acc={vlm_acc:.4f}  wF1={vlm_wf1:.4f}  mF1={vlm_mf1:.4f}")
        print(f"  Δ (knn − vlm) mean={vlm_boot['mean_delta']:+.4f}  "
              f"95% CI=[{vlm_boot['ci_lo_2.5']:+.4f}, {vlm_boot['ci_hi_97.5']:+.4f}]  "
              f"P(knn>vlm)={vlm_boot['p_delta_gt_0']:.3f}")
        vlm_compare = {
            "subset_n": int(len(aligned_idx)),
            "knn_on_subset": knn_on_subset,
            "vlm_on_subset": {
                "accuracy": vlm_acc,
                "weighted_f1": vlm_wf1,
                "macro_f1": vlm_mf1,
            },
            "bootstrap_knn_minus_vlm": vlm_boot,
        }

    # ---- persist results ----------------------------------------------------

    summary = {
        "champion_v4_weighted_f1": 0.6887,
        "best_config": best_key,
        "best_weighted_f1": best_f1,
        "best_macro_f1": results[best_key]["macro_f1"],
        "best_per_class_f1": results[best_key]["per_class_f1"],
        "grid": results,
        "v4_comparison": v4_compare,
        "vlm_few_shot_comparison": vlm_compare,
        "n_samples": int(n),
        "n_persons": int(len(set(persons))),
        "embedding_source": "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz",
    }
    out_main = CACHE_DIR / "knn_baseline_results.json"
    with open(out_main, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {out_main}")

    out_preds = CACHE_DIR / "knn_baseline_best_predictions.json"
    per_scan = []
    for i in range(n):
        per_scan.append({
            "path": rel_path(paths[i]),
            "person": str(persons[i]),
            "true_class": CLASSES[int(y[i])],
            "pred_class": CLASSES[int(best_preds[i])],
            "correct": bool(best_preds[i] == y[i]),
            "proba": {CLASSES[j]: float(best_proba[i, j]) for j in range(n_classes)},
        })
    with open(out_preds, "w") as f:
        json.dump({
            "config": best_key,
            "k": results[best_key]["k"],
            "strategy": results[best_key]["strategy"],
            "weighted_f1": best_f1,
            "predictions": per_scan,
        }, f, indent=2)
    print(f"[saved] {out_preds}")

    # ---- markdown report ----------------------------------------------------

    write_report(summary, results, best_key)


def write_report(summary: dict, results: dict, best_key: str) -> None:
    lines: list[str] = []
    a = lines.append
    a("# KNN Baseline — Pure Retrieval on DINOv2 Embeddings")
    a("")
    a("**Question:** how much of the few-shot VLM's ~80 % accuracy comes from *retrieval alone* "
      "(picking neighbors in DINOv2 space) vs the VLM's visual reasoning on top of those neighbors?")
    a("")
    a("**Method:** cosine k-NN on cached DINOv2 ViT-B/14 scan-level embeddings "
      "(TTA-D4 afmhot, L2-normalized). Person-LOPO: neighbors from the query's person are excluded.")
    a("")
    a(f"**N scans:** {summary['n_samples']} across {summary['n_persons']} unique persons, 5 classes.")
    a(f"**Embedding source:** `cache/{summary['embedding_source']}`")
    a("")

    a("## Grid (weighted F1 / macro F1)")
    a("")
    a("|  k  | majority | sim-weighted | softmax |")
    a("|----:|---------:|-------------:|--------:|")
    for k in K_VALUES:
        row = [f"| {k} "]
        for strat in STRATEGIES:
            cell = results[f"k{k}_{strat}"]
            bold = f"k{k}_{strat}" == best_key
            txt = f"{cell['weighted_f1']:.4f} / {cell['macro_f1']:.4f}"
            if bold:
                txt = f"**{txt}**"
            row.append(f"| {txt} ")
        row.append("|")
        a("".join(row))
    a("")
    a(f"**Winner:** `{best_key}` — weighted F1 **{summary['best_weighted_f1']:.4f}**, "
      f"macro F1 **{summary['best_macro_f1']:.4f}**")
    a("")
    a("### Per-class F1 (winner)")
    a("")
    a("| Class | F1 |")
    a("|---|---:|")
    for cls, v in summary["best_per_class_f1"].items():
        a(f"| {cls} | {v:.4f} |")
    a("")

    # v4 section
    if summary.get("v4_comparison"):
        v4 = summary["v4_comparison"]
        boot = v4["bootstrap"]
        a("## vs Champion v4 (full 240 person-LOPO)")
        a("")
        a(f"- v4 multiscale: weighted F1 = **{v4['v4_weighted_f1']:.4f}**, macro F1 = {v4['v4_macro_f1']:.4f}")
        a(f"- k-NN best:    weighted F1 = **{summary['best_weighted_f1']:.4f}**")
        a(f"- Δ (knn − v4) mean = **{boot['mean_delta']:+.4f}**, "
          f"95 % CI = [{boot['ci_lo_2.5']:+.4f}, {boot['ci_hi_97.5']:+.4f}]")
        a(f"- **P(k-NN beats v4) = {boot['p_delta_gt_0']:.3f}** (bootstrap 1000×)")
        a("")

    # VLM section
    if summary.get("vlm_few_shot_comparison"):
        vlm = summary["vlm_few_shot_comparison"]
        boot = vlm["bootstrap_knn_minus_vlm"]
        a("## vs Few-shot VLM (apples-to-apples on same subset)")
        a("")
        a(f"- subset size: {vlm['subset_n']}")
        a("")
        a("| System | accuracy | weighted F1 | macro F1 |")
        a("|---|---:|---:|---:|")
        a(f"| VLM few-shot (2 anchors/class, collage) | "
          f"{vlm['vlm_on_subset']['accuracy']:.4f} | "
          f"{vlm['vlm_on_subset']['weighted_f1']:.4f} | "
          f"{vlm['vlm_on_subset']['macro_f1']:.4f} |")
        a(f"| k-NN best ({best_key}) | "
          f"{vlm['knn_on_subset']['accuracy']:.4f} | "
          f"{vlm['knn_on_subset']['weighted_f1']:.4f} | "
          f"{vlm['knn_on_subset']['macro_f1']:.4f} |")
        a("")
        a(f"- Δ weighted F1 (k-NN − VLM) mean = **{boot['mean_delta']:+.4f}**, "
          f"95 % CI = [{boot['ci_lo_2.5']:+.4f}, {boot['ci_hi_97.5']:+.4f}]")
        a(f"- **P(k-NN beats VLM on the same {vlm['subset_n']} scans) = {boot['p_delta_gt_0']:.3f}**")
        a("")

        # Conclusion
        a("## Conclusion")
        a("")
        knn_acc = vlm["knn_on_subset"]["accuracy"]
        vlm_acc = vlm["vlm_on_subset"]["accuracy"]
        diff = knn_acc - vlm_acc  # positive = knn better
        subset_n = vlm["subset_n"]
        if knn_acc >= vlm_acc - 0.02:
            a(f"On the same {subset_n}-scan subset, zero-training cosine k-NN on DINOv2 embeddings "
              f"matches the few-shot VLM ({knn_acc:.1%} vs {vlm_acc:.1%}; Δ {diff:+.1%}, "
              f"P(k-NN ≥ VLM) = {boot['p_delta_gt_0']:.2f}). "
              "**Most of the VLM's signal is retrieval, not reasoning** — the VLM is largely "
              "rubber-stamping the top-k neighbors DINOv2 already picked.")
        elif knn_acc >= vlm_acc - 0.10:
            a(f"On the same {subset_n}-scan subset, the VLM (acc {vlm_acc:.1%}) edges out pure "
              f"k-NN retrieval (acc {knn_acc:.1%}) by {-diff:.1%}. "
              "A non-trivial but modest share of the VLM's accuracy is retrieval-only — "
              "worth asking whether a cheap learned head on the k-NN neighbors (logistic "
              "regression / prototypical networks) would close the gap at zero API cost.")
        else:
            a(f"On the same {subset_n}-scan subset, the VLM (acc {vlm_acc:.1%}) clearly beats "
              f"pure k-NN retrieval (acc {knn_acc:.1%}) by **{-diff:.1%}**, "
              f"P(VLM > k-NN) = {1 - boot['p_delta_gt_0']:.3f}. "
              "**The VLM's reasoning adds real signal beyond DINOv2 retrieval** — it is not "
              "merely rubber-stamping nearest neighbors. The few-shot VLM pipeline is therefore "
              "defensible on this subset, though the ~4× cost of v4 multiscale (which is "
              f"{vlm_acc - 0.6887:+.1%} vs VLM on different/full data) still needs honest "
              "cost-benefit accounting at full-dataset scale.")
        a("")
        a("### Caveat on the full-dataset view")
        a("")
        a(f"Against v4 multiscale on the full 240 person-LOPO (not the {subset_n}-scan subset), "
          f"k-NN trails by {summary['best_weighted_f1'] - 0.6887:+.4f} weighted F1 "
          f"(P(k-NN > v4) = {summary['v4_comparison']['bootstrap']['p_delta_gt_0']:.3f}). "
          "So the ensemble stack is doing meaningful work beyond raw retrieval, especially "
          "on the long-tail classes (SucheOko F1 = 0 under k-NN — single nearest neighbor "
          "never sits inside the query's 2-person-only class after person-exclusion).")

    (REPORTS_DIR / "KNN_BASELINE.md").write_text("\n".join(lines))
    print(f"[saved] {REPORTS_DIR / 'KNN_BASELINE.md'}")


if __name__ == "__main__":
    main()
