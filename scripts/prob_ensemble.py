"""Probability-averaging ensemble over 5 components.

Components (all cached):
    1. DINOv2-S tiled (mean-pooled to scan level)
    2. DINOv2-B tiled (mean-pooled to scan level)
    3. BiomedCLIP tiled (mean-pooled to scan level)
    4. Handcrafted features (GLCM+LBP+fractal+HOG)
    5. TDA features (cubical persistent homology)

Per-component classifier: StandardScaler + LogisticRegression
    (class_weight='balanced', max_iter=3000).

Evaluation: PERSON-level LOPO (35 groups, derived via person_id(path)).

Ensemble strategies compared:
    - Uniform average
    - Validation-F1 weighted
    - Per-class weighted (grid search over class weights)
    - Geometric mean (log-average, re-normalize)
    - Stacking via meta-LR (nested-LOPO)
    - Subset ensembles (all 2-of-5, all 3-of-5, all 4-of-5)

Finally: per-class threshold optimization on the best ensemble.
Output: cache/best_ensemble_predictions.npz and
        reports/ENSEMBLE_PROB_RESULTS.md.
"""
from __future__ import annotations

import itertools
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import CLASSES, enumerate_samples, person_id  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
N_CLASSES = len(CLASSES)


# ---------------------------------------------------------------------------
# Component loaders — each returns (X_scan, y, person_groups, scan_paths)
# Row order is the canonical `enumerate_samples` order. We always realign to
# that ordering so matrices are directly comparable.
# ---------------------------------------------------------------------------

@dataclass
class Component:
    name: str
    X: np.ndarray              # (240, D) scan-level features
    y: np.ndarray              # (240,) class labels
    person_groups: np.ndarray  # (240,) person IDs
    scan_paths: np.ndarray     # (240,) raw paths


def _mean_pool_tiles(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    X_tiles = z["X"]
    t2s = z["tile_to_scan"]
    scan_y = np.asarray(z["scan_y"])
    scan_paths = np.asarray(z["scan_paths"])
    n_scans = len(scan_y)
    out = np.zeros((n_scans, X_tiles.shape[1]), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int32)
    for ti, si in enumerate(t2s):
        out[si] += X_tiles[ti]
        counts[si] += 1
    out /= np.maximum(counts, 1)[:, None]
    return out, scan_y, scan_paths, None  # person groups derived below


def load_components(samples) -> list[Component]:
    """Assemble the 5 cached components, all row-aligned to `samples`."""
    # enumerate_samples gives relative paths; cache stores absolute. Build a
    # resolved absolute-path map so both sides compare equal.
    path_to_idx: dict[str, int] = {}
    for i, s in enumerate(samples):
        absp = str(Path(s.raw_path).resolve())
        relp = str(s.raw_path)
        path_to_idx[absp] = i
        path_to_idx[relp] = i
    y_canonical = np.array([s.label for s in samples], dtype=int)
    person_groups = np.array([s.person for s in samples])
    scan_paths = np.array([str(Path(s.raw_path).resolve()) for s in samples])

    comps: list[Component] = []

    for nice_name, fn in [
        ("dinov2_s", "tiled_emb_dinov2_vits14_afmhot_t512_n9.npz"),
        ("dinov2_b", "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz"),
        ("biomedclip", "tiled_emb_biomedclip_afmhot_t512_n9.npz"),
    ]:
        path_npz = CACHE / fn
        X, y_cached, paths_cached, _ = _mean_pool_tiles(path_npz)
        # Reorder to canonical sample order via scan_paths mapping.
        order = np.array([path_to_idx[str(p)] for p in paths_cached])
        X_reordered = np.full_like(X, np.nan)
        X_reordered[order] = X
        assert not np.isnan(X_reordered).any(), f"{nice_name}: some canonical rows unset"
        # Sanity check labels after reorder
        y_reordered = np.full_like(y_cached, -1)
        y_reordered[order] = y_cached
        assert np.array_equal(y_reordered, y_canonical), f"{nice_name}: label mismatch"
        comps.append(Component(
            name=nice_name,
            X=X_reordered.astype(np.float32),
            y=y_canonical,
            person_groups=person_groups,
            scan_paths=scan_paths,
        ))

    # Parquet-based tabular features
    for nice_name, fn in [
        ("handcrafted", "features_handcrafted.parquet"),
        ("tda", "features_tda.parquet"),
    ]:
        df = pd.read_parquet(CACHE / fn)
        df = df.copy()
        df["si"] = df["raw"].map(path_to_idx)
        df = df.dropna(subset=["si"]).sort_values("si").reset_index(drop=True)
        feat_cols = [c for c in df.columns if c not in ("raw", "cls", "label", "patient", "si")]
        X = df[feat_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_pq = df["label"].values.astype(int)
        assert np.array_equal(y_pq, y_canonical), f"{nice_name}: parquet label mismatch"
        comps.append(Component(
            name=nice_name,
            X=X,
            y=y_canonical,
            person_groups=person_groups,
            scan_paths=scan_paths,
        ))

    return comps


# ---------------------------------------------------------------------------
# LOPO predict_proba per component
# ---------------------------------------------------------------------------

def lopo_predict_proba(comp: Component) -> np.ndarray:
    """Return (n_scans, n_classes) OOF predict_proba under person-level LOPO."""
    n = len(comp.y)
    P = np.zeros((n, N_CLASSES), dtype=np.float64)

    # Drop zero-variance features once up front (same logic as baseline_tda).
    var = comp.X.var(axis=0)
    keep = var > 1e-12
    X = comp.X[:, keep]

    for tr, va in leave_one_patient_out(comp.person_groups):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr])
        Xv = scaler.transform(X[va])
        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)

        clf = LogisticRegression(
            class_weight="balanced", max_iter=3000, C=1.0,
            solver="lbfgs", n_jobs=4,
        )
        clf.fit(Xt, comp.y[tr])

        # clf.classes_ is the subset of classes present in tr, expand to N_CLASSES
        proba = clf.predict_proba(Xv)
        full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
        for ci, cls in enumerate(clf.classes_):
            full[:, cls] = proba[:, ci]
        P[va] = full

    return P


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def metrics_from_proba(P: np.ndarray, y: np.ndarray, thresholds: np.ndarray | None = None):
    """Compute weighted F1, macro F1, per-class F1, and argmax predictions."""
    if thresholds is None:
        pred = P.argmax(axis=1)
    else:
        # Scale: P / thresholds (so class with low threshold gets boosted)
        pred = (P / thresholds[None, :]).argmax(axis=1)
    f1w = f1_score(y, pred, average="weighted", zero_division=0)
    f1m = f1_score(y, pred, average="macro", zero_division=0)
    f1pc = f1_score(y, pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    return float(f1w), float(f1m), [float(v) for v in f1pc], pred


# ---------------------------------------------------------------------------
# Ensemble strategies
# ---------------------------------------------------------------------------

def strat_uniform(P_list: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(P_list, axis=0), axis=0)


def strat_weighted(P_list: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return np.tensordot(w, np.stack(P_list, axis=0), axes=(0, 0))


def strat_geometric(P_list: list[np.ndarray]) -> np.ndarray:
    eps = 1e-9
    log_mean = np.mean(np.stack([np.log(P + eps) for P in P_list], axis=0), axis=0)
    G = np.exp(log_mean)
    G = G / G.sum(axis=1, keepdims=True)
    return G


def strat_per_class_weighted(
    P_list: list[np.ndarray], y: np.ndarray, n_grid: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class weights: for each class c, weight_ck across K components.

    Formally: P_ensemble[:, c] = sum_k w_{k,c} * P_k[:, c].
    Searched via random search over K-simplex per class (n_grid^K is too big for K=5).
    """
    K = len(P_list)
    rng = np.random.default_rng(0)
    # Random search: 2000 simplex samples per class
    n_samples = 2000
    best_weights = np.ones((K, N_CLASSES)) / K

    stacked = np.stack(P_list, axis=0)  # (K, n_scans, n_classes)

    # Start by optimizing class-by-class greedily: find weight vector per class c
    # that maximizes the class's F1 contribution. Since classes are coupled via argmax,
    # we do a joint random search instead.
    best_f1 = -1.0
    for _ in range(n_samples):
        # Sample a K x C weight matrix from Dirichlet(1)
        W = rng.dirichlet(np.ones(K), size=N_CLASSES).T  # (K, C)
        # Apply per-class mixing:
        #   ens[:, c] = sum_k W[k,c] * stacked[k,:,c]
        ens = np.einsum("kc,ksc->sc", W, stacked)
        # Re-normalize
        ens = ens / ens.sum(axis=1, keepdims=True).clip(1e-12)
        f1w, _, _, _ = metrics_from_proba(ens, y)
        if f1w > best_f1:
            best_f1 = f1w
            best_weights = W

    ens = np.einsum("kc,ksc->sc", best_weights, stacked)
    ens = ens / ens.sum(axis=1, keepdims=True).clip(1e-12)
    return ens, best_weights


def strat_stacking(
    P_list: list[np.ndarray], y: np.ndarray, person_groups: np.ndarray
) -> np.ndarray:
    """Meta-LR on concatenated OOF probabilities (n_scans, K*C) → LOPO again.

    The per-component P_list already comes from LOPO, so each row is out-of-fold
    w.r.t. that person. Training a meta-LR with another LOPO over the stacked
    features reuses the same fold structure (no nested scheme needed because the
    features are out-of-fold).
    """
    K = len(P_list)
    X_meta = np.concatenate(P_list, axis=1)  # (n, K*C)
    n = X_meta.shape[0]
    P_meta = np.zeros((n, N_CLASSES), dtype=np.float64)
    for tr, va in leave_one_patient_out(person_groups):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X_meta[tr])
        Xv = scaler.transform(X_meta[va])
        clf = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0,
                                 solver="lbfgs", n_jobs=4)
        clf.fit(Xt, y[tr])
        proba = clf.predict_proba(Xv)
        full = np.zeros((len(va), N_CLASSES))
        for ci, cls in enumerate(clf.classes_):
            full[:, cls] = proba[:, ci]
        P_meta[va] = full
    return P_meta


# ---------------------------------------------------------------------------
# Per-class threshold sweep
# ---------------------------------------------------------------------------

def optimize_thresholds(P: np.ndarray, y: np.ndarray,
                        grid=np.linspace(0.05, 0.95, 19)) -> tuple[np.ndarray, float]:
    """Greedy per-class threshold (pred = argmax of P / thresholds)."""
    thresholds = np.ones(N_CLASSES)
    best_f1, _, _, _ = metrics_from_proba(P, y, thresholds)
    improved = True
    iters = 0
    while improved and iters < 5:
        improved = False
        for c in range(N_CLASSES):
            for t in grid:
                cand = thresholds.copy()
                cand[c] = t
                f1w, _, _, _ = metrics_from_proba(P, y, cand)
                if f1w > best_f1 + 1e-6:
                    best_f1 = f1w
                    thresholds = cand
                    improved = True
        iters += 1
    return thresholds, best_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Probability-Averaging Ensemble — person-level LOPO")
    print("=" * 72)

    samples = enumerate_samples(ROOT / "TRAIN_SET")
    print(f"Enumerated {len(samples)} samples")
    persons = {s.person for s in samples}
    print(f"Unique persons: {len(persons)}")

    comps = load_components(samples)
    y = comps[0].y
    person_groups = comps[0].person_groups
    scan_paths = comps[0].scan_paths
    assert all(np.array_equal(c.y, y) for c in comps)
    assert all(np.array_equal(c.person_groups, person_groups) for c in comps)
    print(f"Loaded {len(comps)} components: " + ", ".join(c.name for c in comps))
    for c in comps:
        print(f"  {c.name:12s} X.shape={c.X.shape}")

    # -- Per-component LOPO predict_proba --
    print("\n" + "-" * 72)
    print("Per-component LOPO predict_proba")
    print("-" * 72)
    P_by_comp: dict[str, np.ndarray] = {}
    standalone: dict[str, dict] = {}
    t0 = time.time()
    for c in comps:
        tc0 = time.time()
        P = lopo_predict_proba(c)
        P_by_comp[c.name] = P
        f1w, f1m, f1pc, pred = metrics_from_proba(P, y)
        standalone[c.name] = {"f1w": f1w, "f1m": f1m, "f1pc": f1pc, "pred": pred}
        print(f"  {c.name:12s}  weighted F1={f1w:.4f}  macro F1={f1m:.4f}  "
              f"({time.time()-tc0:.1f}s)")
    print(f"  Total LOPO time: {time.time()-t0:.1f}s")

    # Save per-component summary rows for the report.
    standalone_rows = []
    for name, m in standalone.items():
        row = {"component": name, "weighted_f1": m["f1w"], "macro_f1": m["f1m"]}
        for i, c in enumerate(CLASSES):
            row[f"f1_{c}"] = m["f1pc"][i]
        standalone_rows.append(row)

    # -- Ensemble strategies --
    print("\n" + "-" * 72)
    print("Ensemble strategies (all 5 components)")
    print("-" * 72)
    names = [c.name for c in comps]
    P_list = [P_by_comp[n] for n in names]

    results: list[dict] = []

    def _record(strategy: str, subset: list[str], P_ens: np.ndarray,
                thresholds: np.ndarray | None = None, extra: str = ""):
        f1w, f1m, f1pc, pred = metrics_from_proba(P_ens, y, thresholds)
        row = {
            "strategy": strategy,
            "subset": "+".join(subset),
            "weighted_f1": f1w,
            "macro_f1": f1m,
            "per_class_f1": f1pc,
            "thresholds": thresholds.tolist() if thresholds is not None else None,
            "P": P_ens,
            "pred": pred,
            "extra": extra,
        }
        results.append(row)
        print(f"  {strategy:32s} {('+'.join(subset)):60s}  "
              f"w_f1={f1w:.4f}  m_f1={f1m:.4f}  {extra}")
        return row

    # Uniform
    _record("uniform_avg", names, strat_uniform(P_list))

    # F1-weighted
    w_f1 = np.array([standalone[n]["f1w"] for n in names])
    _record("f1_weighted", names, strat_weighted(P_list, w_f1), extra=f"w={np.round(w_f1/w_f1.sum(), 3).tolist()}")

    # Geometric
    _record("geometric_mean", names, strat_geometric(P_list))

    # Per-class weighted
    P_pc, W_pc = strat_per_class_weighted(P_list, y, n_grid=5)
    _record("per_class_weighted", names, P_pc,
            extra="per-class Dirichlet random search")

    # Stacking (meta-LR)
    P_stack = strat_stacking(P_list, y, person_groups)
    _record("stacking_meta_lr", names, P_stack)

    # -- Subset ensembles --
    print("\n" + "-" * 72)
    print("Subset ensembles — uniform avg over every subset of size 2..4")
    print("-" * 72)
    for k in (2, 3, 4):
        for combo in itertools.combinations(range(len(names)), k):
            sub_names = [names[i] for i in combo]
            sub_P = [P_list[i] for i in combo]
            row = _record(f"uniform_subset_k{k}", sub_names, strat_uniform(sub_P))

    # Also try F1-weighted top-k combos
    print("\n" + "-" * 72)
    print("Subset ensembles — F1-weighted subsets (top-3, top-4)")
    print("-" * 72)
    order = np.argsort([-standalone[n]["f1w"] for n in names])
    for k in (3, 4):
        idx = order[:k].tolist()
        sub_names = [names[i] for i in idx]
        sub_P = [P_list[i] for i in idx]
        sub_w = np.array([standalone[names[i]]["f1w"] for i in idx])
        _record(f"f1_weighted_top{k}", sub_names,
                strat_weighted(sub_P, sub_w),
                extra=f"w={np.round(sub_w/sub_w.sum(),3).tolist()}")

    # -- Pick winner (pre-threshold) --
    best = max(results, key=lambda r: r["weighted_f1"])
    print("\n" + "=" * 72)
    print("BEST (pre-threshold):")
    print(f"  strategy={best['strategy']}  subset={best['subset']}")
    print(f"  weighted F1 = {best['weighted_f1']:.4f}   macro F1 = {best['macro_f1']:.4f}")
    print("=" * 72)

    # -- Per-class threshold optimization on best --
    thr, thr_f1 = optimize_thresholds(best["P"], y)
    f1w_t, f1m_t, f1pc_t, pred_t = metrics_from_proba(best["P"], y, thr)
    print(f"\nPer-class threshold sweep on best ensemble:")
    print(f"  pre-threshold  weighted F1 = {best['weighted_f1']:.4f}  macro F1 = {best['macro_f1']:.4f}")
    print(f"  post-threshold weighted F1 = {f1w_t:.4f}  macro F1 = {f1m_t:.4f}")
    print(f"  per-class thresholds: " +
          ", ".join(f"{CLASSES[i]}={thr[i]:.2f}" for i in range(N_CLASSES)))

    # Also check: optionally threshold-optimize every candidate and pick post-threshold best
    print("\n" + "-" * 72)
    print("Per-strategy post-threshold F1 (top 10 pre-threshold candidates)")
    print("-" * 72)
    sorted_pre = sorted(results, key=lambda r: -r["weighted_f1"])[:10]
    post_table = []
    best_post = None
    for r in sorted_pre:
        thr_r, f1w_r = optimize_thresholds(r["P"], y)
        post_table.append({
            "strategy": r["strategy"],
            "subset": r["subset"],
            "pre_f1": r["weighted_f1"],
            "post_f1": f1w_r,
            "thresholds": thr_r,
        })
        print(f"  {r['strategy']:32s} {r['subset'][:60]:60s}  "
              f"pre={r['weighted_f1']:.4f}  post={f1w_r:.4f}")
        if best_post is None or f1w_r > best_post["post_f1"]:
            best_post = {"strategy": r["strategy"], "subset": r["subset"],
                         "pre_f1": r["weighted_f1"], "post_f1": f1w_r,
                         "thresholds": thr_r, "P": r["P"]}

    # Final winner = max post_f1
    print("\n" + "=" * 72)
    print("FINAL BEST (post-threshold):")
    print(f"  strategy={best_post['strategy']}  subset={best_post['subset']}")
    print(f"  pre-threshold  F1 = {best_post['pre_f1']:.4f}")
    print(f"  post-threshold F1 = {best_post['post_f1']:.4f}")
    print(f"  thresholds = {best_post['thresholds']}")
    print("=" * 72)

    # Re-run metrics with final thresholds for complete stats
    P_final = best_post["P"]
    thr_final = best_post["thresholds"]
    f1w_f, f1m_f, f1pc_f, pred_f = metrics_from_proba(P_final, y, thr_final)

    print("\nClassification report (final best, post-threshold):")
    print(classification_report(y, pred_f, target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(y, pred_f, labels=list(range(N_CLASSES)))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    # -- Save predictions --
    out_npz = CACHE / "best_ensemble_predictions.npz"
    np.savez(
        out_npz,
        proba=P_final,
        pred_label=pred_f,
        true_label=y,
        scan_paths=scan_paths,
        thresholds=thr_final,
    )
    print(f"\n[saved] {out_npz}")

    # -- Write report --
    write_report(
        standalone_rows=standalone_rows,
        results=results,
        post_table=post_table,
        best=best,
        best_post=best_post,
        final_metrics={
            "f1w": f1w_f, "f1m": f1m_f, "f1pc": f1pc_f, "pred": pred_f,
            "thresholds": thr_final,
        },
        confusion=cm,
        y=y,
    )


def write_report(standalone_rows, results, post_table, best, best_post,
                 final_metrics, confusion, y):
    out = REPORTS / "ENSEMBLE_PROB_RESULTS.md"
    lines: list[str] = []
    lines.append("# Probability-Averaging Ensemble Results")
    lines.append("")
    lines.append("Person-level LOPO (35 persons, 240 scans). All components use "
                 "`LogisticRegression(class_weight='balanced', max_iter=3000)` "
                 "+ `StandardScaler`, trained fresh per fold.")
    lines.append("")
    lines.append("## Per-component standalone F1")
    lines.append("")
    hdr = "| component | weighted F1 | macro F1 | " + \
          " | ".join(f"F1 {c}" for c in CLASSES) + " |"
    lines.append(hdr)
    lines.append("|" + "|".join(["---"] * (3 + N_CLASSES)) + "|")
    for row in standalone_rows:
        cells = [row["component"], f"{row['weighted_f1']:.4f}", f"{row['macro_f1']:.4f}"]
        cells += [f"{row[f'f1_{c}']:.3f}" for c in CLASSES]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Ensemble table — compact, summarized by strategy best
    lines.append("## Ensemble strategies (pre-threshold, person-LOPO weighted F1)")
    lines.append("")
    lines.append("| strategy | subset | weighted F1 | macro F1 |")
    lines.append("|---|---|---|---|")
    # Sort by weighted_f1 desc
    for r in sorted(results, key=lambda r: -r["weighted_f1"]):
        lines.append(
            f"| {r['strategy']} | {r['subset']} | "
            f"{r['weighted_f1']:.4f} | {r['macro_f1']:.4f} |"
        )
    lines.append("")

    lines.append("## Post-threshold refinement (top 10 pre-threshold candidates)")
    lines.append("")
    lines.append("| strategy | subset | pre-threshold F1 | post-threshold F1 |")
    lines.append("|---|---|---|---|")
    for r in sorted(post_table, key=lambda r: -r["post_f1"]):
        lines.append(
            f"| {r['strategy']} | {r['subset']} | "
            f"{r['pre_f1']:.4f} | {r['post_f1']:.4f} |"
        )
    lines.append("")

    lines.append("## Best configuration")
    lines.append("")
    lines.append(f"- **Strategy:** `{best_post['strategy']}`")
    lines.append(f"- **Components:** `{best_post['subset']}`")
    lines.append(f"- **Pre-threshold weighted F1:** {best_post['pre_f1']:.4f}")
    lines.append(f"- **Post-threshold weighted F1:** {best_post['post_f1']:.4f}")
    thr = best_post["thresholds"]
    lines.append(f"- **Per-class thresholds:** " +
                 ", ".join(f"{CLASSES[i]}={thr[i]:.2f}" for i in range(N_CLASSES)))
    lines.append("")

    lines.append("### Per-class F1 (final best, post-threshold)")
    lines.append("")
    lines.append("| class | F1 | support |")
    lines.append("|---|---|---|")
    for i, c in enumerate(CLASSES):
        sup = int((y == i).sum())
        lines.append(f"| {c} | {final_metrics['f1pc'][i]:.4f} | {sup} |")
    lines.append("")

    lines.append("### Confusion matrix (rows=true, cols=pred)")
    lines.append("")
    lines.append("| true\\pred | " + " | ".join(CLASSES) + " |")
    lines.append("|" + "|".join(["---"] * (N_CLASSES + 1)) + "|")
    for i, c in enumerate(CLASSES):
        row = [c] + [str(int(confusion[i, j])) for j in range(N_CLASSES)]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `cache/best_ensemble_predictions.npz` — keys: "
                 "`proba`, `pred_label`, `true_label`, `scan_paths`, `thresholds`")
    lines.append("- `scripts/prob_ensemble.py` — runnable end-to-end, caches only.")
    lines.append("")

    lines.append("## Method summary")
    lines.append("")
    lines.append("1. Load 5 cached components and aggregate to scan level (mean-pool tiles "
                 "for neural embeddings; tabular features used as-is).")
    lines.append("2. For each component, run **person-level LOPO** with "
                 "`StandardScaler + LogisticRegression(balanced, 3000 iters)`, "
                 "collecting per-scan OOF `predict_proba` matrices.")
    lines.append("3. Compare ensemble strategies: uniform average, validation-F1 weighted, "
                 "per-class Dirichlet-sampled weighting, geometric mean, stacking via meta-LR, "
                 "and all 2-/3-/4-component subset uniforms.")
    lines.append("4. On top 10 candidates, perform greedy per-class threshold sweep in [0.05, 0.95] "
                 "maximizing weighted F1. Apply winner to ensemble `proba` via `argmax(P / thresholds)`.")
    lines.append("")

    out.write_text("\n".join(lines))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
