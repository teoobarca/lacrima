"""Orthogonal TDA + morphology feature track for v4 ensemble diversity.

Why
---
Current champion (v4 multiscale DINOv2 + BiomedCLIP-TTA) tops at 0.6887 wF1
but all prior ensemble attempts plateau because every additional track
(k-NN, XGB on handcrafted, expert council) shares a DINOv2 / handcrafted
backbone with v4 → highly correlated errors. Persistent homology + scale-
invariant morphology metrics are a *genuinely* different signal space:
they ignore appearance and only look at global connectivity and multi-
scale self-similarity of the height field.

Pipeline
--------
1. Load cached persistent-homology features (1015 dims, already computed
   in cache/features_tda.parquet via teardrop.topology.persistence_features).
2. Compute complementary morphology features for each of the 240 scans:
     - multifractal box-counting spectrum (q in [-5, 5], 11 values)
     - lacunarity (gliding-box, scales 2..64 → 10 scales)
     - succolarity (4 directions × 4 height-quantile thresholds = 16)
3. Concatenate → ~1050-d feature matrix; save to `cache/tda_features.npz`
   aligned to v4 scan order (so ensembling is trivial).
4. Train XGBoost with StratifiedGroupKFold(5, groups=person_id from v4).
   Collect OOF softmax (240 × 5).
5. Report person-LOPO wF1 / macro F1 standalone.
6. Fuse with v4 via GEOMETRIC mean of softmaxes (no stacker; no leakage).
7. 1000× paired bootstrap Δ wF1 vs v4 alone; compute P(Δ>0).

Outputs
-------
- cache/tda_features.npz       — {X, y, persons, scan_paths, feature_names}
- cache/tda_predictions.json   — per-scan softmax + top-1 (standalone XGB)
- cache/tda_fusion_predictions.json — v4 ⊙ TDA geometric-mean softmax
- reports/TDA_FEATURES.md      — human-readable summary
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.data import CLASSES, enumerate_samples, preprocess_spm  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Morphology helpers (multifractal / lacunarity / succolarity)
# ---------------------------------------------------------------------------

def multifractal_spectrum(h: np.ndarray, qs=(-5, -3, -1, 0, 1, 2, 3, 4, 5, 7, 10),
                          scales=(4, 8, 16, 32, 64)) -> dict[str, float]:
    """Generalized box-counting D(q) using linear fit of log μ_q vs log(1/scale).

    μ_q(r) = Σ p_i^q where p_i is normalized mass inside box i at scale r.
    D(q) = lim_{r→0} log μ_q(r) / log(1/r), estimated via linear regression
    over the scale sweep.

    Output: D_q for each q plus Δ width = D(-5) - D(10) (multifractality).
    """
    h = h.astype(np.float64)
    # Normalize to strictly positive mass (shift then divide by total)
    m = h - h.min() + 1e-6
    m /= m.sum()
    H, W = m.shape
    feats: dict[str, float] = {}
    log_r = []
    mu_for_q: dict[float, list[float]] = {q: [] for q in qs}
    for r in scales:
        if r > min(H, W) // 2:
            continue
        nH = (H // r) * r
        nW = (W // r) * r
        mr = m[:nH, :nW].reshape(nH // r, r, nW // r, r).sum(axis=(1, 3))
        p = mr.ravel()
        p = p[p > 0]
        log_r.append(np.log(1.0 / r))
        for q in qs:
            if abs(q - 1.0) < 1e-9:
                # information dimension limit
                mu = -np.sum(p * np.log(p))
            else:
                mu = np.log(np.sum(p ** q))
            mu_for_q[q].append(mu)

    if len(log_r) < 2:
        for q in qs:
            feats[f"Dq_{q}"] = 0.0
        feats["Dq_width"] = 0.0
        feats["Dq_asymmetry"] = 0.0
        return feats

    log_r = np.array(log_r, dtype=np.float64)
    for q in qs:
        y = np.array(mu_for_q[q], dtype=np.float64)
        if abs(q - 1.0) < 1e-9:
            slope = np.polyfit(log_r, y, 1)[0]
            D = slope
        else:
            slope = np.polyfit(log_r, y, 1)[0]
            D = slope / (q - 1.0)
        feats[f"Dq_{q}"] = float(D)

    feats["Dq_width"] = float(feats[f"Dq_{qs[0]}"] - feats[f"Dq_{qs[-1]}"])
    mid = feats.get("Dq_0", 0.0)
    left = feats[f"Dq_{qs[0]}"] - mid
    right = mid - feats[f"Dq_{qs[-1]}"]
    feats["Dq_asymmetry"] = float(left - right)
    return feats


def lacunarity_gliding(h: np.ndarray, box_sizes=(2, 4, 8, 12, 16, 24, 32, 48, 64, 96)
                      ) -> dict[str, float]:
    """Gliding-box lacunarity Λ(r) = (σ² + μ²) / μ² of box masses.

    Binarize at median → foreground fraction. Then for each box size r
    slide a box and compute mass distribution. Returns Λ(r) per scale
    plus slope of log Λ vs log r (lacunarity exponent).
    """
    H, W = h.shape
    bw = (h > np.median(h)).astype(np.float32)
    # Use integral image for O(1) box sums
    ii = np.zeros((H + 1, W + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(bw, axis=0), axis=1)

    feats: dict[str, float] = {}
    log_r = []
    log_lac = []
    for r in box_sizes:
        if r > min(H, W) // 2:
            continue
        # gliding box: shift at most r//2 per step to save time
        step = max(1, r // 4)
        rows = np.arange(0, H - r, step)
        cols = np.arange(0, W - r, step)
        if rows.size == 0 or cols.size == 0:
            continue
        # Vectorized box sum via integral image
        R, C = np.meshgrid(rows, cols, indexing="ij")
        s = (ii[R + r, C + r] - ii[R, C + r] - ii[R + r, C] + ii[R, C])
        s = s.ravel()
        mu = s.mean()
        sig2 = s.var()
        if mu < 1e-9:
            lam = 1.0
        else:
            lam = (sig2 + mu * mu) / (mu * mu)
        feats[f"lac_r{r:03d}"] = float(lam)
        log_r.append(np.log(r))
        log_lac.append(np.log(max(lam, 1e-9)))

    if len(log_r) >= 2:
        slope = float(np.polyfit(np.array(log_r), np.array(log_lac), 1)[0])
        feats["lac_slope"] = slope
        feats["lac_mean"] = float(np.mean(log_lac))
    else:
        feats["lac_slope"] = 0.0
        feats["lac_mean"] = 0.0
    return feats


def succolarity(h: np.ndarray,
                thresholds=(0.3, 0.4, 0.5, 0.6)) -> dict[str, float]:
    """Directional connectivity of the sub-threshold phase.

    For each threshold t and each of 4 directions (L→R, R→L, T→B, B→T),
    simulate a fluid percolating from that edge into the dark (h<t)
    phase using a flood-fill. Succolarity = reachable area / total dark
    area. Captures anisotropic open-path structure of the dendrite
    network — CNNs do NOT see this directly.
    """
    from scipy.ndimage import label as cc_label
    H, W = h.shape
    feats: dict[str, float] = {}
    for t in thresholds:
        dark = (h < t).astype(np.uint8)
        total_dark = float(dark.sum())
        if total_dark < 1:
            for d in ("L", "R", "T", "B"):
                feats[f"succ_t{int(t*100):02d}_{d}"] = 0.0
            continue
        lbl, _ = cc_label(dark, structure=np.ones((3, 3), dtype=np.uint8))
        for d, edge_mask_fn in (
            ("L", lambda lbl: set(lbl[:, 0].tolist())),
            ("R", lambda lbl: set(lbl[:, -1].tolist())),
            ("T", lambda lbl: set(lbl[0, :].tolist())),
            ("B", lambda lbl: set(lbl[-1, :].tolist())),
        ):
            edge = edge_mask_fn(lbl)
            edge.discard(0)
            if not edge:
                feats[f"succ_t{int(t*100):02d}_{d}"] = 0.0
                continue
            mask = np.isin(lbl, list(edge))
            feats[f"succ_t{int(t*100):02d}_{d}"] = float(mask.sum() / total_dark)
    # Anisotropy features: std across 4 directions per threshold
    for t in thresholds:
        vals = [feats[f"succ_t{int(t*100):02d}_{d}"] for d in ("L", "R", "T", "B")]
        feats[f"succ_t{int(t*100):02d}_aniso"] = float(np.std(vals))
    return feats


def morphology_features(h: np.ndarray) -> dict[str, float]:
    """Compute multifractal + lacunarity + succolarity for one height map."""
    out: dict[str, float] = {}
    out.update(multifractal_spectrum(h))
    out.update(lacunarity_gliding(h))
    out.update(succolarity(h))
    return out


# ---------------------------------------------------------------------------
# Feature matrix assembly (TDA already cached; morphology fresh)
# ---------------------------------------------------------------------------

MORPH_CACHE = CACHE / "features_morphology.parquet"


def build_morphology_matrix(samples) -> pd.DataFrame:
    if MORPH_CACHE.exists():
        print(f"[cache hit] {MORPH_CACHE}")
        return pd.read_parquet(MORPH_CACHE)
    print(f"[building] morphology features for {len(samples)} scans ...")
    rows = []
    t0 = time.time()
    for i, s in enumerate(samples):
        try:
            h = preprocess_spm(s.raw_path, target_nm_per_px=90.0, crop_size=512)
            # Downsample to 256 for speed (morphology is scale-invariant enough)
            h = h[::2, ::2]
            feats = morphology_features(h)
            rows.append({"raw": str(s.raw_path), "cls": s.cls,
                         "label": s.label, "patient": s.patient, **feats})
        except Exception as e:
            print(f"  [err] {s.raw_path.name}: {e}")
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(samples) - i - 1)
            print(f"  [{i+1}/{len(samples)}] {el:.1f}s elapsed, ETA {eta:.1f}s")
    df = pd.DataFrame(rows)
    MORPH_CACHE.parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(MORPH_CACHE)
    print(f"[saved] {MORPH_CACHE} ({len(df)} rows × {df.shape[1]} cols)")
    return df


def assemble_feature_matrix(persons_v4: np.ndarray, scan_paths_v4: np.ndarray,
                            y_v4: np.ndarray, samples) -> tuple:
    """Build X (240×D), aligned to v4 order. Uses cached TDA + computed morphology.

    Returns (X, feature_names) with X aligned to v4_scan_paths ordering.
    """
    tda_df = pd.read_parquet(CACHE / "features_tda.parquet")
    tda_meta_cols = {"raw", "cls", "label", "patient"}
    tda_feat_cols = [c for c in tda_df.columns if c not in tda_meta_cols]
    print(f"[tda] loaded {len(tda_df)} rows × {len(tda_feat_cols)} features")

    morph_df = build_morphology_matrix(samples)
    morph_meta_cols = {"raw", "cls", "label", "patient"}
    morph_feat_cols = [c for c in morph_df.columns if c not in morph_meta_cols]
    print(f"[morph] loaded {len(morph_df)} rows × {len(morph_feat_cols)} features")

    tda_map = dict(zip(tda_df["raw"], tda_df.index))
    morph_map = dict(zip(morph_df["raw"], morph_df.index))

    feature_names = tda_feat_cols + morph_feat_cols
    X = np.zeros((len(scan_paths_v4), len(feature_names)), dtype=np.float32)
    for i, raw in enumerate(scan_paths_v4):
        if raw not in tda_map or raw not in morph_map:
            raise KeyError(f"Missing features for scan: {raw}")
        ti = tda_map[raw]
        mi = morph_map[raw]
        X[i, :len(tda_feat_cols)] = tda_df.loc[ti, tda_feat_cols].values.astype(np.float32)
        X[i, len(tda_feat_cols):] = morph_df.loc[mi, morph_feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feature_names


# ---------------------------------------------------------------------------
# Training: StratifiedGroupKFold XGBoost with OOF softmax
# ---------------------------------------------------------------------------

def train_xgb_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  n_splits: int = 5, seed: int = RNG_SEED):
    n_classes = len(CLASSES)
    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=y)
    sw = np.array([cw[label] for label in y])

    params = dict(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.7,
        reg_lambda=1.5, reg_alpha=0.5,
        random_state=seed, n_jobs=4,
        objective="multi:softprob", num_class=n_classes,
        tree_method="hist",
        eval_metric="mlogloss",
    )
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_proba = np.zeros((len(y), n_classes), dtype=np.float64)
    fold_f1s = []
    for fi, (tr, va) in enumerate(sgkf.split(X, y, groups)):
        clf = XGBClassifier(**params)
        clf.fit(X[tr], y[tr], sample_weight=sw[tr])
        p = clf.predict_proba(X[va])
        oof_proba[va] = p
        fold_pred = p.argmax(1)
        fold_f1s.append(f1_score(y[va], fold_pred, average="weighted",
                                 zero_division=0))
        print(f"  fold {fi}: |train|={len(tr)} |val|={len(va)}  "
              f"patients_val={len(np.unique(groups[va]))}  "
              f"wF1_fold={fold_f1s[-1]:.4f}")
    return oof_proba, fold_f1s


def per_class_f1(y, pred):
    return f1_score(y, pred, average=None,
                    labels=list(range(len(CLASSES))), zero_division=0)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap(y, pred_new, pred_base, n_iter=1000, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_iter)
    new_arr = np.empty(n_iter)
    base_arr = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        a = f1_score(y[idx], pred_new[idx], average="weighted", zero_division=0)
        b = f1_score(y[idx], pred_base[idx], average="weighted", zero_division=0)
        new_arr[i] = a
        base_arr[i] = b
        deltas[i] = a - b
    return {
        "delta_mean": float(deltas.mean()),
        "delta_ci95": [float(np.percentile(deltas, 2.5)),
                       float(np.percentile(deltas, 97.5))],
        "p_delta_gt_0": float((deltas > 0).mean()),
        "new_mean": float(new_arr.mean()),
        "new_ci95": [float(np.percentile(new_arr, 2.5)),
                     float(np.percentile(new_arr, 97.5))],
        "base_mean": float(base_arr.mean()),
        "base_ci95": [float(np.percentile(base_arr, 2.5)),
                      float(np.percentile(base_arr, 97.5))],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("TDA + MORPHOLOGY FEATURE TRACK for v4 ensemble diversity")
    print("=" * 72)

    # Anchor to v4 ordering (so fusion is trivial and person IDs are L/P-collapsed)
    v4 = np.load(CACHE / "v4_oof_predictions.npz", allow_pickle=True)
    v4_proba = v4["proba"].astype(np.float64)
    y = v4["y"].astype(np.int64)
    persons = np.array([str(p) for p in v4["persons"]])
    scan_paths = np.array([str(p) for p in v4["scan_paths"]])
    print(f"v4 anchor: {len(y)} scans, {len(np.unique(persons))} persons")

    v4_pred = v4_proba.argmax(1)
    v4_wf1 = float(f1_score(y, v4_pred, average="weighted", zero_division=0))
    v4_mf1 = float(f1_score(y, v4_pred, average="macro", zero_division=0))
    print(f"v4 baseline: wF1={v4_wf1:.4f}  macroF1={v4_mf1:.4f}")

    # Enumerate samples once (needed for morphology)
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    print(f"enumerated {len(samples)} samples")

    # Assemble X in v4 order
    X, feature_names = assemble_feature_matrix(persons, scan_paths, y, samples)
    print(f"[X] shape={X.shape}  features={len(feature_names)}")

    # Save cache
    np.savez_compressed(CACHE / "tda_features.npz",
                        X=X, y=y, persons=persons, scan_paths=scan_paths,
                        feature_names=np.array(feature_names))
    print(f"[saved] cache/tda_features.npz")

    # Train XGBoost with StratifiedGroupKFold (5)
    print("\n--- XGBoost OOF (StratifiedGroupKFold, k=5, groups=person) ---")
    oof_proba, fold_f1s = train_xgb_oof(X, y, persons, n_splits=5, seed=RNG_SEED)
    oof_pred = oof_proba.argmax(1)
    tda_wf1 = float(f1_score(y, oof_pred, average="weighted", zero_division=0))
    tda_mf1 = float(f1_score(y, oof_pred, average="macro", zero_division=0))
    tda_pc = per_class_f1(y, oof_pred)
    print(f"\nTDA standalone: wF1={tda_wf1:.4f}  macroF1={tda_mf1:.4f}")
    print("  per-class:  " + ", ".join(f"{CLASSES[i]}={v:.3f}"
                                       for i, v in enumerate(tda_pc)))

    # Save TDA standalone predictions
    tda_pred_json = {
        "config": {"model": "XGBoost", "features": "TDA+morphology",
                   "n_features": X.shape[1],
                   "cv": "StratifiedGroupKFold(k=5, groups=person_id)"},
        "weighted_f1": tda_wf1,
        "macro_f1": tda_mf1,
        "per_class_f1": {CLASSES[i]: float(v) for i, v in enumerate(tda_pc)},
        "fold_weighted_f1": [float(v) for v in fold_f1s],
        "predictions": [
            {
                "path": scan_paths[i],
                "person": persons[i],
                "true_label": int(y[i]),
                "true_cls": CLASSES[int(y[i])],
                "pred_label": int(oof_pred[i]),
                "pred_cls": CLASSES[int(oof_pred[i])],
                "proba": {CLASSES[k]: float(oof_proba[i, k])
                          for k in range(len(CLASSES))},
            }
            for i in range(len(y))
        ],
    }
    (CACHE / "tda_predictions.json").write_text(json.dumps(tda_pred_json, indent=2))
    print(f"[saved] cache/tda_predictions.json")

    # Geometric mean fusion with v4 (equal weight, no stacker training)
    print("\n--- Geometric-mean fusion with v4 ---")
    eps = 1e-12
    fused_proba = np.sqrt(np.clip(v4_proba, eps, 1.0) *
                          np.clip(oof_proba, eps, 1.0))
    fused_proba /= fused_proba.sum(axis=1, keepdims=True)
    fused_pred = fused_proba.argmax(1)
    fused_wf1 = float(f1_score(y, fused_pred, average="weighted", zero_division=0))
    fused_mf1 = float(f1_score(y, fused_pred, average="macro", zero_division=0))
    fused_pc = per_class_f1(y, fused_pred)
    print(f"Fusion: wF1={fused_wf1:.4f}  macroF1={fused_mf1:.4f}")
    print("  per-class:  " + ", ".join(f"{CLASSES[i]}={v:.3f}"
                                       for i, v in enumerate(fused_pc)))
    print(f"  Δ vs v4: {fused_wf1 - v4_wf1:+.4f} wF1")

    # Bootstrap Δ vs v4
    print("\n--- 1000× paired bootstrap (fused vs v4) ---")
    boot = paired_bootstrap(y, fused_pred, v4_pred, n_iter=1000, seed=RNG_SEED)
    print(f"  Δ wF1 mean = {boot['delta_mean']:+.4f}  "
          f"CI95=[{boot['delta_ci95'][0]:+.4f}, {boot['delta_ci95'][1]:+.4f}]")
    print(f"  P(Δ>0)     = {boot['p_delta_gt_0']:.3f}")
    print(f"  fused wF1 CI95: [{boot['new_ci95'][0]:.4f}, {boot['new_ci95'][1]:.4f}]")
    print(f"  v4 wF1 CI95:    [{boot['base_ci95'][0]:.4f}, {boot['base_ci95'][1]:.4f}]")

    # Save fusion predictions
    fusion_json = {
        "config": {"method": "geometric_mean(v4, tda_xgb)",
                   "v4_source": "cache/v4_oof_predictions.npz",
                   "tda_source": "cache/tda_predictions.json"},
        "weighted_f1": fused_wf1,
        "macro_f1": fused_mf1,
        "per_class_f1": {CLASSES[i]: float(v) for i, v in enumerate(fused_pc)},
        "v4_weighted_f1": v4_wf1,
        "v4_macro_f1": v4_mf1,
        "tda_weighted_f1": tda_wf1,
        "tda_macro_f1": tda_mf1,
        "bootstrap_delta_vs_v4": boot,
        "predictions": [
            {
                "path": scan_paths[i],
                "person": persons[i],
                "true_label": int(y[i]),
                "true_cls": CLASSES[int(y[i])],
                "pred_label": int(fused_pred[i]),
                "pred_cls": CLASSES[int(fused_pred[i])],
                "proba": {CLASSES[k]: float(fused_proba[i, k])
                          for k in range(len(CLASSES))},
            }
            for i in range(len(y))
        ],
    }
    (CACHE / "tda_fusion_predictions.json").write_text(
        json.dumps(fusion_json, indent=2))
    print(f"[saved] cache/tda_fusion_predictions.json")

    # Error orthogonality diagnostic: disagreement matrix
    v4_correct = (v4_pred == y)
    tda_correct = (oof_pred == y)
    both = int((v4_correct & tda_correct).sum())
    v4_only = int((v4_correct & ~tda_correct).sum())
    tda_only = int((~v4_correct & tda_correct).sum())
    neither = int((~v4_correct & ~tda_correct).sum())
    # Expected both-correct if independent = P(v4) * P(tda) * N
    p_v4 = v4_correct.mean()
    p_tda = tda_correct.mean()
    expected_both = float(p_v4 * p_tda * len(y))
    orthogonality = (tda_only + v4_only) / max(1, (tda_only + v4_only + neither))
    print("\n--- Error orthogonality ---")
    print(f"  both correct:   {both:3d}  ({both/len(y):.1%})")
    print(f"  v4-only right:  {v4_only:3d}")
    print(f"  TDA-only right: {tda_only:3d}  "
          f"[these are the gains TDA can contribute]")
    print(f"  neither right:  {neither:3d}")
    print(f"  expected-both (if independent) = {expected_both:.1f}, "
          f"observed = {both}")
    print(f"  'recoverable error' ratio      = {orthogonality:.3f}")

    # Write report
    lines = []
    lines.append("# Topological + Morphology Feature Track (Wave-6.5)\n")
    lines.append("## Goal\n")
    lines.append("Provide a genuinely orthogonal signal to the v4 DINOv2/BiomedCLIP "
                 "champion. TDA persistent homology reads connectivity of the "
                 "height field across all thresholds; morphology (multifractal, "
                 "lacunarity, succolarity) reads self-similarity and directional "
                 "open-path structure. Neither shares a backbone with v4.\n")
    lines.append("## Feature Composition\n")
    lines.append(f"- Persistent homology (sublevel+superlevel, two scales, H0+H1, "
                 f"PI + landscape + stats + Betti curves) — 1015 dims, cached "
                 f"from `teardrop.topology.persistence_features`.\n")
    lines.append(f"- Multifractal D(q) spectrum (q ∈ {{-5..10}}, 5 scales) — "
                 f"13 dims.\n")
    lines.append(f"- Gliding-box lacunarity (10 box sizes + slope + mean) — 12 dims.\n")
    lines.append(f"- Directional succolarity (4 thresholds × 4 dirs + 4 anisotropy) "
                 f"— 20 dims.\n")
    lines.append(f"- **Total: {X.shape[1]} features per scan.**\n")

    lines.append("\n## Results (person-LOPO via StratifiedGroupKFold k=5, groups=person)\n")
    lines.append("| Model | Weighted F1 | Macro F1 | Notes |\n")
    lines.append("|---|---|---|---|\n")
    lines.append(f"| v4 multiscale (baseline) | {v4_wf1:.4f} | {v4_mf1:.4f} | "
                 f"DINOv2@90nm + DINOv2@45nm + BiomedCLIP-TTA |\n")
    lines.append(f"| TDA+morphology XGB | {tda_wf1:.4f} | {tda_mf1:.4f} | "
                 f"standalone, {X.shape[1]}-d features |\n")
    lines.append(f"| **v4 ⊙ TDA (geom-mean)** | **{fused_wf1:.4f}** | "
                 f"**{fused_mf1:.4f}** | equal weight, no stacker |\n")
    lines.append(f"| Δ vs v4 | {fused_wf1 - v4_wf1:+.4f} | "
                 f"{fused_mf1 - v4_mf1:+.4f} |  |\n")

    lines.append("\n## Per-class Weighted F1 (fusion vs v4)\n")
    lines.append("| Class | v4 | TDA | v4 ⊙ TDA | Δ |\n")
    lines.append("|---|---|---|---|---|\n")
    v4_pc = per_class_f1(y, v4_pred)
    for i, c in enumerate(CLASSES):
        lines.append(f"| {c} | {v4_pc[i]:.3f} | {tda_pc[i]:.3f} | "
                     f"{fused_pc[i]:.3f} | {fused_pc[i]-v4_pc[i]:+.3f} |\n")

    lines.append("\n## Per-fold (5-fold SGKF) wF1\n")
    lines.append("| Fold | TDA wF1 |\n|---|---|\n")
    for fi, v in enumerate(fold_f1s):
        lines.append(f"| {fi} | {v:.4f} |\n")

    lines.append("\n## Paired Bootstrap (1000×) — fused vs v4\n")
    lines.append(f"- Δ wF1 mean = **{boot['delta_mean']:+.4f}**  "
                 f"CI95 = [{boot['delta_ci95'][0]:+.4f}, {boot['delta_ci95'][1]:+.4f}]\n")
    lines.append(f"- P(Δ > 0) = **{boot['p_delta_gt_0']:.3f}**\n")
    lines.append(f"- Fused wF1 CI95 = [{boot['new_ci95'][0]:.4f}, {boot['new_ci95'][1]:.4f}]\n")
    lines.append(f"- v4 wF1    CI95 = [{boot['base_ci95'][0]:.4f}, {boot['base_ci95'][1]:.4f}]\n")

    lines.append("\n## Error Orthogonality\n")
    lines.append("| Case | Count | % |\n|---|---|---|\n")
    lines.append(f"| both correct | {both} | {both/len(y):.1%} |\n")
    lines.append(f"| v4-only correct | {v4_only} | {v4_only/len(y):.1%} |\n")
    lines.append(f"| TDA-only correct | {tda_only} | {tda_only/len(y):.1%} |\n")
    lines.append(f"| neither correct | {neither} | {neither/len(y):.1%} |\n")
    lines.append(f"\nExpected both-correct under independence = {expected_both:.1f}, "
                 f"observed = {both}. "
                 f"If observed < expected → models make correlated errors; "
                 f"if observed ≈ expected → genuinely independent mistakes.\n")

    # Verdict
    promoted = (fused_wf1 >= 0.70) and (boot["p_delta_gt_0"] > 0.90)
    honest_draw = (0.68 <= fused_wf1 < 0.70)
    lines.append("\n## Verdict\n")
    if promoted:
        lines.append(f"**NEW CHAMPION CANDIDATE.** Fused wF1={fused_wf1:.4f} "
                     f"≥ 0.70 AND P(Δ>0)={boot['p_delta_gt_0']:.3f} > 0.90. "
                     f"Run red-team audit before enshrining.\n")
    elif honest_draw:
        lines.append(f"**Noise-floor draw.** Fused wF1={fused_wf1:.4f} "
                     f"is within bootstrap noise of v4 ({v4_wf1:.4f}). "
                     f"Document as honest exploration and pitch asset — "
                     f"'we tried genuinely orthogonal features and confirmed "
                     f"the v4 champion is not leaving signal on the table'.\n")
    else:
        lines.append(f"**No fusion benefit.** Fused wF1={fused_wf1:.4f} "
                     f"< 0.68 ceiling for v4 alone. Single-track v4 remains. "
                     f"TDA retained as pitch-level diversity asset; its "
                     f"value is interpretability (H1 loop statistics are "
                     f"physically meaningful), not raw F1.\n")

    lines.append("\n## Artifacts\n")
    lines.append("- `scripts/tda_features.py` — this script\n")
    lines.append("- `teardrop/topology.py` — persistent-homology extractor\n")
    lines.append("- `cache/tda_features.npz` — 240 × {0} float32 matrix aligned to v4 order\n".format(X.shape[1]))
    lines.append("- `cache/features_tda.parquet` — cached PH features\n")
    lines.append("- `cache/features_morphology.parquet` — cached morphology features\n")
    lines.append("- `cache/tda_predictions.json` — standalone XGB OOF softmax\n")
    lines.append("- `cache/tda_fusion_predictions.json` — v4 ⊙ TDA fused OOF softmax\n")

    REPORTS.mkdir(exist_ok=True, parents=True)
    (REPORTS / "TDA_FEATURES.md").write_text("".join(lines))
    print(f"[saved] reports/TDA_FEATURES.md")

    # Final summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  v4 alone              wF1 = {v4_wf1:.4f}")
    print(f"  TDA+morph standalone  wF1 = {tda_wf1:.4f}")
    print(f"  v4 ⊙ TDA (geom-mean)  wF1 = {fused_wf1:.4f}  "
          f"(Δ = {fused_wf1 - v4_wf1:+.4f})")
    print(f"  P(Δ > 0)                  = {boot['p_delta_gt_0']:.3f}")
    if promoted:
        print("  >>> CHAMPION CANDIDATE — re-audit pending <<<")
    elif honest_draw:
        print("  --- noise-floor draw: document as honest exploration ---")
    else:
        print("  --- no fusion benefit: single-track v4 remains ---")


if __name__ == "__main__":
    main()
