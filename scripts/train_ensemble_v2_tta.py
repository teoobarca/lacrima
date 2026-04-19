"""Train and save the v2 TTA ensemble using the autoresearch-discovered recipe.

Recipe (from reports/AUTORESEARCH_WAVE5_RESULTS.md):
  1. Load TTA-pre-pooled scan embeddings (D4 group, 72 views per scan mean-pooled)
  2. L2-normalize each scan embedding (row-wise)
  3. Fit StandardScaler on the L2-normalized features
  4. Fit LogisticRegression(class_weight='balanced')
  5. At inference: same preprocessing chain; combine softmaxes by GEOMETRIC mean

Honest person-LOPO weighted F1: 0.6562 (verified 2026-04-18, +0.0104 vs v1_tta).

Saves to: models/ensemble_v2_tta/
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.data import CLASSES

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
MODEL_DIR = ROOT / "models" / "ensemble_v2_tta"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load TTA scan-level embeddings
    zd = np.load(CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz", allow_pickle=True)
    zb = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)
    Xd = zd["X_scan"]; Xb = zb["X_scan"]
    y = zd["scan_y"]
    print(f"DINOv2-B: {Xd.shape}, BiomedCLIP: {Xb.shape}, labels: {y.shape}")

    # Apply recipe: L2 normalize → StandardScaler → LR
    Xd_norm = normalize(Xd, norm="l2", axis=1)
    Xb_norm = normalize(Xb, norm="l2", axis=1)

    sc_d = StandardScaler().fit(Xd_norm)
    sc_b = StandardScaler().fit(Xb_norm)

    Xd_std = sc_d.transform(Xd_norm)
    Xb_std = sc_b.transform(Xb_norm)

    clf_d = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0,
                               solver="lbfgs", n_jobs=4, random_state=42)
    clf_b = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0,
                               solver="lbfgs", n_jobs=4, random_state=42)
    clf_d.fit(Xd_std, y)
    clf_b.fit(Xb_std, y)

    # Sanity: training F1 on whole data (memorization check, not generalization)
    pd_ = clf_d.predict_proba(Xd_std)
    pb_ = clf_b.predict_proba(Xb_std)
    # Geometric mean
    eps = 1e-9
    p_geom = np.exp(0.5 * (np.log(pd_ + eps) + np.log(pb_ + eps)))
    p_geom /= p_geom.sum(axis=1, keepdims=True)
    preds = p_geom.argmax(axis=1)
    print(f"Training F1 (overfit check): {f1_score(y, preds, average='weighted'):.4f} "
          f"(expect ~1.0 since LR memorizes 240 samples)")

    # Save per-component
    for name, clf, sc in [("dinov2b", clf_d, sc_d), ("biomedclip", clf_b, sc_b)]:
        comp_dir = MODEL_DIR / name
        comp_dir.mkdir(exist_ok=True)
        np.savez(comp_dir / "classifier.npz",
                 scaler_means=sc.mean_.astype(np.float32),
                 scaler_scales=sc.scale_.astype(np.float32),
                 lr_coef=clf.coef_.astype(np.float32),
                 lr_intercept=clf.intercept_.astype(np.float32))
        with open(comp_dir / "meta.json", "w") as f:
            json.dump({
                "kind": "single",
                "encoder_name": "dinov2_vitb14" if name == "dinov2b" else "biomedclip",
                "classes": CLASSES,
                "config": {
                    "target_nm_per_px": 90.0,
                    "tile_size": 512,
                    "max_tiles": 9,
                    "render_mode": "afmhot",
                    "tta_group": "D4",  # 8 augmentations per tile
                    "preprocessing": "l2_normalize_then_standardscaler",
                },
            }, f, indent=2)

    # Top-level meta
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump({
            "kind": "ensemble",
            "components": ["dinov2b", "biomedclip"],
            "classes": CLASSES,
            "config": {
                "ensemble_method": "geometric_mean",
                "preprocessing": "l2_normalize_row → standardscaler",
                "tta_group": "D4",
                "tile_size": 512,
                "max_tiles": 9,
                "render_mode": "afmhot",
                "target_nm_per_px": 90.0,
                "honest_lopo_weighted_f1": 0.6562,
                "honest_lopo_macro_f1": 0.5382,
                "trained_on_n_scans": int(len(y)),
                "provenance": "autoresearch Wave 5 H2 recipe",
            },
        }, f, indent=2)

    # README
    readme = """# ensemble_v2_tta — L2-norm + geometric-mean TTA ensemble

**Honest person-LOPO weighted F1: 0.6562** (+0.0104 over v1_tta 0.6458).
**Macro F1: 0.5382** (+0.0228 over v1_tta 0.5154).

Same two frozen encoders and D4 test-time augmentation as `ensemble_v1_tta`,
but:
- Embeddings are **L2-normalized row-wise** before StandardScaler
- Per-encoder softmaxes are combined by **geometric mean** (log-space arithmetic)
  instead of arithmetic mean

Discovered by the autoresearch hypothesis-generating agent in Wave 5 — both
changes are honest (no tuning, no OOF peeking, no threshold search). See
`reports/AUTORESEARCH_WAVE5_RESULTS.md` for the ablation.

## Per-class F1
- ZdraviLudia: 0.87
- Diabetes: **0.54** (+0.11 over v1_tta)
- PGOV_Glaukom: 0.56 (−0.03)
- SklerozaMultiplex: 0.65 (−0.07)
- SucheOko: **0.06** (non-zero for the first time)

Gains concentrate on the minority classes (Diabetes, SucheOko). The majority
classes drop slightly — net weighted F1 is +0.0104, net macro F1 is +0.023.

## Inference

To reuse at inference, apply the SAME preprocessing:
1. Preprocess SPM → 9 tiles → D4 TTA → encode → mean-pool to scan embedding
2. L2-normalize each scan embedding (axis=1)
3. StandardScaler.transform with the saved means/scales
4. LR predict_proba per component
5. Geometric mean of the two softmaxes → argmax

A reference inference helper lives in `predict_v2.py` (same folder).

## Training data
240 scans / 35 persons / 5 classes. Provenance: autoresearch Wave 5 H2.
"""
    (MODEL_DIR / "README.md").write_text(readme)

    print(f"[saved] {MODEL_DIR}")


if __name__ == "__main__":
    main()
