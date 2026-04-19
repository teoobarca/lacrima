"""Train and save the shippable TTA'd ensemble model: DINOv2-B + BiomedCLIP + D4 TTA.

Same shape as `scripts/train_ensemble_model.py`, but each scan's embedding is
the mean of 72 encoder outputs (9 non-overlapping 512x512 tiles * 8 D4 symmetries)
instead of 9. See `reports/TTA_RESULTS.md` for the honest LOPO estimate
(raw-argmax person-LOPO weighted F1 = 0.6458).

Pipeline (per component):
    cached TTA-augmented embeddings (72 per scan, pre-mean-pooled)
    -> StandardScaler
    -> LogisticRegression(class_weight='balanced', max_iter=3000, C=1.0)

At inference: softmax-average the two components' probabilities, argmax.
No thresholds. Fit on the full 240-scan dataset.

Inference-time requirement: the caller MUST preprocess each scan into
9 tiles * 8 D4 augmentations before calling the bundle. See
`scripts/tta_experiment.py::d4_augmentations` for the canonical D4 orbit.

Usage:
    .venv/bin/python scripts/train_ensemble_tta_model.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.data import CLASSES  # noqa: E402
from teardrop.encoders import load_biomedclip, load_dinov2  # noqa: E402
from teardrop.infer import (  # noqa: E402
    EnsembleClassifierBundle, EnsembleComponent, TearClassifier,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
MODEL_DIR = ROOT / "models" / "ensemble_v1_tta"

COMPONENTS = [
    {
        "name": "dinov2b",
        "encoder_name": "dinov2_vitb14",
        "cache": "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz",
        "embed_dim": 768,
        "loader": lambda: load_dinov2("vitb14"),
    },
    {
        "name": "biomedclip",
        "encoder_name": "biomedclip",
        "cache": "tta_emb_biomedclip_afmhot_t512_n9_d4.npz",
        "embed_dim": 512,
        "loader": load_biomedclip,
    },
]


def fit_component(scan_emb: np.ndarray, scan_y: np.ndarray):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(scan_emb)
    clf = LogisticRegression(
        class_weight="balanced", max_iter=3000, C=1.0,
        solver="lbfgs", n_jobs=4, random_state=42,
    )
    clf.fit(X_std, scan_y)
    return scaler, clf


def main():
    print("=" * 72)
    print("Training TTA ensemble model: DINOv2-B + BiomedCLIP + D4 TTA (72 views / scan)")
    print("=" * 72)

    # Sanity: both caches share the same scan_path ordering (they do by construction
    # because `scripts/tta_experiment.py` iterates the same `enumerate_samples` list).
    ref_paths, ref_y = None, None
    comp_scan_embs: dict[str, np.ndarray] = {}
    fitted_components = []

    for comp in COMPONENTS:
        cache_file = CACHE / comp["cache"]
        if not cache_file.exists():
            print(f"  MISSING cache: {cache_file}")
            print("  Regenerate with:  .venv/bin/python scripts/tta_experiment.py")
            sys.exit(1)
        z = np.load(cache_file, allow_pickle=True)
        X_scan = z["X_scan"]
        scan_y = np.asarray(z["scan_y"])
        scan_paths = [str(p) for p in z["scan_paths"]]
        if ref_paths is None:
            ref_paths, ref_y = scan_paths, scan_y
        else:
            if scan_paths != ref_paths:
                raise RuntimeError(f"Path mismatch between TTA caches at {comp['name']}")
            if not np.array_equal(scan_y, ref_y):
                raise RuntimeError("Label mismatch between caches.")
        assert X_scan.shape[1] == comp["embed_dim"], \
            f"{comp['name']}: expected dim {comp['embed_dim']}, got {X_scan.shape[1]}"
        comp_scan_embs[comp["name"]] = X_scan
        print(f"[{comp['name']}] TTA scan embeddings: {X_scan.shape} (from {cache_file.name})")

    n_scans = len(ref_y)
    print(f"\nTraining on {n_scans} scans")

    for comp in COMPONENTS:
        scan_emb = comp_scan_embs[comp["name"]]
        scaler, clf = fit_component(scan_emb, ref_y)
        train_preds = clf.predict(scaler.transform(scan_emb))
        train_f1 = f1_score(ref_y, train_preds, average="weighted")
        print(f"[{comp['name']}] per-component training F1 (overfit by design): {train_f1:.4f}")
        fitted_components.append({
            "name": comp["name"],
            "encoder_name": comp["encoder_name"],
            "loader": comp["loader"],
            "scaler": scaler,
            "clf": clf,
        })

    # Load encoders for packaged bundle
    print("\nLoading encoders for packaged bundle...")
    components_objs = []
    names = []
    for fc in fitted_components:
        print(f"  loading encoder: {fc['encoder_name']}")
        encoder = fc["loader"]()
        comp_obj = EnsembleComponent(
            encoder=encoder,
            scaler_means=fc["scaler"].mean_.astype(np.float32),
            scaler_scales=fc["scaler"].scale_.astype(np.float32),
            lr_coef=fc["clf"].coef_.astype(np.float32),
            lr_intercept=fc["clf"].intercept_.astype(np.float32),
        )
        components_objs.append(comp_obj)
        names.append(fc["name"])

    bundle = EnsembleClassifierBundle(
        components=components_objs,
        component_names=names,
        classes=CLASSES,
        config={
            "target_nm_per_px": 90.0,
            "tile_size": 512,
            "max_tiles": 9,
            "tta": "D4",
            "tta_group_order": 8,
            "render_mode": "afmhot",
            "classifier": "LogisticRegression_balanced",
            "combine": "softmax_mean",
            "components": [
                {"name": "dinov2b", "encoder": "dinov2_vitb14", "embed_dim": 768},
                {"name": "biomedclip", "encoder": "biomedclip", "embed_dim": 512},
            ],
            "trained_on_n_scans": int(n_scans),
            "honest_lopo_f1_raw_argmax": 0.6458,
            "honest_lopo_f1_non_tta_baseline": 0.6346,
            "delta_vs_non_tta_raw_argmax": 0.0112,
            "note": (
                "TTA'd ensemble: per scan, each of 9 tiles is augmented with the "
                "8 elements of D4 (identity, 3 rotations, 4 flipped rotations). "
                "All 72 PIL images are encoded and mean-pooled to a single scan "
                "embedding, which is then standardized and classified. The caller "
                "MUST expand tiles via the D4 orbit before calling "
                "predict_proba_from_tiles; see scripts/tta_experiment.py::d4_augmentations. "
                "Honest person-LOPO raw-argmax weighted F1 = 0.6458 "
                "(+0.0112 over non-TTA 0.6346 baseline)."
            ),
        },
    )
    bundle.save(MODEL_DIR)
    print(f"\n[saved] {MODEL_DIR}")

    # Ensemble training predictions
    ensemble_probs = bundle.predict_proba_from_embeddings(comp_scan_embs)
    ensemble_preds = ensemble_probs.argmax(axis=1)
    train_f1 = f1_score(ref_y, ensemble_preds, average="weighted")
    print(f"[ensemble] training F1 (full-data fit, overfit by design): {train_f1:.4f}")

    # Save training predictions CSV
    rows = []
    for i, path in enumerate(ref_paths):
        row = {
            "raw_path": path,
            "true_class": CLASSES[int(ref_y[i])],
            "pred_class": CLASSES[int(ensemble_preds[i])],
        }
        for ci, cname in enumerate(CLASSES):
            row[f"prob_{cname}"] = float(ensemble_probs[i, ci])
        rows.append(row)
    pred_df = pd.DataFrame(rows)
    pred_csv = MODEL_DIR / "train_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"[saved] {pred_csv}")

    # Confusion matrix + classification report
    print("\nTraining confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(ref_y, ensemble_preds, labels=list(range(len(CLASSES))))
    print(" " * 20 + " ".join(f"{c:>14}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>20}" + " ".join(f"{v:>14d}" for v in cm[i]))
    print("\nTraining classification report:")
    print(classification_report(ref_y, ensemble_preds,
                                labels=list(range(len(CLASSES))),
                                target_names=CLASSES,
                                zero_division=0))

    # Round-trip sanity
    print("\nRound-trip load sanity (TearClassifier.load) ...")
    clf = TearClassifier.load(MODEL_DIR)
    assert clf.is_ensemble, "Loaded bundle should be ensemble"
    sample_idx = 0
    picked = {name: emb[sample_idx:sample_idx + 1] for name, emb in comp_scan_embs.items()}
    reloaded_probs = clf.bundle.predict_proba_from_embeddings(picked)[0]
    original_probs = bundle.predict_proba_from_embeddings(picked)[0]
    max_diff = float(np.max(np.abs(reloaded_probs - original_probs)))
    print(f"  max probability diff after reload: {max_diff:.3e}")
    print("Done.")


if __name__ == "__main__":
    main()
