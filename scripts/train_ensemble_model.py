"""Train and save the shippable ensemble model: DINOv2-B + BiomedCLIP.

Pipeline (per component):
    cached tiled embeddings  →  mean-pool tiles → scan-level emb
                             →  StandardScaler
                             →  LogisticRegression(class_weight='balanced',
                                                   max_iter=3000, C=1.0)
At inference: softmax-average the two components' probabilities, argmax.

No thresholds. No CV. Full 240-scan dataset → overfits each LR (that's fine;
honest generalization is the LOPO estimate in reports/RED_TEAM_ENSEMBLE_AUDIT.md).

Usage:
    .venv/bin/python scripts/train_ensemble_model.py
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
MODEL_DIR = ROOT / "models" / "ensemble_v1"

COMPONENTS = [
    {
        "name": "dinov2b",
        "encoder_name": "dinov2_vitb14",
        "cache": "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz",
        "embed_dim": 768,
        "loader": lambda: load_dinov2("vitb14"),
    },
    {
        "name": "biomedclip",
        "encoder_name": "biomedclip",
        "cache": "tiled_emb_biomedclip_afmhot_t512_n9.npz",
        "embed_dim": 512,
        "loader": load_biomedclip,
    },
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def mean_pool_scan_level(cache_file: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load tiled embeddings, mean-pool to scan level.

    Returns (scan_emb, scan_y, scan_paths).
    """
    z = np.load(cache_file, allow_pickle=True)
    X_tiles = z["X"]
    tile_to_scan = z["tile_to_scan"]
    scan_y = z["scan_y"]
    scan_paths = z["scan_paths"].tolist()

    n_scans = len(scan_y)
    embed_dim = X_tiles.shape[1]
    scan_emb = np.zeros((n_scans, embed_dim), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int32)
    for ti, si in enumerate(tile_to_scan):
        scan_emb[si] += X_tiles[ti]
        counts[si] += 1
    scan_emb /= np.maximum(counts, 1)[:, None]
    return scan_emb, np.asarray(scan_y), scan_paths


def fit_component(scan_emb: np.ndarray, scan_y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    X_std = scaler.fit_transform(scan_emb)
    clf = LogisticRegression(
        class_weight="balanced", max_iter=3000, C=1.0,
        solver="lbfgs", n_jobs=4, random_state=42,
    )
    clf.fit(X_std, scan_y)
    return scaler, clf


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_and_save() -> tuple[EnsembleClassifierBundle, dict]:
    """Fit both components on the full 240-scan dataset and save to disk.

    Returns (bundle, component_embeddings_dict) — the second is used for
    downstream sanity checks without re-reading disk.
    """
    print("=" * 72)
    print("Training ensemble model: DINOv2-B + BiomedCLIP → softmax-avg")
    print("=" * 72)

    # Sanity: class order & scan paths must align between caches.
    ref_paths, ref_y = None, None
    comp_scan_embs: dict[str, np.ndarray] = {}
    fitted_components: list[dict] = []  # scaler, clf, dims…

    for comp in COMPONENTS:
        cache_file = CACHE / comp["cache"]
        if not cache_file.exists():
            print(f"  MISSING cache: {cache_file}")
            print("  → regenerate with scripts/baseline_tiled_ensemble.py")
            sys.exit(1)
        scan_emb, scan_y, scan_paths = mean_pool_scan_level(cache_file)
        if ref_paths is None:
            ref_paths, ref_y = scan_paths, scan_y
        else:
            if scan_paths != ref_paths:
                raise RuntimeError(
                    f"Scan path ordering mismatch between caches for "
                    f"component {comp['name']} — re-run tiled caches.")
            if not np.array_equal(scan_y, ref_y):
                raise RuntimeError("Label mismatch between caches.")
        assert scan_emb.shape[1] == comp["embed_dim"], \
            f"{comp['name']}: expected dim {comp['embed_dim']}, got {scan_emb.shape[1]}"
        comp_scan_embs[comp["name"]] = scan_emb
        print(f"[{comp['name']}] scan embeddings: {scan_emb.shape} (from {cache_file.name})")

    n_scans = len(ref_y)
    class_counts = dict(zip(*np.unique(ref_y, return_counts=True)))
    print(f"\nTraining on {n_scans} scans")
    print(f"Class distribution (label → count): {class_counts}")

    # Fit both components
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

    # Load encoders and build the EnsembleClassifierBundle
    print("\nLoading encoders for packaged bundle (needed by save()/load())...")
    components_objs: list[EnsembleComponent] = []
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
            "render_mode": "afmhot",
            "classifier": "LogisticRegression_balanced",
            "combine": "softmax_mean",
            "components": [
                {"name": "dinov2b", "encoder": "dinov2_vitb14", "embed_dim": 768},
                {"name": "biomedclip", "encoder": "biomedclip", "embed_dim": 512},
            ],
            "trained_on_n_scans": int(n_scans),
            "honest_lopo_f1_raw_argmax": 0.6346,
            "honest_lopo_f1_with_nested_thresholds": 0.6528,
            "note": "We ship raw softmax-avg argmax (no thresholds). "
                    "0.6528 is only reachable with per-class thresholds tuned "
                    "via nested CV, which we did not package here.",
        },
    )
    bundle.save(MODEL_DIR)
    print(f"\n[saved] {MODEL_DIR}")

    # Ensemble training predictions (via scan_embs — same embeddings the LR saw)
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
    header = " " * 20 + " ".join(f"{c:>14}" for c in CLASSES)
    print(header)
    for i, c in enumerate(CLASSES):
        print(f"{c:>20}" + " ".join(f"{v:>14d}" for v in cm[i]))
    print("\nTraining classification report:")
    print(classification_report(ref_y, ensemble_preds,
                                labels=list(range(len(CLASSES))),
                                target_names=CLASSES,
                                zero_division=0))

    return bundle, comp_scan_embs, ref_paths, ref_y


# -----------------------------------------------------------------------------
# Sanity test: 2 random scans per class
# -----------------------------------------------------------------------------

def sanity_test_10_scans(bundle: EnsembleClassifierBundle,
                         comp_scan_embs: dict[str, np.ndarray],
                         scan_paths: list[str],
                         scan_y: np.ndarray,
                         seed: int = 1) -> None:
    """Pick 2 random scans per class, predict via ensemble (same embeddings),
    print top-2 probas + true class. Expected: 100% correct on training data.
    """
    print("\n" + "=" * 72)
    print(f"Sanity test on 10 scans (2 per class, seed={seed})")
    print("=" * 72)
    rng = np.random.default_rng(seed)
    picks: list[int] = []
    for ci in range(len(CLASSES)):
        idx = np.where(scan_y == ci)[0]
        if len(idx) < 2:
            # class 'SucheOko' has 14 scans so OK, but guard anyway
            chosen = rng.choice(idx, size=min(2, len(idx)), replace=False)
        else:
            chosen = rng.choice(idx, size=2, replace=False)
        picks.extend(int(x) for x in chosen)

    # Predict only the picked scans using cached embeddings
    picked_embs = {name: emb[picks] for name, emb in comp_scan_embs.items()}
    probs = bundle.predict_proba_from_embeddings(picked_embs)
    preds = probs.argmax(axis=1)

    correct = 0
    for i, si in enumerate(picks):
        true_c = CLASSES[int(scan_y[si])]
        pred_c = CLASSES[int(preds[i])]
        ok = "[OK]" if true_c == pred_c else "[MISS]"
        if true_c == pred_c:
            correct += 1
        top2 = np.argsort(probs[i])[::-1][:2]
        top2_str = ", ".join(f"{CLASSES[j]}={probs[i, j]:.3f}" for j in top2)
        path_short = Path(scan_paths[si]).name
        print(f"  {ok} {path_short:<50} true={true_c:<20} pred={pred_c:<20} | top2: {top2_str}")
    print(f"\nSanity accuracy on 10 picked training scans: {correct}/{len(picks)}")


def main() -> None:
    bundle, comp_scan_embs, scan_paths, scan_y = train_and_save()
    sanity_test_10_scans(bundle, comp_scan_embs, scan_paths, scan_y, seed=1)

    # Round-trip: load from disk, verify predictions still match for one scan
    print("\nRound-trip load sanity (TearClassifier.load) ...")
    clf = TearClassifier.load(MODEL_DIR)
    assert clf.is_ensemble, "Loaded bundle should be ensemble"
    sample_idx = 0
    picked = {name: emb[sample_idx:sample_idx + 1] for name, emb in comp_scan_embs.items()}
    reloaded_probs = clf.bundle.predict_proba_from_embeddings(picked)[0]
    original_probs = bundle.predict_proba_from_embeddings(picked)[0]
    max_diff = float(np.max(np.abs(reloaded_probs - original_probs)))
    print(f"  max probability diff after reload: {max_diff:.3e} "
          f"(should be ~0 aside from float precision)")
    print("Done.")


if __name__ == "__main__":
    main()
