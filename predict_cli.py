"""Unified CLI for running the shipped classifier on a test set.

Examples
--------
Default (TTA ensemble champion):
    .venv/bin/python predict_cli.py --input /path/to/TEST_SET --output submission.csv

Non-TTA fallback (faster ~8x):
    .venv/bin/python predict_cli.py --model models/ensemble_v1 --input /path/to/TEST_SET

Single-encoder fallback:
    .venv/bin/python predict_cli.py --model models/dinov2b_tiled_v1 --input /path/to/TEST_SET
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


def _load_predictor(model_dir: Path):
    """Load either TTAPredictor (for ensemble_v1_tta) or TearClassifier (for others)."""
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        sys.exit(f"ERROR: {meta_path} not found — is this a valid model bundle?")

    # TTA bundles ship their own predict.py with TTAPredictor
    tta_predict = model_dir / "predict.py"
    if tta_predict.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("tta_predict", tta_predict)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "TTAPredictor"):
            print(f"[load] TTAPredictor from {model_dir}")
            return mod.TTAPredictor.load(model_dir), "tta"

    # Otherwise use standard TearClassifier
    from teardrop.infer import TearClassifier
    print(f"[load] TearClassifier from {model_dir}")
    return TearClassifier.load(model_dir), "standard"


def main():
    ap = argparse.ArgumentParser(description="Predict tear AFM classes on a directory.")
    ap.add_argument("--model", default="models/ensemble_v2_tta",
                    help="Path to model bundle (default: v2 TTA champion, 0.6562 F1)")
    ap.add_argument("--input", required=True, help="Directory with raw SPM scans (recursive)")
    ap.add_argument("--output", default="submission.csv",
                    help="Output CSV path (default: submission.csv)")
    ap.add_argument("--progress-every", type=int, default=10,
                    help="Print progress every N scans (default: 10)")
    args = ap.parse_args()

    model_dir = Path(args.model).resolve()
    input_dir = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not model_dir.exists():
        sys.exit(f"ERROR: model dir not found: {model_dir}")
    if not input_dir.exists():
        sys.exit(f"ERROR: input dir not found: {input_dir}")

    clf, kind = _load_predictor(model_dir)

    t0 = time.time()
    df = clf.predict_directory(input_dir)
    elapsed = time.time() - t0

    # Ensure column order: file, predicted_class, prob_*
    df = df.sort_values("file").reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print(f"\n[done] {len(df)} scans predicted in {elapsed:.1f} s "
          f"({elapsed / max(1, len(df)):.2f} s/scan)")
    print(f"[saved] {output_path}")

    # Summary
    if "predicted_class" in df.columns:
        print("\nPrediction distribution:")
        for cls, n in df["predicted_class"].value_counts().items():
            print(f"  {cls:20s} {n:5d}")
    if "error" in df.columns:
        n_err = df["error"].notna().sum()
        if n_err > 0:
            print(f"\n⚠ {n_err} scans failed (see 'error' column in output CSV)")


if __name__ == "__main__":
    main()
