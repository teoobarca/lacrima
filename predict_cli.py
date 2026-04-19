"""Unified CLI for running the shipped classifier on a test set.

Examples
--------
Default (v4 multi-scale champion, 0.6887 F1):
    .venv/bin/python predict_cli.py --input /path/to/TEST_SET --output submission.csv

Prior v2 TTA champion (0.6562 F1, faster):
    .venv/bin/python predict_cli.py --model models/ensemble_v2_tta --input /path/to/TEST_SET

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
    ap.add_argument("--model", default="models/ensemble_v4_multiscale",
                    help="Path to model bundle (default: v4 multi-scale champion, 0.6887 F1)")
    ap.add_argument("--input", required=True, help="Directory with raw SPM scans (recursive)")
    ap.add_argument("--output", default="submission.csv",
                    help="Output CSV path (default: submission.csv)")
    ap.add_argument("--progress-every", type=int, default=10,
                    help="Print progress every N scans (default: 10)")
    ap.add_argument("--input-format", default="spm", choices=["spm", "bmp"],
                    help="Input format: 'spm' = raw Bruker .NNN (default, full "
                         "accuracy); 'bmp' = 704x575 BMP previews (fallback, "
                         "degraded accuracy — see teardrop/bmp_infer.py)")
    args = ap.parse_args()

    model_dir = Path(args.model).resolve()
    input_dir = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not model_dir.exists():
        sys.exit(f"ERROR: model dir not found: {model_dir}")
    if not input_dir.exists():
        sys.exit(f"ERROR: input dir not found: {input_dir}")

    if args.input_format == "bmp":
        # Only v4 multi-scale has BMP fallback implemented; other bundles
        # should already have been migrated to SPM by the organizer.
        if "ensemble_v4_multiscale" not in str(model_dir):
            print(f"[warn] --input-format bmp is only validated against "
                  f"models/ensemble_v4_multiscale; got {model_dir.name}")
        from teardrop.bmp_infer import BmpPredictorV4
        print(f"[load] BmpPredictorV4 from {model_dir} (BMP fallback path)")
        clf = BmpPredictorV4.load(model_dir)
    else:
        clf, _ = _load_predictor(model_dir)

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
