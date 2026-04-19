"""Extract advanced + handcrafted features for every scan, cache to parquet.

Usage:
    .venv/bin/python scripts/extract_advanced_features.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.data import enumerate_samples, preprocess_spm  # noqa: E402
from teardrop.features import extract_all_features  # noqa: E402
from teardrop.features_advanced import extract_all_advanced_features  # noqa: E402

CACHE = ROOT / "cache" / "features_advanced.parquet"


def build_feature_matrix(
    samples,
    target_nm_per_px: float = 90.0,
    crop_size: int = 512,
    include_basic: bool = True,
) -> pd.DataFrame:
    if CACHE.exists():
        print(f"[cache hit] {CACHE}")
        return pd.read_parquet(CACHE)
    CACHE.parent.mkdir(exist_ok=True)

    rows: list[dict] = []
    t0 = time.time()
    times: list[float] = []
    for i, s in enumerate(samples):
        t_scan = time.time()
        try:
            h = preprocess_spm(
                s.raw_path, target_nm_per_px=target_nm_per_px, crop_size=crop_size
            )
            feats: dict[str, float] = {}
            if include_basic:
                feats.update(extract_all_features(h))
            feats.update(extract_all_advanced_features(h))
            row = {
                "raw": str(s.raw_path),
                "cls": s.cls,
                "label": s.label,
                "patient": s.patient,
                "person": s.person,
                **feats,
            }
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"  [err] {s.raw_path.name}: {e}")
            continue
        times.append(time.time() - t_scan)
        if (i + 1) % 10 == 0:
            remain = (len(samples) - (i + 1)) * np.mean(times)
            print(
                f"  [{i+1}/{len(samples)}] elapsed={time.time()-t0:.0f}s "
                f"mean={np.mean(times):.2f}s/scan  ETA={remain:.0f}s"
            )

    df = pd.DataFrame(rows)
    df.to_parquet(CACHE)
    print(f"[saved] {CACHE}  rows={len(df)} cols={df.shape[1]}")
    return df


def main() -> None:
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    print(f"Enumerated {len(samples)} samples")
    df = build_feature_matrix(samples)
    print("\nSummary:")
    print(f"  shape: {df.shape}")
    meta_cols = ["raw", "cls", "label", "patient", "person"]
    n_feats = df.shape[1] - len(meta_cols)
    print(f"  n_features = {n_feats}")
    print(f"  n_person groups = {df['person'].nunique()}")
    print(f"  n_patient groups = {df['patient'].nunique()}")


if __name__ == "__main__":
    main()
