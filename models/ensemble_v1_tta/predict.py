"""Inference helper for the TTA'd ensemble.

The saved bundle (`EnsembleClassifierBundle.load(.)`) trained its LRs on
mean-pooled embeddings of 72 views per scan (9 tiles * 8 D4 symmetries). Calling
`bundle.predict_proba_from_tiles(tiles)` with just 9 non-augmented tiles would
feed the LR an out-of-distribution mean and produce miscalibrated probabilities.

This module wires the correct preprocessing chain:

    raw SPM -> 9 tiles -> each tile * D4 -> 72 tiles -> encode per encoder
    -> mean-pool to scan embedding -> per-component LR -> softmax-average

Example:

    from models.ensemble_v1_tta.predict import TTAPredictor
    p = TTAPredictor.load()
    cls, probs = p.predict_scan(Path("TRAIN_SET/ZdraviLudia/2L.001"))

Usage from CLI:

    .venv/bin/python -m models.ensemble_v1_tta.predict TRAIN_SET/path/to/scan.001
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Support being imported or run as script
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from teardrop.data import is_raw_spm  # noqa: E402
from teardrop.encoders import height_to_pil  # noqa: E402
from teardrop.infer import (  # noqa: E402
    EnsembleClassifierBundle, preprocess_and_tile_spm,
)


def d4_augmentations(arr: np.ndarray) -> list[np.ndarray]:
    """Return the 8 elements of the dihedral group D4 applied to a 2D array.

    Kept in lockstep with `scripts/tta_experiment.py::d4_augmentations`.
    """
    rots = [np.rot90(arr, k=k) for k in range(4)]
    flipped = np.fliplr(arr)
    rots_flip = [np.rot90(flipped, k=k) for k in range(4)]
    return rots + rots_flip


class TTAPredictor:
    """Wraps an EnsembleClassifierBundle to do D4-TTA preprocessing before encode."""

    def __init__(self, bundle: EnsembleClassifierBundle):
        self.bundle = bundle
        cfg = bundle.config
        self.target_nm_per_px = cfg.get("target_nm_per_px", 90.0)
        self.tile_size = cfg.get("tile_size", 512)
        self.max_tiles = cfg.get("max_tiles", 9)
        self.render_mode = cfg.get("render_mode", "afmhot")
        assert cfg.get("tta") == "D4", \
            f"Bundle at this path is not configured for D4 TTA: {cfg.get('tta')!r}"

    @classmethod
    def load(cls, model_dir: Path | str | None = None) -> "TTAPredictor":
        if model_dir is None:
            model_dir = _HERE
        bundle = EnsembleClassifierBundle.load(model_dir)
        return cls(bundle)

    def _tta_tiles(self, raw_path: Path) -> list:
        """Preprocess -> 9 tiles -> 8 D4 augs each -> 72 PIL images."""
        tiles = preprocess_and_tile_spm(
            raw_path,
            target_nm_per_px=self.target_nm_per_px,
            tile_size=self.tile_size,
            max_tiles=self.max_tiles,
        )
        pils = []
        for t in tiles:
            for aug in d4_augmentations(t):
                pils.append(height_to_pil(
                    np.ascontiguousarray(aug), mode=self.render_mode))
        return pils

    def predict_scan(self, raw_path: Path) -> tuple[str, np.ndarray]:
        pils = self._tta_tiles(raw_path)
        probs = self.bundle.predict_proba_from_tiles(pils)[0]
        pred_idx = int(probs.argmax())
        return self.bundle.classes[pred_idx], probs

    def predict_directory(self, root: Path | str) -> pd.DataFrame:
        root = Path(root)
        rows = []
        all_raws = sorted([p for p in root.rglob("*") if is_raw_spm(p)])
        for i, p in enumerate(all_raws):
            try:
                cls_pred, probs = self.predict_scan(p)
                row = {"file": str(p.relative_to(root)), "predicted_class": cls_pred}
                for c, pr in zip(self.bundle.classes, probs):
                    row[f"prob_{c}"] = float(pr)
                rows.append(row)
            except Exception as e:
                rows.append({"file": str(p.relative_to(root)), "error": str(e)[:200]})
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(all_raws)}] processed")
        return pd.DataFrame(rows)


def _cli():
    if len(sys.argv) < 2:
        print("Usage: python -m models.ensemble_v1_tta.predict <scan_or_dir>")
        sys.exit(1)
    target = Path(sys.argv[1])
    p = TTAPredictor.load()
    if target.is_dir():
        df = p.predict_directory(target)
        print(df.to_string(index=False))
    else:
        cls, probs = p.predict_scan(target)
        print(f"{target}: {cls}")
        for c, pr in zip(p.bundle.classes, probs):
            print(f"  {c}: {pr:.4f}")


if __name__ == "__main__":
    _cli()
