"""Inference helper for ensemble_v2_tta — L2-norm + geometric-mean recipe.

Key differences vs v1_tta:
  1. Scan embeddings are L2-normalized (axis=1) BEFORE StandardScaler
  2. Per-encoder softmaxes combined by geometric mean (log-space arithmetic)
     instead of arithmetic mean

Honest person-LOPO F1: 0.6562 (weighted), 0.5382 (macro).
See reports/AUTORESEARCH_WAVE5_RESULTS.md for discovery and ablation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from teardrop.data import is_raw_spm  # noqa: E402
from teardrop.encoders import height_to_pil, load_biomedclip, load_dinov2  # noqa: E402
from teardrop.infer import preprocess_and_tile_spm  # noqa: E402


def d4_augmentations(arr: np.ndarray) -> list[np.ndarray]:
    rots = [np.rot90(arr, k=k) for k in range(4)]
    flipped = np.fliplr(arr)
    rots_flip = [np.rot90(flipped, k=k) for k in range(4)]
    return rots + rots_flip


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


class TTAPredictorV2:
    """L2-norm + geometric-mean ensemble over D4 TTA views."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        with open(model_dir / "meta.json") as f:
            self.meta = json.load(f)
        self.classes = self.meta["classes"]
        cfg = self.meta["config"]
        self.target_nm_per_px = cfg.get("target_nm_per_px", 90.0)
        self.tile_size = cfg.get("tile_size", 512)
        self.max_tiles = cfg.get("max_tiles", 9)
        self.render_mode = cfg.get("render_mode", "afmhot")

        # Load components
        dinov2b_arrs = np.load(model_dir / "dinov2b" / "classifier.npz")
        biomedclip_arrs = np.load(model_dir / "biomedclip" / "classifier.npz")
        self.sc_d_mean = dinov2b_arrs["scaler_means"]
        self.sc_d_scale = dinov2b_arrs["scaler_scales"]
        self.lr_d_coef = dinov2b_arrs["lr_coef"]
        self.lr_d_intercept = dinov2b_arrs["lr_intercept"]
        self.sc_b_mean = biomedclip_arrs["scaler_means"]
        self.sc_b_scale = biomedclip_arrs["scaler_scales"]
        self.lr_b_coef = biomedclip_arrs["lr_coef"]
        self.lr_b_intercept = biomedclip_arrs["lr_intercept"]

        # Lazy-load encoders
        self._enc_d = None
        self._enc_b = None

    @classmethod
    def load(cls, model_dir: Path | str | None = None) -> "TTAPredictorV2":
        if model_dir is None:
            model_dir = _HERE
        return cls(Path(model_dir))

    @property
    def encoder_dinov2b(self):
        if self._enc_d is None:
            self._enc_d = load_dinov2("vitb14")
        return self._enc_d

    @property
    def encoder_biomedclip(self):
        if self._enc_b is None:
            self._enc_b = load_biomedclip()
        return self._enc_b

    def _tta_tiles(self, raw_path: Path) -> list:
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

    def _component_proba(self, pils: list, encoder, sc_mean, sc_scale,
                         lr_coef, lr_intercept) -> np.ndarray:
        tile_emb = encoder.encode(pils, batch_size=len(pils))
        scan_emb = tile_emb.mean(axis=0, keepdims=True).astype(np.float32)
        # L2 normalize THEN StandardScaler — the v2 recipe
        scan_emb = normalize(scan_emb, norm="l2", axis=1)
        scan_emb = (scan_emb - sc_mean) / (sc_scale + 1e-9)
        logits = scan_emb @ lr_coef.T + lr_intercept
        return _softmax(logits)

    def predict_scan(self, raw_path: Path) -> tuple[str, np.ndarray]:
        pils = self._tta_tiles(raw_path)
        p_d = self._component_proba(pils, self.encoder_dinov2b,
                                    self.sc_d_mean, self.sc_d_scale,
                                    self.lr_d_coef, self.lr_d_intercept)
        p_b = self._component_proba(pils, self.encoder_biomedclip,
                                    self.sc_b_mean, self.sc_b_scale,
                                    self.lr_b_coef, self.lr_b_intercept)
        # Geometric mean in log space, renormalize
        eps = 1e-9
        log_avg = 0.5 * (np.log(p_d + eps) + np.log(p_b + eps))
        p = np.exp(log_avg - log_avg.max(axis=1, keepdims=True))
        p = (p / p.sum(axis=1, keepdims=True))[0]
        pred_idx = int(p.argmax())
        return self.classes[pred_idx], p

    def predict_directory(self, root: Path | str) -> pd.DataFrame:
        root = Path(root)
        rows = []
        all_raws = sorted([p for p in root.rglob("*") if is_raw_spm(p)])
        for i, p in enumerate(all_raws):
            try:
                cls_pred, probs = self.predict_scan(p)
                row = {"file": str(p.relative_to(root)), "predicted_class": cls_pred}
                for c, pr in zip(self.classes, probs):
                    row[f"prob_{c}"] = float(pr)
                rows.append(row)
            except Exception as e:
                rows.append({"file": str(p.relative_to(root)), "error": str(e)[:200]})
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(all_raws)}] processed")
        return pd.DataFrame(rows)


# Backwards-compat alias for predict_cli.py discovery
TTAPredictor = TTAPredictorV2


def _cli():
    if len(sys.argv) < 2:
        print("Usage: python -m models.ensemble_v2_tta.predict <scan_or_dir>")
        sys.exit(1)
    target = Path(sys.argv[1])
    p = TTAPredictorV2.load()
    if target.is_dir():
        df = p.predict_directory(target)
        print(df.to_string(index=False))
    else:
        cls, probs = p.predict_scan(target)
        print(f"{target}: {cls}")
        for c, pr in zip(p.classes, probs):
            print(f"  {c}: {pr:.4f}")


if __name__ == "__main__":
    _cli()
