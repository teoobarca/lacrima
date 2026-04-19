"""Inference helper for ensemble_v4_multiscale — shipped champion.

Three-component ensemble (Wave 7 Config D):
  A) DINOv2-B at 90 nm/px, no TTA  (9 tiles -> mean-pool)
  B) DINOv2-B at 45 nm/px, no TTA  (9 tiles -> mean-pool) — finer resolution
  C) BiomedCLIP at 90 nm/px with D4 TTA (9 tiles x 8 augs = 72 views -> mean-pool)

Each component: L2-normalize -> saved StandardScaler -> saved LR -> softmax.
Ensemble = geometric mean of the 3 softmaxes, renormalized, argmax.

Honest person-LOPO F1: weighted 0.6887, macro 0.5541.
See README.md for details and reports/RED_TEAM_MULTISCALE.md for audit.
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


def _load_component(comp_dir: Path) -> dict:
    with open(comp_dir / "meta.json") as f:
        meta = json.load(f)
    arrs = np.load(comp_dir / "classifier.npz")
    return {
        "meta": meta,
        "scaler_means": arrs["scaler_means"],
        "scaler_scales": arrs["scaler_scales"],
        "lr_coef": arrs["lr_coef"],
        "lr_intercept": arrs["lr_intercept"],
    }


class TTAPredictorV4:
    """Multi-scale three-component predictor (shipped v4 champion).

    Pipeline per scan:
      1. 90 nm/px tiling -> DINOv2-B encode -> mean-pool -> softmax (p_A)
      2. 45 nm/px tiling -> DINOv2-B encode -> mean-pool -> softmax (p_B)
      3. 90 nm/px tiling -> D4 aug -> BiomedCLIP encode -> mean-pool -> softmax (p_C)
      4. p = geom_mean(p_A, p_B, p_C), renormalize, argmax
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        with open(self.model_dir / "meta.json") as f:
            self.meta = json.load(f)
        self.classes = self.meta["classes"]
        cfg = self.meta["config"]
        self.tile_size = cfg.get("tile_size", 512)
        self.max_tiles = cfg.get("max_tiles", 9)
        self.render_mode = cfg.get("render_mode", "afmhot")

        # Load components
        self.comp_a = _load_component(self.model_dir / "dinov2b_90nm")
        self.comp_b = _load_component(self.model_dir / "dinov2b_45nm")
        self.comp_c = _load_component(self.model_dir / "biomedclip_tta")

        # Lazy encoders — share the DINOv2-B weights between comp_a and comp_b
        self._enc_dinov2b = None
        self._enc_biomedclip = None

    @classmethod
    def load(cls, model_dir: Path | str | None = None) -> "TTAPredictorV4":
        if model_dir is None:
            model_dir = _HERE
        return cls(Path(model_dir))

    @property
    def encoder_dinov2b(self):
        if self._enc_dinov2b is None:
            self._enc_dinov2b = load_dinov2("vitb14")
        return self._enc_dinov2b

    @property
    def encoder_biomedclip(self):
        if self._enc_biomedclip is None:
            self._enc_biomedclip = load_biomedclip()
        return self._enc_biomedclip

    # -- Tile builders ------------------------------------------------------

    def _tiles_at(self, raw_path: Path, target_nm_per_px: float) -> list[np.ndarray]:
        return preprocess_and_tile_spm(
            raw_path,
            target_nm_per_px=target_nm_per_px,
            tile_size=self.tile_size,
            max_tiles=self.max_tiles,
        )

    def _pils_plain(self, tiles: list[np.ndarray]) -> list:
        return [height_to_pil(np.ascontiguousarray(t), mode=self.render_mode)
                for t in tiles]

    def _pils_d4(self, tiles: list[np.ndarray]) -> list:
        pils = []
        for t in tiles:
            for aug in d4_augmentations(t):
                pils.append(height_to_pil(
                    np.ascontiguousarray(aug), mode=self.render_mode))
        return pils

    # -- Per-component probability -----------------------------------------

    def _component_proba(self, pils: list, encoder, comp: dict) -> np.ndarray:
        tile_emb = encoder.encode(pils, batch_size=len(pils))
        scan_emb = tile_emb.mean(axis=0, keepdims=True).astype(np.float32)
        scan_emb = normalize(scan_emb, norm="l2", axis=1)
        scan_emb = (scan_emb - comp["scaler_means"]) / (comp["scaler_scales"] + 1e-9)
        logits = scan_emb @ comp["lr_coef"].T + comp["lr_intercept"]
        return _softmax(logits)

    # -- Public API ---------------------------------------------------------

    def predict_scan(self, raw_path: Path) -> tuple[str, np.ndarray]:
        raw_path = Path(raw_path)

        # Component A: DINOv2-B at 90 nm/px, no TTA
        tiles_90 = self._tiles_at(raw_path, 90.0)
        pils_90_plain = self._pils_plain(tiles_90)
        p_a = self._component_proba(pils_90_plain, self.encoder_dinov2b, self.comp_a)

        # Component B: DINOv2-B at 45 nm/px, no TTA
        tiles_45 = self._tiles_at(raw_path, 45.0)
        pils_45_plain = self._pils_plain(tiles_45)
        p_b = self._component_proba(pils_45_plain, self.encoder_dinov2b, self.comp_b)

        # Component C: BiomedCLIP at 90 nm/px with D4 TTA
        pils_90_d4 = self._pils_d4(tiles_90)
        p_c = self._component_proba(pils_90_d4, self.encoder_biomedclip, self.comp_c)

        # Geometric mean in log space, renormalize
        eps = 1e-9
        log_avg = (np.log(p_a + eps) + np.log(p_b + eps) + np.log(p_c + eps)) / 3.0
        p = np.exp(log_avg - log_avg.max(axis=1, keepdims=True))
        p = (p / p.sum(axis=1, keepdims=True))[0]
        return self.classes[int(p.argmax())], p

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


# Backwards-compat alias — predict_cli.py looks for `TTAPredictor`
TTAPredictor = TTAPredictorV4


def _cli():
    if len(sys.argv) < 2:
        print("Usage: python -m models.ensemble_v4_multiscale.predict <scan_or_dir>")
        sys.exit(1)
    target = Path(sys.argv[1])
    p = TTAPredictorV4.load()
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
