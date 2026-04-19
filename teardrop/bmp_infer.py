"""BMP preview fallback inference path.

If organizers supply BMP preview files (704x575 RGB rendered previews) instead
of raw Bruker .NNN SPM files, we can still run inference — at degraded accuracy
because of rendering + watermark loss.

BMP layout (verified empirically across all 5 classes, see reports/BMP_AND_OPEN_SET.md):
    - 704x575 RGB uint8
    - Data region: rows [8, 531), cols [93, 616)  -> 523x523 pixels
    - Axis labels "0.0 ... 92.5 um" are burned in BELOW row 531
    - Black horizontal axis line at row ~535

The burn-in scale labels (0.0 .. X um) can leak scan-range information, so we
CROP STRICTLY to the data region and discard everything else.

Pipeline (compared to the raw SPM pipeline):
    raw SPM:  load_height -> plane_level -> resample -> robust_normalize -> tile
    BMP:      load RGB    -> crop 523x523 data region          -> tile

There is no height channel to plane-level, no pixel-size unification (everything
is rendered at the same 704x575 regardless of raw scan size), and no robust
normalize (8-bit RGB already).  Tiling then follows the 9x 224^2 scheme so the
encoder input shape matches training.

This file exposes two things:
    preprocess_bmp(path) -> list[PIL.Image]   # 9 x 224-ish tiles
    BmpTTAPredictorV4                         # wraps v4 multi-scale for BMP input

The BMP path is INTENTIONALLY a graceful-degradation fallback.  Do not expect
it to match the raw-SPM F1 of 0.6887.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import normalize


# --- Data-region crop (discovered empirically) ----------------------------

BMP_DATA_ROW_LO = 8
BMP_DATA_ROW_HI = 531   # exclusive
BMP_DATA_COL_LO = 93
BMP_DATA_COL_HI = 616   # exclusive
BMP_EXPECTED_SIZE = (704, 575)  # (W, H) reported by PIL

# Standard encoder input is 224x224.  We cut the 523x523 data region into a
# 3x3 grid of overlapping tiles to produce 9 tiles of ~224^2 each.
BMP_TILE_SIZE = 224
BMP_TILES_PER_SIDE = 3
BMP_N_TILES = BMP_TILES_PER_SIDE * BMP_TILES_PER_SIDE  # 9


def _crop_data_region(arr: np.ndarray) -> np.ndarray:
    """Return the central ~523x523 RGB data region as uint8 array."""
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB array, got shape={arr.shape}")
    H, W, _ = arr.shape
    # Be lenient about size — if the BMP is exactly the expected 704x575, use
    # absolute coords; otherwise fall back to proportional coords.
    if (W, H) == BMP_EXPECTED_SIZE:
        return arr[BMP_DATA_ROW_LO:BMP_DATA_ROW_HI,
                   BMP_DATA_COL_LO:BMP_DATA_COL_HI, :]
    # Proportional fallback: strip 1.5% top, 7.5% bottom, 13% left, 12.5% right
    # (chosen to mimic the absolute crop).  This keeps inference working even
    # if the organizer re-renders at a different size.
    r_lo = int(round(H * BMP_DATA_ROW_LO / BMP_EXPECTED_SIZE[1]))
    r_hi = int(round(H * BMP_DATA_ROW_HI / BMP_EXPECTED_SIZE[1]))
    c_lo = int(round(W * BMP_DATA_COL_LO / BMP_EXPECTED_SIZE[0]))
    c_hi = int(round(W * BMP_DATA_COL_HI / BMP_EXPECTED_SIZE[0]))
    return arr[r_lo:r_hi, c_lo:c_hi, :]


def _grid_tiles(arr: np.ndarray, tile_size: int = BMP_TILE_SIZE,
                tiles_per_side: int = BMP_TILES_PER_SIDE) -> list[np.ndarray]:
    """Cut HxWx3 into a 3x3 grid of `tile_size` square tiles.

    The data region (523x523) is slightly smaller than 3*224=672, so we
    overlap tiles evenly: tile i starts at y = i * (H - tile_size) / (n-1).
    """
    H, W, _ = arr.shape
    if H < tile_size or W < tile_size:
        # pad-reflect to minimum tile size (rare for 523 vs 224)
        pad_h = max(0, tile_size - H)
        pad_w = max(0, tile_size - W)
        arr = np.pad(arr, ((pad_h // 2, pad_h - pad_h // 2),
                           (pad_w // 2, pad_w - pad_w // 2),
                           (0, 0)), mode="reflect")
        H, W, _ = arr.shape

    ys = np.linspace(0, H - tile_size, tiles_per_side).round().astype(int)
    xs = np.linspace(0, W - tile_size, tiles_per_side).round().astype(int)
    out = []
    for y in ys:
        for x in xs:
            out.append(arr[y:y + tile_size, x:x + tile_size, :].copy())
    return out


def preprocess_bmp(path: Path | str,
                   tile_size: int = BMP_TILE_SIZE,
                   tiles_per_side: int = BMP_TILES_PER_SIDE) -> list[Image.Image]:
    """BMP preview -> list of 9 PIL tile images (RGB, `tile_size`^2 each).

    Steps:
        1. Load BMP as RGB numpy.
        2. Crop data region (discard axis labels / scale bar watermark).
        3. Cut into 3x3 grid of tiles, each `tile_size`^2.
    No normalization, plane-leveling or resampling — the BMP is already an
    8-bit rendered image.
    """
    path = Path(path)
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    data = _crop_data_region(arr)
    np_tiles = _grid_tiles(data, tile_size=tile_size,
                           tiles_per_side=tiles_per_side)
    return [Image.fromarray(t, mode="RGB") for t in np_tiles]


# --- v4 BMP predictor wrapper --------------------------------------------

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class _Component:
    scaler_means: np.ndarray
    scaler_scales: np.ndarray
    lr_coef: np.ndarray
    lr_intercept: np.ndarray

    @classmethod
    def load(cls, comp_dir: Path) -> "_Component":
        arrs = np.load(comp_dir / "classifier.npz")
        return cls(
            scaler_means=arrs["scaler_means"],
            scaler_scales=arrs["scaler_scales"],
            lr_coef=arrs["lr_coef"],
            lr_intercept=arrs["lr_intercept"],
        )

    def predict_proba(self, scan_emb: np.ndarray) -> np.ndarray:
        scan_emb = normalize(scan_emb.astype(np.float32), norm="l2", axis=1)
        X_std = (scan_emb - self.scaler_means) / (self.scaler_scales + 1e-9)
        logits = X_std @ self.lr_coef.T + self.lr_intercept
        return _softmax(logits)


class BmpPredictorV4:
    """BMP fallback predictor for v4 multi-scale ensemble.

    Because the BMP renders at a single fixed resolution, we cannot produce
    both 90 nm/px and 45 nm/px views.  We therefore re-use the SAME 9 tiles
    for BOTH DINOv2-B branches plus the BiomedCLIP branch, and take the
    geometric mean of the 3 resulting softmaxes.

    This implicitly treats the 45 nm branch as a second DINOv2-B head with
    its own (independently-trained) LR — useful as a weak ensemble of two
    differently-initialized linear heads over the same embedding.
    """

    def __init__(self, model_dir: Path | str | None = None):
        if model_dir is None:
            model_dir = _ROOT / "models" / "ensemble_v4_multiscale"
        self.model_dir = Path(model_dir)
        with open(self.model_dir / "meta.json") as f:
            self.meta = json.load(f)
        self.classes = self.meta["classes"]

        self.comp_a = _Component.load(self.model_dir / "dinov2b_90nm")
        self.comp_b = _Component.load(self.model_dir / "dinov2b_45nm")
        self.comp_c = _Component.load(self.model_dir / "biomedclip_tta")

        self._enc_dinov2 = None
        self._enc_biomedclip = None

    @classmethod
    def load(cls, model_dir: Path | str | None = None) -> "BmpPredictorV4":
        return cls(model_dir=model_dir)

    @property
    def encoder_dinov2b(self):
        if self._enc_dinov2 is None:
            if str(_ROOT) not in sys.path:
                sys.path.insert(0, str(_ROOT))
            from teardrop.encoders import load_dinov2
            self._enc_dinov2 = load_dinov2("vitb14")
        return self._enc_dinov2

    @property
    def encoder_biomedclip(self):
        if self._enc_biomedclip is None:
            if str(_ROOT) not in sys.path:
                sys.path.insert(0, str(_ROOT))
            from teardrop.encoders import load_biomedclip
            self._enc_biomedclip = load_biomedclip()
        return self._enc_biomedclip

    def _scan_embedding(self, encoder, pil_tiles: list[Image.Image]) -> np.ndarray:
        tile_emb = encoder.encode(pil_tiles, batch_size=len(pil_tiles))
        return tile_emb.mean(axis=0, keepdims=True).astype(np.float32)

    def predict_scan(self, bmp_path: Path | str) -> tuple[str, np.ndarray]:
        pil_tiles = preprocess_bmp(bmp_path)

        emb_dinov2 = self._scan_embedding(self.encoder_dinov2b, pil_tiles)
        emb_biomed = self._scan_embedding(self.encoder_biomedclip, pil_tiles)

        p_a = self.comp_a.predict_proba(emb_dinov2)  # 90nm head
        p_b = self.comp_b.predict_proba(emb_dinov2)  # 45nm head (reused emb)
        p_c = self.comp_c.predict_proba(emb_biomed)

        eps = 1e-9
        log_avg = (np.log(p_a + eps) + np.log(p_b + eps) + np.log(p_c + eps)) / 3.0
        p = np.exp(log_avg - log_avg.max(axis=1, keepdims=True))
        p = (p / p.sum(axis=1, keepdims=True))[0]
        return self.classes[int(p.argmax())], p

    def predict_directory(self, root: Path | str) -> pd.DataFrame:
        root = Path(root)
        rows = []
        all_bmps = sorted([p for p in root.rglob("*.bmp")])
        for i, p in enumerate(all_bmps):
            try:
                cls_pred, probs = self.predict_scan(p)
                row = {"file": str(p.relative_to(root)),
                       "predicted_class": cls_pred}
                for c, pr in zip(self.classes, probs):
                    row[f"prob_{c}"] = float(pr)
                rows.append(row)
            except Exception as e:
                rows.append({"file": str(p.relative_to(root)),
                             "error": str(e)[:200]})
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(all_bmps)}] processed")
        return pd.DataFrame(rows)


# Alias so predict_cli.py can resolve it
BmpTTAPredictorV4 = BmpPredictorV4
