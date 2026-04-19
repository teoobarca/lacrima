"""Data loading + preprocessing for tear AFM scans.

Pipeline:
    raw SPM file →
    load_height (Bruker SPM via AFMReader) →
    plane_level (1st-order polynomial subtraction) →
    resample_to_pixel_size (unify nm/px across all scans) →
    robust_normalize (percentile clip → [0,1]) →
    center_crop_or_pad (fixed pixel size for ML)
"""
from __future__ import annotations

import io
import logging
import re
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

logging.getLogger().setLevel(logging.WARNING)
# AFMReader uses loguru which doesn't respect stdlib logging — silence it directly.
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
except Exception:
    pass

from AFMReader.spm import load_spm  # noqa: E402


CLASSES = ["ZdraviLudia", "Diabetes", "PGOV_Glaukom", "SklerozaMultiplex", "SucheOko"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------------------------
# Filename / patient-id parsing
# ---------------------------------------------------------------------------

_SCAN_INDEX_RE = re.compile(r"\.(\d{3,})$")


def is_raw_spm(p: Path) -> bool:
    """True if filename ends with .NNN where NNN is a numeric scan index."""
    return p.is_file() and bool(_SCAN_INDEX_RE.search(p.name))


def is_bmp_preview(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".bmp"


def patient_id(path: Path) -> str:
    """Extract patient/session ID from filename.

    Drops the `.NNN` scan index and `_1.bmp` BMP suffix.
    NOTE: this ID still treats LEFT and RIGHT eye of the same person as separate.
    For person-level CV use `person_id(path)` instead.
    """
    name = path.name
    if name.lower().endswith(".bmp"):
        name = name[:-4]
        if name.endswith("_1"):
            name = name[:-2]
    name = _SCAN_INDEX_RE.sub("", name)
    return name


# Eye-side tokens that should be collapsed when grouping by *person*.
# L = ľavé (left), P = pravé (right). M/V/O are sample-prep modifiers.
_EYE_TOKENS_RE = re.compile(r"_?(LM|PM|LV|PV|LO|PO)([_-]|$)")


def person_id(path: Path) -> str:
    """Stricter than `patient_id`: also collapses L/R eye of same person.

    Pairs that get merged:
        ZdraviLudia: 2L ↔ 2P, 8L ↔ 8P, 1L_M ↔ 1P (the trailing L/P drops)
        SklerozaMultiplex: 1-SM-LM-18 ↔ 1-SM-PM-18, 20_3-SM-LV-18 ↔ 20_4-SM-PV-18
        PGOV_Glaukom: 21_LV_PGOV ↔ (would-be) 21_PV_PGOV
        Diabetes: DM_01.03.2024_LO ↔ (would-be) DM_01.03.2024_PO

    Rule: replace any 2-char eye token (LM/PM/LV/PV/LO/PO) with `EYE`,
    and strip trailing single L/P (e.g. '2L' → '2'). Then for SM samples
    where files differ only in `_<digit>-SM-` (sample-number) collapse those too.
    """
    pid = patient_id(path)
    pid = _EYE_TOKENS_RE.sub(lambda m: "EYE" + m.group(2), pid)
    # Trailing single L or P (e.g. '2L', '9P', '5P')
    if len(pid) >= 2 and pid[-1] in ("L", "P") and pid[-2].isdigit():
        pid = pid[:-1]
    # 'NL_M' style (e.g. '1L_M' is left-eye-with-M-suffix)
    pid = re.sub(r"^(\d+)L_M$", r"\1", pid)
    # SM sample-number collapse: '100_7-SM-EYE-18' and '100_8-SM-EYE-18' → same
    # Pattern: '<digits>_<digit>-SM-...'  → '<digits>-SM-...'
    pid = re.sub(r"^(\d+)_\d+-SM-", r"\1-SM-", pid)
    # '20_LM_SM-SS' style: had leading number with eye token; already handled by EYE replace
    return pid


def class_of(path: Path, data_root: Path) -> str:
    rel = path.relative_to(data_root)
    return rel.parts[0]


# ---------------------------------------------------------------------------
# SPM loading + preprocessing
# ---------------------------------------------------------------------------

@dataclass
class HeightMap:
    height: np.ndarray  # 2D float array (nm)
    pixel_nm: float     # scale factor — 1 px = pixel_nm nanometers


def load_height(path: Path, channel: str = "Height") -> HeightMap:
    """Load Bruker SPM file → HeightMap. Silences AFMReader logs."""
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        arr, px_nm = load_spm(file_path=path, channel=channel)
    return HeightMap(height=np.asarray(arr, dtype=np.float32), pixel_nm=float(px_nm))


def plane_level(h: np.ndarray) -> np.ndarray:
    """Subtract a 1st-order polynomial fit (removes scanner tilt)."""
    H, W = h.shape
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    A = np.column_stack([np.ones(x.size), x.ravel(), y.ravel()])
    coef, *_ = np.linalg.lstsq(A, h.ravel().astype(np.float64), rcond=None)
    plane = (coef[0] + coef[1] * x + coef[2] * y).astype(np.float32)
    return h - plane


def resample_to_pixel_size(
    h: np.ndarray,
    src_nm_per_px: float,
    target_nm_per_px: float,
) -> np.ndarray:
    """Resample 2D height map so 1 px == target_nm_per_px nanometers."""
    if abs(src_nm_per_px - target_nm_per_px) < 1e-3:
        return h
    scale = src_nm_per_px / target_nm_per_px
    new_H = max(1, int(round(h.shape[0] * scale)))
    new_W = max(1, int(round(h.shape[1] * scale)))
    img = Image.fromarray(h.astype(np.float32), mode="F")
    img = img.resize((new_W, new_H), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def robust_normalize(
    h: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """Clip to [p_low, p_high] percentiles → linearly map to [0, 1]."""
    lo, hi = np.percentile(h, [p_low, p_high])
    if hi - lo < 1e-6:
        return np.zeros_like(h)
    return np.clip((h - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def center_crop_or_pad(h: np.ndarray, size: int) -> np.ndarray:
    """Center-crop if larger than `size`, reflect-pad if smaller."""
    H, W = h.shape
    if H < size or W < size:
        pad_h = max(0, size - H)
        pad_w = max(0, size - W)
        h = np.pad(h, ((pad_h // 2, pad_h - pad_h // 2),
                       (pad_w // 2, pad_w - pad_w // 2)), mode="reflect")
        H, W = h.shape
    y0 = (H - size) // 2
    x0 = (W - size) // 2
    return h[y0:y0 + size, x0:x0 + size]


def tile(h: np.ndarray, size: int, stride: int | None = None) -> list[np.ndarray]:
    """Cut height map into non-overlapping (or strided) square tiles."""
    if stride is None:
        stride = size
    H, W = h.shape
    tiles = []
    for y in range(0, H - size + 1, stride):
        for x in range(0, W - size + 1, stride):
            tiles.append(h[y:y + size, x:x + size])
    return tiles


def preprocess_spm(
    path: Path,
    target_nm_per_px: float = 90.0,
    crop_size: int = 512,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """Full SPM → ML-ready 2D float32 height image."""
    hm = load_height(path)
    h = plane_level(hm.height)
    h = resample_to_pixel_size(h, hm.pixel_nm, target_nm_per_px)
    h = robust_normalize(h, p_low=p_low, p_high=p_high)
    h = center_crop_or_pad(h, crop_size)
    return h


# ---------------------------------------------------------------------------
# Dataset enumeration
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    raw_path: Path
    bmp_path: Path | None
    cls: str
    patient: str   # eye-level grouping (one ID per L/R eye)
    person: str    # person-level grouping (L and R eyes merged)

    @property
    def label(self) -> int:
        return CLASS_TO_IDX[self.cls]


def enumerate_samples(data_root: Path | str) -> list[Sample]:
    """Walk TRAIN_SET, return one Sample per raw SPM file with linked BMP."""
    data_root = Path(data_root)
    samples: list[Sample] = []
    for cls_dir in sorted(data_root.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in CLASS_TO_IDX:
            continue
        for raw in sorted(cls_dir.iterdir()):
            if not is_raw_spm(raw):
                continue
            bmp_candidates = list(cls_dir.glob(f"{raw.stem}{raw.suffix}_1.bmp"))
            samples.append(Sample(
                raw_path=raw,
                bmp_path=bmp_candidates[0] if bmp_candidates else None,
                cls=cls_dir.name,
                patient=patient_id(raw),
                person=person_id(raw),
            ))
    return samples


def samples_dataframe(samples: Iterable[Sample]):
    import pandas as pd
    return pd.DataFrame([
        {"raw": str(s.raw_path), "bmp": str(s.bmp_path) if s.bmp_path else "",
         "cls": s.cls, "label": s.label, "patient": s.patient}
        for s in samples
    ])
