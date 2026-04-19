"""End-to-end inference pipeline: raw SPM directory → class predictions.

Usage:
    from teardrop.infer import TearClassifier
    clf = TearClassifier.load('models/ensemble_v1/')
    predictions = clf.predict_directory('path/to/TEST_SET/')

The model is trained once and saved; inference reads saved scalers + LR
coefficients + encoder weights (or just uses torch.hub for DINOv2).

Two kinds of saved bundles are supported (distinguished by `meta.json['kind']`):
    * 'single'   — legacy: one encoder + one LR (ClassifierBundle).
    * 'ensemble' — multiple encoder/LR components, softmax averaged.

Output format: pandas DataFrame with columns:
    file, predicted_class, prob_ZdraviLudia, prob_Diabetes, ..., prob_SucheOko
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from teardrop.data import (
    CLASSES, is_raw_spm, load_height, plane_level, resample_to_pixel_size,
    robust_normalize, tile,
)
from teardrop.encoders import EncoderBundle, height_to_pil, load_biomedclip, load_dinov2


def _load_encoder(encoder_name: str) -> EncoderBundle:
    if encoder_name.startswith("dinov2_"):
        return load_dinov2(encoder_name.replace("dinov2_", ""))
    if encoder_name == "biomedclip":
        return load_biomedclip()
    raise ValueError(f"Unsupported encoder: {encoder_name}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class ClassifierBundle:
    """Trained single-encoder classifier."""
    encoder: EncoderBundle
    scaler_means: np.ndarray  # (n_features,)
    scaler_scales: np.ndarray  # (n_features,)
    lr_coef: np.ndarray       # (n_classes, n_features)
    lr_intercept: np.ndarray  # (n_classes,)
    classes: list[str]
    config: dict

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_std = (X - self.scaler_means) / (self.scaler_scales + 1e-9)
        logits = X_std @ self.lr_coef.T + self.lr_intercept
        return _softmax(logits)

    def save(self, out_dir: Path | str) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "classifier.npz",
                 scaler_means=self.scaler_means,
                 scaler_scales=self.scaler_scales,
                 lr_coef=self.lr_coef,
                 lr_intercept=self.lr_intercept)
        with open(out_dir / "meta.json", "w") as f:
            json.dump({"kind": "single",
                       "classes": self.classes,
                       "config": self.config,
                       "encoder_name": self.encoder.name}, f, indent=2)

    @classmethod
    def load(cls, in_dir: Path | str,
             encoder: EncoderBundle | None = None) -> "ClassifierBundle":
        in_dir = Path(in_dir)
        with open(in_dir / "meta.json") as f:
            meta = json.load(f)
        arrs = np.load(in_dir / "classifier.npz")
        enc = encoder if encoder is not None else _load_encoder(meta["encoder_name"])
        return cls(
            encoder=enc,
            scaler_means=arrs["scaler_means"],
            scaler_scales=arrs["scaler_scales"],
            lr_coef=arrs["lr_coef"],
            lr_intercept=arrs["lr_intercept"],
            classes=meta["classes"],
            config=meta["config"],
        )


@dataclass
class EnsembleComponent:
    """Single (encoder + scaler + LR) piece of an ensemble."""
    encoder: EncoderBundle
    scaler_means: np.ndarray
    scaler_scales: np.ndarray
    lr_coef: np.ndarray
    lr_intercept: np.ndarray

    def encode_scan(self, pil_tiles) -> np.ndarray:
        tile_emb = self.encoder.encode(pil_tiles, batch_size=len(pil_tiles))
        return tile_emb.mean(axis=0, keepdims=True)  # (1, D)

    def predict_proba(self, scan_emb: np.ndarray) -> np.ndarray:
        X_std = (scan_emb - self.scaler_means) / (self.scaler_scales + 1e-9)
        logits = X_std @ self.lr_coef.T + self.lr_intercept
        return _softmax(logits)


@dataclass
class EnsembleClassifierBundle:
    """Uniform softmax-averaged ensemble.

    For each component: tiles → encode → mean-pool over tiles → standardize →
    LR logits → softmax. Final probability is the arithmetic mean of the
    component softmaxes.
    """
    components: list[EnsembleComponent]
    component_names: list[str]
    classes: list[str]
    config: dict = field(default_factory=dict)

    def predict_proba_from_tiles(self, pil_tiles) -> np.ndarray:
        """Predict from raw PIL tiles (shared preprocessing)."""
        probs = []
        for comp in self.components:
            scan_emb = comp.encode_scan(pil_tiles)
            probs.append(comp.predict_proba(scan_emb))
        return np.mean(np.stack(probs, axis=0), axis=0)  # (1, n_classes)

    def predict_proba_from_embeddings(self, scan_embs: dict[str, np.ndarray]) -> np.ndarray:
        """Predict from pre-computed scan-level embeddings keyed by component name.

        Each value should be (n_scans, D_i) — the mean-pooled scan embedding
        for that component. Returns (n_scans, n_classes).
        """
        probs = []
        for name, comp in zip(self.component_names, self.components):
            if name not in scan_embs:
                raise KeyError(f"Missing scan embedding for component '{name}'")
            probs.append(comp.predict_proba(scan_embs[name]))
        return np.mean(np.stack(probs, axis=0), axis=0)

    def save(self, out_dir: Path | str) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, comp in zip(self.component_names, self.components):
            comp_dir = out_dir / name
            comp_dir.mkdir(parents=True, exist_ok=True)
            np.savez(comp_dir / "classifier.npz",
                     scaler_means=comp.scaler_means,
                     scaler_scales=comp.scaler_scales,
                     lr_coef=comp.lr_coef,
                     lr_intercept=comp.lr_intercept)
            with open(comp_dir / "meta.json", "w") as f:
                json.dump({"kind": "single",
                           "encoder_name": comp.encoder.name,
                           "classes": self.classes,
                           "config": self.config}, f, indent=2)
        with open(out_dir / "meta.json", "w") as f:
            json.dump({"kind": "ensemble",
                       "components": self.component_names,
                       "classes": self.classes,
                       "config": self.config}, f, indent=2)

    @classmethod
    def load(cls, in_dir: Path | str) -> "EnsembleClassifierBundle":
        in_dir = Path(in_dir)
        with open(in_dir / "meta.json") as f:
            meta = json.load(f)
        if meta.get("kind") != "ensemble":
            raise ValueError(f"Not an ensemble bundle: {in_dir}")
        components = []
        for name in meta["components"]:
            comp_dir = in_dir / name
            with open(comp_dir / "meta.json") as f:
                cmeta = json.load(f)
            arrs = np.load(comp_dir / "classifier.npz")
            enc = _load_encoder(cmeta["encoder_name"])
            components.append(EnsembleComponent(
                encoder=enc,
                scaler_means=arrs["scaler_means"],
                scaler_scales=arrs["scaler_scales"],
                lr_coef=arrs["lr_coef"],
                lr_intercept=arrs["lr_intercept"],
            ))
        return cls(
            components=components,
            component_names=list(meta["components"]),
            classes=meta["classes"],
            config=meta.get("config", {}),
        )


def preprocess_and_tile_spm(
    raw_path: Path,
    target_nm_per_px: float = 90.0,
    tile_size: int = 512,
    max_tiles: int = 9,
) -> list[np.ndarray]:
    hm = load_height(raw_path)
    h = plane_level(hm.height)
    h = resample_to_pixel_size(h, hm.pixel_nm, target_nm_per_px)
    h = robust_normalize(h)
    if h.shape[0] < tile_size or h.shape[1] < tile_size:
        pad_h = max(0, tile_size - h.shape[0])
        pad_w = max(0, tile_size - h.shape[1])
        h = np.pad(h, ((pad_h // 2, pad_h - pad_h // 2),
                       (pad_w // 2, pad_w - pad_w // 2)), mode="reflect")
    tiles = tile(h, tile_size, stride=tile_size)
    if not tiles:
        return [h[:tile_size, :tile_size]]
    if len(tiles) > max_tiles:
        idx = np.linspace(0, len(tiles) - 1, max_tiles).astype(int)
        tiles = [tiles[i] for i in idx]
    return tiles


class TearClassifier:
    """High-level predictor. Loads a single or ensemble bundle and exposes
    `predict_scan` / `predict_directory`.

    Supported bundle kinds (auto-detected from `meta.json['kind']`):
        * 'single'   — ClassifierBundle
        * 'ensemble' — EnsembleClassifierBundle
    """

    def __init__(self, bundle):
        self.bundle = bundle
        self.is_ensemble = isinstance(bundle, EnsembleClassifierBundle)

    @classmethod
    def load(cls, model_dir: Path | str) -> "TearClassifier":
        model_dir = Path(model_dir)
        with open(model_dir / "meta.json") as f:
            meta = json.load(f)
        kind = meta.get("kind", "single")
        if kind == "ensemble":
            return cls(EnsembleClassifierBundle.load(model_dir))
        if kind == "single":
            return cls(ClassifierBundle.load(model_dir))
        raise ValueError(f"Unknown bundle kind: {kind}")

    @property
    def classes(self) -> list[str]:
        return self.bundle.classes

    def _predict_probs_from_tiles(self, tiles: list[np.ndarray],
                                  render_mode: str) -> np.ndarray:
        """Returns (n_classes,) probability vector for one scan."""
        if self.is_ensemble:
            # Each encoder may have a different preprocess; share the PIL inputs.
            pils = [height_to_pil(t, mode=render_mode) for t in tiles]
            probs = self.bundle.predict_proba_from_tiles(pils)
            return probs[0]
        else:
            pils = [height_to_pil(t, mode=render_mode) for t in tiles]
            tile_emb = self.bundle.encoder.encode(pils, batch_size=len(pils))
            scan_emb = tile_emb.mean(axis=0, keepdims=True)
            return self.bundle.predict_proba(scan_emb)[0]

    def predict_scan(self, raw_path: Path, render_mode: str = "afmhot",
                     target_nm_per_px: float = 90.0,
                     tile_size: int = 512, max_tiles: int = 9) -> tuple[str, np.ndarray]:
        """Predict one scan. Returns (predicted_class, class_probabilities)."""
        tiles = preprocess_and_tile_spm(raw_path, target_nm_per_px, tile_size, max_tiles)
        probs = self._predict_probs_from_tiles(tiles, render_mode=render_mode)
        pred_idx = int(probs.argmax())
        return self.bundle.classes[pred_idx], probs

    def predict_directory(self, root: Path | str,
                          **kwargs) -> pd.DataFrame:
        """Walk root for raw SPM files, predict each."""
        root = Path(root)
        rows = []
        all_raws = sorted([p for p in root.rglob("*") if is_raw_spm(p)])
        for i, p in enumerate(all_raws):
            try:
                cls_pred, probs = self.predict_scan(p, **kwargs)
                row = {"file": str(p.relative_to(root)), "predicted_class": cls_pred}
                for c, pr in zip(self.bundle.classes, probs):
                    row[f"prob_{c}"] = float(pr)
                rows.append(row)
            except Exception as e:
                rows.append({"file": str(p.relative_to(root)), "error": str(e)[:200]})
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(all_raws)}] processed")
        return pd.DataFrame(rows)
