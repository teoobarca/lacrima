"""Grad-CAM visualizations for the DINOv2-B 90nm component of the v4 ensemble.

The production classifier is:
    image (224x224 RGB) -> DINOv2-B backbone -> (CLS, patches) pooled -> L2 norm
    -> StandardScaler -> Linear (LogReg) -> softmax

Grad-CAM needs an end-to-end differentiable module, so we wrap the encoder,
scaler, and LR head into a single nn.Module (ViTExplain). Gradients flow back
to the last transformer block; pytorch-grad-cam's reshape_transform converts
the (B, 1+N, C) token sequence into a (B, C, 16, 16) spatial map.

Usage:
    python -m scripts.generate_gradcam
    # writes reports/pitch/08_gradcam_per_class.png

Notes:
- DINOv2 at 224x224 with patch 14 -> 16x16 patch grid = 256 tokens + 1 CLS.
- We only visualize component A (DINOv2-B 90nm). The ensemble softmax on the
  full pipeline is computed separately for the title/confidence.
- If a class has no correctly-classified high-confidence scan (e.g. SucheOko in
  OOF), we still pick the 2 scans with highest self-probability so the figure
  shows what pattern the model did/did not latch onto.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from teardrop.data import plane_level, resample_to_pixel_size, robust_normalize, load_height  # noqa: E402
from teardrop.encoders import height_to_pil, load_dinov2  # noqa: E402
from teardrop.infer import preprocess_and_tile_spm  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402


CLASSES = ["ZdraviLudia", "Diabetes", "PGOV_Glaukom", "SklerozaMultiplex", "SucheOko"]
DISPLAY = {
    "ZdraviLudia": "Healthy",
    "Diabetes": "Diabetes",
    "PGOV_Glaukom": "Glaucoma",
    "SklerozaMultiplex": "Mult. Sclerosis",
    "SucheOko": "Dry Eye",
}


# ---------------------------------------------------------------------------
# End-to-end wrapper: DINOv2-B + L2norm + scaler + LR head
# ---------------------------------------------------------------------------

class DinoV2Classifier(nn.Module):
    """Wrap frozen DINOv2-B + L2 + scaler + LR as one differentiable module."""

    def __init__(self, dino_model: nn.Module,
                 scaler_mean: np.ndarray, scaler_scale: np.ndarray,
                 lr_coef: np.ndarray, lr_intercept: np.ndarray,
                 device: str):
        super().__init__()
        self.dino = dino_model
        self.register_buffer("mean", torch.tensor(scaler_mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scaler_scale, dtype=torch.float32))
        self.register_buffer("coef", torch.tensor(lr_coef, dtype=torch.float32))
        self.register_buffer("bias", torch.tensor(lr_intercept, dtype=torch.float32))
        self.to(device)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DINOv2 `forward` returns the CLS token embedding directly.
        emb = self.dino(x)  # (B, 768)
        emb = F.normalize(emb, p=2, dim=1)
        emb = (emb - self.mean) / (self.scale + 1e-9)
        logits = emb @ self.coef.T + self.bias
        return logits


# ---------------------------------------------------------------------------
# ViT reshape transform for Grad-CAM
# ---------------------------------------------------------------------------

def vit_reshape_transform(tensor: torch.Tensor, height: int = 16, width: int = 16):
    """Drop CLS token(s) and reshape (B, 1+N, C) -> (B, C, H, W).

    DINOv2 at 224x224, patch=14 -> 16x16 spatial tokens. The registers (if any)
    come after CLS; by default DINOv2-B hub model has no registers, so we just
    drop the first token.
    """
    # Tensor shape: (B, 1+num_patches, C) or (B, 1+num_reg+num_patches, C)
    num_tokens = tensor.size(1)
    num_patches = height * width
    # Drop everything except the trailing spatial patches.
    patches = tensor[:, -num_patches:, :]
    result = patches.reshape(tensor.size(0), height, width, tensor.size(-1))
    result = result.permute(0, 3, 1, 2)  # (B, C, H, W)
    return result


# ---------------------------------------------------------------------------
# Component-A loader (DINOv2-B 90nm head)
# ---------------------------------------------------------------------------

def build_classifier() -> tuple[DinoV2Classifier, object]:
    """Load DINOv2-B encoder + v4 90nm head, return end-to-end module + preprocess."""
    bundle = load_dinov2("vitb14")
    comp_dir = _ROOT / "models" / "ensemble_v4_multiscale" / "dinov2b_90nm"
    arrs = np.load(comp_dir / "classifier.npz")
    clf = DinoV2Classifier(
        dino_model=bundle.model,
        scaler_mean=arrs["scaler_means"],
        scaler_scale=arrs["scaler_scales"],
        lr_coef=arrs["lr_coef"],
        lr_intercept=arrs["lr_intercept"],
        device=bundle.device,
    )
    clf.eval()
    return clf, bundle.preprocess


# ---------------------------------------------------------------------------
# Per-scan Grad-CAM computation
# ---------------------------------------------------------------------------

def preprocess_scan_to_central_tile(raw_path: Path, target_nm_per_px: float = 90.0,
                                    tile_size: int = 512) -> np.ndarray:
    """Run the production tiler; return the central tile (float32 in [0,1])."""
    tiles = preprocess_and_tile_spm(raw_path, target_nm_per_px=target_nm_per_px,
                                    tile_size=tile_size, max_tiles=9)
    # Middle tile = most representative center-of-scan patch.
    mid = tiles[len(tiles) // 2]
    return mid


def compute_gradcam_for_tile(clf: DinoV2Classifier, preprocess,
                             tile_float01: np.ndarray, target_class: int,
                             render_mode: str = "afmhot") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rgb_224, cam_224, overlay_224) as float arrays in [0,1].

    - rgb_224: the afmhot-rendered resized image that actually enters DINOv2.
    - cam_224: the Grad-CAM saliency upsampled to 224x224, [0,1].
    - overlay_224: heatmap colorized + alpha-blended onto rgb_224.
    """
    # Render + preprocess so we know the *exact* model input.
    pil = height_to_pil(tile_float01, mode=render_mode)
    rgb_224 = np.asarray(pil.resize((224, 224), Image.Resampling.BICUBIC),
                         dtype=np.float32) / 255.0

    input_tensor = preprocess(pil).unsqueeze(0).to(clf.device)
    input_tensor.requires_grad_(True)

    target_layers = [clf.dino.blocks[-1].norm1]  # last transformer block LN1

    with GradCAM(
        model=clf,
        target_layers=target_layers,
        reshape_transform=vit_reshape_transform,
    ) as cam:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # (224,224)

    # Normalize to [0,1]
    if grayscale_cam.max() > grayscale_cam.min():
        grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (
            grayscale_cam.max() - grayscale_cam.min())
    overlay = show_cam_on_image(rgb_224, grayscale_cam, use_rgb=True, colormap=2)  # COLORMAP_JET=2
    overlay = overlay.astype(np.float32) / 255.0
    return rgb_224, grayscale_cam, overlay


# ---------------------------------------------------------------------------
# Scan selection per class
# ---------------------------------------------------------------------------

def pick_iconic_scans_per_class(n_per_class: int = 2) -> dict[str, list[tuple[Path, float, str, str]]]:
    """For each class return list of (raw_path, self_prob, pred_class, note).

    Prefer correctly classified with high confidence.  If none, fall back to
    the scans of that true class with the highest self-probability.
    """
    oof = pd.read_parquet(_ROOT / "reports" / "best_oof_predictions.parquet")
    oof["correct"] = oof["true_class"] == oof["pred_class"]
    out = {}
    for cls in CLASSES:
        prob_col = f"prob_{cls}"
        correct = oof[(oof.true_class == cls) & oof.correct].copy()
        if len(correct) >= n_per_class:
            correct["conf"] = correct[prob_col]
            picks = correct.sort_values("conf", ascending=False).head(n_per_class)
            out[cls] = [(Path(r.raw_path), float(r.conf), r.pred_class,
                         "correct") for _, r in picks.iterrows()]
        else:
            sub = oof[oof.true_class == cls].copy()
            sub["conf"] = sub[prob_col]
            picks = sub.sort_values("conf", ascending=False).head(n_per_class)
            out[cls] = [(Path(r.raw_path), float(r.conf), r.pred_class,
                         "misclassified" if r.pred_class != cls else "correct")
                        for _, r in picks.iterrows()]
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(picks: dict[str, list[tuple[Path, float, str, str]]],
                clf: DinoV2Classifier, preprocess, out_path: Path):
    """Grid: one row per class (5), 6 cols = [orig1, cam1, ovr1, orig2, cam2, ovr2]."""
    n_classes = len(CLASSES)
    n_cols = 6
    fig, axes = plt.subplots(n_classes, n_cols, figsize=(3.0 * n_cols, 3.1 * n_classes))
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    for r, cls in enumerate(CLASSES):
        class_idx = CLASSES.index(cls)
        picks_cls = picks[cls]
        for s, (raw_path, conf, pred_cls, note) in enumerate(picks_cls):
            col_orig = s * 3
            col_cam = s * 3 + 1
            col_ovr = s * 3 + 2
            try:
                tile = preprocess_scan_to_central_tile(raw_path)
                rgb, cam, overlay = compute_gradcam_for_tile(clf, preprocess, tile, class_idx)
            except Exception as e:
                print(f"  ! {raw_path.name}: {e}")
                for c in (col_orig, col_cam, col_ovr):
                    axes[r, c].text(0.5, 0.5, f"err: {str(e)[:40]}", ha="center",
                                    va="center", transform=axes[r, c].transAxes, fontsize=8)
                    axes[r, c].axis("off")
                continue

            axes[r, col_orig].imshow(rgb)
            correct_marker = "OK" if note == "correct" else "X"
            axes[r, col_orig].set_title(
                f"{DISPLAY[cls]} #{s + 1}  [{correct_marker}]\n{raw_path.name}\n"
                f"self-prob={conf:.2f}  pred={DISPLAY.get(pred_cls, pred_cls)}",
                fontsize=8,
            )
            axes[r, col_cam].imshow(cam, cmap="jet", vmin=0, vmax=1)
            axes[r, col_cam].set_title(f"Grad-CAM (target={DISPLAY[cls]})", fontsize=8)
            axes[r, col_ovr].imshow(overlay)
            axes[r, col_ovr].set_title("Overlay", fontsize=8)
            for c in (col_orig, col_cam, col_ovr):
                axes[r, c].axis("off")

        # If fewer than expected picks, blank unused cols
        for s in range(len(picks_cls), 2):
            for c in (s * 3, s * 3 + 1, s * 3 + 2):
                axes[r, c].axis("off")

    fig.suptitle(
        "Grad-CAM on DINOv2-B (90 nm/px) — iconic scan per class\n"
        "Column 1/4: model input (afmhot render).  Col 2/5: class activation.  Col 3/6: overlay.\n"
        "[OK] = model predicts this class correctly.  [X] = misclassified (still shows where the class head is looking).",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading DINOv2-B classifier...")
    clf, preprocess = build_classifier()

    print("Picking iconic scans per class...")
    picks = pick_iconic_scans_per_class(n_per_class=2)
    for cls, lst in picks.items():
        print(f"  {cls}:")
        for p, conf, pred, note in lst:
            print(f"    {note:14s}  self-prob={conf:.3f}  pred={pred}  {p.name}")

    out_path = _ROOT / "reports" / "pitch" / "08_gradcam_per_class.png"
    make_figure(picks, clf, preprocess, out_path)


if __name__ == "__main__":
    main()
