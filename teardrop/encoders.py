"""Frozen vision encoders for embedding-based classification.

Supports:
- DINOv2 (Meta) — strongest general-purpose visual features
- OpenCLIP ViT-L/14 (LAION-2B) — strong CLIP baseline
- BiomedCLIP (Microsoft) — medical-domain pretrained, ideal for tear AFM
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass
class EncoderBundle:
    name: str
    model: torch.nn.Module
    preprocess: callable        # PIL → tensor
    embed_dim: int
    device: str

    @torch.no_grad()
    def encode(self, pil_images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        embs = []
        self.model.eval()
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i + batch_size]
            tensors = torch.stack([self.preprocess(im) for im in batch]).to(self.device)
            with torch.no_grad():
                if hasattr(self.model, "encode_image"):
                    e = self.model.encode_image(tensors)
                else:
                    e = self.model(tensors)
            e = e.float().cpu().numpy()
            embs.append(e)
        return np.concatenate(embs, axis=0)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_dinov2(variant: str = "vitb14") -> EncoderBundle:
    """Load DINOv2 from torch.hub. Variants: vits14, vitb14, vitl14, vitg14."""
    import torchvision.transforms as T

    device = pick_device()
    model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{variant}")
    model = model.to(device).eval()
    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    embed_dim = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}[variant]
    return EncoderBundle(
        name=f"dinov2_{variant}",
        model=model, preprocess=preprocess, embed_dim=embed_dim, device=device,
    )


def load_openclip(model_name: str = "ViT-L-14", pretrained: str = "laion2b_s32b_b82k") -> EncoderBundle:
    import open_clip

    device = pick_device()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    cfg = open_clip.get_model_config(model_name)
    embed_dim = cfg.get("embed_dim", 768)
    return EncoderBundle(
        name=f"openclip_{model_name}_{pretrained}",
        model=model, preprocess=preprocess, embed_dim=embed_dim, device=device,
    )


def load_biomedclip() -> EncoderBundle:
    """BiomedCLIP (Microsoft) via HuggingFace hub. Pretrained on 15M PubMed image-text pairs."""
    import open_clip

    device = pick_device()
    repo = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, _, preprocess = open_clip.create_model_and_transforms(repo)
    model = model.to(device).eval()
    return EncoderBundle(
        name="biomedclip",
        model=model, preprocess=preprocess, embed_dim=512, device=device,
    )


def height_to_pil(h: np.ndarray, mode: str = "RGB") -> Image.Image:
    """Convert normalized height map [0,1] (HxW float32) to PIL image.

    Most foundation models expect 3-channel RGB. We replicate the height channel
    or render with a colormap.
    """
    h8 = (np.clip(h, 0, 1) * 255).astype(np.uint8)
    if mode == "L":
        return Image.fromarray(h8, mode="L")
    if mode == "RGB":
        return Image.fromarray(np.stack([h8, h8, h8], axis=-1), mode="RGB")
    if mode == "afmhot":
        import matplotlib.cm as cm
        rgba = (cm.afmhot(h) * 255).astype(np.uint8)[..., :3]
        return Image.fromarray(rgba, mode="RGB")
    raise ValueError(f"Unknown mode: {mode}")
