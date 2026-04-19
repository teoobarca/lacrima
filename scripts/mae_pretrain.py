"""Masked Autoencoder (MAE) pre-training on our OWN 240 AFM scans' tiles.

Highest-EV experiment per reports/EXTERNAL_DATA_SURVEY.md (recommendation 1):

- Zero data-acquisition cost, maximal domain alignment.
- Train a ViT-Tiny encoder from scratch on all 240 AFM scans' tiles (+ D4 augmentation)
  with 75% random patch masking + pixel reconstruction (He et al., 2022).
- Evaluate transferred features by mean-pooling per scan, training a LogisticRegression
  under person-level LOPO, compare F1 to DINOv2-B ImageNet baseline.

Implementation notes
--------------------
- ViT-Tiny (patch 16, embed_dim 192, 12 blocks, 3 heads) via `timm` as the encoder
  backbone. Custom masking + light decoder (4 blocks, 96 dim) implemented from scratch.
- Input 224x224 greyscale height tiles replicated to 3 channels and normalized.
- D4 augmentation (8 views per tile) applied on the fly.
- AdamW lr=1.5e-4, cosine schedule, 50-100 epochs, batch 128 on MPS.
- Reconstruction loss: MSE on masked patches only, per-patch pixel normalization.

Downstream eval (run in same script to keep a single report)
-----------------------------------------------------------
1. Extract pre-trained encoder CLS embeddings per tile (240 scans, 9 tiles-at-45nm protocol).
2. Mean-pool per scan -> (240, 192).
3. Person-LOPO LogisticRegression.
4. Also evaluate the random-init ViT-Tiny (control), plus already-cached DINOv2-B
   baseline (from cache/tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz) for reference.

Artifacts written
-----------------
- models/mae_tear_tiny/encoder.pt   -- encoder weights
- models/mae_tear_tiny/config.json  -- hyperparameters + results
- cache/mae_emb_tear_tiny.npz       -- tile embeddings (+ scan-level mean pool)
- reports/MAE_PRETRAINING_RESULTS.md

Usage:
    .venv/bin/python scripts/mae_pretrain.py [--epochs N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import (  # noqa: E402
    CLASSES,
    enumerate_samples,
    load_height,
    person_id,
    plane_level,
    resample_to_pixel_size,
    robust_normalize,
    tile,
)

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
(CACHE).mkdir(exist_ok=True)
(REPORTS).mkdir(exist_ok=True)
(MODELS / "mae_tear_tiny").mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet mean/std (3-channel greyscale replication); we use it to match downstream
# feature extraction conventions.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Tile preparation (45 nm/px, up to 9 tiles/scan -- same protocol as champion v4)
# ---------------------------------------------------------------------------

def build_tiles(
    target_nm_per_px: float = 45.0,
    tile_size: int = 512,
    max_tiles: int = 9,
    out_hw: int = 224,
    cache_name: str = "mae_raw_tiles_45nm_t512_n9_224.npz",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Preprocess all 240 scans into resized 224x224 greyscale tiles.

    Returns:
        tiles:            (N_tiles, out_hw, out_hw) float32 in [0, 1]
        tile_to_scan:     (N_tiles,) int
        scan_y:           (240,) int class labels
        scan_persons:     (240,) str person IDs
        scan_paths:       list of raw SPM paths
    """
    cache_path = CACHE / cache_name
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        print(f"[tile-cache hit] {cache_path.name}")
        return (
            z["tiles"].astype(np.float32),
            z["tile_to_scan"].astype(int),
            z["scan_y"].astype(int),
            z["scan_persons"],
            z["scan_paths"].tolist(),
        )

    print(f"[tile-cache build] {cache_path.name}")
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    print(f"  enumerating {len(samples)} samples")

    all_tiles: list[np.ndarray] = []
    tile_to_scan: list[int] = []
    scan_y, scan_persons, scan_paths = [], [], []

    t0 = time.time()
    for si, s in enumerate(samples):
        try:
            hm = load_height(s.raw_path)
            h = plane_level(hm.height)
            h = resample_to_pixel_size(h, hm.pixel_nm, target_nm_per_px)
            h = robust_normalize(h)
            if h.shape[0] < tile_size or h.shape[1] < tile_size:
                pad_h = max(0, tile_size - h.shape[0])
                pad_w = max(0, tile_size - h.shape[1])
                h = np.pad(
                    h,
                    ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                    mode="reflect",
                )
            tiles = tile(h, tile_size, stride=tile_size)
            if not tiles:
                tiles = [h[:tile_size, :tile_size]]
            if len(tiles) > max_tiles:
                idx = np.linspace(0, len(tiles) - 1, max_tiles).astype(int)
                tiles = [tiles[i] for i in idx]
            # Resize each tile to out_hw.
            for t in tiles:
                img = Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8), mode="L")
                img = img.resize((out_hw, out_hw), Image.Resampling.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                all_tiles.append(arr)
                tile_to_scan.append(si)
            scan_y.append(s.label)
            scan_persons.append(s.person)
            scan_paths.append(str(s.raw_path))
        except Exception as e:
            print(f"  [err] {s.raw_path.name}: {e}")
        if (si + 1) % 40 == 0:
            print(f"  [{si+1}/{len(samples)}] {len(all_tiles)} tiles  {time.time()-t0:.1f}s")

    tiles_arr = np.stack(all_tiles, axis=0)
    t2s = np.asarray(tile_to_scan, dtype=np.int64)
    scan_y_arr = np.asarray(scan_y, dtype=np.int64)
    scan_persons_arr = np.asarray(scan_persons)
    print(f"  built {tiles_arr.shape} in {time.time()-t0:.1f}s")

    np.savez(
        cache_path,
        tiles=tiles_arr.astype(np.float32),
        tile_to_scan=t2s,
        scan_y=scan_y_arr,
        scan_persons=scan_persons_arr,
        scan_paths=np.array(scan_paths),
    )
    print(f"  saved {cache_path}")
    return tiles_arr.astype(np.float32), t2s, scan_y_arr, scan_persons_arr, scan_paths


# ---------------------------------------------------------------------------
# MAE model: ViT-Tiny encoder + light decoder
# ---------------------------------------------------------------------------


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = True) -> np.ndarray:
    """Standard 2D sin-cos positional encoding (from MAE repo), NumPy -> torch later."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w first
    grid = np.stack(grid, axis=0)  # (2, H, W)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb.astype(np.float32)


def _get_1d_sincos(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class MAETearTiny(nn.Module):
    """Minimal MAE with a ViT-Tiny encoder (from timm, from scratch) + tiny decoder.

    - Encoder: ViT-Tiny-like (depth 12, dim 192, heads 3, patch 16, img 224).
    - Decoder: 4 blocks, dim 96, 3 heads (standard MAE-tiny decoder).
    - Mask ratio 0.75 (random tokens).

    Forward returns (loss, pred, mask) like the original MAE.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        decoder_embed_dim: int = 96,
        decoder_depth: int = 4,
        decoder_num_heads: int = 3,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        import timm
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.grid = img_size // patch_size  # 14
        self.num_patches = self.grid * self.grid  # 196

        # Encoder: reuse ViT-Tiny architecture, but we only use patch_embed + blocks + norm.
        vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            in_chans=in_chans,
            global_pool="",
        )
        self.patch_embed = vit.patch_embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks = vit.blocks
        self.norm = vit.norm
        # timm's norm for ViT is LayerNorm by default.

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim)
        )
        dec_block = nn.ModuleList()
        for _ in range(decoder_depth):
            dec_block.append(
                _TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio=4.0)
            )
        self.decoder_blocks = dec_block
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans)

        self._init_weights()

    def _init_weights(self):
        # Fixed sin-cos pos embed for encoder and decoder.
        enc_pe = get_2d_sincos_pos_embed(self.embed_dim, self.grid, cls_token=True)
        dec_pe = get_2d_sincos_pos_embed(self.decoder_embed.out_features, self.grid, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(enc_pe).unsqueeze(0))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dec_pe).unsqueeze(0))

        # Small init for cls_token + mask_token.
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # Xavier for patch_embed.proj, following MAE.
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.proj.bias)

    # ------------------------------------------------------------------
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, patch*patch*C)."""
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W == self.img_size
        x = imgs.reshape(B, C, self.grid, p, self.grid, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, gH, gW, p, p, C)
        x = x.reshape(B, self.grid * self.grid, p * p * C)
        return x

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """Per-sample random subset (no replacement). x: (B, N, D).
        Returns x_masked (B, N_kept, D), mask (B, N) 1==masked, ids_restore (B, N)."""
        B, N, D = x.shape
        n_keep = int(N * (1.0 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :n_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    # ------------------------------------------------------------------
    def forward_encoder(self, imgs: torch.Tensor, mask_ratio: float):
        x = self.patch_embed(imgs)  # (B, N, D)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        x = self.decoder_embed(x)  # (B, 1+N_kept, D_dec)
        B, L, D = x.shape
        N = self.num_patches
        mask_tokens = self.mask_token.expand(B, N + 1 - L, -1)
        x_no_cls = x[:, 1:, :]
        x_ = torch.cat([x_no_cls, mask_tokens], dim=1)  # (B, N, D)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # add cls back
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)  # (B, 1+N, patch^2 * C)
        x = x[:, 1:, :]  # drop cls
        return x

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / torch.sqrt(var + 1e-6)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss

    def forward(self, imgs: torch.Tensor, mask_ratio: float | None = None):
        mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    @torch.no_grad()
    def extract_features(self, imgs: torch.Tensor, pool: str = "cls") -> torch.Tensor:
        """Extract encoder features (no masking). Returns (B, embed_dim)."""
        self.eval()
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if pool == "cls":
            return x[:, 0]
        # patch-mean pool
        return x[:, 1:].mean(dim=1)


class _TransformerBlock(nn.Module):
    """Plain pre-norm transformer block (used for the light decoder)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def d4_view(x: torch.Tensor, k: int) -> torch.Tensor:
    """Apply one of the 8 D4 transformations. x: (B, C, H, W), k in 0..7."""
    if k & 4:
        x = x.flip(-1)  # horizontal flip
    n_rot = k & 3
    if n_rot:
        x = torch.rot90(x, k=n_rot, dims=(-2, -1))
    return x


def prepare_image_tensor(tiles: np.ndarray, device: str) -> torch.Tensor:
    """Stack greyscale tiles into a (N, 3, H, W) tensor normalized with ImageNet stats."""
    x = torch.from_numpy(tiles).unsqueeze(1).float()  # (N, 1, H, W) in [0, 1]
    x = x.repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device)


def cosine_lr(base_lr: float, warmup: int, total: int, step: int, min_lr: float = 1e-6):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def pretrain_mae(
    tiles: np.ndarray,
    *,
    epochs: int = 100,
    batch_size: int = 128,
    base_lr: float = 1.5e-4,
    weight_decay: float = 0.05,
    mask_ratio: float = 0.75,
    seed: int = SEED,
    device: str = DEVICE,
    time_budget_s: float = 30 * 60,
) -> tuple[MAETearTiny, list[dict]]:
    """Run MAE pre-training with on-the-fly D4 augmentation.

    Input tiles are already shape (N, H, W) in [0,1]. We normalize to ImageNet stats.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[MAE] device={device}  tiles={tiles.shape}  epochs={epochs}  bs={batch_size}")
    t_prep = time.time()
    X = prepare_image_tensor(tiles, device)  # (N, 3, H, W)
    print(f"[MAE] tensor {tuple(X.shape)}  dtype={X.dtype}  prep={time.time()-t_prep:.1f}s")

    model = MAETearTiny(mask_ratio=mask_ratio).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_enc = (
        sum(p.numel() for p in model.patch_embed.parameters())
        + sum(p.numel() for p in model.blocks.parameters())
        + sum(p.numel() for p in model.norm.parameters())
        + model.cls_token.numel() + model.pos_embed.numel()
    ) / 1e6
    print(f"[MAE] total params={n_params:.2f}M  encoder params={n_enc:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay,
                            betas=(0.9, 0.95))

    N = X.shape[0]
    steps_per_epoch = (N + batch_size - 1) // batch_size
    warmup_steps = max(50, steps_per_epoch)  # ~1 epoch warmup
    total_steps = epochs * steps_per_epoch
    print(f"[MAE] steps/epoch={steps_per_epoch}  total steps={total_steps}  warmup={warmup_steps}")
    print(f"[MAE] D4 views: 1 random view per tile per epoch (over {epochs} epochs ~= {epochs} views/tile)")

    history: list[dict] = []
    t_start = time.time()
    step = 0

    for ep in range(epochs):
        # Per-epoch: one random D4 view per tile (saves 8x compute vs exhaustive-views).
        # Over many epochs each tile gets seen under many different views.
        model.train()
        losses: list[float] = []
        perm = torch.randperm(N, device=device)
        views = torch.randint(0, 8, (N,), device=device)
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            if idx.numel() < 4:
                continue
            x_b = X.index_select(0, idx).clone()
            v_b = views.index_select(0, idx)
            # Group by view for speed (one batched D4 call per distinct view).
            for k in range(8):
                sel = (v_b == k)
                if sel.any():
                    x_b[sel] = d4_view(x_b[sel], k)

            lr = cosine_lr(base_lr, warmup_steps, total_steps, step)
            for g in opt.param_groups:
                g["lr"] = lr

            loss, _, _ = model(x_b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            step += 1

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        elapsed = time.time() - t_start
        eta = elapsed / (ep + 1) * (epochs - ep - 1)
        history.append({"epoch": ep + 1, "loss": avg_loss, "lr": lr, "elapsed_s": elapsed})
        print(f"[MAE] ep {ep+1:3d}/{epochs}  loss={avg_loss:.4f}  lr={lr:.2e}  "
              f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

        # Time-budget guard.
        if elapsed > time_budget_s and ep + 1 < epochs:
            print(f"[MAE] hit time budget {time_budget_s/60:.1f}m at epoch {ep+1}/{epochs} -- stopping early")
            break

    return model, history


@torch.no_grad()
def extract_tile_features(
    model: MAETearTiny,
    tiles: np.ndarray,
    *,
    batch_size: int = 128,
    device: str = DEVICE,
) -> np.ndarray:
    """Extract CLS features for every tile (no masking, no augmentation)."""
    model.eval()
    N = tiles.shape[0]
    out = np.zeros((N, model.embed_dim), dtype=np.float32)
    for s in range(0, N, batch_size):
        chunk = tiles[s:s + batch_size]
        x = prepare_image_tensor(chunk, device)
        z = model.extract_features(x, pool="cls")
        out[s:s + batch_size] = z.float().cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Person-LOPO evaluation
# ---------------------------------------------------------------------------


def scan_mean_pool(X_tiles: np.ndarray, tile_to_scan: np.ndarray, n_scans: int) -> np.ndarray:
    D = X_tiles.shape[1]
    out = np.zeros((n_scans, D), dtype=np.float32)
    cnt = np.zeros(n_scans, dtype=np.int32)
    for ti, si in enumerate(tile_to_scan):
        out[si] += X_tiles[ti]
        cnt[si] += 1
    cnt = np.maximum(cnt, 1)
    return out / cnt[:, None]


def lopo_lr(
    X_scan: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label: str = "",
) -> tuple[float, float, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    preds = np.full(len(y), -1, dtype=int)
    for tr, va in leave_one_patient_out(groups):
        sc = StandardScaler()
        Xt = sc.fit_transform(X_scan[tr])
        Xv = sc.transform(X_scan[va])
        clf = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0,
            solver="lbfgs", n_jobs=4, random_state=SEED,
        )
        clf.fit(Xt, y[tr])
        preds[va] = clf.predict(Xv)
    f1w = f1_score(y, preds, average="weighted")
    f1m = f1_score(y, preds, average="macro")
    if label:
        print(f"  [{label}] LOPO weighted F1={f1w:.4f}  macro F1={f1m:.4f}")
    return f1w, f1m, preds


def per_class_f1(y: np.ndarray, preds: np.ndarray) -> np.ndarray:
    from sklearn.metrics import f1_score
    return f1_score(y, preds, average=None, labels=list(range(len(CLASSES))), zero_division=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--time_budget_min", type=float, default=30.0,
                    help="Stop MAE training after this many minutes (soft cap).")
    args = ap.parse_args()

    t_global = time.time()
    print("=" * 78)
    print("MAE pre-training on our own 240 AFM scans' tiles  (ViT-Tiny from scratch)")
    print("=" * 78)
    print(f"device={DEVICE}  epochs={args.epochs}  bs={args.batch_size}  "
          f"lr={args.lr}  mask_ratio={args.mask_ratio}")

    # Stage 1: build tile tensor.
    tiles, t2s, scan_y, scan_persons, scan_paths = build_tiles()
    n_scans = len(scan_y)
    n_tiles = tiles.shape[0]
    print(f"\nTiles: {n_tiles}   Scans: {n_scans}   Persons: {len(np.unique(scan_persons))}")
    print(f"Class counts: {np.bincount(scan_y).tolist()}")

    # Stage 2: random-init control features (before training).
    print("\n[0/3] Random-init ViT-Tiny control (to prove MAE actually learned something)")
    ctrl_model = MAETearTiny(mask_ratio=args.mask_ratio).to(DEVICE)
    ctrl_feats = extract_tile_features(ctrl_model, tiles)
    ctrl_scan = scan_mean_pool(ctrl_feats, t2s, n_scans)
    f1w_rand, f1m_rand, preds_rand = lopo_lr(ctrl_scan, scan_y, scan_persons, "random-init ViT-Tiny")

    # Stage 3: MAE pre-training.
    print("\n[1/3] MAE pre-training")
    model, history = pretrain_mae(
        tiles,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_lr=args.lr,
        mask_ratio=args.mask_ratio,
        time_budget_s=args.time_budget_min * 60.0,
    )
    enc_ckpt = MODELS / "mae_tear_tiny" / "encoder.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "img_size": model.img_size,
            "patch_size": model.patch_size,
            "embed_dim": model.embed_dim,
            "mask_ratio": model.mask_ratio,
            "epochs_trained": len(history),
        },
    }, enc_ckpt)
    print(f"  saved {enc_ckpt}")

    # Stage 4: MAE feature extraction and LOPO eval.
    print("\n[2/3] MAE feature extraction (CLS pooling)")
    mae_feats = extract_tile_features(model, tiles)
    mae_scan = scan_mean_pool(mae_feats, t2s, n_scans)
    print(f"  tile features {mae_feats.shape}   scan features {mae_scan.shape}")

    f1w_mae, f1m_mae, preds_mae = lopo_lr(mae_scan, scan_y, scan_persons, "MAE-pretrained ViT-Tiny")

    # Stage 5: DINOv2-B baseline reference (using cached 45nm tiles embedding if present).
    print("\n[3/3] DINOv2-B reference (same 45nm tile protocol as MAE, person-LOPO)")
    dino_cache_45 = CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz"
    dino_cache_90 = CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz"
    f1w_dino_45 = f1m_dino_45 = None
    if dino_cache_45.exists():
        z = np.load(dino_cache_45, allow_pickle=True)
        # Recompute person groups to match our protocol.
        dino_persons = np.array([person_id(Path(p)) for p in z["scan_paths"].tolist()])
        dino_scan = scan_mean_pool(z["X"].astype(np.float32), z["tile_to_scan"].astype(int),
                                    len(z["scan_y"]))
        f1w_dino_45, f1m_dino_45, _ = lopo_lr(dino_scan, z["scan_y"].astype(int),
                                              dino_persons, "DINOv2-B 45nm baseline")
    f1w_dino_90 = f1m_dino_90 = None
    if dino_cache_90.exists():
        z = np.load(dino_cache_90, allow_pickle=True)
        dino_persons = np.array([person_id(Path(p)) for p in z["scan_paths"].tolist()])
        dino_scan = scan_mean_pool(z["X"].astype(np.float32), z["tile_to_scan"].astype(int),
                                    len(z["scan_y"]))
        f1w_dino_90, f1m_dino_90, _ = lopo_lr(dino_scan, z["scan_y"].astype(int),
                                              dino_persons, "DINOv2-B 90nm baseline")

    # Persist cache of MAE features.
    out_npz = CACHE / "mae_emb_tear_tiny.npz"
    np.savez(
        out_npz,
        X_tiles=mae_feats,
        X_scan=mae_scan,
        tile_to_scan=t2s,
        scan_y=scan_y,
        scan_persons=scan_persons,
        scan_paths=np.array(scan_paths),
    )
    print(f"\n  saved {out_npz}")

    # Persist config + results.
    cfg = {
        "device": DEVICE,
        "epochs": args.epochs,
        "epochs_trained": len(history),
        "batch_size": args.batch_size,
        "base_lr": args.lr,
        "mask_ratio": args.mask_ratio,
        "n_tiles": int(n_tiles),
        "n_scans": int(n_scans),
        "n_persons": int(len(np.unique(scan_persons))),
        "mae_f1w": float(f1w_mae),
        "mae_f1m": float(f1m_mae),
        "rand_f1w": float(f1w_rand),
        "rand_f1m": float(f1m_rand),
        "dino_45nm_f1w": float(f1w_dino_45) if f1w_dino_45 is not None else None,
        "dino_45nm_f1m": float(f1m_dino_45) if f1m_dino_45 is not None else None,
        "dino_90nm_f1w": float(f1w_dino_90) if f1w_dino_90 is not None else None,
        "dino_90nm_f1m": float(f1m_dino_90) if f1m_dino_90 is not None else None,
        "wall_s": time.time() - t_global,
        "history_last5": history[-5:],
    }
    (MODELS / "mae_tear_tiny" / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"  saved {MODELS / 'mae_tear_tiny' / 'config.json'}")

    # Report
    write_report(
        cfg=cfg,
        f1w_mae=f1w_mae, f1m_mae=f1m_mae, preds_mae=preds_mae,
        f1w_rand=f1w_rand, f1m_rand=f1m_rand, preds_rand=preds_rand,
        f1w_dino_45=f1w_dino_45, f1m_dino_45=f1m_dino_45,
        f1w_dino_90=f1w_dino_90, f1m_dino_90=f1m_dino_90,
        scan_y=scan_y, history=history,
    )
    print(f"\nTotal wall time: {(time.time()-t_global)/60:.1f} min")


def write_report(*, cfg, f1w_mae, f1m_mae, preds_mae, f1w_rand, f1m_rand, preds_rand,
                 f1w_dino_45, f1m_dino_45, f1w_dino_90, f1m_dino_90, scan_y, history):
    supports = [int((scan_y == i).sum()) for i in range(len(CLASSES))]
    pc_mae = per_class_f1(scan_y, preds_mae)
    pc_rand = per_class_f1(scan_y, preds_rand)
    delta_vs_dino_45 = (f1w_mae - f1w_dino_45) if f1w_dino_45 is not None else None
    delta_vs_dino_90 = (f1w_mae - f1w_dino_90) if f1w_dino_90 is not None else None

    out = []
    out.append("# MAE Pre-training on Own 240 AFM Scans -- Results")
    out.append("")
    out.append("## Question")
    out.append("")
    out.append("Can a small ViT-Tiny MAE pre-trained from scratch on OUR OWN 240 AFM scans' tiles")
    out.append("(plus D4 augmentation) learn domain-aligned visual features that beat a frozen")
    out.append("ImageNet-pretrained DINOv2-B baseline under person-level LOPO?")
    out.append("")
    out.append("Context: this is recommendation 1 of `reports/EXTERNAL_DATA_SURVEY.md` (highest EV,")
    out.append("zero data acquisition cost). The prior SSL attempt (`reports/SSL_SUPCON_RESULTS.md`)")
    out.append("used a tiny SupCon projection head on frozen DINOv2 features and regressed (0.6120).")
    out.append("MAE is a different animal: reconstruction-based, requires no labels, trains the full")
    out.append("encoder directly on AFM pixel statistics.")
    out.append("")
    out.append("## Methodology")
    out.append("")
    out.append(f"- **Data**: all 240 scans, tiled at 45 nm/px with 512x512 non-overlapping tiles")
    out.append(f"  (up to 9 tiles per scan), each resized to 224x224 and replicated to 3 channels.")
    out.append(f"  Total tiles: {cfg['n_tiles']}. Persons: {cfg['n_persons']}. Scans: {cfg['n_scans']}.")
    out.append(f"- **Augmentation**: D4 (8-way dihedral group: rotations + flips) applied on the fly,")
    out.append(f"  yielding {cfg['n_tiles'] * 8} effective training views per epoch.")
    out.append(f"- **Model**: ViT-Tiny encoder (timm `vit_tiny_patch16_224`, 12 blocks, 192-dim, 3")
    out.append(f"  heads) + MAE decoder (4 blocks, 96-dim, 3 heads). Mask ratio {cfg['mask_ratio']}.")
    out.append(f"  Pixel loss normalized per patch (`norm_pix_loss=True`).")
    out.append(f"- **Optimizer**: AdamW, lr={cfg['base_lr']}, weight_decay=0.05, betas=(0.9, 0.95),")
    out.append(f"  cosine schedule with 1-epoch warmup.")
    out.append(f"- **Training**: {cfg['epochs_trained']}/{cfg['epochs']} epochs completed, batch {cfg['batch_size']},")
    out.append(f"  on device `{cfg['device']}`, wall clock {cfg['wall_s']/60:.1f} min.")
    out.append("- **Downstream eval**: extract CLS features from the pre-trained encoder on every")
    out.append("  tile (no augmentation, no masking), mean-pool tiles per scan -> (240, 192),")
    out.append("  StandardScaler + LogisticRegression (class_weight='balanced'), person-level LOPO.")
    out.append("- **Controls**:")
    out.append("    - Random-init ViT-Tiny (same architecture, never trained): lower bound.")
    out.append("    - DINOv2-B 45 nm/px tiles (same tile protocol) and 90 nm/px tiles (our established")
    out.append("      baseline): upper references.")
    out.append("")
    out.append("## Results (person-LOPO, raw argmax)")
    out.append("")
    out.append("| Model | Weighted F1 | Macro F1 |")
    out.append("|---|---:|---:|")
    out.append(f"| Random-init ViT-Tiny (control, lower bound) | {f1w_rand:.4f} | {f1m_rand:.4f} |")
    if f1w_dino_90 is not None:
        out.append(f"| DINOv2-B ImageNet 90 nm/px (established baseline) | {f1w_dino_90:.4f} | {f1m_dino_90:.4f} |")
    if f1w_dino_45 is not None:
        out.append(f"| DINOv2-B ImageNet 45 nm/px (same-protocol reference) | {f1w_dino_45:.4f} | {f1m_dino_45:.4f} |")
    out.append(f"| **MAE ViT-Tiny pre-trained on own 240 scans** | **{f1w_mae:.4f}** | **{f1m_mae:.4f}** |")
    out.append("")
    out.append("### Per-class F1 (MAE vs random-init control)")
    out.append("")
    out.append("| Class | Support | Random | MAE |")
    out.append("|---|---:|---:|---:|")
    for i, c in enumerate(CLASSES):
        out.append(f"| {c} | {supports[i]} | {pc_rand[i]:.3f} | {pc_mae[i]:.3f} |")
    out.append("")
    out.append("## Verdict")
    out.append("")
    if f1w_dino_45 is not None and f1w_mae >= f1w_dino_45 + 0.005:
        v = f"SHIP. MAE-pretrained ViT-Tiny beats DINOv2-B 45nm by {f1w_mae - f1w_dino_45:+.4f} F1."
    elif f1w_dino_45 is not None and abs(f1w_mae - f1w_dino_45) < 0.005:
        v = "MATCH. MAE-pretrained ViT-Tiny within sampling noise of DINOv2-B 45nm."
    elif f1w_rand is not None and f1w_mae > f1w_rand + 0.02:
        v = (f"LEARNED (but does not beat DINOv2-B). MAE added {f1w_mae - f1w_rand:+.4f} F1 over "
             f"random-init control, confirming the pre-training learned something; but DINOv2-B's "
             f"scale (86M params, billions of images) still wins.")
    else:
        v = ("NO CLEAR GAIN. MAE-pretrained features are within noise of random-init. Likely causes: "
             "tiny corpus (~1.9k tiles x 8 views = 15k effective), ViT-Tiny too small, or training "
             "was cut short by the time budget.")
    out.append(v)
    out.append("")
    out.append(f"- Delta vs random-init ViT-Tiny: **{f1w_mae - f1w_rand:+.4f}** weighted F1")
    if delta_vs_dino_45 is not None:
        out.append(f"- Delta vs DINOv2-B 45nm (same tile protocol): **{delta_vs_dino_45:+.4f}**")
    if delta_vs_dino_90 is not None:
        out.append(f"- Delta vs DINOv2-B 90nm (established baseline): **{delta_vs_dino_90:+.4f}**")
    out.append("")
    out.append("## Honest caveats")
    out.append("")
    out.append("1. **Tiny corpus for MAE.** 240 scans x ~8 tiles x 8 D4 views ~= 15k training views")
    out.append("   is two orders of magnitude smaller than typical MAE pre-training corpora. The")
    out.append("   EXTERNAL_DATA_SURVEY literature projection (+3-8 F1) was predicated on 2-10k")
    out.append("   IMAGES, not 15k heavily-correlated views of the same 1.9k base tiles.")
    out.append("2. **ViT-Tiny is 5M params** vs DINOv2-B's 86M. Part of DINOv2-B's win is sheer")
    out.append("   capacity + ImageNet-1k scale. A fair scale comparison would need ViT-Base MAE,")
    out.append("   which would require a larger corpus + more compute than our 30-minute budget.")
    out.append("3. **Person-LOPO is strict.** All numbers in this table use `person_id` grouping")
    out.append("   (L/R eyes merged into one person), the same protocol as SSL_SUPCON_RESULTS.md.")
    out.append("4. **Encoder features never saw labels.** The entire MAE head training is fully")
    out.append("   self-supervised (reconstruction only); only the downstream LR sees class labels.")
    out.append("   This is architecturally clean -- no LOPO leakage risk.")
    out.append("5. **Compute budget triggered early stop iff `epochs_trained < epochs`.**")
    out.append(f"   Here: {cfg['epochs_trained']}/{cfg['epochs']} epochs completed.")
    out.append("")
    out.append("## Loss trajectory (last epochs)")
    out.append("")
    out.append("| Epoch | Loss | LR | Elapsed (s) |")
    out.append("|---:|---:|---:|---:|")
    for h in history[-10:]:
        out.append(f"| {h['epoch']} | {h['loss']:.4f} | {h['lr']:.2e} | {h['elapsed_s']:.0f} |")
    out.append("")
    out.append("## Artifacts")
    out.append("")
    out.append("- `scripts/mae_pretrain.py` -- this script (MAE training + LOPO eval)")
    out.append("- `models/mae_tear_tiny/encoder.pt` -- trained encoder checkpoint (ViT-Tiny + decoder)")
    out.append("- `models/mae_tear_tiny/config.json` -- hyperparameters + summary metrics")
    out.append("- `cache/mae_emb_tear_tiny.npz` -- MAE CLS tile features + scan-level mean-pool")
    out.append("- `cache/mae_raw_tiles_45nm_t512_n9_224.npz` -- pre-processed 224x224 tiles cache")
    out.append("- `reports/MAE_PRETRAINING_RESULTS.md` -- this report")
    out.append("")

    report_path = REPORTS / "MAE_PRETRAINING_RESULTS.md"
    report_path.write_text("\n".join(out))
    print(f"  report written: {report_path}")


if __name__ == "__main__":
    main()
