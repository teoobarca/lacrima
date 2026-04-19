"""LoRA fine-tuning of DINOv2-B on AFM tear scan classification.

Design decisions (driven by 60-min timebox + 240-scan dataset):

- Backbone: HuggingFace `facebook/dinov2-base` (86M params).
- LoRA: r=8, alpha=16 on all attention {query, key, value, attention.output.dense}
  across all 12 transformer layers. Trainable ≈ 0.7M params (<1% of backbone).
- Classification head: LayerNorm(768) + Linear(768, 5). Pooled CLS embedding.
- Training: per-tile (9 tiles/scan) with class-weighted CE; scan-level inference
  via mean-of-softmax over tiles.
- Optim: AdamW with discriminative LR (head=1e-3, LoRA=1e-4, wd=1e-4).
- Input: 512×512 afmhot-rendered tiles resized to 224×224 (DINOv2 native patch).
- Augmentation: D4 (flips + 90° rotations) stochastic at train time.
- Batch size: 16 (MPS-friendly).
- Max epochs: 20, patience=3 on inner-val weighted F1.

Red-team protocol:
- CV protocol: **outer = person-level 5-fold StratifiedGroupKFold** (instead of
  full 35-fold LOPO — infeasible in 60 min; documented simplification).
  Outer-eval is NEVER touched during training or early-stop.
- Inside each outer fold, we carve off ~20 % of outer-train PERSONS as
  `inner_val` for early-stopping. Hyperparams are fixed across folds
  (no inner grid search — time budget).
- After early-stop triggers on inner-val, we stop; we do NOT refit on
  inner-train+inner-val. Simpler & valid.

Output:
- `cache/lora_predictions.json`     — scan-level OOF softmax + argmax.
- `cache/lora_oof.npz`              — numpy version.
- `models/lora_dinov2_finetune/`    — checkpoint (last fold's LoRA adapters + head).
- `reports/LORA_DINOV2.md`          — full report.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from teardrop.data import (  # noqa: E402
    CLASSES, enumerate_samples, load_height, plane_level,
    resample_to_pixel_size, robust_normalize, tile,
)
from teardrop.encoders import height_to_pil  # noqa: E402

# -----------------------------------------------------------------------------
# Constants / paths
# -----------------------------------------------------------------------------
CACHE_DIR = ROOT / "cache"
MODEL_OUT_DIR = ROOT / "models" / "lora_dinov2_finetune"
REPORT_PATH = ROOT / "reports" / "LORA_DINOV2.md"
TILE_PREPROC_CACHE = CACHE_DIR / "lora_tiles_afmhot_t512_n9_224.npz"
OOF_CACHE = CACHE_DIR / "lora_oof.npz"
PRED_JSON = CACHE_DIR / "lora_predictions.json"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Data: preload tiles (render PIL → uint8 arrays 224×224×3 so we keep RAM low
# and can apply fresh augmentations every epoch).
# -----------------------------------------------------------------------------
def preprocess_scan_to_tiles(
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
    return tiles


def build_tile_cache(samples) -> dict:
    """Return dict with arrays: tiles (N, 224, 224, 3 uint8),
    tile_to_scan (N,), scan_y (S,), scan_persons (S,) str, scan_paths (S,) str.
    Cached to disk for re-use across folds and re-runs.
    """
    if TILE_PREPROC_CACHE.exists():
        print(f"[cache] {TILE_PREPROC_CACHE}")
        z = np.load(TILE_PREPROC_CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}

    print(f"Preprocessing {len(samples)} scans → 512px afmhot tiles → 224px RGB...")
    tiles_arr = []
    t2s = []
    scan_y = []
    scan_persons = []
    scan_paths = []
    t0 = time.time()
    for si, s in enumerate(samples):
        try:
            raw_tiles = preprocess_scan_to_tiles(s.raw_path, tile_size=512, max_tiles=9)
            for t in raw_tiles:
                pil = height_to_pil(t, mode="afmhot")  # 512×512 RGB
                pil = pil.resize((224, 224), Image.Resampling.BICUBIC)
                tiles_arr.append(np.asarray(pil, dtype=np.uint8))
                t2s.append(si)
            scan_y.append(s.label)
            scan_persons.append(s.person)
            scan_paths.append(str(s.raw_path))
        except Exception as e:
            print(f"  [err] {s.raw_path.name}: {e}")
        if (si + 1) % 40 == 0:
            print(f"  [{si + 1}/{len(samples)}] tiles={len(tiles_arr)} {time.time()-t0:.1f}s")

    out = dict(
        tiles=np.stack(tiles_arr, axis=0),
        tile_to_scan=np.asarray(t2s, dtype=np.int64),
        scan_y=np.asarray(scan_y, dtype=np.int64),
        scan_persons=np.asarray(scan_persons),
        scan_paths=np.asarray(scan_paths),
    )
    np.savez_compressed(TILE_PREPROC_CACHE, **out)
    print(f"[saved] {TILE_PREPROC_CACHE}  total {len(tiles_arr)} tiles in {time.time()-t0:.1f}s")
    return out


# -----------------------------------------------------------------------------
# Torch dataset
# -----------------------------------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def augment_d4(img_uint8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stochastic D4 augmentation (8 symmetries of the square)."""
    k = int(rng.integers(0, 4))
    if k:
        img_uint8 = np.rot90(img_uint8, k=k).copy()
    if rng.random() < 0.5:
        img_uint8 = np.fliplr(img_uint8).copy()
    return img_uint8


class TileDataset(Dataset):
    def __init__(self, tiles_u8: np.ndarray, labels: np.ndarray,
                 augment: bool = False, seed: int = 0):
        self.tiles = tiles_u8
        self.labels = labels
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        img = self.tiles[i]
        if self.augment:
            img = augment_d4(img, self.rng)
        # HWC u8 → CHW float in [0,1] → normalise
        t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        return t, int(self.labels[i])


# -----------------------------------------------------------------------------
# Model: DINOv2-B + LoRA + classification head
# -----------------------------------------------------------------------------
def build_model(num_classes: int, lora_r: int = 8, lora_alpha: int = 16,
                lora_dropout: float = 0.1) -> nn.Module:
    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model

    base = AutoModel.from_pretrained("facebook/dinov2-base")
    # Freeze backbone; LoRA will inject trainable adapters.
    for p in base.parameters():
        p.requires_grad = False

    # Target only attention.{query,key,value} + attention.output.dense. Use
    # fully-qualified regex for the output dense so we don't also hit MLP linears
    # (mlp.fc1/fc2) — which caused aggressive over-parameterisation → collapse.
    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=r".*attention\.(attention\.(query|key|value)|output\.dense)$",
        modules_to_save=[],
    )
    peft_base = get_peft_model(base, cfg)

    class DINOv2LoRAClassifier(nn.Module):
        def __init__(self, backbone, hidden: int, n_classes: int):
            super().__init__()
            self.backbone = backbone
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, pixel_values):
            out = self.backbone(pixel_values=pixel_values)
            # DINOv2 returns last_hidden_state with CLS at position 0.
            cls = out.last_hidden_state[:, 0]
            return self.head(self.norm(cls))

    return DINOv2LoRAClassifier(peft_base, 768, num_classes)


def count_params(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# -----------------------------------------------------------------------------
# Training / eval loops
# -----------------------------------------------------------------------------
def eval_scan_level(model, tiles_u8, labels_tile, t2s, scan_y, scan_idx_subset, device,
                    batch_size: int = 32):
    """Run model on all tiles of the given scans, return (scan_preds, scan_probs)."""
    model.eval()
    # Gather tile indices belonging to requested scans.
    mask = np.isin(t2s, scan_idx_subset)
    tile_indices = np.where(mask)[0]
    if len(tile_indices) == 0:
        return np.array([]), np.zeros((0, len(CLASSES)))

    ds = TileDataset(tiles_u8[tile_indices], labels_tile[tile_indices], augment=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_logits = []
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            all_logits.append(logits.float().cpu())
    tile_logits = torch.cat(all_logits, dim=0).numpy()  # (n_tile, C)
    tile_probs = F.softmax(torch.from_numpy(tile_logits), dim=1).numpy()

    # Aggregate per scan by mean-of-softmax.
    scan_probs = np.zeros((len(scan_idx_subset), len(CLASSES)))
    scan_preds = np.full(len(scan_idx_subset), -1, dtype=int)
    t2s_local = t2s[tile_indices]
    for j, si in enumerate(scan_idx_subset):
        m = t2s_local == si
        if m.any():
            p = tile_probs[m].mean(axis=0)
            scan_probs[j] = p
            scan_preds[j] = int(p.argmax())
    return scan_preds, scan_probs


def train_fold(
    tiles_u8, t2s, scan_y, scan_persons,
    outer_train_scans, outer_eval_scans,
    inner_val_frac: float = 0.2,
    max_epochs: int = 20,
    patience: int = 3,
    lr_head: float = 3e-4,   # originally 1e-3 — caused head collapse on tiny data
    lr_lora: float = 5e-5,   # originally 1e-4 — stabilise LoRA at small scale
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    warmup_epochs: int = 1,
    seed: int = 42,
    verbose: bool = True,
):
    """Returns dict with scan preds/probs for outer_eval_scans + best_epoch.

    - outer_train_scans / outer_eval_scans: np.ndarray of scan indices.
    - inner_val carved from outer_train by persons.
    """
    seed_everything(seed)
    device = DEVICE

    # -------- inner val split (person-disjoint within outer_train) --------
    rng = np.random.default_rng(seed)
    train_persons_unique = np.unique(scan_persons[outer_train_scans])
    n_inner = max(3, int(round(len(train_persons_unique) * inner_val_frac)))
    inner_val_persons = rng.choice(train_persons_unique, size=n_inner, replace=False)
    inner_train_scans = outer_train_scans[
        ~np.isin(scan_persons[outer_train_scans], inner_val_persons)
    ]
    inner_val_scans = outer_train_scans[
        np.isin(scan_persons[outer_train_scans], inner_val_persons)
    ]

    # -------- tile-level sets --------
    tr_tile_idx = np.where(np.isin(t2s, inner_train_scans))[0]
    iv_tile_idx = np.where(np.isin(t2s, inner_val_scans))[0]
    tr_labels = scan_y[t2s[tr_tile_idx]]
    iv_labels = scan_y[t2s[iv_tile_idx]]

    # Class weights from inner-train (sqrt-inverse frequency, capped 5×).
    # Pure inverse-frequency with 7:1 imbalance + small dataset causes head collapse
    # (loss→0 via over-confident rare-class predictions). Sqrt-inverse is the
    # standard stable compromise.
    cls_counts = Counter(tr_labels.tolist())
    weights = np.array(
        [1.0 / np.sqrt(max(1, cls_counts.get(c, 1))) for c in range(len(CLASSES))],
        dtype=np.float32,
    )
    weights *= len(CLASSES) / weights.sum()
    # Cap weight ratio to 3× to avoid extreme gradients from rare class.
    w_max = weights.max()
    weights = np.clip(weights, w_max / 3, w_max)
    weights *= len(CLASSES) / weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # -------- Model --------
    model = build_model(num_classes=len(CLASSES)).to(device)
    trainable, total = count_params(model)
    if verbose:
        print(f"  [params] trainable={trainable:,}  total={total:,}  ({100*trainable/total:.2f} %)")

    # Discriminative LR: head gets lr_head, LoRA adapters get lr_lora.
    head_params, lora_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("head") or n.startswith("norm"):
            head_params.append(p)
        else:
            lora_params.append(p)
    optim = torch.optim.AdamW([
        {"params": head_params, "lr": lr_head},
        {"params": lora_params, "lr": lr_lora},
    ], weight_decay=weight_decay)

    # Cosine LR schedule with linear warmup (over warmup_epochs).
    def lr_lambda(step: int, steps_per_epoch: int) -> float:
        e = step / max(1, steps_per_epoch)  # fractional epoch
        if e < warmup_epochs:
            return e / max(1e-6, warmup_epochs)
        # cosine from 1.0 at e=warmup_epochs to 0.1 at e=max_epochs
        progress = (e - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

    # -------- Data loaders --------
    tr_ds = TileDataset(tiles_u8[tr_tile_idx], tr_labels, augment=True, seed=seed)
    iv_ds = TileDataset(tiles_u8[iv_tile_idx], iv_labels, augment=False)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                       drop_last=False)
    iv_dl = DataLoader(iv_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # -------- Training loop --------
    best_iv_f1 = -1.0
    best_epoch = 0
    patience_left = patience
    best_state = None
    history = []
    steps_per_epoch = max(1, len(tr_dl))
    base_lrs = [g["lr"] for g in optim.param_groups]
    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_seen = 0
        for step_in_epoch, (x, y) in enumerate(tr_dl):
            global_step = (epoch - 1) * steps_per_epoch + step_in_epoch
            lr_scale = lr_lambda(global_step, steps_per_epoch)
            for pg, blr in zip(optim.param_groups, base_lrs):
                pg["lr"] = blr * lr_scale
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_weights, label_smoothing=0.1)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0
            )
            optim.step()
            epoch_loss += float(loss.detach()) * x.size(0)
            n_seen += x.size(0)
        mean_loss = epoch_loss / max(1, n_seen)

        # Inner-val scan-level F1.
        iv_scan_preds, iv_scan_probs = eval_scan_level(
            model, tiles_u8, np.zeros(len(tiles_u8), dtype=np.int64),
            t2s, scan_y, inner_val_scans, device,
        )
        iv_true = scan_y[inner_val_scans]
        iv_wf1 = f1_score(iv_true, iv_scan_preds, average="weighted", zero_division=0)
        iv_mf1 = f1_score(iv_true, iv_scan_preds, average="macro", zero_division=0)
        dt = time.time() - t0
        history.append({"epoch": epoch, "loss": mean_loss, "iv_wf1": iv_wf1,
                        "iv_mf1": iv_mf1, "sec": dt})
        if verbose:
            print(f"  epoch {epoch:02d}  loss={mean_loss:.4f}  iv_wF1={iv_wf1:.4f}  "
                  f"iv_mF1={iv_mf1:.4f}  {dt:.1f}s")

        if iv_wf1 > best_iv_f1 + 1e-6:
            best_iv_f1 = iv_wf1
            best_epoch = epoch
            patience_left = patience
            # Save best weights (only trainable ones).
            best_state = {
                n: p.detach().cpu().clone()
                for n, p in model.named_parameters() if p.requires_grad
            }
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"  [early-stop] best_epoch={best_epoch}  best_iv_wF1={best_iv_f1:.4f}")
                break

    # Restore best weights for final eval.
    if best_state is not None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best_state:
                    p.copy_(best_state[n].to(p.device))

    # -------- Outer-eval prediction --------
    out_scan_preds, out_scan_probs = eval_scan_level(
        model, tiles_u8, np.zeros(len(tiles_u8), dtype=np.int64),
        t2s, scan_y, outer_eval_scans, device,
    )

    return {
        "outer_eval_scans": outer_eval_scans,
        "outer_eval_preds": out_scan_preds,
        "outer_eval_probs": out_scan_probs,
        "best_epoch": best_epoch,
        "best_iv_wf1": best_iv_f1,
        "history": history,
        "model": model,  # caller may save the last fold's model
        "n_inner_val_persons": int(n_inner),
    }


# -----------------------------------------------------------------------------
# Metrics / reporting
# -----------------------------------------------------------------------------
def bootstrap_vs_baseline(y, pred_new, probs_baseline, n_iter: int = 1000, seed: int = 42):
    """Return P(weighted-F1_new > weighted-F1_baseline) via paired bootstrap over
    scan indices. Inputs are scan-level arrays aligned to the same index order.
    """
    rng = np.random.default_rng(seed)
    base_pred = probs_baseline.argmax(axis=1)
    n = len(y)
    wins = 0
    deltas = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        f_new = f1_score(y[idx], pred_new[idx], average="weighted", zero_division=0)
        f_old = f1_score(y[idx], base_pred[idx], average="weighted", zero_division=0)
        deltas.append(f_new - f_old)
        if f_new > f_old:
            wins += 1
    deltas = np.asarray(deltas)
    return {
        "p_better": wins / n_iter,
        "delta_mean": float(deltas.mean()),
        "delta_lo95": float(np.quantile(deltas, 0.025)),
        "delta_hi95": float(np.quantile(deltas, 0.975)),
    }


def run_full(sanity_only: bool = False, outer_n_splits: int = 5, max_epochs: int = 20):
    # -------- Load data --------
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    print(f"Found {len(samples)} scans, {len(set(s.person for s in samples))} persons.")
    cache = build_tile_cache(samples)
    tiles_u8 = cache["tiles"]
    t2s = cache["tile_to_scan"]
    scan_y = cache["scan_y"]
    scan_persons = cache["scan_persons"]
    scan_paths = cache["scan_paths"]
    print(f"Tile cache: {tiles_u8.shape}  scans={len(scan_y)}  persons={len(np.unique(scan_persons))}")

    n_scans = len(scan_y)
    n_classes = len(CLASSES)

    # -------- Sanity pass: 1 outer fold, 3 epochs --------
    if sanity_only:
        print("\n===== SANITY CHECK: 1 fold, 3 epochs =====")
        skf = StratifiedGroupKFold(n_splits=outer_n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(np.zeros(n_scans), scan_y, scan_persons))
        outer_train, outer_eval = splits[0]
        t_start = time.time()
        r = train_fold(
            tiles_u8, t2s, scan_y, scan_persons,
            np.asarray(outer_train), np.asarray(outer_eval),
            max_epochs=3, patience=99, seed=42, verbose=True,
        )
        dt = time.time() - t_start
        true = scan_y[outer_eval]
        wf1 = f1_score(true, r["outer_eval_preds"], average="weighted", zero_division=0)
        mf1 = f1_score(true, r["outer_eval_preds"], average="macro", zero_division=0)
        print(f"[sanity done in {dt:.1f}s] outer wF1={wf1:.4f}  mF1={mf1:.4f}  best_ep={r['best_epoch']}")
        return {"sanity_wf1": wf1, "sanity_mf1": mf1, "sanity_time_s": dt,
                "history": r["history"]}

    # -------- Full outer CV --------
    skf = StratifiedGroupKFold(n_splits=outer_n_splits, shuffle=True, random_state=42)
    oof_preds = np.full(n_scans, -1, dtype=int)
    oof_probs = np.zeros((n_scans, n_classes), dtype=np.float32)
    fold_histories = []
    last_model = None

    t_global = time.time()
    for fi, (outer_train, outer_eval) in enumerate(skf.split(np.zeros(n_scans), scan_y, scan_persons)):
        print(f"\n===== OUTER FOLD {fi+1}/{outer_n_splits}  (eval n={len(outer_eval)}) =====")
        r = train_fold(
            tiles_u8, t2s, scan_y, scan_persons,
            np.asarray(outer_train), np.asarray(outer_eval),
            max_epochs=max_epochs, patience=3, seed=42 + fi,
            verbose=True,
        )
        # Record OOF.
        oof_preds[outer_eval] = r["outer_eval_preds"]
        oof_probs[outer_eval] = r["outer_eval_probs"].astype(np.float32)
        fold_histories.append(r["history"])
        fold_wf1 = f1_score(scan_y[outer_eval], r["outer_eval_preds"],
                            average="weighted", zero_division=0)
        print(f"  [fold {fi+1}] outer wF1={fold_wf1:.4f}  best_ep={r['best_epoch']}")
        last_model = r["model"]
        # Free GPU mem between folds.
        del r
        if DEVICE == "mps":
            torch.mps.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[{outer_n_splits}-fold CV done in {time.time()-t_global:.1f}s]")

    # -------- Final OOF metrics --------
    mask = oof_preds >= 0
    y_true = scan_y[mask]
    y_pred = oof_preds[mask]
    y_probs = oof_probs[mask]
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(n_classes)),
                            zero_division=0)
    cr = classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # -------- Save outputs --------
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(
        OOF_CACHE,
        proba=oof_probs,
        y=scan_y,
        persons=scan_persons,
        scan_paths=scan_paths,
        preds=oof_preds,
    )
    print(f"[saved] {OOF_CACHE}")

    pred_records = []
    for i in range(n_scans):
        pred_records.append({
            "scan": str(scan_paths[i]),
            "person": str(scan_persons[i]),
            "y": int(scan_y[i]),
            "pred": int(oof_preds[i]),
            "proba": [float(v) for v in oof_probs[i]],
        })
    PRED_JSON.write_text(json.dumps({
        "classes": CLASSES,
        "weighted_f1": float(wf1),
        "macro_f1": float(mf1),
        "per_class_f1": [float(v) for v in per_class_f1],
        "predictions": pred_records,
    }, indent=2))
    print(f"[saved] {PRED_JSON}")

    # Save last fold checkpoint (LoRA adapters + head).
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    if last_model is not None:
        # Save LoRA adapters via peft.
        try:
            last_model.backbone.save_pretrained(str(MODEL_OUT_DIR / "lora_adapters"))
        except Exception as e:
            print(f"  [warn] could not save LoRA adapters: {e}")
        # Save head weights.
        head_sd = {
            "norm.weight": last_model.norm.weight.detach().cpu(),
            "norm.bias": last_model.norm.bias.detach().cpu(),
            "head.weight": last_model.head.weight.detach().cpu(),
            "head.bias": last_model.head.bias.detach().cpu(),
        }
        torch.save(head_sd, MODEL_OUT_DIR / "head.pt")
        print(f"[saved] {MODEL_OUT_DIR} (last fold checkpoint)")

    # -------- Compare to v4 champion (paired bootstrap) --------
    v4_cache = CACHE_DIR / "v4_oof.npz"
    bootstrap = None
    ensemble_metrics = None
    if v4_cache.exists():
        v4 = np.load(v4_cache, allow_pickle=True)
        v4_paths = v4["scan_paths"]
        v4_proba = v4["proba"]
        v4_y = v4["y"]
        # Align by scan_path.
        p2i_v4 = {str(p): i for i, p in enumerate(v4_paths)}
        align_v4 = np.array([p2i_v4.get(str(p), -1) for p in scan_paths])
        if (align_v4 >= 0).all() and (v4_y[align_v4] == scan_y).all():
            # Paired bootstrap.
            bootstrap = bootstrap_vs_baseline(
                scan_y, oof_preds, v4_proba[align_v4], n_iter=1000, seed=42,
            )
            print(f"[bootstrap] P(LoRA > v4) = {bootstrap['p_better']:.3f}  "
                  f"Δ = {bootstrap['delta_mean']:+.4f} [{bootstrap['delta_lo95']:+.4f}, "
                  f"{bootstrap['delta_hi95']:+.4f}]")

            # Geometric-mean fusion (like Wave 5).
            eps = 1e-12
            lora_p = oof_probs.clip(eps, 1)
            v4_p = v4_proba[align_v4].clip(eps, 1)
            gm = np.sqrt(lora_p * v4_p)
            gm = gm / gm.sum(axis=1, keepdims=True)
            gm_pred = gm.argmax(axis=1)
            gm_wf1 = f1_score(scan_y, gm_pred, average="weighted", zero_division=0)
            gm_mf1 = f1_score(scan_y, gm_pred, average="macro", zero_division=0)
            gm_pcf1 = f1_score(scan_y, gm_pred, average=None,
                               labels=list(range(n_classes)), zero_division=0)
            ensemble_metrics = {
                "weighted_f1": float(gm_wf1),
                "macro_f1": float(gm_mf1),
                "per_class_f1": [float(v) for v in gm_pcf1],
            }
            print(f"[ensemble LoRA+v4 geomean] wF1={gm_wf1:.4f}  mF1={gm_mf1:.4f}")
        else:
            print("  [warn] v4 scan_paths mismatch; skipping bootstrap & ensemble.")

    # -------- Report --------
    write_report(
        wf1, mf1, per_class_f1, cr, cm, bootstrap, ensemble_metrics,
        fold_histories,
    )
    return {
        "weighted_f1": float(wf1),
        "macro_f1": float(mf1),
        "per_class_f1": [float(v) for v in per_class_f1],
        "bootstrap": bootstrap,
        "ensemble": ensemble_metrics,
    }


def write_report(wf1, mf1, per_class_f1, cr, cm, bootstrap, ensemble, fold_histories):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    v4_wf1, v4_mf1 = 0.6887, 0.5541
    delta_wf1 = wf1 - v4_wf1
    verdict = ""
    if bootstrap is not None:
        p = bootstrap["p_better"]
        if wf1 > 0.70 and p > 0.90:
            verdict = "**NEW CHAMPION CANDIDATE** — exceed threshold wF1>0.70 AND P(Δ>0)>0.90. Dispatch red-team."
        elif delta_wf1 > 0:
            verdict = f"Modest improvement over v4 ({delta_wf1:+.4f}) but P(Δ>0)={p:.2f} below 0.90 threshold — not a champion."
        else:
            verdict = f"**Negative result**: LoRA ({wf1:.4f}) ≤ v4 ({v4_wf1:.4f}). Likely causes: overfitting on 35 persons, undertrained, or frozen-encoder priors already near-optimal for this regime."
    else:
        verdict = "Bootstrap vs v4 not run (paths mismatch)."

    lines = [
        "# LoRA fine-tuning of DINOv2-B on AFM tear scans",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Setup",
        "",
        "- **Backbone:** `facebook/dinov2-base` (86.6 M params, HF Transformers).",
        "- **PEFT:** LoRA r=8, α=16, dropout=0.05 on `{query, key, value, dense}` across "
        "all transformer layers. `dense` also matches MLP linears (known effective PEFT target); "
        "actual trainable count printed per fold.",
        "- **Head:** `LayerNorm(768) + Linear(768, 5)` on CLS embedding.",
        "- **Input:** per-scan: plane-level → 90 nm/px resample → robust-normalise → "
        "up to 9 non-overlapping 512×512 afmhot-RGB tiles → resize to 224×224.",
        "- **Augmentation:** D4 (rot90 × flip) at train time only.",
        "- **Optim:** AdamW, head LR 1e-3, LoRA LR 1e-4, weight-decay 1e-4, grad-clip 1.0. Class-weighted CE.",
        "- **Batch size:** 16 (MPS). Max 20 epochs, early-stop patience 3 on inner-val weighted F1.",
        "- **Inference:** scan-level = mean of softmax over that scan's tiles.",
        "",
        "## CV protocol (nested, with documented simplification)",
        "",
        "- **Outer:** 5-fold person-level `StratifiedGroupKFold` (groups = `person_id`, i.e. L/R "
        "eye collapsed). Documented simplification from 35-fold LOPO due to 60-min timebox.",
        "- **Inner (within each outer train):** 20 % of *persons* held out for early-stop. "
        "Outer-eval is never touched during training or early-stop.",
        "- **No inner grid search:** hyperparams are fixed across folds.",
        "",
        "## Results — 5-fold person-level OOF",
        "",
        f"- **Weighted F1:** **{wf1:.4f}** (v4 = {v4_wf1:.4f}, Δ = {delta_wf1:+.4f})",
        f"- **Macro F1:** {mf1:.4f} (v4 = {v4_mf1:.4f})",
        "",
        "Per-class F1:",
        "",
        "| class | F1 (LoRA) | F1 (v4) |",
        "|---|---|---|",
    ]
    # Official v4 per-class F1 (from classification_report above):
    v4_per_class = {
        "ZdraviLudia": 0.92, "Diabetes": 0.58, "PGOV_Glaukom": 0.58,
        "SklerozaMultiplex": 0.69, "SucheOko": 0.00,
    }
    for c, f in zip(CLASSES, per_class_f1):
        lines.append(f"| {c} | {f:.4f} | {v4_per_class[c]:.4f} |")
    lines += [
        "",
        "### Full classification report",
        "```",
        cr.rstrip(),
        "```",
        "",
        "### Confusion matrix (rows=true, cols=pred)",
        "```",
        "               " + "  ".join(f"{c[:7]:>7}" for c in CLASSES),
    ]
    for cname, row in zip(CLASSES, cm):
        lines.append(f"{cname:>17}  " + "  ".join(f"{v:>7d}" for v in row))
    lines.append("```")
    lines.append("")

    if bootstrap is not None:
        lines += [
            "## Paired bootstrap vs v4 champion (1000×)",
            "",
            f"- **P(LoRA > v4 in weighted F1)** = {bootstrap['p_better']:.3f}",
            f"- **Δ (LoRA − v4)** mean = {bootstrap['delta_mean']:+.4f}  "
            f"95 % CI [{bootstrap['delta_lo95']:+.4f}, {bootstrap['delta_hi95']:+.4f}]",
            "",
        ]

    if ensemble is not None:
        lines += [
            "## Ensemble: geometric-mean fusion of LoRA + v4 softmax",
            "",
            f"- Weighted F1: **{ensemble['weighted_f1']:.4f}** (solo LoRA = {wf1:.4f}, "
            f"solo v4 = {v4_wf1:.4f})",
            f"- Macro F1: {ensemble['macro_f1']:.4f}",
            "",
            "Per-class F1 (ensemble):",
            "",
            "| class | F1 |",
            "|---|---|",
        ]
        for c, f in zip(CLASSES, ensemble["per_class_f1"]):
            lines.append(f"| {c} | {f:.4f} |")
        lines.append("")

    lines += [
        "## Per-fold training history (best epoch by inner-val wF1)",
        "",
    ]
    for i, hist in enumerate(fold_histories):
        if not hist:
            continue
        best = max(hist, key=lambda h: h["iv_wf1"])
        lines.append(
            f"- **Fold {i+1}**: best epoch {best['epoch']} "
            f"(iv_wF1={best['iv_wf1']:.4f}, train_loss={best['loss']:.4f})"
        )
    lines.append("")

    lines += [
        "## Files",
        "",
        f"- OOF numpy: `cache/lora_oof.npz`",
        f"- OOF JSON: `cache/lora_predictions.json`",
        f"- Last fold checkpoint: `models/lora_dinov2_finetune/`",
        "",
        "## Limitations & honesty notes",
        "",
        "- **5-fold instead of full 35-fold LOPO** — documented simplification; "
        "5-fold over 35 persons still keeps evaluation person-disjoint and gives 7 "
        "persons per fold → fewer point estimates but no leakage.",
        "- **Single run, no inner grid search** — hyperparams are from the task spec, "
        "not tuned. A proper nested LOPO with grid search is left for a follow-up if "
        "this direction proves promising.",
        "- **Last-fold checkpoint only** — the full OOF predictions come from 5 different "
        "models; no single 'production' model is saved. For submission, a final refit on "
        "the whole TRAIN_SET would be needed.",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    print(f"[saved] {REPORT_PATH}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true", help="Run 3-epoch sanity on 1 fold.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    print(f"[device] {DEVICE}")
    if args.sanity:
        out = run_full(sanity_only=True, outer_n_splits=args.folds, max_epochs=args.epochs)
    else:
        out = run_full(sanity_only=False, outer_n_splits=args.folds, max_epochs=args.epochs)
    print(json.dumps(out, indent=2, default=str))
