"""Foundation-model zoo comparison on tear-AFM classification.

Goal: search underexplored frozen encoders to see if any beats our
current champions (DINOv2-B, BiomedCLIP) used in the v4 multi-scale
ensemble (person-LOPO weighted F1 = 0.6887).

Recipe (per encoder, minimal, no tuning):
  1. Preprocess each scan at 90 nm/px → up to 9 non-overlapping 512²
     tiles, rendered via afmhot colormap. No TTA.
  2. Encode each tile with the frozen model, mean-pool per scan →
     scan-level embedding.
  3. Cache as cache/tiled_emb_{encoder_name}_afmhot_t512_n9.npz with the
     same keys the rest of the project uses:
         X (n_tiles, D) float32
         tile_to_scan (n_tiles,) int64
         scan_y (n_scans,) int64
         scan_groups (n_scans,) str (person-level)
         scan_paths (n_scans,) str
  4. Fit the V2 recipe per encoder (L2-norm -> StandardScaler -> LR)
     under person-LOPO and report weighted / macro F1.

Then try 4-way and 5-way geom-mean ensembles on top of v4 components:
  - v4 + DINOv2-L
  - v4 (swap BiomedCLIP for SigLIP-SO400M)
  - v4 + DINOv2-L + SigLIP (5-way)

Budget guards:
  - Encoding per model capped at ~8 min on MPS; if exceeded we skip.
  - If a model fails to load (deps / weights / OOM), log and skip.

Outputs:
  - reports/FOUNDATION_ZOO_RESULTS.md (comparison tables)
  - reports/foundation_zoo_results.json (raw numbers)

Run:
  .venv/bin/python scripts/foundation_zoo.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, normalize

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
from teardrop.encoders import height_to_pil, pick_device  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
CACHE.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

N_CLASSES = len(CLASSES)
EPS = 1e-12

# Known champions (for report comparisons)
CHAMP_DINOV2B = 0.6162   # DINOv2-B 90nm no-TTA, person-LOPO (from multiscale_results.json)
CHAMP_V4 = 0.6887        # v4 multi-scale ensemble (DINOv2-B 90, DINOv2-B 45, BiomedCLIP-TTA)

# Per-encoder wall-clock budget (seconds). If encoding exceeds this we still
# finish the current batch but mark the run slow in the report.
# (SigLIP-SO400M at 384² needs ~11 min on MPS, so we allow 15.)
PER_ENCODER_SOFT_BUDGET_S = 15 * 60


# ---------------------------------------------------------------------------
# Tiling (shared across all encoders, identical to the 90-nm cache recipe)
# ---------------------------------------------------------------------------

def preprocess_to_tiles(
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
            ((pad_h // 2, pad_h - pad_h // 2),
             (pad_w // 2, pad_w - pad_w // 2)),
            mode="reflect",
        )
    tiles = tile(h, tile_size, stride=tile_size)
    if not tiles:
        return [h[:tile_size, :tile_size]]
    if len(tiles) > max_tiles:
        idx = np.linspace(0, len(tiles) - 1, max_tiles).astype(int)
        tiles = [tiles[i] for i in idx]
    return tiles


def build_pil_tiles(samples) -> tuple[list[Image.Image], np.ndarray, np.ndarray,
                                       np.ndarray, list[str]]:
    """Build the full tile set once; reused across encoders for fairness."""
    all_pil: list[Image.Image] = []
    tile_to_scan: list[int] = []
    scan_y: list[int] = []
    scan_groups: list[str] = []
    scan_paths: list[str] = []

    t0 = time.time()
    for si, s in enumerate(samples):
        try:
            tiles = preprocess_to_tiles(
                s.raw_path, target_nm_per_px=90.0, tile_size=512, max_tiles=9,
            )
            for t in tiles:
                all_pil.append(height_to_pil(t, mode="afmhot"))
                tile_to_scan.append(si)
            scan_y.append(s.label)
            scan_groups.append(person_id(s.raw_path))
            scan_paths.append(str(s.raw_path))
        except Exception as e:
            print(f"  [tile-err] {s.raw_path.name}: {e}")
        if (si + 1) % 60 == 0:
            print(f"  tiled [{si + 1}/{len(samples)}] pil={len(all_pil)} "
                  f"t={time.time() - t0:.1f}s")
    print(f"  total pil tiles={len(all_pil)} in {time.time() - t0:.1f}s")
    return (
        all_pil,
        np.asarray(tile_to_scan, dtype=np.int64),
        np.asarray(scan_y, dtype=np.int64),
        np.asarray(scan_groups),
        scan_paths,
    )


# ---------------------------------------------------------------------------
# Encoder registry — each entry returns (name, model, preprocess, dim, device)
# but lazily; we only call it when we need to encode.
# ---------------------------------------------------------------------------

@dataclass
class EncSpec:
    name: str                      # short name used in cache filename
    pretty: str                    # display name
    loader: Callable[[], tuple]    # returns (model, preprocess, dim, how)
    # how: "encode_image" for open_clip, "forward" for torchhub
    enabled: bool = True


def _mps():
    return pick_device()


def load_dinov2_l():
    dev = _mps()
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model = model.to(dev).eval()
    import torchvision.transforms as T
    prep = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, prep, 1024, "forward", dev


def load_dinov2_g():
    dev = _mps()
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
    model = model.to(dev).eval()
    import torchvision.transforms as T
    prep = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, prep, 1536, "forward", dev


def load_openclip(model_name: str, pretrained: str):
    import open_clip
    dev = _mps()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained,
    )
    model = model.to(dev).eval()
    cfg = open_clip.get_model_config(model_name)
    dim = cfg.get("embed_dim", 768)
    return model, preprocess, dim, "encode_image", dev


def load_siglip_so400m():
    return load_openclip("ViT-SO400M-14-SigLIP-384", "webli")


def load_eva02_l():
    return load_openclip("EVA02-L-14", "merged2b_s4b_b131k")


def load_openclip_vit_l():
    return load_openclip("ViT-L-14", "laion2b_s32b_b82k")


def load_pubmedclip():
    """PubMedCLIP via HuggingFace transformers (CLIPModel).

    We call `get_image_features(pixel_values=...)` which returns a projected
    embedding tensor (batch, embed_dim). Wrap in an `encode_image` method so the
    uniform encoding loop works.
    """
    from transformers import CLIPModel, CLIPProcessor
    dev = _mps()
    model_id = "flaviagiammarino/pubmed-clip-vit-base-patch32"
    base = CLIPModel.from_pretrained(model_id).to(dev).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    def preprocess_single(img):
        out = processor(images=img, return_tensors="pt")
        return out["pixel_values"][0]

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.clip = m  # registered as submodule

        @torch.no_grad()
        def encode_image(self, x):
            # get_image_features may return a Tensor OR a BaseModelOutputWithPooling
            # (depends on transformers version). The projected image embedding is
            # the Tensor return; when wrapped in ModelOutput, it lives in
            # `pooler_output` (512-d projection), NOT `last_hidden_state[:, 0]`
            # which is the un-projected 768-d vision-tower CLS.
            feats = self.clip.get_image_features(pixel_values=x)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "image_embeds") and feats.image_embeds is not None:
                feats = feats.image_embeds
            return feats

    wrap = _Wrap(base).to(dev).eval()
    return wrap, preprocess_single, 512, "encode_image", dev


ENCODERS: list[EncSpec] = [
    # Ordered small→big so cheap wins land first before anything risks OOM.
    EncSpec("pubmedclip_vitb32",   "PubMedCLIP ViT-B/32 (512-d)",      load_pubmedclip),
    EncSpec("openclip_vitl14_laion2b", "OpenCLIP ViT-L/14 LAION-2B (768-d)", load_openclip_vit_l),
    EncSpec("dinov2_vitl14",       "DINOv2-L (ViT-L/14, 1024-d)",      load_dinov2_l),
    EncSpec("eva02_l14",           "EVA02-L-14 (768-d)",               load_eva02_l),
    EncSpec("siglip_so400m_384",   "SigLIP-SO400M-14-384 (1152-d)",    load_siglip_so400m),
    # DINOv2-G (~4.5 GB weights) is parked last; disabled by default on 16 GB
    # macOS where MPS will OOM. Flip `enabled=True` to attempt it.
    EncSpec("dinov2_vitg14",       "DINOv2-G (ViT-g/14, 1536-d)",      load_dinov2_g,
            enabled=False),
]


# ---------------------------------------------------------------------------
# Encode + cache
# ---------------------------------------------------------------------------

def encode_tiles(
    pil_tiles: list[Image.Image],
    model,
    preprocess,
    how: str,
    device: str,
    batch_size: int = 16,
    hard_budget_s: float = PER_ENCODER_SOFT_BUDGET_S,
) -> Optional[np.ndarray]:
    embs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(pil_tiles), batch_size):
            batch = pil_tiles[i:i + batch_size]
            tensors = torch.stack([preprocess(im) for im in batch]).to(device)
            if how == "encode_image":
                e = model.encode_image(tensors)
            else:
                e = model(tensors)
            # Some HF models return ModelOutput objects
            if hasattr(e, "image_embeds") and e.image_embeds is not None:
                e = e.image_embeds
            elif hasattr(e, "pooler_output") and e.pooler_output is not None:
                e = e.pooler_output
            elif hasattr(e, "last_hidden_state"):
                e = e.last_hidden_state[:, 0]
            e = e.float().cpu().numpy()
            embs.append(e)
            del tensors
            if (i // batch_size) % 10 == 0:
                elapsed = time.time() - t0
                print(f"    enc batch {i // batch_size}  "
                      f"tiles={i + len(batch)}/{len(pil_tiles)}  "
                      f"t={elapsed:.1f}s")
            if time.time() - t0 > hard_budget_s:
                print(f"    [budget-exceeded] {time.time() - t0:.1f}s > "
                      f"{hard_budget_s}s — aborting encode")
                return None
    X = np.concatenate(embs, axis=0)
    return X


def ensure_cache(
    spec: EncSpec,
    pil_tiles: list[Image.Image],
    tile_to_scan: np.ndarray,
    scan_y: np.ndarray,
    scan_groups: np.ndarray,
    scan_paths: list[str],
) -> tuple[Optional[Path], dict]:
    cache_path = CACHE / f"tiled_emb_{spec.name}_afmhot_t512_n9.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        return cache_path, {
            "status": "cached",
            "encode_time_s": float(z["encode_time_s"]) if "encode_time_s" in z.files else None,
            "dim": int(z["X"].shape[1]),
        }

    print(f"[load-enc] {spec.pretty}")
    t_load = time.time()
    try:
        model, preprocess, dim, how, device = spec.loader()
    except Exception as e:
        print(f"  [load-fail] {spec.name}: {e}")
        traceback.print_exc()
        return None, {"status": "load_fail", "err": str(e)}
    print(f"  loaded dim={dim} device={device} t_load={time.time() - t_load:.1f}s")

    # Use smaller batch for big models to avoid MPS OOM
    bs = 16
    if spec.name in {"dinov2_vitg14", "siglip_so400m_384", "eva02_l14"}:
        bs = 8
    if spec.name == "dinov2_vitg14":
        bs = 4

    print(f"  encoding {len(pil_tiles)} tiles (bs={bs})")
    t_enc = time.time()
    try:
        X = encode_tiles(pil_tiles, model, preprocess, how, device, batch_size=bs)
    except Exception as e:
        print(f"  [encode-fail] {spec.name}: {e}")
        traceback.print_exc()
        X = None
    finally:
        # Free memory before next encoder
        try:
            del model
        except Exception:
            pass
        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    if X is None:
        return None, {"status": "encode_fail_or_budget"}

    enc_s = time.time() - t_enc
    print(f"  encoded {X.shape} in {enc_s:.1f}s")
    np.savez(
        cache_path,
        X=X.astype(np.float32),
        tile_to_scan=tile_to_scan,
        scan_y=scan_y,
        scan_groups=scan_groups,
        scan_paths=np.array(scan_paths),
        encode_time_s=np.array(enc_s, dtype=np.float64),
    )
    print(f"  [saved] {cache_path}")
    return cache_path, {"status": "encoded", "encode_time_s": enc_s, "dim": X.shape[1]}


# ---------------------------------------------------------------------------
# Evaluation (v2 recipe, person-LOPO)
# ---------------------------------------------------------------------------

def mean_pool_tiles(X_tiles: np.ndarray, t2s: np.ndarray, n_scans: int) -> np.ndarray:
    d = X_tiles.shape[1]
    out = np.zeros((n_scans, d), dtype=np.float32)
    for si in range(n_scans):
        m = t2s == si
        if m.any():
            out[si] = X_tiles[m].mean(axis=0)
    return out


def lopo_predict_v2(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    n = len(y)
    P = np.zeros((n, N_CLASSES), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups):
        Xt = normalize(X[tr], norm="l2", axis=1)
        Xv = normalize(X[va], norm="l2", axis=1)
        sc = StandardScaler()
        Xt = sc.fit_transform(Xt)
        Xv = sc.transform(Xv)
        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
        clf = LogisticRegression(
            class_weight="balanced", max_iter=3000, C=1.0,
            solver="lbfgs", n_jobs=4, random_state=42,
        )
        clf.fit(Xt, y[tr])
        proba = clf.predict_proba(Xv)
        p_full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
        for ci, cls in enumerate(clf.classes_):
            p_full[:, cls] = proba[:, ci]
        P[va] = p_full
    return P


def metrics_of(P: np.ndarray, y: np.ndarray) -> dict:
    pred = P.argmax(axis=1)
    return {
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(
            y, pred, average=None, labels=list(range(N_CLASSES)), zero_division=0,
        ).tolist(),
    }


def geom_mean_probs(probs_list: list[np.ndarray]) -> np.ndarray:
    log_sum = np.zeros_like(probs_list[0])
    for P in probs_list:
        log_sum = log_sum + np.log(P + EPS)
    G = np.exp(log_sum / len(probs_list))
    G /= G.sum(axis=1, keepdims=True)
    return G


def align_to_reference(paths_ref: list[str], paths_src: list[str],
                       X_src: np.ndarray) -> np.ndarray:
    src_idx = {p: i for i, p in enumerate(paths_src)}
    order = np.array([src_idx[p] for p in paths_ref])
    return X_src[order]


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_markdown(summary: dict) -> None:
    per = summary["per_encoder"]
    ens = summary["ensembles"]

    lines = []
    lines.append("# Foundation-Model Zoo — Results\n")
    lines.append(
        "**Goal:** compare underexplored frozen encoders against the current "
        "champions (DINOv2-B = 0.6162, BiomedCLIP-TTA = 0.6220, v4 ensemble = 0.6887) "
        "on person-LOPO weighted F1.\n"
    )
    lines.append("## Methodology\n")
    lines.append(
        "- **Data:** 240 AFM scans, 35 persons (LOPO via `teardrop.data.person_id`).\n"
        "- **Tiling:** 90 nm/px, `afmhot` render, up to 9 non-overlapping 512² tiles per scan, no TTA.\n"
        "- **Pooling:** mean over tiles → scan-level embedding.\n"
        "- **Recipe:** L2-normalize → StandardScaler → LogisticRegression(class_weight='balanced', C=1.0), same as v2/v4.\n"
        "- **Ensembles:** geometric mean of per-member softmax probabilities.\n"
        "- **Hardware:** MPS (Apple Silicon). Per-encoder budget ~8 min.\n"
    )

    lines.append("## Per-encoder person-LOPO metrics\n")
    lines.append("| Encoder | Status | Dim | Encode time (s) | Weighted F1 | Macro F1 | Δ vs DINOv2-B (0.6162) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in per:
        et = row.get("encode_time_s")
        et_str = f"{et:.1f}" if isinstance(et, (int, float)) else "-"
        dim_str = str(row.get("dim", "")) if row.get("dim") else "-"
        if row.get("weighted_f1") is not None:
            delta = row["weighted_f1"] - CHAMP_DINOV2B
            lines.append(
                f"| {row['pretty']} | {row['status']} | {dim_str} | "
                f"{et_str} | {row['weighted_f1']:.4f} | "
                f"{row['macro_f1']:.4f} | {delta:+.4f} |"
            )
        else:
            lines.append(
                f"| {row['pretty']} | {row['status']} | {dim_str} | "
                f"{et_str} | - | - | - |"
            )
    lines.append("")

    # Per-class for successful encoders
    succ = [r for r in per if r.get("per_class_f1")]
    if succ:
        lines.append("## Per-class F1 (successful encoders)\n")
        lines.append("| Encoder | " + " | ".join(CLASSES) + " |")
        lines.append("|---|" + "|".join([":---:"] * len(CLASSES)) + "|")
        for row in succ:
            pcf1 = row["per_class_f1"]
            lines.append(
                f"| {row['pretty']} | " + " | ".join(f"{v:.4f}" for v in pcf1) + " |"
            )
        lines.append("")

    lines.append("## Ensembles (geom-mean, person-LOPO)\n")
    lines.append("| Config | Members | Weighted F1 | Macro F1 | Δ vs v4 (0.6887) |")
    lines.append("|---|---|---:|---:|---:|")
    for row in ens:
        delta = row["weighted_f1"] - CHAMP_V4
        members = ", ".join(row["members"])
        lines.append(
            f"| **{row['name']}** | {members} | {row['weighted_f1']:.4f} | "
            f"{row['macro_f1']:.4f} | {delta:+.4f} |"
        )
    lines.append("")

    # Interpretation — are any single encoders above DINOv2-B?
    lines.append("## Verdict\n")
    beat_single = [r for r in per
                   if r.get("weighted_f1") and r["weighted_f1"] > CHAMP_DINOV2B + 1e-6]
    if beat_single:
        lines.append("### Single-encoder wins over DINOv2-B (0.6162)\n")
        for r in sorted(beat_single, key=lambda x: -x["weighted_f1"]):
            lines.append(
                f"- **{r['pretty']}**: {r['weighted_f1']:.4f} "
                f"(Δ = {r['weighted_f1'] - CHAMP_DINOV2B:+.4f})"
            )
        lines.append("")
    else:
        lines.append(
            "- No single encoder in this zoo beats DINOv2-B on person-LOPO under the "
            "identical (no-TTA, 9-tile, L2+Scaler+LR) recipe.\n"
        )

    beat_v4 = [r for r in ens if r["weighted_f1"] > CHAMP_V4 + 1e-6]
    if beat_v4:
        lines.append("### Ensembles that beat v4 (0.6887)\n")
        for r in sorted(beat_v4, key=lambda x: -x["weighted_f1"]):
            lines.append(
                f"- **{r['name']}** ({', '.join(r['members'])}): "
                f"W-F1 = {r['weighted_f1']:.4f} (Δ = {r['weighted_f1'] - CHAMP_V4:+.4f}), "
                f"M-F1 = {r['macro_f1']:.4f}"
            )
        lines.append("")
    else:
        lines.append(
            "- No 4-way or 5-way ensemble in this sweep beats the v4 multi-scale champion "
            "(0.6887) under person-LOPO. The v4 recipe remains the champion.\n"
        )

    lines.append("## Honest reporting\n")
    lines.append(
        "- Person-level LOPO (35 folds), V2 recipe only, no threshold tuning, no OOF model selection.\n"
        "- The DINOv2-B 0.6162 baseline uses the IDENTICAL pipeline as every zoo encoder "
        "(no TTA, 9 tiles, afmhot). That is why it sits below the v4 number (0.6887); v4 uses "
        "BiomedCLIP with D4-TTA plus a second DINOv2-B branch at 45 nm/px.\n"
        "- `Δ vs v4` for the ensembles reflects drop-in replacements/additions using "
        "no-TTA zoo encoders and the v4 TTA BiomedCLIP / 45 nm branch where kept.\n"
    )

    (REPORTS / "FOUNDATION_ZOO_RESULTS.md").write_text("\n".join(lines))
    print(f"[saved] reports/FOUNDATION_ZOO_RESULTS.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print("Foundation-Model Zoo — frozen-encoder sweep on tear AFM")
    print("=" * 78)

    # Reference tile set (only built if at least one encoder needs encoding)
    pil_tiles = None
    t2s = scan_y = scan_groups = None
    scan_paths: list[str] = []

    def _ensure_tiles():
        nonlocal pil_tiles, t2s, scan_y, scan_groups, scan_paths
        if pil_tiles is not None:
            return
        print("\n[prep] building PIL tile set (shared across encoders)")
        samples = enumerate_samples(ROOT / "TRAIN_SET")
        print(f"  samples={len(samples)}")
        pil_tiles, t2s, scan_y, scan_groups, scan_paths = build_pil_tiles(samples)

    # ---- 1. Per-encoder encode + evaluate ----
    per_encoder = []
    caches: dict[str, Path] = {}
    for spec in ENCODERS:
        print("\n" + "-" * 78)
        print(f"[zoo] {spec.pretty}")

        if not spec.enabled:
            print(f"  [disabled] skipping (see ENCODERS registry)")
            per_encoder.append({
                "name": spec.name, "pretty": spec.pretty,
                "status": "disabled",
                "encode_time_s": None, "weighted_f1": None,
                "macro_f1": None, "per_class_f1": None, "dim": None,
            })
            continue

        cache_path = CACHE / f"tiled_emb_{spec.name}_afmhot_t512_n9.npz"
        if not cache_path.exists():
            _ensure_tiles()
            path, info = ensure_cache(
                spec, pil_tiles, t2s, scan_y, scan_groups, scan_paths,
            )
        else:
            z = np.load(cache_path, allow_pickle=True)
            info = {
                "status": "cached",
                "encode_time_s": float(z["encode_time_s"])
                                 if "encode_time_s" in z.files else None,
                "dim": int(z["X"].shape[1]),
            }
            path = cache_path
            print(f"  [cache-hit] {cache_path.name}")

        if path is None:
            per_encoder.append({
                "name": spec.name, "pretty": spec.pretty,
                "status": info["status"],
                "err": info.get("err"),
                "encode_time_s": None, "weighted_f1": None,
                "macro_f1": None, "per_class_f1": None, "dim": None,
            })
            continue

        # Evaluate
        z = np.load(path, allow_pickle=True)
        X = z["X"]
        t2s_i = z["tile_to_scan"]
        y_i = z["scan_y"]
        groups_i = z["scan_groups"]
        paths_i = [str(p) for p in z["scan_paths"]]

        # Ensure groups are person-level: if cache was built before person_id fix
        # (e.g. the existing 90nm cache uses patient_id), recompute groups.
        person_groups = np.array([person_id(Path(p)) for p in paths_i])
        if not np.array_equal(person_groups, groups_i):
            print(f"  [note] overriding stored groups with person_id (n_persons="
                  f"{len(set(person_groups))} vs stored {len(set(groups_i))})")
            groups_i = person_groups

        n_scans = len(y_i)
        X_scan = mean_pool_tiles(X, t2s_i, n_scans)
        te = time.time()
        P = lopo_predict_v2(X_scan, y_i, groups_i)
        m = metrics_of(P, y_i)
        print(f"  person-LOPO W-F1={m['weighted_f1']:.4f}  M-F1={m['macro_f1']:.4f}  "
              f"(fit+eval={time.time() - te:.1f}s)")

        per_encoder.append({
            "name": spec.name, "pretty": spec.pretty,
            "status": info["status"],
            "encode_time_s": info.get("encode_time_s"),
            "dim": info.get("dim"),
            "weighted_f1": m["weighted_f1"],
            "macro_f1": m["macro_f1"],
            "per_class_f1": m["per_class_f1"],
        })
        caches[spec.name] = path

    # ---- 2. Load v4 components for ensemble experiments ----
    print("\n" + "-" * 78)
    print("[ensembles] loading v4 components")
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz", allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz",
                  allow_pickle=True)
    zbc_tta = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)

    paths_90 = [str(p) for p in z90["scan_paths"]]
    paths_45 = [str(p) for p in z45["scan_paths"]]
    paths_bc = [str(p) for p in zbc_tta["scan_paths"]]

    groups_ref = np.array([person_id(Path(p)) for p in paths_90])
    y_ref = np.asarray(z90["scan_y"], dtype=np.int64)
    n_persons = len(np.unique(groups_ref))
    print(f"  reference={paths_90[0]!r}... n_scans={len(y_ref)} n_persons={n_persons}")

    X90 = mean_pool_tiles(z90["X"], z90["tile_to_scan"], len(paths_90))
    X45 = mean_pool_tiles(z45["X"], z45["tile_to_scan"], len(paths_45))
    X45 = align_to_reference(paths_90, paths_45, X45)
    Xbc_tta = align_to_reference(paths_90, paths_bc, zbc_tta["X_scan"].astype(np.float32))

    # v4 base softmaxes
    print("  computing v4-component person-LOPO softmax (DINOv2-B 90, 45, BiomedCLIP-TTA)")
    P_d90 = lopo_predict_v2(X90, y_ref, groups_ref)
    P_d45 = lopo_predict_v2(X45, y_ref, groups_ref)
    P_bc = lopo_predict_v2(Xbc_tta, y_ref, groups_ref)
    m_v4 = metrics_of(geom_mean_probs([P_d90, P_d45, P_bc]), y_ref)
    print(f"  v4 reference reproduced: W-F1={m_v4['weighted_f1']:.4f}  "
          f"M-F1={m_v4['macro_f1']:.4f}")

    # Softmaxes for each successful zoo encoder aligned to reference
    zoo_probs: dict[str, np.ndarray] = {}
    for row in per_encoder:
        if row["weighted_f1"] is None:
            continue
        cache_path = caches[row["name"]]
        z = np.load(cache_path, allow_pickle=True)
        paths_k = [str(p) for p in z["scan_paths"]]
        Xk = mean_pool_tiles(z["X"], z["tile_to_scan"], len(paths_k))
        Xk = align_to_reference(paths_90, paths_k, Xk)
        P = lopo_predict_v2(Xk, y_ref, groups_ref)
        zoo_probs[row["name"]] = P

    # ---- 3. Ensemble configurations ----
    ensembles = []

    def _add_ens(name: str, members: list[str], probs_list: list[np.ndarray]):
        m = metrics_of(geom_mean_probs(probs_list), y_ref)
        ensembles.append({
            "name": name, "members": members,
            "weighted_f1": m["weighted_f1"], "macro_f1": m["macro_f1"],
            "per_class_f1": m["per_class_f1"],
        })
        print(f"  {name:48s} W-F1={m['weighted_f1']:.4f}  M-F1={m['macro_f1']:.4f}")

    print("\n  geom-mean ensembles vs v4 (0.6887):")
    _add_ens("v4_baseline",
             ["dinov2b_90", "dinov2b_45", "biomedclip_tta"],
             [P_d90, P_d45, P_bc])

    if "dinov2_vitl14" in zoo_probs:
        _add_ens("v4 + DINOv2-L (4-way)",
                 ["dinov2b_90", "dinov2b_45", "biomedclip_tta", "dinov2_vitl14"],
                 [P_d90, P_d45, P_bc, zoo_probs["dinov2_vitl14"]])

    if "siglip_so400m_384" in zoo_probs:
        _add_ens("v4 swap BiomedCLIP-TTA for SigLIP (3-way)",
                 ["dinov2b_90", "dinov2b_45", "siglip_so400m_384"],
                 [P_d90, P_d45, zoo_probs["siglip_so400m_384"]])

    if "dinov2_vitl14" in zoo_probs and "siglip_so400m_384" in zoo_probs:
        _add_ens("v4 + DINOv2-L + SigLIP (5-way)",
                 ["dinov2b_90", "dinov2b_45", "biomedclip_tta",
                  "dinov2_vitl14", "siglip_so400m_384"],
                 [P_d90, P_d45, P_bc,
                  zoo_probs["dinov2_vitl14"], zoo_probs["siglip_so400m_384"]])

    # Bonus: v4 + EVA02 and v4 + OpenCLIP-L where available
    if "eva02_l14" in zoo_probs:
        _add_ens("v4 + EVA02-L (4-way)",
                 ["dinov2b_90", "dinov2b_45", "biomedclip_tta", "eva02_l14"],
                 [P_d90, P_d45, P_bc, zoo_probs["eva02_l14"]])
    if "openclip_vitl14_laion2b" in zoo_probs:
        _add_ens("v4 + OpenCLIP-L LAION-2B (4-way)",
                 ["dinov2b_90", "dinov2b_45", "biomedclip_tta", "openclip_vitl14_laion2b"],
                 [P_d90, P_d45, P_bc, zoo_probs["openclip_vitl14_laion2b"]])

    # ---- 4. Persist ----
    summary = {
        "person_lopo": True,
        "n_persons": int(n_persons),
        "n_scans": int(len(y_ref)),
        "champions": {"dinov2b_no_tta": CHAMP_DINOV2B, "v4_ensemble": CHAMP_V4},
        "per_encoder": per_encoder,
        "ensembles": ensembles,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (REPORTS / "foundation_zoo_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] reports/foundation_zoo_results.json")
    write_markdown(summary)
    print(f"\n[done] elapsed {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
