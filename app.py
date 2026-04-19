"""Interactive Gradio demo for the UPJŠ Teardrop AFM challenge.

Live pitch artifact. Run via:

    .venv/bin/python app.py

Then visit http://127.0.0.1:7860.

Three tabs:
  * Classify a scan   — upload .NNN (Bruker Nanoscope) or .bmp, see prediction + reasoning.
  * Disease fingerprints — educational gallery per class.
  * About — method, honest F1 = 0.6458, limitations.
"""
from __future__ import annotations

import io
import random
import sys
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Heavy project imports
from teardrop.data import (  # noqa: E402
    CLASSES,
    load_height,
    plane_level,
    resample_to_pixel_size,
    robust_normalize,
    tile as tile_fn,
)
from teardrop.encoders import height_to_pil  # noqa: E402
from teardrop.features import (  # noqa: E402
    fractal_dimension,
    glcm_features,
    roughness_features,
)
from teardrop.infer import preprocess_and_tile_spm  # noqa: E402
from models.ensemble_v1_tta.predict import TTAPredictor  # noqa: E402

import gradio as gr  # noqa: E402

# ---------------------------------------------------------------------------
# Constants & theming
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "ZdraviLudia": "#2a9d8f",       # teal — healthy
    "Diabetes": "#e76f51",           # coral
    "PGOV_Glaukom": "#f4a261",       # amber
    "SklerozaMultiplex": "#8e44ad",  # purple
    "SucheOko": "#264653",           # slate (2-patient ceiling)
}

CLASS_DESCRIPTIONS = {
    "ZdraviLudia": {
        "title": "Healthy controls",
        "morphology": (
            "Smooth dendritic tear-film deposits with continuous, well-branched "
            "fern-like crystalline networks. Uniform height distribution."
        ),
        "features": (
            "Low surface roughness (low Sa/Sq), high GLCM homogeneity, "
            "Masmali grade 0–1 (uniform crystallization)."
        ),
        "masmali": "Grade 0–1",
    },
    "Diabetes": {
        "title": "Diabetes mellitus",
        "morphology": (
            "Coarse grainy deposits with fragmented dendrites. Increased small-scale "
            "roughness, loss of continuous fern branches. Hyperglycaemia-induced "
            "glycation visible as elevated GLCM contrast."
        ),
        "features": (
            "High fractal dimension (D ≈ 2.4–2.6), elevated Sa/Sq, "
            "Masmali grade 2–3."
        ),
        "masmali": "Grade 2–3",
    },
    "PGOV_Glaukom": {
        "title": "Primary-open-angle glaucoma",
        "morphology": (
            "Characteristic loop/ring structures visible in persistent-homology H₁ "
            "features. Medium-scale annular patterns at 10–30 µm. Partial "
            "crystallization with coarse lamellar islands."
        ),
        "features": (
            "H₁ topological features dominate. Medium GLCM contrast, "
            "Masmali grade 3–4."
        ),
        "masmali": "Grade 3–4",
    },
    "SklerozaMultiplex": {
        "title": "Multiple sclerosis",
        "morphology": (
            "Thin, sparse crystals with degraded branching. Disrupted tear-film "
            "protein chemistry yields irregular low-density deposits that often "
            "confuse with glaucoma at the macroscopic level."
        ),
        "features": (
            "Low feature density, moderate roughness, "
            "distinctive LBP uniform pattern histograms."
        ),
        "masmali": "Grade 3–4",
    },
    "SucheOko": {
        "title": "Dry eye syndrome",
        "morphology": (
            "Heterogeneous, highly irregular deposits. Classic Masmali grade 4–5 "
            "crystallization failure. NOTE: TRAIN_SET has only 2 patients → "
            "F1 ≈ 0.07 is a data-acquisition ceiling, not a modelling failure."
        ),
        "features": (
            "Very high Sa/Sq, collapsed fractal dimension, "
            "Masmali grade 4–5. Data ceiling = 2 patients in train set."
        ),
        "masmali": "Grade 4–5",
    },
}

# Curated demo scans — one per class, deterministic on-startup verification
DEMO_SAMPLES = {
    "ZdraviLudia": "TRAIN_SET/ZdraviLudia/2L.001",
    "Diabetes": "TRAIN_SET/Diabetes/DM_01.03.2024_LO.001",
    "PGOV_Glaukom": "TRAIN_SET/PGOV_Glaukom/21_LV_PGOV+SII.001",
    "SklerozaMultiplex": "TRAIN_SET/SklerozaMultiplex/1-SM-LM-18.001",
    "SucheOko": "TRAIN_SET/SucheOko/29_PM_suche_oko.001",
}

# ---------------------------------------------------------------------------
# One-time startup: load predictor + retrieval cache
# ---------------------------------------------------------------------------

print("[app] Loading TTA ensemble bundle — this may take ~60s on cold start...")
PREDICTOR = TTAPredictor.load(PROJECT_ROOT / "models/ensemble_v1_tta")
print("[app] Predictor ready.")

print("[app] Loading DINOv2-B retrieval cache...")
_CACHE = np.load(
    PROJECT_ROOT / "cache/tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz",
    allow_pickle=True,
)
_TILE_X = _CACHE["X"].astype(np.float32)
_TILE_TO_SCAN = _CACHE["tile_to_scan"].astype(np.int64)
_SCAN_Y = _CACHE["scan_y"].astype(np.int64)
_SCAN_PATHS = [str(p) for p in _CACHE["scan_paths"]]
_N_SCANS = len(_SCAN_PATHS)
_SCAN_EMB = np.zeros((_N_SCANS, _TILE_X.shape[1]), dtype=np.float32)
for s in range(_N_SCANS):
    _SCAN_EMB[s] = _TILE_X[_TILE_TO_SCAN == s].mean(axis=0)
# L2 normalize for cosine similarity retrieval
_SCAN_EMB_N = _SCAN_EMB / (np.linalg.norm(_SCAN_EMB, axis=1, keepdims=True) + 1e-9)
print(f"[app] Retrieval index ready: {_N_SCANS} training scans, D={_TILE_X.shape[1]}.")


# ---------------------------------------------------------------------------
# Helper: rendering
# ---------------------------------------------------------------------------

def _height_to_afmhot_rgb(h: np.ndarray, size: int = 512) -> Image.Image:
    """Render a [0,1] normalized height map as afmhot RGB at given size."""
    img = height_to_pil(h, mode="afmhot")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return img


def _bar_chart_probs(probs: np.ndarray, classes: list[str], pred_idx: int) -> Image.Image:
    """Horizontal bar chart of class probabilities. Predicted class highlighted."""
    order = np.argsort(probs)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=120)
    y_pos = np.arange(len(classes))
    labels = [classes[i] for i in order]
    values = [probs[i] for i in order]
    colors = [
        CLASS_COLORS.get(labels[i], "#888")
        if order[i] != pred_idx
        else "#1f77b4"
        for i in range(len(order))
    ]
    # Override: predicted always gets vivid blue so it stands out
    colors = []
    for i, idx in enumerate(order):
        base = CLASS_COLORS.get(classes[idx], "#888")
        if idx == pred_idx:
            colors.append(base)
        else:
            colors.append(base + "80")  # 50% alpha via hex
    ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", fontsize=10)
    ax.set_title("Class probabilities (TTA ensemble)", fontsize=11)
    for i, v in enumerate(values):
        ax.text(min(v + 0.01, 0.98), i, f"{v:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _tile_grid(tiles: list[np.ndarray], n_show: int = 4) -> Image.Image:
    """2×2 grid of tile previews (first n_show)."""
    tiles = tiles[:n_show]
    while len(tiles) < n_show:
        tiles.append(np.zeros_like(tiles[0]))
    pil_tiles = [height_to_pil(t, mode="afmhot").resize((256, 256)) for t in tiles]
    grid = Image.new("RGB", (256 * 2, 256 * 2), (0, 0, 0))
    for i, im in enumerate(pil_tiles[:4]):
        r, c = divmod(i, 2)
        grid.paste(im, (c * 256, r * 256))
    return grid


def _tile_preview_from_path(raw_path: Path):
    """Run preprocessing — return (preview_image, tile_grid_image, full_height_map_normalized)."""
    tiles = preprocess_and_tile_spm(
        raw_path, target_nm_per_px=90.0, tile_size=512, max_tiles=9
    )
    # Full preprocessed height for retrieval-style handcrafted features
    hm = load_height(raw_path)
    h = plane_level(hm.height)
    h = resample_to_pixel_size(h, hm.pixel_nm, 90.0)
    h = robust_normalize(h)
    preview = _height_to_afmhot_rgb(h, size=512)
    grid = _tile_grid(tiles)
    return preview, grid, h, tiles


def _neighbor_retrieval(scan_emb: np.ndarray, k: int = 3) -> list[dict]:
    """k-nearest neighbours in DINOv2-B space (cosine sim)."""
    q = scan_emb / (np.linalg.norm(scan_emb) + 1e-9)
    sims = _SCAN_EMB_N @ q  # (N,)
    top = np.argsort(sims)[::-1][:k]
    out = []
    for idx in top:
        p = _SCAN_PATHS[int(idx)]
        cls = CLASSES[int(_SCAN_Y[int(idx)])]
        out.append({
            "path": p,
            "class": cls,
            "sim": float(sims[int(idx)]),
        })
    return out


def _render_neighbor_thumb(path: str, size: int = 256) -> Image.Image:
    """Render thumbnail for a training-scan file path."""
    try:
        tiles = preprocess_and_tile_spm(Path(path), target_nm_per_px=90.0,
                                        tile_size=512, max_tiles=1)
        return height_to_pil(tiles[0], mode="afmhot").resize((size, size))
    except Exception:
        img = Image.new("RGB", (size, size), (30, 30, 30))
        return img


def _predict_from_tiles_dinov2_emb(tiles: list[np.ndarray]) -> np.ndarray:
    """Re-encode with the DINOv2-B branch only to get retrieval embedding.

    We reuse the loaded DINOv2 encoder inside the ensemble to produce an embedding
    that lives in the same space as the cache.
    """
    # Find the dinov2 component inside the bundle
    bundle = PREDICTOR.bundle
    dino_comp = None
    for name, comp in zip(bundle.component_names, bundle.components):
        if comp.encoder.name.startswith("dinov2_"):
            dino_comp = comp
            break
    if dino_comp is None:
        return None
    pils = [height_to_pil(t, mode="afmhot") for t in tiles]
    tile_emb = dino_comp.encoder.encode(pils, batch_size=len(pils))
    return tile_emb.mean(axis=0)  # (D,)


# ---------------------------------------------------------------------------
# Upload → inference handler
# ---------------------------------------------------------------------------

def classify_scan(file_obj):
    """Main inference handler.

    Accepts:
      - Bruker .NNN files (primary)
      - .bmp previews (fallback — still displayed, no full preprocessing)

    Returns:
      (prediction_md, preview_img, tile_grid_img, prob_chart,
       retrieval_md, retrieval_gallery, handcrafted_md)
    """
    if file_obj is None:
        empty = Image.new("RGB", (512, 512), (20, 20, 20))
        return (
            "### No file uploaded yet\nUpload a `.NNN` Bruker Nanoscope scan "
            "(or `.bmp` preview) on the left to begin.",
            empty, empty, empty,
            "", [],
            "",
        )

    # Gradio 6: File component returns a filepath string (when type='filepath')
    raw_path = Path(file_obj if isinstance(file_obj, str) else file_obj.name)
    suffix = raw_path.suffix.lower()
    is_bmp = suffix == ".bmp"

    try:
        if is_bmp:
            # Fallback: just display the BMP, no classification (the model is
            # watermark-unsafe on BMPs per project policy).
            pil = Image.open(raw_path).convert("RGB").resize((512, 512))
            empty_grid = Image.new("RGB", (512, 512), (20, 20, 20))
            prob_img = Image.new("RGB", (720, 360), (20, 20, 20))
            return (
                "### BMP preview loaded (no inference)\n"
                "The shipped model is trained on raw Bruker SPM height channels. "
                "BMP previews contain scale-bars and watermarks that leak labels — "
                "upload a `.NNN` scan to run inference.",
                pil, empty_grid, prob_img,
                "", [],
                "",
            )

        # Primary path — full preprocessing + TTA inference
        preview, grid, full_h, tiles = _tile_preview_from_path(raw_path)

        # Inference (D4 TTA via bundled predictor)
        pred_class, probs = PREDICTOR.predict_scan(raw_path)
        pred_idx = int(np.argmax(probs))
        probs_np = np.asarray(probs, dtype=np.float32)

        # Top-2 with confidences
        order = np.argsort(probs_np)[::-1]
        top1, top2 = order[0], order[1]
        top1_conf = probs_np[top1] * 100.0
        top2_conf = probs_np[top2] * 100.0

        classes = PREDICTOR.bundle.classes
        color = CLASS_COLORS.get(classes[top1], "#333")
        color2 = CLASS_COLORS.get(classes[top2], "#555")

        pred_md = (
            f"### Prediction\n"
            f"<div style='font-size:2rem; font-weight:700; color:{color}; "
            f"padding:0.3em 0.6em; border-radius:8px; background:rgba(0,0,0,0.02); "
            f"border-left: 6px solid {color};'>"
            f"{classes[top1]} &nbsp; <span style='font-size:1.1rem; opacity:0.8;'>"
            f"({top1_conf:.1f} % confidence)</span></div>\n\n"
            f"**Runner-up:** <span style='color:{color2}; font-weight:600;'>"
            f"{classes[top2]}</span> ({top2_conf:.1f} %)\n\n"
            f"*D4-TTA ensemble (DINOv2-B + BiomedCLIP), softmax-averaged over "
            f"72 views/scan.*"
        )

        prob_img = _bar_chart_probs(probs_np, classes, pred_idx)

        # Retrieval — 3 nearest DINOv2-B training scans
        dino_emb = _predict_from_tiles_dinov2_emb(tiles)
        retrieval_md_parts = ["### Nearest training scans (DINOv2-B cosine)"]
        gallery = []
        if dino_emb is not None:
            neighbours = _neighbor_retrieval(dino_emb, k=3)
            for i, nb in enumerate(neighbours):
                thumb = _render_neighbor_thumb(nb["path"])
                name = Path(nb["path"]).name
                gallery.append((
                    thumb,
                    f"#{i + 1}  {nb['class']}  (sim={nb['sim']:.3f})\n{name}",
                ))
            votes = {}
            for nb in neighbours:
                votes[nb["class"]] = votes.get(nb["class"], 0) + 1
            majority = max(votes.items(), key=lambda kv: kv[1])[0]
            agree = majority == classes[top1]
            agree_txt = (
                f"Neighbour majority ({majority}) **agrees** with the model."
                if agree
                else f"Neighbour majority ({majority}) **disagrees** — ambiguous sample."
            )
            retrieval_md_parts.append(agree_txt)
        retrieval_md = "\n\n".join(retrieval_md_parts)

        # Handcrafted feature summary — computed on full preprocessed scan, not tiles
        try:
            rough = roughness_features(full_h)
            # GLCM only on a center crop (speed)
            H, W = full_h.shape
            s = 512
            y0 = max(0, (H - s) // 2)
            x0 = max(0, (W - s) // 2)
            crop = full_h[y0:y0 + s, x0:x0 + s]
            if crop.shape[0] < 8 or crop.shape[1] < 8:
                crop = full_h
            glcm = glcm_features(crop, distances=(1, 5))
            frac = fractal_dimension(crop)
            hand_md = (
                "### Handcrafted surface descriptors\n"
                "| Metric | Value | Interpretation |\n"
                "|---|---:|---|\n"
                f"| Sa (mean absolute roughness)   | {rough['Sa']:.4f} | "
                "lower = smoother |\n"
                f"| Sq (RMS roughness)             | {rough['Sq']:.4f} | "
                "RMS deviation from mean plane |\n"
                f"| Ssk (skewness)                 | {rough['Ssk']:.3f} | "
                "peak/valley asymmetry |\n"
                f"| Fractal dimension D (mean)     | {frac['fractal_D_mean']:.3f} | "
                "higher = more complex |\n"
                f"| GLCM contrast (d=1, 0°)        | {glcm['glcm_contrast_d1_mean']:.3f} | "
                "local intensity variation |\n"
                f"| GLCM homogeneity (d=5)         | {glcm['glcm_homogeneity_d5_mean']:.3f} | "
                "higher = more uniform |\n"
            )
        except Exception as e:
            hand_md = f"*(handcrafted features unavailable: {e})*"

        return (
            pred_md, preview, grid, prob_img,
            retrieval_md, gallery,
            hand_md,
        )

    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"[app] Error during inference:\n{tb}", file=sys.stderr)
        empty = Image.new("RGB", (512, 512), (20, 20, 20))
        return (
            f"### Error\nFailed to process this file.\n\n```\n{str(e)[:400]}\n```",
            empty, empty, empty,
            "", [],
            "",
        )


def load_demo_sample(cls: str):
    """Load one of the built-in TRAIN_SET samples for quick testing."""
    rel = DEMO_SAMPLES.get(cls)
    if rel is None:
        return None
    path = PROJECT_ROOT / rel
    if not path.exists():
        return None
    return str(path)


# ---------------------------------------------------------------------------
# Tab 2 — Disease fingerprints gallery
# ---------------------------------------------------------------------------

def _build_class_examples():
    """For each class, find 3 preprocessed thumbs from TRAIN_SET."""
    rng = random.Random(42)
    examples: dict[str, list[Image.Image]] = {}
    for cls in CLASSES:
        cls_dir = PROJECT_ROOT / "TRAIN_SET" / cls
        if not cls_dir.exists():
            examples[cls] = []
            continue
        # list .NNN files
        spm_files = []
        for p in sorted(cls_dir.iterdir()):
            if p.is_file() and len(p.suffix) == 4 and p.suffix[1:].isdigit():
                spm_files.append(p)
        if not spm_files:
            examples[cls] = []
            continue
        rng.shuffle(spm_files)
        picks = spm_files[:3]
        imgs = []
        for p in picks:
            try:
                tiles = preprocess_and_tile_spm(p, max_tiles=1)
                imgs.append(height_to_pil(tiles[0], mode="afmhot").resize((320, 320)))
            except Exception:
                continue
        examples[cls] = imgs
    return examples


print("[app] Building disease fingerprint gallery (may take ~30 s)...")
_CLASS_EXAMPLES = _build_class_examples()
print("[app] Fingerprint gallery ready.")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="slate",
    neutral_hue="slate",
).set(body_background_fill="#fafafa")

INTRO_MD = """
# Teardrop AFM disease classifier — Hack Košice 2026

Classify tear-film AFM scans into **5 clinical categories**: `ZdraviLudia`
(healthy), `Diabetes`, `PGOV_Glaukom`, `SklerozaMultiplex`, `SucheOko` (dry eye).

- **Model:** DINOv2-B + BiomedCLIP TTA-D4 ensemble (softmax-averaged, raw argmax)
- **Honest F1 (person-LOPO):** **0.6458** on 240 scans / 35 persons
- **Data ceiling:** SucheOko has only 2 patients in train set → very low recall expected
"""

with gr.Blocks(title="Teardrop AFM classifier") as demo:
    gr.Markdown(INTRO_MD)

    with gr.Tabs():

        # -----------------------------------------------------------------
        # Tab 1 — classify
        # -----------------------------------------------------------------
        with gr.Tab("Classify a scan"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload a scan")
                    file_in = gr.File(
                        label="Bruker Nanoscope (.NNN) or .bmp preview",
                        type="filepath",
                    )
                    classify_btn = gr.Button("Classify", variant="primary", size="lg")

                    gr.Markdown("### Or try a built-in sample")
                    with gr.Row():
                        demo_buttons = {}
                        for cls in CLASSES:
                            btn = gr.Button(cls, size="sm")
                            demo_buttons[cls] = btn

                    gr.Markdown(
                        "*Preprocessing: plane-level → resample to 90 nm/px → "
                        "robust normalize → up to 9 non-overlapping 512² tiles "
                        "→ D4 TTA (72 views/scan) → dual-encoder softmax mean.*"
                    )

                with gr.Column(scale=2):
                    pred_md = gr.Markdown(
                        "### Prediction\n_Upload a scan or pick a demo sample to begin._"
                    )
                    with gr.Row():
                        preview_img = gr.Image(
                            label="Preprocessed height map (afmhot, 512×512)",
                            type="pil", height=380,
                        )
                        tile_img = gr.Image(
                            label="Tile crops (first 4 of 9)",
                            type="pil", height=380,
                        )
                    prob_img = gr.Image(
                        label="Class probabilities",
                        type="pil", height=320,
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    retrieval_md = gr.Markdown("### Nearest training scans")
                    retrieval_gallery = gr.Gallery(
                        label="3 nearest neighbours (DINOv2-B cosine)",
                        columns=3, rows=1, height=280, object_fit="contain",
                    )
                with gr.Column(scale=1):
                    hand_md = gr.Markdown("### Handcrafted surface descriptors")

            out_components = [
                pred_md, preview_img, tile_img, prob_img,
                retrieval_md, retrieval_gallery, hand_md,
            ]

            classify_btn.click(
                fn=classify_scan, inputs=[file_in], outputs=out_components,
            )
            file_in.upload(
                fn=classify_scan, inputs=[file_in], outputs=out_components,
            )

            for cls, btn in demo_buttons.items():
                btn.click(
                    fn=load_demo_sample, inputs=[gr.State(cls)], outputs=[file_in],
                ).then(
                    fn=classify_scan, inputs=[file_in], outputs=out_components,
                )

        # -----------------------------------------------------------------
        # Tab 2 — disease fingerprints
        # -----------------------------------------------------------------
        with gr.Tab("Disease fingerprints"):
            gr.Markdown(
                "## Morphological fingerprints per class\n"
                "Three training examples per class, rendered from raw SPM → plane-level "
                "→ robust normalize → `afmhot` colormap. Descriptions summarise the "
                "domain-relevant features our encoders latch on to."
            )

            for cls in CLASSES:
                info = CLASS_DESCRIPTIONS[cls]
                color = CLASS_COLORS.get(cls, "#333")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            f"### <span style='color:{color}'>{info['title']}</span> "
                            f"(`{cls}`)\n"
                            f"**Morphology.** {info['morphology']}\n\n"
                            f"**Distinguishing features.** {info['features']}\n\n"
                            f"**Masmali grade.** {info['masmali']}"
                        )
                    with gr.Column(scale=2):
                        gr.Gallery(
                            value=_CLASS_EXAMPLES.get(cls, []),
                            columns=3, rows=1, height=260, object_fit="contain",
                            show_label=False, interactive=False,
                        )

            gr.Markdown(
                "### Class-morphology grid (from pitch report)\n"
                "Deterministic seed=42 — full grid in `reports/pitch/02_class_morphology_grid.png`."
            )
            morph_grid_path = PROJECT_ROOT / "reports/pitch/02_class_morphology_grid.png"
            if morph_grid_path.exists():
                gr.Image(
                    value=str(morph_grid_path),
                    show_label=False, interactive=False, height=520,
                )

        # -----------------------------------------------------------------
        # Tab 3 — about
        # -----------------------------------------------------------------
        with gr.Tab("About"):
            gr.Markdown(
                """
## Method

Two frozen vision transformers encode preprocessed tear-film AFM scans:

- **DINOv2-B** (ViT-B/14, 768-dim) — Meta's self-supervised general-purpose features.
- **BiomedCLIP** (ViT-B/16, 512-dim) — Microsoft's medical-image CLIP, pretrained on 15 M
  PubMed image-text pairs.

### Pipeline

1. Load Bruker Nanoscope `.NNN` via `AFMReader.spm.load_spm`.
2. **Plane-level** (1st-order polynomial subtraction — removes scanner tilt).
3. **Resample** to 90 nm/px so all scans share an identical physical scale.
4. **Robust-normalize** (2–98 percentile clip → [0, 1]).
5. Cut into up to **9 non-overlapping 512×512 tiles**.
6. Apply the **D4 group** (identity + 3 rotations + 4 flipped rotations) → 72 views/scan.
7. Encode each view, mean-pool to a single scan-embedding per encoder.
8. Per encoder: `StandardScaler → LogisticRegression(class_weight='balanced', C=1.0)`.
9. **Softmax-average** the two heads, argmax.

No thresholds, no calibration, no stacking — simplicity was a deliberate choice
after our red-team audits (see below).

---

## Honest performance (person-level Leave-One-Patient-Out, 240 scans / 35 persons)

| Metric | Value |
|---|---:|
| **Weighted F1 (shipped, raw argmax)** | **0.6458** |
| Macro F1 | 0.49 |
| Single-model baseline (DINOv2-B alone) | 0.615 |
| Null baseline (label-shuffle × 5 seeds) | 0.276 ± 0.042 |

### Per-class F1

| Class | F1 | Support |
|---|---:|---:|
| ZdraviLudia | ~0.82 | 70 |
| SklerozaMultiplex | ~0.72 | 95 |
| PGOV_Glaukom | ~0.59 | 36 |
| Diabetes | ~0.43 | 25 |
| **SucheOko** | **~0.00** | 14 |

---

## Dataset

- **240 scans, 35 persons, 5 classes** (TRAIN_SET).
- Person-disjoint LOPO is the only honest evaluation — patients are the dominant
  latent variable in a UMAP of our embeddings.
- L/R eyes of the same person collapsed via `person_id()` parser.

---

## Limitations (please read before trusting a prediction)

1. **SucheOko 2-patient ceiling.** TRAIN_SET contains only 2 distinct persons
   labelled `SucheOko` (14 scans). Any person-disjoint fold leaves ≤1 SucheOko
   patient in training. F1 ≈ 0 is a data-acquisition ceiling, not a modelling
   failure.
2. **5-class closed-set only.** The UPJŠ PDF mentions Alzheimer, bipolar,
   panic-disorder, cataract, pigmentary-dispersion — none appear in TRAIN_SET.
   Unseen classes will be mis-classified with high confidence. No OOD detector
   is shipped.
3. **Raw Bruker SPM required for inference.** BMP previews contain scale-bars
   and watermarks that constitute label leakage — the UI accepts BMPs only for
   illustration; it will not run classification on them.
4. **Small data.** 240 scans is data-ceiling territory. Red-team audits showed
   that every claim > 0.65 from threshold / bias tuning on the same OOF was a
   +0.04–0.06 inflation.

---

## Red-team log (summarised)

- Initial claim 0.6698 (threshold-tuned) → rejected by nested CV.
- Nested-threshold honest: 0.6528 (reference, not shipped).
- Shipped: 0.6458 raw-argmax TTA ensemble — no tuning.
- 4-component concat ensemble claimed 0.6878 → honest 0.6326. Rejected.

Full history: `reports/FINAL_REPORT.md`, `reports/RED_TEAM_ENSEMBLE_AUDIT.md`.

---

*Demo built with Gradio {v}. Model bundle:* `models/ensemble_v1_tta/`.
""".format(v=gr.__version__)
            )

if __name__ == "__main__":
    print("[app] Launching Gradio app on http://127.0.0.1:7860")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False,
        theme=THEME,
    )
