# Teardrop AFM demo — how to run

Pitch-ready Gradio app for the UPJŠ / Hack Košice 2026 challenge. Exposes the
shipped TTA ensemble (`models/ensemble_v1_tta/`, honest person-LOPO F1 = 0.6458)
via a three-tab interactive UI.

## Run it

```bash
cd /Users/rafael/Programming/teardrop-challenge
.venv/bin/pip install gradio       # one-time — Gradio 6.x
.venv/bin/python app.py
```

The app binds to `http://127.0.0.1:7860`. On **cold start** it:

1. Loads DINOv2-B via `torch.hub` (cached in `~/.cache/torch/hub/` after first run).
2. Loads BiomedCLIP via `transformers` / `open_clip` (cached in `~/.cache/huggingface/`).
3. Loads the TTA ensemble bundle from `models/ensemble_v1_tta/`.
4. Builds a retrieval index by mean-pooling the 811 tile embeddings in
   `cache/tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz` to 240 per-scan vectors.
5. Pre-renders 3 training examples per class for the "Disease fingerprints" gallery.

Expect ~60–120 s cold start on CPU, ~30–40 s on MPS (Apple Silicon).
Subsequent invocations skip the torch-hub download and are much faster.

## Tabs

### 1. Classify a scan
- Upload **raw Bruker Nanoscope `.NNN`** (primary) or **`.bmp`** (loaded for
  display only — BMP previews contain scale-bar watermarks, no inference run).
- "**Run demo sample**" buttons auto-load one curated scan per class from
  `TRAIN_SET/` for instant testing during the pitch.
- Output panel shows:
  - Large colored prediction with confidence.
  - Preprocessed height map (`afmhot`, 512×512) + first 4 of 9 tile crops.
  - Bar chart of all 5 class probabilities with predicted class highlighted.
  - 3 nearest-neighbour training scans (DINOv2-B cosine similarity, from the
    cached `tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz`) with thumbnails.
  - Handcrafted descriptor summary: roughness Sa/Sq/Ssk, fractal dimension,
    GLCM contrast / homogeneity at d=1 and d=5.

### 2. Disease fingerprints
- Per-class explainer: morphology description, distinguishing features,
  Masmali grade range, 3 example thumbnails (deterministic seed=42).
- Full 5×3 class-morphology grid from `reports/pitch/02_class_morphology_grid.png`
  shown at the bottom.

### 3. About
- Full method description (preprocessing pipeline, dual-encoder softmax-mean).
- Honest performance table (F1 = 0.6458 person-LOPO, macro F1 = 0.49).
- Per-class F1 and support counts.
- Limitations (SucheOko 2-patient ceiling, 5-class closed-set, BMP leakage).
- Red-team log summary.

## Screenshots for the pitch

Recommended capture sequence (5 min pitch fits these in order):

1. Tab 1 landing state (before upload), to show the upload widget.
2. After clicking **ZdraviLudia** demo button — screenshot the prediction panel,
   probability chart, and 3 nearest-neighbour gallery (all three should be
   `ZdraviLudia` — good signal of model coherence).
3. After clicking **PGOV_Glaukom** demo button — shows confusion with `SklerozaMultiplex`
   in the runner-up slot (honest weakness).
4. After clicking **SucheOko** demo button — model will likely predict
   `SklerozaMultiplex` or similar. Use this as the visual for the
   "2-patient data ceiling" slide.
5. Tab 2: per-class fingerprints — one full-page screenshot works.
6. Tab 3: About — crop just the performance table section.

Screenshot tooling: macOS `Cmd+Shift+4`, save to `reports/demo_screenshots/`.

## Common issues

- **"No module named gradio"** — run `.venv/bin/pip install gradio`.
- **DINOv2 weights download on first run** — ~350 MB via `torch.hub`. Cache lives
  in `~/.cache/torch/hub/`. The app does not re-download on subsequent runs.
- **Port 7860 already in use** — kill any stale `python app.py` (`lsof -i :7860`)
  or edit `demo.launch(server_port=...)` in `app.py`.
- **File upload fails for `.NNN`** — Gradio's browser file picker may hide
  numeric extensions on some OSes. Use "Browse → All files" in the file dialog.
- **SucheOko misclassified in the demo** — expected. TRAIN_SET has 2 patients
  only; this is a data-acquisition ceiling, not a model failure (the About tab
  explains this explicitly).

## Files produced by the app

All outputs are rendered in-memory (PIL images, pyplot figures); the app does
not write files. Exit cleanly via `Ctrl+C`.

## Reproducing honest metrics

```bash
.venv/bin/python scripts/prob_ensemble.py     # 2-component ensemble leaderboard
.venv/bin/python scripts/redteam_v2.py        # nested-CV audit
```

Headline numbers also recorded in `SUBMISSION.md` and `STATE.md`.
