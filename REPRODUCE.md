# Reproduce

Step-by-step to rebuild everything from scratch.

## Prerequisites

- **macOS or Linux**, Python **3.13** (NOT 3.14 — `pySPM` breaks on 3.14)
- ~10 GB disk (TRAIN_SET.zip = 3.2 GB, extracted = 4 GB, caches = ~2 GB, models = ~3 MB)
- GPU nice-to-have: Apple MPS or CUDA speeds up encoding ~5x. Fallback CPU works in ~30–60 min total.

## 1. Clone + environment

```bash
git clone <this-repo>
cd teardrop-challenge
python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Note: `cripser` may require a C++ toolchain. On macOS: `xcode-select --install`.

## 2. Data

Download the original dataset (3.2 GB ZIP):

```bash
curl -L "https://temp.kotol.cloud/api/download/TRAIN_SET.zip?code=7IDU" -o TRAIN_SET.zip
```

Extract (Windows-style filename encoding — must use Python, not `unzip`):

```bash
.venv/bin/python - <<'PY'
import zipfile
from pathlib import Path

z = zipfile.ZipFile('TRAIN_SET.zip')
out = Path('TRAIN_SET')
out.mkdir(exist_ok=True)
for info in z.infolist():
    if info.flag_bits & 0x800:
        name = info.filename
    else:
        try:
            name = info.filename.encode('cp437').decode('cp1250')
        except Exception:
            name = info.filename
    target = out / name
    if info.is_dir():
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as src, open(target, 'wb') as dst:
            dst.write(src.read())
# Fix the one Slovak-encoded directory name
import os
if (out / 'Skler˘zaMultiplex').exists():
    os.rename(out / 'Skler˘zaMultiplex', out / 'SklerozaMultiplex')
PY
```

## 3. Data audit

```bash
.venv/bin/python scripts/data_audit.py        # reports/DATA_AUDIT.md
.venv/bin/python scripts/probe_raw_afm.py     # reports/raw_afm_probe.csv
.venv/bin/python scripts/visualize_samples.py # reports/samples/*.png
.venv/bin/python scripts/visualize_raw_afm.py # reports/raw_samples/*.png
```

## 4. Build baseline embeddings (~10 min on MPS)

```bash
# Tiled non-TTA for each encoder
.venv/bin/python scripts/baseline_tiled_ensemble.py dinov2_vits14
.venv/bin/python scripts/baseline_tiled_ensemble.py dinov2_vitb14
.venv/bin/python scripts/baseline_tiled_ensemble.py biomedclip

# Handcrafted features
.venv/bin/python scripts/baseline_handcrafted_xgb.py

# TDA features (optional, slow)
.venv/bin/python scripts/baseline_tda.py
```

## 5. TTA embeddings + shipped v2 champion (~10 min MPS)

```bash
# Build D4 TTA-pooled embeddings for both encoders (~10 min on MPS)
.venv/bin/python scripts/tta_experiment.py

# v1 TTA ensemble (arith-mean baseline)
.venv/bin/python scripts/train_ensemble_tta_model.py   # → models/ensemble_v1_tta/

# v2 TTA ensemble — SHIPPED CHAMPION (L2-norm + geom-mean, +0.011 over v1)
.venv/bin/python scripts/train_ensemble_v2_tta.py      # → models/ensemble_v2_tta/
```

v2 recipe discovered by Wave-5 autoresearch agent: `normalize → StandardScaler → LR → geometric-mean softmax combination`. See `reports/AUTORESEARCH_WAVE5_RESULTS.md`.

## 6. Verify

```bash
# Canonical benchmark (writes reports/BENCHMARK_DASHBOARD.md)
.venv/bin/python scripts/benchmark_dashboard.py

# End-to-end CLI smoke test (single scan)
mkdir -p /tmp/smoke && cp TRAIN_SET/Diabetes/37_DM.010 /tmp/smoke/
.venv/bin/python predict_cli.py --input /tmp/smoke --output /tmp/smoke.csv
# Expected: predicted_class=Diabetes, confidence > 0.9
```

## 7. Inference on a new test set

```bash
.venv/bin/python predict_cli.py \
    --model models/ensemble_v2_tta \
    --input /path/to/TEST_SET \
    --output submission.csv
```

Expected wall time: ~8 s/scan × N scans (D4 TTA is the bottleneck).

## 8. Optional: interactive demo

```bash
.venv/bin/python app.py  # starts Gradio on localhost:7860
```

## Key files by directory

| Path | Purpose |
|---|---|
| `teardrop/` | Python library (data, CV, features, encoders, infer, graph, gin_model, topology, llm_reason) |
| `scripts/` | Runnable experiments + utilities |
| `cache/` | Computed embeddings and feature parquet files (gitignored) |
| `models/` | Shippable bundles (JSON + NPZ, git-tracked) |
| `reports/` | All markdown outputs, visualizations, CSVs (git-tracked) |
| `STATE.md` | Orchestration ledger — single source of truth for experiment state |
| `predict_cli.py` | Entry point for inference on new data |

## Known gotchas

- `AFMReader` uses `loguru` which prints to stderr; our library silences it on import
- `pySPM` warns about `tqdm` — benign
- `ripser` (fallback for `giotto-tda`) needs C++ build tools
- Filename encoding in TRAIN_SET.zip is cp1250 (Windows Central European); standard `unzip` fails on `SklerózaMultiplex`
- `person_id` parser merges L/R eyes of same person (35 persons, not 44); `patient_id` does NOT
- LOPO evaluation must use `s.person`, not `s.patient` — otherwise F1 is inflated by ~1 %
