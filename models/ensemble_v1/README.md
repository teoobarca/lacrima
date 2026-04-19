# Ensemble v1 — DINOv2-B + BiomedCLIP (scan-mean) softmax-average

This is the shippable hackathon model for 5-class tear AFM disease classification.

## What this model is

Two frozen vision encoders, each followed by mean-pooling over tiles and a
separate `StandardScaler → LogisticRegression(class_weight='balanced', C=1.0,
max_iter=3000)` head. At inference we average the two softmax outputs
uniformly and take argmax. No thresholds, no calibration, no stacking.

Pipeline per scan:

1. Load Bruker SPM height map → plane-level subtract → resample to 90 nm/px
   → robust-normalize (2-98 percentile) → cut into up to 9 non-overlapping
   512x512 tiles.
2. Render each tile as an `afmhot` RGB image (PIL).
3. For each component:
   - encode all tiles via the frozen encoder (DINOv2-B ViT-B/14 — 768 dim,
     BiomedCLIP ViT-B/16 — 512 dim)
   - mean-pool tile embeddings → one scan-level vector
   - standardize with the saved scaler → LR logits → softmax
4. Arithmetic mean of the two softmax outputs, argmax → predicted class.

Trained on the full 240-scan TRAIN_SET (no CV split held out — we keep all data
for the final shippable model). LOPO numbers below come from earlier honest
validation runs.

## Expected honest performance (person-level LOPO, from red-team audit)

- **Weighted F1 raw argmax (what we ship): 0.6346**
- Weighted F1 with nested-CV per-class thresholds (reference, not packaged): 0.6528
- Baseline (DINOv2-B alone, single-encoder v1): 0.615

Per-class expected F1 (person-level LOPO):

| Class              | F1    |
|--------------------|------:|
| ZdraviLudia        | ~0.82 |
| SklerozaMultiplex  | ~0.68 |
| PGOV_Glaukom       | ~0.57 |
| Diabetes           | ~0.46 |
| SucheOko           | ~0.00 |

## Known limitations

- **SucheOko 2-patient ceiling.** The training set contains only 2 distinct
  persons labelled SucheOko (14 scans total). Any person-level
  held-out-patient evaluation necessarily has 0 SucheOko training patients
  for that fold, so F1 on SucheOko under honest LOPO is ~0.0. In-distribution
  (e.g. the full training fit we just produced) the model memorises them
  perfectly; out-of-distribution generalisation to a *third* SucheOko
  patient is not measurable from this dataset.
- **5 classes only.** The hackathon slides reference additional classes
  (Alzheimer, bipolar, panic, cataract, PDS) that are not present in the
  TRAIN_SET. This model only predicts `ZdraviLudia`, `Diabetes`,
  `PGOV_Glaukom`, `SklerozaMultiplex`, `SucheOko`. If the TEST_SET contains
  unseen classes, they will be mis-classified with high confidence.
- **Patient is a dominant latent variable.** UMAP shows patient clusters
  stronger than class clusters — small-sample effect.
- **Training F1 is 1.0.** That is expected (LR fits perfectly in these
  dimensions on 240 samples) and is *not* a validation number.
- **No thresholds.** We intentionally ship raw argmax because threshold tuning
  on the same 240-scan OOF inflated F1 by ~0.04 and did not generalise under
  nested CV (see `reports/RED_TEAM_ENSEMBLE_AUDIT.md`). If you need the
  0.6528 variant, you'd have to re-introduce per-class thresholds fitted
  under nested CV.

## Directory layout

```
models/ensemble_v1/
├── dinov2b/
│   ├── classifier.npz       # scaler means/scales + LR coef/intercept
│   └── meta.json            # encoder_name = dinov2_vitb14
├── biomedclip/
│   ├── classifier.npz
│   └── meta.json            # encoder_name = biomedclip
├── meta.json                # kind='ensemble', components=['dinov2b','biomedclip']
├── train_predictions.csv    # 240 training-scan predictions (sanity only)
└── README.md
```

## How to load and use

```python
from teardrop.infer import TearClassifier

clf = TearClassifier.load('models/ensemble_v1')

# Single scan:
pred_class, probs = clf.predict_scan('path/to/scan.015')

# Whole directory (walks recursively for raw SPM files):
df = clf.predict_directory('/path/to/TEST_SET')
df.to_csv('submission.csv', index=False)
```

The classifier auto-detects bundle kind via `meta.json['kind']` and will
load either a single-encoder bundle (`models/dinov2b_tiled_v1/`) or an
ensemble bundle (this directory).

First call will download DINOv2 via `torch.hub` (~350 MB) and BiomedCLIP via
HuggingFace hub. Subsequent runs are cached locally.

## Reproducing

```
.venv/bin/python scripts/train_ensemble_model.py
```

Requires cached tiled embeddings under `cache/`:

- `cache/tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz`
- `cache/tiled_emb_biomedclip_afmhot_t512_n9.npz`

Both are produced by `scripts/baseline_tiled_ensemble.py`.
