# Architecture & Methodology

## ⭐ SHIPPED (Wave 7 + Wave 13-19 confirmed): v4 multiscale ensemble

**Honest person-LOPO weighted F1 = 0.6887** (patient-disjoint regime, matches expected test scenario).

```mermaid
flowchart TB
    A[Raw AFM scan .spm] --> B[Preprocess<br/>plane level<br/>resample 90nm/px<br/>9 tiles 512x512<br/>render afmhot RGB]
    B --> C1[DINOv2-B encode<br/>at 90 nm/px]
    B --> C2[DINOv2-B encode<br/>at 45 nm/px<br/>finer detail]
    B --> C3[BiomedCLIP encode<br/>D4 TTA = 8 rotations<br/>mean embedding]
    C1 --> D1[L2 normalize<br/>+ StandardScaler<br/>+ LR head]
    C2 --> D2[L2 normalize<br/>+ StandardScaler<br/>+ LR head]
    C3 --> D3[L2 normalize<br/>+ StandardScaler<br/>+ LR head]
    D1 --> E[Geometric mean<br/>of 3 softmaxes]
    D2 --> E
    D3 --> E
    E --> F[argmax → predicted class<br/>+ calibrated probabilities]
```

### Why 3 streams not 1

- **DINOv2-B @ 90nm**: overall view of dendritic ferning structure
- **DINOv2-B @ 45nm**: fine-grained crystal edges, branch tip morphology
- **BiomedCLIP @ 90nm + D4 TTA**: medical imaging prior (PubMed pretraining), rotation-invariant
- **3 independent error patterns** → geometric mean cancels uncorrelated mistakes

### What is FROZEN vs TRAINED

- Frozen (no training): all 3 backbone encoders (~150M params total). 240 scans is too small to fine-tune safely (LoRA attempt: -4.1 pp wF1).
- Trained (per LOPO fold): 3 × LR heads (~12k params total). Class-weighted, balanced.

### Bundle artifacts (`models/ensemble_v4_multiscale/`)

```
ensemble_v4_multiscale/
├── meta.json                  # honest_lopo_weighted_f1: 0.6887
├── predict.py                 # inference entrypoint
├── README.md
├── dinov2b_90nm/              # encoder + head + scaler per stream
├── dinov2b_45nm/
└── biomedclip_tta/
```

### Performance breakdown

| Metric | v4 |
|---|---|
| Weighted F1 | **0.6887** |
| Macro F1 | 0.5541 |
| Per-patient F1 | 0.8011 |
| Top-2 accuracy | 88% |
| Per-class F1 | Healthy 0.92 / SM 0.69 / Glaukom 0.58 / Diabetes 0.58 / SucheOko 0.00 |
| SucheOko ceiling | 2 patients = structural limit |

### Why simpler ensembles failed

| Variant | Wave | wF1 | Why not v4 |
|---|---|---|---|
| DINOv2-B alone | 1 | 0.6162 | Single-stream, no diversity |
| 2-stream (v2) | 5 | 0.6562 | Missing 45nm detail |
| 4-stream (zoo+) | 18 | 0.6627 | Diminishing returns, weaker encoders drag mean |
| LoRA fine-tune | 18 | 0.6476 | Overfits 240 samples |

---

## [LEGACY] 1. Inference pipeline (shipped — `models/ensemble_v2_tta/`, F1 = 0.6562)

```mermaid
flowchart LR
    A[Raw Bruker SPM] --> B[preprocess<br>level + resample + normalize]
    B --> C[9 tiles × D4 → 72 views]
    C --> D1[DINOv2-B encode<br>mean-pool tiles]
    C --> D2[BiomedCLIP encode<br>mean-pool tiles]
    D1 --> E1[L2-normalize row-wise]
    D2 --> E2[L2-normalize row-wise]
    E1 --> F1[StandardScaler<br>+ LR balanced]
    E2 --> F2[StandardScaler<br>+ LR balanced]
    F1 --> G[softmax log-space<br>arithmetic mean]
    F2 --> G
    G --> H[renormalize<br>→ argmax]
```

**v2 recipe changes (discovered by Wave-5 autoresearch agent):**
- **L2-normalize** scan embeddings BEFORE StandardScaler (+0.003)
- **Geometric mean** of softmaxes INSTEAD of arithmetic (+0.008)
- Both honest (no tuning), stack cleanly → +0.010 ensemble, +0.023 macro

## 1b. Inference pipeline v1 (earlier champion — `models/ensemble_v1_tta/`, F1 = 0.6458)

```mermaid
flowchart LR
    A[Raw Bruker SPM<br>scan.NNN] --> B[load_height<br>AFMReader]
    B --> C[plane_level<br>1st-order poly subtract]
    C --> D[resample<br>90 nm/px]
    D --> E[robust_normalize<br>2–98 pct clip]
    E --> F[tile<br>9× 512² non-overlap]
    F --> G{For each of 9 tiles}
    G --> H1[D4 augmentations<br>8 views per tile]
    H1 --> I1[DINOv2-B ViT/14<br>frozen]
    H1 --> I2[BiomedCLIP ViT-B/16<br>frozen]
    I1 --> J1[mean-pool<br>72 views → 1]
    I2 --> J2[mean-pool<br>72 views → 1]
    J1 --> K1[StandardScaler<br>+ LR balanced]
    J2 --> K2[StandardScaler<br>+ LR balanced]
    K1 --> L[softmax avg]
    K2 --> L
    L --> M[argmax<br>5-class prediction]
```

**Key invariants:**
- Frozen encoders — no fine-tuning (small-data discipline)
- Each encoder has its own scaler + LR head
- Ensemble is the ARITHMETIC mean of softmaxes
- No thresholds, no bias tuning — honest, robust

## 2. Training protocol

```mermaid
flowchart TD
    A[240 scans × 9 tiles = 2160 tile inputs] --> B[D4 augmentation × 8<br>= 17,280 tile views]
    B --> C[Encode with 2 foundation models<br>DINOv2-B + BiomedCLIP]
    C --> D[Mean-pool per scan<br>240 × 768 and 240 × 512]
    D --> E{Person-level LOPO<br>35 folds}
    E --> F[Train LR per fold<br>class_weight='balanced']
    F --> G[OOF predictions for<br>honest F1 estimate]
    G --> H[Final model:<br>retrain on ALL 240 scans]
    H --> I[Save as ClassifierBundle]
```

## 3. Orchestration pattern

```mermaid
flowchart TB
    ORCH[Orchestrator<br>Claude Opus 4.7<br>tracks STATE.md]
    ORCH -->|dispatch| A[Researcher agents<br>domain literature]
    ORCH -->|dispatch| B[Implementer agents<br>build + run experiments]
    ORCH -->|dispatch| C[Validator agents<br>red-team every win]
    ORCH -->|dispatch| D[Synthesizer agents<br>consolidate results]
    ORCH -->|dispatch| E[Specialist agents<br>class-specific, novel tracks]
    A --> F[Report back]
    B --> F
    C --> F
    D --> F
    E --> F
    F --> G[Update STATE.md]
    G --> ORCH
```

Typical round = 3–5 parallel agents dispatched with self-contained prompts.
Red-team agent audits every F1 > baseline claim before adoption.

## 4. Red-team discipline

For every F1 claim above baseline:
1. **Check grouping**: eye-level vs person-level LOPO
2. **Check tuning leakage**: if any parameter (threshold, subset, bias, α) was tuned, is the eval on DIFFERENT data than the tuning?
3. **Nested CV**: re-evaluate with inner-CV tuning inside each outer fold
4. **Label-shuffle sanity**: is the gap vs null (0.28) substantially larger than our claim's delta?

Three initially-headline claims (0.67–0.69) all collapsed to ≤0.65 under this audit.

## 5. Feature caches layout

```
cache/
├── tiled_emb_dinov2_vits14_afmhot_t512_n9.npz   # (n_tiles, 384) + tile_to_scan mapping
├── tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz   # (n_tiles, 768)
├── tiled_emb_biomedclip_afmhot_t512_n9.npz      # (n_tiles, 512)
├── tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz  # (240, 768) — D4 TTA pre-pooled
├── tta_emb_biomedclip_afmhot_t512_n9_d4.npz     # (240, 512)
├── features_handcrafted.parquet                 # 94-dim classical texture
├── features_tda.parquet                         # 1015-dim persistent homology
├── features_advanced.parquet                    # (if Wave 5) multi-fractal, lacunarity, ...
├── best_ensemble_predictions.npz                # OOF proba from 2-comp ensemble
├── cascade_oof.npz                              # binary specialist OOF
└── supcon_projected_emb.npz                     # (240, 128) SupCon-projected features
```

## 6. Model bundle layout

```
models/
├── dinov2b_tiled_v1/           # single-encoder fallback, 0.615 F1
│   ├── classifier.npz
│   └── meta.json
├── ensemble_v1/                # 2-encoder, no TTA, 0.6346 F1 (fast inference)
│   ├── meta.json (kind=ensemble, components=[dinov2b, biomedclip])
│   ├── dinov2b/
│   └── biomedclip/
└── ensemble_v1_tta/            # SHIPPED CHAMPION, 0.6458 F1
    ├── meta.json
    ├── dinov2b/
    ├── biomedclip/
    ├── predict.py              # TTAPredictor (D4 augmentation at inference)
    └── README.md
```

Legacy bundles load via `TearClassifier.load(...)` which auto-detects `kind` field.
TTA bundle uses its own `TTAPredictor` class to wire D4 augmentation at inference so callers can't accidentally feed 9-view embeddings to an LR trained on 72-view means.
