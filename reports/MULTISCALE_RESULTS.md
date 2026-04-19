# Multi-Scale Tile Experiment — Results

**Hypothesis:** tear crystallization has multi-scale structure (fine crystal lattice at 10-30 nm, macro dendrite at 100+ nm). Combining TWO tile scales (90 nm/px + 45 nm/px) may capture complementary signal.

## Methodology

- **Data:** 240 AFM scans, 35 persons (LOPO groups via `teardrop.data.person_id`).
- **Scales:**
  - 90 nm/px (existing champion cache, `tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz`).
  - 45 nm/px (new cache, `tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz`).
- **Tiling:** `preprocess_spm` followed by `tile(size=512)`, then cap at 9 tiles/scan (evenly-spaced subset). At 45 nm/px a 1024² original scan becomes ~2048² and yields 16 raw tiles before capping.
- **Encoder:** DINOv2-B (vitb14), `afmhot` render, no D4 TTA this round.
- **Pooling:** tile embeddings mean-pooled per scan (scan-level features).
- **Recipe (V2, no tuning):** L2-norm → StandardScaler(fit-on-train) → LogisticRegression(class_weight='balanced', C=1.0, max_iter=3000).
- **Ensemble:** geometric mean of per-member softmax probabilities.
- **Evaluation:** person-level LOPO (35 folds), weighted & macro F1, per-class F1.

## Per-member LOPO metrics

| Member | Weighted F1 | Macro F1 |
|---|---:|---:|
| `dinov2_90nm` | 0.6162 | 0.4941 |
| `dinov2_45nm` | 0.6544 | 0.5186 |
| `biomedclip_tta_90nm` | 0.6220 | 0.4915 |

## Configurations (geom-mean ensembles)

| Config | Members | Weighted F1 | Macro F1 | Δ v2 (0.6562) | Δ E7 (0.6645) |
|---|---|---:|---:|---:|---:|
| **A_dinov2_90nm** | dinov2_90nm | 0.6162 | 0.4941 | -0.0400 | -0.0483 |
| **B_dinov2_45nm** | dinov2_45nm | 0.6544 | 0.5186 | -0.0018 | -0.0101 |
| **C_dinov2_90+45** | dinov2_90nm, dinov2_45nm | 0.6427 | 0.5125 | -0.0135 | -0.0218 |
| **D_dinov2_90+45+biomedclip_tta** | dinov2_90nm, dinov2_45nm, biomedclip_tta_90nm | 0.6887 | 0.5541 | +0.0325 | +0.0242 |

## Per-class F1

| Config | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|:---:|:---:|:---:|:---:|:---:|
| **A_dinov2_90nm** | 0.8169 | 0.5106 | 0.5195 | 0.6237 | 0.0000 |
| **B_dinov2_45nm** | 0.8472 | 0.5000 | 0.5610 | 0.6848 | 0.0000 |
| **C_dinov2_90+45** | 0.8531 | 0.5200 | 0.5333 | 0.6561 | 0.0000 |
| **D_dinov2_90+45+biomedclip_tta** | 0.9167 | 0.5833 | 0.5789 | 0.6915 | 0.0000 |

## Interpretation

- **Best multi-scale config:** `D_dinov2_90+45+biomedclip_tta` with weighted F1 **0.6887** (Δ v2 = +0.0325, Δ E7 = +0.0242).

- 90+45 fusion (Config C) does NOT beat the stronger single-scale (-0.0117). Fine-scale (45 nm/px) features appear redundant with, or noisier than, 90 nm/px features at DINOv2-B. Multi-scale is not the winning axis.

- Config D (0.6887) matches/beats v2 champion by ≥ 0.005 — multi-scale stacks usefully with BiomedCLIP-TTA (multi-encoder × multi-scale).


## Honest reporting
- Person-level LOPO (35 folds), V2 recipe only, no threshold tuning, no OOF model selection.
- Champions are numerically optimistic baselines: v2 uses D4-TTA on both encoders; this experiment uses NO TTA for the DINOv2 members (budget constraint), so the Config A number here is slightly below the TTA-boosted champion.
- Fair comparison: look at the **lift** of C - A and D - (A+BiomedCLIP), not the absolute deltas vs champions.
