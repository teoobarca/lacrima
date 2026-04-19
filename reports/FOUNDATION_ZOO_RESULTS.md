# Foundation-Model Zoo — Results

**Goal:** compare underexplored frozen encoders against the current champions (DINOv2-B = 0.6162, BiomedCLIP-TTA = 0.6220, v4 ensemble = 0.6887) on person-LOPO weighted F1.

## Methodology

- **Data:** 240 AFM scans, 35 persons (LOPO via `teardrop.data.person_id`).
- **Tiling:** 90 nm/px, `afmhot` render, up to 9 non-overlapping 512² tiles per scan, no TTA.
- **Pooling:** mean over tiles → scan-level embedding.
- **Recipe:** L2-normalize → StandardScaler → LogisticRegression(class_weight='balanced', C=1.0), same as v2/v4.
- **Ensembles:** geometric mean of per-member softmax probabilities.
- **Hardware:** MPS (Apple Silicon). Per-encoder budget ~8 min.

## Per-encoder person-LOPO metrics

| Encoder | Status | Dim | Encode time (s) | Weighted F1 | Macro F1 | Δ vs DINOv2-B (0.6162) |
|---|---|---:|---:|---:|---:|---:|
| PubMedCLIP ViT-B/32 (512-d) | cached | 512 | 7.7 | 0.5167 | 0.3712 | -0.0995 |
| OpenCLIP ViT-L/14 LAION-2B (768-d) | cached | 768 | 129.9 | 0.5699 | 0.4345 | -0.0463 |
| DINOv2-L (ViT-L/14, 1024-d) | cached | 1024 | 135.4 | 0.5418 | 0.4128 | -0.0744 |
| EVA02-L-14 (768-d) | cached | 768 | 213.7 | 0.5832 | 0.4563 | -0.0330 |
| SigLIP-SO400M-14-384 (1152-d) | cached | 1152 | 722.9 | 0.5735 | 0.4401 | -0.0427 |
| DINOv2-G (ViT-g/14, 1536-d) | disabled | - | - | - | - | - |

## Per-class F1 (successful encoders)

| Encoder | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|:---:|:---:|:---:|:---:|:---:|
| PubMedCLIP ViT-B/32 (512-d) | 0.7383 | 0.1224 | 0.4286 | 0.5668 | 0.0000 |
| OpenCLIP ViT-L/14 LAION-2B (768-d) | 0.8027 | 0.2979 | 0.4865 | 0.5856 | 0.0000 |
| DINOv2-L (ViT-L/14, 1024-d) | 0.7260 | 0.1905 | 0.5000 | 0.5851 | 0.0625 |
| EVA02-L-14 (768-d) | 0.7832 | 0.4314 | 0.4578 | 0.6092 | 0.0000 |
| SigLIP-SO400M-14-384 (1152-d) | 0.8052 | 0.3111 | 0.5000 | 0.5843 | 0.0000 |

## Ensembles (geom-mean, person-LOPO)

| Config | Members | Weighted F1 | Macro F1 | Δ vs v4 (0.6887) |
|---|---|---:|---:|---:|
| **v4_baseline** | dinov2b_90, dinov2b_45, biomedclip_tta | 0.6887 | 0.5541 | -0.0000 |
| **v4 + DINOv2-L (4-way)** | dinov2b_90, dinov2b_45, biomedclip_tta, dinov2_vitl14 | 0.6557 | 0.5260 | -0.0330 |
| **v4 swap BiomedCLIP-TTA for SigLIP (3-way)** | dinov2b_90, dinov2b_45, siglip_so400m_384 | 0.6480 | 0.5220 | -0.0407 |
| **v4 + DINOv2-L + SigLIP (5-way)** | dinov2b_90, dinov2b_45, biomedclip_tta, dinov2_vitl14, siglip_so400m_384 | 0.6533 | 0.5275 | -0.0354 |
| **v4 + EVA02-L (4-way)** | dinov2b_90, dinov2b_45, biomedclip_tta, eva02_l14 | 0.6520 | 0.5195 | -0.0367 |
| **v4 + OpenCLIP-L LAION-2B (4-way)** | dinov2b_90, dinov2b_45, biomedclip_tta, openclip_vitl14_laion2b | 0.6627 | 0.5286 | -0.0260 |

## Verdict

- No single encoder in this zoo beats DINOv2-B on person-LOPO under the identical (no-TTA, 9-tile, L2+Scaler+LR) recipe.

- No 4-way or 5-way ensemble in this sweep beats the v4 multi-scale champion (0.6887) under person-LOPO. The v4 recipe remains the champion.

## Honest reporting

- Person-level LOPO (35 folds), V2 recipe only, no threshold tuning, no OOF model selection.
- The DINOv2-B 0.6162 baseline uses the IDENTICAL pipeline as every zoo encoder (no TTA, 9 tiles, afmhot). That is why it sits below the v4 number (0.6887); v4 uses BiomedCLIP with D4-TTA plus a second DINOv2-B branch at 45 nm/px.
- `Δ vs v4` for the ensembles reflects drop-in replacements/additions using no-TTA zoo encoders and the v4 TTA BiomedCLIP / 45 nm branch where kept.
