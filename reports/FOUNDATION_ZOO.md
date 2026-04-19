# Foundation-Model Zoo — Wave 6 Report

**Hypothesis:** foundation-model diversity (DINOv2-L, SigLIP-SO400M, EVA-02, OpenCLIP-L, PubMedCLIP) beats the single-track v4 (DINOv2-B 90nm + DINOv2-B 45nm + BiomedCLIP-TTA, wF1=0.6887).

## Protocol

- Person-level LOPO (35 folds, `StratifiedGroupKFold` via `teardrop.cv`).
- Identical encode recipe for all zoo members: 90 nm/px resample, plane-level, robust-norm, 9 non-overlapping 512² afmhot tiles, mean-pool to scan embedding, no TTA.
- Head: L2-norm → StandardScaler → LogisticRegression(class_weight='balanced', C=1.0). Identical across encoders; no per-encoder tuning.
- Ensembles: geometric mean of class softmaxes.
- Bootstrap: 1000× paired resample on wF1, scan-level.

## Per-encoder standalone (person-LOPO)

| Encoder | Status | Dim | Encode(s) | wF1 | Macro F1 |
|---|---|---:|---:|---:|---:|
| PubMedCLIP ViT-B/32 (512-d) | cached | 512 | 7.7 | 0.5167 | 0.3712 |
| OpenCLIP ViT-L/14 LAION-2B (768-d) | cached | 768 | 129.9 | 0.5699 | 0.4345 |
| DINOv2-L (ViT-L/14, 1024-d) | cached | 1024 | 135.4 | 0.5418 | 0.4128 |
| EVA02-L-14 (768-d) | cached | 768 | 213.7 | 0.5832 | 0.4563 |
| SigLIP-SO400M-14-384 (1152-d) | cached | 1152 | 722.9 | 0.5735 | 0.4401 |
| DINOv2-G (ViT-g/14, 1536-d) | disabled | - | - | - | - |

## Per-class F1 (standalone)

| Encoder | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|:---:|:---:|:---:|:---:|:---:|
| PubMedCLIP ViT-B/32 (512-d) | 0.7383 | 0.1224 | 0.4286 | 0.5668 | 0.0000 |
| OpenCLIP ViT-L/14 LAION-2B (768-d) | 0.8027 | 0.2979 | 0.4865 | 0.5856 | 0.0000 |
| DINOv2-L (ViT-L/14, 1024-d) | 0.7260 | 0.1905 | 0.5000 | 0.5851 | 0.0625 |
| EVA02-L-14 (768-d) | 0.7832 | 0.4314 | 0.4578 | 0.6092 | 0.0000 |
| SigLIP-SO400M-14-384 (1152-d) | 0.8052 | 0.3111 | 0.5000 | 0.5843 | 0.0000 |

## Ensemble sweep (geom-mean softmax, person-LOPO)

| Config | Members | wF1 | Macro F1 | Δ vs v4 (0.6887) |
|---|---|---:|---:|---:|
| **v4_baseline** | dinov2b_90, dinov2b_45, biomedclip_tta | 0.6887 | 0.5541 | -0.0000 |
| **v4 + DINOv2-L (4-way)** | dinov2b_90, dinov2b_45, biomedclip_tta, dinov2_vitl14 | 0.6557 | 0.5260 | -0.0330 |
| **v4 swap BiomedCLIP-TTA for SigLIP (3-way)** | dinov2b_90, dinov2b_45, siglip_so400m_384 | 0.6480 | 0.5220 | -0.0407 |
| **v4 + DINOv2-L + SigLIP (5-way)** | dinov2b_90, dinov2b_45, biomedclip_tta, dinov2_vitl14, siglip_so400m_384 | 0.6533 | 0.5275 | -0.0354 |
| **v4 + EVA02-L (4-way)** | dinov2b_90, dinov2b_45, biomedclip_tta, eva02_l14 | 0.6520 | 0.5195 | -0.0367 |
| **v4 + OpenCLIP-L LAION-2B (4-way)** | dinov2b_90, dinov2b_45, biomedclip_tta, openclip_vitl14_laion2b | 0.6627 | 0.5286 | -0.0260 |
| **greedy_forward_from_v4** | dinov2b_90, dinov2b_45, biomedclip_tta | 0.6887 | 0.5541 | -0.0000 |

## Greedy forward selection (start from v4)

Rule: at each step, try adding every remaining zoo encoder; keep the one that improves wF1 the most. Stop when no candidate improves.

| Step | Added | Members | wF1 | Δ vs v4 |
|---:|---|---|---:|---:|
| 0 | - | dinov2b_90, dinov2b_45, biomedclip_tta | 0.6887 | +0.0000 |

## Bootstrap best-zoo vs v4 (1000× paired)

- Best ensemble: **v4 + OpenCLIP-L LAION-2B (4-way)** (members: dinov2b_90, dinov2b_45, biomedclip_tta, openclip_vitl14_laion2b)
- Best wF1 (point): 0.6627
- v4 wF1 (point): 0.6887
- Bootstrap mean Δ = -0.0261
- 95% CI on Δ = [-0.0486, -0.0068]
- **P(Δ > 0) = 0.000**

## Verdict

**v4 REMAINS CHAMPION.** No zoo ensemble reaches the required bar (wF1 > 0.70 AND P(Δ>0) > 0.90). Best zoo ensemble: **v4 + OpenCLIP-L LAION-2B (4-way)** @ 0.6627 wF1 (Δmean = -0.0261, P(Δ>0) = 0.000).

### Why diversity did not help

- Every added zoo encoder underperforms DINOv2-B standalone (best zoo single = EVA02-L @ 0.5832). Geom-mean of softmaxes penalises members that confidently disagree with the majority, so a weaker member pulls the ensemble down, not up.
- The v4 stack already has latent-space diversity from 3 complementary sources: two DINOv2-B scales (90 nm vs 45 nm) plus D4-TTA BiomedCLIP. Adding a fourth non-medical encoder is marginal redundancy rather than new information.
- No zoo encoder solves SucheOko (F1=0.00 across the board). That is the class bottleneck, not encoder capacity.

## Artefacts

- `reports/foundation_zoo_results.json` — structured summary.
- `cache/zoo_predictions.json` — per-variant OOF softmax + preds.
- `cache/tiled_emb_<encoder>_afmhot_t512_n9.npz` — cached tile embeddings (reusable for future ensembles, linear probes, stacking).
