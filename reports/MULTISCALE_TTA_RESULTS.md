# Multi-Scale + D4 TTA Experiment — Results

**Hypothesis:** Adding D4 TTA to the 45 nm/px DINOv2-B branch (the 90 nm/px branch and BiomedCLIP already use D4 TTA) will push the Wave 7 Config D multi-scale ensemble further. The fine scale shows structure the canonical DINOv2 orientation priors may not be invariant to, so TTA has theoretical foothold there as well.

## Methodology

- **Data:** 240 AFM scans, 35 persons (LOPO groups via `teardrop.data.person_id`).
- **Scales & encoders:**
  - DINOv2-B @ 90 nm/px, D4 TTA (72 views/scan, mean-pooled) — `cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`.
  - DINOv2-B @ 45 nm/px, D4 TTA (72 views/scan, mean-pooled) — `cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4_45nm.npz` (**new**).
  - BiomedCLIP @ 90 nm/px, D4 TTA — `cache/tta_emb_biomedclip_afmhot_t512_n9_d4.npz`.
  - DINOv2-B @ 90/45 nm/px non-TTA (for reference) — tiled caches, mean-pooled.
- **Recipe (V2, no tuning):** per scan embedding, L2-normalize -> StandardScaler(fit-on-train) -> LogisticRegression(class_weight='balanced', C=1.0, max_iter=3000).
- **Ensemble:** geometric mean of per-member softmax probabilities.
- **Evaluation:** person-level LOPO (35 folds), weighted & macro F1, per-class F1.

## Per-member LOPO metrics

| Member | Weighted F1 | Macro F1 |
|---|---:|---:|
| `dinov2_90nm_tta` | 0.6464 | 0.5286 |
| `dinov2_45nm_tta` | 0.6255 | 0.4964 |
| `dinov2_90nm_nt` | 0.6162 | 0.4941 |
| `dinov2_45nm_nt` | 0.6544 | 0.5186 |
| `biomedclip_tta` | 0.6220 | 0.4915 |

## Configurations (geom-mean ensembles)

| Config | Members | Weighted F1 | Macro F1 | Δ v2 (0.6562) | Δ D (0.6887) |
|---|---|---:|---:|---:|---:|
| **D_reproduce_nonTTA_multiscale_plus_BC** | dinov2_90nm_nt, dinov2_45nm_nt, biomedclip_tta | 0.6887 | 0.5541 | +0.0325 | -0.0000 |
| **D_TTA_multiscale_plus_BC** | dinov2_90nm_tta, dinov2_45nm_tta, biomedclip_tta | 0.6666 | 0.5371 | +0.0104 | -0.0221 |
| **E_TTA_multiscale_dinov2_only** | dinov2_90nm_tta, dinov2_45nm_tta | 0.6473 | 0.5205 | -0.0089 | -0.0414 |
| **F_TTA_dinov2_45nm_alone** | dinov2_45nm_tta | 0.6255 | 0.4964 | -0.0307 | -0.0632 |

## Per-class F1

| Config | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|:---:|:---:|:---:|:---:|:---:|
| **D_reproduce_nonTTA_multiscale_plus_BC** | 0.9167 | 0.5833 | 0.5789 | 0.6915 | 0.0000 |
| **D_TTA_multiscale_plus_BC** | 0.8951 | 0.5490 | 0.5823 | 0.6593 | 0.0000 |
| **E_TTA_multiscale_dinov2_only** | 0.8811 | 0.5600 | 0.5195 | 0.6417 | 0.0000 |
| **F_TTA_dinov2_45nm_alone** | 0.8531 | 0.4706 | 0.5316 | 0.6264 | 0.0000 |

## Deltas

- `D_TTA_minus_D_reproduced` = -0.0221
- `D_TTA_minus_v2` = +0.0104
- `D_TTA_minus_multiscale_D_target` = -0.0221

## Verdict

**D-TTA REGRESSES from D: 45 nm/px TTA hurts but multi-scale still beats v2. Do NOT ship D-TTA; Config D (non-TTA 45 nm) remains the best multi-scale option.**

## Honest reporting

- Person-level LOPO (35 folds), V2 recipe only, no threshold tuning, no OOF model selection.
- Config D (non-TTA 45 nm/px) reproduction in this script may differ slightly from the Wave 7 published 0.6887 due to minor version drift; the meaningful comparison is **D-TTA - D_reproduced** within this run.
- If D-TTA regresses vs D, the honest conclusion is that TTA is redundant (or harmful) at 45 nm/px — fine-scale tiles may already average out orientation variance because there are ~9 of them covering different parts of the scan.
