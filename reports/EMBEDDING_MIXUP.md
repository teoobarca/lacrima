# Embedding-space Mixup / CutMix — Results

**Question:** Does Mixup / CutMix in *embedding space* (on top of frozen DINOv2-B + BiomedCLIP features) improve the v4 LR-head ensemble?

Rationale: prior `augmented_head_training.py` T3 (D4-expand) failed because rotated views live close to the original in embedding space. Mixup interpolates *between classes*, a fundamentally different regularizer.

## Treatments

- **M0 sanity** — averaged-head sklearn LR (== v4 T1, must reproduce 0.6887).

- **M1 Mixup (any)** — add N mixup samples per fold, λ ~ Beta(0.4, 0.4), any pair.

- **M2 Mixup (within)** — only within-class pairs (pure regularization, no soft labels).

- **M3 Mixup (minority)** — 50%% of pairs anchored on a minority-class sample (SucheOko or Diabetes).

- **M4 CutMix tiles** — build synthetic scan by swapping half the tiles of scan A with tiles of scan B (different class); mean-pool; soft label by tile-fraction.

## Setup

- Person-level LOPO, 35 folds, 240 scans.

- Mixup α = 0.4, mixup:original ratio = 1, mixup sample_weight = 0.5.

- Head: torch LBFGS softmax soft-CE (L2=1, balanced class weights).

- 3-way geometric-mean ensemble over DINOv2-90, DINOv2-45, BiomedCLIP-90.

- v4 reference weighted F1 (reports): **0.6887**.

## Ensemble-level results

| Treatment | Weighted F1 | Macro F1 | Δ vs v4 |
|---|---:|---:|---:|
| **M0_sanity** | 0.6887 | 0.5541 | -0.0000 |
| **M1_mixup_any** | 0.6576 | 0.5241 | -0.0311 |
| **M2_mixup_within** | 0.6887 | 0.5541 | -0.0000 |
| **M3_mixup_minority** | 0.6614 | 0.5276 | -0.0273 |
| **M4_cutmix** | 0.6544 | 0.5206 | -0.0343 |

## Per-class F1

| Treatment | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|:---:|:---:|:---:|:---:|:---:|
| **M0_sanity** | 0.9167 | 0.5833 | 0.5789 | 0.6915 | 0.0000 |
| **M1_mixup_any** | 0.8784 | 0.5000 | 0.5789 | 0.6631 | 0.0000 |
| **M2_mixup_within** | 0.9167 | 0.5833 | 0.5789 | 0.6915 | 0.0000 |
| **M3_mixup_minority** | 0.8844 | 0.5306 | 0.5526 | 0.6703 | 0.0000 |
| **M4_cutmix** | 0.8828 | 0.4889 | 0.5753 | 0.6561 | 0.0000 |

## Paired bootstrap (1000×) vs M0 sanity

| Treatment | Δ mean | Δ p05 | Δ p95 | P(Δ > 0) |
|---|---:|---:|---:|---:|
| **M1** | -0.0316 | -0.0532 | -0.0118 | 0.004 |
| **M2** | +0.0000 | +0.0000 | +0.0000 | 0.000 |
| **M3** | -0.0279 | -0.0467 | -0.0103 | 0.005 |
| **M4** | -0.0352 | -0.0618 | -0.0105 | 0.004 |

## Verdicts

- **M1**: ROLLBACK (delta=-0.0316, p(>0)=0.00)
- **M2**: NOISE_FLOOR (delta=+0.0000, p(>0)=0.00)
- **M3**: ROLLBACK (delta=-0.0279, p(>0)=0.01)
- **M4**: ROLLBACK (delta=-0.0352, p(>0)=0.00)


Promotion rule: **wF1 ≥ 0.70 AND P(Δ>0) ≥ 0.90** → champion candidate; Δ ≥ +0.02 AND P(Δ>0) ≥ 0.90 → promote; |Δ| < 0.01 → noise floor.
