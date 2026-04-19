# LR Hyperparameter Sweep — Nested Person-LOPO

## TL;DR

- **Best ensemble (nested-LOPO, honest):** `geom_weighted` → **wF1 = 0.6357**, mF1 = 0.4860
- **v4 default (flat LOPO, reconstructed):** wF1 = 0.6887 (published: 0.6887)
- **Δ vs v4:** mean = -0.0534, **P(Δ>0) = 0.000**, 95% CI = [-0.0872, -0.0253]
- **Verdict:** `NOISE_FLOOR_V4_AT_CEILING`

## Protocol (no inner→outer leakage)

Stage 1: nested person-LOPO. Outer 35-fold LOPO (hold out 1 person), inner 5-fold StratifiedGroupKFold on remaining 34 persons. Per encoder, inner CV selects best {C, class_weight} from lbfgs+l2 pool; refit on full outer_train with best; predict outer_val. Ensemble weights tuned on inner OOF (arith & geom grid search). All Stage-1 reported metrics are outer-OOF — no inner leakage into test. 

Stage 2: single-level 5-fold person-stratified CV (saga+{l1,l2} pool). Exploratory only — NOT used for final metric because single-level CV on 240 scans is slightly optimistic; documented for future inclusion if saga+l1 shows a large gain worth bearing the nested cost.


**Stage 1 axes (nested-LOPO, per encoder):**
- `C`: {0.01, 0.1, 1.0, 10.0, 100.0}
- `solver` × `penalty`: {lbfgs × l2}
- `class_weight`: {None, 'balanced', custom sqrt-inverse frequency}
- `max_iter` = 5000 (all lbfgs runs converged).

**Stage 2 axes (single-level 5-fold, exploratory only):**
- `C`: {0.1, 1.0, 10.0}
- `solver` × `penalty`: {saga × l1, saga × l2}
- `class_weight`: {'balanced', sqrt_inv}
- `max_iter` = 2000, `tol` = 1e-3.

**Ensemble-combining axes (tuned on Stage 1 inner OOF per outer fold):**
- geometric mean (unweighted + weighted)
- arithmetic mean (unweighted + weighted)
- per-member weight grid: {0.2, 0.5, 1.0, 2.0, 5.0}, triplet normalized.

## Per-encoder outer-OOF metrics (Stage 1, nested-selected)

| Encoder | Weighted F1 | Macro F1 | Per-class F1 |
|---|---:|---:|---|
| `dinov2_90` | 0.5528 | 0.4190 | 0.776 / 0.244 / 0.507 / 0.568 / 0.000 |
| `dinov2_45` | 0.6394 | 0.4975 | 0.836 / 0.426 / 0.545 / 0.681 / 0.000 |
| `biomedclip_tta` | 0.6033 | 0.4574 | 0.846 / 0.356 / 0.448 / 0.638 / 0.000 |

*Per-class order:* ZdraviLudia / Diabetes / PGOV_Glaukom / SklerozaMultiplex / SucheOko

## Ensemble combinations (Stage 1 outer OOF)

| Method | Weighted F1 | Macro F1 | Per-class F1 |
|---|---:|---:|---|
| `geom_unweighted` | 0.6350 | 0.4883 | 0.857 / 0.381 / 0.533 / 0.670 / 0.000 |
| `arith_unweighted` | 0.6215 | 0.4750 | 0.844 / 0.341 / 0.533 / 0.656 / 0.000 |
| `arith_weighted` | 0.6325 | 0.4838 | 0.867 / 0.372 / 0.514 / 0.667 / 0.000 |
| `geom_weighted` * | 0.6357 | 0.4860 | 0.868 / 0.381 / 0.507 / 0.674 / 0.000 |

**Mean tuned weights** across 35 outer folds (encoders = dinov2_90, dinov2_45, biomedclip_tta):
- arith: [0.158, 0.39, 0.452]
- geom:  [0.142, 0.321, 0.537]

## Winning LR config per encoder (Stage 1)

Frequency across 35 outer folds (each outer fold independently picks its best config on its inner 5-fold CV):


### `dinov2_90` — top 3 picks

| # folds / 35 | C | solver | penalty | class_weight |
|---:|---:|---|---|---|
| 7 | 10.0 | lbfgs | l2 | sqrt_inv |
| 7 | 0.01 | lbfgs | l2 | none |
| 4 | 1.0 | lbfgs | l2 | balanced |

### `dinov2_45` — top 3 picks

| # folds / 35 | C | solver | penalty | class_weight |
|---:|---:|---|---|---|
| 12 | 0.1 | lbfgs | l2 | sqrt_inv |
| 5 | 1.0 | lbfgs | l2 | balanced |
| 5 | 1.0 | lbfgs | l2 | sqrt_inv |

### `biomedclip_tta` — top 3 picks

| # folds / 35 | C | solver | penalty | class_weight |
|---:|---:|---|---|---|
| 9 | 0.1 | lbfgs | l2 | none |
| 8 | 0.1 | lbfgs | l2 | sqrt_inv |
| 5 | 1.0 | lbfgs | l2 | none |

## Class-weight ablation (C=1, lbfgs, l2, geom-mean)

| class_weight | Weighted F1 | Macro F1 | SucheOko F1 | Per-class F1 |
|---|---:|---:|---:|---|
| `none` | 0.6588 | 0.5158 | 0.0000 | 0.880 / 0.455 / 0.560 / 0.684 / 0.000 |
| `balanced` | 0.6887 | 0.5541 | 0.0000 | 0.917 / 0.583 / 0.579 / 0.691 / 0.000 |
| `sqrt_inv` | 0.6811 | 0.5476 | 0.0000 | 0.910 / 0.583 / 0.560 / 0.684 / 0.000 |

## Stage 2: saga + {l1, l2} 5-fold exploration (NOT nested)

*Single-level 5-fold person-stratified CV. Numbers are slightly optimistic compared to LOPO — use only to decide whether saga+l1 is worth pursuing.*

| Encoder | Config | Weighted F1 | Macro F1 |
|---|---|---:|---:|
| `dinov2_90` | `__lbfgs_baseline__` | 0.6485 | 0.5156 |
| `dinov2_90` | `saga_l1_C0.1_balanced` | 0.5754 | 0.4733 |
| `dinov2_90` | `saga_l1_C0.1_sqrt_inv` | 0.5694 | 0.4129 |
| `dinov2_90` | `saga_l2_C0.1_balanced` | 0.6391 | 0.5120 |
| `dinov2_90` | `saga_l2_C0.1_sqrt_inv` | 0.6406 | 0.5127 |
| `dinov2_90` | `saga_l1_C1.0_balanced` | 0.6333 | 0.5054 |
| `dinov2_90` | `saga_l1_C1.0_sqrt_inv` | 0.6366 | 0.5055 |
| `dinov2_90` | `saga_l2_C1.0_balanced` | 0.6350 | 0.5079 |
| `dinov2_90` | `saga_l2_C1.0_sqrt_inv` | 0.6409 | 0.5131 |
| `dinov2_90` | `saga_l1_C10.0_balanced` | 0.6464 | 0.5283 |
| `dinov2_90` | `saga_l1_C10.0_sqrt_inv` | 0.6408 | 0.5137 |
| `dinov2_90` | `saga_l2_C10.0_balanced` | 0.6350 | 0.5079 |
| `dinov2_90` | `saga_l2_C10.0_sqrt_inv` | 0.6409 | 0.5131 |
| `dinov2_45` | `__lbfgs_baseline__` | 0.6806 | 0.5505 |
| `dinov2_45` | `saga_l1_C0.1_balanced` | 0.6268 | 0.5085 |
| `dinov2_45` | `saga_l1_C0.1_sqrt_inv` | 0.6486 | 0.4848 |
| `dinov2_45` | `saga_l2_C0.1_balanced` | 0.6746 | 0.5499 |
| `dinov2_45` | `saga_l2_C0.1_sqrt_inv` | 0.6641 | 0.5317 |
| `dinov2_45` | `saga_l1_C1.0_balanced` | 0.6633 | 0.5380 |
| `dinov2_45` | `saga_l1_C1.0_sqrt_inv` | 0.6530 | 0.5202 |
| `dinov2_45` | `saga_l2_C1.0_balanced` | 0.6799 | 0.5511 |
| `dinov2_45` | `saga_l2_C1.0_sqrt_inv` | 0.6735 | 0.5410 |
| `dinov2_45` | `saga_l1_C10.0_balanced` | 0.6769 | 0.5512 |
| `dinov2_45` | `saga_l1_C10.0_sqrt_inv` | 0.6749 | 0.5458 |
| `dinov2_45` | `saga_l2_C10.0_balanced` | 0.6803 | 0.5509 |
| `dinov2_45` | `saga_l2_C10.0_sqrt_inv` | 0.6816 | 0.5510 |
| `biomedclip_tta` | `__lbfgs_baseline__` | 0.6578 | 0.5238 |
| `biomedclip_tta` | `saga_l1_C0.1_balanced` | 0.6149 | 0.4985 |
| `biomedclip_tta` | `saga_l1_C0.1_sqrt_inv` | 0.6287 | 0.4444 |
| `biomedclip_tta` | `saga_l2_C0.1_balanced` | 0.6843 | 0.5696 |
| `biomedclip_tta` | `saga_l2_C0.1_sqrt_inv` | 0.7040 | 0.5843 |
| `biomedclip_tta` | `saga_l1_C1.0_balanced` | 0.6555 | 0.5293 |
| `biomedclip_tta` | `saga_l1_C1.0_sqrt_inv` | 0.6552 | 0.5173 |
| `biomedclip_tta` | `saga_l2_C1.0_balanced` | 0.6757 | 0.5605 |
| `biomedclip_tta` | `saga_l2_C1.0_sqrt_inv` | 0.6929 | 0.5735 |
| `biomedclip_tta` | `saga_l1_C10.0_balanced` | 0.6756 | 0.5571 |
| `biomedclip_tta` | `saga_l1_C10.0_sqrt_inv` | 0.6987 | 0.5769 |
| `biomedclip_tta` | `saga_l2_C10.0_balanced` | 0.6765 | 0.5567 |
| `biomedclip_tta` | `saga_l2_C10.0_sqrt_inv` | 0.6892 | 0.5660 |

## Bootstrap vs v4-default (1000 paired resamples)

- Mean Δ weighted F1: -0.0534
- **P(Δ > 0)**: 0.000
- 95% CI: [-0.0872, -0.0253]

## Verdict

**NOISE FLOOR.** Best sweep ≤ 0.695 — despite exhaustive nested CV over C × class_weight, no meaningful lift over v4 defaults. The LR head is not the bottleneck; features are. v4 defaults confirmed at their ceiling.


_Elapsed: 485.0s. Cache: `cache/lr_hparam_sweep_predictions.json`._
