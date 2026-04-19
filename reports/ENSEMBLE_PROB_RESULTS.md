# Probability-Averaging Ensemble Results

Person-level LOPO (35 persons, 240 scans). All components use `LogisticRegression(class_weight='balanced', max_iter=3000)` + `StandardScaler`, trained fresh per fold.

## Per-component standalone F1

| component | weighted F1 | macro F1 | F1 ZdraviLudia | F1 Diabetes | F1 PGOV_Glaukom | F1 SklerozaMultiplex | F1 SucheOko |
|---|---|---|---|---|---|---|---|
| dinov2_s | 0.5927 | 0.4782 | 0.819 | 0.465 | 0.459 | 0.589 | 0.059 |
| dinov2_b | 0.6150 | 0.4910 | 0.814 | 0.500 | 0.513 | 0.628 | 0.000 |
| biomedclip | 0.5841 | 0.4385 | 0.775 | 0.269 | 0.507 | 0.642 | 0.000 |
| handcrafted | 0.4882 | 0.3707 | 0.746 | 0.211 | 0.368 | 0.481 | 0.047 |
| tda | 0.4777 | 0.3615 | 0.781 | 0.255 | 0.333 | 0.438 | 0.000 |

## Ensemble strategies (pre-threshold, person-LOPO weighted F1)

| strategy | subset | weighted F1 | macro F1 |
|---|---|---|---|
| per_class_weighted | dinov2_s+dinov2_b+biomedclip+handcrafted+tda | 0.6474 | 0.4989 |
| uniform_subset_k2 | dinov2_b+biomedclip | 0.6346 | 0.4934 |
| f1_weighted_top4 | dinov2_b+dinov2_s+biomedclip+handcrafted | 0.6305 | 0.4877 |
| uniform_subset_k2 | dinov2_s+biomedclip | 0.6286 | 0.4833 |
| uniform_subset_k3 | dinov2_s+dinov2_b+biomedclip | 0.6285 | 0.4868 |
| f1_weighted_top3 | dinov2_b+dinov2_s+biomedclip | 0.6285 | 0.4868 |
| geometric_mean | dinov2_s+dinov2_b+biomedclip+handcrafted+tda | 0.6155 | 0.4744 |
| uniform_subset_k4 | dinov2_s+dinov2_b+biomedclip+handcrafted | 0.6146 | 0.4720 |
| uniform_subset_k3 | dinov2_b+biomedclip+handcrafted | 0.6133 | 0.4677 |
| uniform_avg | dinov2_s+dinov2_b+biomedclip+handcrafted+tda | 0.6108 | 0.4687 |
| stacking_meta_lr | dinov2_s+dinov2_b+biomedclip+handcrafted+tda | 0.6078 | 0.5492 |
| f1_weighted | dinov2_s+dinov2_b+biomedclip+handcrafted+tda | 0.6073 | 0.4650 |
| uniform_subset_k4 | dinov2_s+dinov2_b+biomedclip+tda | 0.6060 | 0.4607 |
| uniform_subset_k3 | dinov2_s+dinov2_b+tda | 0.6050 | 0.4693 |
| uniform_subset_k2 | dinov2_s+dinov2_b | 0.6017 | 0.4700 |
| uniform_subset_k3 | dinov2_s+biomedclip+handcrafted | 0.5967 | 0.4562 |
| uniform_subset_k3 | dinov2_s+biomedclip+tda | 0.5961 | 0.4499 |
| uniform_subset_k3 | dinov2_b+biomedclip+tda | 0.5950 | 0.4482 |
| uniform_subset_k4 | dinov2_s+dinov2_b+handcrafted+tda | 0.5943 | 0.4570 |
| uniform_subset_k4 | dinov2_s+biomedclip+handcrafted+tda | 0.5857 | 0.4485 |
| uniform_subset_k4 | dinov2_b+biomedclip+handcrafted+tda | 0.5850 | 0.4429 |
| uniform_subset_k3 | dinov2_s+dinov2_b+handcrafted | 0.5787 | 0.4558 |
| uniform_subset_k2 | dinov2_b+handcrafted | 0.5762 | 0.4446 |
| uniform_subset_k2 | dinov2_s+handcrafted | 0.5736 | 0.4555 |
| uniform_subset_k3 | dinov2_s+handcrafted+tda | 0.5725 | 0.4416 |
| uniform_subset_k3 | dinov2_b+handcrafted+tda | 0.5713 | 0.4492 |
| uniform_subset_k2 | dinov2_b+tda | 0.5665 | 0.4369 |
| uniform_subset_k2 | biomedclip+tda | 0.5575 | 0.4145 |
| uniform_subset_k2 | dinov2_s+tda | 0.5553 | 0.4223 |
| uniform_subset_k2 | biomedclip+handcrafted | 0.5483 | 0.4042 |
| uniform_subset_k3 | biomedclip+handcrafted+tda | 0.5427 | 0.4093 |
| uniform_subset_k2 | handcrafted+tda | 0.4688 | 0.3480 |

## Post-threshold refinement (top 10 pre-threshold candidates)

| strategy | subset | pre-threshold F1 | post-threshold F1 |
|---|---|---|---|
| uniform_subset_k2 | dinov2_b+biomedclip | 0.6346 | 0.6698 |
| f1_weighted_top3 | dinov2_b+dinov2_s+biomedclip | 0.6285 | 0.6618 |
| uniform_subset_k4 | dinov2_s+dinov2_b+biomedclip+handcrafted | 0.6146 | 0.6587 |
| per_class_weighted | dinov2_s+dinov2_b+biomedclip+handcrafted+tda | 0.6474 | 0.6572 |
| geometric_mean | dinov2_s+dinov2_b+biomedclip+handcrafted+tda | 0.6155 | 0.6569 |
| uniform_subset_k3 | dinov2_s+dinov2_b+biomedclip | 0.6285 | 0.6564 |
| f1_weighted_top4 | dinov2_b+dinov2_s+biomedclip+handcrafted | 0.6305 | 0.6563 |
| uniform_avg | dinov2_s+dinov2_b+biomedclip+handcrafted+tda | 0.6108 | 0.6512 |
| uniform_subset_k3 | dinov2_b+biomedclip+handcrafted | 0.6133 | 0.6431 |
| uniform_subset_k2 | dinov2_s+biomedclip | 0.6286 | 0.6384 |

## Best configuration

- **Strategy:** `uniform_subset_k2`
- **Components:** `dinov2_b+biomedclip`
- **Pre-threshold weighted F1:** 0.6346
- **Post-threshold weighted F1:** 0.6698
- **Per-class thresholds:** ZdraviLudia=0.95, Diabetes=1.00, PGOV_Glaukom=1.00, SklerozaMultiplex=0.85, SucheOko=1.00

### Per-class F1 (final best, post-threshold)

| class | F1 | support |
|---|---|---|
| ZdraviLudia | 0.8649 | 70 |
| Diabetes | 0.4286 | 25 |
| PGOV_Glaukom | 0.5915 | 36 |
| SklerozaMultiplex | 0.7179 | 95 |
| SucheOko | 0.0000 | 14 |

### Confusion matrix (rows=true, cols=pred)

| true\pred | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|---|---|---|---|---|
| ZdraviLudia | 64 | 5 | 0 | 1 | 0 |
| Diabetes | 11 | 9 | 0 | 2 | 3 |
| PGOV_Glaukom | 0 | 0 | 21 | 15 | 0 |
| SklerozaMultiplex | 1 | 3 | 14 | 70 | 7 |
| SucheOko | 2 | 0 | 0 | 12 | 0 |

## Artifacts

- `cache/best_ensemble_predictions.npz` — keys: `proba`, `pred_label`, `true_label`, `scan_paths`, `thresholds`
- `scripts/prob_ensemble.py` — runnable end-to-end, caches only.

## Method summary

1. Load 5 cached components and aggregate to scan level (mean-pool tiles for neural embeddings; tabular features used as-is).
2. For each component, run **person-level LOPO** with `StandardScaler + LogisticRegression(balanced, 3000 iters)`, collecting per-scan OOF `predict_proba` matrices.
3. Compare ensemble strategies: uniform average, validation-F1 weighted, per-class Dirichlet-sampled weighting, geometric mean, stacking via meta-LR, and all 2-/3-/4-component subset uniforms.
4. On top 10 candidates, perform greedy per-class threshold sweep in [0.05, 0.95] maximizing weighted F1. Apply winner to ensemble `proba` via `argmax(P / thresholds)`.
