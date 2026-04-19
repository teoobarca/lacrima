# Fusion Ensemble — v4 + k-NN + XGBoost (Clean, VLM-free)

**Question:** can combining three clean (non-VLM) OOF tracks — v4 multiscale (0.6887), DINOv2 k-NN retrieval (0.6117), XGBoost on 440 handcrafted features — push past v4's weighted F1 under person-LOPO?

**VLM contamination guard:** no Haiku/Sonnet/Opus features are used. Wave-14 filename-leak finding explicitly disqualifies those tracks.

**Sample size:** 240 scans × 35 persons × 5 classes (person-LOPO).

## 1. Component baselines

| Model | Weighted F1 | Macro F1 |
|---|---:|---:|
| v4_multiscale | 0.6887 | 0.5541 |
| knn_dinov2_k1_majority | 0.6117 | 0.4864 |
| knn_dinov2_k5_softw | 0.5859 | 0.4396 |
| xgb_handcrafted | 0.5458 | 0.3856 |

## 2. Fusion results (person-LOPO)

| Strategy | wF1 | mF1 | Accuracy | Δ vs v4 | 95% CI | P(Δ>0) |
|---|---:|---:|---:|---:|---|---:|
| 3way_hardknn_geo_mean_equal | 0.6117 | 0.4864 | 0.6167 | -0.0767 | [-0.1330, -0.0185] | 0.003 |
| 3way_hardknn_geo_mean_f1w | 0.6117 | 0.4864 | 0.6167 | -0.0767 | [-0.1330, -0.0185] | 0.003 |
| 3way_hardknn_arith_f1w | 0.6229 | 0.4870 | 0.6417 | -0.0663 | [-0.1082, -0.0230] | 0.002 |
| 3way_hardknn_lr_stacker | 0.6109 | 0.5065 | 0.5917 | -0.0782 | [-0.1415, -0.0201] | 0.006 |
| 3way_hardknn_class_routing | 0.6253 | 0.4888 | 0.6458 | -0.0638 | [-0.1068, -0.0177] | 0.001 |
| 3way_softknn_geo_mean_equal | 0.6241 | 0.4688 | 0.6500 | -0.0641 | [-0.1094, -0.0237] | 0.001 |
| 3way_softknn_geo_mean_f1w | 0.6407 | 0.4924 | 0.6625 | -0.0478 | [-0.0892, -0.0100] | 0.007 |
| 3way_softknn_arith_f1w | 0.6359 | 0.4892 | 0.6542 | -0.0527 | [-0.0883, -0.0193] | 0.000 |
| 3way_softknn_lr_stacker | 0.6120 | 0.5094 | 0.6000 | -0.0772 | [-0.1372, -0.0179] | 0.006 |
| 3way_softknn_class_routing | 0.6359 | 0.4892 | 0.6542 | -0.0527 | [-0.0883, -0.0193] | 0.000 |
| 2way_v4xgb_geo_mean_equal | 0.6299 | 0.4715 | 0.6583 | -0.0593 | [-0.1054, -0.0178] | 0.003 |
| 2way_v4xgb_geo_mean_f1w | 0.6518 | 0.5069 | 0.6708 | -0.0374 | [-0.0746, -0.0029] | 0.017 |
| 2way_v4xgb_arith_f1w | 0.6559 | 0.5125 | 0.6708 | -0.0336 | [-0.0643, -0.0073] | 0.008 |
| 2way_v4xgb_lr_stacker | 0.6065 | 0.5079 | 0.5875 | -0.0823 | [-0.1425, -0.0243] | 0.002 |
| 2way_v4xgb_class_routing | 0.6613 | 0.5207 | 0.6750 | -0.0280 | [-0.0554, -0.0031] | 0.015 |
| 2way_v4knn_geo_mean_equal | 0.6523 | 0.5150 | 0.6667 | -0.0368 | [-0.0672, -0.0104] | 0.008 |
| 2way_v4knn_geo_mean_f1w | 0.6540 | 0.5161 | 0.6667 | -0.0350 | [-0.0645, -0.0077] | 0.008 |
| 2way_v4knn_arith_f1w | 0.6495 | 0.5114 | 0.6625 | -0.0395 | [-0.0686, -0.0153] | 0.003 |
| 2way_v4knn_lr_stacker | 0.6024 | 0.5217 | 0.5833 | -0.0876 | [-0.1456, -0.0317] | 0.001 |
| 2way_v4knn_class_routing | 0.6583 | 0.5214 | 0.6708 | -0.0306 | [-0.0564, -0.0086] | 0.007 |
| **v4 (reference)** | **0.6887** | **0.5541** | — | — | — | — |

## 3. Winner: `2way_v4xgb_class_routing`

- **Weighted F1:** 0.6613
- **Macro F1:** 0.5207
- **Δ vs v4:** -0.0274
- **P(Δ > 0):** 0.015

### Per-class F1 (winner)

| Class | F1 |
|---|---:|
| ZdraviLudia | 0.8859 |
| Diabetes | 0.5000 |
| PGOV_Glaukom | 0.5333 |
| SklerozaMultiplex | 0.6842 |
| SucheOko | 0.0000 |

## 4. Method detail

- **`3way_hardknn_geo_mean_equal`** — [3way_hardknn] geometric mean, equal weights (proven v2 pattern)
- **`3way_hardknn_geo_mean_f1w`** — [3way_hardknn] geometric mean, F1-weighted exponents ([0.373, 0.331, 0.296])
- **`3way_hardknn_arith_f1w`** — [3way_hardknn] arithmetic mean, weights=wF1 ([0.373, 0.331, 0.296])
- **`3way_hardknn_lr_stacker`** — [3way_hardknn] LR stacker, NESTED LOPO (no stacker-fold leakage)
- **`3way_hardknn_class_routing`** — [3way_hardknn] per-class weights from inner-fold per-class F1 (nested LOPO)
- **`3way_softknn_geo_mean_equal`** — [3way_softknn] geometric mean, equal weights (proven v2 pattern)
- **`3way_softknn_geo_mean_f1w`** — [3way_softknn] geometric mean, F1-weighted exponents ([0.378, 0.322, 0.3])
- **`3way_softknn_arith_f1w`** — [3way_softknn] arithmetic mean, weights=wF1 ([0.378, 0.322, 0.3])
- **`3way_softknn_lr_stacker`** — [3way_softknn] LR stacker, NESTED LOPO (no stacker-fold leakage)
- **`3way_softknn_class_routing`** — [3way_softknn] per-class weights from inner-fold per-class F1 (nested LOPO)
- **`2way_v4xgb_geo_mean_equal`** — [2way_v4xgb] geometric mean, equal weights (proven v2 pattern)
- **`2way_v4xgb_geo_mean_f1w`** — [2way_v4xgb] geometric mean, F1-weighted exponents ([0.558, 0.442])
- **`2way_v4xgb_arith_f1w`** — [2way_v4xgb] arithmetic mean, weights=wF1 ([0.558, 0.442])
- **`2way_v4xgb_lr_stacker`** — [2way_v4xgb] LR stacker, NESTED LOPO (no stacker-fold leakage)
- **`2way_v4xgb_class_routing`** — [2way_v4xgb] per-class weights from inner-fold per-class F1 (nested LOPO)
- **`2way_v4knn_geo_mean_equal`** — [2way_v4knn] geometric mean, equal weights (proven v2 pattern)
- **`2way_v4knn_geo_mean_f1w`** — [2way_v4knn] geometric mean, F1-weighted exponents ([0.54, 0.46])
- **`2way_v4knn_arith_f1w`** — [2way_v4knn] arithmetic mean, weights=wF1 ([0.54, 0.46])
- **`2way_v4knn_lr_stacker`** — [2way_v4knn] LR stacker, NESTED LOPO (no stacker-fold leakage)
- **`2way_v4knn_class_routing`** — [2way_v4knn] per-class weights from inner-fold per-class F1 (nested LOPO)

## 5. Red-team self-check on leakage

- **v4 probs** come from `StratifiedGroupKFold(groups=person_id)`; each scan's probability is produced by a fold where its person was in the val set.
- **k-NN probs** mask out ALL scans sharing the query's person before voting (see `scripts/knn_baseline.py`, `mask_same_person`).
- **XGBoost probs** come from `StratifiedGroupKFold(groups=person_id)` in `scripts/expert_council.py::xgboost_oof`; no same-person leakage.

- **Geometric mean / weighted arithmetic mean** require NO training — purely deterministic transforms of already-OOF probs. No leakage possible.

- **Logistic-regression stacker** uses **NESTED LOPO**: for every held-out person, the stacker is fit on rows belonging to *other* persons (whose probs are themselves already person-LOPO). The held-out person's row is NEVER used in any stacker training step. This defuses the "stacker sees its own eval fold" trap flagged in the task spec.

- **Class-specific routing** computes per-class F1 weights using only rows from persons other than the held-out person. The held-out person's fused probs depend only on inner-fold diagnostics.

- **Possible residual leakage:** k-NN and v4 both use DINOv2 embeddings. They are therefore correlated, which weakens diversity gains — this is an *efficacy* concern, not a *leakage* concern.

## 6. Verdict

**Fusion hurts.** Best candidate (`2way_v4xgb_class_routing`, wF1 = 0.6613) trails v4 by 0.0274 weighted F1 (P(Δ>0) = 0.015). **Keep v4 alone as the shipped champion.** The three clean components are too correlated (v4 ensemble already contains DINOv2, and k-NN rides on the same DINOv2 embedding) and the XGBoost head is substantially weaker than v4. Adding any of them to v4 injects more noise than signal. A genuinely orthogonal track (CGNN, TDA persistent homology, physics-informed prior) is required to break past v4.

## 7. Files

- `cache/fusion_ensemble_predictions.json` — all fusion probs + winner preds
- `reports/FUSION_ENSEMBLE.md` — this file
- `scripts/fusion_ensemble.py` — reproduction script
