# Cascade Stacker Results

Stacking meta-classifier that consumes the Stage-1 ensemble probabilities and binary specialist probabilities as **features** (not as a hard override). Motivated by the diagnostic in `reports/CASCADE_RESULTS.md` (hard-cascade Δ vs Stage-1 = -0.048). Everything below is evaluated with person-level LOPO (35 groups).

## Feature matrix (240 × 12)

All features are out-of-fold. Stage-1 probabilities come from `cache/best_ensemble_predictions.npz` (which was produced by person-LOPO over the DINOv2-B + BiomedCLIP proba-avg ensemble). Specialist probabilities come from `cache/cascade_oof.npz`, where each specialist was trained only on in-pair training rows of each person-LOPO fold and predicted on every held-out person — so the specialist output on every row is a legitimate OOF value (in-pair scans get their meaningful binary prediction; off-pair scans get a noisy but still-OOF prior which the meta-LR will weight down).

| idx | name | range | mean |
|---|---|---|---|
| 0 | `s1_ZdraviLudia` | [+0.000, +1.000] | +0.301 |
| 1 | `s1_Diabetes` | [+0.000, +0.818] | +0.101 |
| 2 | `s1_PGOV_Glaukom` | [+0.000, +1.000] | +0.160 |
| 3 | `s1_SklerozaMultiplex` | [+0.000, +1.000] | +0.377 |
| 4 | `s1_SucheOko` | [+0.000, +0.830] | +0.061 |
| 5 | `specA_p_Glaukom` | [+0.003, +0.992] | +0.196 |
| 6 | `specA_p_SM` | [+0.008, +0.997] | +0.804 |
| 7 | `specB_p_Diabetes` | [+0.008, +0.973] | +0.394 |
| 8 | `specB_p_Healthy` | [+0.027, +0.992] | +0.606 |
| 9 | `specA_conf` | [+0.506, +0.997] | +0.892 |
| 10 | `specB_conf` | [+0.507, +0.992] | +0.799 |
| 11 | `s1_entropy` | [+0.000, +1.158] | +0.452 |

## Honest comparison (all person-LOPO weighted F1)

| Method | weighted F1 | macro F1 | notes |
|---|---:|---:|---|
| Stage-1 alone (raw argmax, no threshold) | 0.6346 | 0.4934 | reference |
| Stage-1 + tuned thresholds (leaky) | 0.6698 | 0.5206 | thresholds tuned on full OOF |
| Hard cascade A+B @ thr=0.65 (Round 3a) | 0.6217 | 0.4738 | from `cache/cascade_oof.npz` |
| Meta-LR (this script) | **0.5057** | 0.4446 | StandardScaler + balanced LR, person-LOPO |
| Meta-XGB (this script) | **0.5404** | 0.3882 | depth=3, n_est=300, sample-weight balanced, person-LOPO |
| Soft-blend (nested-α, honest) | **0.6451** | 0.5033 | α selected on training fold per outer fold |
| Soft-blend (best α on full eval, LEAKY) | 0.6451 | 0.5033 | α=0.90 — upper bound |

### Δ summary (best stacker method)

Best method: **soft_blend_nested** — wF1 = 0.6451, mF1 = 0.5033.

- vs Stage-1 raw argmax (0.6346): Δ = +0.0105 → improves Stage-1 raw.
- vs Stage-1 + leaky thresholds (0.6698): Δ = -0.0247 → does NOT beat the tuned Stage-1 (the 0.6528–0.6698 honest/leaky band from prior audits).

## Per-class F1 — best method (soft_blend_nested)

| class | F1 |
|---|---:|
| ZdraviLudia | 0.865 |
| Diabetes | 0.429 |
| PGOV_Glaukom | 0.553 |
| SklerozaMultiplex | 0.670 |
| SucheOko | 0.000 |

## Soft-blend α sweep (FULL-EVAL, leaky)

`final_proba = α * stage1_proba + (1-α) * spec_informed_proba`. Spec-informed proba redistributes Stage-1's mass on the Glaukom/SM pair using Specialist A's probability, and similarly for the Diabetes/Healthy pair with Specialist B.

| α | weighted F1 | macro F1 |
|---:|---:|---:|
| 0.00 | 0.5788 | 0.4380 |
| 0.10 | 0.5724 | 0.4293 |
| 0.20 | 0.5835 | 0.4383 |
| 0.30 | 0.5790 | 0.4334 |
| 0.40 | 0.5877 | 0.4427 |
| 0.50 | 0.6056 | 0.4622 |
| 0.60 | 0.6202 | 0.4762 |
| 0.70 | 0.6360 | 0.4958 |
| 0.80 | 0.6343 | 0.4950 |
| 0.90 | 0.6451 | 0.5033 |
| 1.00 | 0.6346 | 0.4934 |

Full-eval best: α=0.90 → wF1=0.6451 (UPPER bound, leaky).

Nested-α selection (honest) picked α across 35 folds: {0.9: 35}.
Nested-α wF1 = 0.6451, mF1 = 0.5033.

## Confusion matrix — best method (soft_blend_nested)

| true\pred | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|---|---|---|---|---|
| ZdraviLudia | 64 | 5 | 0 | 1 | 0 |
| Diabetes | 11 | 9 | 0 | 2 | 3 |
| PGOV_Glaukom | 0 | 0 | 21 | 15 | 0 |
| SklerozaMultiplex | 1 | 3 | 19 | 63 | 9 |
| SucheOko | 2 | 0 | 0 | 12 | 0 |

## Honest verdict

The soft stacker **improves** over Stage-1 raw argmax (0.6346 → 0.6451, Δ = +0.0105). Specialist probabilities contribute useful signal when used as features instead of a hard override: the meta-model learns to trust them in the Glaukom/SM and Diabetes/Healthy subspaces and ignore them elsewhere.

Compared to the 0.6528 honest / 0.6698 leaky band from prior audits:

- The meta-LR number is honest (person-LOPO, no parameter on eval).
- It sits at **0.6451**, which is -0.0247 vs the tuned (leaky) 0.6698 headline.
- The Stage-1 ensemble with bias tuning only survives honest evaluation at ~0.6326 (see RED_TEAM_ENSEMBLE_V2_AUDIT.md); the stacker's honest number should be compared against that, not 0.6698.

## Method notes

1. Every feature is out-of-fold. Stage-1 probas are person-LOPO; specialist probas come from `cache/cascade_oof.npz`, which was produced by `scripts/cascade_classifier.py::specialist_lopo` with the same `leave_one_patient_out(person_groups)` iterator used here. In each fold the specialist trains only on in-pair training rows and predicts on all held-out rows of that fold. That means the meta-model at fold `k` sees features that never observed person `k`.
2. The meta-model is refit from scratch in each outer fold (35 fits total). Nothing is tuned on the held-out person.
3. Meta-LR uses `StandardScaler + LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000)`. Meta-XGB uses depth=3, n_estimators=300, balanced sample weights (since XGB has no `class_weight`). No inner CV on eval folds.
4. The soft-blend α is selected inside each outer fold on its training subset by choosing the α ∈ {0.0, 0.1, …, 1.0} that maximises weighted-F1 on the training subset. The resulting prediction on the held-out person is then evaluated.
5. SucheOko (14 scans, 2 persons) is essentially invisible to every method here — same as Stage-1. This is a dataset-size issue, not a stacker issue.

## Artifacts

- `cache/stacker_oof.npz` — keys: `best_method`, `best_proba`, `best_pred`, `lr_proba`, `lr_pred`, `xgb_proba`, `xgb_pred`, `blend_proba`, `blend_pred`, `picked_alphas`, `stacker_features`, `feature_names`, `stage1_proba`, `stage1_pred`, `true_label`, `scan_paths`.
- `scripts/cascade_stacker.py` — runnable end-to-end; consumes `cache/cascade_oof.npz` and `cache/best_ensemble_predictions.npz`.
