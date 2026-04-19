# Cascade Classifier Results

Confidence-gated cascade over Stage-1 ensemble (DINOv2-B + BiomedCLIP proba-avg + per-class thresholds).

Two (+ optional third) binary specialists trained person-LOPO:

- **Specialist A**: PGOV_Glaukom vs SklerozaMultiplex — features = DINOv2-B scan-mean || TDA (1015-dim) — XGBoost.
- **Specialist B**: Diabetes vs ZdraviLudia — features = DINOv2-B scan-mean || handcrafted (94-dim) — XGBoost.
- **(Bonus) Specialist C**: SklerozaMultiplex vs ZdraviLudia — features = DINOv2-B scan-mean || handcrafted — XGBoost.

Gating: if Stage-1 top-1 proba > `thr` keep Stage-1 prediction; otherwise if Stage-1 top-2 classes match a specialist's pair, route the scan to that specialist (which yields an out-of-fold binary prediction because it used the SAME person-LOPO splits).

## Specialist binary F1 (person-LOPO, restricted to pair scans)

| specialist | pair | n_scans | binary F1 | accuracy | per-class F1 |
|---|---|---|---|---|---|
| A | A (Glaukom vs SM) | 131 | 0.7831 | 0.6870 | PGOV_Glaukom=0.438, SklerozaMultiplex=0.783 |
| B | B (Diabetes vs Healthy) | 95 | 0.8108 | 0.7053 | Diabetes=0.333, ZdraviLudia=0.811 |
| C | C (SM vs Healthy) | 165 | 0.9577 | 0.9636 | SklerozaMultiplex=0.968, ZdraviLudia=0.958 |

## Stage-1 vs cascade variants (gating threshold = 0.65)

| variant | weighted F1 | macro F1 | F1 ZdraviLudia | F1 Diabetes | F1 PGOV_Glaukom | F1 SklerozaMultiplex | F1 SucheOko | routed A | routed B | routed C | total routed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Stage-1 baseline | 0.6698 | 0.5206 | 0.865 | 0.429 | 0.592 | 0.718 | 0.000 | - | - | - | - |
| stage1_only | 0.6698 | 0.5206 | 0.865 | 0.429 | 0.592 | 0.718 | 0.000 | 0 | 0 | 0 | 0 |
| stage1 + A (Glaukom/SM) | 0.6338 | 0.4893 | 0.865 | 0.429 | 0.486 | 0.667 | 0.000 | 24 | 0 | 0 | 24 |
| stage1 + B (Diabetes/Healthy) | 0.6577 | 0.5050 | 0.844 | 0.372 | 0.592 | 0.718 | 0.000 | 0 | 22 | 0 | 22 |
| stage1 + A + B | 0.6217 | 0.4738 | 0.844 | 0.372 | 0.486 | 0.667 | 0.000 | 24 | 22 | 0 | 46 |
| stage1 + A + B + C | 0.6130 | 0.4681 | 0.819 | 0.372 | 0.486 | 0.663 | 0.000 | 24 | 22 | 6 | 52 |

## Pair-level accuracy for routed confused pairs

| variant | Glaukom/SM pair F1 | Diabetes/Healthy pair F1 |
|---|---|---|
| stage1_only | 0.778 | 0.883 |
| stage1 + A (Glaukom/SM) | 0.723 | 0.883 |
| stage1 + B (Diabetes/Healthy) | 0.778 | 0.861 |
| stage1 + A + B | 0.723 | 0.861 |
| stage1 + A + B + C | 0.716 | 0.853 |

## Gating-threshold sweep (A+B cascade)

| threshold | weighted F1 | macro F1 | routed A | routed B | total routed |
|---|---|---|---|---|---|
| 0.50 | 0.6770 | 0.5272 | 1 | 1 | 2 |
| 0.60 | 0.6330 | 0.4846 | 20 | 17 | 37 |
| 0.65 | 0.6217 | 0.4738 | 24 | 22 | 46 |
| 0.70 | 0.6135 | 0.4663 | 28 | 24 | 52 |
| 0.75 | 0.6058 | 0.4581 | 32 | 29 | 61 |

Best threshold in sweep: **0.50** → weighted F1 = **0.6770** (honest caveat: this value is tuned on eval — do not claim as headline).

## Double-gated cascade (S1 thr=0.65 + specialist conf threshold)

Route only when Stage-1 is uncertain AND the specialist is confident in its binary call. This reduces the bad-change rate.

| spec_thr | routed A | routed B | weighted F1 | macro F1 |
|---|---|---|---|---|
| 0.6 | 19 | 19 | 0.6380 | 0.4887 |
| 0.7 | 13 | 17 | 0.6526 | 0.5007 |
| 0.8 | 10 | 14 | 0.6558 | 0.5041 |
| 0.9 | 5 | 8 | 0.6731 | 0.5242 |

Best double-gate setting: `spec_thr=0.9` → weighted F1 = **0.6731** (Δ = +0.0033 vs Stage-1). Also eval-tuned, but shows that a confidence-weighted merge (rather than a hard override) is the promising direction.

## Honest headline

- **Stage-1 baseline weighted F1 (person-LOPO):** 0.6698
- **Cascade A+B (thr=0.65, honest — no tuning on eval) weighted F1:** 0.6217
- **Δ vs Stage-1:** -0.0482  → cascade does NOT beat Stage-1.

> **Why it may not help much:** The Stage-1 ensemble already puts very high confidence (>0.65) on easy scans; the confused pairs are confused precisely because the ensemble has *low* confidence there, and the specialists face the same underlying feature-vs-label difficulty. Adding TDA/handcrafted features does yield somewhat complementary signal, but on such a small dataset the specialists are themselves noisy.

## Diagnostic: why routing hurts at thr=0.65

- **Specialist A (Glaukom/SM)**: routed n=24, Stage-1 was already correct on 18/24 (75%). Cascade made 11 changes: **1 good** (fixed a mistake) vs **10 bad** (broke a correct). Net -9.
- **Specialist B (Diabetes/Healthy)**: routed n=22, Stage-1 was already correct on 15/22 (68%). Cascade made 5 changes: **1 good** (fixed a mistake) vs **4 bad** (broke a correct). Net -3.

**Key insight:** low Stage-1 confidence does NOT imply Stage-1 is wrong. The ensemble produces naturally flatter posteriors on confused pairs, but its 5-class argmax still beats a 2-class specialist trained on the same features. The specialist loses the "definitely-not-SucheOko/Diabetes" signal that the full ensemble carries.

**What might still work** (not fully explored, out of time budget):

1. Soft cascade — blend Stage-1 proba with specialist proba (weighted average) instead of hard override.
2. Specialists that additionally take Stage-1 proba as a feature (stacking-meta, pair-restricted).
3. Abstain-then-route — the double-gated table above hints this direction with a tiny positive delta, but it is eval-tuned.

## Baseline confusion matrix (Stage-1)

| true\pred | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|---|---|---|---|---|
| ZdraviLudia | 64 | 5 | 0 | 1 | 0 |
| Diabetes | 11 | 9 | 0 | 2 | 3 |
| PGOV_Glaukom | 0 | 0 | 21 | 15 | 0 |
| SklerozaMultiplex | 1 | 3 | 14 | 70 | 7 |
| SucheOko | 2 | 0 | 0 | 12 | 0 |

## Cascade A+B confusion matrix (thr=0.65)

| true\pred | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|---|---|---|---|---|
| ZdraviLudia | 62 | 7 | 0 | 1 | 0 |
| Diabetes | 12 | 8 | 0 | 2 | 3 |
| PGOV_Glaukom | 0 | 0 | 18 | 18 | 0 |
| SklerozaMultiplex | 1 | 3 | 20 | 64 | 7 |
| SucheOko | 2 | 0 | 0 | 12 | 0 |

## Artifacts

- `cache/cascade_oof.npz` — keys: `pred_label`, `true_label`, `scan_paths`, `stage1_pred`, `stage1_proba`, `gating_threshold`, `spec_A_proba_pos`, `spec_A_pred`, `spec_B_proba_pos`, `spec_B_pred`, `spec_C_proba_pos`, `spec_C_pred`.
- `scripts/cascade_classifier.py` — runnable end-to-end.

## Method notes

1. Specialists use XGBoost (`max_depth=4`, `n_estimators=200`, `lr=0.08`, `scale_pos_weight = #neg/#pos`). Fresh model per fold.
2. Every specialist fold trains only on in-pair training scans of that fold (excluding the held-out person). At prediction time we predict on ALL held-out scans, not just the in-pair ones, so the cascade can route any scan.
3. The cascade uses RAW Stage-1 proba (un-thresholded) to measure confidence; the fallback Stage-1 prediction is the thresholded argmax that produced the 0.67 baseline.
4. No tuning on eval for the headline: gating threshold is fixed at 0.65 (a Schelling point, not a sweep winner).
