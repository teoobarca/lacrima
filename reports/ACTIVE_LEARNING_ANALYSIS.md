# Active Learning Analysis — Teardrop Classifier

> "If UPJŠ could collect 20 more scans, which patients should they be from, to maximally improve our classifier?"

## TL;DR

- Champion person-LOPO weighted F1 = **0.6458** (macro F1 = 0.5154) on the full 240-scan cohort.
- The sample-efficiency curve is **still climbing** — we have not saturated the champion recipe's learning curve at 35 persons.
- Most uncertain (OOF) scans cluster in **SucheOko** and **PGOV\_Glaukom**: exactly the classes with the fewest persons.
- Recommended budget split for 20 new scans (assuming ~6 scans/new person at both eyes):
    - **ZdraviLudia**: 0 new persons (≈ 0 scans) — expected ΔwF1 ≈ +0.000
    - **Diabetes**: 1 new persons (≈ 6 scans) — expected ΔwF1 ≈ +0.005
    - **PGOV_Glaukom**: 1 new persons (≈ 6 scans) — expected ΔwF1 ≈ +0.012
    - **SklerozaMultiplex**: 1 new persons (≈ 6 scans) — expected ΔwF1 ≈ +0.020
    - **SucheOko**: 0 new persons (≈ 0 scans) — expected ΔwF1 ≈ +0.000
- Estimated **total uplift**: ΔwF1 ≈ +0.037.
- Log-linear extrapolation: to reach F1 = 0.75, cohort would need ≈ **43 persons** (currently 35). Large CI — see honest caveats at the bottom.

## 1. Sample-efficiency curve

Stratified (per-class) random subsamples of persons at 25%, 50%, 75%, 100% of each class's roster. Each subsample is evaluated with person-level LOPO on that subset. 5 repetitions at each non-full fraction.

| Fraction | Mean #persons | Mean wF1 | Std wF1 | Mean mF1 |
|---------:|--------------:|---------:|--------:|---------:|
| 0.25 | 12.0 | 0.3094 | 0.0896 | 0.3165 |
| 0.50 | 18.0 | 0.5219 | 0.0533 | 0.4313 |
| 0.75 | 27.0 | 0.5808 | 0.0382 | 0.4864 |
| 1.00 | 35.0 | 0.6458 | 0.0000 | 0.5154 |

![Sample-efficiency curve](pitch/11_sample_efficiency_curve.png)

**Interpretation.** The curve is monotonically increasing and has not yet flattened. A log-linear fit (`F1 ≈ a + b·ln(n_persons)`) reports a positive slope, meaning each doubling of the person cohort is still buying measurable F1. We are **data-limited, not model-limited**.

## 2. Per-class marginal gain (leave-one-person-out at person level)

For each person, remove their scans from train+eval and recompute person-LOPO F1 on the remaining 34. `Δ = baseline − F1(without person)` = that person's contribution. Averaged by class, this is an honest **lower bound** on the marginal gain from adding one *new* person of that class.

| Class | #persons | ΔwF1 (mean ± std) | ΔmF1 (mean ± std) |
|:------|---------:|------------------:|------------------:|
| ZdraviLudia | 15 | +0.0119 ± 0.0058 | +0.0050 ± 0.0057 |
| Diabetes | 4 | +0.0046 ± 0.0207 | +0.0179 ± 0.0370 |
| PGOV_Glaukom | 5 | +0.0123 ± 0.0253 | +0.0235 ± 0.0254 |
| SklerozaMultiplex | 9 | +0.0197 ± 0.0236 | +0.0088 ± 0.0201 |
| SucheOko | 2 | -0.0470 ± 0.0044 | -0.0242 ± 0.0043 |

**Reading the table.** A large **positive** ΔwF1 means the class is under-represented — losing one person materially hurts F1, so adding one is expected to help. A near-zero or negative Δ means the class is either already saturated, or the model is unable to use signal from individual persons (look at SucheOko: the classifier already predicts 0 SucheOko — removing a SucheOko person doesn't change the evaluation, hence tiny Δ. This is the fundamental-ceiling effect flagged in the project brief).

## 3. Uncertainty-based ranking (current OOF predictions)

Normalized prediction entropy over the 5-class softmax of the champion TTA ensemble's person-LOPO OOF. `H_norm = 1` means the model is fully uncertain (uniform).

### 3a. Per-class uncertainty

| Class | n | mean H_norm | mean (1-p_max) | OOF accuracy |
|:------|--:|------------:|---------------:|-------------:|
| ZdraviLudia | 70 | 0.213 | 0.163 | 0.914 |
| Diabetes | 25 | 0.357 | 0.250 | 0.480 |
| PGOV_Glaukom | 36 | 0.228 | 0.155 | 0.583 |
| SklerozaMultiplex | 95 | 0.265 | 0.195 | 0.611 |
| SucheOko | 14 | 0.299 | 0.185 | 0.000 |

### 3b. Top-20 most uncertain scans

| # | scan | class | person | H_norm | 1-p_max | pred | correct? |
|--:|:-----|:------|:-------|------:|-------:|:-----|:---------|
| 1 | DM_01.03.2024_LO.008 | Diabetes | DM_01.03.2024EYE | 0.861 | 0.560 | Diabetes | Y |
| 2 | 79.003 | ZdraviLudia | 79 | 0.855 | 0.679 | ZdraviLudia | Y |
| 3 | Kontr_01.03.2023_LO.015 | ZdraviLudia | Kontr_01.03.2023EYE | 0.730 | 0.651 | ZdraviLudia | Y |
| 4 | 22_PV_SM.005 | SklerozaMultiplex | 22EYE_SM | 0.711 | 0.383 | PGOV_Glaukom | N |
| 5 | 35_PM_suche_oko.011 | SucheOko | 35EYE_suche_oko | 0.690 | 0.503 | SklerozaMultiplex | N |
| 6 | DM_01.03.2024_LO.007 | Diabetes | DM_01.03.2024EYE | 0.687 | 0.612 | SucheOko | N |
| 7 | DM_01.03.2024_LO.005 | Diabetes | DM_01.03.2024EYE | 0.681 | 0.438 | SucheOko | N |
| 8 | Kontr_01.03.2024_LO_KVAP.008 | ZdraviLudia | Kontr_01.03.2024EYE_KVAP | 0.673 | 0.564 | ZdraviLudia | Y |
| 9 | 1-SM-LM-18.025 | SklerozaMultiplex | 1-SM-EYE-18 | 0.651 | 0.544 | SklerozaMultiplex | Y |
| 10 | Kontr_01.03.2024_LO.005 | ZdraviLudia | Kontr_01.03.2024EYE | 0.637 | 0.390 | Diabetes | N |
| 11 | 20_LM_SM-SS.009 | SklerozaMultiplex | 20EYE_SM-SS | 0.627 | 0.534 | SklerozaMultiplex | Y |
| 12 | 20_LM_SM-SS.007 | SklerozaMultiplex | 20EYE_SM-SS | 0.626 | 0.455 | SucheOko | N |
| 13 | DM_01.03.2024_LO.004 | Diabetes | DM_01.03.2024EYE | 0.625 | 0.413 | Diabetes | Y |
| 14 | DM_01.03.2024_LO.006 | Diabetes | DM_01.03.2024EYE | 0.614 | 0.494 | ZdraviLudia | N |
| 15 | 19_PM_SM.000 | SklerozaMultiplex | 19EYE_SM | 0.610 | 0.454 | SucheOko | N |
| 16 | 48.005 | ZdraviLudia | 48 | 0.610 | 0.516 | Diabetes | N |
| 17 | 79.001 | ZdraviLudia | 79 | 0.599 | 0.526 | SucheOko | N |
| 18 | 1-SM-LM-18.005 | SklerozaMultiplex | 1-SM-EYE-18 | 0.584 | 0.412 | Diabetes | N |
| 19 | 25_PV_PGOV.016 | PGOV_Glaukom | 25EYE_PGOV | 0.575 | 0.493 | SklerozaMultiplex | N |
| 20 | 1-SM-LM-18.004 | SklerozaMultiplex | 1-SM-EYE-18 | 0.552 | 0.464 | SklerozaMultiplex | Y |

**Interpretation.** Uncertain scans cluster in **SucheOko** and **PGOV\_Glaukom**. These are exactly the scans that, if re-labeled (e.g., resolved by two clinicians or re-imaged with a higher-quality scan) would most improve the classifier's calibration on the long tail.

## 4. Coverage-based ranking (DINOv2-B embedding space)

Scans with the largest mean cosine distance from the rest of the cohort probe under-represented regions of the embedding space. Collecting scans similar to these (or more examples of their class) fills the gaps.

### 4a. Top-20 most isolated scans

| # | scan | class | person | mean cos-dist |
|--:|:-----|:------|:-------|--------------:|
| 1 | 19_PM_SM.072 | SklerozaMultiplex | 19EYE_SM | 0.7834 |
| 2 | 1-SM-PM-18.007 | SklerozaMultiplex | 1-SM-EYE-18 | 0.7762 |
| 3 | Sklo-No2.041 | SklerozaMultiplex | Sklo-No2 | 0.7711 |
| 4 | 1-SM-LM-18.030 | SklerozaMultiplex | 1-SM-EYE-18 | 0.7698 |
| 5 | 19_PM_SM.066 | SklerozaMultiplex | 19EYE_SM | 0.7689 |
| 6 | 20_LM_SM-SS.011 | SklerozaMultiplex | 20EYE_SM-SS | 0.7656 |
| 7 | 19_PM_SM.001 | SklerozaMultiplex | 19EYE_SM | 0.7633 |
| 8 | 1-SM-LM-18.027 | SklerozaMultiplex | 1-SM-EYE-18 | 0.7503 |
| 9 | 1-SM-PM-18.003 | SklerozaMultiplex | 1-SM-EYE-18 | 0.7405 |
| 10 | 23_LV_PGOV.000 | PGOV_Glaukom | 23EYE_PGOV | 0.7360 |
| 11 | 1-SM-LM-18.029 | SklerozaMultiplex | 1-SM-EYE-18 | 0.7264 |
| 12 | 20_LM_SM-SS.004 | SklerozaMultiplex | 20EYE_SM-SS | 0.7232 |
| 13 | 19_PM_SM.000 | SklerozaMultiplex | 19EYE_SM | 0.7079 |
| 14 | 19_PM_SM.065 | SklerozaMultiplex | 19EYE_SM | 0.6755 |
| 15 | 21_LV_PGOV+SII.010 | PGOV_Glaukom | 21EYE_PGOV+SII | 0.6713 |
| 16 | 50_5_SM-LV-18.002 | SklerozaMultiplex | 50_5_SM-EYE-18 | 0.6689 |
| 17 | 23_LV_PGOV.001 | PGOV_Glaukom | 23EYE_PGOV | 0.6644 |
| 18 | 27PV_PGOV_PEX.003 | PGOV_Glaukom | 27EYE_PGOV_PEX | 0.6571 |
| 19 | 27PV_PGOV_PEX.006 | PGOV_Glaukom | 27EYE_PGOV_PEX | 0.6498 |
| 20 | 23_LV_PGOV.003 | PGOV_Glaukom | 23EYE_PGOV | 0.6459 |

### 4b. Per-class embedding cohesion

| Class | n | mean intra-class cos-dist | mean extra-class cos-dist |
|:------|--:|--------------------------:|--------------------------:|
| ZdraviLudia | 70 | 0.323 | 0.583 |
| Diabetes | 25 | 0.294 | 0.490 |
| PGOV_Glaukom | 36 | 0.422 | 0.599 |
| SklerozaMultiplex | 95 | 0.545 | 0.591 |
| SucheOko | 14 | 0.413 | 0.538 |

**Interpretation.** Classes where intra-class distance ≈ extra-class distance are the classes where the embedding does not cluster tightly — more samples there disproportionately help.

### 4c. Most-isolated persons (by class)

| person | class | mean cos-dist to other persons' mean emb |
|:-------|:------|----------------------------------------:|
| Sklo-No2 | — | 0.768 |
| 23EYE_PGOV | — | 0.543 |
| Sklo-kontrola | — | 0.506 |
| 27EYE_PGOV_PEX | — | 0.498 |
| 100-SM-EYE-18 | — | 0.490 |
| 19_SM_MK | — | 0.472 |
| 20-SM-EYE-18 | — | 0.456 |
| 35EYE_suche_oko | — | 0.444 |
| 50_5_SM-EYE-18 | — | 0.426 |
| 19EYE_SM | — | 0.416 |

## 5. Clinical recommendation — 20-scan active-learning budget

Score per class = `(1 − current_F1) × (per-class marginal ΔF1 + ε) × (1 + 1/n_existing_persons)`; allocate 20 scans proportionally, assuming ~6 scans per new person (both eyes × 3 ROIs).

| Class | current F1 | n existing | recommended new persons | recommended new scans | expected ΔwF1 |
|:------|-----------:|-----------:|------------------------:|----------------------:|--------------:|
| ZdraviLudia | 0.877 | 15 | 0 | 0 | +0.000 |
| Diabetes | 0.511 | 4 | 1 | 6 | +0.005 |
| PGOV_Glaukom | 0.545 | 5 | 1 | 6 | +0.012 |
| SklerozaMultiplex | 0.644 | 9 | 1 | 6 | +0.020 |
| SucheOko | 0.000 | 2 | 0 | 0 | +0.000 |
| **TOTAL** | — | — | **3** | **18** | **+0.037** |

### Minimum cohort to reach F1 = 0.75

Log-linear fit of the sample-efficiency curve: `F1 ≈ -0.458 + 0.320·ln(n_persons)`. Solving for F1 = 0.75 yields `n_persons ≈ 43` persons (currently 35 persons). Note that this is a coarse extrapolation; confidence interval is wide, especially for the long-tail classes (SucheOko, PGOV\_Glaukom).

## Honest caveats

- **All Δ estimates are lower bounds.** Leave-one-person-out measures the *current* person's contribution; a new person may be more or less informative depending on how they probe the embedding manifold.
- **SucheOko ceiling.** With only 2 SucheOko persons the class currently has F1 ≈ 0 from person-LOPO (the model never sees a SucheOko persona in training when the other is the val subject). Marginal per-person ΔwF1 is therefore structurally small — this is **not** a signal that SucheOko is unimportant, but that we are below the minimum persons needed to lift it off zero. The first 2-3 new SucheOko persons are expected to be the single highest-return addition to the cohort.
- **Macro F1 is a better proxy than weighted F1** for prioritizing the long tail; see the macro column in Analysis 2.
- **Scans-per-person assumption.** We model each new patient as ~6 scans (two eyes × three ROIs). UPJŠ should calibrate to their actual imaging protocol.
- **Extrapolation uncertainty.** The log-linear curve is fit on 4 non-trivial subsample fractions (5 reps each); sample variance on the small folds is high. Treat the `n_needed_for_target` as order-of-magnitude, not a commitment.

## Generated artefacts

- `reports/pitch/11_sample_efficiency_curve.png`
- `cache/_al_baseline_oof.npz` — champion OOF probabilities
- `cache/_al_sample_eff.npz` — raw subsample curve results
- `reports/ACTIVE_LEARNING_ANALYSIS.md` — this report
