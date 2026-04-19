# Advanced Texture Features — Results

_Person-level LOPO on TRAIN_SET (n=240 scans, 35 persons, 5 classes)._


## Methodology

We engineered 349 advanced texture features targeted at tear-crystallization
physics and concatenated them with the existing 94-dim handcrafted set to give
a combined feature bank of 443 dims. All evaluations are person-level
leave-one-out (35 persons → 35 folds).

- Handcrafted features (reused from `teardrop/features.py`): **94**
- Advanced features (`teardrop/features_advanced.py`): **349**
- Combined: **443**

Feature families implemented:
1. Multifractal spectrum f(α) via box-counting q-moments (q ∈ {-5,-3,-1,0,1,3,5})
2. Lacunarity (gliding box, Allain-Cloitre) at scales {4,8,16,32,64}
3. Succolarity (4 directional flood-permeabilities + summary)
4. Wavelet-packet tree energies (db4, level=3 → 64 subbands)
5. Hurst exponent (R/S) + DFA on row/col/flat signals + row-trend CV
6. Extended GLCM (8 distances × 8 angles × 6 Haralick props, summarized)
7. Gabor bank (8 orientations × 4 frequencies, mean/std/energy/entropy)
8. Multi-scale HOG (cells ∈ {8,16,32}, 9 orientations)

Feature selection is **nested** inside each LOPO fold (F-test or XGB
importance computed on TRAIN only, never on OOF). We sweep top-k ∈ {20, 50}.

## Results summary (weighted F1, person-LOPO)

| Experiment | n features | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| A1_handcrafted94_XGB_full | 94 | 0.4816 | 0.3312 |
| A2_advanced_XGB_full | 349 | 0.4813 | 0.3371 |
| A3_combined_XGB_full | 443 | 0.4757 | 0.3185 |
| B1_combined_top50_ftest_XGB | 443 | 0.4497 | 0.3011 |
| B2_combined_top50_xgbimp_XGB | 443 | 0.4312 | 0.2863 |
| C1_combined_top50_ftest_LR | 443 | 0.4154 | 0.3307 |
| B3_advanced_top50_ftest_XGB | 349 | 0.4435 | 0.3055 |
| B4_combined_top20_ftest_XGB | 443 | 0.4769 | 0.3289 |
| C2_combined_top20_ftest_LR | 443 | 0.4468 | 0.3553 |
| D1_top50adv_plus_DINOv2B_XGB | 1117 | 0.5711 | 0.4151 |
| D0_DINOv2B_only_XGB | 768 | 0.5888 | 0.4336 |
| D0b_DINOv2B_only_LR | 768 | 0.6150 | 0.4910 |
| D2_advanced_plus_DINOv2B_LR | 1117 | 0.6251 | 0.4956 |

## Comparison to baselines

| Baseline | Weighted F1 |
|---|---:|
| Handcrafted 94 (prior) | 0.490 |
| DINOv2-B alone (prior) | 0.615 |
| DINOv2-B+BiomedCLIP TTA ensemble (person-LOPO) | 0.6458 |
| **Best advanced experiment (D2_advanced_plus_DINOv2B_LR)** | **0.6251** |

Δ (D2) vs handcrafted-only: **+0.135**  (caveat: this gain is from DINOv2; adv features alone give 0.4813 ≈ handcrafted 0.4816)
Δ (D2) vs DINOv2-B-alone (LR): **+0.010**  (honest contribution of the advanced features on top of DINOv2)
Δ (D2) vs TTA ensemble (shipped): **-0.021**

## Top-30 global XGB feature importances (fit on full 240, all 349 features)

| Feature | Importance |
|---|---:|
| xglcm_homogeneity_d4 | 0.0368 |
| glcm_homogeneity_d5_mean | 0.0277 |
| xglcm_energy_d16 | 0.0163 |
| lbp_22 | 0.0153 |
| mhog_c8_p75 | 0.0151 |
| mhog_c16_min | 0.0123 |
| wp_energy_entropy | 0.0118 |
| mhog_c16_p90 | 0.0117 |
| wp_ahh_energy | 0.0113 |
| glcm_dissimilarity_d9_mean | 0.0110 |
| lbp_1 | 0.0103 |
| gabor_f3_o0_std | 0.0100 |
| lac_s16 | 0.0099 |
| gabor_f0_o1_mean | 0.0096 |
| xglcm_homogeneity_d32 | 0.0093 |
| fractal_D_std | 0.0090 |
| xglcm_homogeneity_all_std | 0.0082 |
| lbp_24 | 0.0076 |
| wp_hhv_energy | 0.0075 |
| glcm_correlation_d1_mean | 0.0072 |
| gabor_f0_o7_mean | 0.0072 |
| lbp_11 | 0.0069 |
| gabor_f1_o2_entropy | 0.0068 |
| xglcm_contrast_d4 | 0.0067 |
| wp_energy_max_frac | 0.0066 |
| lbp_14 | 0.0066 |
| gabor_f2_o4_mean | 0.0064 |
| xglcm_correlation_d8 | 0.0064 |
| hog_mean | 0.0063 |
| wp_dah_energy | 0.0062 |

_Important note_: these are fit on all data, so they only indicate which
features carry the most usable signal globally — they are NOT the selected
features during nested LOPO evaluation.

## Honest commentary

**Headline:** the advanced feature bank **does not improve the feature-only
classifier** (~0.48 weighted F1 with or without the new families), but when
**concatenated with DINOv2-B tiled embeddings** under a linear classifier it
gives a **small honest gain of +0.010 F1** (0.6251 vs 0.6150). This is still
**below the shipped TTA ensemble** (0.6458).

### Feature-only regime

| Regime | w-F1 |
|---|---:|
| Handcrafted 94 alone | 0.4816 |
| Advanced 349 alone   | 0.4813 |
| Combined 443 (full)  | 0.4757 |
| Combined top-50 F-test + XGB | 0.4497 |
| Combined top-50 XGB-imp + XGB | 0.4312 |
| Combined top-20 F-test + XGB | 0.4769 |
| Combined top-20 F-test + LR  | 0.4468 |

- Advanced features **alone give the same F1 as handcrafted-alone** — no real
  feature-level win, despite 4× more dimensions. The physical motivation
  (multifractal, lacunarity, Gabor, wavelet-packet) did not uncover a signal
  that the original GLCM + LBP + fractal + HOG did not already capture.
- **Nested feature selection (top-50 by any method) hurts**. F-test and
  XGB-importance both underperform the full-dimensional XGBoost. XGBoost is
  regularized enough to handle 443 raw features on 240 samples; pre-selection
  discards useful soft signal.
- Top-20 is slightly better than top-50 but still below the full set — consistent
  with "each individual feature has low signal; XGB compresses them jointly".

### Combining with DINOv2-B (bonus experiment)

| Regime | w-F1 | m-F1 |
|---|---:|---:|
| DINOv2-B alone, XGBoost | 0.5888 | 0.4336 |
| DINOv2-B alone, LR+scaler | 0.6150 | 0.4910 |
| **Advanced 349 + DINOv2-B concat, LR+scaler** | **0.6251** | **0.4956** |
| Advanced top-50 + DINOv2-B concat, XGB  | 0.5711 | 0.4151 |

- **LR beats XGB on DINOv2 features** by 0.027. Known finding from
  prior experiments — reproduced here.
- **Adding all 349 advanced features as extra LR inputs gives a small but real
  +0.010 F1** over DINOv2-B alone. This is the only regime where the new
  features add measurable value. Inside XGBoost they are redundant with each
  other and with DINOv2; under a linear classifier they provide a few
  complementary directions that scaler + L2 can use.
- **Does the concat beat the TTA ensemble?** No — 0.6251 vs 0.6458 (-0.021).
  Even with 1117 dimensions the single-model concat does not reach the two-model
  (DINOv2-B + BiomedCLIP) TTA ensemble, and the gap is comparable to the TTA
  boost (+0.011 on the ensemble).

### Which advanced families actually show up in XGBoost importance?

Looking at the top-30 global-fit importances:
- **Extended GLCM (xglcm_*)**: 6 of top-30, including homogeneity at d=4, d=32
  (the single most-used feature), energy at d=16, contrast at d=4, correlation
  at d=8. Advanced-GLCM extensions do carry signal beyond the original 4-distance
  set.
- **Multi-scale HOG (mhog_*)**: 3 features (c8 p75, c16 min, c16 p90). Finer
  cell sizes help.
- **Wavelet-packet (wp_*)**: 4 features (entropy, subband-ahh, subband-hhv,
  max-frac). The decomposition captures scale-localised energy the 1-level DWT
  in the handcrafted set did not have.
- **Gabor (gabor_*)**: 5 features across all 4 frequency bands. Orientation
  statistics add something orthogonal to GLCM.
- **Lacunarity (lac_s16)**: 1 feature, mid-scale gappiness.
- **Multifractal, Hurst/DFA, succolarity**: essentially absent. These families
  are either not discriminative for this corpus or degenerate under preprocess
  (percentile clip + plane-level collapses much of the heavy-tail content that
  multifractal q<0 moments probe).

So GLCM extensions, multi-scale HOG, wavelet-packet, and Gabor carry new
signal; lacunarity/succolarity/multifractal/Hurst largely do not.

### Why doesn't more feature engineering help?

The 240-scan regime is the real constraint. STATE.md and prior red-team work
already converged on "data ceiling around 0.65". Every complex manipulation
(stacking, threshold tuning, big concat) has produced either flat results or
inflation-then-regression after honest nesting. These new features follow the
same pattern: they don't hurt at full-dimensional XGB, but they don't give a
strong isolated gain either. The one honest win (+0.010 when concatenated
with DINOv2 under LR) is within the noise band of this evaluation.

## Files
- `teardrop/features_advanced.py` — extractor module (8 families, 349 feats)
- `scripts/extract_advanced_features.py` — cache builder
- `scripts/eval_advanced_features.py` — this evaluator
- `cache/features_advanced.parquet` — 240 × (443 features + 5 meta) cache
- `reports/advanced_features_results.json` — numerical results