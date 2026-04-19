# VLM Numeric Reasoner — Sonnet 4.6 on Features Only

Date: 2026-04-18. Model: `claude-sonnet-4-6`.

## Why this experiment

Every prior VLM-vision attempt failed because AFM tear-ferning is out-of-distribution for vision encoders trained on natural images (zero-shot 30 %, few-shot honest 34 %, binary re-ranker 35 %). The class signatures, however, are well-described in TEXT — published Masmali grading, AFM literature on diabetic / MS / glaucoma tear-film biomarkers. Text reasoning is IN-distribution for a frontier LLM. This script tests whether Sonnet 4.6, given only 16 quantitative features per scan + the written class signatures (no image, no path, no filename), can reason back to the correct class.

## Methodology

- **Features sent** (per scan, 16 + 1 derived):
  `Sa, Sq, Ssk, Sku, glcm_contrast_d1_mean, glcm_homogeneity_d1_mean, glcm_correlation_d1_mean, fractal_D_mean, fractal_D_std, mf_alpha_width, lac_slope, hog_mean, lbp_0, lbp_10, lbp_25, gabor_f2_aniso` + heuristic `masmali_grade` (0-4).
- **Source**: `cache/features_advanced.parquet` (240 scans, 448 cols).
- **No image, no path, no filename** is ever included in the prompt — only a zero-padded anonymous scan index.
- **Safety guard**: `teardrop.safe_paths.assert_prompt_safe(prompt)` is called before every API request; raises `PromptLeakError` if any class name appears in a path-like context or any raw-filename fragment slips in.
- **Concurrency**: `ThreadPoolExecutor`, default 4 workers.
- **Determinism**: `temperature=0.0`, `max_tokens=1024`.
- **Scale-up gate**: if pilot wF1 >= 0.50, run the remaining scans.

## Pilot run (60-scan stratified subset, seed=42)

- **n processed / parsed**: 60 / 60  
- **Accuracy**: 0.233  
- **Weighted F1**: 0.126  
- **Macro F1**: 0.126  
- **Cost**: $0.339

```
                   precision    recall  f1-score   support

      ZdraviLudia      0.261     1.000     0.414        12
         Diabetes      0.100     0.083     0.091        12
     PGOV_Glaukom      0.000     0.000     0.000        12
SklerozaMultiplex      0.250     0.083     0.125        12
         SucheOko      0.000     0.000     0.000        12

         accuracy                          0.233        60
        macro avg      0.122     0.233     0.126        60
     weighted avg      0.122     0.233     0.126        60

```

Confusion matrix (rows=true, cols=pred, order = ZdraviLudia, Diabetes, PGOV_Glaukom, SklerozaMultiplex, SucheOko):

```
ZdraviLudia           12   0   0   0   0
Diabetes              11   1   0   0   0
PGOV_Glaukom           7   4   0   1   0
SklerozaMultiplex      8   3   0   1   0
SucheOko               8   2   0   2   0
```

## vs v4 ensemble (paired overlap on scored scans)

Direct head-to-head on the **same 60-scan subset**:

| Model                           | Accuracy | Weighted F1 | Macro F1 |
|---------------------------------|---------:|------------:|---------:|
| v4 ensemble (OOF argmax)        |    0.600 |       0.563 |    0.563 |
| Numeric Reasoner (Sonnet 4.6)   |    0.233 |       0.126 |    0.126 |
| **Delta (Reasoner − v4)**       |   −0.367 |      −0.437 |   −0.437 |

Paired bootstrap (1000×) on the n=60 overlap:
- Delta weighted F1 = **−0.437** (95 % CI [−0.587, −0.299])
- P[Delta > 0] = **0.00** — v4 strictly dominates in every bootstrap sample.

## Verdict

- **STOP** (do not scale to full 240 scans, do not include in ensemble). Weighted F1 = **0.126** is below the 0.50 gate — worse than a majority-class baseline on this five-class problem. The 60-scan confusion matrix shows catastrophic mode collapse: the model predicted `ZdraviLudia` in 46 of 60 cases (all 12 Diabetes rows got routed through ZdraviLudia / Diabetes, **0/12** PGOV_Glaukom and **0/12** SucheOko recovered). On the same 60 scans, v4 achieves 56.3 % weighted F1 — roughly 4.5× better than pure-text numeric reasoning.
- **Why it failed.** The 16 features are mostly overlapping (all five class ranges for `fractal_D_mean` intersect heavily; `glcm_correlation_d1_mean` is near-saturated at ~0.99 for every healthy-ish scan; roughness metrics are scale-variant and were not z-normalised before the prompt). Sonnet anchors to the first bullet in the class-signature list (healthy) whenever the fractal dimension falls in 1.70–1.85 and defers to that regardless of the other 15 features. A richer discrimination signal (per-feature z-scores vs. class priors, or actual image + features) would be required to recover separation.
- **Cost.** $0.34 for pilot (below $1.30 budget). No full run executed.


### Sample: correctly classified

**C1** — true = ZdraviLudia, pred = ZdraviLudia, conf = 0.82

  - Reasoning: _Fractal D mean of 1.771 falls squarely within the healthy signature range (1.70–1.85), indicating a dense, well-branched dendritic fern structure. GLCM homogeneity of 0.8215 is moderate and consistent with healthy morphology, while GLCM correlation of 0.998 signals strong local regularity rather than the chaotic granular texture of PGOV_Glaukom. Masmali grade 1 confirms near-complete ferning with minimal fragmentation, ruling out SucheOko (grade 3–4). Ssk of 0.1572 is only mildly positive and Sq is low, arguing against the elevated roughness expected in Diabetes._

**C2** — true = ZdraviLudia, pred = ZdraviLudia, conf = 0.62

  - Reasoning: _Fractal D mean of 1.745 falls squarely in the ZdraviLudia range (1.70–1.85), indicating a well-developed dendritic network. GLCM correlation of 0.9914 is very high, consistent with the structured, globally coherent branching seen in healthy ferns rather than the locally chaotic pattern of PGOV_Glaukom. GLCM homogeneity of 0.7304 is moderate, aligning with healthy morphology rather than the elevated contrast expected in Diabetes (Ssk is slightly negative at −0.2327, not the positive skew associated with diabetic coarse packing). Masmali grade 2 is borderline but fractal dimension remains the dominant discriminator here._

**C3** — true = ZdraviLudia, pred = ZdraviLudia, conf = 0.62

  - Reasoning: _Fractal D mean of 1.755 falls squarely within the ZdraviLudia signature range (1.70–1.85), indicating a well-developed dendritic fern structure. GLCM correlation of 0.9885 is very high, consistent with the organized, repetitive branching of healthy tear ferning rather than the locally chaotic pattern of glaucoma or the heterogeneous spread of MS. GLCM homogeneity of 0.6742 is moderate, also matching healthy. The main uncertainty is Masmali grade 2 (slightly above the 0–1 healthy ideal) and mf_alpha_width of 2.21 which hints at some intra-sample heterogeneity, but the fractal dimension and correlation are the strongest discriminators here._


### Sample: misclassified (diagnostic)

**W1** — true = Diabetes, pred = ZdraviLudia, conf = 0.72

  - Reasoning: _Fractal D mean of 1.759 falls squarely in the healthy range (1.70–1.85), and GLCM correlation of 0.9833 is very high, indicating structured, coherent branching rather than the chaotic local texture of PGOV_Glaukom. GLCM homogeneity at 0.6765 is moderate — consistent with a dense dendritic fern rather than fragmented dry-eye or coarse diabetic morphology. Ssk of −0.361 (negative skew) argues against Diabetes (which expects positive Ssk), and masmali_grade 2 is borderline but the fractal and correlation signatures dominate toward healthy over SucheOko (grade 3–4). mf_alpha_width of 2.1 is elevated but not extreme enough to favour SklerozaMultiplex without other heterogeneity markers._

**W2** — true = Diabetes, pred = ZdraviLudia, conf = 0.72

  - Reasoning: _Fractal dimension of 1.756 falls squarely in the healthy range (1.70–1.85), indicating a well-developed dendritic fern network. GLCM correlation of 0.9832 is very high, reflecting strong local structural regularity consistent with healthy branching rather than the chaotic granularity of PGOV_Glaukom. The negative Ssk (−0.8436) indicates a height distribution skewed toward valleys — typical of dense fern arms over a flat substrate — arguing against Diabetes (which expects positive Ssk). Masmali grade 2 introduces mild uncertainty (borderline sparse), but the combination of elevated fractal D, high GLCM correlation, and moderate HOG mean (0.142) aligns most strongly with ZdraviLudia._

**W3** — true = Diabetes, pred = ZdraviLudia, conf = 0.72

  - Reasoning: _Fractal D mean of 1.75 falls squarely in the ZdraviLudia signature range (1.70–1.85), indicating a dense, well-branched dendritic network. GLCM correlation of 0.9937 is very high, consistent with the structured, locally coherent ferning pattern of healthy tears rather than the low-correlation chaotic texture of glaucoma. GLCM homogeneity of 0.7658 is moderate as expected for healthy, and Ssk of 0.086 is near-zero (not the positive skew typical of Diabetes). Masmali grade 2 introduces mild doubt — slightly above the 0–1 ideal for ZdraviLudia — but the fractal and textural features outweigh this heuristic, possibly reflecting mild desiccation artefact rather than true pathology._


## Raw artefacts

- Predictions JSON: `cache/vlm_numeric_reasoner_predictions.json`

- Per-call JSONL: `cache/vlm_numeric_reasoner_raw.jsonl`

- Wall time: 180.7 s
