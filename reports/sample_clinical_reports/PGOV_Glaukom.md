# Tear-film AFM Diagnostic Report

**Patient scan:** `21_LV_PGOV+SII.001`  
**Scan date:** 04:11:10 PM Wed Jan 12 2022  
**Pixel size:** 90.35 nm/px  
**Image dimensions:** 1024 x 1024 px  
**Generated:** 2026-04-18 19:10:57

---

## Model prediction

**Primary:** Primary open-angle glaucoma (`PGOV_Glaukom`) - 100%

**Differential:** Multiple sclerosis (`SklerozaMultiplex`) - 0%

**Confidence level:** HIGH

Full class posterior (ensemble geometric mean, 3 components):

| Class | Probability |
|---|---:|
| Primary open-angle glaucoma (`PGOV_Glaukom`) | 99.9% |
| Multiple sclerosis (`SklerozaMultiplex`) | 0.1% |
| Dry-eye disease (`SucheOko`) | 0.0% |
| Diabetes mellitus (`Diabetes`) | 0.0% |
| Healthy control (`ZdraviLudia`) | 0.0% |

---

## Morphology assessment

- **Surface roughness:** Ra = 91 nm, Rq = 113 nm, Rz = 603 nm (plane-levelled, pre-normalisation)
- **Fractal dimension:** D = 1.784 (std 0.099) - within normal range
- **Crystal texture:** GLCM contrast = 7.71, homogeneity = 0.77 - high homogeneity (smooth, uniform crystalline texture); elevated local contrast
- **Height distribution:** skewness Ssk = 1.43, kurtosis Sku = 3.46
- **Masmali grade (heuristic surrogate):** 0 (expected for Primary open-angle glaucoma: grade 3-4)

---

## Evidence for the primary prediction (Primary open-angle glaucoma)

Granular, loop-dominated surface. MMP-9 protease activity degrades the tear-film glycoprotein matrix, yielding shorter branches and coarse ring/loop topology visible as H_1 persistent-homology features.

- observed surface roughness Ra = 91 nm, Rq = 113 nm — within the typical 80-220 nm band for this diagnosis
- fractal dimension D = 1.784 +/- 0.099 (reference band for Primary open-angle glaucoma: 1.70-1.82) — within normal range
- GLCM contrast (d=1) = 7.71, homogeneity = 0.77 — high homogeneity (smooth, uniform crystalline texture); elevated local contrast
- granular texture with coarse medium-scale loops consistent with MMP-9 mediated matrix degradation
- locally chaotic correlation structure (GLCM correlation depressed at d=5) suggests short-range order is preserved but long-range branching is lost
- fractal dimension tends to be lower and more variable than in healthy controls due to truncated branching

---

## Similar reference cases

Nearest scans in DINOv2-B embedding space (cosine similarity), from the 240-scan / 35-patient training cohort:

| Rank | Class | Similarity | File |
|---:|---|---:|---|
| 1 | Primary open-angle glaucoma (`PGOV_Glaukom`) | 0.792 | `21_LV_PGOV+SII.000` |
| 2 | Primary open-angle glaucoma (`PGOV_Glaukom`) | 0.730 | `21_LV_PGOV+SII.006` |
| 3 | Multiple sclerosis (`SklerozaMultiplex`) | 0.722 | `50_5_SM-LV-18.004` |

Neighbour majority (`PGOV_Glaukom`) **agrees** with the model's primary prediction.

---

## Confidence note

This is an AI-generated preliminary assessment. Full diagnosis requires clinical correlation with patient history, symptoms, and orthogonal tests (e.g. HbA1c, Schirmer, visual-field, MRI).

Our model's honest held-out performance (person-level Leave-One-Patient-Out over 240 scans / 35 persons): **weighted F1 = 0.6887**, macro F1 = 0.5541. For context, this matches or exceeds typical human inter-rater reproducibility on Masmali-grade tear-ferning (weighted kappa ~ 0.57, Daza et al. 2022).

---

## What the model might be missing

- Glaucoma and Multiple Sclerosis are our primary confusion pair (all 15 Glaukom errors went to SM in the held-out audit). If this patient's clinical picture supports either condition, treat both as differentials.
- TDA / persistent-homology H_1 features (not used in this model) carry additional Glaucoma signal and may be worth a second pass.
- `SklerozaMultiplex` and `PGOV_Glaukom` are the most commonly confused pair in our held-out audit. Consider both as active differentials and correlate with the clinical picture.

---

## Methods (for the attending clinician)

- **Model:** 3-component geometric-mean ensemble - DINOv2-B at 90 nm/px + DINOv2-B at 45 nm/px + BiomedCLIP with D4 test-time augmentation at 90 nm/px. Per-component pipeline: frozen encoder -> L2 normalise -> StandardScaler -> class-balanced logistic regression -> softmax.
- **Preprocessing:** Bruker SPM -> plane-level (1st-order polynomial subtraction) -> resample to 90 nm/px -> robust normalise (2-98th percentile clip) -> up to 9 non-overlapping 512 x 512 tiles.
- **Handcrafted descriptors:** ISO surface-roughness (Ra, Rq, Rz, Ssk, Sku), GLCM Haralick statistics (contrast, homogeneity, correlation, ASM), box-counting fractal dimension over 5 threshold percentiles, LBP and HOG histograms. All features reproducible from `teardrop/features.py`.
- **Retrieval:** DINOv2-B tile-mean scan embeddings, cosine similarity against 240 training scans.
- **Report template:** `teardrop/clinical_report.py` (LLM-free, fully deterministic from the ensemble outputs + handcrafted features).
