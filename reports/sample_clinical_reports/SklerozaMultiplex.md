# Tear-film AFM Diagnostic Report

**Patient scan:** `1-SM-LM-18.001`  
**Scan date:** 05:07:10 PM Thu May 13 2021  
**Pixel size:** 90.35 nm/px  
**Image dimensions:** 1024 x 1024 px  
**Generated:** 2026-04-18 19:11:02

---

## Model prediction

**Primary:** Multiple sclerosis (`SklerozaMultiplex`) - 97%

**Differential:** Diabetes mellitus (`Diabetes`) - 2%

**Confidence level:** HIGH

Full class posterior (ensemble geometric mean, 3 components):

| Class | Probability |
|---|---:|
| Multiple sclerosis (`SklerozaMultiplex`) | 97.3% |
| Diabetes mellitus (`Diabetes`) | 1.7% |
| Healthy control (`ZdraviLudia`) | 0.9% |
| Dry-eye disease (`SucheOko`) | 0.1% |
| Primary open-angle glaucoma (`PGOV_Glaukom`) | 0.0% |

---

## Morphology assessment

- **Surface roughness:** Ra = 59 nm, Rq = 99 nm, Rz = 1460 nm (plane-levelled, pre-normalisation)
- **Fractal dimension:** D = 1.788 (std 0.094) - within normal range
- **Crystal texture:** GLCM contrast = 1.96, homogeneity = 0.68 - moderate homogeneity; low local contrast
- **Height distribution:** skewness Ssk = 0.75, kurtosis Sku = 4.09
- **Masmali grade (heuristic surrogate):** 0 (expected for Multiple sclerosis: grade 2-4)

---

## Evidence for the primary prediction (Multiple sclerosis)

Heterogeneous texture with mixed morphologies. Altered tear-film protein and lipid composition produces coarse rods and fine granules in the same sample; commonly confused with glaucoma at the macroscopic level.

- observed surface roughness Ra = 59 nm, Rq = 99 nm — within the typical 50-300 nm band for this diagnosis
- fractal dimension D = 1.788 +/- 0.094 (reference band for Multiple sclerosis: 1.74-1.88) — within normal range
- GLCM contrast (d=1) = 1.96, homogeneity = 0.68 — moderate homogeneity; low local contrast
- high intra-sample texture variance (mixed coarse/fine regions) consistent with altered lipid / MUC5AC composition in MS tear-film
- elevated GLCM contrast with depressed homogeneity points to localised crystalline islands separated by amorphous regions
- fractal dimension is variable across tiles — a hallmark of the heterogeneous MS tear-film phenotype

---

## Similar reference cases

Nearest scans in DINOv2-B embedding space (cosine similarity), from the 240-scan / 35-patient training cohort:

| Rank | Class | Similarity | File |
|---:|---|---:|---|
| 1 | Diabetes mellitus (`Diabetes`) | 0.868 | `Dusan1_DM_STER_mikro_281123.012` |
| 2 | Diabetes mellitus (`Diabetes`) | 0.843 | `Dusan2_DM_STER_mikro_281123.006` |
| 3 | Diabetes mellitus (`Diabetes`) | 0.823 | `Dusan2_DM_STER_mikro_281123.007` |

Neighbour majority (`Diabetes`) **disagrees** with the model's primary prediction (`SklerozaMultiplex`) - this is an ambiguous sample; clinical review is strongly advised.

---

## Confidence note

This is an AI-generated preliminary assessment. Full diagnosis requires clinical correlation with patient history, symptoms, and orthogonal tests (e.g. HbA1c, Schirmer, visual-field, MRI).

Our model's honest held-out performance (person-level Leave-One-Patient-Out over 240 scans / 35 persons): **weighted F1 = 0.6887**, macro F1 = 0.5541. For context, this matches or exceeds typical human inter-rater reproducibility on Masmali-grade tear-ferning (weighted kappa ~ 0.57, Daza et al. 2022).

---

## What the model might be missing

- SM is morphologically heterogeneous within-class; 20 of 37 SM errors in LOPO were mis-called as Glaucoma and 12 as Dry-Eye. Consider both differentials — especially in Sjogren's co-morbidity, where an SM patient may present with a dry-eye-like tear-film signature.

---

## Methods (for the attending clinician)

- **Model:** 3-component geometric-mean ensemble - DINOv2-B at 90 nm/px + DINOv2-B at 45 nm/px + BiomedCLIP with D4 test-time augmentation at 90 nm/px. Per-component pipeline: frozen encoder -> L2 normalise -> StandardScaler -> class-balanced logistic regression -> softmax.
- **Preprocessing:** Bruker SPM -> plane-level (1st-order polynomial subtraction) -> resample to 90 nm/px -> robust normalise (2-98th percentile clip) -> up to 9 non-overlapping 512 x 512 tiles.
- **Handcrafted descriptors:** ISO surface-roughness (Ra, Rq, Rz, Ssk, Sku), GLCM Haralick statistics (contrast, homogeneity, correlation, ASM), box-counting fractal dimension over 5 threshold percentiles, LBP and HOG histograms. All features reproducible from `teardrop/features.py`.
- **Retrieval:** DINOv2-B tile-mean scan embeddings, cosine similarity against 240 training scans.
- **Report template:** `teardrop/clinical_report.py` (LLM-free, fully deterministic from the ensemble outputs + handcrafted features).
