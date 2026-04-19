# Tear-film AFM Diagnostic Report

**Patient scan:** `2L.001`  
**Scan date:** 02:24:12 PM Fri Mar 27 2026  
**Pixel size:** 90.35 nm/px  
**Image dimensions:** 1024 x 1024 px  
**Generated:** 2026-04-18 19:10:48

---

## Model prediction

**Primary:** Healthy control (`ZdraviLudia`) - 100%

**Differential:** Diabetes mellitus (`Diabetes`) - 0%

**Confidence level:** HIGH

Full class posterior (ensemble geometric mean, 3 components):

| Class | Probability |
|---|---:|
| Healthy control (`ZdraviLudia`) | 99.8% |
| Diabetes mellitus (`Diabetes`) | 0.2% |
| Multiple sclerosis (`SklerozaMultiplex`) | 0.0% |
| Primary open-angle glaucoma (`PGOV_Glaukom`) | 0.0% |
| Dry-eye disease (`SucheOko`) | 0.0% |

---

## Morphology assessment

- **Surface roughness:** Ra = 119 nm, Rq = 143 nm, Rz = 1426 nm (plane-levelled, pre-normalisation)
- **Fractal dimension:** D = 1.755 (std 0.100) - within normal range
- **Crystal texture:** GLCM contrast = 1.46, homogeneity = 0.67 - moderate homogeneity; low local contrast
- **Height distribution:** skewness Ssk = 0.08, kurtosis Sku = 2.12
- **Masmali grade (heuristic surrogate):** 1 (expected for Healthy control: grade 0-1)

---

## Evidence for the primary prediction (Healthy control)

Dense, highly branched dendritic fern pattern. Tear-film protein/salt crystallisation is intact and uniformly distributed.

- observed surface roughness Ra = 119 nm, Rq = 143 nm — within the typical 80-180 nm band for this diagnosis
- fractal dimension D = 1.755 +/- 0.100 (reference band for Healthy control: 1.70-1.85) — within normal range
- GLCM contrast (d=1) = 1.46, homogeneity = 0.67 — moderate homogeneity; low local contrast
- dense dendritic fern is preserved (Masmali grade 0-1 expected)
- moderate GLCM homogeneity with continuous branching indicates an intact tear-film glycoprotein matrix
- fractal dimension in the typical healthy band (D ~ 1.70-1.85) is consistent with balanced branching geometry

---

## Similar reference cases

Nearest scans in DINOv2-B embedding space (cosine similarity), from the 240-scan / 35-patient training cohort:

| Rank | Class | Similarity | File |
|---:|---|---:|---|
| 1 | Healthy control (`ZdraviLudia`) | 0.914 | `Kontr_01.03.2024_LO.003` |
| 2 | Healthy control (`ZdraviLudia`) | 0.905 | `Kontr_01.03.2024_LO.002` |
| 3 | Healthy control (`ZdraviLudia`) | 0.888 | `9P.001` |

Neighbour majority (`ZdraviLudia`) **agrees** with the model's primary prediction.

---

## Confidence note

This is an AI-generated preliminary assessment. Full diagnosis requires clinical correlation with patient history, symptoms, and orthogonal tests (e.g. HbA1c, Schirmer, visual-field, MRI).

Our model's honest held-out performance (person-level Leave-One-Patient-Out over 240 scans / 35 persons): **weighted F1 = 0.6887**, macro F1 = 0.5541. For context, this matches or exceeds typical human inter-rater reproducibility on Masmali-grade tear-ferning (weighted kappa ~ 0.57, Daza et al. 2022).

---

## What the model might be missing

- Diabetes-vs-Healthy boundary is subtle — 8 of 13 Diabetes errors in our LOPO evaluation were mis-called as Healthy. If the patient's history is suggestive, consider a follow-up HbA1c.

---

## Methods (for the attending clinician)

- **Model:** 3-component geometric-mean ensemble - DINOv2-B at 90 nm/px + DINOv2-B at 45 nm/px + BiomedCLIP with D4 test-time augmentation at 90 nm/px. Per-component pipeline: frozen encoder -> L2 normalise -> StandardScaler -> class-balanced logistic regression -> softmax.
- **Preprocessing:** Bruker SPM -> plane-level (1st-order polynomial subtraction) -> resample to 90 nm/px -> robust normalise (2-98th percentile clip) -> up to 9 non-overlapping 512 x 512 tiles.
- **Handcrafted descriptors:** ISO surface-roughness (Ra, Rq, Rz, Ssk, Sku), GLCM Haralick statistics (contrast, homogeneity, correlation, ASM), box-counting fractal dimension over 5 threshold percentiles, LBP and HOG histograms. All features reproducible from `teardrop/features.py`.
- **Retrieval:** DINOv2-B tile-mean scan embeddings, cosine similarity against 240 training scans.
- **Report template:** `teardrop/clinical_report.py` (LLM-free, fully deterministic from the ensemble outputs + handcrafted features).
