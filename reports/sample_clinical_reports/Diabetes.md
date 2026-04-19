# Tear-film AFM Diagnostic Report

**Patient scan:** `DM_01.03.2024_LO.001`  
**Scan date:** 04:41:21 PM Fri Mar 01 2024  
**Pixel size:** 90.35 nm/px  
**Image dimensions:** 1024 x 1024 px  
**Generated:** 2026-04-18 19:10:53

---

## Model prediction

**Primary:** Diabetes mellitus (`Diabetes`) - 99%

**Differential:** Healthy control (`ZdraviLudia`) - 1%

**Confidence level:** HIGH

Full class posterior (ensemble geometric mean, 3 components):

| Class | Probability |
|---|---:|
| Diabetes mellitus (`Diabetes`) | 98.7% |
| Healthy control (`ZdraviLudia`) | 1.3% |
| Dry-eye disease (`SucheOko`) | 0.0% |
| Multiple sclerosis (`SklerozaMultiplex`) | 0.0% |
| Primary open-angle glaucoma (`PGOV_Glaukom`) | 0.0% |

---

## Morphology assessment

- **Surface roughness:** Ra = 222 nm, Rq = 274 nm, Rz = 1597 nm (plane-levelled, pre-normalisation)
- **Fractal dimension:** D = 1.742 (std 0.101) - within normal range
- **Crystal texture:** GLCM contrast = 0.71, homogeneity = 0.77 - high homogeneity (smooth, uniform crystalline texture); low local contrast
- **Height distribution:** skewness Ssk = -0.45, kurtosis Sku = 2.39
- **Masmali grade (heuristic surrogate):** 1 (expected for Diabetes mellitus: grade 2-3)

---

## Evidence for the primary prediction (Diabetes mellitus)

Thickened, densely packed dendrites with elevated small-scale roughness. Hyperglycaemia-induced glycation drives denser crystal packing and a coarser surface.

- observed surface roughness Ra = 222 nm, Rq = 274 nm — within the typical 150-350 nm band for this diagnosis
- fractal dimension D = 1.742 +/- 0.101 (reference band for Diabetes mellitus: 1.73-1.85) — within normal range
- GLCM contrast (d=1) = 0.71, homogeneity = 0.77 — high homogeneity (smooth, uniform crystalline texture); low local contrast
- elevated surface roughness (Ra often > 180 nm) consistent with hyperglycaemia-induced glycation of tear-film proteins
- higher GLCM contrast reflects denser crystal packing and loss of smooth lamellar structure
- skewness tends to shift positive — taller crystalline peaks dominate over trough regions

---

## Similar reference cases

Nearest scans in DINOv2-B embedding space (cosine similarity), from the 240-scan / 35-patient training cohort:

| Rank | Class | Similarity | File |
|---:|---|---:|---|
| 1 | Diabetes mellitus (`Diabetes`) | 0.896 | `DM_01.03.2024_LO.002` |
| 2 | Diabetes mellitus (`Diabetes`) | 0.859 | `DM_01.03.2024_LO.009` |
| 3 | Healthy control (`ZdraviLudia`) | 0.838 | `9L.002` |

Neighbour majority (`Diabetes`) **agrees** with the model's primary prediction.

---

## Confidence note

This is an AI-generated preliminary assessment. Full diagnosis requires clinical correlation with patient history, symptoms, and orthogonal tests (e.g. HbA1c, Schirmer, visual-field, MRI).

Our model's honest held-out performance (person-level Leave-One-Patient-Out over 240 scans / 35 persons): **weighted F1 = 0.6887**, macro F1 = 0.5541. For context, this matches or exceeds typical human inter-rater reproducibility on Masmali-grade tear-ferning (weighted kappa ~ 0.57, Daza et al. 2022).

---

## What the model might be missing

- This class has only 25 scans across 4-5 patients in training; session-level variance is large (DM_01.03.2024 session contributes 6 of 13 total Diabetes errors).
- Diabetes and Healthy share a soft decision boundary driven by glucose-dependent salt-lattice expression — always correlate with HbA1c.

---

## Methods (for the attending clinician)

- **Model:** 3-component geometric-mean ensemble - DINOv2-B at 90 nm/px + DINOv2-B at 45 nm/px + BiomedCLIP with D4 test-time augmentation at 90 nm/px. Per-component pipeline: frozen encoder -> L2 normalise -> StandardScaler -> class-balanced logistic regression -> softmax.
- **Preprocessing:** Bruker SPM -> plane-level (1st-order polynomial subtraction) -> resample to 90 nm/px -> robust normalise (2-98th percentile clip) -> up to 9 non-overlapping 512 x 512 tiles.
- **Handcrafted descriptors:** ISO surface-roughness (Ra, Rq, Rz, Ssk, Sku), GLCM Haralick statistics (contrast, homogeneity, correlation, ASM), box-counting fractal dimension over 5 threshold percentiles, LBP and HOG histograms. All features reproducible from `teardrop/features.py`.
- **Retrieval:** DINOv2-B tile-mean scan embeddings, cosine similarity against 240 training scans.
- **Report template:** `teardrop/clinical_report.py` (LLM-free, fully deterministic from the ensemble outputs + handcrafted features).
