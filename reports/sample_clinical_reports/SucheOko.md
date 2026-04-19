# Tear-film AFM Diagnostic Report

**Patient scan:** `29_PM_suche_oko.001`  
**Scan date:** 08:41:45 AM Tue Feb 22 2022  
**Pixel size:** 26.77 nm/px  
**Image dimensions:** 3456 x 3456 px  
**Generated:** 2026-04-18 19:11:14

---

## Model prediction

**Primary:** Dry-eye disease (`SucheOko`) - 100%

**Differential:** Multiple sclerosis (`SklerozaMultiplex`) - 0%

**Confidence level:** HIGH

Full class posterior (ensemble geometric mean, 3 components):

| Class | Probability |
|---|---:|
| Dry-eye disease (`SucheOko`) | 99.6% |
| Multiple sclerosis (`SklerozaMultiplex`) | 0.3% |
| Healthy control (`ZdraviLudia`) | 0.1% |
| Diabetes mellitus (`Diabetes`) | 0.0% |
| Primary open-angle glaucoma (`PGOV_Glaukom`) | 0.0% |

---

## Morphology assessment

- **Surface roughness:** Ra = 51 nm, Rq = 61 nm, Rz = 614 nm (plane-levelled, pre-normalisation)
- **Fractal dimension:** D = 1.818 (std 0.060) - within normal range
- **Crystal texture:** GLCM contrast = 14.34, homogeneity = 0.50 - low homogeneity (irregular / fragmented texture); elevated local contrast
- **Height distribution:** skewness Ssk = 0.65, kurtosis Sku = 2.43
- **Masmali grade (heuristic surrogate):** 2 (expected for Dry-eye disease: grade 3-4)

---

## Evidence for the primary prediction (Dry-eye disease)

Fragmented, sparse crystalline network with large amorphous regions. Severe tear-film deficit leaves only isolated crystalline islands; Masmali grade 3-4.

- observed surface roughness Ra = 51 nm, Rq = 61 nm — within the typical 40-120 nm band for this diagnosis
- fractal dimension D = 1.818 +/- 0.060 (reference band for Dry-eye disease: 1.72-1.85) — within normal range
- GLCM contrast (d=1) = 14.34, homogeneity = 0.50 — low homogeneity (irregular / fragmented texture); elevated local contrast
- sparse, fragmented ferning network consistent with Masmali grade 3-4 dry-eye phenotype
- depressed fractal dimension (often D < 1.78) reflects loss of dendritic branching complexity
- large flat/amorphous regions dominate LBP histogram toward uniform bins — the hallmark of severe crystallisation failure

---

## Similar reference cases

Nearest scans in DINOv2-B embedding space (cosine similarity), from the 240-scan / 35-patient training cohort:

| Rank | Class | Similarity | File |
|---:|---|---:|---|
| 1 | Dry-eye disease (`SucheOko`) | 0.912 | `29_PM_suche_oko.004` |
| 2 | Healthy control (`ZdraviLudia`) | 0.868 | `79.001` |
| 3 | Healthy control (`ZdraviLudia`) | 0.853 | `Sklo-kontrola.018` |

Neighbour majority (`ZdraviLudia`) **disagrees** with the model's primary prediction (`SucheOko`) - this is an ambiguous sample; clinical review is strongly advised.

---

## Confidence note

This is an AI-generated preliminary assessment. Full diagnosis requires clinical correlation with patient history, symptoms, and orthogonal tests (e.g. HbA1c, Schirmer, visual-field, MRI).

Our model's honest held-out performance (person-level Leave-One-Patient-Out over 240 scans / 35 persons): **weighted F1 = 0.6887**, macro F1 = 0.5541. For context, this matches or exceeds typical human inter-rater reproducibility on Masmali-grade tear-ferning (weighted kappa ~ 0.57, Daza et al. 2022).

---

## What the model might be missing

- HIGH UNCERTAINTY: our training set contains only 2 unique Dry-Eye patients (14 scans). Person-LOPO F1 for this class is 0.00 — any Dry-Eye prediction is a data-acquisition ceiling artefact, not a reliable clinical signal. Treat as a rule-out, not a rule-in.

---

## Methods (for the attending clinician)

- **Model:** 3-component geometric-mean ensemble - DINOv2-B at 90 nm/px + DINOv2-B at 45 nm/px + BiomedCLIP with D4 test-time augmentation at 90 nm/px. Per-component pipeline: frozen encoder -> L2 normalise -> StandardScaler -> class-balanced logistic regression -> softmax.
- **Preprocessing:** Bruker SPM -> plane-level (1st-order polynomial subtraction) -> resample to 90 nm/px -> robust normalise (2-98th percentile clip) -> up to 9 non-overlapping 512 x 512 tiles.
- **Handcrafted descriptors:** ISO surface-roughness (Ra, Rq, Rz, Ssk, Sku), GLCM Haralick statistics (contrast, homogeneity, correlation, ASM), box-counting fractal dimension over 5 threshold percentiles, LBP and HOG histograms. All features reproducible from `teardrop/features.py`.
- **Retrieval:** DINOv2-B tile-mean scan embeddings, cosine similarity against 240 training scans.
- **Report template:** `teardrop/clinical_report.py` (LLM-free, fully deterministic from the ensemble outputs + handcrafted features).
