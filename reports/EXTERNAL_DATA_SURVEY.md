# External AFM Data Survey — Research Report

**Date:** 2026-04-18
**Method:** Multi-Ask (8x perplexity_ask, parallel)
**Project:** Teardrop AFM 5-class disease classification, DINOv2-B, 240 samples, LOPO F1 = 0.6887

---

## Executive Summary

No large, ready-to-use public AFM height-image dataset exists that closely mirrors dried tear droplet topography. Best available for domain-adaptive pre-training:

- **QUAM-AFM** (165M simulated molecular images, CC-BY-NC-SA-4.0) — biggest but simulated
- **Zenodo / Dryad real AFM datasets** — tiny (7–500 images each), real experimental
- **No AFM-specific pretrained encoder exists publicly** (April 2026)
- **VOPTICAL tear interferometry** (128–406 images) — same biology, different modality, email-gated

Literature on domain-adaptive SSL consistently shows **+7–14% accuracy gains** over ImageNet baselines on small downstream sets, if a 2k-10k image in-domain corpus is assembled.

---

## Top 5 candidate datasets

| Rank | Dataset | Size | License | Match | URL |
|---|---|---|---|---|---|
| 1 | QUAM-AFM | 165M simulated | CC-BY-NC-SA | Low-medium | https://edatos.consorciomadrono.es/dataset.xhtml?persistentId=doi:10.21950/UTGMZ7 |
| 2 | AAU Undersampled AFM | Thousands sim. | Academic | Medium | https://vbn.aau.dk/en/datasets/algorithms-for-reconstruction-of-undersampled-atomic-force-micros-2/ |
| 3 | Dryad polymer transfer film | 1.29 GB real | CC-BY | Medium-high | https://datadryad.org/dataset/doi:10.5061/dryad.zpc866thd |
| 4 | VOPTICAL tear interferometry | 128–406 images | Academic, free | Low pixel / **HIGH biological** | http://www.varpa.es/research/optics.html |
| 5 | Zenodo exosomes/liposomes | ~hundreds | CC-BY | Medium | https://zenodo.org/records/15113782 |

---

## Key finding: AFM-specific pretrained encoders don't exist

Searches across arXiv, GitHub, Zenodo, Hugging Face (April 2026) return **nothing matching AFM topography SSL pre-training**. Only supervised, narrow-purpose models:
- SINGROUP/Graph-AFM, SINGROUP/ED-AFM: task-specific supervised CNNs
- "Reverse AFM Height Map Search" (Springer 2024): SSL for retrieval, no public weights

**Releasing an AFM foundation model would itself be a novel contribution.**

---

## Actionable recommendations ranked by impact/effort

### 🏆 Recommendation 1 — MAE on our own 240 tiles (highest EV, zero data cost)

**Run MAE pretraining on our own tile extractions BEFORE seeking external data.**

- 240 scans × 9 tiles × D4 augmentation = 17,280 training patches
- MAE with 75% mask ratio + ViT-B reconstruction head
- Then fine-tune linear head on 5-class classification
- Expected gain: **+3–8 F1 points** (literature median for MAE on small medical datasets)

### Recommendation 2 — Hybrid real AFM corpus (medium effort)

Target 3,000-10,000 real AFM images by combining:
- Dryad polymer film (direct download)
- Zenodo exosomes/liposomes/cells (direct download)  
- Author outreach for 2015 / 2023 AFM tear studies (long-shot but high upside)
- Manufacturer galleries (Bruker, Nanosurf) — internal use only, license unclear

Pre-train MAE ViT-B on this corpus → fine-tune on 240 tears.
Expected gain: **+5–12 F1 points** at 3k+ corpus.

### Recommendation 3 — VOPTICAL multi-task fine-tuning

Email bremeseiro@uniovi.es for access. Use as biological-domain augmentation (same tear physiology, different modality). Expected gain: +2–6 F1 points.

### Recommendation 4 — QUAM-AFM filtered pre-training (fallback)

Download 50k subsample, filter by spatial frequency profile matching our scan statistics. Only if primary strategies stall.

---

## Corpus-size literature guidance

| Corpus size | Expected gain over ImageNet | Confidence |
|---|---|---|
| ~1,000 images | Marginal (<2%) | Low |
| ~2,000–5,000 images | +7–8% accuracy | High (MICCAI 2024) |
| ~10,000–50,000 images | +10–14% accuracy | High |
| >100,000 images | Diminishing returns | — |

**For our 240-sample downstream task:** the literature strongly supports in-domain pre-training even on small corpora. With 3,000–10,000 real AFM images, a +5–10 F1-point gain is realistic.

---

## Realistic F1 trajectory projection

| Strategy | Current | After | Gain |
|---|---|---|---|
| Baseline (ImageNet DINOv2-B) | 0.6887 | — | — |
| MAE on own 240 tiles | 0.6887 | 0.70–0.75 | +0.02–0.06 |
| + 3-10k external AFM corpus | 0.6887 | 0.73–0.79 | +0.05–0.10 |
| + QUAM-AFM filtered | (marginal) | — | +0.00–0.04 |
| + VOPTICAL fine-tune | (complement) | — | +0.01–0.03 |

**Conclusion:** MAE-on-own-tiles first (Recommendation 1) is the critical next experiment to attempt.

---

*Full raw synthesis available in agent transcript; committed as reference.*
