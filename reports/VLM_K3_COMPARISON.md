> [!WARNING]
> **CONTAMINATED — DO NOT CITE.** This report used `cache/vlm_few_shot_collages/<CLASS>__<scan>.png` paths whose filename leaked the class label to the VLM. Caught by red-team audit `reports/RED_TEAM_SONNET_0_8873.md` on 2026-04-18.
> Honest replacement: `reports/VLM_SONNET_HONEST.md` (Sonnet honest wF1 = 0.3424, inflation +0.545).
> Leakage prevention infra: `teardrop/safe_paths.py` + `reports/LEAKAGE_PREVENTION.md`.

---

# VLM Few-Shot: k=3 vs k=2 Comparison

_Generated: 2026-04-18T21:20:07_

## Setup

- **Task:** 5-way tear-scan classification via Claude Haiku 4.5 reading a retrieval-augmented collage.
- **Subset:** 60 scans, person-stratified (per_class=12, seed=42).
- **Retrieval:** DINOv2-B cached embeddings (`tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`), cosine sim.
- **Person-LOPO:** anchors strictly exclude query's person.
- **k=2:** 2 nearest anchors per class (10 total) in a 5x2 grid. Existing script `scripts/vlm_few_shot.py`.
- **k=3:** 3 nearest anchors per class (15 total) in a 5x3 grid. New script `scripts/vlm_few_shot_k3.py`.
- **Prompts** identical apart from anchor count wording (see `PROMPT_TEMPLATE`).
- **Workers:** 8 parallel (ProcessPoolExecutor) for k=3.

## Headline

| Metric | k=2 (baseline) | k=3 | Δ (k3 − k2) |
|---|---|---|---|
| Accuracy | **0.7333** | **0.6667** | -0.0667 |
| **Weighted F1** | **0.7293** | **0.6491** | **-0.0802** |
| Macro F1 | 0.7293 | 0.6491 | -0.0802 |
| Mean confidence | 0.811 | 0.797 | -0.014 |
| Total cost (60 scans) | $0.99 | $1.15 | +0.17 |

## Per-class F1 (precision / recall / **F1**)

| Class | Support | k=2 | k=3 | ΔF1 |
|---|---|---|---|---|
| ZdraviLudia | 12 | 0.600 / 1.000 / **0.750** | 0.647 / 0.917 / **0.759** | +0.009 |
| Diabetes | 12 | 0.727 / 0.667 / **0.696** | 0.500 / 0.750 / **0.600** | -0.096 |
| PGOV_Glaukom | 12 | 0.714 / 0.833 / **0.769** | 0.750 / 0.750 / **0.750** | -0.019 |
| SklerozaMultiplex | 12 | 1.000 / 0.667 / **0.800** | 0.889 / 0.667 / **0.762** | -0.038 |
| SucheOko | 12 | 0.857 / 0.500 / **0.632** | 0.750 / 0.250 / **0.375** | -0.257 |

## Head-to-Head on Shared Subset

- Overlap scored by both: **60/60**
- Agreement on predicted class: **63.3%**
- Both correct: **34**
- Both wrong: **10**
- k=3 fixes (k=2 wrong → k=3 right): **6**
- k=3 breaks (k=2 right → k=3 wrong): **10**
- Net change attributable to k=3: **-4** correct predictions

### Confusion matrix — k=2

| true\pred | ZdraviLudi | Diabetes | PGOV_Glauk | SklerozaMu | SucheOko |
|---|---|---|---|---|---|
| ZdraviLudi | 12 | 0 | 0 | 0 | 0 |
| Diabetes | 3 | 8 | 0 | 0 | 1 |
| PGOV_Glauk | 1 | 1 | 10 | 0 | 0 |
| SklerozaMu | 1 | 2 | 1 | 8 | 0 |
| SucheOko | 3 | 0 | 3 | 0 | 6 |

### Confusion matrix — k=3

| true\pred | ZdraviLudi | Diabetes | PGOV_Glauk | SklerozaMu | SucheOko |
|---|---|---|---|---|---|
| ZdraviLudi | 11 | 1 | 0 | 0 | 0 |
| Diabetes | 2 | 9 | 1 | 0 | 0 |
| PGOV_Glauk | 2 | 1 | 9 | 0 | 0 |
| SklerozaMu | 1 | 1 | 1 | 8 | 1 |
| SucheOko | 1 | 6 | 1 | 1 | 3 |

### Flips where k=3 rescued a k=2 error

| Scan | Truth | k=2 | k=3 |
|---|---|---|---|
| `TRAIN_SET/Diabetes/Dusan2_DM_STER_mikro_281123.002` | Diabetes | ZdraviLudia | Diabetes |
| `TRAIN_SET/Diabetes/Dusan2_DM_STER_mikro_281123.005` | Diabetes | ZdraviLudia | Diabetes |
| `TRAIN_SET/Diabetes/DM_01.03.2024_LO.003` | Diabetes | ZdraviLudia | Diabetes |
| `TRAIN_SET/SklerozaMultiplex/22_PV_SM.002` | SklerozaMultiplex | Diabetes | SklerozaMultiplex |
| `TRAIN_SET/SklerozaMultiplex/100_7-SM-LV-18.001` | SklerozaMultiplex | Diabetes | SklerozaMultiplex |
| `TRAIN_SET/SucheOko/35_PM_suche_oko.001` | SucheOko | PGOV_Glaukom | SucheOko |

### Flips where k=3 regressed from k=2

| Scan | Truth | k=2 | k=3 |
|---|---|---|---|
| `TRAIN_SET/ZdraviLudia/48.001` | ZdraviLudia | ZdraviLudia | Diabetes |
| `TRAIN_SET/Diabetes/37_DM.012` | Diabetes | Diabetes | ZdraviLudia |
| `TRAIN_SET/Diabetes/Dusan2_DM_STER_mikro_281123.003` | Diabetes | Diabetes | ZdraviLudia |
| `TRAIN_SET/PGOV_Glaukom/25_PV_PGOV.012` | PGOV_Glaukom | PGOV_Glaukom | Diabetes |
| `TRAIN_SET/SklerozaMultiplex/Sklo-No2.041` | SklerozaMultiplex | SklerozaMultiplex | PGOV_Glaukom |
| `TRAIN_SET/SklerozaMultiplex/1-SM-LM-18.000` | SklerozaMultiplex | SklerozaMultiplex | Diabetes |
| `TRAIN_SET/SucheOko/29_PM_suche_oko.000` | SucheOko | SucheOko | Diabetes |
| `TRAIN_SET/SucheOko/35_PM_suche_oko.010` | SucheOko | SucheOko | Diabetes |
| `TRAIN_SET/SucheOko/29_PM_suche_oko.005` | SucheOko | SucheOko | SklerozaMultiplex |
| `TRAIN_SET/SucheOko/35_PM_suche_oko.011` | SucheOko | SucheOko | Diabetes |

## Decision

> **DO NOT scale k=3 — regression relative to k=2.**

**Observations**

- **SucheOko recall collapses** from 0.50 (k=2) to 0.25 (k=3) — 10 of the 12 k=3 SucheOko queries end up predicted as Diabetes / ZdraviLudia / SM / PGOV. Six of the ten k=3 regressions are SucheOko scans. With only 2 dry-eye persons in TRAIN_SET and person-LOPO exclusion, the SucheOko anchor pool is restricted to a single remaining person (~7 scans). Going from 2 → 3 anchors shows the VLM three very similar within-person crystals; instead of the expected "more evidence = easier classification", the added anchor appears to dilute the "dry / fragmented" prototype and push the query toward Diabetes/ZdraviLudia look-alikes.
- **Diabetes and PGOV_Glaukom** per-class F1 are roughly flat (-0.10, -0.02). The extra anchor does not materially help classes that already had decent retrieval at k=2.
- **ZdraviLudia** gains a little on precision (0.600 → 0.647) but loses one recall slot; net +0.01.
- **Head-to-head:** k=3 fixes 6 errors but introduces 10 new ones (net -4). Agreement drops to 63% — k=3 is not just a strictly-better superset of k=2.
- **Diminishing / negative returns:** k=2 already surfaces the top-2 nearest neighbours per class; the third neighbour at k=3 is often morphologically less similar (or identical to anchors 1-2 for small classes) and the model reads it as a distractor rather than extra signal. The 5x3 collage is also more visually cluttered — each anchor is smaller (240 px vs 256 px in k=2) which may reduce per-tile clarity for a vision model.
- **Cost** of k=3 is 17% higher ($1.15 vs $0.99 on 60 scans) despite the accuracy drop.

## Files

- Script: `scripts/vlm_few_shot_k3.py`
- k=3 predictions: `cache/vlm_few_shot_k3_predictions.json`
- k=3 collages:    `cache/vlm_few_shot_k3_collages/*.png`
- k=2 predictions: `cache/vlm_few_shot_predictions.json` + `cache/vlm_few_shot_k2_extend_predictions.json`
- Comparison script: `scripts/vlm_k3_vs_k2_compare.py`