> [!WARNING]
> **CONTAMINATED — DO NOT CITE.** This report used `cache/vlm_few_shot_collages/<CLASS>__<scan>.png` paths whose filename leaked the class label to the VLM. Caught by red-team audit `reports/RED_TEAM_SONNET_0_8873.md` on 2026-04-18.
> Honest replacement: `reports/VLM_SONNET_HONEST.md` (Sonnet honest wF1 = 0.3424, inflation +0.545).
> Leakage prevention infra: `teardrop/safe_paths.py` + `reports/LEAKAGE_PREVENTION.md`.

---

# VLM Sonnet 4.6 Few-Shot on FULL 240 Scans

Generated: 2026-04-18T21:40:48

## Setup

- Retrieval: DINOv2-B (`cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`), cosine sim, **person-disjoint** anchors.
- k=2 anchors per class -> 10-anchor collage + QUERY tile.
- VLM: `claude-sonnet-4-6`, `claude -p --output-format json` via CLI subprocess.
- Parallelism: 8 worker threads (subprocess-bound; threads > processes).
- Scans scored: 240 / 240.
- Total API cost: $7.205
- Wall-clock time: 351.5 s

## Scan-level metrics (primary)

- Accuracy: **0.8875**
- **Weighted F1: 0.8873**  <- challenge metric
- Macro F1: 0.8518
- Mean confidence: 0.789

## Per-class F1

| Class | P | R | F1 | Support |
|---|---|---|---|---|
| ZdraviLudia | 0.932 | 0.986 | 0.958 | 70 |
| Diabetes | 0.828 | 0.960 | 0.889 | 25 |
| PGOV_Glaukom | 0.720 | 1.000 | 0.837 | 36 |
| SklerozaMultiplex | 1.000 | 0.789 | 0.882 | 95 |
| SucheOko | 0.750 | 0.643 | 0.692 | 14 |

## Confusion matrix (rows=true, cols=pred)

| true\pred | ZdraviLudi | Diabetes | PGOV_Glauk | SklerozaMu | SucheOko |
|---|---|---|---|---|---|
| ZdraviLudi | 69 | 1 | 0 | 0 | 0 |
| Diabetes | 0 | 24 | 1 | 0 | 0 |
| PGOV_Glauk | 0 | 0 | 36 | 0 | 0 |
| SklerozaMu | 1 | 3 | 13 | 75 | 3 |
| SucheOko | 4 | 1 | 0 | 0 | 9 |

## Per-person aggregation (majority vote across scans of same person)

- N persons: 35
- Accuracy: 0.9714
- Weighted F1: 0.9671
- Macro F1: 0.9269

## Per-patient-eye aggregation (majority vote across scans of same eye)

- N patient-eyes: 44
- Accuracy: 0.9545
- Weighted F1: 0.9520
- Macro F1: 0.9016

## Bootstrap 1000x vs v4 multiscale champion (weighted F1 = 0.6887)

- Sonnet weighted F1 bootstrap mean: 0.8877  (95% CI [0.8483, 0.9288])
- Delta (Sonnet - v4) mean: **+0.1990**  (95% CI [+0.1596, +0.2401])
- **P(Delta > 0 vs v4) = 1.000**

## Bootstrap 1000x vs Haiku-full-240 (weighted F1 = 0.6755) — unpaired

- Delta (Sonnet - Haiku) mean: **+0.2123**  (95% CI [+0.1701, +0.2498])
- **P(Delta > 0 vs Haiku) = 1.000**

## Paired bootstrap 1000x vs Haiku-full-240 (tighter, same-scan resampling)

- N overlap: 240
- Sonnet wF1 mean (paired resamples): 0.8877  (95% CI [0.8483, 0.9288])
- Haiku  wF1 mean (paired resamples): 0.6753  (95% CI [0.6159, 0.7360])
- Delta (Sonnet - Haiku) mean: **+0.2124**  (95% CI [+0.1496, +0.2751])
- **P(Delta > 0 vs Haiku, paired) = 1.000**

## Sonnet vs Haiku agreement on 240 scans

- Both correct: 148
- Both wrong: 15
- Sonnet right only (Haiku wrong): 65
- Haiku right only (Sonnet wrong): 12
- Agreement rate: 65.8%

## >>> NEW CHAMPION CANDIDATE <<<

- Weighted F1 = 0.8873 >= 0.75
- P(Delta > 0 vs v4) = 1.000 > 0.95
- **ACTION**: Red-team before promotion (person-LOPO audit, anchor-leak audit, prompt-sensitivity, cost/latency ops review).

## Comparison summary

| Run | N | Weighted F1 | Macro F1 | Notes |
|---|---|---|---|---|
| Sonnet 60-subset (`VLM_FEW_SHOT_RESULTS.md`) | 60 | 0.8454 | 0.8454 | stratified, person-disjoint |
| Haiku full 240 (`VLM_FEW_SHOT_FULL_240.md`) | 240 | 0.6755 | 0.5925 | champion-beating attempt |
| **Sonnet full 240 (this run)** | 240 | **0.8873** | **0.8518** | 10-anchor collage, k=2 |
| v4 multiscale ensemble | 240 | 0.6887 | — | current Wave-5 champion |

## Reproducibility

- Script: `scripts/vlm_few_shot_sonnet_full_240.py`
- Predictions cache: `cache/vlm_sonnet_full_predictions.json`
- Collages: `cache/vlm_few_shot_collages/`
- Model slug: `claude-sonnet-4-6`
- Embeddings: `cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz` (L2-normalized)
