# VLM Sonnet 4.6 Few-Shot — HONEST (filename-leak fixed)

Generated: 2026-04-18T22:02:41

## TL;DR

- **Honest weighted F1: 0.3424** (n=237 scans, model=claude-sonnet-4-6)
- Honest macro F1: 0.2797
- vs leaky-Sonnet claim (0.8873): **inflation = +0.5449** (scalar diff)
- vs v4 champion (0.6887): delta = -0.3463
- Bootstrap P(Delta > 0 vs v4) = 0.000
- Verdict: **DEAD — retract.** Honest wF1 below 0.50 means the few-shot VLM path is not competitive and was essentially random-ish after the leak fix.

## Setup

- Fix: collage file written to `cache/vlm_few_shot_collages_honest/scan_XXXX.png` (shuffled, zero-padded index; no class, person, or raw-scan name in path).
- Manifest (NOT passed to VLM): `cache/vlm_sonnet_honest_manifest.json`.
- Retrieval & anchor composition identical to the leaky pipeline — only the collage filename differs. This isolates the filename leak as the causal factor.
- Prompt passes `{img_path}` = obfuscated path; self-test verifies no class keyword in path.
- Model: `claude-sonnet-4-6`; workers: 8; wall-clock: 449.2s; cost: $7.333.

## Scan-level metrics

- Accuracy: 0.3924
- **Weighted F1: 0.3424**  <- primary challenge metric
- Macro F1: 0.2797
- Mean confidence: 0.767

## Per-class F1

| Class | P | R | F1 | Support |
|---|---|---|---|---|
| ZdraviLudia | 0.550 | 0.870 | 0.674 | 69 |
| Diabetes | 0.000 | 0.000 | 0.000 | 24 |
| PGOV_Glaukom | 0.297 | 0.528 | 0.380 | 36 |
| SklerozaMultiplex | 0.480 | 0.128 | 0.202 | 94 |
| SucheOko | 0.143 | 0.143 | 0.143 | 14 |

## Confusion matrix (rows=true, cols=pred)

| true\pred | ZdraviLudi | Diabetes | PGOV_Glauk | SklerozaMu | SucheOko |
|---|---|---|---|---|---|
| ZdraviLudi | 60 | 5 | 0 | 4 | 0 |
| Diabetes | 18 | 0 | 2 | 2 | 2 |
| PGOV_Glauk | 2 | 7 | 19 | 7 | 1 |
| SklerozaMu | 21 | 12 | 40 | 12 | 9 |
| SucheOko | 8 | 1 | 3 | 0 | 2 |

## Per-person aggregation (majority vote)

- N persons: 35
- Accuracy: 0.5429
- Weighted F1: 0.4571
- Macro F1: 0.2933

## Per-patient-eye aggregation (majority vote)

- N patient-eyes: 44
- Accuracy: 0.5455
- Weighted F1: 0.4757
- Macro F1: 0.2825

## Bootstrap 1000x vs v4 multiscale (baseline wF1 = 0.6887)

- Honest Sonnet wF1 bootstrap mean: 0.3423  (95% CI [0.2733, 0.4170])
- Delta (Sonnet - v4) mean: **-0.3464**  (95% CI [-0.4154, -0.2717])
- **P(Delta > 0 vs v4) = 0.000**

## Bootstrap 1000x vs leaky Sonnet (baseline wF1 = 0.8873) — inflation quantification

- Delta (honest - leaky) mean: **-0.5440**  (95% CI [-0.6077, -0.4718])
- P(honest > leaky) = 0.000
- Inflation caused by filename leak (scalar): **+0.5449**

## Paired bootstrap 1000x: honest vs leaky (same scans) — tighter inflation estimate

- N overlap: 237
- Honest wF1 mean (paired): 0.3423  (95% CI [0.2733, 0.4170])
- Leaky  wF1 mean (paired): 0.8852  (95% CI [0.8448, 0.9251])
- **Inflation (leaky - honest) = +0.5429  (95% CI [+0.4702, +0.6100])**
- P(leaky > honest) = 1.000

## Decision

- **DEAD: retract few-shot VLM path.** Mark `VLM_SONNET_FULL_240.md` and all prior Sonnet/Haiku few-shot tables as CONTAMINATED.
- Do NOT ensemble into Expert Council. Do NOT cite in pitch.

## Reproducibility

- Script: `scripts/vlm_few_shot_sonnet_honest.py`
- Predictions cache: `cache/vlm_sonnet_honest_predictions.json`
- Manifest: `cache/vlm_sonnet_honest_manifest.json`
- Collages: `cache/vlm_few_shot_collages_honest/scan_XXXX.png`
- Model slug: `claude-sonnet-4-6`
- Leaky cache retained for comparison: `cache/vlm_sonnet_full_predictions.json` (do NOT overwrite).