> [!WARNING]
> **CONTAMINATED — DO NOT CITE.** This report used `cache/vlm_few_shot_collages/<CLASS>__<scan>.png` paths whose filename leaked the class label to the VLM. Caught by red-team audit `reports/RED_TEAM_SONNET_0_8873.md` on 2026-04-18.
> Honest replacement: `reports/VLM_SONNET_HONEST.md` (Sonnet honest wF1 = 0.3424, inflation +0.545).
> Leakage prevention infra: `teardrop/safe_paths.py` + `reports/LEAKAGE_PREVENTION.md`.

---

# VLM Opus 4.7 vs Sonnet 4.6 vs Haiku 4.5 — Head-to-Head

Generated: 2026-04-18T21:46:30

## Setup

- Subset: 60 scans, person-stratified (seed=123, 12 per class).
- Pipeline: DINOv2-B kNN retrieval -> 10-anchor (2/class) collage -> VLM classifies query.
- Person-LOPO hard-assertion: anchor person != query person for every sample.
- Total cost this run (Opus): $4.00

## Headline comparison

| Model | N | wF1 | mF1 | Accuracy |
|---|---:|---:|---:|---:|
| Haiku 4.5 (cached note) | 19 | 0.9482 | 0.9492 | 0.9474 |
| Sonnet 4.6 | 60 | 0.8454 | 0.8454 | 0.8500 |
| **Opus 4.7** | 60 | **0.7974** | **0.7974** | **0.8000** |

**Δ wF1 (Opus − Sonnet) on overlapping scans: -0.0480**

## Paired bootstrap CI (1000 resamples, Opus − Sonnet)

- Δ wF1 mean: -0.0493
- 95% CI: [-0.1675, +0.0713]
- Two-sided significant at alpha=0.05: **False**
- Fraction of bootstraps with Δ <= 0: 0.781

## Per-class F1

| Class | Haiku F1 | Sonnet F1 | Opus F1 | Support |
|---|---:|---:|---:|---:|
| ZdraviLudia | 0.857 | 0.889 | 0.870 | 12 |
| Diabetes | 1.000 | 0.917 | 0.815 | 12 |
| PGOV_Glaukom | 1.000 | 0.828 | 0.833 | 12 |
| SklerozaMultiplex | 0.889 | 0.857 | 0.769 | 12 |
| SucheOko | 1.000 | 0.737 | 0.700 | 12 |

## Opus 4.7 Confusion Matrix (rows=true, cols=pred)

| true\pred | ZdraviLudi | Diabetes | PGOV_Glauk | SklerozaMu | SucheOko |
|---|---:|---:|---:|---:|---:|
| ZdraviLudi | 10 | 2 | 0 | 0 | 0 |
| Diabetes | 1 | 11 | 0 | 0 | 0 |
| PGOV_Glauk | 0 | 0 | 10 | 2 | 0 |
| SklerozaMu | 0 | 0 | 1 | 10 | 1 |
| SucheOko | 0 | 2 | 1 | 2 | 7 |

## Cost comparison (this run only)

- Opus 60-scan cost: $4.00
- Sonnet reference: ~$0.50 for 60 scans historically (~5x cheaper).
- Full 240 Opus projected: ~$16.01

## Decision

**STOP** — Opus does NOT clear the +3 pp threshold. Δ wF1 = -0.0480.
- Sonnet 4.6 is the economic winner at ~1/5 the cost.
- Paired bootstrap Δ 95% CI: [-0.1675, +0.0713] — includes 0.
