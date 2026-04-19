# Augmented Head Training — Results

**Question:** Does training the v4 linear head on D4-expanded samples (8x per scan) improve weighted F1 vs the standard averaged-embedding recipe?

## Treatments

- **T1 averaged-head**: train head on mean-over-views embedding per scan (identical to v4 recipe; sanity baseline).
- **T2 noise-injected**: augment in embedding space (Gaussian sigma=0.01 on L2-normalized embeddings, 8 copies).
- **T3 D4-augmented**: each of the 8 D4 views = separate training sample (8x data for head). Test-time still averages views.

## Setup

- Person-level LOPO, 35 folds, 240 scans.
- v4 reference weighted F1 (reports): **0.6887**.

## Ensemble-level results

| Treatment | Weighted F1 | Macro F1 | delta vs v4 |
|---|---:|---:|---:|
| **T1 averaged** | 0.6887 | 0.5541 | -0.0000 |
| **T2 noise-inj** | 0.6788 | 0.5465 | -0.0099 |
| **T3 D4-aug** | 0.6518 | 0.5205 | -0.0369 |

## Per-class F1

| Treatment | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|:---:|:---:|:---:|:---:|:---:|
| **T1 averaged** | 0.9167 | 0.5833 | 0.5789 | 0.6915 | 0.0000 |
| **T2 noise-inj** | 0.9116 | 0.5833 | 0.5600 | 0.6774 | 0.0000 |
| **T3 D4-aug** | 0.9041 | 0.5306 | 0.5263 | 0.6413 | 0.0000 |

## Paired bootstrap (1000x) vs T1 (in-experiment baseline)

| Treatment | delta_mean | delta_p05 | delta_p95 | P(delta>0) |
|---|---:|---:|---:|---:|
| **T2 vs T1** | -0.0098 | -0.0233 | +0.0024 | 0.088 |
| **T3 vs T1** | -0.0381 | -0.0638 | -0.0124 | 0.003 |

## Verdict

- **T2 (noise-injected):** NOISE_FLOOR (delta=-0.0098, p(>0)=0.09)

- **T3 (D4-augmented):**  ROLLBACK (delta=-0.0381, p(>0)=0.00)


Promotion rule: `delta >= +0.02 weighted F1 AND P(delta>0) >= 0.90`. Noise floor: `|delta| < 0.01`.
