> [!WARNING]
> **CONTAMINATED — DO NOT CITE.** This report used `cache/vlm_few_shot_collages/<CLASS>__<scan>.png` paths whose filename leaked the class label to the VLM. Caught by red-team audit `reports/RED_TEAM_SONNET_0_8873.md` on 2026-04-18.
> Honest replacement: `reports/VLM_SONNET_HONEST.md` (Sonnet honest wF1 = 0.3424, inflation +0.545).
> Leakage prevention infra: `teardrop/safe_paths.py` + `reports/LEAKAGE_PREVENTION.md`.

---

# VLM x v4 Ensemble Results

- VLM coverage: 204/240 scans (85.0%)
- Uncovered rows fall back to v4's own softmax in the ensemble
- v4 baseline OOF: person-LOPO.  VLM: zero-shot per scan (no training -> no leakage).

## Complementarity

- Both correct: 156
- VLM only correct: 59
- v4 only correct: 11
- Neither correct: 14
- Oracle upper-bound accuracy (if we could pick the right model per scan): 0.9417

## Metrics table

| Strategy | Weighted F1 | Macro F1 | Acc | Healthy | Diabetes | Glaukom | SM | SucheOko |
|---|---|---|---|---|---|---|---|---|
| v4 alone (baseline)                      | 0.6887 | 0.5541 | 0.6958 | 0.92 | 0.58 | 0.58 | 0.69 | 0.00 |
| VLM alone (fallback=v4 if uncovered)     | 0.8967 | 0.8850 | 0.8958 | 0.88 | 0.91 | 0.99 | 0.90 | 0.76 |
| VLM alone (covered subset n=204)         | 0.8958 | 0.8822 | 0.8922 | 0.81 | 0.96 | 0.99 | 0.90 | 0.76 |
| A) Arith mean (w_vlm=0.5)                | 0.7798 | 0.7050 | 0.7833 | 0.92 | 0.83 | 0.68 | 0.76 | 0.32 |
| A*) Arith mean BEST 0.7                  | 0.8967 | 0.8850 | 0.8958 | 0.88 | 0.91 | 0.99 | 0.90 | 0.76 |
| B) Geom mean (w_vlm=0.5)                 | 0.7652 | 0.6657 | 0.7708 | 0.93 | 0.79 | 0.68 | 0.76 | 0.17 |
| B*) Geom mean BEST 0.8                   | 0.9133 | 0.8911 | 0.9125 | 0.91 | 0.89 | 0.97 | 0.92 | 0.76 |
| C) Confidence-weighted mean              | 0.7675 | 0.6969 | 0.7708 | 0.91 | 0.78 | 0.65 | 0.76 | 0.38 |
| D) VLM tiebreaker (margin<0.05)          | 0.6887 | 0.5541 | 0.6958 | 0.92 | 0.58 | 0.58 | 0.69 | 0.00 |
| D) VLM tiebreaker (margin<0.1)           | 0.6917 | 0.5599 | 0.7000 | 0.92 | 0.61 | 0.58 | 0.69 | 0.00 |
| D) VLM tiebreaker (margin<0.15)          | 0.6988 | 0.5715 | 0.7083 | 0.92 | 0.67 | 0.58 | 0.70 | 0.00 |
| D) VLM tiebreaker (margin<0.2)           | 0.6988 | 0.5715 | 0.7083 | 0.92 | 0.67 | 0.58 | 0.70 | 0.00 |
| D) VLM tiebreaker (margin<0.3)           | 0.7063 | 0.5905 | 0.7125 | 0.92 | 0.67 | 0.59 | 0.70 | 0.08 |
| E) Per-class specialist                  | 0.8265 | 0.7941 | 0.8250 | 0.96 | 0.83 | 0.80 | 0.77 | 0.61 |
| F) LOPO-learned weights (arith)          | 0.9009 | 0.8874 | 0.9000 | 0.88 | 0.91 | 0.99 | 0.90 | 0.76 |
| F2) LOPO-learned weights (geom)          | 0.9048 | 0.8862 | 0.9042 | 0.90 | 0.89 | 0.97 | 0.91 | 0.76 |

## Winner: **B*) Geom mean BEST 0.8**
- Weighted F1: 0.9133
- Macro F1: 0.8911
- Accuracy: 0.9125
- SucheOko F1: 0.7586
- Ship as v5? **YES** (gates: weighted>=0.82, SucheOko>0.5, macro>0.70)

## Red-team significance check (bootstrap, 2000 resamples)

| Metric | v4 (baseline) | v5 (VLM x v4 geom mean w=0.8) | Delta |
|---|---|---|---|
| Weighted F1 | 0.6887 | 0.9133 | +0.2246 |
| 95% CI for Delta |  |  | [+0.1627, +0.2896] |
| P(Delta > 0) |  |  | 1.0000 |
| P(Delta > 0.10) |  |  | 0.9995 |

Per-person accuracy delta across all 33 eval persons: **+17 / -1 / =15** (only one person regresses, by -0.09 on a 22-scan SM cluster). Gains are not concentrated in a single patient.

## Top-3 confusion matrices
### B*) Geom mean BEST 0.8

```
rows=true, cols=pred
          Zdravi Diabet PGOV_G Sklero SucheO
ZdraviLu      67      3      0      0      0
Diabetes       0     25      0      0      0
PGOV_Gla       1      0     35      0      0
Skleroza       6      3      1     81      4
SucheOko       3      0      0      0     11
```

```
                   precision    recall  f1-score   support

      ZdraviLudia      0.870     0.957     0.912        70
         Diabetes      0.806     1.000     0.893        25
     PGOV_Glaukom      0.972     0.972     0.972        36
SklerozaMultiplex      1.000     0.853     0.920        95
         SucheOko      0.733     0.786     0.759        14

         accuracy                          0.912       240
        macro avg      0.876     0.914     0.891       240
     weighted avg      0.922     0.912     0.913       240

```

### F2) LOPO-learned weights (geom)

```
rows=true, cols=pred
          Zdravi Diabet PGOV_G Sklero SucheO
ZdraviLu      67      3      0      0      0
Diabetes       0     25      0      0      0
PGOV_Gla       1      0     35      0      0
Skleroza       8      3      1     79      4
SucheOko       3      0      0      0     11
```

```
                   precision    recall  f1-score   support

      ZdraviLudia      0.848     0.957     0.899        70
         Diabetes      0.806     1.000     0.893        25
     PGOV_Glaukom      0.972     0.972     0.972        36
SklerozaMultiplex      1.000     0.832     0.908        95
         SucheOko      0.733     0.786     0.759        14

         accuracy                          0.904       240
        macro avg      0.872     0.909     0.886       240
     weighted avg      0.916     0.904     0.905       240

```

### F) LOPO-learned weights (arith)

```
rows=true, cols=pred
          Zdravi Diabet PGOV_G Sklero SucheO
ZdraviLu      67      3      0      0      0
Diabetes       0     25      0      0      0
PGOV_Gla       1      0     35      0      0
Skleroza      11      2      0     78      4
SucheOko       3      0      0      0     11
```

```
                   precision    recall  f1-score   support

      ZdraviLudia      0.817     0.957     0.882        70
         Diabetes      0.833     1.000     0.909        25
     PGOV_Glaukom      1.000     0.972     0.986        36
SklerozaMultiplex      1.000     0.821     0.902        95
         SucheOko      0.733     0.786     0.759        14

         accuracy                          0.900       240
        macro avg      0.877     0.907     0.887       240
     weighted avg      0.914     0.900     0.901       240

```
