# Ensemble Optimization — výsledky

Aktualizované: 2026-04-18.


## TL;DR


**Champion: `dinov2_s+dinov2_b+biomedclip+handcrafted|TUNED` / LR_tuned**

- LOPO weighted F1: **0.6878**
- LOPO macro F1: **0.5622**
- Baseline (RESULTS.md): 0.6280
- Δ vs baseline: **+0.0598**


## Všetky configy (sorted by weighted F1)

| Config | Classifier | Weighted F1 | Macro F1 |
|---|---|---:|---:|
| `dinov2_s+dinov2_b+biomedclip+handcrafted|TUNED` | LR_tuned | 0.6878 | 0.5622 |
| `dinov2_s+dinov2_b+biomedclip+handcrafted|AVG3+TUNED` | AVG3_tuned | 0.6582 | 0.5132 |
| `dinov2_s+dinov2_b+biomedclip+handcrafted` | LR | 0.6474 | 0.5190 |
| `dinov2_s+dinov2_b+biomedclip` | LR | 0.6388 | 0.5075 |
| `dinov2_s+dinov2_b+biomedclip+handcrafted|MLP+TUNED` | MLP_tuned | 0.6376 | 0.4991 |
| `dinov2_s+biomedclip` | LR | 0.6286 | 0.4889 |
| `dinov2_s+dinov2_b+biomedclip+handcrafted|AVG3` | AVG3 | 0.6042 | 0.4560 |
| `dinov2_s` | LR | 0.6004 | 0.4882 |
| `dinov2_s+dinov2_b` | LR | 0.5947 | 0.4722 |
| `dinov2_s+dinov2_b+biomedclip+handcrafted` | XGB | 0.5888 | 0.4152 |
| `dinov2_s+biomedclip` | XGB | 0.5800 | 0.4121 |
| `dinov2_s+dinov2_b+biomedclip` | XGB | 0.5562 | 0.3822 |
| `dinov2_s+dinov2_b` | XGB | 0.5425 | 0.3856 |
| `dinov2_s` | XGB | 0.5363 | 0.3905 |
| `dinov2_s+dinov2_b+biomedclip+handcrafted|MLP` | MLP | 0.5124 | 0.3807 |

## Per-class F1 (top 3)

| Config | Classifier | wF1 | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|---|---:|---:|---:|---:|---:|---:|
| `dinov2_s+dinov2_b+biomedclip+handcrafted|TUNED` | LR_tuned | 0.6878 | 0.905 | 0.640 | 0.585 | 0.680 | 0.000 |
| `dinov2_s+dinov2_b+biomedclip+handcrafted|AVG3+TUNED` | AVG3_tuned | 0.6582 | 0.865 | 0.522 | 0.469 | 0.710 | 0.000 |
| `dinov2_s+dinov2_b+biomedclip+handcrafted` | LR | 0.6474 | 0.884 | 0.542 | 0.528 | 0.641 | 0.000 |

## Champion confusion matrix

```
                   ZdraviLudia  Diabetes  PGOV_Glaukom  SklerozaMultiplex  SucheOko
ZdraviLudia                 62         4             0                  4         0
Diabetes                     5        16             0                  4         0
PGOV_Glaukom                 0         0            24                 12         0
SklerozaMultiplex            0         4            22                 66         3
SucheOko                     0         1             0                 13         0
```


## Postup experimentov

1. **Concat ensembles** — mean-pooled tile embeddingy (DINOv2-S/B, BiomedCLIP) + handcrafted.
2. **Per-class threshold/bias tuning** — coordinate-ascent na log-prob biasoch (sweep ±3 v 25 krokoch, opakované).
3. **MLP head** — PyTorch [256, 128, dropout=0.3, label_smoothing=0.1, AdamW] na najlepšom concat configu (LOPO + early stopping na 10 % held-out patient val split).
4. **AVG3 ensemble** — priemer probabilít LR + XGB + MLP, opcionálne s threshold tuningom.
5. **TTA** — preskočené, výsledky bez TTA prekonali 0.66 cieľ; TTA 8× re-encoding by zabralo ~10 min navyše.
