# CGNN (CPU) — Results

Date: 2026-04-18  |  Wall time: **404.9s** (6.7 min)  |  Device: **CPU** (MPS hangs on GINEConv backward)

## Config

- Epochs: **40**
- Hidden: **48**
- Layers: **2**
- Dropout: **0.3**
- LR: **0.001**
- Batch: **4**
- Weight decay: **0.0001**
- Class-weighting: balanced (sklearn), label_smoothing=0.1, grad_clip=1.0
- Model: GINEConv × N_LAYERS, 5-dim node feats, 5-dim edge feats, mean+max+sum pool, MLP head, 5 classes

## Graph feature sanity

- n_graphs = 240
- node x: min=0.000  max=1.400  mean=0.441  (normalized)
- edge_attr: min=-1.000  max=18.500  mean=0.331
- graphs with 0 edges: 0

## Diagnostic run (fold-0, 5 epochs)

- best val F1w: **0.2571**
- loss epoch 1: 17.2579  → epoch 5: 1.8555
- loss Δ: +15.4024  (learning)
- time: 4.5s

## Eye-level grouping (`patient`, 44 groups)

- Mean fold F1w: **0.4287 ± 0.0682**
- Mean fold F1m: **0.3045 ± 0.0329**
- OOF F1w: **0.4589**  |  OOF F1m: **0.3242**
- Total time: 197.4s

**Per-fold:**

| Fold | n_val | F1w | F1m | classes | sec |
|---:|---:|---:|---:|---|---:|
| 0 | 49 | 0.4180 | 0.3260 | [0, 2, 3, 4] | 38.8 |
| 1 | 45 | 0.3836 | 0.2950 | [0, 1, 2, 3] | 39.3 |
| 2 | 47 | 0.5583 | 0.3360 | [0, 1, 2, 3] | 40.4 |
| 3 | 51 | 0.3638 | 0.2444 | [0, 1, 2, 3, 4] | 38.7 |
| 4 | 48 | 0.4197 | 0.3210 | [0, 1, 2, 3] | 40.1 |

**Per-class F1 (OOF):**

| Class | F1 |
|---|---:|
| ZdraviLudia | 0.600 |
| Diabetes | 0.316 |
| PGOV_Glaukom | 0.000 |
| SklerozaMultiplex | 0.622 |
| SucheOko | 0.083 |

**Classification report:**

```
                   precision    recall  f1-score   support

      ZdraviLudia      0.562     0.643     0.600        70
         Diabetes      0.235     0.480     0.316        25
     PGOV_Glaukom      0.000     0.000     0.000        36
SklerozaMultiplex      0.612     0.632     0.622        95
         SucheOko      0.100     0.071     0.083        14

         accuracy                          0.492       240
        macro avg      0.302     0.365     0.324       240
     weighted avg      0.437     0.492     0.459       240
```

**Confusion matrix:**

```
                   ZdraviLudia  Diabetes  PGOV_Glaukom  SklerozaMultiplex  SucheOko
ZdraviLudia                 45        21             0                  4         0
Diabetes                     7        12             0                  6         0
PGOV_Glaukom                 4         6             0                 23         3
SklerozaMultiplex           16        12             1                 60         6
SucheOko                     8         0             0                  5         1
```

## Person-level grouping (`person_id`, ~35 groups)

- Mean fold F1w: **0.3208 ± 0.0422**
- Mean fold F1m: **0.2092 ± 0.0555**
- OOF F1w: **0.3649**  |  OOF F1m: **0.2203**
- Total time: 203.0s

**Per-fold:**

| Fold | n_val | F1w | F1m | classes | sec |
|---:|---:|---:|---:|---|---:|
| 0 | 50 | 0.3718 | 0.2571 | [0, 2, 3, 4] | 38.2 |
| 1 | 50 | 0.2478 | 0.1304 | [0, 1, 2, 3] | 41.0 |
| 2 | 46 | 0.3487 | 0.2077 | [0, 1, 2, 3, 4] | 42.0 |
| 3 | 47 | 0.3087 | 0.2819 | [0, 1, 2, 3] | 41.3 |
| 4 | 47 | 0.3271 | 0.1688 | [0, 1, 2, 3] | 40.4 |

**Per-class F1 (OOF):**

| Class | F1 |
|---|---:|
| ZdraviLudia | 0.478 |
| Diabetes | 0.000 |
| PGOV_Glaukom | 0.087 |
| SklerozaMultiplex | 0.537 |
| SucheOko | 0.000 |

**Classification report:**

```
                   precision    recall  f1-score   support

      ZdraviLudia      0.391     0.614     0.478        70
         Diabetes      0.000     0.000     0.000        25
     PGOV_Glaukom      0.200     0.056     0.087        36
SklerozaMultiplex      0.537     0.537     0.537        95
         SucheOko      0.000     0.000     0.000        14

         accuracy                          0.400       240
        macro avg      0.226     0.241     0.220       240
     weighted avg      0.357     0.400     0.365       240
```

**Confusion matrix:**

```
                   ZdraviLudia  Diabetes  PGOV_Glaukom  SklerozaMultiplex  SucheOko
ZdraviLudia                 43         0             0                 14        13
Diabetes                    11         0             0                  9         5
PGOV_Glaukom                13         0             2                 20         1
SklerozaMultiplex           30         0             8                 51         6
SucheOko                    13         0             0                  1         0
```

## Comparison to DINOv2 baseline

| Model | Eye-LOPO F1w | Person-LOPO F1w | Macro F1 |
|---|---:|---:|---:|
| DINOv2-B tiled scan-mean + LR | 0.628 | **0.615** | 0.491 |
| CGNN (GINE, CPU, this run) | 0.459 | 0.365 | 0.220 |

## Interpretation

- CGNN is **clearly weaker** than DINOv2 (Δ = -0.250). Value is in **interpretability** (node = junction/endpoint, edge = skeleton segment length/tortuosity) rather than raw F1. Graph-topology features alone do not capture the full AFM appearance.
- Worth keeping as an **ensemble component** only if its softmax is diverse w.r.t. DINOv2 errors — a separate analysis.