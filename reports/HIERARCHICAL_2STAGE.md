# Hierarchical 2-Stage Classifier — Stage A (binary) + Stage B (4-way)

**Thesis (user-stated):** isolating healthy first simplifies the 4-way disease decision.  Hypothesis: (easy binary Stage A) + (focused 4-way Stage B) beats the flat 5-way v4 champion (W-F1 = 0.6887).

## Methodology

- **Data:** 240 AFM scans, 35 persons (`teardrop.data.person_id`).
- **CV:** person-level LOPO (35 folds); every fold refits both Stage A and Stage B on the 34 non-held-out persons.
- **Encoders:** DINOv2-B @ 90 nm/px tiled, DINOv2-B @ 45 nm/px tiled, BiomedCLIP @ 90 nm/px D4-TTA  (the three components of v4).
- **Recipe per encoder:** row-wise L2-normalize -> StandardScaler (fit-on-train) -> LogisticRegression(class_weight='balanced', C=1, max_iter=3000).
- **Stage A:** binary labels — ZdraviLudia=0 (healthy), all others=1 (diseased).  Geom-mean of 3 encoder softmaxes.
- **Stage B:** 4-class labels (Diabetes, PGOV_Glaukom, SklerozaMultiplex, SucheOko) trained on the disease-only subset of each train fold.  Geom-mean of 3 encoder softmaxes.  Predicted for every val-fold scan (so we have P_B for all 240 scans out-of-fold).
- **Hard fusion:** argmax(P_A). If healthy -> ZdraviLudia, else argmax P_B.
- **Soft fusion:** P5 = [P_A(h), P_A(d) * P_B(class_i)].
- **Bootstrap vs flat v4:** B=1000 resamples of persons with replacement; recompute weighted F1 of flat v4 (re-derived) and each hierarchical variant on each bootstrap; report mean Δ, P(Δ>0), 95% CI.

## Stage A — Healthy vs Diseased (binary)

- **Counts:** healthy n=70, diseased n=170.

| Encoder | BinF1 (diseased) | BinF1 (healthy) | Weighted F1 | AUROC |
|---|---:|---:|---:|---:|
| `dinov2_90nm` | 0.9245 | 0.8322 | 0.8976 | 0.9622 |
| `dinov2_45nm` | 0.9277 | 0.8378 | 0.9015 | 0.9592 |
| `biomedclip_tta_90nm` | 0.9009 | 0.7755 | 0.8643 | 0.9403 |
| **ensemble (geom-mean)** | **0.9394** | **0.8667** | **0.9182** | **0.9736** |

- Strong Stage A: binary F1 (diseased) = 0.9394, AUROC = 0.9736.  Healthy isolation is indeed easy.

## Stage B — 4-way disease (standalone, diseased subset only)

- **Counts:** 170 diseased scans across 20 persons.

| Encoder | Weighted F1 | Macro F1 |
|---|---:|---:|
| `dinov2_90nm` | 0.5767 | 0.4904 |
| `dinov2_45nm` | 0.6153 | 0.5125 |
| `biomedclip_tta_90nm` | 0.5641 | 0.4771 |
| **ensemble (geom-mean)** | **0.6236** | **0.5247** |

Per-class F1 (disease-local order):
| Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|:---:|:---:|:---:|:---:|
| 0.8462 | 0.5789 | 0.6738 | 0.0000 |

- **Stage B does NOT rescue SucheOko** even without healthy competing: per-class F1 = 0.0000. The binding constraint is n=14 scans from 2 persons, not softmax competition.

## Hierarchical 5-class — combined fusion

| Method | Weighted F1 | Macro F1 | Δ vs flat v4 (0.6887) |
|---|---:|---:|---:|
| flat v4 (re-derived this run) | 0.6887 | 0.5541 | -0.0000 |
| **hier-hard** | **0.6551** | 0.5155 | **-0.0336** |
| **hier-soft** | **0.6551** | 0.5155 | **-0.0336** |

Per-class F1 (5-class ordering):
| Method | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|:---:|:---:|:---:|:---:|:---:|
| flat v4 (re-derived) | 0.9167 | 0.5833 | 0.5789 | 0.6915 | 0.0000 |
| hier-hard | 0.8667 | 0.4545 | 0.5789 | 0.6774 | 0.0000 |
| hier-soft | 0.8667 | 0.4545 | 0.5789 | 0.6774 | 0.0000 |

## Bootstrap (B=1000, person-resample, vs flat v4)

| Variant | mean ΔF1 | median ΔF1 | P(Δ > 0) | 95% CI |
|---|---:|---:|---:|:---:|
| hier-hard | -0.0331 | -0.0311 | 0.029 | [-0.0781, +0.0001] |
| hier-soft | -0.0331 | -0.0311 | 0.029 | [-0.0781, +0.0001] |

The bootstrap resamples **persons** (not scans) with replacement and recomputes weighted F1 on each resample, giving a CI that reflects person-level uncertainty — the relevant statistic given there are only 35 persons in the cohort.

## Verdict

- **DO NOT promote.**  Best variant `hard` W-F1 = 0.6551 (target > 0.70) with P(Δ>0) = 0.029 (target > 0.90 for promotion).

- Gap: -0.0336 vs flat v4 champion.  The hierarchical architecture LOSES because:
  1. Stage B is trained on only 170 diseased scans (vs 240 for flat), a ~30 % data loss that hurts the hardest 4-way decision.
  2. The flat 5-class softmax uses healthy as a "relief valve" for ambiguous diseased scans; hierarchical removes that escape hatch, forcing every scan the gate calls "diseased" into one of four slots.
  3. Stage A errors on healthy scans propagate as pure loss on ZdraviLudia F1, partially cancelling the gain on disease classes.
  4. Hard and soft fusion are numerically very close because P_B is typically peaky (max q ~ 0.9), so argmax(P5_soft) collapses to the hard rule in the vast majority of scans.

- **Minority-class hope failed:** Stage B does not rescue SucheOko (n=14 scans, 2 persons).  The binding constraint is dataset size, not softmax competition.

- Recommend: flat v4 remains the champion.  No red-team dispatch.

## Honest reporting

- Person-level LOPO (35 folds) for BOTH stages; no patient leakage.
- No threshold tuning (hard gate = natural 0.5 boundary); no OOF model selection; no post-hoc per-class calibration.
- Flat v4 baseline is re-derived here with the same v2 recipe on the same three encoders (no TTA on DINOv2 branches — matching the champion definition) and lands within 0.001 of the 0.6887 from STATE.md, validating the pipeline.
- Stage B training loses ~30 % of the data (170/240) because only diseased scans are used; this is acknowledged as a likely cause of the W-F1 regression.
