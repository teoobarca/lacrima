# Prototypical Networks — Results

**Hypothesis:** Prototypical Networks (Snell et al., 2017) should help with extreme few-shot classes — our dataset has 2 persons for SucheOko and 4 for Diabetes, for which logistic regression struggles because the decision boundary is fit with very few positive samples. A prototype (class centroid in embedding space) gives each class equal footing regardless of its sample count.

## Setup

- **Data:** 240 AFM scans, 35 persons, 5 classes.
- **Encoders (matching v4 multiscale champion):** DINOv2-B 90 nm/px, DINOv2-B 45 nm/px, BiomedCLIP 90 nm/px with D4 TTA.
- **Features:** scan-level (mean-pool tile embeddings).
- **Evaluation:** PERSON-level LOPO (35 folds, `teardrop.data.person_id`).
- **Prototype:** L2-normalize support embeddings per class → take mean → L2-normalize again.
- **Classification:** softmax over negative distances (squared-Euclidean after L2-norm, or cosine).
- **Ensemble:** geometric mean of per-encoder softmaxes (same recipe as v2/v4).
- **Baseline to beat:** v4 multi-scale LR ensemble = W-F1 0.6887, M-F1 0.5541.


## Experiments

1. **baseline_sqeuclid** — training-free, squared-Euclidean distance after L2-normalization, temperature T=1.
2. **cosine** — training-free, cosine distance (1 - <q,p>).
3. **weighted_prototypes** — inverse-distance-to-centroid weighting (outlier-robust, Mahalanobis-ish).
4. **adapter_sqeuclid / adapter_cosine** — per-fold trained MLP (D → 256 → 128) with episodic ProtoNet loss (4-support, 2-query, 400 episodes, AdamW, learnable temperature). Then prototype + NN in the learned space.

**A note on fixed temperature:** under a geometric-mean ensemble with a temperature shared across all encoders, changing T only rescales the log-probabilities uniformly per encoder and does NOT alter the ensemble argmax — a fixed-T sweep is therefore uninformative. The temperature experiment in this report is the LEARNABLE per-fold T inside the adapter (exp (4)), which does change what the adapter optimizes and thus affects the ensemble.


## Per-encoder ProtoNet — baseline (sqEuclidean, T=1)

| Encoder | Weighted F1 | Macro F1 | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|---:|---:|:---:|:---:|:---:|:---:|:---:|
| `dinov2_90nm` | 0.5284 | 0.4315 | 0.726 | 0.386 | 0.559 | 0.486 | 0.000 |
| `dinov2_45nm` | 0.5144 | 0.4120 | 0.712 | 0.291 | 0.577 | 0.479 | 0.000 |
| `biomedclip_tta` | 0.4913 | 0.3823 | 0.676 | 0.200 | 0.556 | 0.480 | 0.000 |

## Ensemble (geometric mean across 3 encoders)

| Experiment | Weighted F1 | Macro F1 | Δ v4 | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| **baseline_sqeuclid** | 0.5042 | 0.3966 | -0.1845 | 0.703 | 0.255 | 0.542 | 0.483 | 0.000 |
| **cosine** | 0.5042 | 0.3966 | -0.1845 | 0.703 | 0.255 | 0.542 | 0.483 | 0.000 |
| **weighted_prototypes** | 0.4991 | 0.3890 | -0.1896 | 0.681 | 0.246 | 0.522 | 0.497 | 0.000 |
| **adapter_sqeuclid** | 0.6506 | 0.5120 | -0.0381 | 0.838 | 0.478 | 0.553 | 0.691 | 0.000 |
| **adapter_cosine** | 0.6276 | 0.4946 | -0.0611 | 0.819 | 0.468 | 0.526 | 0.660 | 0.000 |

## SucheOko minority-class analysis

SucheOko has only 2 persons (14 scans). With person-level LOPO the training fold for each SucheOko person has only ONE remaining SucheOko person — a true 1-shot (on person axis) regime. Under the LR recipe this collapses to F1 = 0 on those folds.

- **Best SucheOko F1 (ensemble):** 0.0000 — experiment `baseline_sqeuclid`.
- **Best SucheOko F1 (single encoder, any experiment):** 0.0741 — `adapter_sqeuclid / biomedclip_tta`.

- PARTIAL SIGNAL: SucheOko F1 climbs to **0.074** on the `biomedclip_tta` encoder under the adapter (`adapter_sqeuclid`), but the geometric-mean ensemble dilutes it back to 0 — the other two encoders never predict SucheOko and the geom-mean of three softmaxes where only one gives SucheOko ≠ 0 does not recover the positive. Useful follow-up: switch to MAX or weighted-mean ensembling for the minority-class channel, or condition on a ProtoNet gate.


## Verdict vs v4 LR champion

- Best overall ProtoNet variant: `adapter_sqeuclid` — W-F1 0.6506 (Δ v4 = -0.0381), M-F1 0.5120 (Δ v4 = -0.0421).

- ProtoNet is **below** v4 LR on weighted F1. The LR recipe is hard to beat when the majority classes each have ≥ 5 persons; the prototype loses discriminative capacity in favor of symmetry. Expected, per Snell et al.: ProtoNet helps strictly-few-shot but not well-populated classes.

- If Diabetes (4 persons) or SucheOko (2 persons) per-class F1 is noticeably higher than under v4, keep ProtoNet predictions as a minority-class rescue channel even if the overall score doesn't improve.


## Honest reporting

- Person-LOPO (35 folds) for every number here; no OOF model selection; no threshold tuning; no train/val leakage in the adapter (the adapter is trained fresh on each fold's training set).
- Adapter runs use fixed hyperparameters (no per-fold HP search) to prevent nested-CV leakage.
- Caches are the same three encoder outputs used by the v4 multi-scale champion → apples-to-apples comparison.
