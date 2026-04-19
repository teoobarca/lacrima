# Prototypical Network — SucheOko Rescue Experiment

Single-encoder ProtoNet rescue attempt for the 2-person SucheOko class, which the v4 champion predicts F1 = 0.000.

## Setup

- **Data:** 240 scans, 35 persons, 5 classes.
- **Embeddings:** `cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz` (DINOv2-B, TTA D4, 240 x 768).
- **CV:** person-level LOPO (35 folds), groups = `teardrop.data.person_id`. Prototypes computed strictly from training persons; query person never in prototype set.
- **Distance:** cosine on L2-normalized embeddings. Softmax temperature T=0.1.
- **Baseline:** v4 multiscale LR champion, wF1=0.6887, mF1=0.5541, SucheOko F1=0.000.


## Variants

1. **Standard** — L2-normed mean per class, cosine -> softmax.
2. **Weighted** — per-class logit bias = log(1/sqrt(N_c)); shifts decision boundary toward rare classes.
3. **Person-averaged** — mean within person first, then mean across persons -- reduces dominant-person bias inside class.
4. **K-NN-weighted** — query-dependent soft kNN voting, kernel bandwidth tau=0.15; per-class normalized.


## Per-variant results (person-LOPO)

| Variant | wF1 | mF1 | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|---:|---:|:---:|:---:|:---:|:---:|:---:|
| **standard** | 0.5193 | 0.4231 | 0.713 | 0.373 | 0.549 | 0.480 | 0.000 |
| **weighted** | 0.3965 | 0.3778 | 0.452 | 0.447 | 0.563 | 0.323 | 0.105 |
| **person_avg** | 0.5081 | 0.4233 | 0.667 | 0.339 | 0.512 | 0.494 | 0.105 |
| **knn_weighted** | 0.4821 | 0.4340 | 0.617 | 0.425 | 0.612 | 0.403 | 0.113 |


## Ensembles with v4

| Ensemble | wF1 | mF1 | Δ vs v4 | SucheOko F1 |
|---|---:|---:|---:|---:|
| `v4_x_protoflat` | 0.6698 | 0.5461 | -0.0189 | 0.0000 |
| `v4_x_protogated` | 0.6487 | 0.5476 | -0.0400 | 0.0526 |
| `v4_x_protogated_oracle_alpha` | 0.6566 | 0.5410 | -0.0321 | 0.0000 |


## Bootstrap (1000x person-level, vs v4)

| Candidate | mean ΔwF1 | 95% CI | P(Δ>0) |
|---|---:|---:|---:|
| protonet_best_by_w | -0.1638 | [-0.2610, -0.0666] | 0.001 |
| v4_x_protoflat | -0.0187 | [-0.0376, -0.0007] | 0.022 |
| v4_x_protogated | -0.0384 | [-0.0899, +0.0107] | 0.065 |
| v4_x_protogated_oracle_alpha | -0.0306 | [-0.0726, +0.0096] | 0.065 |


## Verdict

- **Partial SucheOko rescue:** best variant `knn_weighted` reaches F1 = 0.113 (vs v4 = 0.000). Below the 0.30 threshold; useful signal but not pitch-ready.

- **Best ensemble:** `v4_x_protoflat` -> wF1 = 0.6698 (Δ v4 = -0.0189), SucheOko = 0.0000.

- **No reliable ensemble gain** over v4. Ensemble with v4 does not beat the champion.


## Honest reporting

- Prototypes built strictly from training persons (LOPO guarantee).
- Softmax T fixed at 0.1 (no per-fold tuning). Oracle alpha boost for the gated ensemble was chosen on the OOF set (alpha=0.10); this number is an OPTIMISTIC ceiling and is labeled as such.
- Bootstrap resamples persons with replacement (not scans) to respect person-level evaluation.
