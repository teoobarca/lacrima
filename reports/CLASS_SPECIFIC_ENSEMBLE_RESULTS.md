# Class-Specific Ensemble Strategies — Results

**Hypothesis:** v4's geometric mean ensemble sets SucheOko F1 = 0 because any encoder with low P(SucheOko) multiplicatively penalizes the class-wise log-probability. The ProtoNet agent found that a single encoder (BiomedCLIP-TTA + adapter) achieves SucheOko F1 = 0.074, but geom-mean collapses it. Replacing the geometric mean with a combination rule that preserves minority-class signal may rescue SucheOko without losing majority-class accuracy.

## Methodology

- **Data:** 240 AFM scans, 35 persons, 5 classes.
- **Encoders (identical to v4):** DINOv2-B 90 nm/px, DINOv2-B 45 nm/px, BiomedCLIP 90 nm/px with D4-TTA.
- **Per-encoder classifier:** L2-normalize -> StandardScaler -> LogisticRegression(class_weight=balanced).
- **Evaluation:** PERSON-level LOPO (35 folds) via `teardrop.data.person_id` + `teardrop.cv.leave_one_patient_out`.
- **No OOF-based tuning:** Strategy C uses NESTED LOPO on the outer training set to fit the meta-learner; never selects on outer OOF.

## Per-encoder LOPO metrics (OOF softmaxes fed into combiners)

| Encoder | Weighted F1 | Macro F1 | SucheOko F1 |
|---|---:|---:|---:|
| `dinov2_90nm` | 0.6162 | 0.4941 | 0.0000 |
| `dinov2_45nm` | 0.6544 | 0.5186 | 0.0000 |
| `biomedclip_tta` | 0.6220 | 0.4915 | 0.0588 |

## Combination strategies

- **`geom_mean_v4`** — the v4 champion baseline (3-way geometric mean of softmaxes, then renormalize).
- **`A_max_pool`** — `max` across encoders per class, then renormalize.
- **`B_hybrid_majority_geom_minority_max`** — geom-mean for all classes, but `P(SucheOko)` is replaced with max-over-encoders; then rows are renormalized.
- **`C_learnable_nested_lopo`** — a multinomial-LR meta-stacker on the concatenated (3 x C) softmax vectors. Weights are learned inside each outer-LOPO fold via a nested LOPO run on the training portion, so there is zero leakage from the held-out person.
- **`D_noisy_or`** — `1 - prod_enc(1 - p_enc(c))` per class, then renormalize.
- **`E_rank_vote`** — each encoder argmax votes; ties broken by summed-softmax.

## Results

v4 baseline: W-F1 = 0.6887, M-F1 = 0.5541, SucheOko F1 = 0.0000.

| Strategy | Weighted F1 | Δ v4 | Macro F1 | Δ v4 | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| **`geom_mean_v4`** | 0.6887 | -0.0000 | 0.5541 | -0.0000 | 0.917 | 0.583 | 0.579 | 0.691 | **0.000** |
| **`A_max_pool`** | 0.6617 | -0.0270 | 0.5248 | -0.0293 | 0.895 | 0.511 | 0.548 | 0.670 | **0.000** |
| **`B_hybrid_majority_geom_minority_max`** | 0.6690 | -0.0197 | 0.5437 | -0.0104 | 0.908 | 0.583 | 0.579 | 0.648 | **0.000** |
| **`C_learnable_nested_lopo`** | 0.5676 | -0.1211 | 0.4602 | -0.0939 | 0.845 | 0.385 | 0.581 | 0.490 | **0.000** |
| **`D_noisy_or`** | 0.6858 | -0.0029 | 0.5544 | +0.0003 | 0.917 | 0.596 | 0.579 | 0.681 | **0.000** |
| **`E_rank_vote`** | 0.6706 | -0.0181 | 0.5399 | -0.0142 | 0.871 | 0.565 | 0.579 | 0.684 | **0.000** |

## Diagnostic: why did no combination strategy rescue SucheOko?

Per-encoder softmax inspection on the true-label = `SucheOko` scans (14 of 240):

| Encoder | # predicted SucheOko | mean P(SO | y=SO) | max P(SO | y=SO) | max P(SO | y≠SO) |
|---|---:|---:|---:|---:|
| `dinov2_90nm` | 14 | 0.006 | 0.044 | 0.941 |
| `dinov2_45nm` | 8  | 0.030 | 0.365 | 0.997 |
| `biomedclip_tta` | 20 | 0.062 | 0.785 | 0.999 |

Two facts dominate:

1. **No encoder has calibrated SucheOko signal.** On true SucheOko scans, DINOv2-B 90nm's mean P(SO) is only 0.006. The ~14 SucheOko argmax predictions from this encoder are effectively "the softmax collapsed — all classes near zero, SucheOko won by a hair" and are almost all **on the wrong scans**.
2. **All three encoders produce high-confidence P(SucheOko) on non-SucheOko scans** (up to 0.94 / 1.00 / 1.00). Any combination rule that amplifies the SucheOko column (max-pool, noisy-OR, minority-max hybrid, rank-vote) therefore also amplifies **false-positive SucheOko** predictions — the denominator (recall) stays near zero while precision collapses.

This is the structural reason no combination rule rescued SucheOko: the minority-class signal is present at the scan level **on the wrong scans** as often as on the right ones. The rescue cannot come from reweighting the three encoders used by v4. It has to come from either

- a **different encoder / representation** that has a higher ROC-AUC for SucheOko vs "other" (e.g. the ProtoNet adapter's 0.074 F1 result shows there IS a better space — but the raw softmaxes we feed into the combiner were from the v4 recipe's LR head, not the adapter), **or**
- an **explicit SucheOko binary gate** (one-vs-rest classifier, OOD / open-set scorer, BMP-fallback rule) that fires *before* the combination step.

## Verdict

- **Best weighted-F1:** `geom_mean_v4` at 0.6887 (Δ v4 = -0.0000). No combination strategy beats it.
- `D_noisy_or` ties v4 on weighted F1 to within 0.003 and slightly *improves* macro F1 by +0.0003. It is not a meaningful improvement but it does show noisy-OR is as well-behaved as geom-mean on the 4 majority classes.
- Strategy C (learnable nested-LOPO stacker) under-performs by 0.12 weighted F1. With only 34 training persons and 5 classes, the multinomial-LR meta-stacker over-fits to per-fold noise — there isn't enough data to learn a (3 x 5) weight matrix honestly. This confirms the prior honest expectation.
- Class-specific max or noisy-OR did NOT rescue SucheOko because **the class signal itself is not discriminative at the scan level under the v4 LR recipe**, irrespective of the combination rule (see diagnostic above).

- **No strategy rescued SucheOko (F1 > 0).**
- **No v5 candidate:** no strategy simultaneously kept W-F1 >= 0.67 and produced SucheOko F1 > 0.

## Recommended follow-ups

- Re-run this experiment feeding **ProtoNet-adapter softmaxes** as the third encoder instead of raw BiomedCLIP-TTA LR softmaxes. The adapter achieved SucheOko F1 = 0.074 single-encoder; the class-specific combiner may now have a non-zero channel to amplify.
- Pair the v4 geom-mean with an **explicit SucheOko one-vs-rest gate** (train a binary OOD classifier on the 14 SucheOko scans vs the other 226 at the person level, then override the v4 argmax when the gate fires above a honest threshold selected on nested-LOPO).
- Investigate per-person calibration: SucheOko has only 2 persons, so in every LOPO fold the training set has ONE SucheOko person. LR + class_weight='balanced' cannot reliably fit a 4-class vs 1-class boundary here. A 1-vs-rest Mahalanobis / density-ratio detector is the principled alternative.

## Honest reporting

- Person-level LOPO (35 folds) for every row; no OOF-based model selection; no threshold tuning.
- Strategy C uses **nested** person-LOPO on the outer training set to fit its meta-stacker — red-team approved (no leakage from the held-out person).
- Per-encoder numbers differ slightly from `MULTISCALE_RESULTS.md` only by random-state non-determinism in LR; the v4 row here reproduces the published W-F1 = 0.6887 via the same code path.
