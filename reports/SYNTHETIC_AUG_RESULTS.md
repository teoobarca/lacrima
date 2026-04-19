# Synthetic Augmentation Results — MixUp in Embedding Space

## TL;DR (Honest Negative Result)

- **Baseline v4 (no aug)**: weighted-F1 = **0.6887**, macro-F1 = 0.5541, **SucheOko F1 = 0.000**.
- **Best MixUp config (`noise_sigma03_n100`)**: weighted-F1 = 0.6850 (Δ = -0.0037), SucheOko F1 = **0.000** (Δ = +0.000).
- **None of the 9 tested MixUp / extrapolation / noise-injection
  configurations produced a single correct SucheOko prediction under
  person-LOPO.**  SucheOko F1 remained exactly 0.000.
- Weighted-F1 mildly regressed in every aug config (−0.003 to −0.017),
  because synthesising minor-class noise slightly dilutes the LR
  decision boundary for the majority classes without improving the
  minority.

This is a clean negative result: **Option C (feature-space MixUp) is
insufficient for the SucheOko problem, and we diagnose why below.**

## Method

Three synthesis modes evaluated, all operating in the
frozen-encoder embedding spaces of the v4 champion (DINOv2-B 90 nm,
DINOv2-B 45 nm, BiomedCLIP-90-TTA):

1. **`interp` (classic MixUp):** convex combinations `lam·A + (1-lam)·B`
   with `lam ~ Beta(α, α)` squashed to `[0.2, 0.8]`.  A and B are
   drawn from the same class, **different persons** when available.
2. **`extrap` (extrapolative MixUp):** `lam ~ Uniform(-0.5, 1.5)` or
   `Uniform(-1.0, 2.0)` — synthetic points live **outside** the
   convex hull of the parents, inflating the effective class variance.
3. **`noise` (Gaussian noise injection):** `X_synth = X_real + σ·ε`,
   `ε ~ N(0, I)`.  Captures local variation without mixing persons.

Tile-level synthesis for DINOv2 caches (more diversity: SucheOko has
45 tiles at 90 nm/px, 110 at 45 nm/px), scan-level for BiomedCLIP.
Synthetic tiles are mean-pooled into scan-level examples that mirror
the real pipeline exactly.

**Leakage safety:** in each LOPO fold, synthesis draws ONLY from the
fold's training tiles — the held-out person never contributes a tile
to any synthetic sample.  Verified by running `baseline_no_aug` and
matching the published champion F1 exactly (0.6887 W-F1).

Classifier per v4 component: `L2-row-norm → StandardScaler →
LogisticRegression(class_weight='balanced', C)`.  Ensemble =
geometric mean of the 3 component softmaxes.

## Configurations & Results

| Config | Classes | n_synth | Ens. W-F1 | Ens. M-F1 | Healthy | Diabetes | Glaukom | SM | **SucheOko** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_no_aug` | [] | 0 | 0.6887 | 0.5541 | 0.917 | 0.583 | 0.579 | 0.691 | **0.000** |
| `mixup_interp_n100` | [4] | 100 | 0.6839 | 0.5501 | 0.904 | 0.583 | 0.571 | 0.691 | **0.000** |
| `mixup_interp_n200` | [4] | 200 | 0.6824 | 0.5493 | 0.904 | 0.583 | 0.571 | 0.688 | **0.000** |
| `mixup_extrap_n100` | [4] | 100 | 0.6826 | 0.5507 | 0.910 | 0.583 | 0.579 | 0.681 | **0.000** |
| `mixup_extrap_n200_aggr` | [4] | 200 | 0.6841 | 0.5514 | 0.910 | 0.583 | 0.579 | 0.684 | **0.000** |
| `noise_sigma03_n100` | [4] | 100 | 0.6850 | 0.5516 | 0.904 | 0.583 | 0.579 | 0.691 | **0.000** |
| `noise_sigma05_n200` | [4] | 200 | 0.6826 | 0.5507 | 0.910 | 0.583 | 0.579 | 0.681 | **0.000** |
| `mixup_extrap_noise_n200` | [4] | 200 | 0.6822 | 0.5502 | 0.904 | 0.583 | 0.579 | 0.684 | **0.000** |
| `mixup_SO+Diab_n100` | [4, 1] | 100 | 0.6721 | 0.5313 | 0.893 | 0.500 | 0.571 | 0.691 | **0.000** |
| `mixup_extrap_n200_lowC` | [4] | 200 | 0.6822 | 0.5502 | 0.904 | 0.583 | 0.579 | 0.684 | **0.000** |

## Per-component breakdown (ensemble member metrics)

| Config | DINOv2-90 W-F1 | DINOv2-45 W-F1 | BiomedCLIP W-F1 |
|---|---:|---:|---:|
| `baseline_no_aug` | 0.6162 | 0.6544 | 0.6220 |
| `mixup_interp_n100` | 0.6264 | 0.6381 | 0.6059 |
| `mixup_interp_n200` | 0.6261 | 0.6380 | 0.6040 |
| `mixup_extrap_n100` | 0.6180 | 0.6453 | 0.6281 |
| `mixup_extrap_n200_aggr` | 0.6200 | 0.6421 | 0.6218 |
| `noise_sigma03_n100` | 0.6235 | 0.6430 | 0.6205 |
| `noise_sigma05_n200` | 0.6320 | 0.6397 | 0.6168 |
| `mixup_extrap_noise_n200` | 0.6243 | 0.6420 | 0.6168 |
| `mixup_SO+Diab_n100` | 0.6189 | 0.6358 | 0.6049 |
| `mixup_extrap_n200_lowC` | 0.6250 | 0.6417 | 0.6153 |


## Red-team: nearest-neighbour analysis (why it fails)

For the held-out SucheOko fold we compared synthetic SucheOko
embeddings to (a) their nearest **real training** SucheOko scan, and
(b) the **held-out test** person's SucheOko scans.

```json
{
  "held_out_person": "29EYE_suche_oko",
  "n_synthetic": 100,
  "n_real_train_SucheOko_scans": 6,
  "nn_cosine_distance_to_train_SucheOko": {
    "mean": 0.06935916095972061,
    "median": 0.06946039199829102,
    "min": 0.028871655464172363,
    "max": 0.11910438537597656
  },
  "nn_cosine_distance_to_HELDOUT_test_SucheOko": {
    "mean": 0.38294827938079834,
    "median": 0.38029271364212036,
    "min": 0.3338744044303894,
    "max": 0.4477241039276123
  },
  "memorisation_check": {
    "mean_dist_to_train": 0.06935916095972061,
    "mean_dist_to_heldout": 0.38294827938079834,
    "interpretation": "synthetic should not be dramatically closer to train (<< 0.01) \u2014 that would indicate memorisation of real tiles"
  }
}
```

**The key diagnostic number: the held-out SucheOko person sits at
mean cosine distance ≈ 0.38 from training SucheOko**, while synthetic
samples live at ≈ 0.07 from training.  In LOPO on a SucheOko fold the
training set contains tiles from exactly ONE SucheOko person (the
other one is held out), so MixUp of same-person tiles stays inside a
tight ball around that person and cannot cover the held-out person's
region of DINOv2-B space.  Extrapolative MixUp (up to λ ∈ [-1, 2])
pushes further, but still along the straight line between two points
of the SAME person, so it expands the cloud in an almost-random
direction rather than toward the missing phenotype.

In classifier terms: under LOPO, LR puts `P(SucheOko | held-out
scan) ≈ 0.01` and ranks SucheOko 3rd-5th on every held-out scan.  SM
(95-scan majority) dominates.  Augmentation that does not push
synthetic points toward where the *unseen* SucheOko person lives
cannot fix this.

## Why Option C Fails for This Problem (Root Cause)

1. **2 persons ≠ 2 modes.**  Feature-space MixUp assumes convex
   combinations of existing examples cover the class distribution.
   With only 2 SucheOko persons and LOPO holding one out, training
   sees **one single-person cloud**, whose convex/extrapolated hull
   is still centred near that person.
2. **Inter-person distance (0.38) ≫ intra-person spread.**  The two
   SucheOko persons live in distant regions of the embedding
   manifold.  Neither MixUp nor Gaussian noise at realistic σ bridges
   that gap.
3. **Class-weight already maxed.**  The baseline uses
   `class_weight='balanced'` (14× SucheOko up-weighting), yet SucheOko
   probs still average ~0.01 — the LR direction for SucheOko is
   simply not distinguishable from SM in the held-out-person region.
4. **Synthetic dilution hurts majority.**  Adding 100-200 synthetic
   SucheOko embeddings that look like the single training person
   marginally shifts the SM boundary, causing −0.005 to −0.017 W-F1
   without any minority gain.

This is the **fundamental few-persons-per-class limitation** that
Option C cannot overcome.  A genuine fix needs either:
- **More SucheOko patients** (data acquisition),
- **A generative model that learns the tear-ferning manifold across
  classes and can synthesise unseen-phenotype samples** (Option A/B:
  class-conditional VAE / latent diffusion on raw AFM tiles),
- **Cross-task transfer** (pretrain on OTHER dry-eye AFM datasets,
  e.g. public tear-ferning images from Masmali scale studies).

## Recommendation

**Do not ship MixUp augmentation for SucheOko.**  Keep the v4 champion
as-is (0.6887 W-F1).  Future work:
- Option A (class-conditional VAE on 64×64 AFM tiles) has the same
  2-persons-in-train problem and will almost certainly mode-collapse
  to the single training persona — risky.
- Option B (latent diffusion, 5M params) is even more data-hungry:
  from 45 SucheOko tiles it will memorise or collapse.
- **Most promising direction**: semi-supervised / cross-dataset
  pretraining on external tear-ferning images (Masmali scale),
  which isn't "synthetic" but is the only source of additional
  SucheOko phenotype diversity.

The negative result IS useful: it precisely quantifies the ceiling
of embedding-space augmentation when the minority class has only 2
persons, and should be cited against future "just MixUp it"
suggestions.

## Reproduction

```bash
.venv/bin/python -W ignore scripts/synthetic_augmentation.py
```
Runs in ~80 seconds CPU.  Produces `reports/synthetic_aug_results.json`
and this file.  Baseline `no_aug` W-F1 exactly reproduces the published
v4 champion (0.6887) — proves the
pipeline faithfully mirrors `scripts/train_ensemble_v4_multiscale.py`.
