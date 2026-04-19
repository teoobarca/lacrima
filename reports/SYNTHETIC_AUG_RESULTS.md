# Synthetic Augmentation Results — MixUp in Embedding Space

**TL;DR.** Person-LOPO weighted F1, 35 folds.  Champion v4 (no aug): **0.6887** (SucheOko F1 = 0).  MixUp augmentation of minority class embeddings is evaluated against this baseline.

## Method

MixUp in the frozen-encoder embedding space.  For each LOPO fold we
synthesise `n` new per-scan embeddings for minority class(es) by:

1. Drawing two **tiles** (or two scans for BiomedCLIP TTA cache) of the
   target class from the fold's **training set only**, preferably from
   DIFFERENT persons when multiple are available.
2. Mixing with weight `lam ~ Beta(α, α)` squashed into `[0.2, 0.8]` to
   guarantee genuine interpolation rather than near-copy.
3. Averaging `tiles_per_synth_scan` such mixed tiles to form one
   synthetic scan-level embedding (mirrors the real mean-pool pipeline).

Appended to the fold's training set before fitting v2-recipe LR
(L2-row-norm → StandardScaler → LR(class_weight='balanced')).

The held-out person's scans/tiles are **never** used as MixUp ingredients,
so there is no cross-fold leakage.

## Configurations & Results

| Config | Classes | n_synth | Ens. W-F1 | Ens. M-F1 | Healthy | Diabetes | Glaukom | SM | **SucheOko** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_no_aug` | [] | 0 | 0.6887 | 0.5541 | 0.917 | 0.583 | 0.579 | 0.691 | **0.000** |
| `mixup_SucheOko_n50` | [4] | 50 | 0.6854 | 0.5521 | 0.910 | 0.583 | 0.579 | 0.688 | **0.000** |
| `mixup_SucheOko_n100` | [4] | 100 | 0.6839 | 0.5501 | 0.904 | 0.583 | 0.571 | 0.691 | **0.000** |
| `mixup_SucheOko_n200` | [4] | 200 | 0.6824 | 0.5493 | 0.904 | 0.583 | 0.571 | 0.688 | **0.000** |
| `mixup_SucheOko_n100_alpha5` | [4] | 100 | 0.6843 | 0.5506 | 0.910 | 0.583 | 0.571 | 0.688 | **0.000** |
| `mixup_all_min_n100` | [4, 1] | 100 | 0.6721 | 0.5313 | 0.893 | 0.500 | 0.571 | 0.691 | **0.000** |

## Per-component breakdown (ensemble member metrics)

| Config | DINOv2-90 W-F1 | DINOv2-45 W-F1 | BiomedCLIP W-F1 |
|---|---:|---:|---:|
| `baseline_no_aug` | 0.6162 | 0.6544 | 0.6220 |
| `mixup_SucheOko_n50` | 0.6222 | 0.6460 | 0.6037 |
| `mixup_SucheOko_n100` | 0.6264 | 0.6381 | 0.6059 |
| `mixup_SucheOko_n200` | 0.6261 | 0.6380 | 0.6040 |
| `mixup_SucheOko_n100_alpha5` | 0.6264 | 0.6397 | 0.5987 |
| `mixup_all_min_n100` | 0.6189 | 0.6358 | 0.6049 |

## Headline

- **Baseline (no aug, 3-component geom-mean)**: W-F1 = 0.6887, SucheOko F1 = **0.000**.
- **Best MixUp config (`mixup_SucheOko_n50`)**: W-F1 = 0.6854 (Δ = -0.0033), SucheOko F1 = **0.000** (Δ = +0.000).
- **Champion v4 reference** (trained ensemble, published in `models/ensemble_v4_multiscale/`): W-F1 = 0.6887.


## Red-team: nearest-neighbour distance

We check that synthetic SucheOko embeddings are **not** near-copies of
real training tiles (memorisation) and are similarly close/far from
real train vs held-out SucheOko.

```json
{
  "held_out_person": "29EYE_suche_oko",
  "n_synthetic": 50,
  "n_real_train_SucheOko_scans": 6,
  "nn_cosine_distance_to_train_SucheOko": {
    "mean": 0.0730462446808815,
    "median": 0.07190331816673279,
    "min": 0.028871655464172363,
    "max": 0.11910438537597656
  },
  "nn_cosine_distance_to_HELDOUT_test_SucheOko": {
    "mean": 0.3813220262527466,
    "median": 0.37998151779174805,
    "min": 0.3338744044303894,
    "max": 0.4334055781364441
  },
  "memorisation_check": {
    "mean_dist_to_train": 0.0730462446808815,
    "mean_dist_to_heldout": 0.3813220262527466,
    "interpretation": "synthetic should not be dramatically closer to train (<< 0.01) \u2014 that would indicate memorisation of real tiles"
  }
}
```

Interpretation: MixUp operates in the L2-normed 768-d DINOv2 space.
Cosine distances of ≥ 0.01 to the nearest real training tile indicate
the synthetic samples are genuine interpolations, not duplicates.  If
the heldout-test distance is comparable to the train distance, the
synthetic samples have plausibly captured class-level texture.

## Limitations & Honest Caveats

- **Only 2 SucheOko persons** — MixUp can only linearly interpolate
  between what exists.  If the missing SucheOko phenotypes live in a
  non-linear manifold region, convex combinations can't reach them.
- **No image-space validation** — we trust the encoder's inductive
  bias (DINOv2-B was pretrained on natural images, generalises well to
  AFM texture, as evidenced by the champion F1).  A true VAE would
  also need pixel-level sanity checks.
- **Does not escape feature manifold** — any bias in the frozen encoder
  is preserved.  If DINOv2-B lacks SucheOko-specific directions, MixUp
  can't create them.  That's Option A / B's job (future work).
- **Class weight already boosts SucheOko** — the baseline already uses
  `class_weight='balanced'`.  Gains from MixUp must exceed the gains
  from simple reweighting to be meaningful.
