# Test-Time Augmentation (TTA) Results

## Question

Current honest champion: DINOv2-B + BiomedCLIP proba-avg, raw-argmax weighted F1 = 0.6346 (person-LOPO, 240 scans, 35 persons).
Does adding D4 test-time augmentation to the encoded tiles close the gap to the threshold-tuned 0.6528 without any tuning?

## Methodology

For each of the 240 scans:

1. Load via `AFMReader.spm`, plane-level, resample to 90 nm/px, robust-normalize (2..98 percentile clip) -- identical to the tiled baseline.
2. Split into up to 9 non-overlapping 512x512 tiles.

3. For each tile, apply the dihedral group D4 = {id, r90, r180, r270, flipLR, flipLR+r90, flipLR+r180, flipLR+r270}. This yields 8 augmented views per tile.
4. Render each augmented tile with the `afmhot` colormap (matplotlib) -> PIL RGB.
5. Feed all 72 PIL images to the encoder in batches of 16; mean-pool over the 72 embeddings to a single (1, D) scan vector.

Encoders: DINOv2-B (ViT-B/14, D=768) and BiomedCLIP (ViT-B/16, D=512).

Evaluation: person-level LOPO (via `teardrop.data.person_id` -> 35 groups). Per fold: `StandardScaler` -> `LogisticRegression(class_weight='balanced', max_iter=3000, C=1.0)`. No threshold tuning, no nested CV. Raw argmax.

Non-TTA baselines are re-computed with the exact same LOPO loop using the cached `tiled_emb_*_t512_n9.npz` embeddings mean-pooled to scan level (1 tile * 1 aug = 9 embeddings per scan -> mean).

## Results (person-LOPO, raw argmax)

### Weighted F1

| model | non-TTA | TTA (D4) | Delta |
|---|---:|---:|---:|
| DINOv2-B scan-mean | 0.6150 | 0.6434 | +0.0284 |
| BiomedCLIP scan-mean | 0.5841 | 0.6135 | +0.0294 |
| **Ensemble proba-avg (raw argmax)** | **0.6346** | **0.6458** | **+0.0112** |

### Macro F1

| model | non-TTA | TTA (D4) | Delta |
|---|---:|---:|---:|
| DINOv2-B scan-mean | 0.4910 | 0.5275 | +0.0366 |
| BiomedCLIP scan-mean | 0.4385 | 0.4838 | +0.0453 |
| Ensemble proba-avg (raw argmax) | 0.4934 | 0.5154 | +0.0221 |

## Verdict

**SHIP (with honest caveat): TTA gives +0.0112 weighted F1 on the ensemble (right at the +0.01 threshold). Strong +0.028..+0.029 per-encoder gains confirm the signal is real; most of it redundantly overlaps between the two encoders so ensemble gain is smaller.**

Decision rule (pre-registered): Ensemble Delta-weighted-F1 >= +0.01 -> ship TTA'd model as new champion and save to `models/ensemble_v1_tta/`. +0.005..+0.01 -> marginal, report but stay with non-TTA. |Delta| < +0.005 -> noise, stay. Delta < -0.005 -> negative, stay.

Observed ensemble Delta = +0.0112 weighted F1, Delta macro = +0.0221.

**Important nuance:** the per-encoder lifts (DINOv2 +0.0284, BiomedCLIP +0.0294) are ~2-3x larger than the ensemble lift (+0.0112). TTA reduces orientation-noise in each encoder's embedding, but the *corrections* overlap heavily between encoders (both see the same augmented tiles; both learn dihedrally-invariant features when averaged). The ensemble was already doing some of TTA's job via two encoders voting. Net: TTA still wins, but it also implies that a single TTA'd encoder is almost as good as the non-TTA 2-encoder ensemble (0.6434 vs 0.6346), which is a simpler deployment if you want to drop BiomedCLIP.

At 240 scans, +0.01 is on the edge of sampling noise (~+/-0.02 bootstrap). The consistency of the gain across DINOv2, BiomedCLIP, *and* ensemble (all positive, all in the +0.011..+0.029 band for weighted F1, +0.022..+0.045 for macro F1) is stronger evidence than any single number: TTA's effect is systematic, not a one-draw fluke.

## Compute cost (wall clock)

- DINOv2-B TTA encode (240 scans * up to 72 images): 327.2 s
- BiomedCLIP TTA encode: 262.3 s
- LOPO eval (5 LR runs over 35 persons): 2.8 s
- Total wall: 595.3 s

Device: MPS (Apple Silicon). Batch size 16.

## Interpretation

Mean-pooling over the D4 orbit approximately enforces dihedral invariance on the final scan embedding. For AFM scans, the physical sample has no preferred orientation (the scanner head / sample rotation is arbitrary), so the ground-truth class is D4-invariant. Any variance in the encoder's output under D4 is therefore noise, and averaging over the orbit should reduce it -- provided the encoder is not already nearly-equivariant.

DINOv2 and BiomedCLIP are trained on natural/medical images with a *canonical* orientation (faces up, horizons level, etc.), so they are probably NOT orientation-invariant by default. This is what gives TTA a theoretical foothold.

With only 240 scans the sampling noise of a single weighted-F1 point is roughly +/- 0.02 (back-of-envelope: one class flip changes F1 by 1/240 in a uniform ideal, practically more due to class imbalance). So a gain has to clear +0.01 to even be distinguishable from bootstrap variation, and +0.02+ to be confidently real.

## Comparison to the existing champion landscape

| model | person-LOPO weighted F1 | notes |
|---|---:|---|
| DINOv2-B scan-mean (baseline, non-TTA) | 0.6150 | single-model simple baseline |
| DINOv2-B scan-mean + D4 TTA (this work) | **0.6434** | +0.028 from TTA alone |
| BiomedCLIP scan-mean (non-TTA) | 0.5841 | |
| BiomedCLIP scan-mean + D4 TTA (this work) | 0.6135 | +0.029 from TTA alone |
| DINOv2-B + BiomedCLIP proba-avg (non-TTA, raw argmax) | 0.6346 | previous honest champion |
| **DINOv2-B + BiomedCLIP proba-avg + D4 TTA (raw argmax, this work)** | **0.6458** | new honest champion |
| DINOv2-B + BiomedCLIP proba-avg (non-TTA) + nested per-class thresholds | 0.6528 | still slightly higher, but needs threshold tuning (nested CV) |

TTA closes most of the gap between the 0.6346 raw-argmax ensemble and the 0.6528 threshold-tuned ensemble, *without* tuning anything. That is the main pitch-relevant finding: the +0.0112 lift from D4 TTA is roughly half of the +0.0182 lift from nested per-class thresholding, but it is obtained entirely from honest invariance averaging, no OOF-peeking or threshold-sweeping required.

## Artifacts

- `cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz` -- (240, 768) mean-pooled TTA DINOv2-B embeddings
- `cache/tta_emb_biomedclip_afmhot_t512_n9_d4.npz` -- (240, 512) mean-pooled TTA BiomedCLIP embeddings
- `scripts/tta_experiment.py` -- end-to-end experiment (encoding + LOPO eval + report generation)
- `scripts/train_ensemble_tta_model.py` -- trains and saves the shippable ensemble
- `models/ensemble_v1_tta/` -- saved `EnsembleClassifierBundle`. **Inference-time requirement:** caller must expand tiles via the D4 orbit (see `scripts/tta_experiment.py::d4_augmentations`) before calling `bundle.predict_proba_from_tiles`. Config records `tta=D4`, `tta_group_order=8`.
- `reports/TTA_RESULTS.md` (this file)
