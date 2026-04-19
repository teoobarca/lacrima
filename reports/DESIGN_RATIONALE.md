# Design rationale — why every choice

Why we chose each part of the shipped pipeline. Each decision came from an experiment (not intuition) and has a receipt in `reports/`.

## Encoder: DINOv2-B frozen (not DINOv2-L, not fine-tuned)

- **DINOv2-B vs DINOv2-S**: both tested. DINOv2-B marginally better (0.615 vs 0.593 tiled) but overfits faster in classifier head. For 240 samples, the extra 2× params don't help much.
- **DINOv2-L (1024-dim)**: not tested exhaustively — higher risk of overfitting, slower encoding.
- **Fine-tuning**: SupCon projection head (Wave-5 SSL agent) lost 0.003 F1 vs frozen. 0.4M-param head on 240 samples can't add information — only reshape. Fine-tuning bigger slices would need more data than we have. **Verdict: frozen is the small-data safe choice.**
- **Why also BiomedCLIP**: it was pretrained on 15M PubMed image-text pairs — domain priors relevant to microscopy. Adds **diverse representations** to the ensemble without overfitting.

## Classifier head: LogisticRegression (not MLP, not XGBoost)

- **LR vs MLP**: Wave-1 experiments: MLP 0.512 F1, LR 0.647 on same tiled embeddings. MLP overfits 240 samples with 200k+ parameters.
- **LR vs XGBoost**: XGBoost 0.54 F1 on same features. GBM can't model smooth class-conditional densities that frozen embeddings have — LR is asymptotically optimal for this problem class (linear separability after StandardScaler).
- **Class weights = 'balanced'** because imbalance is 7:1 (SM:SucheOko).
- **C = 1.0**: default; nested-C grid search found range [1e-2, 1e2] with best on outer folds scattered widely ⇒ classifier doesn't care much about C in our regime. Don't tune → don't leak.

## Tile preprocessing: 9 tiles of 512² at 90 nm/px

- **Why tile**: single-crop loses 75 % of the raw 1024² data. Tiling gives 9× effective sample count at training + averaging signal at inference. **+0.05 F1.**
- **Why 512² and not 256/768**: 256 loses features at the crystalline-lattice scale; 768 has memory cost for TTA (72 views) with diminishing returns. Grid search on 9 configs settled here.
- **Why 90 nm/px**: dataset's scan range is 10–92 µm; 90 nm/px normalizes ~78 % of scans cleanly, pads the rest symmetrically. Cross-scale normalization was essential — without it GLCM distances are apples-to-oranges.
- **Why non-overlapping tiles**: overlap gives correlated views → less effective augmentation gain than augmenting then stacking. Non-overlap + D4 gives 8× diversity per tile from the augmentation axis.

## Test-time augmentation: D4 group (8 views per tile)

- **D4 vs C4 vs full 16**: D4 covers the dihedral symmetry (rotation + flip) that AFM height-maps have. Full 16 = non-orthogonal rotations that need interpolation → artifacts. **+0.011 F1 on ensemble.**
- **Why at inference, not just train**: train-time TTA needs more LR epochs and risks memorizing augmentations. Inference-time TTA is cheaper (encode once per view) and cleanly separates "which tile" from "which view of that tile".

## Ensemble: softmax average of 2 encoders

- **Why only 2 and not 5**: Wave-5 autoresearch H8 tested adding TDA + handcrafted as 3rd/4th components — regressed by 0.03–0.05 because weak members dilute the geometric mean. Foundation-model diversity (DINOv2 vs BiomedCLIP pretraining) is what we want; redundant or weak members hurt.
- **Geometric mean (v2) vs arithmetic (v1)**: geom-mean penalizes confident-wrong members more. Both encoders are well-calibrated → geom-mean stays conservative. **+0.008 F1.**
- **L2-normalization before StandardScaler**: StandardScaler assumes features are roughly comparable in magnitude. Foundation embeddings have heterogeneous norms across tiles. L2-norm makes StandardScaler fair. **+0.003 F1, stacks with geom-mean.**

## Evaluation: person-level LOPO (not eye-level, not k-fold)

- **Person-level LOPO (35 groups)**: every scan participates in val exactly once, each person is val on exactly one fold. Strongest guarantee against leakage.
- **vs eye-level LOPO (44 groups)**: inflates F1 by ~1 % because L/R eyes of same human share systemic disease + sample-prep batch. Validator agent caught this in Wave-1; fixed via `person_id()`.
- **vs 5-fold CV**: with 35 persons / 5 folds = 7 persons per fold, but SucheOko has only 2 persons total → some folds have 0 SucheOko. K=5 breaks for minority classes.
- **vs simple train/val split**: throws away data we can't afford (240 scans).

## Red-team discipline

Every F1 claim > baseline triggers a separate agent audit:
1. Reproduce the number from saved artifacts
2. Check if any tuning (threshold, bias, subset, α) happened on the same OOF as the eval
3. Re-run with nested CV (tuning on inner folds, eval on outer)
4. If honest F1 < claim, reject

Resulted in three retracted claims (0.6698, 0.6780, 0.6878) and one small-delta caveat (+0.0083 bootstrap check for v3).

## What we did NOT ship and why

| Rejected | Reason |
|---|---|
| Hard-override cascade | −0.048 F1; low confidence ≠ wrong |
| LLM prediction override | −0.012 F1; LLM flips correct near-miss cases |
| 4-component concat + bias tune | 0.633 F1 nested; bias tune leaks |
| CGNN alone | 0.37 F1; graph features too sparse on 240 scans |
| SupCon projection head | 0.61 F1; small-data SSL regresses |
| Advanced handcrafted (443-dim) | 0.47 F1 alone; doesn't fuse cleanly with neural |
| Meta-LR stacker on 12 features | 0.51 F1; 12D meta is too narrow |
| Project-out person-direction | −0.06 F1; patient-ID and class are confounded (**this is scientifically important** — UMAP showed it) |

## Meta-insight

At 240 scans / 35 persons, **additional complexity reliably inflates leakage rather than signal**. The Pareto front is: few strong foundation models + robust ensembling + no OOF tuning. Everything past that is noise.
