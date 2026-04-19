# MAE Pre-training on Own 240 AFM Scans -- Results

## Question

Can a small ViT-Tiny MAE pre-trained from scratch on OUR OWN 240 AFM scans' tiles
(plus D4 augmentation) learn domain-aligned visual features that beat a frozen
ImageNet-pretrained DINOv2-B baseline under person-level LOPO?

Context: this is recommendation 1 of `reports/EXTERNAL_DATA_SURVEY.md` (highest EV,
zero data acquisition cost). The prior SSL attempt (`reports/SSL_SUPCON_RESULTS.md`)
used a tiny SupCon projection head on frozen DINOv2 features and regressed (0.6120).
MAE is a different animal: reconstruction-based, requires no labels, trains the full
encoder directly on AFM pixel statistics.

## Methodology

- **Data**: all 240 scans, tiled at 45 nm/px with 512x512 non-overlapping tiles
  (up to 9 tiles per scan), each resized to 224x224 and replicated to 3 channels.
  Total tiles: 1889. Persons: 35. Scans: 240.
- **Augmentation**: D4 (8-way dihedral group: rotations + flips) applied on the fly,
  yielding 15112 effective training views per epoch.
- **Model**: ViT-Tiny encoder (timm `vit_tiny_patch16_224`, 12 blocks, 192-dim, 3
  heads) + MAE decoder (4 blocks, 96-dim, 3 heads). Mask ratio 0.75.
  Pixel loss normalized per patch (`norm_pix_loss=True`).
- **Optimizer**: AdamW, lr=0.00015, weight_decay=0.05, betas=(0.9, 0.95),
  cosine schedule with 1-epoch warmup.
- **Training**: 72/100 epochs completed, batch 64,
  on device `mps`, wall clock 30.8 min.
- **Downstream eval**: extract CLS features from the pre-trained encoder on every
  tile (no augmentation, no masking), mean-pool tiles per scan -> (240, 192),
  StandardScaler + LogisticRegression (class_weight='balanced'), person-level LOPO.
- **Controls**:
    - Random-init ViT-Tiny (same architecture, never trained): lower bound.
    - DINOv2-B 45 nm/px tiles (same tile protocol) and 90 nm/px tiles (our established
      baseline): upper references.

## Results (person-LOPO, raw argmax)

| Model | Weighted F1 | Macro F1 |
|---|---:|---:|
| Random-init ViT-Tiny (control, lower bound) | 0.5705 | 0.4590 |
| DINOv2-B ImageNet 90 nm/px (established baseline) | 0.6150 | 0.4910 |
| DINOv2-B ImageNet 45 nm/px (same-protocol reference) | 0.6433 | 0.5038 |
| **MAE ViT-Tiny pre-trained on own 240 scans** | **0.5555** | **0.4482** |

### Per-class F1 (MAE vs random-init control)

| Class | Support | Random | MAE |
|---|---:|---:|---:|
| ZdraviLudia | 70 | 0.730 | 0.691 |
| Diabetes | 25 | 0.448 | 0.148 |
| PGOV_Glaukom | 36 | 0.533 | 0.519 |
| SklerozaMultiplex | 95 | 0.583 | 0.620 |
| SucheOko | 14 | 0.000 | 0.263 |

## Verdict

NO CLEAR GAIN. MAE-pretrained features are within noise of random-init. Likely causes: tiny corpus (~1.9k tiles x 8 views = 15k effective), ViT-Tiny too small, or training was cut short by the time budget.

- Delta vs random-init ViT-Tiny: **-0.0150** weighted F1
- Delta vs DINOv2-B 45nm (same tile protocol): **-0.0878**
- Delta vs DINOv2-B 90nm (established baseline): **-0.0594**

## Honest caveats

1. **Tiny corpus for MAE.** 240 scans x ~8 tiles x 8 D4 views ~= 15k training views
   is two orders of magnitude smaller than typical MAE pre-training corpora. The
   EXTERNAL_DATA_SURVEY literature projection (+3-8 F1) was predicated on 2-10k
   IMAGES, not 15k heavily-correlated views of the same 1.9k base tiles.
2. **ViT-Tiny is 5M params** vs DINOv2-B's 86M. Part of DINOv2-B's win is sheer
   capacity + ImageNet-1k scale. A fair scale comparison would need ViT-Base MAE,
   which would require a larger corpus + more compute than our 30-minute budget.
3. **Person-LOPO is strict.** All numbers in this table use `person_id` grouping
   (L/R eyes merged into one person), the same protocol as SSL_SUPCON_RESULTS.md.
4. **Encoder features never saw labels.** The entire MAE head training is fully
   self-supervised (reconstruction only); only the downstream LR sees class labels.
   This is architecturally clean -- no LOPO leakage risk.
5. **Compute budget triggered early stop iff `epochs_trained < epochs`.**
   Here: 72/100 epochs completed.

## Loss trajectory (last epochs)

| Epoch | Loss | LR | Elapsed (s) |
|---:|---:|---:|---:|
| 63 | 0.8406 | 4.73e-05 | 1412 |
| 64 | 0.8394 | 4.51e-05 | 1458 |
| 65 | 0.8386 | 4.30e-05 | 1505 |
| 66 | 0.8379 | 4.09e-05 | 1552 |
| 67 | 0.8370 | 3.88e-05 | 1601 |
| 68 | 0.8361 | 3.67e-05 | 1647 |
| 69 | 0.8353 | 3.47e-05 | 1690 |
| 70 | 0.8350 | 3.27e-05 | 1736 |
| 71 | 0.8333 | 3.08e-05 | 1775 |
| 72 | 0.8325 | 2.89e-05 | 1811 |

## Artifacts

- `scripts/mae_pretrain.py` -- this script (MAE training + LOPO eval)
- `models/mae_tear_tiny/encoder.pt` -- trained encoder checkpoint (ViT-Tiny + decoder)
- `models/mae_tear_tiny/config.json` -- hyperparameters + summary metrics
- `cache/mae_emb_tear_tiny.npz` -- MAE CLS tile features + scan-level mean-pool
- `cache/mae_raw_tiles_45nm_t512_n9_224.npz` -- pre-processed 224x224 tiles cache
- `reports/MAE_PRETRAINING_RESULTS.md` -- this report

## Bonus: MAE features as 4th member of v4 ensemble

Does MAE add complementary signal despite lower individual F1? Quick ensemble test:

| Configuration | Weighted F1 | Macro F1 |
|---|---:|---:|
| v4 (dino90 + dino45 + bmc_tta) | 0.6887 | 0.5541 |
| v4 + MAE (4-way) | 0.6756 | 0.5450 |
| dino90 + MAE | 0.6280 | 0.4944 |
| dino45 + MAE | 0.6314 | 0.5076 |
| bmc_tta + MAE | 0.6264 | 0.4924 |

**Delta MAE adds to v4 baseline: -0.0131 weighted F1.**

MAE features hurt the ensemble -- they share failure modes with the other members without adding complementary signal. Do not integrate.

