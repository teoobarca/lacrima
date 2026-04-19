# MAE Pre-training + Fine-tune vs v4 Champion

## TL;DR

**NEGATIVE RESULT** (delta -0.1160 wF1, worse by >2 pp). In-domain MAE on 240 scans does not beat frozen DINOv2-B. Documented.

- v4 champion (frozen DINOv2-B + BiomedCLIP geomean): wF1 = 0.6887, mF1 = 0.5541
- Best MAE variant (`mlp` head on MAE-ViT-Tiny): wF1 = 0.5727, mF1 = 0.4632
- Bootstrap delta (MAE - v4), 1000 resamples: mean -0.1165 wF1, CI95 [-0.1756, -0.0569], P(delta>0) = 0.000

## Question

Literature (MICCAI 2024) reports +3-8 F1 from MAE pretraining on similar-size medical datasets. Does in-domain MAE pretraining on our 240 AFM scans beat frozen generic-pretrained DINOv2-B for tear-ferning classification?

## Protocol

1. **Pretrain** (already done; see `reports/MAE_PRETRAINING_RESULTS.md` and `scripts/mae_pretrain.py`): ViT-Tiny MAE from scratch on all 240 scans' tiles with D4 augmentation, 75% mask ratio, 72/100 epochs on MPS (time-budget stop). `norm_pix_loss=True`. Loss trajectory 0.85 -> 0.83, still declining at stop but shallow slope.
2. **Fine-tune**: load MAE encoder, extract CLS features (mean-pool of tiles per scan), train classification head under person-level LOPO. Two head variants:
   - `linear`: BatchNorm + single Linear layer (MAE paper's canonical probe)
   - `mlp`:    BatchNorm + 2-layer MLP (192 -> 128 -> 5) with dropout 0.3
3. **Optimizer**: AdamW, lr=1e-3, weight_decay=1e-4, 200 epochs full-batch, cosine LR, cross-entropy with balanced class weights.

### Head choice: why frozen (not LoRA)

- **Overfit risk**: 72-epoch encoder trained on ~15k heavily-correlated views (1.9k base tiles x 8 D4). Updating encoder weights per fold likely adapts to the 34-person train set and hurts 1-person val. LOPO has no per-fold early-stop signal.
- **Apples-to-apples**: `mae_pretrain.py`'s LR-probe (0.5555 wF1) already tested the frozen-encoder quality. A linear probe here is the neural equivalent; large gap would indicate bug, not encoder weakness.
- **MAE paper precedent**: He et al. 2022 use linear probe as the canonical metric for representation quality.

## Results (person-LOPO, 35 folds)

| Variant | Weighted F1 | Macro F1 | Delta vs v4 |
|---|---:|---:|---:|
| MAE-ViT-Tiny + linear head | 0.5523 | 0.4624 | -0.1363 |
| MAE-ViT-Tiny + mlp head | 0.5727 | 0.4632 | -0.1160 |
| **v4 champion (DINOv2-B + BiomedCLIP geomean)** | **0.6887** | **0.5541** | reference |

### Per-class F1 (best MAE variant vs v4)

| Class | Support | Best MAE | v4 |
|---|---:|---:|---:|
| ZdraviLudia | 70 | 0.771 | 0.917 |
| Diabetes | 25 | 0.240 | 0.583 |
| PGOV_Glaukom | 36 | 0.506 | 0.579 |
| SklerozaMultiplex | 95 | 0.593 | 0.691 |
| SucheOko | 14 | 0.205 | 0.000 |

## Bootstrap (1000 resamples, weighted F1 delta MAE - v4)

- Mean delta: **-0.1165** wF1
- 95% CI: [-0.1756, -0.0569]
- P(MAE > v4): **0.000**
- P(MAE > v4) macro: 0.005 (CI95 [-0.1545, -0.0276])

## Verdict

**NEGATIVE RESULT** (delta -0.1160 wF1, worse by >2 pp). In-domain MAE on 240 scans does not beat frozen DINOv2-B. Documented.

## Caveats (important)

1. **Corpus size**: 240 scans x ~8 tiles x 8 D4 views = ~15k effective views is two orders of magnitude below typical MAE corpora. MICCAI-grade gains were predicated on ~2-10k distinct medical images, not highly-correlated patches of 240 scans.
2. **Model size**: ViT-Tiny (5M params) vs v4's DINOv2-B (86M params, trained on 142M curated web images). Capacity and pretraining-data advantages favor v4 intrinsically.
3. **Time-budget stop at epoch 72/100**: loss was still declining at a shallow rate (0.8325 at ep.72; see mae_pretrain history). More epochs *might* improve 1-2 wF1 pp; unlikely to close a >10 pp gap.
4. **Alternative not pursued**: ViT-S MAE initialized from MAE-IN1k checkpoint and fine-tuned on AFM tiles. This hybrid (in-domain continual MAE) is the next-most-promising follow-up per EXTERNAL_DATA_SURVEY.md.

## Artifacts

- `scripts/mae_pretrain.py`            - MAE pre-training (step 1)
- `scripts/mae_finetune.py`            - fine-tune + LOPO eval (this script)
- `models/mae_tear_tiny/encoder.pt`    - MAE ViT-Tiny weights (72 epochs)
- `cache/mae_emb_tear_tiny.npz`        - cached CLS tile features
- `cache/mae_predictions.json`         - OOF predictions + bootstrap payload
- `reports/MAE_PRETRAINING.md`         - this report
- `reports/MAE_PRETRAINING_RESULTS.md` - prior LR-probe eval (reference)
