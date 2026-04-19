# LoRA fine-tuning of DINOv2-B on AFM tear scans

## Verdict

**Negative result.** Solo LoRA wF1 = **0.6476** trails v4 champion (0.6887) by **−0.041**; paired bootstrap gives **P(LoRA > v4) = 0.097** — far below the 0.90 champion threshold. Geometric-mean ensemble with v4 (0.6774) also comes in **below** v4 solo — the LoRA probabilities actively *drag the ensemble down*. The single silver lining is SucheOko improves from 0.000 → 0.091 (1/14 correct), but net per-class F1 regresses on Diabetes (0.58 → 0.38) and ZdraviLudia (0.92 → 0.82).

**Likely causes:**
- **Too little data for full fine-tuning.** 240 scans / 35 persons with 595 k trainable params → high risk of overfitting despite LoRA.
- **Frozen DINOv2-B embeddings are already near-optimal** for this regime — the Wave 5 logistic head on frozen features extracts most of what a linear classifier can use, and LoRA's added capacity mainly hurts Diabetes (noisier texture prior).
- **Inter-fold variance is large** (per-fold inner-val wF1 ranged 0.61 → 0.90, best-epoch ranged 2 → 11) — a single run is not robustly better/worse than v4, but the bootstrap CI `[-0.103, +0.018]` does not cross the champion threshold.
- **5-fold outer, not full 35-fold LOPO** (timebox simplification) — but that simplification *should if anything help* LoRA (more training data per fold), so it isn't the reason for the deficit.

## Setup

- **Backbone:** `facebook/dinov2-base` (86.6 M params, HF Transformers).
- **PEFT:** LoRA r=8, α=16, dropout=0.1 targeting `attention.{query, key, value, output.dense}` across all 12 transformer layers (regex-scoped, MLP FCs NOT included). 595 205 trainable params (0.68 % of backbone).
- **Head:** `LayerNorm(768) + Linear(768, 5)` on CLS embedding.
- **Input:** per-scan: plane-level → 90 nm/px resample → robust-normalise → up to 9 non-overlapping 512×512 afmhot-RGB tiles → resize to 224×224.
- **Augmentation:** D4 (rot90 × flip) at train time only.
- **Optim:** AdamW, head LR **3e-4**, LoRA LR **5e-5** (lowered from task-spec defaults after sanity-check revealed head collapse), weight-decay 1e-4, grad-clip 1.0. Cosine-decay schedule with 1-epoch linear warmup.
- **Loss:** Class-weighted cross-entropy (sqrt-inverse-frequency, ratio capped at 3×) with label-smoothing 0.1.
- **Batch size:** 16 (MPS). Max 12 epochs, early-stop patience 3 on inner-val weighted F1.
- **Inference:** scan-level = mean of softmax over that scan's tiles.

### Debugging note
First-pass config (head LR 1e-3, pure inverse-frequency weighting, no label smoothing) produced immediate training collapse: loss fell from 1.6 → 0 in one epoch, model picked a single class at eval time (wF1 = 0.020). Fix was the stabilised regime above; second-pass sanity showed smooth loss descent (1.60 → 1.07 → 0.96) and inner-val F1 climbing (0.50 → 0.62 → 0.65) over 3 epochs. The full run used those hyperparams.

## CV protocol (nested, with documented simplification)

- **Outer:** 5-fold person-level `StratifiedGroupKFold` (groups = `person_id`, i.e. L/R eye collapsed). Documented simplification from 35-fold LOPO due to 60-min timebox.
- **Inner (within each outer train):** 20 % of *persons* held out for early-stop. Outer-eval is never touched during training or early-stop.
- **No inner grid search:** hyperparams are fixed across folds.

## Results — 5-fold person-level OOF

- **Weighted F1:** **0.6476** (v4 = 0.6887, Δ = -0.0411)
- **Macro F1:** 0.5144 (v4 = 0.5541)

Per-class F1:

| class | F1 (LoRA) | F1 (v4) |
|---|---|---|
| ZdraviLudia | 0.8182 | 0.9200 |
| Diabetes | 0.3784 | 0.5800 |
| PGOV_Glaukom | 0.5867 | 0.5800 |
| SklerozaMultiplex | 0.6979 | 0.6900 |
| SucheOko | 0.0909 | 0.0000 |

### Full classification report
```
                   precision    recall  f1-score   support

      ZdraviLudia     0.7500    0.9000    0.8182        70
         Diabetes     0.5833    0.2800    0.3784        25
     PGOV_Glaukom     0.5641    0.6111    0.5867        36
SklerozaMultiplex     0.6907    0.7053    0.6979        95
         SucheOko     0.1250    0.0714    0.0909        14

         accuracy                         0.6667       240
        macro avg     0.5426    0.5136    0.5144       240
     weighted avg     0.6448    0.6667    0.6476       240
```

### Confusion matrix (rows=true, cols=pred)
```
               ZdraviL  Diabete  PGOV_Gl  Skleroz  SucheOk
      ZdraviLudia       63        5        0        1        1
         Diabetes       14        7        0        4        0
     PGOV_Glaukom        0        0       22       14        0
SklerozaMultiplex        5        0       17       67        6
         SucheOko        2        0        0       11        1
```

## Paired bootstrap vs v4 champion (1000×)

- **P(LoRA > v4 in weighted F1)** = 0.097
- **Δ (LoRA − v4)** mean = -0.0410  95 % CI [-0.1031, +0.0176]

## Ensemble: geometric-mean fusion of LoRA + v4 softmax

- Weighted F1: **0.6774** (solo LoRA = 0.6476, solo v4 = 0.6887)
- Macro F1: 0.5344

Per-class F1 (ensemble):

| class | F1 |
|---|---|
| ZdraviLudia | 0.8919 |
| Diabetes | 0.5000 |
| PGOV_Glaukom | 0.5753 |
| SklerozaMultiplex | 0.7047 |
| SucheOko | 0.0000 |

## Per-fold training history (best epoch by inner-val wF1)

- **Fold 1**: best epoch 4 (iv_wF1=0.6547, train_loss=0.8209)
- **Fold 2**: best epoch 11 (iv_wF1=0.6194, train_loss=0.5332)
- **Fold 3**: best epoch 6 (iv_wF1=0.8766, train_loss=0.6316)
- **Fold 4**: best epoch 5 (iv_wF1=0.9021, train_loss=0.6840)
- **Fold 5**: best epoch 2 (iv_wF1=0.6089, train_loss=1.0943)

## Files

- OOF numpy: `cache/lora_oof.npz`
- OOF JSON: `cache/lora_predictions.json`
- Last fold checkpoint: `models/lora_dinov2_finetune/`

## Limitations & honesty notes

- **5-fold instead of full 35-fold LOPO** — documented simplification; 5-fold over 35 persons still keeps evaluation person-disjoint and gives 7 persons per fold → fewer point estimates but no leakage.
- **Single run, no inner grid search** — hyperparams are from the task spec, not tuned. A proper nested LOPO with grid search is left for a follow-up if this direction proves promising.
- **Last-fold checkpoint only** — the full OOF predictions come from 5 different models; no single 'production' model is saved. For submission, a final refit on the whole TRAIN_SET would be needed.