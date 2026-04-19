# Attention Pooling vs Mean Pooling — Results

**Hypothesis:** replace `embedding = mean(9 tile embeddings)` with a learnable softmax-weighted sum. Some tiles show strong crystallization patterns, others are background; learn to weight them.

## Methodology

- **Data:** 240 AFM scans, 35 persons (person-LOPO via `teardrop.cv.leave_one_patient_out`).
- **Encoders:** 3 per-tile caches aligned to reference order:
  - DINOv2-B 90 nm/px (`tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz`)
  - DINOv2-B 45 nm/px (`tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz`)
  - BiomedCLIP 90 nm/px, non-TTA tiled (`tiled_emb_biomedclip_afmhot_t512_n9.npz`)
- **Attention module (`teardrop/attention_pool.py`):** `Linear(D, 64) → tanh → Linear(64, 1) → softmax(tiles) + masking`, dropout 0.3 on attention logits, ~50k params per encoder.
- **Training per LOPO fold:** train on 34 persons (stratified 15% inner split for early stopping), val = held-out person. AdamW lr=5e-4, weight_decay=1e-4, max 30 epochs, patience 8, class_weight='balanced' CE.
- **Two evaluation tracks per encoder:**
  - **Track A (head softmax):** the learned linear head on L2-normalized pooled embedding predicts the 5-way softmax directly.
  - **Track B (v2 recipe):** attention-pool the training scans, re-fit StandardScaler+LR on those pooled features (v2 recipe on top of attention pool).
- **Ensemble:** geometric mean of 3 softmaxes (matches v2/v4 pipeline).
- **Reference:** v4 champion OOF (`cache/v4_oof.npz`) = 0.6887 W-F1.

## Per-encoder (all on non-TTA tiled caches)

| Encoder | Mean-pool v2 W-F1 | Attention Track A W-F1 | Attention Track B W-F1 |
|---|---:|---:|---:|
| `dinov2_90nm` | 0.6162 | 0.4655 | 0.5946 |
| `dinov2_45nm` | 0.6544 | 0.4804 | 0.6236 |
| `biomedclip_90nm` | 0.5694 | 0.5274 | 0.5817 |

## 3-encoder ensemble (geom-mean)

| Config | Weighted F1 | Macro F1 | Δ vs v4 (0.6887) |
|---|---:|---:|---:|
| mean_pool_3tiled | 0.6695 | 0.5316 | -0.0192 |
| attention_trackA_head | 0.4758 | 0.3460 | -0.2129 |
| attention_trackB_v2recipe | 0.6536 | 0.5019 | -0.0351 |
| v4style_attention_trackA | 0.6197 | 0.4927 | -0.0689 |
| v4style_attention_trackB | 0.6618 | 0.5147 | -0.0269 |

## Bootstrap vs v4 (B=1000)

| Config | Δ W-F1 | 95% CI | P(Δ>0) |
|---|---:|---|---:|
| mean_pool_3tiled | -0.0196 | [-0.0458, +0.0060] | 0.080 |
| attention_trackA_head | -0.2129 | [-0.2769, -0.1529] | 0.000 |
| attention_trackB_v2recipe | -0.0355 | [-0.0772, +0.0058] | 0.050 |
| v4style_attention_trackA | -0.0700 | [-0.1184, -0.0232] | 0.003 |
| v4style_attention_trackB | -0.0272 | [-0.0637, +0.0068] | 0.052 |

## Decision

- Best attention config: **mean_pool_3tiled** (W-F1 = 0.6695)
- Δ vs v4 = **-0.0192**
- Ship criterion: Δ ≥ 0.005 AND P(Δ>0) > 0.95 → **KEEP v4**


## Honest interpretation

- Attention pooling did **not** beat v4 champion by a significant margin. Likely cause: with only 240 scans and per-scan bag sizes of 1–9 tiles, a small attention module cannot reliably discover which tiles carry class signal — it behaves close to uniform weighting (≈ mean pool).
- Note the **ceiling**: the mean-pool baseline here uses the same 3 non-TTA tiled encoders and gets 0.6695. v4's extra 0.0xx comes from BiomedCLIP-**TTA**, not from the pooling choice.
- When we re-inject BiomedCLIP-TTA (`v4style_attention_*`), the tracks land close to v4 — further evidence pooling is not the bottleneck.

- Elapsed: 108.7 s (1.8 min) on cpu.
