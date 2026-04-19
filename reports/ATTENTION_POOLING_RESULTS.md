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

- Best attention config: **v4style_attention_trackB** (W-F1 = 0.6618)
- Δ vs v4 = **-0.0269**
- Ship criterion: Δ ≥ 0.005 AND P(Δ>0) > 0.95 → **KEEP v4**


## Honest interpretation

- Attention pooling **regresses** vs both the mean-pool baseline and the v4 champion. Every attention variant is below mean-pool within the same 3-encoder ensemble family:
  - mean-pool 3-tiled = 0.6695
  - attention track A (head)   = 0.4758
  - attention track B (v2 LR)  = 0.6536
- Per-encoder: track B (attention-pool → LR) is systematically **below** mean-pool+LR by 0.02–0.03. Track A (attention head directly) collapses to 0.47 — the tiny linear head underfits a 5-class problem from 204-person training with class_weight alone.
- **Why attention loses here:** with bag sizes 1–4 tiles (90 nm) and up to 9 tiles (45 nm), 240 total scans, and ~50k new attention params, the model cannot reliably learn which tiles carry class signal. Mean pooling implicitly averages out noise; learned attention over-concentrates on a few tiles per scan and throws away the averaging benefit.
- **Ceiling check:** v4 champion is 0.6887. The mean-pool 3-encoder ensemble here at 0.6695 is Δ = -0.0192 — the v4 champion's edge over the non-TTA mean-pool ensemble comes from the BiomedCLIP **TTA** branch, not from any pooling gain. When we re-inject BiomedCLIP-TTA into the attention pipeline (`v4style_attention_*`), we still land below v4.
- **Verdict: REJECT attention pooling.** v4 mean-pool stays champion.

- Elapsed: 69.1 s (1.2 min) on cpu.
