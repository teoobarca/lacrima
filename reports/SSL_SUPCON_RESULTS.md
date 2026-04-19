# SSL-style SupCon Projection-Head Results

## Question

Can a small (768 -> 256 -> 128) supervised-contrastive projection head specialize
ImageNet-pretrained DINOv2-B features to the tear-AFM domain and beat:
(a) the frozen-DINOv2-B single-model baseline (~0.615 weighted F1), and
(b) the current shipped TTA D4 ensemble champion (0.6458)?

## Methodology

- Source features: `cache/tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz` (DINOv2-B-tiled, 811 tiles / 240 scans, 768-dim).
- Projection head: 2-layer MLP (Linear 768->256 -> GELU -> Linear 256->128), L2-normalized output.
- Loss: Supervised Contrastive (Khosla et al. 2020), temperature = 0.1, batch 128, AdamW lr = 3e-4, weight_decay = 1e-4, 40 epochs (per fold) / 60 (final).
- Tile-level training: each tile inherits its scan's class label; SupCon pulls same-class tiles together, pushes different-class tiles apart.
- **Honest person-LOPO protocol**: for each of 35 persons, a fresh projection head is trained from scratch on the 34 remaining persons' tiles, then:
    1. project all tiles via this fold's head;
    2. mean-pool tiles per scan -> 240 x 128;
    3. StandardScaler + LogisticRegression fit on train scans, predict held-out person's scans.
- Ensemble: 0.5/0.5 proba-average of the SupCon head's LR probs and the frozen-DINOv2-B LR probs, LOPO-honest.
- Device: `mps`. Total wall clock: 97.8 s.

## Results (person-LOPO, raw argmax)

| Model | Weighted F1 | Macro F1 |
|---|---:|---:|
| Frozen DINOv2-B single (this run, reference)  | 0.6150 | 0.4910 |
| **SupCon projection head (honest LOPO)**      | **0.6120** | **0.4448** |
| SupCon + Frozen proba-average (honest LOPO)   | 0.6318 | 0.4870 |
| Reference champion: DINOv2-B + BiomedCLIP + D4 TTA | 0.6458 | 0.5154 |
| Reference baseline (STATE.md): DINOv2-B single    | 0.6150 | 0.4910 |

### Per-class F1 (weighted)

| Class | Support | Frozen | SupCon | SupCon+Frozen |
|---|---:|---:|---:|---:|
| ZdraviLudia | 70 | 0.814 | 0.816 | 0.834 |
| Diabetes | 25 | 0.500 | 0.167 | 0.359 |
| PGOV_Glaukom | 36 | 0.513 | 0.548 | 0.571 |
| SklerozaMultiplex | 95 | 0.628 | 0.693 | 0.670 |
| SucheOko | 14 | 0.000 | 0.000 | 0.000 |

## Verdict

NO CLEAR GAIN. SupCon is within sampling noise of the frozen baseline.

- Delta vs frozen DINOv2-B (this run): **-0.0030** weighted F1
- Delta vs TTA D4 champion (0.6458)  : **-0.0338** weighted F1

## Honest caveats

1. **240 scans is tiny for SSL.** Even SupCon with real labels is at risk of overfitting the 34-person training pool; cross-person generalization is what LOPO actually measures.
2. **SupCon uses labels.** This is NOT self-supervised in the strict sense; it is *supervised contrastive*. A purely self-supervised variant (e.g. augmentation-consistency) would skip the labels at the cost of losing the class-pulling signal.
3. **Head sees only 768-dim cached DINOv2 features, not raw images.** It cannot repair genuinely missing information -- it can only re-shape the existing feature geometry.
4. **Tile labels are noisy.** All tiles of a scan inherit the scan's class even if some tiles are mostly background or show weak evidence.
5. **The per-fold head retraining is the honest baseline.** Training a head on all 240 scans and re-using it for evaluation would be label-leakage -- and we deliberately separate the LOPO numbers from the final cached embedding.

## Artifacts

- `scripts/supcon_projection.py` -- this script
- `cache/supcon_projected_emb.npz` -- (240, 128) projected scan embeddings from the final head trained on all 240 scans, plus (n_tiles, 128) projected tile embeddings. **Not safe for LOPO re-eval** (the head saw all persons); use only for downstream / visualization.
- `reports/SSL_SUPCON_RESULTS.md` (this file)
