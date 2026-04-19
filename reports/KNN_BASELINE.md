# KNN Baseline — Pure Retrieval on DINOv2 Embeddings

**Question:** how much of the few-shot VLM's ~80 % accuracy comes from *retrieval alone* (picking neighbors in DINOv2 space) vs the VLM's visual reasoning on top of those neighbors?

**Method:** cosine k-NN on cached DINOv2 ViT-B/14 scan-level embeddings (TTA-D4 afmhot, L2-normalized). Person-LOPO: neighbors from the query's person are excluded.

**N scans:** 240 across 35 unique persons, 5 classes.
**Embedding source:** `cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`

## Grid (weighted F1 / macro F1)

|  k  | majority | sim-weighted | softmax |
|----:|---------:|-------------:|--------:|
| 1 | **0.6117 / 0.4864** | 0.6117 / 0.4864 | 0.6117 / 0.4864 |
| 3 | 0.5809 / 0.4447 | 0.5809 / 0.4447 | 0.6049 / 0.4865 |
| 5 | 0.5859 / 0.4396 | 0.5859 / 0.4396 | 0.5625 / 0.4476 |
| 7 | 0.5698 / 0.4215 | 0.5698 / 0.4215 | 0.5721 / 0.4635 |
| 10 | 0.5717 / 0.4196 | 0.5717 / 0.4196 | 0.5796 / 0.4659 |

**Winner:** `k1_majority` — weighted F1 **0.6117**, macro F1 **0.4864**

### Per-class F1 (winner)

| Class | F1 |
|---|---:|
| ZdraviLudia | 0.8188 |
| Diabetes | 0.4898 |
| PGOV_Glaukom | 0.5000 |
| SklerozaMultiplex | 0.6237 |
| SucheOko | 0.0000 |

## vs Champion v4 (full 240 person-LOPO)

- v4 multiscale: weighted F1 = **0.6887**, macro F1 = 0.5541
- k-NN best:    weighted F1 = **0.6117**
- Δ (knn − v4) mean = **-0.0767**, 95 % CI = [-0.1330, -0.0185]
- **P(k-NN beats v4) = 0.003** (bootstrap 1000×)

## vs Few-shot VLM (apples-to-apples on same subset)

- subset size: 53

| System | accuracy | weighted F1 | macro F1 |
|---|---:|---:|---:|
| VLM few-shot (2 anchors/class, collage) | 0.7547 | 0.7528 | 0.7562 |
| k-NN best (k1_majority) | 0.5472 | 0.5367 | 0.4830 |

- Δ weighted F1 (k-NN − VLM) mean = **-0.2148**, 95 % CI = [-0.3705, -0.0580]
- **P(k-NN beats VLM on the same 53 scans) = 0.002**

## Conclusion

On the same 53-scan subset, the VLM (acc 75.5%) clearly beats pure k-NN retrieval (acc 54.7%) by **20.8%**, P(VLM > k-NN) = 0.998. **The VLM's reasoning adds real signal beyond DINOv2 retrieval** — it is not merely rubber-stamping nearest neighbors. The few-shot VLM pipeline is therefore defensible on this subset, though the ~4× cost of v4 multiscale (which is +6.6% vs VLM on different/full data) still needs honest cost-benefit accounting at full-dataset scale.

### Caveat on the full-dataset view

Against v4 multiscale on the full 240 person-LOPO (not the 53-scan subset), k-NN trails by -0.0770 weighted F1 (P(k-NN > v4) = 0.003). So the ensemble stack is doing meaningful work beyond raw retrieval, especially on the long-tail classes (SucheOko F1 = 0 under k-NN — single nearest neighbor never sits inside the query's 2-person-only class after person-exclusion).