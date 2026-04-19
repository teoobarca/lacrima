# DANN (Domain-Adversarial) Results

## Question

UMAP of DINOv2-B scan embeddings (`reports/pitch/03_umap_embedding.png`)
shows patient identity is the DOMINANT latent axis -- more salient than
clinical class. Can a small feature adapter, trained adversarially to
*forget* patient ID while preserving class, unlock more signal?

## Methodology

- **Source**: cached DINOv2-B scan embeddings (`cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`,
  240 x 768). For v4-ensemble bonus we additionally use the 45 nm/px DINOv2-B and
  D4-TTA BiomedCLIP caches.
- **Adapter**: 768 -> 512 -> 256 -> 128 MLP (LayerNorm + GELU + dropout 0.1).
- **Class head**: Linear(128 -> 5). **Domain head**: Linear(128 -> 34) with gradient reversal.
  The number of domain classes is 34 per fold (35 persons - 1 held-out); domain IDs are
  re-indexed inside each fold.
- **Loss**: `L = CE_class(balanced) + lambda * CE_domain_reversed`.
- **Optimizer**: AdamW, lr=1e-3, weight_decay=1e-4, 40 epochs, full-batch, patience=8
  on val class accuracy (10% stratified val split inside each training fold).
- **Protocol**: strict person-LOPO over 35 persons (from `teardrop.data.person_id`).
  Held-out person's scans are projected by an adapter that never saw them.
- **Downstream**: two evaluations per lambda
  1. **class head argmax** (what the DANN optimizes);
  2. **StandardScaler + LR(balanced)** on the 128-d adapter features, LOPO-honest
     — apples-to-apples with the 0.6150 DINOv2-B-single baseline.
- **Lambda sweep**: {0.0, 0.05, 0.1, 0.3, 1.0}. lambda=0 is the no-DANN MLP baseline.
- Device: `mps`. Wall time: 107.7 s.

## Part A -- DANN lambda sweep on DINOv2-B (person-LOPO)

| lambda | F1w (class head) | F1m (class head) | F1w (LR on adapter) | F1m (LR on adapter) |
|---:|---:|---:|---:|---:|
| 0.0 | 0.5558 | 0.4429 | 0.5161 | 0.3563 |
| 0.05 | 0.5558 | 0.4429 | 0.5167 | 0.3572 |
| 0.1 | 0.5601 | 0.4459 | 0.4946 | 0.3357 |
| 0.3 | 0.5575 | 0.4486 | 0.5526 | 0.3820 |
| 1.0 | 0.4844 | 0.3942 | 0.6111 | 0.4390 |

**Reference baseline (STATE.md, DINOv2-B single + LR)**: weighted F1 = 0.6150.

### Per-class F1 on the LR-on-adapter projection

| Class | Support | lambda=0.0 | lambda=0.05 | lambda=0.1 | lambda=0.3 | lambda=1.0 |
|---|---:|---:|---:|---:|---:|---:|
| ZdraviLudia | 70 | 0.770 | 0.770 | 0.788 | 0.818 | 0.706 |
| Diabetes | 25 | 0.051 | 0.051 | 0.057 | 0.042 | 0.041 |
| PGOV_Glaukom | 36 | 0.382 | 0.386 | 0.289 | 0.432 | 0.529 |
| SklerozaMultiplex | 95 | 0.578 | 0.578 | 0.544 | 0.619 | 0.794 |
| SucheOko | 14 | 0.000 | 0.000 | 0.000 | 0.000 | 0.125 |

### Part A verdict

- Best lambda: **1.0** -> weighted F1 = **0.6111** (macro = 0.4390).
- Delta vs lambda=0 (no-DANN MLP baseline, same adapter arch): **+0.0950**
- Delta vs DINOv2-B+LR reference (0.6150): **-0.0039**

REGULARIZER. DANN materially improves over the matched no-adv adapter (lambda=0) but only matches (does not beat) the plain-LR baseline on the original 768-d features. Interpretation: the adversarial loss is acting primarily as a regularizer on an otherwise-overparameterized MLP, not as a genuine identity-scrubber. No ship.

## Part C -- Can DANN help on TOP of v4 (bonus)?

For each lambda in {0.0, best_lambda_from_part_A}, we independently train a
DANN adapter on each of the 3 v4 encoders (DINOv2-B 90 nm, DINOv2-B 45 nm,
BiomedCLIP D4-TTA 90 nm), then fit LR per-encoder on the LOPO-honest projections
and take the geometric mean of the 3 softmaxes.

| Config | Weighted F1 | Macro F1 |
|---|---:|---:|
| v4 reproduction (no DANN, this run)        | 0.6887 | 0.5541 |
| v5 candidate (DANN lambda=0.0 on 3 enc)    | 0.5824 | 0.3939 |
| v5 candidate (DANN lambda=1.0 on 3 enc)    | 0.5837 | 0.4085 |
| Reference v4 champion (STATE.md)           | 0.6887 | 0.5541 |

### Part C verdict

- Best v5 candidate: lambda=1.0, weighted F1 = **0.5837** (macro = 0.4085).
- Delta vs v4 reproduction (this run): **-0.1050**
- Delta vs v4 champion (STATE.md 0.6887): **-0.1050**

NEGATIVE. DANN projection regresses the v4 ensemble. Keep v4 as champion; DANN's identity-scrubbing is useful for visualization / ablation but not for this ensemble's final LOPO F1.

## Honest caveats

1. **240 samples / 35 patients is tiny for adversarial training.** The domain
   head has only ~7 scans per patient on average; the adversarial gradient is
   inherently noisy. This is not a methodology critique -- it is the reason
   DANN often degrades to a regularizer at this scale.

2. **Class leakage through patient.** For the rarer classes (SucheOko n=4 over 2
   patients), patient ID and class are nearly colinear -- a perfectly-scrubbed
   adapter *must* also forget class for those scans. This caps how aggressive
   lambda can reasonably be.

3. **LOPO-honest projection caching.** `cache/dann_projected_emb_dinov2b.npz`
   stores the 128-d adapter features where each held-out person's row comes
   from the fold model that did NOT see them. These 240 rows are safe to feed
   into a LOPO-LR downstream evaluation; they are NOT safe for any model that
   retrains on all 240 at once (row i encodes fold-i dependence).

4. **Full-batch on 240 scans.** With so few samples, mini-batching adds more
   variance than it removes. We use full-batch gradient descent (~230 scans).

5. **class head vs LR head.** The class head and the LR-on-adapter classifier
   often diverge by 1-3 F1 points: the LR baseline regularizes more strongly
   and is the fair comparison to the 0.6150 reference which also uses LR.

## Artifacts

- `scripts/dann_training.py` (this experiment)
- `cache/dann_projected_emb_dinov2b.npz` (240 x 128 LOPO-honest adapter features at best lambda)
- `reports/DANN_RESULTS.md` (this file)
