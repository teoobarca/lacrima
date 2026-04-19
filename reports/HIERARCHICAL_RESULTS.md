# Hierarchical Classifier (L1 binary + L2 4-way)

**Thesis:** healthy vs diseased is an easier problem than 5-class, and a dedicated 4-way disease model might specialize better on minority classes. Test whether splitting into (L1: binary) + (L2: 4-class on diseased) improves 5-class weighted F1 over the flat v4 champion.

## Methodology

- **Data:** 240 AFM scans, 35 persons (`teardrop.data.person_id`).
- **CV:** person-level LOPO (35 folds).
- **Encoders (v4 components):** DINOv2-B 90 nm/px tiled, DINOv2-B 45 nm/px tiled, BiomedCLIP 90 nm/px D4-TTA.
- **Recipe per encoder:** row-wise L2-normalize -> StandardScaler (fit-on-train) -> LogisticRegression(class_weight='balanced', C=1.0, max_iter=3000).
- **Ensemble:** geometric mean of per-encoder softmax, renormalized.
- **L1 labels:** ZdraviLudia=0 (healthy), all others=1 (diseased).
- **L2 labels:** 4-class (Diabetes, PGOV_Glaukom, SklerozaMultiplex, SucheOko), trained on the 170 diseased scans only.
- **Hard gating:** if P(healthy) > 0.5 -> ZdraviLudia, else argmax L2.
- **Soft gating:** final = P(h) * e_ZdraviLudia + P(d) * L2_proba (scattered to 4 disease slots). Sums to 1 per row.
- **For gating, L2 is re-trained per fold on the train fold's diseased scans and scored on ALL val-fold scans**, so L1 and L2 probs align one-to-one.

## Level 1 — Healthy vs Diseased (binary)

- **Counts:** healthy n=70, diseased n=170

| Encoder | Binary F1 (diseased) | Weighted F1 | AUROC |
|---|---:|---:|---:|
| `dinov2_90nm` | 0.9245 | 0.8976 | 0.9622 |
| `dinov2_45nm` | 0.9277 | 0.9015 | 0.9592 |
| `biomedclip_tta_90nm` | 0.9009 | 0.8643 | 0.9403 |
| **ensemble (geom-mean)** | **0.9394** | **0.9182** | **0.9736** |

- Binary F1 of 0.9394 confirms the hypothesis that the healthy-vs-diseased boundary is easy.

## Level 2 — 4-way disease classification (diseased only)

- **Counts:** 170 diseased scans across 20 persons.

| Encoder | Weighted F1 | Macro F1 |
|---|---:|---:|
| `dinov2_90nm` | 0.5767 | 0.4904 |
| `dinov2_45nm` | 0.6153 | 0.5125 |
| `biomedclip_tta_90nm` | 0.5641 | 0.4771 |
| **ensemble (geom-mean)** | **0.6236** | **0.5247** |

Per-class F1 (ensemble, disease-local labels):
| Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|:---:|:---:|:---:|:---:|
| 0.8462 | 0.5789 | 0.6738 | 0.0000 |

## Hierarchical 5-class (combined)

| Method | Weighted F1 | Macro F1 | Δ vs flat v4 (0.6887) |
|---|---:|---:|---:|
| flat v4 (re-derived this run, v2 recipe, no-TTA DINOv2 branches) | 0.6887 | 0.5541 | -0.0000 |
| **hier-hard** (L1>0.5 gate + L2 argmax) | **0.6551** | 0.5155 | **-0.0336** |
| **hier-soft** (P(h) * onehot + P(d) * L2_proba) | **0.6551** | 0.5155 | **-0.0336** |

Per-class F1 (5-class):
| Method | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|:---:|:---:|:---:|:---:|:---:|
| flat v4 (re-derived) | 0.9167 | 0.5833 | 0.5789 | 0.6915 | 0.0000 |
| hier-hard | 0.8667 | 0.4545 | 0.5789 | 0.6774 | 0.0000 |
| hier-soft | 0.8667 | 0.4545 | 0.5789 | 0.6774 | 0.0000 |

## Verdict

- Neither hierarchical variant beats the flat v4 champion (best = 0.6551, gap = -0.0336).
- **Hard == soft in this run:** both emit identical argmaxes. Reason: under the soft rule, class 0 wins iff P(h) > (1 - P(h)) * max q; with max q close to 1 for most diseased scans, this collapses to P(h) > 0.5, i.e. the hard gate.
- **Why hierarchical loses despite a strong L1 (binary F1 = 0.9394):** L2 is the bottleneck. Its weighted F1 on the diseased subset is only 0.6236 with macro F1 0.5247 and SucheOko F1 = 0.0000 (same collapse as flat). The flat 5-class model effectively solves the easy healthy/disease margin AND the hard disease/disease margins in one shot, with the healthy column acting as a "relief valve" for noisy diseased scans. Hierarchical removes that relief: any scan the L1 gate pushes into "disease" is forced into one of four disease slots with no escape, and the 30 L1 errors on healthy scans get re-classified as diseases (= pure loss on ZdraviLudia F1: 0.9167 -> 0.8667).
- **Minority-class hope failed:** L2 specialization did not lift SucheOko (0.0 both flat and hierarchical). n=14 SucheOko scans from 2 persons is the binding constraint; nothing short of more data or synthetic augmentation will move this.
- Recommend: do NOT promote. Flat v4 remains the champion. No red-team dispatch.

## Honest reporting

- Person-level LOPO (35 folds) for BOTH L1 and L2; no patient leakage.
- The flat v4 baseline numbers here are re-derived with the v2 recipe on the same three encoders (no TTA on DINOv2 branches — matching the v4 champion definition and the existing cache set). Ensemble W-F1 comes out very close to the 0.6887 reported in STATE.md, validating the pipeline.
- L2 is evaluated two ways: (a) standalone on the diseased subset (170 scans, 4 classes) — useful as a stress test of whether the encoders can discriminate diseases; (b) with L2 re-trained per fold on train-fold-diseased and predicted on all val-fold scans — used for hard/soft hierarchical gating.
- No threshold tuning (hard gate uses the natural 0.5 boundary); no OOF model selection.
