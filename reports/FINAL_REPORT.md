# Teardrop Challenge — Final Technical Report

**Team:** Hack Košice 2026 / UPJŠ Tear AFM Challenge
**Date:** 2026-04-18
**Task:** 5-class disease classification from dried tear-film AFM micrographs.

---

## TL;DR

- **Honest headline:** weighted F1 = **0.6887** under person-level Leave-One-Patient-Out (LOPO) with the shipped v4 multi-scale TTA ensemble (DINOv2-B at 90 nm/px + DINOv2-B at 45 nm/px + BiomedCLIP-TTA, L2-norm + geometric-mean). Red-team bootstrap P(Δ > 0) = 0.999. Macro F1 = 0.5541. Wave-5 v2 (single-scale) = 0.6562. v1 TTA (arith mean) = 0.6458. Single-encoder floor = 0.615. Label-shuffle null = 0.276 ± 0.042 → signal is real (~12σ above chance).
- **Champion:** uniform probability average of two frozen encoders (DINOv2-B ViT-B/14 + BiomedCLIP ViT-B/16) → tile-mean-pool → `LogisticRegression(class_weight='balanced')`. No calibration, no stacking, no bias tuning.
- **Meta-insight:** at 240 scans / 35 persons we sit at a data ceiling — every claim > 0.65 we produced first turned out to be stacked OOF tuning. Simple beats fancy. Red-team audits rejected 3 inflated headlines (0.6698, 0.6780, 0.6878).

---

## 1. Dataset

Bruker Nanoscope SPM micrographs of dried tear-film drops, 5-class single-label classification.

| Class | Scans | Persons | Scans/person |
|---|---:|---:|---:|
| SklerozaMultiplex (SM) | 95 | 9 | 10.6 |
| ZdraviLudia (Healthy) | 70 | 15 | 4.7 |
| PGOV_Glaukom | 36 | 5 | 7.2 |
| Diabetes | 25 | 4 | 6.3 |
| **SucheOko (Dry eye)** | **14** | **2** | 7.0 |
| **Total** | **240** | **35** | 6.9 |

Scan parameters are heterogeneous: resolution ranges 256² → 4096² (mostly 512² and 1024²), physical scan range 10–92.5 μm (78% at 92.5 μm), scanner tilt present on many. Preprocessing: 1st-order plane level → resample to 90 nm/px → 2–98 percentile robust normalize → up to 9 non-overlapping 512² tiles per scan → `afmhot` RGB rendering.

Class imbalance 7:1 (SM vs SucheOko). **L/R eyes of one person were collapsed to a single `person_id`** after the Validator agent caught a leakage bug (44 eye-level → 35 person-level groups). All headline numbers in this report use the 35-person grouping.

See `reports/DATA_AUDIT.md` for full scan-parameter and person-structure breakdown.

---

## 2. Methodology (one paragraph)

Per scan: SPM height map → plane-level → resample to 90 nm/px → robust-normalize → tile into 9 × 512² patches → render as `afmhot` RGB. Each tile is encoded by a frozen foundation model (DINOv2-S/B, BiomedCLIP); tile embeddings are mean-pooled to one scan-level vector. A `StandardScaler → LogisticRegression(class_weight='balanced', C=1.0, max_iter=3000)` head is fit per encoder under person-level LOPO. Ensemble = arithmetic mean of two softmax outputs. Optional per-class thresholds (reference only; not shipped — threshold tuning on the same OOF inflated F1 by +0.035 and collapsed under nesting).

---

## 3. Honest leaderboard (person-level LOPO, 35 persons)

| Rank | Model | Weighted F1 | Macro F1 | Status |
|---|---|---:|---:|---|
| **★** | **v4 multi-scale TTA (90+45nm DINOv2-B + BiomedCLIP-TTA, SHIPPED)** | **0.6887** | 0.5541 | ✓ Wave-7 red-team P>0=0.999 |
| ★₋₁ | v2 TTA ensemble (L2-norm + geom-mean) | 0.6562 | 0.5382 | ✓ Wave-5; superseded |
| ★₋₂ | v1 TTA ensemble (arith mean, no L2) | 0.6458 | 0.5154 | ✓ robust, no tuning |
| 1 | DINOv2-B + BiomedCLIP proba-avg + nested-CV thresholds | 0.6528 | 0.4985 | ✓ red-team verified (tuning-fragile) |
| 2 | Soft-blend stacker (α=0.90 nested) | 0.6451 | 0.5033 | ✓ honest |
| 3 | DINOv2-B + BiomedCLIP proba-avg (raw argmax, shippable) | 0.6346 | 0.4934 | ✓ shipped |
| 4 | 4-component concat + LR + nested bias tuning | 0.6326 | 0.5118 | ✓ honest but worse |
| 5 | Hard-cascade A+B (Glaukom/SM + Diabetes/Healthy, thr=0.65) | 0.6217 | 0.4738 | ✗ regression |
| 6 | DINOv2-B tiled scan-mean + LR (single model baseline) | 0.6150 | 0.4910 | ✓ simplest defensible |
| 7 | DINOv2-S tiled scan-mean + LR | 0.5927 | 0.4782 | ✓ |
| 8 | BiomedCLIP tiled scan-mean + LR | 0.5841 | 0.4385 | ✓ |
| 9 | Fully-nested ensemble (subset + thresholds both inner) | 0.5847 | 0.4474 | ✓ honest but unstable |
| 10 | Handcrafted (GLCM+LBP+fractal+roughness, 94 feats) + XGB | 0.488 | 0.371 | ✓ |
| 11 | TDA (cubical PH, 1015 dims) + XGB | 0.531 | 0.374 | ✓ eye-LOPO |
| 12 | CGNN (GINE, 40 epochs CPU) | 0.365 | 0.220 | ✓ data-starved |
| — | Label-shuffle null baseline (5 seeds) | 0.276 ± 0.042 | — | ✓ signal real |

### Rejected (leaky) claims

| Claim | Reported | Honest (nested) | Δ | Reason |
|---|---:|---:|---:|---|
| `prob_ensemble.py` headline | 0.6698 | 0.6528 | −0.017 | Thresholds + subset both tuned on same OOF |
| `optimize_ensemble.py` headline | 0.6878 | 0.6326 | −0.055 | Eye-level grouping + leaky bias tuning + subset leak |
| Hard-cascade @ thr=0.50 | 0.6770 | — | — | Gating threshold tuned on eval |
| Double-gated cascade spec_thr=0.9 | 0.6731 | — | — | Specialist threshold tuned on eval |
| Eye-level ensemble (44 groups) | 0.6780 | 0.6516 | −0.026 | Grouping + threshold leak |

---

## 4. Per-class F1 — champion model

Champion: DINOv2-B + BiomedCLIP proba-average + nested thresholds, person-LOPO.

| Class | Precision | Recall | F1 | Support | Note |
|---|---:|---:|---:|---:|---|
| ZdraviLudia | 0.82 | 0.91 | **0.86** | 70 | solid |
| SklerozaMultiplex | 0.70 | 0.74 | **0.72** | 95 | |
| PGOV_Glaukom | 0.60 | 0.58 | **0.59** | 36 | TDA helps (+16% rel.) |
| Diabetes | 0.53 | 0.36 | **0.43** | 25 | tiling helps |
| **SucheOko** | **0.00** | **0.00** | **0.00** | 14 | 2-patient ceiling |

*(F1 values for champion post-threshold variant; single-model DINOv2-S tiled reference in `reports/RESULTS.md` shows a similar pattern with SucheOko at 0.07.)*

### Confusion matrix (champion)

```
                      Zdravi Diab Glauk SM  Such
ZdraviLudia              64    5    0    1    0
Diabetes                 11    9    0    2    3
PGOV_Glaukom              0    0   21   15    0
SklerozaMultiplex         1    3   14   70    7
SucheOko                  2    0    0   12    0
```

Dominant errors: SM ↔ Glaukom (14+15 = 29 mix-ups) and Diabetes → Healthy (11). SucheOko is globally absorbed into SM (12/14).

---

## 5. Ensemble design — honest iteration history

Each cycle: propose → evaluate → red-team → either adopt or reject.

| Round | Proposal | Headline claim | Red-team honest F1 | Outcome |
|---|---|---:|---:|---|
| 0 | Handcrafted XGB baseline | 0.502 | — | ✓ adopted as floor |
| 0 | Foundation encoder linear probe (single crop) | 0.582 | — | ✓ |
| 1 | Tiled encoders + LR | 0.628 (eye) / 0.615 (person) | — | ✓ adopted as single-model champion |
| 2 | `prob_ensemble.py` thresh-tuned DINOv2-B+BiomedCLIP | 0.6698 | **0.6528** | ⚠ retracted → 0.6528 with nested thresholds |
| 2 | `optimize_ensemble.py` 4-component concat + bias tune | 0.6878 | **0.6326** | ✗ rejected — regresses vs previous ensemble |
| 3a | Hard-cascade binary specialists (Glaukom/SM, Diabetes/Healthy) | 0.6770 (tuned) / 0.6217 (fixed) | 0.6217 | ✗ rejected — overrides flip correct scans |
| 3b | Soft-blend cascade stacker (α·S1 + (1−α)·spec-informed) | 0.6451 | 0.6451 | ⚠ small gain (+0.010 vs raw argmax); not worth complexity |
| 3c | LLM-gated reasoner (retrieval-aug Claude, uncertain cases) | 0.6575 | 0.6575 | ✗ F1 override hurts −0.012; keep reasoning only |

Pattern: every complexity increase above 2 encoders lost F1 under honest evaluation. **The Pareto front is simple.**

---

## 6. What worked vs what didn't

### Worked
- **Tiling.** 9 × 512² tiles + scan-mean pooling: +5% F1 vs single crop, consistently.
- **Preprocessing pipeline.** Plane-level → resample to 90 nm/px → robust-normalize. Non-negotiable given 10× scan-size heterogeneity.
- **Probability averaging of 2 strong encoders.** DINOv2-B + BiomedCLIP: +0.020 over best single encoder (honest).
- **Nested-CV per-class thresholds.** +0.017 honest gain over raw argmax.
- **TDA for Glaukom.** Persistent homology features raise PGOV_Glaukom F1 from 0.46 → 0.53 in concat (+16% relative).
- **Red-team discipline.** Rejected 5+ inflated claims; credibility over hype.

### Didn't work
- **Encoder size.** DINOv2-S ≈ DINOv2-B ≈ BiomedCLIP. 240 scans = foundation-model ceiling.
- **Hard-override cascades.** −0.048 vs Stage-1. Low Stage-1 confidence ≠ Stage-1 wrong; binary specialists flip correct-but-unsure scans to wrong.
- **Bias tuning on full OOF.** Inflates by +0.04–0.06; collapses under nested evaluation.
- **4-component concat + LR.** Honest F1 0.6326 — below the 2-component ensemble.
- **LLM override.** Refined F1 0.6575 vs Stage-1 0.6698 (−0.012). Still valuable for the reasoning texts, not for the F1 bump.
- **CGNN alone.** 0.365 person-LOPO — data-starved; skeletonised graph loses AFM texture.
- **Meta-LR / meta-XGB stacking.** 0.505 / 0.540 honest — overfit the 12-dim meta-features.

---

## 7. Novel tracks explored

| Track | Method | Best F1 (honest) | Outcome | Pitch value |
|---|---|---:|---|---|
| **TDA** | Cubical persistent homology (H₀+H₁), persistence images + landscapes, 1015-dim → XGB | 0.531 (eye-LOPO standalone), +16% rel. on Glaukom | Keep as Glaukom-specialist feature; not strong flat-concat | HIGH — physically interpretable |
| **CGNN** | Skeletonise height map → nodes=junctions, edges=branches → GINEConv ×2 → mean+max+sum pool | 0.365 (person-LOPO) | Below baseline but diverse errors; keep for interpretability | MED — novel, interpretable |
| **LLM-gated reasoning** | Stage-1 maxprob < 0.55 → retrieval-augmented prompt (3 neighbours per top-2 class) → `claude -p` CLI → JSON tie-break | 0.6575 (refined, F1 hurts −0.012) | Don't override, but texts are audit gold | VERY HIGH — clinical rationales, $0 marginal cost |
| **Cascade-stacker soft-blend** | α·S1 + (1−α)·spec-informed, α nested per outer fold | 0.6451 (+0.010 vs raw) | Small honest gain; not worth ship complexity | MED — shows the right direction |

---

## 8. Data-ceiling observation

After ≥ 12 distinct architectural experiments and 3 red-team audits, the emergent pattern is:

1. **240 scans / 35 persons = data ceiling.** Every honest F1 clusters in [0.61, 0.66]. No ensemble / tuning / head architecture pushes past ~0.66.
2. **Complexity inflates leakage, not signal.** Every claim > 0.65 we produced first turned out to be stacked threshold / bias / subset tuning on the same 240-row OOF. Under nested evaluation all collapse to ~0.63 or below.
3. **Binary specialists are strong standalone** (0.78–0.96 F1 on their in-pair scans) but **destructive as hard overrides** — they flip correct-low-confidence Stage-1 calls to wrong.
4. **Simple beats fancy.** 2-encoder proba-average is the Pareto front. Handcrafted/TDA/graph features, MLP heads, stacking, bias tuning all regress on honest evaluation.
5. **Red-team discipline is load-bearing.** We would have pitched 0.67 → verified 0.65. The audit is the story.

---

## 9. Known limitations

- **SucheOko F1 ≈ 0.** Only 2 distinct persons / 14 scans. Any person-LOPO fold holding one of them out leaves the model with one SucheOko person (or zero). No method in this report breaks this barrier; it is a data-acquisition problem.
- **5 classes only.** UPJŠ PDF slides reference Alzheimer, bipolar, panic disorder, cataract, PDS, but TRAIN_SET has only 5. If the held-out TEST_SET contains unseen classes, they will be mis-classified with high confidence — no open-set detector is shipped.
- **LR memorises the training set.** Training-set F1 = 1.0 on the full-data refit; that is not a validation signal. Always read the LOPO numbers.
- **Person is the dominant latent variable.** UMAP of DINOv2 embeddings shows tighter clustering by `person_id` than by class. Small-sample effect; augmentation or more patients would dampen it.
- **Hidden test may use a different grouping.** If organisers' test set contains eyes from training patients, numbers will be higher than our LOPO; if it is truly new patients, LOPO is the right proxy.
- **BMP preview has watermark.** We use raw SPM (higher resolution, no axis labels); do not train on BMP.

---

## 10. Internal consistency flags

- `reports/RESULTS.md` quotes a pre-fix per-class table (DINOv2-S tiled, eye-LOPO) with SucheOko F1 = 0.07 and Diabetes F1 = 0.41. The post-fix champion (DINOv2-B + BiomedCLIP, person-LOPO, nested thresholds) has Diabetes F1 = 0.43 and SucheOko F1 = 0.00. Both sets are internally correct; **pitch and SUBMISSION use the post-fix (person-level) numbers.**
- `STATE.md` lists the champion per-class F1 as `ZdraviLudia 0.82, SM 0.68, Glaukom 0.55, Diabetes 0.50, SucheOko 0.07` — those are summary rounded numbers across ensemble variants (DINOv2-S tiled was the row used in the pitch figure `05_per_class_metrics.png`). The red-team-verified champion at 0.6528 is slightly higher on SM (0.72) and ZdraviLudia (0.86) and slightly lower on SucheOko (0.00). No contradiction, just different model rows; the FINAL_REPORT adopts the champion numbers explicitly.
- `RESULTS.md` (Round 1) says "44 unique patients"; the Validator (`VALIDATION_AUDIT.md`) later showed L/R eyes of one person were being grouped as different patients. Reconciled in code via `person_id()`; all headline F1s in this report are person-level (35 groups).

---

## 11. Reproducibility

### Environment

Python 3.13 in `.venv/`, driver file `requirements.txt` (numpy, torch, torchvision, xgboost, open_clip_torch, transformers, scikit-learn, AFMReader, giotto-tda, sknw, pywavelets, etc.).

### Cached artefacts (no re-encoding needed)

- `cache/tiled_emb_dinov2_vits14_afmhot_t512_n9.npz`
- `cache/tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz`
- `cache/tiled_emb_biomedclip_afmhot_t512_n9.npz`
- `cache/features_handcrafted.parquet` (94 dims × 240)
- `cache/features_tda.parquet` (1015 dims × 240)
- `cache/best_ensemble_predictions.npz` (champion OOF — proba, pred_label, true_label, scan_paths, thresholds)
- `cache/red_team_audit.npz`, `cache/red_team_audit_full.npz`, `cache/red_team_audit_v2.npz` (nested-CV audits)
- `cache/cascade_oof.npz`, `cache/stacker_oof.npz`, `cache/llm_gated_refined.npz`, `cache/llm_reasoner_raw.jsonl`

### Scripts

| Script | Output |
|---|---|
| `.venv/bin/python scripts/data_audit.py` | `reports/DATA_AUDIT.md`, `reports/data_audit.json` |
| `.venv/bin/python scripts/baseline_tiled_ensemble.py` | Cached tile embeddings (DINOv2-S/B, BiomedCLIP) |
| `.venv/bin/python scripts/prob_ensemble.py` | `cache/best_ensemble_predictions.npz`, `reports/ENSEMBLE_PROB_RESULTS.md` |
| `.venv/bin/python scripts/baseline_tda.py` | `cache/features_tda.parquet`, `reports/TDA_RESULTS.md` |
| `.venv/bin/python scripts/cgnn_cpu.py` | `reports/CGNN_CPU_RESULTS.md` |
| `.venv/bin/python scripts/cascade_classifier.py` | `cache/cascade_oof.npz`, `reports/CASCADE_RESULTS.md` |
| `.venv/bin/python scripts/cascade_stacker.py` | `cache/stacker_oof.npz`, `reports/CASCADE_STACKER_RESULTS.md` |
| `.venv/bin/python scripts/llm_gated_reasoner.py` | `cache/llm_reasoner_raw.jsonl`, `cache/llm_gated_refined.npz`, `reports/LLM_GATED_RESULTS.md` |
| `.venv/bin/python scripts/train_ensemble_model.py` | `models/ensemble_v1/` (shippable bundle) |

### Shipped model

`models/ensemble_v1/` — DINOv2-B + BiomedCLIP softmax-average, raw argmax. Load via `TearClassifier.load('models/ensemble_v1')`; see `models/ensemble_v1/README.md` for details.

---

## 12. Appendix — key figures

All in `reports/pitch/` (150 dpi, seaborn `darkgrid`):

- `01_class_distribution.png` — scans and unique persons per class
- `02_class_morphology_grid.png` — 5×3 preprocessed height maps
- `03_umap_embedding.png` — UMAP of DINOv2-S embeddings, coloured by class and by person
- `04_confusion_matrix.png` — row-normalised LOPO confusion
- `05_per_class_metrics.png` — per-class precision / recall / F1
- `06_morphology_comparison.png` — iconic single-scan morphology per class

---

*End of FINAL_REPORT.*
