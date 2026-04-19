# Pitch Slides — 10-slide outline

Ready for Google Slides / PowerPoint. Bullet-point format, no prose. Figure references map to `reports/pitch/`.

---

## Slide 1 — Title

- **Teardrop Challenge: detecting disease from a single tear drop**
- Hack Košice 2026 / UPJŠ
- Team name, members, date
- (Background image: a single dendritic AFM scan, afmhot colormap)

---

## Slide 2 — The problem

- A dried tear drop on mica → dendritic crystallisation
- Each disease alters protein / electrolyte composition → distinct crystal signature
- Task: 5-class single-label classification
  - ZdraviLudia · SklerozaMultiplex · PGOV_Glaukom · Diabetes · SucheOko
- *Figure:* `02_class_morphology_grid.png` (5×3 grid of AFM height maps)

---

## Slide 3 — The data

- 240 Bruker Nanoscope SPM scans, 35 unique persons
- Class imbalance 7:1 (SM 95 scans vs SucheOko 14)
- Scan parameters heterogeneous: 10–92.5 μm range, 256² → 4096² pixels
- **SucheOko has only 2 persons** — a structural ceiling
- *Figure:* `01_class_distribution.png`

---

## Slide 4 — The catch (person identity leakage)

- UMAP of DINOv2 embeddings: scans cluster by **person** more than by **class**
- Our initial parser treated L/R eye of one human as two patients
- Validator agent caught the bug → fixed via `person_id`
- All honest numbers: person-level LOPO (35 groups), not eye-level (44)
- *Figure:* `03_umap_embedding.png` (left: by class | right: by person)

---

## Slide 5 — The approach: orchestrator of agents

- Claude orchestrator + specialist agents + red-team auditor
- Rounds of parallel experiments: tiling, TDA, CGNN, ensemble, LLM, cascade
- **Every claim audited before adoption** → 5 inflated headlines rejected
- Result: 0.6528 honest (not 0.6878 leaky)

---

## Slide 6 — The architecture: 3-tier pipeline

- **Stage 1:** DINOv2-B + BiomedCLIP proba-average → argmax (~80 % of cases)
- **Stage 2:** binary specialists (Glaukom↔SM, Diab↔Healthy) as SOFT-BLEND features
- **Stage 3:** Claude CLI with retrieval-augmented prompts for uncertain cases
  - Returns JSON + clinical rationale
  - $0 marginal cost (CLI subscription)
- Simple, tile + mean-pool + LR — nothing fancier survived honest evaluation

---

## Slide 7 — Headline results

- **Weighted F1 = 0.6562** (person-LOPO, v2 TTA ensemble, L2-norm + geom-mean, no tuning)
- v1 TTA (arith mean, no L2): 0.6458
- Threshold-tuned reference: 0.6528 (nested-CV, tuning-fragile)
- Non-TTA ensemble: 0.6346
- Single-model baseline: 0.615
- Null baseline: 0.276 ± 0.042 → **~9σ above chance**
- v2 recipe gains concentrated on minorities: Diabetes 0.43→0.54, Dry Eye 0→0.06
- Per-class: Healthy 0.87 · Diabetes 0.54 · Glaukom 0.56 · SM 0.65 · Dry Eye 0.06
- *Figure:* `04_confusion_matrix.png` + `05_per_class_metrics.png`

---

## Slide 8 — What we ruled out (honest negatives)

| Approach | Honest F1 | Status |
|---|---:|---|
| Hard-override cascade | 0.6217 | −0.048 |
| LLM prediction override | 0.6575 | −0.012 |
| 4-component concat + bias tune | 0.6326 | −0.020 |
| CGNN alone | 0.365 | −0.29 |
| Meta-LR / meta-XGB stacker | 0.51 / 0.54 | −0.14 / −0.11 |

- Meta-insight: **240 scans = data ceiling.** Complexity inflates leakage, not signal.

---

## Slide 9 — Interpretability bonuses

- **TDA** (persistent homology H₀+H₁): physically sensible Glaukom signature
  - Sparse dendrites, few but long-lived loops
  - +16 % relative F1 on PGOV_Glaukom
- **LLM reasoning texts:** every uncertain case ships with a clinical rationale
  - Cites nearest-neighbour references and specific GLCM / roughness features
  - Sample: *"Query's glcm_contrast_d5_mean (19.43) nearly matches Diabetes (19.61) and is far lower than Healthy (37.8) ..."*
- *Figure:* `06_morphology_comparison.png`

---

## Slide 10 — Future work & closing

- Future
  - More patients, especially SucheOko (2 persons → ceiling)
  - 9-class version + open-set detector (UPJŠ PDF references 9)
  - Patient-level meta-features (age, sex, scan day)
  - Synthetic data for rare classes (diffusion-conditioned AFM)
- Ship
  - `TearClassifier.load('models/ensemble_v1')` → `.predict_directory(...)`
  - Full red-team history in `reports/`
- **Credibility > hype. 0.6562 honest, shipped. 0.6528 threshold-tuned reference.**
- **Thank you.**

---

## Backup slides (for Q&A)

- **B1.** Full honest leaderboard (13 rows) — from `FINAL_REPORT.md` section 3
- **B2.** Rejected leaky claims table (5 rows + reasons) — from `FINAL_REPORT.md` section 3
- **B3.** Per-class persistence signatures (TDA) — from `reports/TDA_RESULTS.md`
- **B4.** CV variance: Repeated StratifiedGroupKFold mean F1 = 0.6375 ± 0.0795 — from `VALIDATION_AUDIT.md`
- **B5.** LLM cost analysis: 47 calls × ~9 s, $0.106 if billed via API — from `LLM_GATED_RESULTS.md`
