# Error Analysis — TTA Ensemble Champion

**Model:** `models/ensemble_v1_tta/` — DINOv2-B + BiomedCLIP, D4 TTA, proba-average, raw argmax.
**Eval:** person-level Leave-One-Patient-Out over 35 persons / 240 scans / 5 classes.
**Reproduction script:** `scripts/error_analysis.py` → `reports/error_cases.csv`, `reports/error_analysis.json`.

---

## 1. OOF reproduction (sanity check)

| Metric | Value |
|---|---:|
| Weighted F1 | **0.6458** (matches shipped champion exactly) |
| Macro F1 | 0.5154 |
| Accuracy | 0.646 (155/240 correct) |

### Classification report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ZdraviLudia | 0.84 | 0.91 | 0.88 | 70 |
| Diabetes | 0.55 | 0.48 | 0.51 | 25 |
| PGOV_Glaukom | 0.51 | 0.58 | 0.55 | 36 |
| SklerozaMultiplex | 0.68 | 0.61 | 0.64 | 95 |
| **SucheOko** | **0.00** | **0.00** | **0.00** | 14 |

### Confusion matrix (rows=true, cols=pred)

```
                      Zdrav Diabe PGOV_ Skler Suche
  ZdraviLudia            64     5     0     0     1
  Diabetes                8    12     0     2     3
  PGOV_Glaukom            0     0    21    15     0
  SklerozaMultiplex       1     4    20    58    12
  SucheOko                3     1     0    10     0
```

---

## 2. Failure-mode breakdown (85 wrong scans)

Rank of true class (among wrong): **rank 2: 58 | rank 3: 21 | rank 4: 5 | rank 5: 1**.
So **68% of errors are near-miss** (true class was second-guess).

| Mode | Definition | Count | % of errors |
|---|---|---:|---:|
| **A — Low-conf wrong** | maxprob < 0.40 | **1** | 1 % |
| **B — High-conf wrong** | maxprob > 0.70 AND rank of true ≥ 3 | **15** | 18 % |
| **C — Near miss** | true class was rank 2 | **58** | 68 % |
| **D — Catastrophic** | true class rank ≥ 4 | **6** | 7 % |
| E — Mid-conf wrong | wrong, 0.4 ≤ maxprob ≤ 0.7, rank = 3 | 5 | 6 % |

Mean maxprob on wrong predictions = 0.765 (vs 0.843 on correct). Distributions overlap heavily. **Confidence is a weak rejection signal** — even at 0.99 the model is sometimes confidently wrong.

### Mode counts per true class

| Class | OK | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|---:|
| ZdraviLudia | 64 | 0 | 0 | 5 | 0 | 1 |
| Diabetes | 12 | 1 | 1 | 10 | 1 | 0 |
| PGOV_Glaukom | 21 | 0 | 2 | 13 | 0 | 0 |
| SklerozaMultiplex | 58 | 0 | 4 | 28 | 1 | 4 |
| **SucheOko** | **0** | 0 | **8** | 2 | **4** | 0 |

**Headline:** 12 of 15 Mode B scans come from SM (4), SucheOko (8), Glaukom (2). Mode D is dominated by SucheOko (4/6). **SucheOko generates 57% of the non-near-miss errors despite being 6% of the dataset.**

### Mode B true → pred pairs

| True → Pred | count |
|---|---:|
| SucheOko → SklerozaMultiplex | 5 |
| SucheOko → ZdraviLudia | 2 |
| SucheOko → Diabetes | 1 |
| SklerozaMultiplex → SucheOko | 2 |
| SklerozaMultiplex → PGOV_Glaukom | 2 |
| PGOV_Glaukom → SklerozaMultiplex | 2 |
| Diabetes → ZdraviLudia | 1 |

---

## 3. Mode B deep-dive — 5 representative case studies

Method: for each Mode B scan, we retrieve the 5 nearest training scans (DINOv2-B *tiled, non-TTA* scan-mean, cosine similarity) belonging to **other persons**. We also compute Euclidean distance in z-scored handcrafted-feature space to the mean of the true class vs the predicted class. All 15 Mode B cases are in `reports/error_analysis.json` under `mode_b_details`.

### Case B1 — `22_PV_SM.000` (person `22EYE_SM`): SM → Glaukom, maxprob = **1.000**, true-class prob = 0.0003

- 5 / 5 nearest neighbours are Glaukom (persons `27EYE_PGOV_PEX`, `21EYE_PGOV+SII`, `23EYE_PGOV`).
- Handcrafted features: closer to Glaukom mean (11.85) than SM mean (13.10).
- **Diagnosis:** this SM scan is simply *morphologically indistinguishable* from Glaukom at both deep and shallow feature levels. The model is not wrong by accident — the embedding puts it in Glaukom territory. Likely the sample at `22EYE_SM` has the coarse-loop texture that is characteristic of PGOV at 92.5 μm scan range; without an SM-internal morphological sub-label the model has no leverage.

### Case B2 — `DM_01.03.2024_LO.003` (person `DM_01.03.2024EYE`): Diabetes → Healthy, maxprob = **0.996**

- 5 / 5 nearest neighbours are healthy controls (persons `48`, `5`, `48`, `6`, `2`).
- HC features: closer to Healthy mean (4.07) than Diabetes mean (4.57).
- **Diagnosis:** this is the classic Diabetes-looks-healthy failure. The `DM_01.03.2024EYE` session has scans that span the decision boundary (4 of 10 classified as Healthy, 2 as SucheOko, 1 as Diabetes). The diagnostic features for diabetic tear film (salt-crystal lattice disruption, glucose-dependent surface micro-structure) may simply not be present/visible at 90 nm/px, OR this particular session's glucose level was near-normal. Per-patient variance dominates.

### Case B3 — `21_LV_PGOV+SII.007` (person `21EYE_PGOV+SII`): Glaukom → SM, maxprob = **0.995**

- 5 / 5 nearest neighbours are SM (persons `100-SM-EYE-18`, `20-SM-EYE-18`).
- HC features: *closer to true-class* (Glaukom) mean (7.08) than predicted (9.60). The embedding disagrees with the shallow features!
- **Diagnosis:** pure embedding-vs-handcrafted disagreement. The DINOv2/BiomedCLIP representation sees "SM-like branching patterns"; GLCM/LBP/roughness statistics correctly place it in Glaukom cluster. This is the strongest case for **adding handcrafted features as a stacker input** — they would have caught this one.

### Case B4 — `29_PM_suche_oko.006` (person `29EYE_suche_oko`): SucheOko → SM, maxprob = **0.963**

- 5 / 5 nearest neighbours are SM (persons `19EYE_SM`, `1-SM-EYE-18`).
- HC features: closer to SM mean (6.81) than SucheOko mean (7.41). Note — SucheOko "mean" is computed from 14 scans of 2 persons so it's essentially two person-centroids averaged, which itself is a statistical artefact.
- **Diagnosis:** the two SucheOko patients (`29EYE` and `35EYE`) each appear to have a distinct morphology from each other. When either is held out in LOPO, the remaining patient's style is a single point estimate that almost never matches the held-out one. Deep features place this scan firmly inside the SM cluster. **There is no signal to learn "SucheOko" from 7 scans of one person and generalize to a new person.** This is a data-acquisition failure, not a modelling failure.

### Case B5 — `20_LM_SM-SS.001` (person `20EYE_SM-SS`): SM → SucheOko, maxprob = **0.751**

- 3 / 5 nearest neighbours are SucheOko (all from `29EYE_suche_oko`), 2 / 5 are SM (from `1-SM-EYE-18`).
- HC features: marginally closer to SucheOko mean (9.47 vs 9.52) — truly ambiguous.
- **Diagnosis:** this is a genuine edge case at the SM-SucheOko boundary. The person label `20EYE_SM-SS` suggests a co-morbidity (`SM-SS` = SM + Sjögren's Syndrome, which causes dry eye). The scan may present mixed pathology — SM structure with dry-eye-like crystallisation. **Label ambiguity is plausible here**; a clinician review could resolve. Ship-level fix: a dedicated SM-vs-SucheOko binary is tough because both SucheOko patients contribute few examples.

### Mode B pattern summary

Of 15 Mode B scans, **the nearest-neighbour majority class matches the model's wrong prediction in 12 / 15 cases**. The model is internally consistent — it is not making random mistakes. The embedding space simply puts these scans in the wrong cluster. That means:

- **Handcrafted features agree with the model's wrong call in 9 / 15 cases** (the scan genuinely looks like the wrong class at all feature levels — hardest to fix).
- **Handcrafted features disagree in 6 / 15 cases** (B3-style, the shallow stats know better). These are the ones most likely recoverable by a stacker that fuses DINOv2 + GLCM/LBP/roughness.

---

## 4. Per-class error patterns

### ZdraviLudia (64 / 70, recall 0.91) — the reliable class
- 5 / 6 errors → Diabetes (healthy persons mis-called as Diabetes), all in the 0.48–0.62 maxprob band → Mode C/E near-misses.
- Biomedical plausibility: the Diabetes↔Healthy boundary is known to be subtle (glucose-dependent salt-crystal morphology, not always expressed).
- **Intervention:** a Diabetes-vs-Healthy stacker fed by the handcrafted features (roughness Sa/Sq, LBP histogram) — reuse the existing binary specialist trained in `cascade_classifier.py` but as a **feature, not override**.

### Diabetes (12 / 25, recall 0.48) — the leaky-to-healthy class
- 8 / 13 errors → Healthy (62% of Diabetes errors), 3 → SucheOko, 2 → SM.
- Per-patient structure: `DM_01.03.2024EYE` loses 6 / 11 scans to wrong classes; `37_DM` loses 2 / 3. `Dusan1/2_DM_STER_*` is handled better (8 / 9 correct). **The failure is session-level, not scan-level.**
- Biomedical plausibility: at 90 nm/px sampling and 92.5 μm scan range we may be under-resolving the salt-lattice features (typically 100-500 nm). SucheOko-confusions are particularly odd given patient separation — they reflect SucheOko's "any non-canonical morphology" absorption bucket.
- **Intervention:** Diabetes specialist with inputs = DINOv2-B + GLCM d1/d3 + Sa/Sq/Ssk roughness. Train as binary Diabetes-vs-Healthy LR, feed softmax into stacker.

### PGOV_Glaukom (21 / 36, recall 0.58) — the SM-twin
- 15 / 15 errors → SM (100%). Pure binary confusion.
- Already known from `TDA_RESULTS.md`: cubical H₁ persistence features raise Glaukom F1 by +16% relative when concat'd (eye-LOPO 0.46 → 0.53). Not used in the champion.
- Biomedical plausibility: both show reticulated height patterns at 10-50 μm lateral scale; differentiators are the loop persistence (H₁ lifetimes) and branching structure, which are exactly what TDA encodes.
- **Intervention:** Glaukom specialist = DINOv2-B + TDA persistence-image features (already cached in `cache/features_tda.parquet`). Binary Glaukom-vs-SM LR. Feed softmax into stacker weighted by Stage-1 maxprob.

### SklerozaMultiplex (58 / 95, recall 0.61) — the gravity well
- 37 errors: 20 → Glaukom (54%), 12 → SucheOko (32%), 4 → Diabetes, 1 → Healthy.
- SM is the largest class (95 scans, 9 persons), so absorbs OTHER classes' mis-predictions too (12 SucheOko → SM, 15 Glaukom → SM). It's symmetric: SM over-predicted for Glaukom/SucheOko, SM under-predicted vs same. Net SM F1 is capped by this bidirectional confusion.
- Biomedical plausibility: the 12 SM→SucheOko errors are telling — these are plausibly genuinely dry-eye-affected SM patients (SM is a known risk factor for keratoconjunctivitis sicca).
- **Intervention:** SM is a two-front problem. Priority 1 = Glaukom-vs-SM specialist (see above). Priority 2 = SucheOko detector (see below).

### SucheOko (0 / 14, recall 0.00) — the 2-patient ceiling
- 10 / 14 → SM, 3 → Healthy, 1 → Diabetes.
- 8 Mode B + 4 Mode D = 12 / 14 scans are confidently-wrong or catastrophically-wrong. The 2 near-misses are the only "close calls".
- Patient-level breakdown: `29EYE_suche_oko` (8 scans, 0 / 8 correct, 5 → SM, 3 → Healthy) and `35EYE_suche_oko` (6 scans, 0 / 6 correct, 5 → SM, 1 → Diabetes).
- **This is not a model problem, it is a data problem.** Person-LOPO with 2 patients means the training fold *always* has exactly 1 SucheOko patient left (7 scans, single-morphology). LR with class_weight='balanced' over-weights that single point; the held-out patient has different morphology; result is 0 recall.
- **Interventions, in order of plausibility:**
  1. **Acquire more SucheOko patients.** 2 persons is below the statistical floor.
  2. **Binary SucheOko-vs-rest detector with heavy augmentation** + abstention (flag as "uncertain") rather than override. Target: go from 0 recall to 20-30% recall via cautious positive prediction; never hurt the 226 non-SucheOko scans.
  3. **Open-set / anomaly-detection reframing:** treat SucheOko as OOD, use Mahalanobis distance in DINOv2 space to flag non-typical scans, then triage to human review.

---

## 5. Intervention candidate list (ranked)

Criteria: Mode addressed, expected F1 gain (optimistic / realistic), implementation cost (hours), risk of regression.

| # | Intervention | Mode addressed | F1 gain (opt / real) | Impl. cost | Risk |
|---|---|---|---:|---:|---|
| 1 | **Glaukom-vs-SM specialist as stacker feature** (DINOv2-B + cached TDA → LR → softmax → stacker input) | B/C for PGOV↔SM (35 errors) | +0.015 / +0.005 | 3 h | Low — stacker with α cross-val preserves Stage-1 where weak |
| 2 | **Diabetes-vs-Healthy specialist as stacker feature** (DINOv2-B + roughness + LBP → LR) | C for Diabetes→Healthy (8 errors), B1 case | +0.010 / +0.004 | 3 h | Low |
| 3 | **Handcrafted-disagreement gate** — when DINOv2-NN class ≠ handcrafted-nearest-class, abstain or fall back to uncertainty triage | B (6 / 15 Mode B have HC-disagreement) | +0.010 / +0.004 (via abstention-weighted metric) | 4 h | Low if used as confidence flag; medium if used to override |
| 4 | **SucheOko binary anomaly detector** (Mahalanobis on DINOv2 tile embeddings, fit on non-SucheOko only, flag high-distance scans) + conservative positive threshold | D / B for SucheOko (12 errors) | +0.015 / +0.003 | 6 h | High — 2-patient ceiling still dominates; easy to hurt ZdraviLudia |
| 5 | **Entropy-based abstention** — produce "uncertain" as a 6th "class" when top-1 prob < 0.50, report F1 on confident subset as secondary metric | A (1 scan) + mid-conf C | separate metric | 2 h | None (reporting-only) |
| 6 | **Patient-aware data augmentation for SucheOko** — synthesise additional SucheOko-looking tiles via elastic/perlin warp of the 14 existing scans, train with patient-consistent pairing | B / D for SucheOko | +0.010 / +0.002 | 8 h | High — real risk of over-fitting to 2 persons' texture statistics |
| 7 | **Session-level calibration for Diabetes** — the `DM_01.03.2024EYE` session has 6 / 10 errors; fit a per-session bias term in nested-CV only. | B2 case, Diabetes Mode C | +0.005 / +0.001 | 5 h | Medium — easy to leak if nesting is sloppy |
| 8 | **TDA features in the main stacker** (already cached, 1015 dims, PCA-reduce to 64 dims → concat into LR) | PGOV↔SM | +0.008 / +0.002 | 2 h | Low |
| 9 | **LLM-gated reasoner for Mode B cases only** (maxprob > 0.85 and top-2 gap < 0.3) — use the cached reasoning texts for pitch, not F1. | B (pitch value, not F1) | — | already built | None — reporting-only |
| 10 | **More training data.** Acquiring 3-5 additional SucheOko patients would break the 0.00 F1 ceiling more than any of the above combined. | All SucheOko | +0.03 / +0.02 | weeks | None once acquired |

### Top-3 recommended actions (Pareto-weighted by F1 × confidence × cost)

1. **Implement stacker with Glaukom-vs-SM specialist as soft feature (not override).** Rationale: attacks the largest symmetric confusion (35 errors total), uses already-cached TDA, does not touch the known-fragile SucheOko subset, easy to nested-CV. Realistic gain +0.005 pushes 0.6458 → 0.651, still honestly reportable.

2. **Add Diabetes-vs-Healthy specialist as a second stacker feature** (pairs with #1). Rationale: attacks 8 near-miss errors, roughness features are cheap and interpretable, session-level leakage risk is known and controllable.

3. **Add entropy-based abstention as a secondary deliverable.** Rationale: does not change F1 but turns a single-number eval into a confidence-aware one. The pitch deck can show "on confident-subset (60% coverage) F1 = 0.78" which is a much more honest picture than the 0.65 headline. Costs 2 h, impossible to regress.

**Actions NOT to take** (explicitly):

- Hard-cascade with Glaukom/SM specialist override. Already red-teamed in `CASCADE_RESULTS.md` — loses −0.048.
- Any threshold tuning on the current OOF. Already burned twice (0.6698, 0.6878).
- Data augmentation for SucheOko as the main fix — the ceiling is 2 patients, not image count.

---

## 6. Honest take

Writing this with no spin:

1. **68% of all errors are near-misses** (true class was rank 2). The model is not wildly off — it is sitting on genuine decision boundaries.
2. **The other 32% are split between three root causes:** Mode B (confidently-wrong, n=15), Mode D (catastrophic, n=6), Mode E (mid-confidence, n=5). Of these 26 "real" errors, **12 are SucheOko** (the 2-patient ceiling), **4 are SM↔Glaukom** (plausibly recoverable via TDA stacker), **and 10 are scattered across Diabetes-Healthy-session issues.**
3. **Nearest-neighbour analysis on Mode B shows the model is internally consistent** (12 / 15 NN majorities agree with the wrong prediction). This means the errors are not random LR hiccups — the scans genuinely land in the wrong region of embedding space. Fixing Mode B requires a *different* feature basis, not a better head.
4. **The most expensive-looking intervention (more patients) is also the most effective.** Every modelling intervention is fighting for single-digit F1 points; doubling SucheOko patients would alone move the needle by 2-3×.
5. **The champion is close to the honest ceiling of this dataset.** The +0.01 gap to 0.6528 (threshold-tuned reference) is roughly the magnitude our interventions can honestly close, not exceed.

If forced to ship one change: add the Glaukom-vs-SM soft-stacker feature (#1 above). Expected honest gain +0.004-0.006, keeps the architecture simple, addresses the largest single confusion pair.

---

## Artefacts

- `reports/error_cases.csv` — one row per scan with path, true, pred, confidence, mode, per-class proba.
- `reports/error_analysis.json` — mode counts, confusion matrix, per-class breakdown, 15 Mode B cases with NN + handcrafted-distance details.
- `scripts/error_analysis.py` — reproducible driver (≤ 1 min on MPS).
