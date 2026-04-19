# Pitch Narrative — 5-minute script
## Hack Košice 2026 / UPJŠ Teardrop Challenge

---

## 0. Hook (~20 s)

> **"Can you detect a disease from a single drop of a tear?"**

UPJŠ posed that question. A tear drops onto a mica slide, dries, and under an atomic-force microscope it crystallises into a fingerprint. Our job: map that fingerprint to one of 5 diseases — Healthy, Multiple Sclerosis, Diabetes, Glaucoma, Dry Eye.

---

## 1. The data (~30 s)

> One slide. Five classes. 240 AFM micrographs. Only 35 real human patients.

- 240 Bruker Nanoscope SPM height maps
- 5 classes: ZdraviLudia (Healthy), SklerozaMultiplex (SM), Diabetes, PGOV_Glaukom, SucheOko (Dry Eye)
- **35 unique persons** (after fixing an L/R eye leakage bug — more on that)
- Class imbalance 7:1 (SM has 95 scans; SucheOko has 14, from only **2 patients**)
- Heterogeneous scan parameters: 10 μm → 92.5 μm physical range, 256² → 4096² pixels

**Visual:** `reports/pitch/01_class_distribution.png` — bar chart per class (scans vs persons), SucheOko's 2-patient cliff visually obvious.

**Visual:** `reports/pitch/02_class_morphology_grid.png` — 5×3 grid of preprocessed AFM height maps. Each class has a distinct dendritic signature. Healthy = dense radial ferns; Glaukom = sparse granular; SM = heterogeneous crystalline rods; Dry Eye = fragmented.

---

## 2. The challenge (~40 s)

**Three things make this hard:**

1. **Tiny dataset.** 240 scans, 35 persons. Any fancy deep model memorises.
2. **Patient is the dominant latent variable.** UMAP of DINOv2 embeddings shows scans cluster by *person* more tightly than by *class*.
3. **L/R eye leakage.** Left and right eye of the same human are NOT independent samples — shared genetics, hydration, scan day. Our initial parser grouped them as separate "patients" — leakage. Our Validator agent caught this; we re-ran everything under `person_id`, F1 dropped by ~1 % (the honest signal was ~1 % inflated).

**Visual:** `reports/pitch/03_umap_embedding.png` — UMAP coloured by class on the left, by person on the right. Person-colouring clusters tighter. That's the red flag we caught.

---

## 3. Our approach — the orchestrator pattern (~45 s)

> We didn't just train a model. We ran a *research organisation* of AI agents.

- **Orchestrator (Claude):** kept `STATE.md` as single source of truth, dispatched rounds of specialist agents in parallel.
- **Specialist agents:** each explored a different direction — tiling, TDA, CGNN, ensembles, LLM reasoning, cascades.
- **Red-team agent:** audited every new winning claim. Re-evaluated with strict nested cross-validation. If the number didn't survive, we retracted it.

Over 3 rounds × ~10 agents we ran 12+ architectural experiments and **rejected 5 inflated headlines** (0.6698, 0.6780, 0.6878, cascade 0.6770, LLM-refined 0.6575). The discipline is the story.

---

## 4. The architecture — dynamic 3-tier inference (~60 s)

We ship a **confidence-routed 3-tier pipeline**:

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1  │  DINOv2-B + BiomedCLIP proba-average         │
│  (fast)   │  ~80 % of scans decided here                 │
└─────────────────────────────────────────────────────────┘
              │
              │ maxprob < 0.55  ────┐
              ▼                     ▼
┌─────────────────────────┐  ┌──────────────────────────┐
│ Stage 2                 │  │ Stage 3                  │
│ Binary specialists as   │  │ Claude CLI with          │
│ SOFT-BLEND features      │  │ retrieval-augmented      │
│ (not overrides!)         │  │ domain-knowledge prompts │
│ Glaukom↔SM, Diab↔Healthy │  │ → JSON + rationale       │
└─────────────────────────┘  └──────────────────────────┘
```

Key design decisions:

- **Tile → mean-pool → LR.** 9 × 512² non-overlapping tiles per scan, encoded by frozen ViT, mean-pooled, logistic regression head. +5 % F1 vs single crop.
- **Stage 2 = soft blend, not override.** We learned the hard way (cascade −0.048) that low Stage-1 confidence ≠ Stage-1 wrong. So binary specialists feed into a meta-score, not a flip.
- **Stage 3 = reasoning, not classification.** Claude CLI runs on the ~20 % uncertain scans, returns a clinical rationale citing nearest-neighbour reference cases and specific texture features. Marginal cost $0 (CLI subscription).

---

## 5. The headline number (~40 s)

> **0.6528 weighted F1, person-level LOPO. Baseline 0.615. Null baseline 0.276 ± 0.042.**

- Shipped raw-argmax model: **0.6346** (no thresholds, no tuning, guaranteed honest).
- With nested-CV per-class thresholds: **0.6528** (reference variant, reproducibly fair).
- Single-model floor: 0.615 (DINOv2-B alone).
- Random-labels null: 0.276 ± 0.042 — our model is **7 standard deviations above chance**.

**Visual:** `reports/pitch/04_confusion_matrix.png` — LOPO confusion matrix, green diagonal clearly visible.
**Visual:** `reports/pitch/05_per_class_metrics.png` — precision/recall/F1 per class:

| Class | F1 |
|---|---:|
| ZdraviLudia | 0.86 |
| SklerozaMultiplex | 0.72 |
| PGOV_Glaukom | 0.59 |
| Diabetes | 0.43 |
| SucheOko | 0.00 |

---

## 6. Why 0.6528 and not higher (~40 s)

> **Because credibility > hype.**

We produced five claims above 0.65:

| Claim | Rejected? | Why |
|---|---|---|
| 0.6878 (4-component concat + bias tune) | ✗ | Eye-level grouping + bias tuned on same OOF |
| 0.6780 (eye-level ensemble + thresholds) | ✗ | Eye-level grouping |
| 0.6770 (cascade @ thr=0.50) | ✗ | Gating threshold tuned on eval set |
| 0.6731 (double-gated cascade) | ✗ | Specialist threshold tuned on eval set |
| 0.6698 (thresholds tuned on OOF) | ✗ | Thresholds + subset both tuned on full OOF |

Red-team discipline: if the number embeds test-set reuse, we retract. Our honest 0.6528 is what judges and any downstream user can trust.

---

## 7. What doesn't work — honest negative results (~40 s)

> **We searched honestly and here's what we ruled out.**

| Approach | Honest F1 | Δ vs champion |
|---|---:|---:|
| Hard-override cascade (Glaukom/SM + Diab/Healthy specialists) | 0.6217 | −0.048 |
| LLM override (Claude picks top-1 vs top-2) | 0.6575 (on full 240) | −0.012 vs leaky 0.6698 |
| 4-component concat + LR + bias tuning (nested) | 0.6326 | −0.020 |
| CGNN alone (skeleton graph → GINE) | 0.365 | −0.29 |
| Meta-LR / meta-XGB stacker on 12-dim features | 0.51 / 0.54 | −0.14 / −0.11 |
| Fully-nested ensemble (subset + thresholds both inner) | 0.5847 | −0.068 |

**Root-cause framing:** at 240 scans we are at a data ceiling. Complexity inflates leakage, not signal. **Simple beats fancy under honest evaluation.**

---

## 8. Interpretability bonus (~40 s)

Two components of our pipeline are interpretability wins that judges cannot get from plain CNN scores:

### 8a. TDA — topological signatures per class

Persistent homology (H₀ + H₁, cubical complex, multi-scale) gives a **deformation-invariant, global connectivity fingerprint** of each scan:

- **Glaukom has fewest significant loops but highest max persistence.** Sparse dendrites with a few dominant enclosed cells.
- **Healthy has intermediate feature count with short persistences.** Smooth fern topology.
- **SM and Dry Eye share dense topology.** Explaining the SM↔Dry-Eye confusion directly.

TDA flat-concat didn't beat DINOv2 overall, but **boosted PGOV_Glaukom F1 from 0.46 → 0.53 (+16 % relative)**. It's a per-class specialist feature. Physical intuition: glaucoma's MMP-9 degradation signature *is* topological.

### 8b. LLM reasoning texts — clinical rationales for uncertain cases

For each of 47 uncertain cases, Claude returns a JSON verdict + short paragraph citing nearest-neighbour references and specific features.

**Example (true class = Diabetes, Stage-1 said Healthy, LLM correctly flipped):**
> *"Query's glcm_contrast_d5_mean (19.43) nearly matches Diabetes [4] (19.61) and is far lower than Healthy [1] (37.8); glcm_contrast_d1_mean (1.922) precisely matches Diabetes [4] (1.912), with elevated Sa (0.2154) and dissimilarity (2.094) consistent with denser tear lattice and higher osmolarity."*

This is what a clinician would want to see next to any uncertain prediction. Marginal cost: $0 (CLI subscription).

---

## 9. Future directions (~20 s)

- **More patients** — especially SucheOko (currently 2 persons; any fourth person breaks the ceiling).
- **9-class version** — the UPJŠ PDF references 4 additional diseases. Open-set detector needed.
- **Patient-level meta-features** — age, sex, scan-day — to explain residual variance currently absorbed by the person latent.
- **Synthetic data for rare classes** — diffusion model conditioned on SucheOko morphology to break the 2-patient barrier.
- **Clinical-site deployment** — real-time inference on scan capture with automatic reasoning for low-confidence cases.

---

## 10. Closing (~15 s)

> **0.6528 F1 with audit trail. Simple model, honest evaluation, clinical rationales.**

- One simple shippable model (`TearClassifier.load('models/ensemble_v1')`)
- One-line prediction API (`clf.predict_directory('/path/to/test_set')`)
- Full red-team history published (`reports/RED_TEAM_*.md`)
- LLM reasoning texts for every uncertain case (`cache/llm_reasoner_raw.jsonl`)

**Thank you. Questions?**

---

## Speaker notes / time budget

| Section | Target | Cumulative |
|---|---:|---:|
| 0. Hook | 20 s | 0:20 |
| 1. Data | 30 s | 0:50 |
| 2. Challenge | 40 s | 1:30 |
| 3. Orchestrator | 45 s | 2:15 |
| 4. Architecture | 60 s | 3:15 |
| 5. Headline | 40 s | 3:55 |
| 6. Why 0.6528 | 40 s | 4:35 |
| 7. Negative results | 40 s | 5:15 *(overflow buffer)* |
| 8. Interpretability | 40 s | 5:55 *(overflow buffer)* |
| 9. Future | 20 s | 6:15 |
| 10. Close | 15 s | 6:30 |

**Strict 5-min cut:** skip section 7 or 8; they are listed as bonus.

**Figure handoffs (all in `reports/pitch/`):**
1. Section 1 → `01_class_distribution.png`, `02_class_morphology_grid.png`
2. Section 2 → `03_umap_embedding.png`
3. Section 5 → `04_confusion_matrix.png`, `05_per_class_metrics.png`
4. Section 8 → `06_morphology_comparison.png` + LLM text quote from `reports/LLM_GATED_RESULTS.md`
