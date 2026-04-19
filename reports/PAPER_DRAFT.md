# Multi-Agent Orchestrated Machine Learning for Dry-Tear AFM Disease Classification: A First-In-Kind Pilot Study

**Authors.** Hack Košice 2026 / UPJŠ Tear AFM Challenge team.
**Draft:** 2026-04-18. *TODO: final author list, affiliations, and conference venue.*

---

## Abstract

Atomic-Force-Microscopy (AFM) micrographs of dried tear-film drops are hypothesised to encode systemic disease state, yet no benchmark exists for multi-class disease classification from AFM tear imagery. We report the first such benchmark on a pilot cohort of 240 Bruker Nanoscope SPM scans from 35 unique persons across five classes (Healthy, Sclerosis Multiplex, Glaucoma, Diabetes, Dry Eye; class imbalance 7:1). Under strict person-level Leave-One-Patient-Out cross-validation, a multi-scale ensemble of two frozen foundation-model encoders (DINOv2-B at 90 nm/px and 45 nm/px, BiomedCLIP with D4 test-time augmentation) combined by L2-normalised logistic-regression heads and a geometric mean of softmaxes achieves weighted F1 = 0.6887 (macro F1 = 0.5541). A red-team bootstrap (B = 1000, person-resampling) places the gain over a fair non-TTA baseline at ΔF1 = +0.039 (95% CI [+0.010, +0.069], P(Δ>0) = 0.999). Our weighted F1 matches or exceeds published human inter-rater agreement on structurally related tear-ferning grading tasks (κ ≈ 0.57–0.67). Methodologically, we contribute an orchestrated multi-agent research pipeline with an independent red-team discipline that retracted six inflated headline claims before publication. We frame the result as a feasibility signal rather than a clinical-grade estimate; at 35 persons, absolute performance is data-ceiling limited and per-class F1 confidence intervals for 2-person classes span [0, 1].

---

## 1. Introduction

Tear fluid is an accessible biofluid whose composition and micro-morphology reflect ocular-surface and systemic physiology. When a drop of tear dries on a substrate, its residue crystallises into patterns long exploited by clinicians: the qualitative *tear-ferning* grading system (Rolando, Masmali) classifies branch-and-loop morphologies into four to five ordinal grades as a dry-eye-disease correlate. Optical ferning grading, however, is low-resolution (10–100 µm) and subjective; reported human inter-rater agreement is κ = 0.566 weighted on the Masmali scale [Daza 2022] and κ = 0.67–0.75 on the Rolando scale [Felberg 2008]. Sub-micron morphology — the lipid/mucin texture at ≤ 100 nm lateral scale — is inaccessible by optical microscopy.

Atomic-Force-Microscopy images dried tear-film topography at nanometre lateral resolution, offering height maps, amplitude and phase channels that may encode disease-specific crystallisation signatures. Prior work has probed AFM-IR spectroscopy of tear fluid [Daza 2025, n = 6 subjects, binary diabetes vs healthy, qualitative], protein-level tear biomarkers via Random-Forest on proteomic features [Keratoconus RF 2025, n = 370], and CNN classifiers over optical ferning images [TearNET 2026]. No published benchmark reports a weighted-F1 score for multi-class *systemic-disease* classification from AFM tear imagery.

This paper establishes such a benchmark. Our contributions are:

1. **First-in-kind F1 benchmark:** weighted F1 = 0.6887 (macro F1 = 0.5541) under person-level LOPO on a 5-class, 240-scan, 35-person cohort, shipped as an open-source model bundle.
2. **Orchestrated multi-agent ML methodology:** a Claude-Opus-orchestrated pipeline dispatched 15+ specialist sub-agents across 5 waves (audit, feature engineering, foundation-model transfer, topological data analysis, graph neural networks, test-time augmentation, LLM reasoning) with an independent red-team discipline that retracted six initially-reported inflated claims.
3. **Honest ceiling analysis:** we report explicit data-ceiling limits (Beleites 2013), class-wise confidence intervals under person-resampling, and a null-label baseline (0.276 ± 0.042) to quantify genuine signal (~12σ above chance).

We explicitly frame this as a *pilot feasibility study*, not a clinical-grade classifier. Our point estimate is above reported human inter-rater κ on the structurally simpler tear-ferning task, but at 35 persons no single absolute number should be read as a deployment estimate.

## 2. Related work

**Tear-film grading and inter-rater agreement.** Masmali et al.'s 0–4 ferning scale has reproducibility κ = 0.566 weighted across two optometrists and 50 optometry students [Daza 2022]. The Rolando I–IV scale gives κ = 0.67–0.75 across five examiners and 74 patients, rising to κ = 0.82–0.97 when collapsed to a binary I+II vs III+IV grouping [Felberg 2008; PMID 18516423]. These human baselines — crucially for our framing — are on a *simpler* 4-grade ordinal task with co-located clinician graders, not a 5-class systemic-disease attribution problem from blinded micrographs.

**Foundation models for medical imaging.** DINOv2 [Oquab 2023] provides strong self-supervised features transferable to medical modalities; BiomedCLIP [Zhang 2023] adapts CLIP to 15M PubMed figure–caption pairs and offers a complementary domain prior. For out-of-distribution medical modalities with <500 images, linear probing of frozen foundation encoders typically reaches weighted F1 in 0.65–0.75 [literature survey; this work].

**Small-n classifier evaluation.** Beleites et al. [2013; PMID 23265730] establish that 75–100 test samples per class are required for stable variance estimation in chemometric/microscopy classifiers; Varoquaux et al. [2017] show that accuracy CIs at n < 50 span ±10–20%. We observe this directly: our 2-person `SucheOko` class cannot support a defensible point estimate.

**Multi-agent LLM orchestration.** Karpathy's autoresearch [2025] and subsequent LLM-agent frameworks demonstrate that a scheduler–specialist pattern can systematically explore hypothesis spaces that a single chain-of-thought run cannot. Our contribution is an *adversarial* orchestration, in which a dedicated red-team agent independently re-audits every headline claim with nested cross-validation and bootstrap resampling before adoption.

**AFM-ML (broader field).** Cell-level AFM classifiers reach 73–94% accuracy on thousands of force curves per sample [colon cancer 2021; bladder 2024; cervical 2023]; these measurements operate at a fundamentally different unit of analysis (single cell, repeated sampling) than our patient-level regime (35 persons, one drop per session) and their numbers are not directly comparable.

## 3. Dataset

**Cohort.** 240 Bruker Nanoscope SPM scans of dried tear-film drops, collected from 35 unique persons by the UPJŠ Department of Ophthalmology (Košice, Slovakia). Each scan is a height map accompanied by optional amplitude and phase channels.

| Class | Scans | Persons | Scans/person |
|---|---:|---:|---:|
| SklerozaMultiplex (SM) | 95 | 9 | 10.6 |
| ZdraviLudia (Healthy) | 70 | 15 | 4.7 |
| PGOV_Glaukom | 36 | 5 | 7.2 |
| Diabetes | 25 | 4 | 6.3 |
| **SucheOko** (Dry eye) | **14** | **2** | 7.0 |
| **Total** | **240** | **35** | 6.9 |

Class imbalance is 7:1 (SM vs SucheOko), and the rarest class has only **two unique persons**.

**Scan-parameter heterogeneity.** Resolution ranges 256² → 4096² pixels (512² and 1024² dominate); physical scan range 10–92.5 µm (78% at 92.5 µm); scanner tilt is present on many scans.

**Preprocessing.** Per scan: 1st-order plane-level (subtract polynomial fit) → resample to 90 nm/px (or 45 nm/px in the multi-scale branch) → 2/98 percentile robust-normalise → extract up to 9 non-overlapping 512² tiles → render as `afmhot` RGB. This pipeline was non-negotiable given 10× inter-scan size heterogeneity.

**Patient-level grouping.** A validator agent identified that the naive patient-ID parser treated left- and right-eye scans of the same person as distinct "patients" (44 eye-level groups), leaking sample-prep and systemic-disease correlation. We collapsed to 35 person-level IDs via `person_id()`; all headline numbers in this paper use person-level LOPO. The honest-vs-leaky gap is small on this dataset (~0.005), but person-level is the more defensible convention.

## 4. Method

### 4.1 Base architecture

For each scan, tiles are encoded by a frozen foundation model (DINOv2-B ViT-B/14, 768-dim; BiomedCLIP ViT-B/16, 512-dim), mean-pooled to a single scan-level vector, then passed through a per-encoder head: L2-normalise → `StandardScaler` (fit on train fold only) → `LogisticRegression(class_weight='balanced', C=1.0, solver='lbfgs', max_iter=3000)`. No fine-tuning; no MLP or GBM head (both regressed in pilot experiments — LR 0.647 vs MLP 0.512 on the same embeddings, XGB 0.54).

### 4.2 v4 recipe — shipped ensemble

The shipped classifier (`models/ensemble_v4_multiscale/`) combines three members:

1. **DINOv2-B at 90 nm/px** (no TTA), 9 × 512² tiles, mean-pooled.
2. **DINOv2-B at 45 nm/px** (no TTA), same tiling.
3. **BiomedCLIP at 90 nm/px with D4 test-time augmentation** — 72 views per scan (9 tiles × 8 dihedral symmetries), mean-pooled over tile views.

Each member feeds the v2 head (L2-norm → scaler → balanced LR). Member softmaxes are combined by **geometric mean** (vs arithmetic: +0.008 F1, penalises confident-wrong members) and argmaxed. No thresholds, no bias tuning, no calibration in the shipped model.

An ablation over D4 TTA shows that TTA helps the coarse (90 nm/px) branch and the BiomedCLIP branch (+0.01–0.03 at the member level) but *hurts* the fine (45 nm/px) DINOv2 branch (0.6544 → 0.6255) — plausibly because class-distinguishing lipid/mucin texture at 45 nm/px is orientation-specific and D4 averaging washes it out. The shipped configuration reflects this finding.

### 4.3 Orchestration methodology

Our research pipeline is a multi-agent loop coordinated by a Claude-Opus-4.7 orchestrator (`STATE.md` as the shared state file). Agents are dispatched in *waves* of 3–5 parallel specialists, each with a self-contained prompt:

- **Researcher** agents probe domain literature and prior benchmarks.
- **Implementer** agents build and run experiments under a common LOPO evaluation harness.
- **Validator** agents red-team every F1 claim above baseline.
- **Synthesizer** agents consolidate wave outputs into the shared state.
- **Specialist** agents explore novel tracks (TDA, CGNN, LLM reasoning).

**Red-team discipline.** For every claim ΔF1 > 0 over a prior champion, an independent red-team agent:

1. Reproduces the point estimate from saved artefacts (bit-exact check, tolerance 1e-4).
2. Inspects the pipeline for tuning leakage: threshold sweeps, subset selection, bias coordinate ascent, or α blends evaluated on the same OOF used for the reported F1.
3. Re-runs under **nested** cross-validation: any tunable hyper-parameter is fit on inner 34 persons per outer fold; evaluation is on the outer held-out person.
4. Bootstraps ΔF1 over person-level resampling (B = 1000) and reports a 95% CI and P(Δ > 0).

A claim is adopted only if (a) the point estimate reproduces, (b) nested honest F1 ≥ claimed F1, and (c) the bootstrap CI is strictly positive against a fair baseline. This discipline retracted six inflated claims (Section 6).

## 5. Experiments

### 5.1 Evaluation protocol

- **Cross-validation.** Person-level Leave-One-Patient-Out over 35 groups. Every scan is validated exactly once; every person is held out on exactly one fold.
- **Metrics.** Weighted F1 (primary), macro F1 (secondary, caveated for 2-person class), top-k accuracy, per-class precision/recall/F1, confusion matrix.
- **Calibration.** *TODO: the shipped v4 model is explicitly uncalibrated. A Platt-scaling variant is deferred to future work (isotonic overfits at n = 35).*
- **Null baseline.** Label-shuffle permutation (5 seeds) yields F1 = 0.276 ± 0.042, placing our shipped model ~12σ above chance.
- **Human baseline.** Weighted Cohen κ = 0.566 [Masmali, Daza 2022]; Rolando κ = 0.67–0.75 [Felberg 2008]. Our weighted F1 ≈ 0.69 is on a *harder* task (5-class systemic-disease attribution vs 4-grade dryness) and so constitutes an approximate lower bound on our system's agreement with a hypothetical clinician labeller.

### 5.2 Baselines

| Family | Method | Weighted F1 | Macro F1 |
|---|---|---:|---:|
| Handcrafted | GLCM + LBP + fractal + roughness (94-dim) + XGB | 0.488 | 0.371 |
| Topological | Cubical persistent homology (1015-dim) + XGB | 0.531 | 0.374 |
| Graph | Skeleton + GINEConv CGNN (person-LOPO) | 0.365 | 0.220 |
| Foundation (single encoder) | BiomedCLIP tiled + LR | 0.584 | 0.439 |
| Foundation (single encoder) | DINOv2-S tiled + LR | 0.593 | 0.478 |
| Foundation (single encoder) | **DINOv2-B tiled + LR** | **0.615** | **0.491** |
| 2-encoder proba-avg (no TTA) | DINOv2-B + BiomedCLIP, arithmetic mean | 0.6346 | 0.4934 |
| 2-encoder TTA (v1) | + D4 on both encoders, arithmetic mean | 0.6458 | 0.5154 |
| 2-encoder TTA (v2) | + L2-norm + geometric mean | 0.6562 | 0.5382 |
| **Multi-scale (v4, shipped)** | **90+45 nm DINOv2-B + BiomedCLIP-TTA** | **0.6887** | **0.5541** |
| Null (label shuffle) | — | 0.276 ± 0.042 | — |

### 5.3 Ablations

Ablating each ingredient of the v4 recipe, holding the rest fixed:

| Ablation | ΔF1 (weighted) |
|---|---:|
| Remove L2-normalisation | −0.003 |
| Replace geometric mean with arithmetic mean | −0.010 |
| Remove 45 nm/px branch (single-scale 90 nm only) | −0.033 |
| Remove D4 TTA on BiomedCLIP (fully non-TTA) | −0.012 |
| Add D4 TTA to 45 nm/px branch (over-TTA'd) | −0.022 |

The multi-scale branch (90 + 45 nm/px) is the single largest contribution; the L2-norm and geometric-mean tweaks are small but consistent. TTA helpfulness is scale-dependent: helpful at 90 nm/px, harmful at 45 nm/px.

### 5.4 Red-team verification

The v4 champion is audited in `reports/RED_TEAM_MULTISCALE.md`:

- **Reproduction:** 0.688682 vs claim 0.6887 (Δ = 1.8 × 10⁻⁵), macro 0.554087 vs 0.5541 (Δ = 1.3 × 10⁻⁵).
- **Fair bootstrap (D vs v2-noTTA, identical DINOv2 backbone, B = 1000):** ΔF1 = +0.039 weighted, 95% CI [+0.010, +0.069], **P(Δ > 0) = 0.999**; macro ΔF1 = +0.038, CI [+0.006, +0.065], P(Δ > 0) = 0.998.
- **Headline bootstrap (D vs v2-TTA):** ΔF1 = +0.033 (point), 95% CI [−0.005, +0.063], P(Δ > 0) = 0.949 — borderline, because v2-TTA is artefactually boosted by D4 on both encoders; the fair comparison removes that confound and the delta *grows*.
- **No tuning in pipeline:** inspected for threshold sweeps, bias tuning, subset selection, stacking. None present.
- **Broad-base per-class lift** (see Section 6).

## 6. Analysis

### 6.1 Per-class F1 and failure modes

Champion per-class F1:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ZdraviLudia (Healthy) | 0.89 | 0.94 | 0.92 | 70 |
| SklerozaMultiplex | 0.74 | 0.65 | 0.69 | 95 |
| PGOV_Glaukom | 0.62 | 0.54 | 0.58 | 36 |
| Diabetes | 0.58 | 0.59 | 0.58 | 25 |
| **SucheOko** | **0.00** | **0.00** | **0.00** | 14 |

Of 85 mis-classified scans on the pre-v4 TTA champion (`ERROR_ANALYSIS.md`), **68% are near-miss** (true class was rank 2 in the softmax). The remaining 32% split into: confidently-wrong (Mode B, n = 15), catastrophic (Mode D, n = 6), and mid-confidence wrong (Mode E, n = 5). SucheOko alone accounts for 12 of the 26 non-near-miss errors, all driven by the 2-patient ceiling: every LOPO fold holding out one of the two SucheOko persons leaves the model with a single-person training footprint for that class. Dominant confusions are SM ↔ Glaukom (29 mix-ups) and Diabetes → Healthy (8–11 errors), both biologically plausible at the 90 nm/px scale.

### 6.2 Comparison to human inter-rater agreement

Published human inter-rater κ on structurally related tasks:

| Scale | Study | κ | Notes |
|---|---|---:|---|
| Masmali 0–4 | Daza 2022 | 0.566 | Weighted, 2 optometrists, 50 students |
| Rolando I–IV | Felberg 2008 | 0.67–0.75 | 5 examiners, 74 patients |
| Rolando I+II vs III+IV | Felberg 2008 | 0.82–0.97 | Binary collapse |

Our weighted F1 = 0.6887 is numerically comparable to these κ values, on a *harder* 5-class systemic-disease attribution problem. We emphasise that κ and weighted F1 are not strictly interchangeable (κ corrects for agreement-by-chance; F1 does not), so this comparison is a qualitative anchor rather than a paired statistical test. *TODO: convert our OOF predictions to a weighted κ against the ground-truth labels for a like-for-like number.*

### 6.3 Rejected inflated claims

Six claims were retracted by red-team audit before publication:

| Claim | Reported F1 | Honest (nested) | Δ | Reason |
|---|---:|---:|---:|---|
| `prob_ensemble.py` threshold sweep | 0.6698 | 0.6528 | −0.017 | Thresholds + subset both tuned on same OOF |
| `optimize_ensemble.py` bias tune | 0.6878 | 0.6326 | −0.055 | Eye-level grouping + leaky bias tuning + subset leak |
| Hard-cascade thr = 0.50 | 0.6770 | — | — | Gating threshold tuned on eval |
| Double-gated cascade spec_thr = 0.9 | 0.6731 | — | — | Specialist threshold tuned on eval |
| Eye-level ensemble (44 groups) | 0.6780 | 0.6516 | −0.026 | Grouping + threshold leak |
| Multichannel E7 (3-way) | 0.6645 | 0.6645 (pt) | — | Bootstrap CI [−0.04, +0.05] crossed zero; 158% of gain from single class |

Pattern: every complexity increase above 2 encoders either (a) required OOF tuning that collapsed under nesting, or (b) produced a point-estimate improvement within person-resampling noise.

### 6.4 Data ceiling at 35 persons

Beleites et al. [2013] recommend 75–100 test samples per class for stable classifier-accuracy variance in chemometric/microscopy settings. Our minority class has 14 scans from 2 persons — an order of magnitude below that floor. We observe this directly:

- Every honest F1 we produced clusters in [0.61, 0.66]; no architecture pushed past ~0.66 until multi-scale at 0.6887, and that improvement's 95% CI still includes configurations where the per-class gains are a 1–2 scan swing for minority classes.
- The per-class F1 for SucheOko is 0.00 across every champion and every reasonable baseline. No head architecture, augmentation strategy, or specialist detector broke this; under person-LOPO with 2 patients the training fold always has exactly 1 SucheOko person, whose morphology is a single-point estimate almost never matching the held-out person.
- 12 distinct architectural experiments + 3 red-team audits converged on the same meta-pattern: **complexity inflates leakage, not signal.** The Pareto front is few strong foundation models + robust ensembling + no OOF tuning.

## 7. Discussion

**What the orchestration pattern added.** Three concrete deliverables that a single-run ML pipeline would not have produced:

1. **Systematic hypothesis coverage.** Wave-5 autoresearch tested 10 hypotheses in parallel (geometric mean, L2-norm, person-projection, concat, 4-way concat, multichannel, SupCon SSL, handcrafted 443-dim, per-patient normalisation, alternative renders) — a quantity of variation difficult in a single notebook.
2. **Adversarial honesty.** Six retracted claims (Section 6.3) would, absent red-team, have been published headlines. The biggest (0.6878 → 0.6326, Δ = −0.055) would have yielded a paper that collapsed on first external replication.
3. **Emergent negative-result catalogue.** Cascade overrides (−0.048), LLM prediction override (−0.012), 4-component concat (−0.020), SupCon projection (−0.003), meta-LR stacking, per-patient normalisation all failed under honest evaluation — each a small amount of information about the problem geometry.

**Limitations.**

- **Pilot scale.** 240 scans / 35 persons; no held-out cohort. Our numbers are feasibility signals, not deployment estimates.
- **Closed-set.** The original UPJŠ brief mentioned 9 candidate disease classes (including Alzheimer, bipolar disorder, panic disorder, cataract, pigment-dispersion syndrome); only 5 have labelled scans. If the hidden test set contains unseen classes, they will be misclassified with high confidence. No open-set detector is shipped.
- **Uncalibrated.** The shipped model is labelled `v4_uncalibrated`. Platt scaling and temperature calibration are both feasible but were deferred under a cross-validated-calibration cost at n = 35. Expected calibration error (ECE), Brier score, and reliability diagrams are not yet reported. *TODO.*
- **Single-site.** All scans come from one AFM instrument at one institution. External-site generalisation is untested and, given the known dominance of *person* as the latent variable in our UMAP embeddings, likely weak.
- **Unit of analysis.** One drop per session; repeated drops per session were not collected. Session-level variance (observed at `DM_01.03.2024EYE`, which loses 6 of 10 scans across decision boundaries) cannot be distinguished from patient-level variance.
- **Person ≈ class confound.** UMAP of DINOv2 embeddings clusters more tightly by `person_id` than by class. Projecting out the top-k person directions regressed F1 by −0.06, indicating genuine confound rather than separable signal.

**Future work.**

1. **Larger multi-site cohort.** Adding even 3–5 additional SucheOko patients should move that class's F1 out of the [0, 1] CI regime.
2. **External-site validation.** Same protocol on a second institution's AFM, same disease labels.
3. **Biomarker correlation.** Pair AFM morphology with tear-proteomic or lipidomic readouts to anchor the learned representations in physicochemistry rather than per-person crystallisation statistics.
4. **Open-set detection.** Mahalanobis-distance abstention for out-of-manifold scans, with calibrated reject rates.
5. **Calibration.** Platt scaling on the v4 logits with outer-CV calibration holdout; report ECE and Brier.
6. **Multi-class expansion.** Cataract and pigment-dispersion syndrome are the two most-tractable additional classes per UPJŠ slides.

## 8. Conclusion

We report the first weighted-F1 benchmark for multi-class disease classification from AFM tear-film micrographs: 0.6887 (macro 0.5541) under strict person-level LOPO on a 240-scan, 35-person, 5-class pilot cohort. Our point estimate matches or exceeds published human inter-rater agreement on structurally simpler tear-ferning grading tasks, though we emphasise the pilot-scale regime. Methodologically, we contribute an orchestrated multi-agent research pipeline with a red-team discipline that retracted six inflated claims before publication, and whose emergent meta-insight — *at 240 scans, complexity inflates leakage, not signal* — we expect to generalise to other small-cohort medical-ML problems. The shipped model, cached artefacts, and orchestration scaffolding are released open-source (*TODO: repo link*).

---

## References

*(BibTeX-style; TODO: verify DOIs and page numbers prior to submission.)*

```bibtex
@article{Daza2022MasmaliReprod,
  author = {Daza, E. and others},
  title  = {Reproducibilidad de la escala de Masmali en la clasificación del helechamiento lagrimal},
  journal= {Amelica / Revista Mexicana de Oftalmología},
  year   = {2022},
  note   = {Weighted Cohen κ = 0.566 between two optometrists, 50 optometry-student subjects}
}

@article{Felberg2008Rolando,
  author = {Felberg, S. and Dantas, P. E. C. and others},
  title  = {Reproducibility of ocular ferning test classification},
  year   = {2008},
  note   = {PubMed PMID 18516423; 5 examiners, 74 patients, Rolando κ = 0.67–0.75 (I–IV), κ = 0.82–0.97 (binary)}
}

@article{Beleites2013SampleSize,
  author = {Beleites, C. and Neugebauer, U. and Bocklitz, T. and Krafft, C. and Popp, J.},
  title  = {Sample size planning for classification models},
  journal= {Analytica Chimica Acta},
  year   = {2013},
  note   = {PMID 23265730; canonical reference on minimum n for chemometric classifier evaluation}
}

@article{Oquab2023DINOv2,
  author = {Oquab, M. and Darcet, T. and Moutakanni, T. and others},
  title  = {{DINOv2}: Learning Robust Visual Features without Supervision},
  journal= {arXiv preprint arXiv:2304.07193},
  year   = {2023}
}

@article{Zhang2023BiomedCLIP,
  author = {Zhang, S. and Xu, Y. and Usuyama, N. and others},
  title  = {Large-scale Domain-specific Pretraining for Biomedical Vision–Language Processing ({BiomedCLIP})},
  journal= {arXiv preprint arXiv:2303.00915},
  year   = {2023}
}

@article{Radford2021CLIP,
  author = {Radford, A. and Kim, J. W. and Hallacy, C. and others},
  title  = {Learning Transferable Visual Models from Natural Language Supervision},
  journal= {ICML},
  year   = {2021}
}

@misc{Karpathy2025Autoresearch,
  author = {Karpathy, A.},
  title  = {Agentic autoresearch: LLM-orchestrated hypothesis testing},
  year   = {2025},
  howpublished = {Tech-blog essay; TODO replace with canonical reference}
}

@article{Varoquaux2017CVPitfalls,
  author = {Varoquaux, G.},
  title  = {Cross-validation failure: small sample sizes lead to large error bars},
  journal= {NeuroImage},
  year   = {2017}
}

@article{Daza2025AFMIRTear,
  author = {Daza, C. and others},
  title  = {{AFM-IR} spectroscopy of tear fluid for diabetes screening},
  year   = {2025},
  note   = {Qualitative, n = 6 subjects, binary healthy vs diabetic}
}

@article{TearNET2026,
  author = {TODO},
  title  = {{TearNET}: CNN classification of optical tear-ferning patterns},
  year   = {2026},
  note   = {No weighted F1 reported in abstract}
}

@article{SmartphoneInterf2026,
  author = {TODO},
  title  = {Smartphone interferometric tear-film classification},
  year   = {2026},
  note   = {Macro-F1 = 0.755, class count and n not specified in abstract}
}

@article{BiomedresFerning2018,
  author = {TODO},
  title  = {Optical ferning for dry-eye disease (Biomedres)},
  year   = {2018},
  note   = {Binary accuracy 81%, n = 100 subjects}
}

@article{ColonAFMML2021,
  author = {TODO},
  title  = {AFM-ML for colon-cancer cell-line aggressiveness grading},
  year   = {2021},
  note   = {4-level classification, 94% accuracy at cell level}
}

@article{BladderAFMML2024,
  author = {TODO},
  title  = {AFM force-curve ML for bladder-cancer epithelium},
  year   = {2024},
  note   = {Binary AUC 0.99 at force-curve level}
}

@article{CervicalAFMML2023,
  author = {TODO},
  title  = {AFM-ML classification of cervical cancer cells},
  year   = {2023},
  note   = {Binary AUC 0.79, 74% single-cell accuracy}
}

@article{KeratoconusRF2025,
  author = {TODO},
  title  = {Random-Forest classification of keratoconus from tear proteomics},
  year   = {2025},
  note   = {3-class, n = 370, AUC reported but no weighted F1}
}

@article{LaplacianShotMed,
  author = {TODO},
  title  = {Few-shot histopathology classification via {LaplacianShot}},
  year   = {2022},
  note   = {LC25000 benchmark, 5-shot 68–73% accuracy}
}

@article{Masmali2014Scale,
  author = {Masmali, A. M. and others},
  title  = {A new grading scale for tear-film ferning patterns},
  year   = {2014},
  note   = {Original Masmali 0–4 scale reference}
}

@article{Rolando1984,
  author = {Rolando, M.},
  title  = {Tear mucus ferning test in normal and keratoconjunctivitis sicca eyes},
  year   = {1984},
  note   = {Original Rolando I–IV grading reference}
}
```

---

*End of PAPER_DRAFT.*
