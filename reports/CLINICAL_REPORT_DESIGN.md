# Clinical Report Generator - Design Document

**Artifact:** `teardrop/clinical_report.py` + `reports/sample_clinical_reports/*.md`
**Author:** Rafael (Hack Kosice 2026, Teardrop challenge)
**Last updated:** 2026-04-18

---

## 1. Why this exists (the pitch killer)

Every ML hackathon deliverable in this track will stop at *"here is a softmax
over 5 classes."* That is not what a clinician can use. A clinician expects a
**report** - prose that grounds the prediction in observable quantities, names
the differential, and acknowledges its own uncertainty.

This module closes that loop with no additional modelling. Given any Bruker
SPM scan it produces a self-contained Markdown diagnostic report in under a
minute, combining:

- the ensemble's top-1 / top-2 prediction with confidence,
- quantitative morphology (Ra / Rq in nm, fractal D, GLCM contrast /
  homogeneity, skewness / kurtosis, heuristic Masmali grade),
- a plain-English interpretation keyed to the predicted class,
- **3 nearest training cases** in DINOv2-B embedding space,
- an honest confidence note with our validated F1 + human baseline,
- **class-specific caveats** distilled from `reports/ERROR_ANALYSIS.md`.

It is the difference between *"our model scores 0.69 F1"* and *"here is what
the model would tell a doctor, line by line, for this patient."*

## 2. Design constraints

| Constraint | Rationale |
|---|---|
| **LLM-free** | Reproducible, deterministic, works offline, no API key or Anthropic dependency. Every byte of output is a pure function of the scan + the ensemble weights + the cached reference stats. |
| **Uses the shipped champion** (`ensemble_v4_multiscale`) | Same model that reported the headline F1 = 0.6887 - no separate "report-only" branch that could diverge. |
| **Honest-by-construction** | Every report carries the weighted F1 + macro F1 + human-kappa baseline in the confidence note. The "what the model might be missing" section names the predicted class's documented failure modes. No report implies better performance than the audit. |
| **Per-scan latency <= 10 s** (after model load) | So it can drop into the Gradio demo as a new tab without making the UI sluggish. Model load itself is ~30 s, amortised across calls. |
| **No new training or feature extraction** | Re-uses `teardrop.features` (94-dim handcrafted) and `teardrop.infer.preprocess_and_tile_spm` verbatim. Zero new failure surface. |

## 3. Template structure (what each report contains)

The report template (rendered by `generate_clinical_report`) has 6 fixed
sections, in this order:

1. **Header block** - filename, scan date (harvested best-effort from the SPM
   ASCII header), pixel size, image dimensions, generation timestamp.
2. **Model prediction** - top-1 class name + confidence, top-2 differential +
   confidence, discrete confidence level (HIGH/MEDIUM/LOW), and a full
   posterior table sorted descending.
3. **Morphology assessment** - five quantitative bullets: Ra/Rq/Rz in
   nanometers (from the *plane-levelled raw* height map), fractal dimension
   D with std, GLCM contrast + homogeneity, height-distribution skewness +
   kurtosis, and a heuristic Masmali grade 0-4 (see Section 5).
4. **Evidence for the primary prediction** - a one-sentence narrative pulled
   from the class template, followed by 3-6 bullets combining the *observed*
   feature values with *class-biology explanations*.
5. **Similar reference cases** - top-3 nearest training scans in DINOv2-B
   embedding space (cosine similarity), with a "majority agrees/disagrees"
   voter note. The querying scan itself is excluded if it's in the index.
6. **Confidence note** - weighted F1 + macro F1 + human inter-rater kappa,
   framed as *"this is a preliminary assessment, requires clinical
   correlation."*
7. **What the model might be missing** - class-specific caveats distilled
   from `reports/ERROR_ANALYSIS.md`. For the two most fragile pairs
   (SM<->Glaukom, SM<->SucheOko) we also append a generic differential
   warning when they appear as top-1/top-2.
8. **Methods** - a short paragraph naming the ensemble, preprocessing,
   handcrafted-feature list, and retrieval index so the attending clinician
   has a footnote they can inspect.

### Why this structure

- **Prediction first** mirrors how radiology reports are read - most readers
  only need to see the top-1 and top-2. Everything below is audit trail.
- **Observed quantities before class interpretation** - a clinician will
  trust *"Ra = 222 nm"* more than *"consistent with hyperglycaemia"*. We
  always lead with the measurement.
- **Honest by default** - the F1 number and the human-kappa baseline appear
  in *every* report, not only in edge cases. You cannot read the output
  without being told how trustworthy it is.
- **Named failure modes** - every predicted class carries its own worst-case
  story from our LOPO audit. A Dry-Eye prediction explicitly warns
  *"2-patient ceiling, treat as a rule-out."* This is information the pure
  softmax cannot convey.

## 4. Class biology templates (where the prose comes from)

The five per-class templates in `CLASS_TEMPLATES` are hand-written, grounded
in:

- `reports/LITERATURE_BENCHMARK.md` (Masmali, Rolando, Daza et al. 2022,
  Felberg et al. 2008 - for inter-rater kappa and grade-semantic ranges).
- `teardrop/llm_reason.py::DOMAIN_CONTEXT` (class-specific morphology notes
  that were originally written for the LLM reasoner prompt).
- Cohort statistics computed in-house on the 240-scan TRAIN_SET (see
  `cache/clinical_reference_nm.json`).

Each template specifies:

- a one-sentence `summary` (what the tear-film looks like for this class),
- expected Masmali grade range,
- an `expected_sa_nm` band - the typical Ra in nanometers a clinician can
  sanity-check against (taken from the IQR of our TRAIN_SET cohort, cross-
  referenced with the class description in `app.py::CLASS_DESCRIPTIONS`),
- an `expected_fractal` band for fractal dimension,
- 3-4 `biomarker_bullets` - plain-English explanations linking *observed*
  features to *disease biology* (e.g. "MMP-9 mediated matrix degradation"
  for Glaucoma, "hyperglycaemia-induced glycation" for Diabetes).

At report time we first emit 3 *observation* bullets keyed to the scan's
measured values (using `_interpret_roughness / _interpret_fractal /
_interpret_glcm`), then append the biology bullets verbatim. The result is
a list that reads like *"here is what we see, here is why the predicted
disease would produce that pattern."*

## 5. Heuristic Masmali grade

Masmali 0-4 is a **visual** grading scale that requires a trained
optometrist. We do not claim to reproduce it. What we do produce is a crude
surrogate score that rises with:

- low fractal dimension (D < 1.76 - sparse branching),
- low GLCM homogeneity (< 0.55 - fragmented texture),
- high GLCM contrast (> 8.0),
- very thin fern layer (Ra < 50 nm).

The score clips at 4 and is reported next to the class-expected range so the
clinician can see whether the scan's shape matches the typical grade for the
predicted disease. Any mismatch (e.g. SucheOko prediction with heuristic
grade = 2) is itself a soft uncertainty signal - the report does not hide
it. This is reported as `Masmali grade (heuristic surrogate): N` so no-one
mistakes it for a real Masmali score.

## 6. Retrieval note

The `RetrievalIndex` uses the cached DINOv2-B tile embeddings
(`cache/tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz`). We mean-pool over the
tiles per scan, L2-normalise, and do cosine lookup. At query time we encode
the incoming scan through the same DINOv2-B branch of the shipped ensemble
(so the embedding space is exactly the same). The querying scan is excluded
from the candidate list via normalised absolute-path match.

If the cache is missing the report simply omits the "Similar reference
cases" section - nothing else breaks. This makes the module portable to
deployments without the full cache.

## 7. What is explicitly NOT in the report

To keep the template defensible we refused to include:

- **Masmali grade as an absolute number.** Only our heuristic surrogate,
  always labelled as such.
- **Per-class probability interpretations** ("consistent with keratoconus
  with 78% likelihood"). The class is already in the prediction block;
  repeating it in probability language would be spurious precision.
- **Patient personal identifiers or session metadata** beyond what is in
  the SPM header.
- **Any claim stronger than "consistent with" or "suggests"** in the
  biology bullets. Every bullet is softened to acknowledge the
  interpretation layer, never direct diagnosis.

## 8. Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| SPM header has no Date - clinician may want scan date | Report prints `(not embedded in SPM header)` rather than fabricating. |
| Retrieval cache unavailable | Section 5 is silently dropped; core report still renders. |
| Observed Ra is way outside expected band | `_interpret_roughness` explicitly flags *"markedly lower / elevated above the typical band"*. We never pretend the scan matches its class. |
| Top-1 is SucheOko | Section 7 hard-codes the 2-patient-ceiling warning. |
| Top-1 and top-2 are SM and Glaukom | Additional generic caveat appended warning of the most-confused pair. |
| Ensemble prediction changes upstream | Report re-renders from the *current* ensemble at call time; nothing is cached between runs. |

## 9. Deployment

- `teardrop/clinical_report.py` exposes `generate_clinical_report(scan_path,
  model_dir)` returning a Markdown string.
- `scripts/gen_sample_clinical_reports.py` regenerates the 5 pitch artefacts.
- `app.py` - the Gradio demo - wires the function into a new 4th tab
  ("Clinical Report") that runs the current loaded predictor and renders
  the Markdown inline.

### Per-scan latency (measured)

| Stage | Time |
|---|---:|
| Cold start (TTAPredictorV4 + DINOv2-B + BiomedCLIP load) | ~30 s |
| Per-scan encode + prediction + retrieval + render | 5-12 s |

The Gradio app keeps one live predictor so all subsequent reports are in the
fast-path.

## 10. Checklist: what a reviewer can verify

1. Open any of the 5 sample reports in `reports/sample_clinical_reports/`.
   Every number in the morphology block is reproducible by running
   `teardrop.features.extract_all_features` on the same scan.
2. The weighted F1 claim (0.6887) matches
   `models/ensemble_v4_multiscale/meta.json::config.honest_lopo_weighted_f1`.
3. Every "what the model might be missing" bullet traces to a statement in
   `reports/ERROR_ANALYSIS.md` (the class-level sections).
4. No report overstates confidence - even 99.9% predictions carry the human-
   kappa baseline and the class-specific failure caveat.
5. No report contains a Slovak class label without also spelling it out in
   English, for a non-native reader.
