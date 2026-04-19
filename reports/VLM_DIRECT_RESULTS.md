# VLM Direct-Classification Baseline — HONEST Evaluation

Send each AFM scan as a rendered afmhot PNG directly to Claude Haiku 4.5 via the `claude -p` CLI and parse its JSON prediction. No training, no features — just image + biologically informed prompt.

## Key finding (up front)

**The off-the-shelf VLM cannot classify AFM tear-ferning scans above chance when the filename does not leak the class.** On all 240 scans with obfuscated tile names (`scan_XXXX.png`), Claude Haiku 4.5 scores **accuracy = 0.3083**, **weighted-F1 = 0.2259**, **macro-F1 = 0.1971**. Random baseline for 5 classes is 20% accuracy. The v4 champion remains at weighted-F1 = 0.6887.

## The contamination we had to fix first

An earlier version of `scripts/vlm_direct_classify.py` rendered tiles with class-name-prefixed filenames (`cache/vlm_tiles/Diabetes__37_DM.png`) and embedded that path into the prompt. Claude's Read tool saw the path before reading the image, and the model shortcut-classified from the filename. That run reported a falsely-high accuracy of ~88% and is retracted in full. Re-run artefacts:

- archived leaky predictions: `cache/vlm_predictions_LEAKY.json.bak`
- archived leaky tiles: `cache/vlm_tiles_LEAKY.bak/`
- prior red-team note: `reports/VLM_CONTAMINATION_FINDING.md`

Leaky run score, for reference: accuracy 0.9083, weighted-F1 0.9085, macro-F1 0.8998 (these numbers are NOT real VLM performance — they're the VLM reading a class name out of a path).

The fix is trivial but required: tile names are now class-neutral (`scan_0000.png` under `cache/vlm_tiles_honest/`), and the prompt only exposes that neutral filename. Ground truth is held in `cache/vlm_honest_manifest.json`, which the VLM never sees.

## Methodology

- **Image rendering.** Each SPM file: `preprocess_spm(target=90 nm/px, crop=512)` → Matplotlib `afmhot` colormap → PNG. Identical preprocessing to the v4 multi-scale pipeline.
- **Filename obfuscation.** `cache/vlm_tiles_honest/scan_NNNN.png`, randomised mapping stored in `cache/vlm_honest_manifest.json`. The VLM sees only the neutral filename in the prompt.
- **Model.** `claude-haiku-4-5` (2026 release). ~16 s/call, ~$0.01/call on this prompt size.
- **Prompt.** Same biologically-informed prompt as the leaky run (Masmali grades, fractal dimensions, MMP-9 hints, per-class morphological signatures). Only the path changed.
- **Parallelism.** 8 worker processes via `ProcessPoolExecutor` (`scripts/vlm_honest_parallel.py`). 240 scans finished in ~7 minutes wall-clock.
- **Scoring.** Hard argmax from `predicted_class`. Pseudo-softmax for ensembling: `conf` on predicted class, `(1-conf)/4` elsewhere.

## Coverage

| Class | Scored | Total |
|---|---:|---:|
| ZdraviLudia | 70 | 70 |
| Diabetes | 25 | 25 |
| PGOV_Glaukom | 36 | 36 |
| SklerozaMultiplex | 95 | 95 |
| SucheOko | 14 | 14 |
| **TOTAL** | **240** | **240** |

## Subset (stratified, person-disjoint, 5 per class)

n = 21  |  accuracy = **0.2381**  |  macro-F1 = **0.1333**  |  weighted-F1 = **0.1587**  |  mean confidence = 0.781

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ZdraviLudia | 0.500 | 1.000 | 0.667 | 5 |
| Diabetes | 0.000 | 0.000 | 0.000 | 4 |
| PGOV_Glaukom | 0.000 | 0.000 | 0.000 | 5 |
| SklerozaMultiplex | 0.000 | 0.000 | 0.000 | 5 |
| SucheOko | 0.000 | 0.000 | 0.000 | 2 |

Confusion matrix (subset):

| true \ pred | Zdravi | Diabet | PGOV_G | Sklero | SucheO |
|---|---:|---:|---:|---:|---:|
| ZdraviLudia | 5 | 0 | 0 | 0 | 0 |
| Diabetes | 3 | 0 | 1 | 0 | 0 |
| PGOV_Glaukom | 0 | 3 | 0 | 0 | 2 |
| SklerozaMultiplex | 1 | 1 | 3 | 0 | 0 |
| SucheOko | 1 | 0 | 0 | 1 | 0 |

## Full 240 scans

n = 240  |  accuracy = **0.3083**  |  macro-F1 = **0.1971**  |  weighted-F1 = **0.2259**  |  mean confidence = 0.777

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ZdraviLudia | 0.500 | 0.900 | 0.643 | 70 |
| Diabetes | 0.098 | 0.200 | 0.132 | 25 |
| PGOV_Glaukom | 0.174 | 0.111 | 0.136 | 36 |
| SklerozaMultiplex | 0.000 | 0.000 | 0.000 | 95 |
| SucheOko | 0.051 | 0.143 | 0.075 | 14 |

Confusion matrix:

| true \ pred | Zdravi | Diabet | PGOV_G | Sklero | SucheO |
|---|---:|---:|---:|---:|---:|
| ZdraviLudia | 63 | 7 | 0 | 0 | 0 |
| Diabetes | 19 | 5 | 1 | 0 | 0 |
| PGOV_Glaukom | 7 | 14 | 4 | 0 | 11 |
| SklerozaMultiplex | 27 | 25 | 17 | 0 | 26 |
| SucheOko | 10 | 0 | 1 | 1 | 2 |

The class-conditional pattern is telling: recall is above-chance only for `ZdraviLudia` (the model defaults to healthy when uncertain) and very weak elsewhere. `SklerozaMultiplex` recall is 0 — SM scans are consistently called `ZdraviLudia` or, less often, `Diabetes`.

## Comparison to Champion v4 (honest LOPO)

v4 multi-scale: weighted-F1 = **0.6887**, macro-F1 = **0.5541** (per `models/ensemble_v4_multiscale/meta.json`).

- v4 on its full 240 OOF: macro-F1 = 0.5541, weighted-F1 = 0.6887
- Overlap with VLM scored samples: n = 240
  - VLM alone: macro-F1 = 0.1971, weighted-F1 = 0.2259
  - v4 alone (on same overlap): macro-F1 = 0.5541, weighted-F1 = 0.6887

### Blend of v4 and VLM probability vectors

| Weighting | macro-F1 | weighted-F1 |
|---|---:|---:|
| v4*0.9 + vlm*0.1 | 0.5527 | 0.6859 |
| v4*0.8 + vlm*0.2 | 0.5509 | 0.6837 |
| v4*0.7 + vlm*0.3 | 0.5291 | 0.6671 |
| v4*0.5 + vlm*0.5 | 0.4610 | 0.5988 |

Every blend that gives VLM non-trivial weight *hurts* the ensemble, as expected when the second voter is near-random.

## Qualitative reasoning — top-confidence correct

Despite the poor F1, the VLM does produce fluent, clinically-plausible morphology narratives. These are genuinely useful as clinical-report scaffolding and are the pitch-worthy deliverable of this experiment.

**8P.001** — true/pred=`ZdraviLudia` (confidence 0.92):

> Dense, uniform dendritic ferning with well-organized branching and consistent crystal thickness throughout the field—classic Masmali grade 0-1 healthy tear morphology. Regular packing density and fine-grained structure with no fragmentation or granular chaos.

**49.004** — true/pred=`ZdraviLudia` (confidence 0.88):

> Dense, hierarchical dendritic ferning with uniform fine-branching structure and well-organized spacing are characteristic of healthy tears. No granular texture, fragmentation, or morphological heterogeneity observed.

**9P.001** — true/pred=`ZdraviLudia` (confidence 0.88):

> Dense, uniform dendritic ferning with regular branching geometry and high network density characteristic of healthy tear morphology. No visible granularity, fragmentation, or coarse crystal thickening; consistent topology across scan field consistent with Masmali grade 0-1.

**9P.002** — true/pred=`ZdraviLudia` (confidence 0.88):

> Dense radial dendritic ferning with uniform fine-branch distribution and regular crystalline organization throughout the field. No granularity, fragmentation, or heterogeneity observed—morphology consistent with healthy tear fluid crystallization (Masmali grade 0-1, estimated fractal D ~1.75).

**7.002** — true/pred=`ZdraviLudia` (confidence 0.88):

> Dense, uniformly branching dendritic ferning pattern with fine, consistent-thickness branches and complete field coverage. Lack of fragmentation, granular texture, or heterogeneous coarse/fine mixing indicates healthy tear crystallization morphology consistent with Masmali grade 0-1.

## Qualitative reasoning — top-confidence WRONG

Each of these shows the model writing confident prose that looks like a clinical note while getting the class wrong — exactly the behaviour we'd expect from a VLM with no in-domain training. These are still useful as prompts for follow-up experiments (e.g., few-shot, prompt rewriting).

**DM_01.03.2024_LO.003** — true=`Diabetes`, pred=`ZdraviLudia` (confidence 0.92):

> Dense, uniform dendritic ferning with fine, consistent branching throughout the field of view; elevated crystal topography (bright coloration) indicates well-organized crystalline structure typical of Masmali grade 0-1 healthy tear morphology. No amorphous regions, granularity, or fragmentation evident.

**22_PV_SM.005** — true=`SklerozaMultiplex`, pred=`ZdraviLudia` (confidence 0.88):

> Dense hierarchical dendritic ferning with uniform fine branching throughout, minimal fragmentation or amorphous regions, and elevated fractal complexity consistent with healthy tear crystallization (Masmali 0–1 grade, estimated D ~1.78).

**Dusan2_DM_STER_mikro_281123.003** — true=`Diabetes`, pred=`ZdraviLudia` (confidence 0.88):

> Dense, hierarchically organized dendritic ferning with uniform fine branching throughout the field. The regular, tapered branch structure and high packing density of the network are consistent with Masmali grade 0-1 morphology and estimated fractal dimension 1.75-1.80.

## Reproducibility

```bash
# 25-scan stratified subset, class-neutral tile filenames
python scripts/vlm_direct_classify.py --subset 5

# Full 240 scans, parallel workers, same obfuscation scheme
python scripts/vlm_honest_parallel.py --full --workers 8

# Regenerate this report from the cache
python scripts/vlm_report.py
```

## Takeaways for the pitch

1. **Null result that is itself informative.** A foundation VLM with strong performance on natural images cannot classify AFM tear scans (weighted-F1 ~0.23). The morphology of dried biological films is sufficiently out-of-distribution that zero-shot transfer fails outright.
2. **Red-team method validated.** The contamination audit (renaming tiles to `scan_NNNN.png` and re-running) caught a 65-percentage-point phantom gain that would otherwise have been claimed as novel SOTA. The audit itself is a transferable result.
3. **Clinical narrative remains useful.** The reasoning strings above are well-formed and cite morphological features — they can scaffold a clinical-report generator even though the VLM's label is unreliable. Combine with v4 predictions for the actual class; use the VLM only for narrative.
4. **Cost.** Full 240 scans at Haiku = roughly **$2.5** of compute. Parallel runtime ~7 min on 8 workers.
5. **Do not ensemble with v4.** Every blend weight > 0 hurts the ensemble. The v4 champion stands alone.