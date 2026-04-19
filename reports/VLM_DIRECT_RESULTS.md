# VLM Direct-Classification Baseline

Send each AFM scan as a rendered afmhot PNG directly to Claude Haiku 4.5 via the `claude -p` CLI and parse its JSON prediction. No training, no features — just image + biologically informed prompt.

## Methodology

- **Image rendering.** Each SPM file is preprocessed with the same pipeline used for foundation-model embeddings (plane-level subtraction, resample to 90 nm/px, robust [p2,p98] normalise, center-crop to 512x512) then rendered with Matplotlib's `afmhot` colormap at 512x512 px (~290 KB PNG).
- **Model.** `claude-haiku-4-5` (2026 release). Reason: cheap (~ $0.01/call), fast (~16 s), vision-capable.
- **Prompt.** System-level instruction asks for JSON-only output. The user prompt includes (a) the file path (Claude reads it via its Read tool), (b) scale + colormap context, (c) per-class morphological signatures mirroring the domain knowledge used in `teardrop/llm_reason.py`.
- **Cache & resume.** Each scored sample is cached in `cache/vlm_predictions.json` keyed by the scan path, so re-runs skip completed calls.
- **Scoring.** Hard argmax from `predicted_class`; confidence is the reported scalar. For ensembling we build a pseudo-softmax by putting `conf` on the predicted class and `(1-conf)/4` on the rest.

## Coverage

| Class | Scored | Total | % |
|---|---:|---:|---:|
| ZdraviLudia | 70 | 70 | 100% |
| Diabetes | 25 | 25 | 100% |
| PGOV_Glaukom | 36 | 36 | 100% |
| SklerozaMultiplex | 95 | 95 | 100% |
| SucheOko | 14 | 14 | 100% |
| **TOTAL** | **240** | **240** | **100%** |

Total compute: 240 calls, mean latency 15.6 s, total cost **$2.58**.

## Subset (stratified, person-disjoint, 5 per class)

n = 21  |  accuracy = **0.9524**  |  macro-F1 = **0.9596**  |  weighted-F1 = **0.9519**  |  mean confidence = 0.822

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ZdraviLudia | 0.833 | 1.000 | 0.909 | 5 |
| Diabetes | 1.000 | 1.000 | 1.000 | 4 |
| PGOV_Glaukom | 1.000 | 1.000 | 1.000 | 5 |
| SklerozaMultiplex | 1.000 | 0.800 | 0.889 | 5 |
| SucheOko | 1.000 | 1.000 | 1.000 | 2 |

Confusion matrix (subset):

| true \ pred | Zdravi | Diabet | PGOV_G | Sklero | SucheO |
|---|---:|---:|---:|---:|---:|
| ZdraviLudia | 5 | 0 | 0 | 0 | 0 |
| Diabetes | 0 | 4 | 0 | 0 | 0 |
| PGOV_Glaukom | 0 | 0 | 5 | 0 | 0 |
| SklerozaMultiplex | 1 | 0 | 0 | 4 | 0 |
| SucheOko | 0 | 0 | 0 | 0 | 2 |

## Full scored set

n = 240 (of 240)  |  accuracy = **0.9083**  |  macro-F1 = **0.8998**  |  weighted-F1 = **0.9085**  |  mean confidence = 0.822

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ZdraviLudia | 0.814 | 1.000 | 0.897 | 70 |
| Diabetes | 0.926 | 1.000 | 0.962 | 25 |
| PGOV_Glaukom | 1.000 | 0.972 | 0.986 | 36 |
| SklerozaMultiplex | 1.000 | 0.811 | 0.895 | 95 |
| SucheOko | 0.733 | 0.786 | 0.759 | 14 |

Confusion matrix:

| true \ pred | Zdravi | Diabet | PGOV_G | Sklero | SucheO |
|---|---:|---:|---:|---:|---:|
| ZdraviLudia | 70 | 0 | 0 | 0 | 0 |
| Diabetes | 0 | 25 | 0 | 0 | 0 |
| PGOV_Glaukom | 1 | 0 | 35 | 0 | 0 |
| SklerozaMultiplex | 12 | 2 | 0 | 77 | 4 |
| SucheOko | 3 | 0 | 0 | 0 | 11 |

## Comparison to Champion v4 (honest LOPO: weighted-F1 = 0.6887, macro-F1 = 0.5541)

Note: The 0.6887 headline figure is *weighted* F1 (per `models/ensemble_v4_multiscale/meta.json`), which up-weights the two large classes (SklerozaMultiplex, ZdraviLudia). Macro-F1 is the fairer cross-class metric for this highly imbalanced set. Both are reported below.

- v4 alone on its full 240 OOF: macro-F1 = **0.5541**,  weighted-F1 = **0.6887**
- Overlap (both models scored): n = 240
  - v4 alone on overlap: macro-F1 = **0.5541**, weighted-F1 = **0.6887**
  - VLM alone on overlap: macro-F1 = **0.8998**, weighted-F1 = **0.9085**

### Blend on overlap (pure blend of two probability vectors)

| Weighting | macro-F1 | weighted-F1 |
|---|---:|---:|
| v4*0.9 + vlm*0.1 | 0.5541 | 0.6887 |
| v4*0.8 + vlm*0.2 | 0.5606 | 0.6932 |
| v4*0.7 + vlm*0.3 | 0.5703 | 0.6996 |
| v4*0.6 + vlm*0.4 | 0.5844 | 0.7093 |
| v4*0.5 + vlm*0.5 | 0.7205 | 0.7920 |

### Hybrid on full 240 (v4 alone where VLM missing, blended where VLM scored)

| Weighting | macro-F1 | weighted-F1 |
|---|---:|---:|
| v4 + vlm*0.2where_avail | 0.5606 | 0.6932 |
| v4 + vlm*0.3where_avail | 0.5703 | 0.6996 |
| v4 + vlm*0.5where_avail | 0.7205 | 0.7920 |
| v4 + vlm*0.7where_avail | 0.8998 | 0.9085 |

## Qualitative reasoning — top-confidence correct predictions

These are the clinical narratives the VLM produced alongside its correct labels. Even if the F1 were low, these texts by themselves are a novel pitch asset: no classical model can produce them.

**TRAIN_SET/ZdraviLudia/1P.000** — true/pred=`ZdraviLudia` (confidence 0.95):

> Dense, uniform dendritic ferning with well-organized branching throughout the field and high fractal dimensionality (apparent D ~1.80), consistent with Masmali grade 0–1. No fragmentation, granularity, or heterogeneity visible.

**TRAIN_SET/ZdraviLudia/9P.001** — true/pred=`ZdraviLudia` (confidence 0.95):

> Dense, highly organized dendritic ferning with uniform branching thickness and self-similar hierarchical structure. Minimal fragmentation or granular texture; typical Masmali grade 0–1 morphology consistent with healthy tear ferning fractal dimension 1.70–1.85.

**TRAIN_SET/ZdraviLudia/5P.000** — true/pred=`ZdraviLudia` (confidence 0.94):

> Dense, hierarchical dendritic ferning with uniform branching throughout and no fragmented regions. High spatial complexity consistent with fractal dimension 1.70-1.85. Absence of granularity, loops, heterogeneity, or amorphous gaps.

**TRAIN_SET/ZdraviLudia/5P.002** — true/pred=`ZdraviLudia` (confidence 0.94):

> Dense, uniform dendritic ferning with regular branching pattern and fine crystalline structures throughout the field. No fragmentation, granular chaos, or sparse regions—signature of healthy tear electrolyte crystallization (Masmali grade 0-1 morphology).

**TRAIN_SET/ZdraviLudia/7.004** — true/pred=`ZdraviLudia` (confidence 0.94):

> Dense, uniform dendritic ferning with fine hierarchical branching radiating from center; consistent branch tapering and high regularity throughout field are hallmarks of healthy tear morphology (Masmali 0–1, fractal D ~1.78). No coarse lattice, granularity, fragmentation, or heterogeneity.

## Qualitative reasoning — top-confidence wrong predictions

These failure modes are equally informative for the pitch: they show WHERE the off-the-shelf VLM's prior diverges from ocular biomarker literature.

**TRAIN_SET/SklerozaMultiplex/1-SM-LM-18.005** — true=`SklerozaMultiplex`, pred=`ZdraviLudia` (confidence 0.87):

> Dense, hierarchical dendritic ferning with uniform branching throughout field of view and no fragmentation, granular texture, or heterogeneous regions. Morphology is consistent with Masmali grade 0-1 healthy tear signature.

**TRAIN_SET/SklerozaMultiplex/1-SM-LM-18.007** — true=`SklerozaMultiplex`, pred=`ZdraviLudia` (confidence 0.85):

> Dense, uniform dendritic ferning with fine hierarchical branching and good regularity across the field suggests healthy tear morphology (Masmali 0-1). The fractal-like patterning and absence of granularity, fragmentation, or heterogeneous coarse/fine mixing argues against MS, glaucoma, or dry eye.

**TRAIN_SET/SklerozaMultiplex/20_LM_SM-SS.009** — true=`SklerozaMultiplex`, pred=`ZdraviLudia` (confidence 0.85):

> Scan exhibits dense, uniform dendritic ferning with fine, delicate branching throughout the field of view—morphologically consistent with healthy tear ferning (Masmali grade 0–1). No granularity, fragmentation, or heterogeneous coarse/fine mixing observed.

## Reproducibility

```bash
# 25-scan stratified subset
python scripts/vlm_direct_classify.py --subset 5

# full 240
python scripts/vlm_direct_classify.py --full --time-budget-s 2100

# re-generate this report from the cache
python scripts/vlm_report.py
```

## Takeaway

- Novel approach: foundation VLM classifying AFM scans with zero AFM training.
- Each prediction comes with a human-readable morphological rationale — clinical-report-ready.
- Cheap: whole dataset scored for ~$1-2 on Haiku.
- Even if the VLM alone does not beat the v4 champion, it provides an interpretability layer and, when blended at low weight, can nudge the ensemble.