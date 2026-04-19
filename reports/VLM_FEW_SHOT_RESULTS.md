> [!WARNING]
> **CONTAMINATED — DO NOT CITE.** This report used `cache/vlm_few_shot_collages/<CLASS>__<scan>.png` paths whose filename leaked the class label to the VLM. Caught by red-team audit `reports/RED_TEAM_SONNET_0_8873.md` on 2026-04-18.
> Honest replacement: `reports/VLM_SONNET_HONEST.md` (Sonnet honest wF1 = 0.3424, inflation +0.545).
> Leakage prevention infra: `teardrop/safe_paths.py` + `reports/LEAKAGE_PREVENTION.md`.

---

# VLM Few-Shot Classification Results

Generated: 2026-04-18T21:26:20

## Setup

- Retrieval: DINOv2-B (`cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`), cosine sim.
- k=2 anchors per class (10 total), nearest-neighbor, **person-disjoint**.
- Prompt: one composite image (5x2 labeled anchor grid + query tile below).
- Model: claude-sonnet-4-6
- Samples scored: 60
- Total API cost (this run + cached): $2.038

## Accuracy

- Accuracy: **0.8500**
- Macro-F1: **0.8454**
- Mean confidence: 0.780
- Reasoning quality: VLM's `most_similar_anchor_class` agrees with `predicted_class` **100.0%** of the time.
- Sanity: true class in anchor set 100.0% (should be 100%).

## Per-class F1

| Class | P | R | F1 | Support |
|---|---|---|---|---|
| ZdraviLudia | 0.800 | 1.000 | 0.889 | 12 |
| Diabetes | 0.917 | 0.917 | 0.917 | 12 |
| PGOV_Glaukom | 0.706 | 1.000 | 0.828 | 12 |
| SklerozaMultiplex | 1.000 | 0.750 | 0.857 | 12 |
| SucheOko | 1.000 | 0.583 | 0.737 | 12 |

## Confusion Matrix

Rows=true, cols=pred.

| true\pred | ZdraviLudi | Diabetes | PGOV_Glauk | SklerozaMu | SucheOko |
|---|---|---|---|---|---|
| ZdraviLudi | 12 | 0 | 0 | 0 | 0 |
| Diabetes | 0 | 11 | 1 | 0 | 0 |
| PGOV_Glauk | 0 | 0 | 12 | 0 | 0 |
| SklerozaMu | 0 | 0 | 3 | 9 | 0 |
| SucheOko | 3 | 1 | 1 | 0 | 7 |

## All available zero-shot baselines (for context)

| Baseline | N overlap | Zero-shot acc | Few-shot acc | Delta | FS fixes | FS regresses |
|---|---|---|---|---|---|---|
| `honest` | 60 | 0.283 | 0.850 | +0.567 | 35 | 1 |
| `vlm_variant_1_predictions.json` | 28 | 0.929 | 0.786 | -0.143 | 2 | 6 |
| `vlm_variant_2_predictions.json` | 28 | 1.000 | 0.786 | -0.214 | 0 | 6 |
| `vlm_variant_3_predictions.json` | 28 | 0.964 | 0.786 | -0.179 | 1 | 6 |
| `vlm_variant_4_predictions.json` | 28 | 0.750 | 0.786 | +0.036 | 4 | 3 |
| `vlm_haiku_predictions_subset.json` | 19 | 0.947 | 0.895 | -0.053 | 1 | 2 |
| `vlm_sonnet_predictions_subset.json` | 19 | 0.947 | 0.895 | -0.053 | 0 | 1 |
| `vlm_opus_predictions_subset.json` | 19 | 0.789 | 0.895 | +0.105 | 3 | 1 |

## Few-Shot vs Zero-Shot Head-to-Head (largest overlap)

- Zero-shot source: `vlm_honest_predictions.json (via manifest)`
- Overlap: 60 scans scored by both.
- **Zero-shot accuracy**: 0.2833
- **Few-shot accuracy**:  0.8500
- **Delta**: +0.5667
- Agreement: 33.3%
- Both right: 16  |  Both wrong: 8
- Few-shot fixes (ZS-wrong -> FS-right): 35
- Few-shot regresses (ZS-right -> FS-wrong): 1

## Baselines to compare against

- Zero-shot VLM (honest, 40 overlap): 0.225 accuracy
- Zero-shot VLM (curated first-scan-per-person subsets): 0.77 - 1.00 on 30-scan overlaps
- v4 multiscale ensemble: 0.6887 macro-F1 (full 240)

## Reasoning quality (sample VLM explanations)

### Correct predictions (cite anchors explicitly)

- **true=ZdraviLudia** pred=ZdraviLudia (conf 0.92): The query shows a classic dense dendritic ferning pattern with a clear central nucleation point and uniform fine branching radiating outward — virtually identical to both ZdraviLudia anchors (#1 and #2), which display the same high-fractal-dimension fern-like crystalline network. The branching is fine, symmetric, and fully covers the field, unlike the coarser lattice of Diabetes, the granular/looped texture of Glaukom, the heterogeneous rods of SklerozaMultiplex, or the sparse amorphous regions of SucheOko.
- **true=ZdraviLudia** pred=ZdraviLudia (conf 0.91): The query shows dense, fine dendritic ferning radiating from a bright central origin with uniform branching — closely matching both ZdraviLudia anchors (#1 and #2), which display the same high-fractal radial fern pattern with thin, well-defined branches. Diabetes anchors show coarser packing, Glaukom shows loop/rod textures, and SucheOko shows fragmentation — none of which appear in the query.
- **true=ZdraviLudia** pred=ZdraviLudia (conf 0.91): The query tile shows dense, fine dendritic ferning radiating from a bright central nucleation point with uniform branching structure, closely matching both ZdraviLudia anchors (#1 and #2) which display the same high fractal dimension fine-branch network. The Diabetes anchors show coarser/thicker crystals and the SucheOko anchors show sparser networks, neither of which matches the query.

### Wrong predictions (failure modes)

- **true=Diabetes** pred=PGOV_Glaukom (conf 0.72): The query shows thick, stubby, chaotically arranged crystal branches with loop-like structures and irregular spacing, closely matching the PGOV_Glaukom anchors which display similar coarse, shorter-thicker branches with granular chaotic texture and partial ring/loop formations. The crystal arms are bulkier and less uniformly branched than the Diabetes anchors.
- **true=SklerozaMultiplex** pred=PGOV_Glaukom (conf 0.72): The query shows a granular, chaotic texture with scattered bright spots and no clear dendritic branching, closely resembling the PGOV_Glaukom anchors which display similar granular/looping disorganized morphology with short fragmented structures rather than the elongated fern-like crystals seen in ZdraviLudia or Diabetes.
- **true=SklerozaMultiplex** pred=PGOV_Glaukom (conf 0.72): The query shows prominent horizontal banding/striping with a granular, chaotic texture and no coherent dendritic branching, closely matching the PGOV_Glaukom anchors which display similar horizontal stripe artifacts, degraded short structures, and disorganized granular patterns consistent with MMP-9 degradation.
- **true=SklerozaMultiplex** pred=PGOV_Glaukom (conf 0.72): The query shows a dense granular texture with small rounded nodules and loop-like structures distributed uniformly across the field, closely matching both PGOV_Glaukom anchors which display the same chaotic granular/ring morphology consistent with MMP-9 degradation and shortened thick branches. The SklerozaMultiplex anchors show a more regular lattice pattern with longer rod structures, which the query lacks.
