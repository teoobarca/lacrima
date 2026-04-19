# VLM Few-Shot Classification Results

Generated: 2026-04-18T20:41:49

## Setup

- Retrieval: DINOv2-B (`cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`), cosine sim.
- k=2 anchors per class (10 total), nearest-neighbor, **person-disjoint**.
- Prompt: one composite image (5x2 labeled anchor grid + query tile below).
- Model: claude-haiku-4-5
- Samples scored: 5
- Total API cost (this run + cached): $0.084

## Accuracy

- Accuracy: **1.0000**
- Macro-F1: **1.0000**
- Mean confidence: 0.798
- Reasoning quality: VLM's `most_similar_anchor_class` agrees with `predicted_class` **100.0%** of the time.
- Sanity: true class in anchor set 100.0% (should be 100%).

## Per-class F1

| Class | P | R | F1 | Support |
|---|---|---|---|---|
| ZdraviLudia | 1.000 | 1.000 | 1.000 | 1 |
| Diabetes | 1.000 | 1.000 | 1.000 | 1 |
| PGOV_Glaukom | 1.000 | 1.000 | 1.000 | 1 |
| SklerozaMultiplex | 1.000 | 1.000 | 1.000 | 1 |
| SucheOko | 1.000 | 1.000 | 1.000 | 1 |

## Confusion Matrix

Rows=true, cols=pred.

| true\pred | ZdraviLudi | Diabetes | PGOV_Glauk | SklerozaMu | SucheOko |
|---|---|---|---|---|---|
| ZdraviLudi | 1 | 0 | 0 | 0 | 0 |
| Diabetes | 0 | 1 | 0 | 0 | 0 |
| PGOV_Glauk | 0 | 0 | 1 | 0 | 0 |
| SklerozaMu | 0 | 0 | 0 | 1 | 0 |
| SucheOko | 0 | 0 | 0 | 0 | 1 |

## Few-Shot vs Zero-Shot Head-to-Head

- Overlap: 5 scans scored by both.
- **Zero-shot accuracy**: 1.0000
- **Few-shot accuracy**:  1.0000
- **Delta**: +0.0000
- Agreement: 100.0%
- Both right: 5  |  Both wrong: 0
- Few-shot fixes (ZS-wrong -> FS-right): 0
- Few-shot regresses (ZS-right -> FS-wrong): 0

## Baselines to compare against

- Zero-shot VLM (prior run): ~0.88 accuracy on 30-scan subset
- v4 multiscale ensemble: 0.6887 macro-F1 (full 240)
