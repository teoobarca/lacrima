# VLM Binary Re-ranker on v4 uncertain predictions

Generated: 2026-04-18T23:24:35

## TL;DR

- Margin threshold: **0.30**  (abstain set = 20/240 scans, confident = 220)
- v4 accuracy on abstain: **0.550**
- VLM accuracy on abstain: **0.350**  (20 evaluated, 0 errored)
- Overall wF1: v4-only = 0.6887  →  v4+VLM-rerank = **0.6675**  (Δ = -0.0211)
- Overall mF1: v4-only = 0.5541  →  v4+VLM-rerank = **0.5380**
- Paired bootstrap P(Δ > 0) = **0.000**  (95% CI [-0.0380, -0.0072])
- Cost: $0.6523  (20 calls, model=claude-sonnet-4-6)
- **Verdict:** **RETRACT — VLM hurts on abstain.** VLM acc=0.350 < v4 acc=0.550 on the same subset.

## Protocol

1. Load v4 multiscale OOF softmax (240 scans × 5 classes) from `cache/v4_oof.npz`.
2. Abstain set: scans where top-1 prob − top-2 prob < **0.30** (auto-raised from 0.20 to reach target_min=20).
3. For each abstain scan, retrieve 3 nearest DINOv2-B anchors for top-1 class and 3 nearest for top-2 class (cosine sim; person-disjoint from query).
4. Compose an obfuscated collage: 2 columns (headers only "Class A" / "Class B") × 3 rows + query at bottom with red border. **No actual class names appear anywhere in the collage or the filename.**
5. Save via `teardrop.safe_paths.safe_tile_path(idx, subdir='binary_reranker')`. Call `assert_prompt_safe(prompt, extra_forbidden=CLASSES)` immediately before every `claude -p`.
6. VLM returns `{choice: 'A' | 'B', confidence, reasoning}`. Internal map: A = top-1 class, B = top-2 class.
7. Final fused prediction = VLM's choice on abstain scans, v4 argmax on confident scans.

## Abstain subset analysis

| Scan idx | True | v4 top-1 | v4 top-2 | margin | VLM choice | VLM pred | VLM correct? | v4 correct? |
|---|---|---|---|---|---|---|---|---|
| 0 | Diabetes | SucheOko | SklerozaMultiplex | 0.123 | B | SklerozaMultiplex | ✗ | ✗ |
| 7 | Diabetes | SucheOko | ZdraviLudia | 0.082 | B | ZdraviLudia | ✗ | ✗ |
| 9 | Diabetes | ZdraviLudia | Diabetes | 0.135 | A | ZdraviLudia | ✗ | ✗ |
| 12 | Diabetes | Diabetes | ZdraviLudia | 0.285 | A | Diabetes | ✓ | ✓ |
| 19 | Diabetes | Diabetes | ZdraviLudia | 0.243 | A | Diabetes | ✓ | ✓ |
| 36 | PGOV_Glaukom | PGOV_Glaukom | SklerozaMultiplex | 0.081 | B | SklerozaMultiplex | ✗ | ✓ |
| 63 | SklerozaMultiplex | SklerozaMultiplex | SucheOko | 0.217 | A | SklerozaMultiplex | ✓ | ✓ |
| 69 | SklerozaMultiplex | SklerozaMultiplex | ZdraviLudia | 0.287 | B | ZdraviLudia | ✗ | ✓ |
| 79 | SklerozaMultiplex | SklerozaMultiplex | SucheOko | 0.056 | A | SklerozaMultiplex | ✓ | ✓ |
| 81 | SklerozaMultiplex | SklerozaMultiplex | SucheOko | 0.292 | B | SucheOko | ✗ | ✓ |
| 95 | SklerozaMultiplex | SklerozaMultiplex | SucheOko | 0.031 | A | SklerozaMultiplex | ✓ | ✓ |
| 109 | SklerozaMultiplex | SklerozaMultiplex | Diabetes | 0.213 | B | Diabetes | ✗ | ✓ |
| 126 | SklerozaMultiplex | SklerozaMultiplex | SucheOko | 0.108 | A | SklerozaMultiplex | ✓ | ✓ |
| 131 | SklerozaMultiplex | SucheOko | ZdraviLudia | 0.299 | B | ZdraviLudia | ✗ | ✗ |
| 149 | SklerozaMultiplex | PGOV_Glaukom | SklerozaMultiplex | 0.238 | A | PGOV_Glaukom | ✗ | ✗ |
| 158 | SucheOko | SklerozaMultiplex | ZdraviLudia | 0.226 | B | ZdraviLudia | ✗ | ✗ |
| 162 | SucheOko | SklerozaMultiplex | ZdraviLudia | 0.115 | A | SklerozaMultiplex | ✗ | ✗ |
| 163 | SucheOko | ZdraviLudia | SklerozaMultiplex | 0.255 | A | ZdraviLudia | ✗ | ✗ |
| 216 | ZdraviLudia | ZdraviLudia | SucheOko | 0.192 | A | ZdraviLudia | ✓ | ✓ |
| 218 | ZdraviLudia | Diabetes | ZdraviLudia | 0.261 | A | Diabetes | ✗ | ✗ |

## Per-class F1 (v4 → fused)

| Class | v4 F1 | fused F1 | Δ | Support |
|---|---|---|---|---|
| ZdraviLudia | 0.917 | 0.892 | -0.025 | 70 |
| Diabetes | 0.583 | 0.571 | -0.012 | 25 |
| PGOV_Glaukom | 0.579 | 0.560 | -0.019 | 36 |
| SklerozaMultiplex | 0.691 | 0.667 | -0.025 | 95 |
| SucheOko | 0.000 | 0.000 | +0.000 | 14 |

## Confusion matrices

### v4-only

| true\pred | ZdraviLudi | Diabetes | PGOV_Glauk | SklerozaMu | SucheOko |
|---|---|---|---|---|---|
| ZdraviLudi | 66 | 4 | 0 | 0 | 0 |
| Diabetes | 7 | 14 | 0 | 2 | 2 |
| PGOV_Glauk | 0 | 0 | 22 | 14 | 0 |
| SklerozaMu | 0 | 4 | 18 | 65 | 8 |
| SucheOko | 1 | 1 | 0 | 12 | 0 |

### v4 + VLM-rerank

| true\pred | ZdraviLudi | Diabetes | PGOV_Glauk | SklerozaMu | SucheOko |
|---|---|---|---|---|---|
| ZdraviLudi | 66 | 4 | 0 | 0 | 0 |
| Diabetes | 8 | 14 | 0 | 3 | 0 |
| PGOV_Glauk | 0 | 0 | 21 | 15 | 0 |
| SklerozaMu | 2 | 5 | 18 | 62 | 8 |
| SucheOko | 2 | 1 | 0 | 11 | 0 |

## Paired bootstrap 1000x (fused − v4)

- Fused wF1 bootstrap mean: 0.6681  (95% CI [0.6065, 0.7305])
- v4    wF1 bootstrap mean: 0.6890  (95% CI [0.6308, 0.7525])
- Delta (fused − v4) mean: **-0.0208**  (95% CI [-0.0380, -0.0072])
- **P(Δ > 0) = 0.000**

## Safety verification

- Collage filename pattern: `cache/vlm_safe/binary_reranker/scan_XXXX.png` (no class, no person, no raw name).
- Column headers rendered in the collage are strictly `"Class A"` and `"Class B"` — actual class names never appear in pixels.
- `assert_prompt_safe(prompt, extra_forbidden=CLASSES)` called before every `claude -p`. This catches both path-context leaks AND any literal class-name string.
- Manifest (mapping `scan_XXXX` → true_class + top1/top2 + person) persisted at `cache/vlm_safe/binary_reranker/manifest.json` — never passed to the VLM.

## Decision

- **VLM DOES NOT HELP even on the narrow binary task.** On the abstain subset — where v4 is already unsure — the VLM picks the *wrong* class between top-1 and top-2 more often than v4 does.
- This closes the 'maybe VLM still helps on uncertain cases' hypothesis. Full-5-way VLM is dead (F1 0.34) AND binary-VLM is dead on the narrow subset.
- Recommendation: **do not ensemble VLM into the final pipeline**. v4 multiscale remains the champion.

## Reproducibility

- Script: `scripts/vlm_binary_reranker.py`
- Predictions JSON: `cache/vlm_binary_reranker_predictions.json`
- Raw VLM cache: `cache/vlm_binary_reranker_predictions.json` (keyed by obf_key)
- Collages: `cache/vlm_safe/binary_reranker/scan_XXXX.png`
- Manifest: `cache/vlm_safe/binary_reranker/manifest.json`
- Model: `claude-sonnet-4-6`   Threshold used: 0.30