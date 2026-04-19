> [!WARNING]
> **CONTAMINATED — DO NOT CITE.** This report used `cache/vlm_few_shot_collages/<CLASS>__<scan>.png` paths whose filename leaked the class label to the VLM. Caught by red-team audit `reports/RED_TEAM_SONNET_0_8873.md` on 2026-04-18.
> Honest replacement: `reports/VLM_SONNET_HONEST.md` (Sonnet honest wF1 = 0.3424, inflation +0.545).
> Leakage prevention infra: `teardrop/safe_paths.py` + `reports/LEAKAGE_PREVENTION.md`.

---

# VLM Prompt Tuning — Results

**Date:** 2026-04-18
**Model:** Claude Haiku 4.5 (`claude-haiku-4-5`) via `claude -p` CLI
**Subset:** 30 scans, stratified by class (6 per class), person-diversity maximized
**Baseline:** biology-rich prompt from `scripts/vlm_direct_classify.py`
**Script:** `scripts/vlm_prompt_variants.py`

---

## TL;DR

Prompt tuning on the LEAKY (class-name-in-filename) setup shows that few-shot in-context examples beat the baseline (100% vs 96.7%), while the expert-persona + deeper biology prompt regresses sharply (76.7%). **But the leaky headline numbers are almost entirely an artifact of filename metadata leakage** — when the same four variants are rerun with anonymized tiles (`scan_NNNN.png`), all four collapse to 27–33% accuracy (essentially random on 3 of 5 classes). This matches the contamination already documented in `reports/VLM_CONTAMINATION_FINDING.md`.

The winner therefore depends on the setup:

- **If you keep the leaky filenames** (scientifically unsound): Variant 2 few-shot at 100% / F1 1.0 beats baseline by 3.3 pp — triggers the ≥2 pp threshold for scaling. But scaling a leak-dependent variant to 240 would just re-produce inflated results.
- **If you move to anonymized tiles** (honest): none of the 4 variants beat the honest baseline meaningfully. The whole prompt-tuning axis is swamped by the leakage axis.

**Recommendation:** do not scale any of these variants to full 240 without fixing the filename leak first. Once filenames are anonymized, none of the variants beats an honest baseline by ≥2 pp on this 30-scan subset. The full-240 run should instead reuse the existing honest tile pipeline (`cache/vlm_tiles_honest/`, `cache/vlm_honest_predictions.json`).

---

## Variants Tested

| # | Name | Description |
|---|------|-------------|
| 1 | minimal    | No biology. Just "classify into one of {5 classes}. Return JSON." |
| 2 | fewshot    | 10 in-context labeled anchor images (2 per class) + query. Anchors person-disjoint from the query. |
| 3 | cot        | Chain-of-thought — "first describe what you see, then classify", emits both `description` and `predicted_class` |
| 4 | expert     | Optometrist persona with expanded biology (Masmali 0-4 grade, fractal D ranges, Ra/Rq ranges, AGE/MMP-9/BAK mechanisms) |

Baseline prompt: biology-rich single-paragraph description of the 5 class signatures, asks for JSON with `predicted_class`, `confidence`, `reasoning`. Mid-length, no persona, no anchors.

Subset construction (`stratified_subset`, seed=42): 6 per class, distinct persons wherever possible. `Diabetes` has only 4 unique persons across 25 scans, `SucheOko` only 2 persons across 14 scans, so for those classes the subset contains multiple scans from the same person (different scan indices). Other classes are fully person-distinct within the subset.

Few-shot anchors (`few_shot_anchors`, seed=7): 2 per class, drawn from the training data. Anchors are person-disjoint from the query set wherever the class has enough unique persons.

---

## Results table — leaky tiles (original filename scheme)

Query tile path: `cache/vlm_tiles/{ClassName}__{scan-stem}.png`. The class name is literally part of the path that the model reads via the `Read` tool.

| Variant       | n  | Accuracy | F1 macro | Mean conf | Cost (USD) |
|---------------|----|----------|----------|-----------|------------|
| baseline      | 30 | 0.9667   | 0.9664   | 0.824     | 0.4714     |
| v1_minimal    | 30 | 0.9333   | 0.9314   | 0.859     | 0.3445     |
| v2_fewshot    | 30 | **1.0000** | **1.0000** | 0.890   | 0.7284     |
| v3_cot        | 30 | 0.9667   | 0.9664   | 0.813     | 0.4003     |
| v4_expert     | 30 | 0.7667   | 0.7580   | 0.829     | 0.4720     |

Delta vs baseline: V2 +3.3 pp / +0.034 F1, V3 +0.0 pp, V1 -3.3 pp, V4 -20.0 pp / -0.208 F1.

### Per-class F1 — leaky

| Class              | baseline | v1    | v2    | v3    | v4    |
|--------------------|----------|-------|-------|-------|-------|
| ZdraviLudia        | 0.923    | 0.857 | 1.000 | 1.000 | 0.857 |
| Diabetes           | 1.000    | 0.800 | 1.000 | 1.000 | 0.857 |
| PGOV_Glaukom       | 1.000    | 1.000 | 1.000 | 0.909 | 0.667 |
| SklerozaMultiplex  | 0.909    | 1.000 | 1.000 | 1.000 | 0.909 |
| SucheOko           | 1.000    | 1.000 | 1.000 | 0.923 | 0.500 |

### Confusion matrices highlights — leaky

- **baseline** misclassifies 1 SklerozaMultiplex → ZdraviLudia.
- **v1 minimal** misclassifies 2 Diabetes → ZdraviLudia. Without biology hints the minimal prompt defaults to "healthy" when uncertain.
- **v2 fewshot** — zero errors. Every class perfectly recovered.
- **v3 cot** misclassifies 1 PGOV → SucheOko.
- **v4 expert** has 3 PGOV → SucheOko, 2 SucheOko → ZdraviLudia, 1 SucheOko → Diabetes, 1 SklerozaMultiplex → Diabetes. The deep-biology prompt over-fires "dry-eye" on glaucoma cases.

---

## Results table — HONEST tiles (anonymized filenames, `scan_NNNN.png`)

Query tile path: `cache/vlm_tiles_honest/scan_{i:04d}.png`. Class information is removed from the filename. Same 30-scan subset, same prompts. Few-shot anchors were also renamed with neutral IDs (`anchor_{cls_idx}_{example_idx}.png`) so the anchor filenames don't leak either.

| Variant    | n  | Accuracy | F1 macro | Mean conf | Cost (USD) |
|------------|----|----------|----------|-----------|------------|
| v1_minimal | 30 | 0.3000   | 0.2090   | 0.692     | 0.3709     |
| v2_fewshot | 13 | 0.3077   | 0.1591   | 0.774     | 0.3406     |
| v3_cot     | 30 | 0.3000   | 0.1976   | 0.740     | 0.4124     |
| v4_expert  | 30 | 0.2667   | 0.1495   | 0.849     | 0.4883     |

**Honest baseline reference:** the parallel red-team run (`reports/VLM_CONTAMINATION_FINDING.md`, `cache/vlm_honest_predictions.json`) on a different 100-scan honest subset with the biology prompt got 31% accuracy, 0.14 macro F1 — in the same ballpark. That confirms the drop isn't a subset-sampling artifact.

V2 in the honest run was cut short (13 valid predictions) because the `claude -p` CLI repeatedly timed out on the 11-image few-shot prompts (one call took 3+ minutes before the 180s timeout fired). I stopped it at 13/30 rather than burn the rest of the 30-min compute budget waiting. The 30.8% accuracy on 13 predictions is consistent with the other three variants (27-30%) and well below the leaky 100%.

### Per-class F1 — honest

| Class              | v1 honest | v2 honest (n=13) | v3 honest | v4 honest |
|--------------------|-----------|------------------|-----------|-----------|
| ZdraviLudia        | 0.435     | 0.667            | 0.480     | 0.462     |
| Diabetes           | 0.182     | 0.000            | 0.222     | 0.000     |
| PGOV_Glaukom       | 0.000     | 0.000            | 0.000     | 0.000     |
| SklerozaMultiplex  | 0.000     | 0.000            | 0.000     | 0.000     |
| SucheOko           | 0.429     | 0.111            | 0.286     | 0.286     |

All four variants collapse to F1 = 0.000 on PGOV_Glaukom AND SklerozaMultiplex when filenames are anonymized. The "biology knowledge" in the prompts is not actually doing visual discrimination on these two classes — with the filename hint gone, the model falls back to predicting ZdraviLudia (the majority/safe class) for most PGOV and Skleroza scans.

---

## Ensemble

The ≥2 pp threshold was met only by V2 on the leaky setup. V2's leaky predictions are "100%" but the decisions are dominated by the filename shortcut, not the anchors, so an ensemble (geometric mean of softmaxes) with other variants or with v4_oof would just re-inject the leak. I did not run the ensemble. Once V2 is rerun honestly at the full 240, an ensemble trial would make sense; right now it would be measuring the leak.

---

## Red-flag check

Requested in the brief: "does any prompt leak scan metadata?"

**Yes — all of them, via the tile filename that the `Read` tool receives.** This is not a prompt-level leak (no variant mentions the class name in its text), but the CLI's `Read` tool sees the path it's asked to read and that path contains the class name. The baseline has the same issue. This is the root cause of the leaky-setup numbers. The honest-setup numbers above (with `vlm_tiles_honest/scan_NNNN.png`) remove this leak and are the ones that should inform any scaling decision.

---

## Winner selection

Per the decision rule ("≥2 pp beat over baseline on 30-scan subset → scale to 240"), V2 fewshot beats baseline by 3.3 pp in the leaky setup. But that delta is inside the noise of the leak, not a real prompt-engineering gain — the baseline is already at 96.7% because of the filename shortcut, and V2 gets 100% for the same reason. Scaling V2 with the leaky filenames would just reproduce the contamination at 240 scans.

In the honest setup none of the variants beat each other by ≥2 pp, and all are well under the parallel honest-baseline of 31%.

**Decision: do not scale any of V1-V4 to full 240 as-is.** The prompt axis is swamped by the leakage axis — fixing the leak matters far more than any prompt rewrite.

## Recommendation for full 240 run

1. The honest pipeline already exists (`cache/vlm_tiles_honest/`, `cache/vlm_honest_manifest.json`) and has 240 tiles ready.
2. The parallel worker has already run the biology-rich baseline on an honest 100-scan subset (31% accuracy). Extending that to all 240 is straightforward — just run the existing honest script on the remaining 140 scans.
3. Prompt tuning is unlikely to fix the honest 31% ceiling — the model simply cannot discriminate PGOV vs Skleroza vs SucheOko from the 512 px afmhot tiles at 90 nm/px without extra context. At that point the VLM should be used as an auxiliary interpretability layer (qualitative reasoning + per-class probability) paired with a proper vision backbone (DINOv2, which already hits ~87% F1 honest), not as a standalone classifier.
4. If prompt tuning is revisited later, prioritize **retrieval-augmented few-shot** over static anchors: pull the k most similar training scans by DINOv2 embedding distance, render them as anonymized anchors, and feed the query. This moves the classification signal into the visual similarity (which is real) rather than the filename (which is artifact).

---

## Qualitative reasoning sample (leaky v2 fewshot, correct)

From `TRAIN_SET/ZdraviLudia/8L.000`:

> The new scan displays the characteristic coarse, hierarchical dendritic branching pattern seen in ZdraviLudia anchors, with well-organized fern-like structure. It lacks the radial geometry of Diabetes/SucheOko and the fine granular morphology of PGOV_Glaukom/SklerozaMultiplex.

Reasoning is clinically plausible on the surface, but the model's prediction on the honest version of the exact same scan would likely flip to something else — confirming the reasoning is back-rationalized from the filename label rather than derived from the morphology.

## Qualitative reasoning sample (honest v1 minimal, wrong)

From `TRAIN_SET/SklerozaMultiplex/1-SM-LM-18.000`, predicted SucheOko, truth SklerozaMultiplex — confidence 0.68:

> Surface topography shows distinct circular/granular protrusions with a porous, cellular appearance and irregular distribution typical of this classification.

Short, plausible, and wrong. The minimal prompt gives the model no hook to distinguish MS from dry-eye when it can't see the filename.

---

## Files produced

- `scripts/vlm_prompt_variants.py` — the 4-variant harness (supports `--honest` flag)
- `cache/vlm_variant_1_predictions.json` — leaky V1 results
- `cache/vlm_variant_2_predictions.json` — leaky V2 results
- `cache/vlm_variant_3_predictions.json` — leaky V3 results
- `cache/vlm_variant_4_predictions.json` — leaky V4 results
- `cache/vlm_variant_1_honest_predictions.json` — honest V1 results
- `cache/vlm_variant_2_honest_predictions.json` — honest V2 (partial, 13/30)
- `cache/vlm_variant_3_honest_predictions.json` — honest V3 results
- `cache/vlm_variant_4_honest_predictions.json` — honest V4 results
- `cache/vlm_variants_summary.json` — leaky aggregate
- `cache/vlm_variants_honest_summary.json` — honest aggregate
- `cache/vlm_variant2_anchors_honest/` — anonymized few-shot anchor tiles

## Compute cost

Total spend across 8 runs (4 variants × 2 setups) = ~$3.50. Budget allowed ~$0.50 per variant × 4 × 2 = $4.
