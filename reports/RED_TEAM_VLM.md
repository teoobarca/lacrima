# RED TEAM AUDIT: VLM Direct-Classify (Claude Haiku 4.5)

**Date:** 2026-04-18
**Claim under audit:** VLM direct-image classification with domain-knowledge prompt scores ~88% accuracy / F1_macro ~0.87 on 240 AFM tear-droplet scans, beating v4 ensemble (F1 = 0.6887) by ~18 F1 points.
**Auditor:** skeptical red team (tone: loud flagging).

## TL;DR — verdict: **INFLATED / UNTRUSTWORTHY — do NOT ship as v5**

The 88% accuracy is almost entirely explained by **class-name leakage through the image filename**. The prompt template interpolates the image path directly into the text:

```
The image at cache/vlm_tiles/Diabetes__37_DM.png shows surface topography ...
```

Every tile name starts with the ground-truth class (`Diabetes__`, `PGOV_Glaukom__`, `SklerozaMultiplex__`, `SucheOko__`, `ZdraviLudia__`). The VLM trivially reads the answer from the filename it is asked to classify.

**Honest F1_macro when filenames are hashed (SHA-1, class-stripped):**
- 20-scan re-audit (4 per class): **F1_macro = 0.160, accuracy = 21%** (down from 0.89 / 90% with leaked names on the same scans)
- Pre-existing `vlm_honest_predictions.json` (n=25, class-stripped): **F1_macro = 0.14, accuracy = 28%**

Both independent blind re-runs converge on **F1 ~ 0.15** — i.e. worse than a naïve majority-class classifier (Sklo-prevalence baseline ~0.40 weighted acc). The VLM without the filename leak is **clearly worse than v4**.

---

## The 7-point checklist

### 1. Class-name / shortcut leak — **FAIL (critical)**

`scripts/vlm_direct_classify.py`, line 221:
```python
img_path = TILE_DIR / f"{s.cls}__{s.raw_path.stem}.png"
```

And line 117:
```python
prompt = PROMPT_TEMPLATE.format(img_path=str(img_path))
```

And line 66 of the prompt:
```
The image at {img_path} shows surface topography rendered with the afmhot colormap ...
```

→ The filename is literally embedded in the prompt text. Any Claude Haiku 4.5 call that parses the prompt sees `cache/vlm_tiles/Diabetes__37_DM.png` — the class is in cleartext. This is textbook shortcut leakage.

Evidence: full `ls cache/vlm_tiles/` shows every single tile prefixed with its class:
- `Diabetes__37_DM.png`, `PGOV_Glaukom__21_LV_PGOV+SII.png`, `SklerozaMultiplex__1-SM-LM-18.png`, `SucheOko__29_PM_suche_oko.png`, `ZdraviLudia__6L.png`, …

This alone invalidates the result.

### 2. Prompt-cheating check — **PASS** (the biology is OK; the filename leak is the problem)

The class descriptions (lines 67–72 of `vlm_direct_classify.py`) cite standard tear-ferning morphology: Masmali grades, fractal-D ranges, MMP-9 biology, Rolando/Masmali literature. Nothing scan-specific, no numeric pixel cues, no coordinates. This portion of the prompt is acceptable and not cheating. If the filename leak were fixed, the prompt itself is a reasonable zero-shot specification.

### 3. Prompt-to-answer alignment — **PARTIAL PASS**

Sampling the qualitative reasoning (from `vlm_honest_predictions.json` where the model was forced to classify without class-leaked names), the model overwhelmingly parks on "ZdraviLudia" (7 of 7 healthy → correct but also 2/2 Diabetes, 0/3 Glaukom, 0/13 SM predicted Zdravi). The descriptions it generates ("dense hierarchical dendritic ferning") are boilerplate — they cite the prompt template rather than actual image features. This is a mode-collapse failure mode, not a prompt-language mismatch.

### 4. Evaluation integrity — **PASS (math is correct, claim is inflated)**

Re-computed F1 from scratch using `sklearn.metrics.f1_score` on the file that was present at start of audit (n=199 scored):

- accuracy = 0.889, F1_macro = **0.877**, F1_weighted = 0.894
- Bootstrap 95% CI on F1_macro (2000 resamples): **[0.821, 0.924]**
- Coverage was only 199/240 (83%) due to 35-min time budget — pessimistic F1 (treat 41 unscored as wrong) = 0.736, optimistic = 0.900

`true_class` field in cache was cross-checked against `enumerate_samples()`: **zero label mismatches** in the 199 scored. So the reported F1 correctly reflects the cache. The arithmetic is not the problem — the cache itself is polluted by filename-leakage.

### 5. Reproducibility / variance — **PASS (mostly deterministic), but moot**

3-trial repeat on a blind tile: 2/2 identical before CLI raised an unrelated parse error on trial 3. Claude Haiku on temperature 0 is largely deterministic for this task. Variance is not the issue; the leakage is.

### 6. Fair comparison to v4 — **FAIL (apples vs. cheating-apples)**

- v4 is evaluated with person-LOPO (patient-disjoint folds), F1 = 0.6887. Honest.
- VLM was evaluated with filenames that announce the label. Every VLM prediction is trivially inflated by direct shortcut.
- In the honest blind audit, VLM F1 ≈ 0.15 ≪ v4 F1 = 0.69. **v4 beats the VLM by ~55 F1 points on a fair comparison**.

So: not only is the comparison unfair, it is inverted — the VLM without the leak loses badly to the trained classical ensemble.

### 7. Embedding-space check — **SKIPPED (not needed to reject claim)**

Given the overwhelming evidence from audits 1 and 6, a DINOv2 embedding t-SNE would be scientific cosplay. The question "does the VLM use visual features or priors?" has already been answered: it uses **filename priors**. When the filename is scrambled, the VLM falls back to a near-majority guess and fails.

---

## Re-run evidence (≤15 min of compute spent)

### Blind 20-scan re-audit (`cache/red_team_blind/results.json`)

Same tile pixels, but renamed to `scan_<sha1-hash>.png` and placed in `cache/red_team_blind_tiles/`. Same prompt template (including filename-mention), same model (`claude-haiku-4-5`), same SYSTEM append, same `--output-format json`.

| Condition                             | n  | accuracy | F1_macro |
| ------------------------------------- | -- | -------- | -------- |
| Original (class-leaked filenames)     | 29 | 0.897    | 0.892    |
| **Blind (hashed filenames)**          | 29 | **0.207**| **0.160**|
| Drop                                  |    | −0.690   | −0.732   |

### Prediction distribution under blind renaming

- Blind predictions: `{ZdraviLudia: 18, SucheOko: 6, Diabetes: 4, SklerozaMultiplex: 1, PGOV_Glaukom: 0}`
- True distribution (balanced-ish): `{ZdraviLudia: 6, Diabetes: 6, PGOV_Glaukom: 6, SklerozaMultiplex: 6, SucheOko: 5}`

The VLM collapses to "ZdraviLudia" (18/29 = 62% of predictions) — that is its prior for "dense dendritic ferning" when it has no filename to crib from.

### Independent corroboration (`cache/vlm_honest_predictions.json`, n=25)

Someone else in the project has already re-run a 25-scan subset with class-stripped filenames: **F1_macro = 0.14, accuracy = 28%**. Same collapse: 7/7 ZdraviLudia correct, 0/13 SM correct, 0/3 Glaukom, 0/2 Diabetes, 0/0 SucheOko. Two independent blind experiments now put honest VLM F1 in the **0.14–0.16** range.

---

## Honest performance estimate

If we take the geometric mean of the two blind experiments (weighted by support):

- 20-scan blind: F1 = 0.160 (support 29)
- 25-scan blind: F1 = 0.140 (support 25)
- Combined estimate: **F1_macro ≈ 0.15 ± 0.05**

This is **below chance on a balanced 5-class problem** (1/5 = 0.20 accuracy, ~0.20 F1 under balanced support — the VLM's 0.14–0.21 is no better than random, just biased toward a single class).

**Honest F1_macro for Claude Haiku 4.5 + prompt + AFM image ≈ 0.15.**
**Inflation due to filename leak ≈ +0.73 F1.**

---

## Recommendation

1. **Do NOT ship VLM as v5.** The 0.88 number is a filename-reading artifact.
2. **Blacklist the "VLM-direct-classify" approach in its current form** — at zero-shot, the VLM cannot match v4 (or even a majority-class baseline) on AFM tear scans.
3. **Do not publish any claim based on `vlm_summary.json` / `vlm_predictions.json`** without a footnote disclosing the leak. The `vlm_summary.json` F1 numbers are artifacts, not evidence.
4. **If the VLM path is kept at all**, it must:
   - strip class from filenames (`scan_<hash>.png`) AND
   - remove the `{img_path}` interpolation from the prompt (pass image via `Read` tool only, do not name it in the prompt) AND
   - re-run the full 240 scans before any metric is reported.
5. **A better retention of the idea**: use the VLM as a **reasoning/narrative generator** given an already-classified scan (feeding it the v4 probability vector + cropped tile), not as a primary classifier. The natural-language clinical narrative is still valuable even though the VLM's labelling is not.

## Files & evidence

- Leaky script: `/Users/rafael/Programming/teardrop-challenge/scripts/vlm_direct_classify.py` (lines 64–76 prompt, line 221 tile name)
- Leaked tiles: `/Users/rafael/Programming/teardrop-challenge/cache/vlm_tiles/` (every tile name begins with class)
- Original (leaked) predictions: `/Users/rafael/Programming/teardrop-challenge/cache/vlm_summary.json`
- Blind re-audit tiles: `/Users/rafael/Programming/teardrop-challenge/cache/red_team_blind_tiles/`
- Blind re-audit results: `/Users/rafael/Programming/teardrop-challenge/cache/red_team_blind/results.json`
- Pre-existing blind audit: `/Users/rafael/Programming/teardrop-challenge/cache/vlm_honest_predictions.json`
- v4 reference: `/Users/rafael/Programming/teardrop-challenge/cache/v4_oof.npz` (F1 = 0.6887 person-LOPO)
