# Zero-Shot Claude Opus 4.7 on AFM Tear Scans

**Date:** 2026-04-18
**Script:** `scripts/vlm_zero_shot_opus.py`
**Predictions:** `cache/vlm_zero_shot_opus_predictions.json`
**Manifest:** `cache/vlm_zero_shot_opus_manifest.json`
**Summary:** `cache/vlm_zero_shot_opus_summary.json`

---

## TL;DR

**Zero-shot Opus 4.7 is still near random.** Anchors / few-shot retrieval
remain essential regardless of model strength.

| Setup | wF1 | Macro F1 | Accuracy |
|---|---|---|---|
| Random baseline (5-class, majority) | ~0.20 | ~0.20 | ~0.20 |
| **Zero-shot Haiku 4.5 (honest)** | 0.226 | ~0.14 | 0.280 |
| **Zero-shot Opus 4.7 (honest)** *(this run)* | **0.2773** | **0.2570** | **0.3478** |
| Few-shot Haiku 4.5 (with anchors) | **0.7351** | — | — |

**Verdict:** Opus zero-shot scores in the **< 0.3 F1 bucket** per the predefined
decision rule → confirms anchors are essential. A 10× stronger model buys ~5
percentage points of accuracy over Haiku zero-shot, but cannot close the 0.45
F1 gap that retrieval anchors provide.

---

## Design

- **Subset:** `stratified_person_disjoint_subset(per_class=6, seed=42)` → 23
  scans (capped by the 2 unique SucheOko persons and 4 Diabetes persons; each
  scan comes from a distinct person within its class).
- **Obfuscation:** Tiles rendered to
  `cache/vlm_zero_shot_opus_tiles/scan_XXXX.png`. Manifest kept separately; the
  CLI prompt references only the obfuscated tile path so the model cannot read
  the class from the filename.
- **Prompt:** Identical morphology signatures as `vlm_honest_parallel.py`
  (ZdraviLudia = dense dendritic ferning, Diabetes = coarser branches,
  PGOV_Glaukom = granular loops, SklerozaMultiplex = heterogeneous,
  SucheOko = fragmented). No in-context examples, no retrieval anchors.
- **Model:** `claude-opus-4-7`.
- **Runtime:** 4 parallel CLI workers, 23 calls, 68 s wall-clock,
  $1.7434 total, 11.1 s mean latency per call.

## Per-class breakdown

```
                   precision    recall  f1-score   support
      ZdraviLudia       0.45      0.83      0.59         6
         Diabetes       0.00      0.00      0.00         4
     PGOV_Glaukom       0.00      0.00      0.00         5
SklerozaMultiplex       0.40      0.33      0.36         6
         SucheOko       0.25      0.50      0.33         2
         accuracy                           0.35        23
        macro avg       0.22      0.33      0.26        23
     weighted avg       0.24      0.35      0.28        23
```

- **ZdraviLudia default bias:** 83 % recall on Healthy — same failure mode seen
  with zero-shot Haiku. When uncertain, Opus defaults to "dense dendritic
  ferning = healthy".
- **Diabetes / Glaukom = 0 F1:** the coarse-branch vs. granular-loop
  distinction from text alone is insufficient; the model cannot ground the
  morphology terms visually without examples.
- **SM F1 = 0.36:** only two correct out of six; most confused with healthy /
  dry-eye / glaucoma — heterogeneity signature is too broad textually.
- **SucheOko F1 = 0.33:** tiny support (2), but at least both scans got some
  pull toward "fragmented/sparse".
- **Mean confidence 0.665** — Opus is *confidently wrong* on the negative
  classes, which matches the earlier contamination-era finding that LLMs
  generate plausible reasoning post-hoc.

## Comparison to prior runs

| Run | Model | Setup | N | wF1 | Source |
|---|---|---|---|---|---|
| Zero-shot Haiku (honest) | claude-haiku-4-5 | text only | 25 | 0.226 | `reports/VLM_CONTAMINATION_FINDING.md` |
| **Zero-shot Opus (this)** | **claude-opus-4-7** | **text only** | **23** | **0.277** | **this run** |
| Few-shot Haiku (k=3 anchors) | claude-haiku-4-5 | anchors + text | 240 | 0.7351 | `reports/VLM_FEW_SHOT_FULL_240.md` |

**Delta Opus − Haiku zero-shot:** +0.051 wF1 (+~7pp accuracy). Modest gain,
not transformative. Well below the 0.5 F1 threshold that would have made a
retrieval-free pipeline attractive.

## Decision (per the predefined rule)

- If Opus zero-shot > 0.5 F1 → simpler pipeline viable → **NO** (0.277)
- If 0.3 – 0.5 F1 → Opus helps but anchors still needed → **NO** (0.277 is below 0.3)
- **If < 0.3 F1 → confirms anchors are essential regardless of model strength → YES**

**Keep the anchor-based few-shot VLM path.** Do not invest engineering effort
in a zero-shot Opus flow for the hackathon submission — it would regress
wF1 from ~0.735 to ~0.28 while costing ~2× more per call.

## Why zero-shot fails (hypothesis)

1. **Tear ferning morphology is out-of-distribution** for any pretraining
   corpus. Even BiomedCLIP's PubMed images rarely show AFM scans of dried
   tears. Textual descriptors like "granular loops" have no visual anchor.
2. **Class prototypes need visual grounding.** With k=3 anchors per class,
   the VLM can compare a query tile to concrete examples; with only text, it
   falls back to colormap priors (bright=dense=healthy).
3. **Opus's stronger reasoning doesn't help** when the bottleneck is visual
   grounding of unfamiliar microstructure — extra reasoning just produces
   more fluent but still wrong rationales.

## Files

- Script: `/Users/rafael/Programming/teardrop-challenge/scripts/vlm_zero_shot_opus.py`
- Predictions: `/Users/rafael/Programming/teardrop-challenge/cache/vlm_zero_shot_opus_predictions.json`
- Manifest: `/Users/rafael/Programming/teardrop-challenge/cache/vlm_zero_shot_opus_manifest.json`
- Summary: `/Users/rafael/Programming/teardrop-challenge/cache/vlm_zero_shot_opus_summary.json`
- Tiles: `/Users/rafael/Programming/teardrop-challenge/cache/vlm_zero_shot_opus_tiles/scan_XXXX.png`
