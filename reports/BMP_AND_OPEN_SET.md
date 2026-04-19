# Defensive infra: BMP fallback + open-set abstention

Two test-time scenarios the champion doesn't cover natively:

1. Hidden test set is **BMP preview images**, not raw Bruker `.NNN` SPM files.
2. Hidden test set contains **novel classes** (up to 9, from the PDF brief)
   beyond the 5 seen in `TRAIN_SET`.

This note documents the defensive infra we built and the honest evaluation
numbers for each fallback.

---

## 1. BMP fallback — `teardrop/bmp_infer.py`

### Crop region
All 240 BMPs are rendered at **704×575 RGB** with a consistent data region
at rows `[8, 531)` × cols `[93, 616)` — i.e. a **523×523** square.  Empirically
verified across Diabetes / Glaukom / SM / SucheOko / ZdraviLudia: the bbox
never moves.  The axis label `0.0 … 92.5 μm` (the scan range watermark) sits
strictly BELOW row 531 and is cropped away.

A proportional fallback kicks in if the organizer re-renders at a different
size — we strip the same fractional borders.

### Pipeline
```
BMP (704x575x3) -> crop to 523x523 data region
                -> 3x3 grid of overlapping 224² tiles (9 tiles)
                -> DINOv2-B encode (9 tiles) -> mean-pool scan emb
                -> BiomedCLIP encode (9 tiles) -> mean-pool scan emb
                -> 3x v2-recipe LR heads -> geom-mean -> argmax
```
No plane-level, no pixel-size resample, no robust-normalize — the BMP is
already a rendered 8-bit image.

### Honest person-LOPO F1 (238/240 scans with BMP)

| Model | weighted F1 | macro F1 |
|---|---:|---:|
| Raw-SPM v4 champion (reference) | 0.6887 | 0.5541 |
| **BMP v4-style (D+D+Bc geom-mean)** | **0.6574** | **0.5270** |
| BMP geom-mean(D+Bc) | 0.6490 | 0.5187 |
| BMP DINOv2-B only | 0.6202 | 0.5152 |
| BMP BiomedCLIP only | 0.6028 | 0.4538 |

Per-class F1 (v4-style BMP):
- ZdraviLudia: 0.84 (vs 0.87 raw)
- SklerozaMultiplex: 0.67 (vs 0.69 raw)
- PGOV_Glaukom: 0.65 (vs 0.58 raw — **actually better!**)
- Diabetes: 0.39 (vs 0.58 raw — worst delta)
- SucheOko: 0.08 (2-patient ceiling persists)

**Conclusion:** BMP degradation is only **-0.031 weighted F1 / -0.027 macro F1**
vs the raw-SPM champion.  The 523² data region retains enough structural
information for the foundation encoders; 704×575 rendering loses the fine
dendritic tips that matter for Diabetes disambiguation but not the broad
topology.  This is a *very* graceful degradation — much better than the
0.50–0.60 we expected.

### CLI

```bash
.venv/bin/python predict_cli.py \
    --model models/ensemble_v4_multiscale \
    --input /path/to/TEST_SET_BMP \
    --input-format bmp \
    --output submission_bmp.csv
```

---

## 2. Open-set abstention — `teardrop/open_set.py`

### Wrapper
```python
from teardrop.open_set import OpenSetPredictor
from models.ensemble_v4_multiscale.predict import TTAPredictorV4

base = TTAPredictorV4.load()
open_clf = OpenSetPredictor(base, threshold=0.60)
label, probs = open_clf.predict_scan(path)  # label can be "UNKNOWN"
```
Threshold selection helpers in `teardrop.open_set` pick T from OOF quantiles:
`pick_threshold_from_oof(proba, y, correct_floor_pct=10)` returns the 10th
percentile of max-softmax on correctly-classified OOF scans.

### Honest simulation (`scripts/eval_open_set.py`)

We hold out **SucheOko** entirely from training (14 scans, 2 persons).  Train
a 4-class v4-recipe LR ensemble on the remaining 226 scans; then score ALL
240 scans.  The max-softmax on the 14 SucheOko scans is the OOD score.

| Metric | Value | Notes |
|---|---:|---|
| 4-class OOF weighted F1 (known classes) | 0.7420 | up from 5-class 0.69 — fewer confusers |
| 4-class OOF macro F1 (known classes) | 0.6944 | |
| AUROC in-sample known vs OOD | **0.8483** | optimistic — known = training in-sample |
| AUROC OOF known vs OOD | **0.6220** | **honest number** |

### Threshold scan

| T choice | T | TPR (unknown flagged) | FPR known (OOF) |
|---|---:|---:|---:|
| 10th pctile of correct conf | 0.775 | 21.4 % (3/14) | 14.6 % (33/226) |
| 25th pctile | 0.899 | 50.0 % (7/14) | 30.1 % (68/226) |
| 50th pctile | 0.990 | 71.4 % (10/14) | 53.1 % (120/226) |
| Fixed T=0.60 | 0.600 | 7.1 % (1/14) | 3.5 % (8/226) |
| Fixed T=0.70 | 0.700 | 14.3 % (2/14) | 11.1 % (25/226) |

Interpretation: to flag even **half** of unknowns, we must abstain on **30 %**
of correctly-classifiable known-class scans — a bad trade.  12/14 SucheOko
scans get confidently misclassified as SklerozaMultiplex, meaning the
foundation-encoder embedding places them inside the SM cluster — there is no
"low-confidence" exit.

### Why max-softmax is weak here

1. 4-class LR w/ `class_weight="balanced"` is inherently over-confident on
   small-dataset LOPO (calibration is poor at n=226).
2. SucheOko and SM share morphology (fragmented dendritic networks) so the
   OOD-ness is tiny in embedding space — the model is legitimately "sure"
   when it sees SucheOko.
3. At 35 persons / 4 classes the max-softmax distribution of correct
   predictions already has 90th-percentile ≥ 0.99, leaving no room for a
   sharper cutoff.

### On the 5-class production OOF

Same thresholds applied to the shipped 5-class OOF (`best_ensemble_predictions.npz`):

| T | pass-correct | pass-wrong |
|---:|---:|---:|
| 0.775 (p10) | 62.3 % | 41.9 % |
| 0.899 (p25) | 44.8 % | 32.6 % |
| 0.600 | 76.6 % | 65.1 % |

The gap `pass_correct − pass_wrong` tops out at ≈ 20 percentage points — the
confidence signal is real but weak.  Using abstention at T = 0.77 would
correctly hold back 58 % of the model's wrong answers but also drop 38 % of
its right ones.

---

## 3. Honest verdict

| Scenario | Infra shipped | Expected deployment behavior |
|---|---|---|
| Organizer ships BMP instead of SPM | `teardrop/bmp_infer.py` + `--input-format bmp` | **-0.03 F1** vs raw champion.  Usable as-is. |
| Organizer ships 9 classes instead of 5 | `teardrop/open_set.py` OpenSetPredictor | **AUROC 0.62** on honest simulation.  Use threshold T ≈ 0.60 for a *soft* abstention (~7 % TPR, 3.5 % FPR) only if the brief gives partial credit for UNKNOWN; do NOT abstain aggressively. |

**Recommendation for the hackathon submission:**

- If BMP-only input: ship v4 with `--input-format bmp`.  Expect 0.65 weighted
  F1, still well above the 0.276 null.
- If suspected unseen classes: set `threshold=0.60`, gaining some defensive
  abstention (1/14 unknowns flagged) without destroying known-class recall.
  But be clear in the pitch that this is **not a strong OOD detector** —
  max-softmax abstention on a small-data LR head is fundamentally limited.

---

## 4. Files added / modified

- `teardrop/bmp_infer.py` — new; `preprocess_bmp()` + `BmpPredictorV4` wrapper
- `teardrop/open_set.py` — new; `OpenSetPredictor` + threshold helpers
- `predict_cli.py` — added `--input-format {spm,bmp}` flag
- `scripts/eval_bmp_fallback.py` — new
- `scripts/eval_open_set.py` — new
- `reports/BMP_FALLBACK_EVAL.md` — raw numbers
- `reports/OPEN_SET_EVAL.md` — raw numbers
- `cache/bmp_scan_emb_dinov2_vitb14.npz`,
  `cache/bmp_scan_emb_biomedclip.npz`,
  `cache/bmp_fallback_oof.npz`,
  `cache/open_set_sim_sucheoko.npz` — artifacts
