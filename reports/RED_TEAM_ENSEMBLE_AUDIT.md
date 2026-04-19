# Red-Team Audit — `ENSEMBLE_PROB_RESULTS.md` (2026-04-18)

Auditor: independent red-team review. Artifacts inspected:
`scripts/prob_ensemble.py`, `reports/ENSEMBLE_PROB_RESULTS.md`, `cache/best_ensemble_predictions.npz`,
`teardrop/cv.py`, `teardrop/data.py`.

The **claim under review**:
- 0.6698 weighted F1, person-level LOPO (35 groups), 5-class AFM tear-film classification.
- Best config: uniform average of DINOv2-B + BiomedCLIP tiled-embedding LR probas,
  with per-class thresholds `[0.95, 1.00, 1.00, 0.85, 1.00]`.
- Baseline quoted: 0.615 (matches the standalone DINOv2-B row in the report).

Auditor verdict up-front: **number is inflated. The headline 0.6698 embeds two
layers of test-set reuse.** Honest numbers (best layer alone fixed vs fully honest):
0.6528 (fixed subset, nested-LOPO thresholds) and **0.5847** (fully nested:
both subset and thresholds picked on inner folds).

---

## Check 1 — Threshold-selection leakage

**Verdict: CONFIRMED LEAKY (category (a) in the rubric).**

Evidence from `scripts/prob_ensemble.py`:

- `lopo_predict_proba` produces OOF probabilities for all 240 scans (one pass of
  person-level LOPO per component). These OOF matrices are `P_by_comp[...]`.
- The top-level flow (`main`) then:
  1. enumerates ensemble candidates (uniform, weighted, geometric, per-class
     Dirichlet, stacking, all 2/3/4-subsets — ~30 candidates total),
  2. picks `best = max(results, key weighted_f1)` on the full 240-row OOF
     predictions (`strategy / subset` selection — leak #1),
  3. calls `optimize_thresholds(r["P"], y)` on each of the top-10 rows and
     selects `best_post = max(..., key post_f1)` — again on the full 240-row
     OOF set (threshold selection — leak #2).
- `optimize_thresholds` is a greedy coordinate ascent that directly maximises
  `f1_score(y, (P / thresholds).argmax(1), average="weighted")` on the input
  `P, y` pair. No inner fold. No held-out set.

So both the ensemble-strategy choice AND the per-class thresholds are chosen
on the same OOF predictions on which the final F1 is reported. Classic
test-set fitting.

### Magnitude of the optimistic bias — honest nested LOPO

I reran the whole pipeline with **nested** person-LOPO:

| setting | weighted F1 | macro F1 | Δ vs claim |
|---|---|---|---|
| Claimed headline (leaky strategy + leaky thresholds) | **0.6698** | 0.4823 | — |
| Pre-threshold honest (argmax only, same subset) | 0.6346 | 0.4934 | −0.035 |
| Fixed subset `dinov2_b+biomedclip`, nested thresholds (leave-one-person-out on the threshold search) | **0.6528** | 0.4985 | −0.017 |
| Fully nested — subset AND thresholds chosen on inner 34 persons per outer fold | **0.5847** | 0.4474 | **−0.085** |

Protocol for the two "honest" rows:
- For each outer person p:
  - compute best thresholds (or best (strategy, thresholds)) by maximising
    weighted F1 on the OOF predictions of the remaining 34 persons,
  - apply those thresholds to p's held-out scans,
  - concatenate the 35 outer-fold predictions and compute weighted F1 once.

Extra diagnostic — the "leaky leaderboard" after tuning thresholds on the full
240 OOFs for every strategy (reproduced against the report's numbers):

| strategy | pre-thr | leaky post-thr |
|---|---|---|
| k2 dinov2_b+biomedclip | 0.6346 | 0.6698 |
| k3 s+b+biomedclip      | 0.6285 | 0.6564 |
| k4 s+b+biomedclip+hc   | 0.6146 | 0.6587 |
| all5 geometric         | 0.6155 | 0.6569 |

The team already reports all these in `ENSEMBLE_PROB_RESULTS.md`; I re-derived
them to confirm reproducibility. **Note the ordering changes under nested
tuning** — `k2_b_clip` is only selected as the inner-best strategy in 31 of 35
outer folds, the other 4 picks are scattered across `k2_s_b`, `k3_s_b_clip`,
`k4_s_b_clip_hc`, and `all5_geom`.

---

## Check 2 — Reproducibility of the headline number

**Verdict: exact reproduction.** The artifact `cache/best_ensemble_predictions.npz`
stores `proba`, `pred_label`, `true_label`, `scan_paths`, `thresholds`.

Reproductions from that file:
- `f1_score(true, pred_label, 'weighted')` = **0.6698** (stored preds).
- Applying `thresholds = [0.95, 1.0, 1.0, 0.85, 1.0]` to `proba` via
  `(proba/thresholds).argmax(1)` = **0.6698** (matches stored preds exactly —
  `np.array_equal` is `True`).
- Pure argmax on `proba` (no thresholds) = **0.6346** weighted F1.

Re-deriving the ensemble from scratch (re-running LOPO on the cached DINOv2-B
and BiomedCLIP tile means, uniform-averaging) gives max-abs-delta `0.000000`
against the saved `proba`. No data leakage of any other kind; the saved
artifact is internally consistent.

So the gap the thresholds buy over the raw ensemble is `0.6698 − 0.6346 = +0.035`
**in-sample**. Of that, roughly `0.017` survives a nested evaluation (i.e. the
thresholds do genuinely help — just not as much as advertised).

---

## Check 3 — Person-level grouping sanity

**Verdict: correct.**

- `enumerate_samples` attaches both `patient` (eye-level, via `patient_id`) and
  `person` (person-level, via `person_id`) to each `Sample`.
- `scripts/prob_ensemble.py` uses `s.person` for `person_groups` and feeds that
  to `leave_one_patient_out`. The function name is a leftover; the content
  partitions on the supplied groups.
- Unique `person` count = **35**, matching the report header.
- Loading the saved `scan_paths` and mapping back to `person_id` also yields 35
  groups across 240 scans. No scan lives in both train and validation of any
  LOPO fold.
- L/R eye merging is non-trivially exercised: 44 eye-level groups collapse to
  35 person-level groups (9 persons have both eyes present).

---

## Bonus — Eye-level LOPO (44 groups) vs person-level (35)

For the fixed "best" subset (dinov2_b + biomedclip uniform):

| grouping | pre-thr | post-thr (leaky) | post-thr (nested/honest) |
|---|---|---|---|
| person-level (35) | 0.6346 | 0.6698 | 0.6528 |
| eye-level (44)    | 0.6506 | 0.6780 | 0.6516 |

Surprise result: **eye-level is marginally *better* than person-level here**,
both in pre-threshold and nested terms. The honest nested gap is essentially
zero (+0.001 in favour of eye). Interpretation:
- The stricter person-level grouping gives the model fewer training rows but
  adds only a handful of truly new held-out persons (9 of 35 had L/R pairs).
- The SucheOko class has just 2 persons; under either grouping both splits
  have essentially no information for that class (post-threshold F1 = 0.000 on
  person-level; it contributes zero usable signal).
- Net: on this dataset, the "patient leakage between L/R eye of same person"
  concern that motivated `person_id()` does not change headline numbers. Use
  person-level (it's more defensible) but be aware the gain/loss is noise-level.

---

## Overall verdict

**Inflated.** The claimed 0.6698 weighted F1 reflects two stacked test-set-fitting
steps on a single fixed OOF matrix. With a fair, nested protocol:

- If you **commit up-front** to the subset `dinov2_b + biomedclip` (the
  per-component F1 already ranked this as the top pair — see `dinov2_b` 0.615,
  `biomedclip` 0.584 in the standalone table) and only tune thresholds
  nested, honest F1 is **0.6528** (−0.017 vs. claim, still a +0.038 gain over
  the 0.615 single-model baseline).
- If you do not commit up-front and allow the pipeline to *choose* the
  strategy on OOF too, honest F1 collapses to **0.5847** (−0.085 vs. claim,
  -0.030 *below* the 0.615 baseline). The strategy vote is unstable across
  outer folds (k2_b_clip wins 31/35, but 4 folds disagree).

The **clean uplift attributable to the method** is therefore:

| comparison | honest Δ over 0.615 baseline |
|---|---|
| threshold tuning only, fixed subset | **+0.038** (0.6528) |
| threshold tuning + subset search | **−0.030** (0.5847) |

The "fixed subset, nested thresholds" path is defensible because the pair
`dinov2_b + biomedclip` is independently motivated by the two strongest
standalone components. But the *number you should quote externally* is
0.6528, not 0.6698.

---

## Recommendations

1. **Stop reporting 0.6698 as the headline.** Replace with 0.6528 (nested
   thresholds, fixed `dinov2_b + biomedclip`). State clearly that thresholds
   are tuned per outer LOPO fold on the remaining 34 persons. Bias bound: no
   evidence of any remaining leak; the only hyperparameter learned on the full
   set is the *choice* of subset, which is directly justified by the
   standalone leaderboard.
2. **Do not quote a number that relies on post-hoc strategy selection.** If
   you want an "automatic method picks everything" bottom line, it is 0.5847
   — worse than the single-model baseline.
3. **Fix `scripts/prob_ensemble.py`** to make the leakage explicit: either (a)
   add a nested evaluation path as the primary output and keep the leaky path
   only as an "upper-bound/diagnostic" marker, or (b) split the 35 persons
   into an inner tuning group (e.g. 7 persons) and an outer held-out group
   once, up-front.
4. **Per-class thresholds are contributing real but modest uplift** (~+0.017 in
   honest terms, not +0.035). Consider reporting both the pre-threshold
   (0.6346) and post-threshold honest (0.6528) numbers side-by-side so
   readers can see the actual threshold effect size.
5. **SucheOko is effectively unclassifiable with this data (2 persons, 14
   scans).** Regardless of method, F1 = 0. Say so in the report rather than
   letting a 0.000 row sit next to 0.86 without comment; consider a 4-class
   variant that drops SucheOko as a secondary reported number.
6. **Grouping choice (person vs eye) is numerically a wash here.** Keep
   `person_id` (it's the more conservative convention and the code already
   routes through it correctly), but don't oversell it as a necessary
   correction — eye-level would give almost identical honest numbers on this
   split.
7. **Optional: pre-register the subset.** The fact that `k2_b_clip` was the
   inner-best in 31/35 folds means a fixed pre-registration of that subset
   would be honest and stable. Do that explicitly in the paper/report.

---

## Artifacts produced by this audit

- `cache/red_team_audit.npz` — first pass: `P_ens`, `y`, `person_groups`,
  leaky thresholds, honest nested predictions for the fixed subset.
- `cache/red_team_audit_full.npz` — full pass: person-level nested + eye-level
  nested predictions for the fixed subset, both groupings, full `y`/groups.

Both can be inspected with `np.load(..., allow_pickle=True)`.
