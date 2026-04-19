# Red-Team Audit v2 — `optimize_ensemble.py` / `ENSEMBLE_RESULTS.md` (2026-04-18)

Auditor: independent red-team review. Artifacts inspected:
`scripts/optimize_ensemble.py`, `reports/ENSEMBLE_RESULTS.md`,
`reports/best_oof_predictions.parquet`, `reports/ensemble_results.json`,
`cache/tiled_emb_*.npz`, `cache/features_handcrafted.parquet`,
`teardrop/cv.py`, `teardrop/data.py`.
Prior audit referenced: `reports/RED_TEAM_ENSEMBLE_AUDIT.md`.

**Claim under review (self-reported in `ENSEMBLE_RESULTS.md`):**
- LOPO weighted F1 = **0.6878**, macro F1 = **0.5622**
- Method: concat `[dinov2_s + dinov2_b + biomedclip + handcrafted]` → LR (balanced)
  → per-class log-prob bias tuning (coordinate ascent, 25 steps × 3 passes on [-3, +3])
- Tuned biases: `[Zdrav −0.75, Diab 0, PGOV +2.5, SM +1.25, Suche −1.5]`
- Per-class F1: `Zdrav=0.905 | Diab=0.640 | PGOV=0.585 | SM=0.680 | Suche=0.000`
- Δ claimed over "baseline 0.6280": `+0.0598`

**Up-front verdict: INFLATED.** Same failure pattern as the prior ensemble:
both the *subset* and the *per-class biases* are selected by maximising
weighted F1 on the exact 240-row OOF matrix whose F1 is then reported.
On top of that, the grouping is **eye-level (44 patient IDs) not person-level
(35 persons)**, which is the stricter convention used elsewhere in the project.

Honest numbers (same method, same features, nested or stricter grouping):

| protocol | wF1 | mF1 | Δ vs claim |
|---|---:|---:|---:|
| Claim headline (eye-level, leaky bias tune) | **0.6878** | 0.5622 | — |
| Eye-level, raw argmax (no bias)             | 0.6474 | 0.5190 | −0.040 |
| Eye-level, **nested** bias tuning           | 0.6403 | 0.5181 | −0.048 |
| Person-level, leaky bias tune               | 0.6859 | 0.5609 | −0.002 |
| Person-level, raw argmax                    | 0.6474 | 0.5190 | −0.040 |
| Person-level, **nested** bias tuning (honest) | **0.6326** | 0.5118 | **−0.055** |

Single honest number I trust: **0.6326 weighted F1** (person-level LOPO,
nested bias tuning, same 4-component concat + LR).

---

## Check 1 — Grouping level: **eye-level, not person-level**

**Verdict: FAIL — optimiser uses `patient_id` (44 eye-level groups), not
`person_id` (35 person-level groups).**

Evidence:
- `scripts/optimize_ensemble.py` reads `scan_groups` directly from the cached
  `tiled_emb_*_afmhot_t512_n9.npz` files (see `load_tiled`, line 57). Those
  caches store `scan_groups` as `patient_id(path)`, producing **44 unique IDs**
  (verified in `cache/tiled_emb_dinov2_vits14_afmhot_t512_n9.npz`).
- `reports/best_oof_predictions.parquet`, `patient` column, has **44 unique
  values** and matches `patient_id` tokens (`19_PM_SM`, `1-SM-LM-18`, …).
- The script never touches `person_id()` and never references `s.person`.
- Same dataset, same artefact, compared against `teardrop.data.person_id()`
  merging: 44 eye-level → 35 person-level.

Magnitude of gap from grouping alone, holding method fixed:

| method | eye (44) | person (35) |
|---|---:|---:|
| raw argmax (no bias)           | 0.6474 | 0.6474 |
| leaky bias tune                | 0.6878 | 0.6859 |
| **nested bias tune (honest)**  | 0.6403 | **0.6326** |

Gap is ~0.005–0.008, matching the "eye-vs-person gap is noise-level on this
dataset" finding from the prior red-team. Keep `person_id`: it is the stricter
and already-canonical convention in the repo (`s.person` exists on `Sample`).

---

## Check 2 — Bias-tuning leakage (dominant issue)

**Verdict: CONFIRMED LEAKY — identical failure pattern to the previous
ensemble.** The entire +0.040 uplift from `0.6474 → 0.6878` is in-sample
over-fit and does **not** survive nested evaluation.

Evidence from `scripts/optimize_ensemble.py`:

- `lopo_lr` (lines 95–114) produces per-scan OOF probabilities for the full
  240 rows.
- `per_class_threshold_tuning` (lines 153–194) runs coordinate ascent over
  25 values in `[-3, +3]`, 3 passes, directly maximising
  `f1_score(y, preds_with(trial), average="weighted")` against the full
  `y`/`probs` pair. No inner holdout, no outer split.
- In `main` (lines 404–410), this is called on `best_probs` (the full OOF
  matrix of the winning config) and the tuned F1 is what gets reported as the
  headline.

Leaky vs honest, same champion subset, **eye-level**:

| setting | wF1 | mF1 | per-class F1 (Zdrav, Diab, PGOV, SM, Suche) |
|---|---:|---:|---|
| raw argmax                        | 0.6474 | 0.5190 | 0.88 · 0.54 · 0.53 · 0.64 · 0.00 |
| **leaky bias tune (= claim)**     | **0.6878** | 0.5622 | 0.91 · 0.64 · 0.59 · 0.68 · 0.00 |
| nested bias tune (outer-LOPO over inner bias search) | **0.6403** | 0.5181 | 0.87 · 0.60 · 0.48 · 0.63 · 0.00 |

Same, **person-level**:

| setting | wF1 | mF1 |
|---|---:|---:|
| raw argmax                         | 0.6474 | 0.5190 |
| leaky bias tune                    | 0.6859 | 0.5609 |
| **nested bias tune (honest)**      | **0.6326** | 0.5118 |

Nested protocol (same as prior red-team):
- For each outer LOPO fold `p`:
  - tune biases by coordinate ascent on the OOF rows of the remaining
    34 persons (or 43 patients) using identical 25-step / 3-pass search;
  - apply those biases to `p`'s held-out OOF rows;
- concatenate all outer-fold predictions, compute weighted F1 once.

Interesting side-finding: the median per-fold bias across 35 person-level
outer folds is `[−0.75, 0, +2.5, +1.25, −3.0]` — very close to the claim's
global biases. In other words the bias pattern *is* stable across folds;
what the leaky protocol over-estimates is the **effectiveness** of that
pattern on held-out data. Stable biases that each shift class priors still
move test-fold predictions in ways that the global-fit F1 can't see, and the
specific value that maximises in-sample F1 (e.g. Suche=−1.5 drives all 14
SucheOko scans to non-SucheOko but inflates the weighted average because
SucheOko is only 14/240) over-shoots slightly for held-out folds. Net effect:
leaky headline − honest headline ≈ **0.05** weighted F1, bigger than the prior
audit's 0.017 gap.

Bias tuning is *not* useless here — it does lift macro-level per-class F1 in
classes that are underweighted in LR's prior-balanced logits (PGOV, SM). But
the honest uplift over the raw ensemble argmax is essentially zero on
person-level (−0.015) and −0.007 on eye-level. **The claimed +0.040 bias-tune
uplift is almost entirely overfit.**

---

## Check 3 — Subset / component selection

**Verdict: additional (small) leak, dominated by the bias-tune leak.**

The optimiser evaluates **5 concat configs × 2 classifiers = 10** pre-tuning
configurations against the full 240-row LOPO OOF (`main`, lines 371–386) and
then picks the argmax (line 399):

```
dinov2_s, dinov2_s+dinov2_b, dinov2_s+biomedclip,
dinov2_s+dinov2_b+biomedclip, dinov2_s+dinov2_b+biomedclip+handcrafted
  × {LR, XGB}
```

Bias tuning is applied only to the winner. The winner is
`dinov2_s+dinov2_b+biomedclip+handcrafted / LR` with raw wF1 0.6474. The
runner-up at raw is `dinov2_s+dinov2_b+biomedclip / LR` at 0.6388, so the
selected config is genuinely the top pre-tuning row. Gap is only 0.009, so
the subset-selection leak on top of the bias-tune leak contributes
at most ~0.01 extra optimistic bias and is not the main issue. A strictly
honest protocol would pre-register the feature set (e.g. fix it as "all four
components" as a defensible design choice) and skip the subset search
altogether — which lands us back at the same nested headline of **0.6326**
person-level.

The MLP and AVG3 branches don't enter the champion because LR+leaky-bias is
the best leaky number by +0.03; nothing changes if one fixes them.

---

## Check 4 — Baseline honesty

The report quotes `Baseline (RESULTS.md): 0.6280` and claims `Δ = +0.0598`.

- `0.6280` is the **eye-level**, standalone single-component LR baseline
  (matches `RESULTS.md`).
- The honest comparison target from the prior audit is **person-level** and
  fully nested: ensemble honest = **0.6528** (nested thresholds on
  `dinov2_b + biomedclip`) vs single-model person-level baseline ≈ 0.615.

Re-stated apples-to-apples:

| baseline / target                           | protocol            | wF1 |
|---|---|---:|
| Single-model LR on `dinov2_b` (RESULTS.md)  | eye-level LOPO      | 0.615 |
| Prior ensemble (red-teamed, fixed subset)   | person-level nested | **0.6528** |
| This ensemble claim                         | eye-level, leaky    | 0.6878 |
| This ensemble, SAME method, honest          | **person-level nested** | **0.6326** |
| This ensemble, SAME method, honest          | eye-level nested    | 0.6403 |

**The "honest number for this claim is 0.6326 person-level nested, which is
−0.020 *below* the previous red-teamed ensemble's 0.6528.** Adding handcrafted
features and changing from uniform-averaged two-model ensemble to concat-LR
did not help on honest evaluation — it hurt slightly. The +0.060 headline
improvement is entirely due to (a) looser grouping (eye vs person, +0.007)
plus (b) leaky bias tuning (+0.040), plus (c) minor subset-selection leak.

---

## Overall verdict

**Inflated by ~0.055 weighted F1.** The claim's 0.6878 eye-level leaky number
collapses to **0.6326 person-level nested** under the same honest protocol
applied to the prior ensemble. That is **worse than** the previous
red-team-certified headline of 0.6528, not better.

Summary of inflation budget:

| contributor | magnitude |
|---|---:|
| Grouping (eye → person)        | −0.0019 (leaky) |
| Bias-tune leakage (person-level leaky → nested) | **−0.0533** |
| Subset-selection leakage (bounded by raw leaderboard)   | ≤ −0.010 |
| **Total observed: 0.6878 − 0.6326** | **−0.0552** |

Comparison to prior red-team's 0.6528 (fixed subset, nested thresholds,
person-level):

| headline | honest number | Δ vs prior honest |
|---|---:|---:|
| Prior: `prob_ensemble.py` 0.6698 claim | 0.6528 | — |
| This: `optimize_ensemble.py` 0.6878 claim | **0.6326** | **−0.020** |

This claim is not an improvement. It looks bigger because it uses a looser
grouping and a more aggressive tuning-on-test loop.

---

## Recommendations

1. **Retract `0.6878` as the headline.** Replace with **0.6326** (person-level
   LOPO, nested per-class bias tuning on the same 4-component concat + LR).
   State grouping (person-level, 35 persons) and protocol explicitly.
2. **If a single external number is desired, the best we can defend from
   *this* pipeline is the nested person-level 0.6326, and it is worse than
   the prior ensemble's 0.6528.** Revert to the prior audit's recommended
   pipeline (uniform average of `dinov2_b + biomedclip`, nested thresholds)
   or show clearly why this new pipeline is preferred despite losing honest
   F1.
3. **Fix `optimize_ensemble.py` to use person-level groups.** Either
   recompute `scan_groups` via `teardrop.data.person_id()` or re-derive
   groups from `samples` directly rather than relying on the cached npz's
   eye-level `scan_groups`. Loading path: replace `scan_groups = s[3]` with
   groups built from `[Sample.person for Sample in samples]` aligned by
   `scan_paths`.
4. **Make the tuning protocol honest.** In `per_class_threshold_tuning`, add
   an outer-LOPO wrapper: for each outer fold, tune on inner-LOPO OOF of the
   remaining persons, apply to held-out person. Keep the leaky number only
   as an explicit upper-bound diagnostic, clearly labelled.
5. **SucheOko remains unclassifiable** (2 persons / 14 scans, F1 = 0.000 in
   every honest row). Note this explicitly instead of letting it sit as a
   zero line in the per-class table. Consider reporting a 4-class variant.
6. **The `handcrafted` branch doesn't help much on honest evaluation.** Raw
   argmax on full 4-component concat (0.6474 eye, 0.6474 person) is
   essentially the same as 3-component concat (0.6388 / leaky 0.6388 row in
   the report). If you want to keep handcrafted, report its honest marginal
   contribution separately, not as part of a leaky headline.

---

## Artifacts produced by this audit

- `cache/red_team_audit_v2.npz`: full OOF matrices and predictions for both
  groupings and both tuning protocols. Keys: `y`, `person_groups`,
  `eye_groups`, `oof_eye`, `oof_pers`, `leaky_pred_eye`, `honest_pred_eye`,
  `leaky_pred_pers`, `honest_pred_pers`, `biases_leaky_eye`,
  `biases_leaky_pers`, `biases_per_fold_eye`, `biases_per_fold_pers`.
- `scripts/redteam_v2.py`: reproduction script (≈ 30 s wall-clock on this
  machine for the full nested pass).

Both reproducible from cached embeddings; no re-encoding needed.
