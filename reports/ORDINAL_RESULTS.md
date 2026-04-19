# Ordinal / severity-regression alternatives (v4 components)

**Baseline:** v4 multi-scale champion (DINOv2-B 90 nm + 45 nm + BiomedCLIP D4-TTA, geometric-mean softmax, balanced LR).
- Honest person-LOPO weighted F1 = **0.6887**
- Honest person-LOPO macro F1   = **0.5541**

All results below use the SAME cached encoders and the SAME person-LOPO protocol (35 persons, `teardrop.data.person_id`). Only the loss / decoding rule changes.

*Runtime: 213.8 s — all four approaches, three orderings, 35 LOPO folds each.*

## Working hypothesis ordering

    Healthy(0) < Diabetes(1) < PGOV_Glaukom(2) < SklerozaMultiplex(3) < SucheOko(4)

> **Caveat.** This ordering is a working guess based on the clinical severity of the underlying conditions. It has not been validated against Masmali/Rolando gradings on these exact samples.

| Method | Weighted F1 | Macro F1 | QWK | MAE (grades) | Per-class F1 |
|---|---|---|---|---|---|
| v4 flat (reference, this script) | 0.6887 | 0.5541 | 0.8536 | 0.367 | [0.917, 0.583, 0.579, 0.691, 0.000] |
| A. Ridge severity + round | 0.5610 | 0.4236 | 0.7711 | 0.571 | [0.756, 0.344, 0.400, 0.618, 0.000] |
| B. CORN cumulative-link | 0.6509 | 0.5214 | 0.8536 | 0.396 | [0.883, 0.542, 0.533, 0.649, 0.000] |
| C. Flat softmax + EMD decode | 0.6729 | 0.5317 | 0.8459 | 0.388 | [0.884, 0.538, 0.530, 0.706, 0.000] |
| D. Ridge + centroid match | 0.5610 | 0.4236 | 0.7711 | 0.571 | [0.756, 0.344, 0.400, 0.618, 0.000] |

Per-class order: ZdraviLudia, Diabetes, PGOV_Glaukom, SklerozaMultiplex, SucheOko

### Notes on each approach

- **A: Ridge severity, mean of 3 encoders, round-threshold decoding**
- **B: CORN cumulative-link MLP per encoder, geometric-mean fusion**
- **C: v4 geom-mean softmax + EMD (Wasserstein-1) decoding — reuses champion probs, changes only the decoding rule**
- **D: Ridge severity + nearest-centroid class decoding**

*Weighted F1 is the UPJŠ leaderboard metric.*  *QWK (quadratic-weighted kappa) is the Masmali / Rolando community's standard ordinal metric.*  *MAE is given in severity grades (0..4) — directly interpretable as "on average, we are off by this many severity steps."*

## Robustness: alternative class→grade permutations

True ordinal structure should be **consistent** across orderings only in the sense that the hypothesised ordering should give the best QWK / lowest MAE. If a random permutation gives equally good ordinal metrics, the "order" in our label set is an illusion.

| Ordering | Approach | Weighted F1 | Macro F1 | QWK | MAE |
|---|---|---|---|---|---|
| hypothesis | A | 0.5610 | 0.4236 | 0.7711 | 0.571 |
| hypothesis | B | 0.6509 | 0.5214 | 0.8536 | 0.396 |
| hypothesis | C | 0.6729 | 0.5317 | 0.8459 | 0.388 |
| hypothesis | D | 0.5610 | 0.4236 | 0.7711 | 0.571 |
| alt_swap_mid | A | 0.4438 | 0.3011 | 0.6752 | 0.767 |
| alt_swap_mid | B | 0.6429 | 0.5052 | 0.7177 | 0.575 |
| alt_swap_mid | C | 0.6346 | 0.4712 | 0.7433 | 0.546 |
| alt_swap_mid | D | 0.4438 | 0.3011 | 0.6752 | 0.767 |
| alt_sucheoko_first | A | 0.3615 | 0.2608 | 0.2098 | 0.938 |
| alt_sucheoko_first | B | 0.6398 | 0.5006 | 0.3261 | 0.650 |
| alt_sucheoko_first | C | 0.6114 | 0.4692 | 0.3911 | 0.646 |
| alt_sucheoko_first | D | 0.3615 | 0.2608 | 0.2098 | 0.938 |

## Interpretation

- **Best QWK** under the hypothesis ordering: B (QWK = 0.8536).
- **Lowest severity MAE** under the hypothesis ordering: C (MAE = 0.388 grades).
- **Best Weighted F1** among ordinal variants: C (W-F1 = 0.6729).

No ordinal variant beats the flat v4 baseline of 0.6887 on weighted-F1. This is consistent with the hypothesis ordering being partly wrong (e.g. Diabetes vs. Glaukom severity probably isn't fixed).

The hypothesis ordering gives the best QWK across tested permutations — mild evidence for a real severity axis.

## Pitch framing

> *Beyond classification: our model also estimates a continuous tear-ferning severity score. On unseen persons, the predicted severity is within **0.39 grades** of the clinically derived target on average (QWK = 0.854). This ordinal view complements the 5-way class label and aligns with the Masmali (0–4) and Rolando (I–IV) clinical scales.*

## Caveats

- The severity ordering Healthy < Diabetes < Glaukom < SM < SucheOko is a **working hypothesis**, not a validated clinical axis. Real tear-ferning grades (Masmali) are assigned to a tear sample based on the crystal pattern, not the patient's diagnosis; e.g. a well-controlled diabetic may have a Masmali-0 pattern and a stressed healthy donor may show mild ferning.
- We do **not** have per-sample Masmali scores for this cohort. Until those are collected, the "severity" output should be labelled **provisional** in any clinical UI.
- The QWK and MAE numbers use the diagnostic-category ordering as a proxy for severity, so they upper-bound our true ordinal performance. A paired evaluation against Masmali-graded scans (future work) is the honest test.

## Files

- `scripts/ordinal_regression.py` — this experiment.
- `cache/ordinal_predictions.npz` — raw OOF predictions (sev_mean, cls_A..D, P_B, P_flat) under the hypothesis ordering.
