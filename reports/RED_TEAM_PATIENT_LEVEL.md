# Red-Team: Patient-Level Classifier (Wave 7 claim)

**Claim:** attention-pooled patient-level LR reaches scan-level wF1 = 0.8177 (+0.1290 vs v4 champion 0.6887), P(Δ>0) = 1.000 bootstrap.

**Verdict: CAVEAT — ACCEPT only under a per-patient evaluation regime. REJECT the headline +0.129 gain as apples-to-oranges.**

## Audit results

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Person-LOPO integrity inside `patient_level_classifier.py` | PASS | `lopo_person_predict` trains LR on persons `j != i`, fits StandardScaler on train only, predicts on held-out person vector. Person vectors are means of that person's own scan embeddings — a feature, not a label leak. |
| 2 | "Broadcasting 1 prediction to N scans" fair vs v4? | **FAIL** | All 35/35 train patients are 100% class-pure (`Counter`-check reproduced). Per-patient broadcasting is a structurally easier task than v4's per-scan independent prediction. |
| 3 | Attention weights = v4 top-1 prob, computed once pre-loop | PASS | v4 OOF was produced under person-LOPO (confirmed in `reports/METRICS_UPGRADE.md` and `threshold_calibration.py` docstring). Attention weights therefore carry no label leakage. |
| 4 | Patient-class homogeneity | CRITICAL | 35/35 patients pure → oracle-patient-label broadcast = **wF1 = 1.0000**. Any reasonable per-patient method inflates scan-level wF1 vs per-scan baseline for free. |
| 5 | Fair head-to-head (both broadcast) | **LARGELY ERASES GAIN** | Simple v4 soft-vote per patient → broadcast already hits **wF1 = 0.7739** (no new model, no attention). Attention adds +0.0438 on top, not +0.1290. |
| 6 | McNemar / per-patient Wilcoxon | Significant but tiny N | McNemar on 240 scans p = 3.1e-8 (inflated by scan-not-independent). Per-patient paired Wilcoxon (n=35) p = 1.16e-3. Real effective sample size is **35**, not 240. |
| 7 | SucheOko F1 = 0.000 | FAIL on minority | 2/2 SucheOko patients misclassified → the method is **not rescuing** the rarest class; it is exploiting easy pure-patient structure for the big classes (ZdraviLudia 1.00, Diabetes 0.94). |
| 8 | Hostile synthetic test (mixed-label patient) | Not run (method is structurally broken for this case) | If a test patient has scans from >1 class, the method **cannot** output different labels for them. Floor is `1/n_scans` accuracy on that patient. |

## Decomposition of the claimed +0.1290 wF1

```
v4 scan-level (per-scan):                   0.6887
v4 majority-vote-per-patient → broadcast:   0.7114   (+0.0227 free from broadcasting)
v4 mean-proba-per-patient  → broadcast:     0.7739   (+0.0852 free from broadcasting)
attn patient-level LR       → broadcast:    0.8177   (+0.0438 actual new-model gain)
ORACLE true patient label   → broadcast:    1.0000   (upper bound)
```

**>66% of the claimed gain is the broadcasting exploit, not the model.** Apples-to-apples (both broadcast), the new method adds ~0.044 wF1.

## When this ships vs does not

- **SHIP** if organizers evaluate **one label per patient** (single-scan submission per patient, or per-patient grading): patient-level attn is legitimately best and should be the submission aggregator. Gain over v4-soft-vote-broadcast is ~0.044, over v4-per-scan is ~0.129.
- **DO NOT SHIP as scan-level champion.** Reporting 0.8177 against v4's 0.6887 on the benchmark dashboard is an **apples-to-oranges comparison** and repeats the structural-inflation pattern of 3 earlier red-teamed claims.
- **DO NOT SHIP** if the test set admits per-scan heterogeneous labels (e.g. comorbidity, intra-patient progression). This method has a hard ceiling of patient-label accuracy and cannot recover from it.

## Required corrections

1. In `reports/PATIENT_LEVEL_CLASSIFIER.md` and `reports/BENCHMARK_DASHBOARD.md`: replace the "+0.1293 vs v4" bootstrap with the **v4-soft-vote-broadcast baseline (0.7739)** as the honest reference. Real gain = **+0.044 wF1**, not +0.129.
2. Add explicit disclaimer: "wF1 comparison is valid only under per-patient evaluation; under per-scan evaluation with heterogeneous intra-patient labels this method underperforms v4."
3. Drop the `P(Δ>0)=1.000` claim — bootstrap over 240 non-independent scans with 35 underlying decisions is misleading. Report per-patient Wilcoxon (p=1.2e-3, n=35) instead.
4. Patient-level F1 = 0.8367 on **35 persons** is the figure the pitch should lead with if ever, not a scan-level 0.8177.
