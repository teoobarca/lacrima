# Red-team audit — E7 (multichannel 3-way) vs E1 (v2 champion)

Date: 2026-04-18.  Auditor: red-team bootstrap.  B=1000 person-level resamples, 5 LR seeds, person-LOPO only (`teardrop.cv.leave_one_patient_out`, groups from `teardrop.data.person_id`).

## TL;DR

**Do not ship E7 as v3. Stay with v2.**

Reproduction passes exactly (E1 = 0.6562, E7 = 0.6645), but the person-level bootstrap 95% CI for ΔF1 is **[−0.0405, +0.0515]** with **P(ΔF1 > 0) = 0.598** — the claimed +0.0083 gain is well within person-resampling noise. The full support-weighted improvement comes from +0.125 on Diabetes alone (contributes +0.0130 to ΔW-F1, while Glaukom gives back −0.0086 and SucheOko −0.0038). That's a single class driving a sub-sigma global move, with two classes regressing. Do not crown.

## 1. Reproduction (claim reproduces?)

| Ensemble | Claim W-F1 | Recomputed W-F1 | Match? |
|---|---:|---:|:---:|
| E1 (champion v2) | 0.6562 | 0.6562 | PASS |
| E7 (challenger) | 0.6645 | 0.6645 | PASS |

- Macro F1: E1 = 0.5382, E7 = 0.5435  (claimed 0.5382 / 0.5435).
- Saved npz E7 softmaxes reproduce E7 W-F1 = 0.6645 (integrity check passes).
- Saved npz `pred` agrees with fresh re-run on 240/240 scans.

## 2. Person-level bootstrap CI for ΔF1 = F1(E7) - F1(E1)

- Resampling: persons with replacement (n=35 per bootstrap), B = 1000.

- Point estimate (observed): **+0.0083**.
- Bootstrap mean: +0.0061, median: +0.0055.
- **95% CI: [-0.0405, +0.0515]**.
- **P(ΔF1 > 0) = 0.598**.
- Verdict: 95% CI crosses 0 — improvement is within person-resampling noise.

## 3. Seed sensitivity (LR `random_state`)

Five seeds re-run: [0, 1, 13, 42, 123]. Each seed recomputes full LOPO for both E1 and E7.

| seed | E1 W-F1 | E7 W-F1 | Δ |
|---:|---:|---:|---:|
| 0 | 0.6562 | 0.6645 | +0.0083 |
| 1 | 0.6562 | 0.6645 | +0.0083 |
| 13 | 0.6562 | 0.6645 | +0.0083 |
| 42 | 0.6562 | 0.6645 | +0.0083 |
| 123 | 0.6562 | 0.6645 | +0.0083 |

- E7 across seeds: mean = **0.6645**, std = **0.0000**, range = [0.6645, 0.6645].
- E1 across seeds: mean = 0.6562, std = 0.0000.
- Δ across seeds: mean = +0.0083, std = 0.0000, all positive = True.
- Seed-variance threshold: std > 0.008 ⇒ unstable. E7 std = 0.0000 ⇒ **stable** on this axis.

**Caveat:** `LogisticRegression(solver="lbfgs")` is deterministic given the fit data — it does not consume `random_state` for anything that varies the fit on a fixed problem. So this check only rules out optimizer noise; it does not rule out data-sampling noise. The bootstrap in §2 is the load-bearing uncertainty estimate, and it does not support shipping.

## 4. Per-class breakdown (E1 → E7)

| class | E1 F1 | E7 F1 | Δ | claimed Δ | match? |
|---|---:|---:|---:|---:|:---:|
| ZdraviLudia | 0.8690 | 0.8844 | +0.0154 |  |  |
| Diabetes | 0.5417 | 0.6667 | +0.1250 | +0.125 | PASS |
| PGOV_Glaukom | 0.5641 | 0.5067 | -0.0574 | -0.057 | PASS |
| SklerozaMultiplex | 0.6517 | 0.6596 | +0.0079 |  |  |
| SucheOko | 0.0645 | 0.0000 | -0.0645 |  |  |

- SucheOko went from one correct scan (F1 = 0.0645) to zero correct (F1 = 0). Given only 2 SucheOko persons, that's 1 scan's worth of change.
- Support-weighted contributions to ΔW-F1:
  - ZdraviLudia: +0.0045
  - Diabetes: +0.0130
  - PGOV_Glaukom: -0.0086
  - SklerozaMultiplex: +0.0031
  - SucheOko: -0.0038

- Diabetes alone contributes **+0.0130** to ΔW-F1; all other classes together contribute -0.0048. The +0.0083 total is dominantly driven by Diabetes.

## 5. Verdict

Ship decision inputs:
- Reproduction: E1 PASS, E7 PASS.
- Bootstrap 95% CI for ΔF1: [-0.0405, +0.0515].
- P(ΔF1 > 0) under person-resampling: 0.598.
- Seed std(E7) = 0.0000 (threshold 0.008).
- Seed Δ all positive: True.
- Diabetes drives 158% of ΔW-F1.

### Final call: **Do not ship E7 as v3. Stay with v2 (0.6562).**

Reasoning: reproduction passes (point estimate is real), but (a) the 95% bootstrap CI [−0.0405, +0.0515] comfortably includes 0 — P(ΔF1 > 0) is only 0.598, barely above a coin flip; (b) the full gain rides on a single class (Diabetes), with two other classes regressing — a fragile bet on a 36-scan minority class; (c) the seed check is degenerate for lbfgs LR and therefore not evidence of robustness. The agent's own caveat ("+0.0083 ≈ 2 scans out of 240") is correct and determinative: at this dataset size, a 2-scan swing is not a real improvement. Keep v2 as champion and only revisit E7 if (i) a larger held-out cohort confirms the Diabetes lift, or (ii) a fold-level paired test (e.g., sign test over the 35 person-folds) is significant.

