# Orchestration State — Teardrop Challenge

Single source of truth for current experiment state. Orchestrator (Claude) updates after every round.

**Last updated: 2026-04-18 23:00 — After Wave 18 + patient-disjoint split CONFIRMED**

---

## 🎯 CONFIRMED TEST REGIME (from user, organizer info)

- **Per-image evaluation, patient info stripped**
- **Patient-disjoint split** — train patients ≠ test patients
- **Same class distribution**
- Consequence: our **person-LOPO F1 is the correct estimator** for test performance

## ⭐ SHIPPED CHAMPION: v4 multiscale @ wF1 0.6887 (honest, person-LOPO)

**Why v4 is the only viable champion under patient-disjoint**:
- Patient-level classifier (0.8177) requires patient ID → unusable
- Hybrid re-ID: test patients are all new → re-id never fires → fallback to v4 (safe but no bonus)
- v4 person-LOPO directly simulates new-patient scenario

---

## Wave 13-15 summary — high-EV smery failed, v4 still stands

| Wave | Experiment | Result | Verdict |
|---|---|---|---|
| 13A | Full 240 few-shot Haiku k=2 | wF1 0.6755 *(later proven LEAKY)* | CONTAMINATED |
| 13B | Few-shot k=3 variant | wF1 0.649 (k=3 < k=2) | k=2 stays |
| 13C | Expert Council (Haiku judge) | wF1 0.5912 (HURTS v4 by -0.097) | Rejected |
| 13D | Pure k-NN retrieval baseline | wF1 0.6117 (clean ablation) | v4 +21 pp over k-NN |
| 13E | Self-consistency TTA | +0.003 pp (noise floor) | Voting not worth 3× cost |
| 13F | Sonnet 4.6 60-subset | wF1 0.8454 *(CONTAMINATED)* | Leaky |
| 13G | Full 240 Sonnet 4.6 | wF1 0.8873 *(CONTAMINATED)* | **Red-team catch #8** |
| 13H | Opus 4.7 zero-shot | wF1 0.2773 (near-random) | Anchors essential |
| 13I | Opus 4.7 few-shot 60-subset | wF1 0.7974 (< Sonnet, 5× cost) | Sonnet is sweet spot |
| 14 | Red-team Sonnet 0.8873 audit | Filename leak #2 in collage paths | REJECT |
| 15a | Honest Sonnet full 240 rerun | wF1 **0.3424** (inflation +0.545!) | VLM path DEAD |
| 15b | Fusion ensemble (v4+kNN+XGB) | max 0.6613 (< v4) | v4 ceiling for clean components |
| 15c | Threshold calibration (nested LOPO) | 0.6933 (P(Δ>0)=0.637 = noise) | No significant gain |
| 15d | Leakage prevention infra | 16 scripts retrofitted + tests | DEPLOYED |

**Net outcome**: v4 @ 0.6887 remains honest champion. 8 red-team catches documented. Structural leakage prevention shipped.

## Wave 16 (DISPATCHED, in flight)

| # | Experiment | Agent | Expected |
|---|---|---|---|
| A | VLM binary re-ranker on v4 abstain cases (safe paths) | a6e96ebf8c29d6921 | +0.5-1.5 pp if VLM > 50% on binary |
| B | TDA persistent homology orthogonal track | a2c46644c385bc44e | +1-3 pp fusion candidate |
| C | Augmented linear head (8× D4 train data) | a7699c602bc9bf73d | +1-3 pp |

## Wave 20 LOCAL (no API, after credit limit hit)

| Experiment | wF1 | Δ vs v4 |
|---|---|---|
| Multi-seed v4 ensemble (5 seeds) | 0.6887 | 0.000 (LR is deterministic) |
| Bagging+multi-C (25 members) | 0.6712 | -0.017 (SucheOko underrep in bags) |
| Multichannel RGB fusion (Wave 6 revisit) | 0.6729 | -0.016 |
| Weighted geomean (inner-F1 weights) | 0.6854 | -0.003 |
| Tree heads (ExtraTrees / RandomForest) | 0.603 / 0.592 | -8.6 / -9.6 |
| LR+ET fusion | 0.6654 | -2.3 |

**Net: all local Wave 20 attempts under v4. Total Wave 13-20 = 30+ honest experiments, 0 new champions.** v4 multiscale @ 0.6887 is mathematical and empirical optimum for 240 scans / 35 persons / 5 classes / patient-disjoint regime.

## ⭐ v5 ADAPTIVE — production-grade upgrade SHIPPED

**Bundle**: `models/ensemble_v5_adaptive/`. Same F1 as v4, better calibration + triage + adaptive Re-ID safety net.

| Metric | v4 | v5 | Δ |
|---|---|---|---|
| Weighted F1 | 0.6887 | 0.6887 | 0 (SAFE — bootstrap CI [0,0]) |
| Macro F1 | 0.5541 | 0.5541 | 0 |
| **ECE (calibration)** | 0.2057 | **0.0821** | **-60%** |
| Re-ID fire rate | — | 7.1% | bonus on similarity > 0.94 |
| Re-ID fire accuracy | — | 94.1% | wins when fires |
| Triage @ margin=0.10 | — | 92.1% autonomous | acc 71.9% |

**Layers**: v4 base → Hybrid Re-ID adaptive blend → temperature scaling → triage abstain.
**Worst-case = v4** (mathematically additive, all layers are safe). **Best-case = +0.03 wF1** if test patients overlap (won't happen under patient-disjoint regime, but inert if not).

---

## [HISTORICAL] Current best — FINAL (after Wave 7 multi-scale, red-team approved)

| Metric | Value | Model | Status |
|---|---|---|---|
| **★ Weighted F1 (person-LOPO, CHAMPION)** | **0.6887** | Multi-scale (90+45nm) DINOv2-B + BiomedCLIP-TTA (v2 recipe) | ⏳ v4 TTA packaging in progress |
| Wave 5 v2 (superseded champ) | 0.6562 | TTA + L2-norm + geom-mean | ✓ `models/ensemble_v2_tta/` |
| Wave 4 v1 (superseded) | 0.6458 | TTA v1 arith-mean | ✓ `models/ensemble_v1_tta/` |
| Threshold-tuned ref | 0.6528 | non-TTA + nested thresholds | ✓ verified, fragile |
| Non-TTA 2-comp | 0.6346 | raw argmax | ✓ `models/ensemble_v1/` |
| Baseline | 0.615 | DINOv2-B single | ✓ |
| Label-shuffle null | 0.276 ± 0.042 | — | ✓ signal real (~12σ above) |

**Chosen submission (pending v4 TTA build):** multi-scale 90+45+BiomedCLIP-TTA via v2 recipe. Red-team bootstrap P(>0)=0.999 on fair comparison.

### Red-team findings (from `reports/RED_TEAM_ENSEMBLE_AUDIT.md`)
- Original 0.6698 had stacked leakage: threshold sweep + subset selection both done on same 240-row OOF.
- Fixed subset `dinov2_b + biomedclip` + nested-CV thresholds: **0.6528** — honest +0.038 over baseline.
- Fully nested (both subset & threshold inner): 0.5847 — below baseline. Means the 2-encoder subset is a fragile choice that only holds because we fixed it by hand; but it's independently justified by standalone F1 leaderboard.
- Eye-level vs person-level: only ~0.002 difference in nested numbers — grouping is not the main issue.

**Shippable models:**
- `models/dinov2b_tiled_v1/` — single-model baseline, 0.615 expected.
- Ensemble: not yet packaged (TODO in Round 3).

**Candidate ensemble predictions:** `cache/best_ensemble_predictions.npz` (0.6346 raw argmax, 0.6528 nested-threshold).

---

## Per-class ceiling so far (person-level LOPO)

| Class | Best F1 | Support | Issue |
|---|---:|---:|---|
| ZdraviLudia | 0.82 | 70 | ✓ solid |
| SklerozaMultiplex | 0.66 | 95 | confuses with Glaukom |
| PGOV_Glaukom | 0.55 | 36 | confuses with SM; TDA helps +16% |
| Diabetes | 0.50 | 25 | better with tiling |
| **SucheOko** | **0.07** | 14 | **2 patients — fundamental ceiling** |

---

## Ledger of tried experiments

| Experiment | Status | F1 weighted | F1 macro | Notes |
|---|---|---:|---:|---|
| Handcrafted (GLCM+LBP+fractal+HOG) + XGB | ✓ eye-LOPO | 0.502 | 0.344 | 94 features |
| DINOv2-S single crop + LR | ✓ eye-LOPO | 0.582 | 0.451 | |
| DINOv2-B single crop + LR | ✓ eye-LOPO | 0.581 | 0.430 | |
| BiomedCLIP single crop + LR | ✓ eye-LOPO | 0.577 | 0.441 | |
| DINOv2-S tiled (tile-LR + mean-proba) | ✓ eye-LOPO | 0.628 | 0.501 | |
| DINOv2-B tiled scan-mean + LR | ✓ **person-LOPO** | **0.615** | **0.491** | current champion |
| DINOv2-S tiled scan-mean + LR | ✓ person-LOPO | 0.593 | 0.478 | |
| BiomedCLIP tiled scan-mean + LR | ✓ person-LOPO | 0.580 | 0.434 | |
| TDA (cubical PH, 1015 dims) + XGB | ✓ eye-LOPO | 0.531 | 0.374 | |
| TDA + DINOv2-S concat + LR | ✓ eye-LOPO | 0.589 | — | noisy concat; proba-average better? |
| Label-shuffle null baseline | ✓ eye-LOPO | 0.276±0.042 | — | validated — signal is real |
| CGNN (graph from skeleton → GINE), CPU retry | ✓ person-LOPO | 0.365 | 0.220 | below baseline; graph-alone insufficient |
| Proba-ensemble (DINOv2-B + BiomedCLIP + thresholds, LEAKY) | ✗ rejected | ~~0.6698~~ | ~~0.506~~ | threshold leakage |
| **Proba-ensemble + NESTED thresholds** | ✓ **person-LOPO** | **0.6528** | — | honest new champion |
| Production-Optimizer (still running) | ⏳ | ? | ? | |

---

## Key domain insights

1. **Tiling > single crop** (+5 % F1). We use 9 non-overlapping 512² tiles per scan.
2. **Encoder size irrelevant.** DINOv2-S ≈ DINOv2-B ≈ BiomedCLIP. Small data ceiling.
3. **TDA is per-class specialist.** Boosts Glaukom via H₁ loop features at coarse scale. Doesn't help overall when flat-concatenated.
4. **Patient = dominant latent variable.** UMAP shows patient clusters stronger than class clusters.
5. **SucheOko ceiling = 2 patients.** Not a model problem, a data problem.
6. **L/R eyes of same person** were NOT merged in original parser. Fixed via `person_id()` — F1 drops ~1 % (real signal but was ~1 % inflated).

---

## Queue (prioritized by expected F1 gain × probability)

### ROUND 2 — dispatch after Production-Optimizer returns

| # | Hypothesis | EV | Agent type | Status |
|---|---|---|---|---|
| 1 | Synthesize 3 components (DINOv2-B + TDA probs + Production-Opt best) via probability-weighted voting | HIGH | Synthesizer | pending |
| 2 | Red-team: cherry-pick check on Round 1 winners — re-eval with held-out 5 patients, not LOPO | HIGH | Validator | pending |
| 3 | CGNN retry on CPU with 40 epochs | MED | Implementer | pending |
| 4 | LLM-reasoning layer: JSON-out classifier with domain knowledge prompting | MED-HIGH | Specialist (pitch killer) | infra complete, blocked on `ANTHROPIC_API_KEY` — see `reports/LLM_REASON_RESULTS.md` |
| 5 | SucheOko-specific rescue: 50x augmentation per-scan, 1-vs-rest binary detector | MED | Specialist | pending |
| 6 | Probability-averaging ensemble (not concat): DINOv2 softmax + TDA softmax × weights | MED-HIGH | Implementer | pending |

### ROUND 3 — after Round 2 convergence

| # | Task | When |
|---|---|---|
| 1 | Final submission ensemble | Round 2 done |
| 2 | Pitch narrative + slide deck | T-6h |
| 3 | Submission format validation | T-3h |
| 4 | Test on synthetic held-out | T-3h |

---

## Agents launched (Round 1)

| Agent | Role | Status | Return |
|---|---|---|---|
| Production-Optimizer | implementer/ensemble | ⏳ running | pending |
| TDA-Researcher | implementer | ✓ done | 0.531 TDA-alone, 0.589 concat |
| Validator | red-team | ✓ done | found L/R eye bug — fixed |
| Pitch-Visualizer | designer | ✓ done | 6 figures in `reports/pitch/` |

## Agents launched (Round 2)

| Agent | Role | Status | Return |
|---|---|---|---|
| Probability-Ensemble | implementer | ✓ done | claimed 0.6698 → rejected, honest 0.6528 |
| CGNN-CPU-retry | implementer | ✓ done | 0.365 person-LOPO — below baseline, keep for interpretability only |
| LLM-reasoning (API) | specialist | ⨯ blocked | no API key — superseded by Round 3 LLM-gated agent |
| Red-Team (ensemble audit) | validator | ✓ done | honest F1 = 0.6528, not 0.6698 |

## Agents launched (Round 3 — dynamic specialists)

| Agent | Role | Status | Return |
|---|---|---|---|
| Cascade-Specialists | implementer | ✓ done | **cascade HURTS** (-0.048) — binary specialists strong standalone (0.78-0.96) but override flips correct-low-confidence → wrong. Soft-blend/stacker is right direction. |
| Cascade-Stacker (soft-blend) | implementer | ✓ done | **+0.0105 over raw argmax** (0.6451 vs 0.6346), α=0.90 consistent across nested folds. Meta-LR/XGB overfit. Small honest gain; not enough to justify deployment complexity. |
| LLM-Gated-Reasoner (CLI) | specialist | ✓ done | **F1 override HURTS** (-0.012). But reasoning texts + architecture are pitch gold. 47 cases processed, $0 marginal cost. |
| Red-Team v2 (Production-Optimizer) | validator | ⏳ running | pending — auditing 0.6878 claim from Production-Optimizer |

### Cascade findings (insightful negative result)
- Specialist F1 (standalone binary, person-LOPO):
  - Glaukom vs SM: 0.78
  - Diabetes vs Healthy: 0.81
  - SM vs Healthy: 0.96 (bonus)
- Hard-override cascade @ thr=0.65: 0.6217 (vs Stage-1 0.6698 claimed, 0.6528 honest)
- **Root cause**: low Stage-1 confidence ≠ Stage-1 wrong. Of 24 routed to spec A, Stage-1 already correct on 18/24. Specialist flipped 10 of those to wrong.
- **Correct takeaway**: binary specialist as FEATURE not OVERRIDE (stacker / soft-blend). Not implemented yet.

## Production-Optimizer (Round 1) — red-teamed

| Metric | Claimed | Honest nested | Status |
|---|---|---|---|
| F1 weighted (eye-LOPO) | 0.6878 | 0.6403 | ✗ inflated −0.048 |
| F1 weighted (person-LOPO) | — | **0.6326** | ✗ WORSE than 0.6528 champion |

**Outcome:** 4-component concat + log-prob bias tuning **regresses** vs 2-component ensemble. Simpler ensemble wins. Occam's razor confirmed.

Meta-insight across all red-teams: at 240 scans we are at data ceiling. Bias/threshold tuning on OOF consistently inflates by +0.04–0.06. Only SIMPLE methods + NESTED CV give honest numbers.

---

## Strategic decisions taken

- **person_id over patient_id** — after Validator flagged eye leakage
- **DINOv2-B over DINOv2-S** — marginally better, same cost
- **Tiled over single crop** — +5 % F1 consistently
- **Skip BMP preview path** — raw SPM is strictly better (no watermark, higher resolution)
- **Submit 1 model first, ensemble second** — baseline always better than over-engineered

---

## Blocked / parked

- **CGNN on MPS** — PyTorch Geometric GINEConv backward on MPS hangs. Park until Round 2 CPU retry.
- **9-class scenario** — PDF slides mentioned Alzheimer/bipolar/panic/cataract/PDS but TRAIN_SET has only 5 classes. If test set has more, need open-set strategy. For now assume 5-class.

---

## Invariants I commit to as orchestrator

1. **Never do 30+ min gruntwork myself.** Delegate.
2. **Every win gets red-teamed** before adoption.
3. **Update this file after every round.** Not optional.
4. **Parallelize independent work.** Serial only when blocked.
5. **T-6h cutoff: stop exploring, start consolidating.** Pitch > optimal F1.

## ⭐ Wave 7 multi-scale D — APPROVED by red-team (NEW CHAMPION CANDIDATE)

- Config D: DINOv2-B 90 nm + DINOv2-B 45 nm + BiomedCLIP-TTA via v2 recipe (L2-norm + StandardScaler + LR + geom-mean)
- Honest F1: **0.6887 weighted, 0.5541 macro** (+0.033 weighted, +0.016 macro over v2 champion 0.6562)
- Red-team bootstrap (B=1000, person-level):
  - vs v2-champion (TTA-ful): Δ=+0.0325, 95% CI [-0.005, +0.063], P(>0) = 0.949 (borderline)
  - **vs v2-noTTA (fair apples-to-apples): Δ=+0.039, CI [+0.010, +0.069], P(>0) = 0.999** — strictly positive
- Per-class broad-base lifts: Healthy +0.048, Diabetes +0.042, SM +0.040, Glaukom +0.015, SucheOko -0.065 (one scan)
- Report: `reports/RED_TEAM_MULTISCALE.md`
- **VERDICT: SHIP as v4 champion.**
- v4 TTA agent running in parallel; will add D4 TTA to 90 nm + 45 nm branches (should widen lead further).

## Wave 6 multichannel E7 — REJECTED by red-team

- Claim: 0.6645 (+0.0083 over v2 champion)
- Bootstrap 95% CI for ΔF1: [−0.04, +0.05] → crosses zero
- P(gain > 0) = 0.598 (essentially coin flip)
- 158% of point gain from Diabetes (36 scans); Glaukom −0.057, SucheOko 1→0
- Report: `reports/RED_TEAM_E7_BOOTSTRAP.md`
- **Champion stays at v2 (0.6562).** Red-team saved us from shipping noise.

---

## Error analysis findings (post-TTA, from `reports/ERROR_ANALYSIS.md`)

Of 85 misclassified scans out of 240:
- **Mode C (near-miss, truth = rank 2): 58 (68%)** — model is on decision boundary, not wildly wrong
- **Mode B (high-conf wrong): 15** — 12/15 the embedding legitimately puts the scan in wrong cluster; cannot be fixed by changing head
- **Mode D (catastrophic, truth rank ≥ 4): 6** — 4/6 are SucheOko
- **Mode A (low-conf): 1**
- **SucheOko contributes 57% of non-near-miss errors** — 2-patient ceiling dominates
- **Diabetes → Healthy session-level failure** at one specific session (`DM_01.03.2024` LO/PO eyes)
- **Realistic upper bound from further interventions: +0.005 to +0.01** — we are at the data ceiling, not model ceiling

---

## Key meta-insights (orchestration win)

After 3+ round × ~8 agents, the emergent pattern is crystal clear:

1. **240 scans = data ceiling.** No amount of clever ensemble/tuning pushes honest F1 past ~0.66.
2. **Complexity inflates leakage, not signal.** Every claim > 0.65 turned out to be stacked threshold/bias/subset tuning on same OOF. After nesting: regresses to ~0.63 or below.
3. **Binary specialists are strong standalone** (0.78–0.96 F1) but **destructive as hard overrides** because low Stage-1 confidence ≠ Stage-1 wrong.
4. **Simple beats fancy.** DINOv2-B + BiomedCLIP proba-avg (2 components, no bias tuning) is the Pareto front.
5. **Red-team discipline saved us.** Would have pitched 0.67 → verified 0.65. Credibility.
6. **Negative results are pitch-viable** when you frame them as "we searched honestly".
