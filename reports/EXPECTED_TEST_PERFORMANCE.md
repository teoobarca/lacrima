# Expected test-time performance — scenario analysis

The hidden test set from organizers is unknown. This document predicts model behavior for several plausible scenarios.

## Scenario 1: Same 35 persons, new scans (temporal split)

- Expected F1: **~0.66 weighted, ~0.54 macro** (matches person-LOPO)
- Best case. Person identity priors align perfectly.
- Same per-class pattern: ZdraviLudia ~0.87, SM ~0.65, Diabetes ~0.54, Glaukom ~0.56, SucheOko ~0.06.

## Scenario 2: New persons, same 5 classes (strict person-disjoint)

- Expected F1: **0.58 – 0.66** (wide interval from nested LOPO analysis)
- Per `reports/VALIDATION_AUDIT.md` repeated-5-fold variance was 0.64 ± 0.08 → 1σ below is 0.56.
- SucheOko: will remain near-zero F1 — 2 training persons cannot generalize.
- Diabetes: depends on whether test Diabetes persons are systemically similar to our 4 training Diabetes persons.

## Scenario 3: New persons + new subpopulations (distribution shift)

- Expected F1: **0.50 – 0.62**
- Example shifts: different scanner, different sample-prep protocol, different season (protein hydration varies), different age range.
- Preprocessing pipeline (plane-level + resample + robust-normalize) is designed to absorb scanner and sample-size variation.
- TTA D4 + L2-normalization should help with test-time variance.
- BUT: if BioClips/amplitude/phase channels are not available at test time, multichannel agent findings don't transfer (we already ship non-multichannel v2).

## Scenario 4: 9 classes (PDF's original scope includes Alzheimer, Bipolar, Panic, Cataract, PDS)

- Expected F1: **UNKNOWN — our model outputs only 5 classes**.
- We'd need an open-set strategy: threshold on max-softmax, route unknown classes to "abstain".
- `reports/LLM_GATED_RESULTS.md` has precedent for gating on uncertainty; the `predict_proba` output could feed into an external "is this one of the 5 known classes?" rule.
- Rough intuition: the 5 classes in TRAIN_SET likely cover ~70% of the test set if organizers use the same patient pool; the other 30% would need graceful abstention.

## Scenario 5: Test set is BMP (rendered images) not raw SPM

- Expected F1: **~0.40 – 0.55**
- Our pipeline is built for raw SPM. BMP has the scale-bar + axis labels burned-in → data leakage risk (models learn axis markings, not biology).
- `preprocess_spm` in `teardrop/data.py` expects SPM → won't read BMP directly.
- Fix: add BMP path in `preprocess_bmp` that crops the central data region (discards axis labels) and feeds to the same encoder.
- **Recommend: ask organizers which format first.** Building the BMP fallback is ~30 min of work if needed.

## Scenario 6: Extreme imbalance or different class distribution

- If test has 1 SucheOko patient with 5 scans: F1 still ~0 for that class.
- If test has no SucheOko: macro F1 normalization changes; weighted F1 likely goes up.
- If test has more Diabetes: weighted F1 may benefit slightly (Diabetes F1 = 0.54 is better than Glaukom 0.56 but worse than SM 0.65).

## Scenario 7: Patient-incremental retraining allowed

- If organizers allow us to see test SPM files WITHOUT labels and retrain:
  - Could do semi-supervised adaptation: pseudo-label test → fine-tune LR
  - Expected additional gain: +0.02 – 0.05 depending on distribution shift
- If organizers allow FEW test labels (e.g., 5 per class):
  - Meta-learning / few-shot tuning possible
  - Expected additional gain: +0.05 – 0.10
- Not implementing these unless rules explicitly allow.

## Our 3 shipped model tiers

| Bundle | F1 expected | Inference time per scan | When to use |
|---|---:|---:|---|
| `models/ensemble_v2_tta/` ★ | 0.6562 | ~45 s | **Default shipped champion** |
| `models/ensemble_v1_tta/` | 0.6458 | ~8 s | If v2 predict.py fails at test time |
| `models/ensemble_v1/` | 0.6346 | ~1 s | If TTA infrastructure unavailable |
| `models/dinov2b_tiled_v1/` | 0.6150 | ~0.5 s | Single-encoder fallback |

All four load via the same `predict_cli.py --model <dir>` interface.

## Submission checklist

- [x] Shipped model at `models/ensemble_v2_tta/`
- [x] Backup bundles (v1_tta, v1, single)
- [x] CLI entry-point `predict_cli.py`
- [x] CSV output schema documented in `SUBMISSION.md`
- [x] Reproducible from scratch via `REPRODUCE.md`
- [x] Git-committed scripts + models + reports (16 commits)
- [x] Interactive demo `app.py` (Gradio, pitch artifact)
- [ ] Pending: organizer's test-set format clarification (SPM vs BMP)
- [ ] Pending: organizer's class set (5 vs 9)
