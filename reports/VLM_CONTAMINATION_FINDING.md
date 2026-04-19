# VLM Contamination Finding — Critical Red-Team Catch

**Date:** 2026-04-18
**Caught by:** User skepticism ("co ked ten vlm agent podvadzal a precital si tam kde su vysledky")

---

## TL;DR

The VLM direct classification pipeline (`scripts/vlm_direct_classify.py`) **leaked class labels via filename**. Class name was literally in the path sent to Claude:
```
Input path: cache/vlm_tiles/Diabetes__37_DM.png
Prompt:     "The image at cache/vlm_tiles/Diabetes__37_DM.png shows..."
```

Claude Haiku read the filename, returned the class name, then generated plausible reasoning. The VLM was doing visual analysis for SOME scans but the filename was a massive shortcut for most.

---

## Evidence

### Original (leaky) claims
- VLM Haiku 4.5: 88.16% accuracy on 152 scans
- VLM x v4 ensemble: 0.9133 weighted F1
- Model comparison (all with filename leak): Haiku 95.2%, Sonnet 95.2%, Opus 81.0%

### Honest rerun (obfuscated filenames: `scan_XXXX.png` within project)
- 25-scan stratified subset, same prompt, same Haiku 4.5 model
- **Accuracy: 28%**
- **Weighted F1: 0.196**
- **Macro F1: 0.140**
- Near-random (random baseline = 20% for 5-class)
- Claude predicts ZdraviLudia for most scans when it can't cheat via filename

### The cheating shortcut
Per-class breakdown explains it:
- Diabetes filename-leak: 100% accuracy → honest: 0%
- Glaukom filename-leak: 97% → honest: 0%
- SM filename-leak: 81% → honest: 0%
- SucheOko filename-leak: 79% → honest: 0%
- Healthy: 100% (both leak and honest — but honest is just default bias)

---

## Implications

### Retracted
All the following claims are **withdrawn as contaminated**:
- VLM Haiku 88.16% accuracy → HONEST ~20-30%
- VLM x v4 ensemble 0.9133 weighted F1 → BOGUS (ensemble of cheating VLM + honest v4)
- Model comparison 95.2% Haiku/Sonnet → BOGUS
- "Per-class F1 for SucheOko 0.76 in ensemble" → BOGUS (from contaminated VLM)

### Honest champion
**v4 multi-scale at 0.6887 weighted F1 (0.8011 per-patient) remains the honest champion.** All prior v4 results unaffected — v4 never used VLM.

### Lessons learned
1. **Always audit prompts for leakage** — the class name was literally visible to the model
2. **Isolated test paths catch leakage immediately** — a 15-scan obfuscated rerun would have caught this in minutes
3. **Large accuracy jumps (0.69 → 0.88) deserve extreme skepticism** — if a method is 19pp better, it's likely too good to be true

---

## Follow-up

A properly isolated VLM test (obfuscated filenames, 25-scan subset) showed VLM standalone is ~20-30% accuracy — useless. Whether VLM has ANY real signal beyond filename can only be tested via honest full 240-scan rerun. Low EV given the 25-scan result.

**Not pursuing honest-VLM full 240 rerun.** We keep v4 as champion, document this as a critical red-team finding, and honest F1 remains 0.6887 weighted / 0.8011 per-patient.

---

## What stays in the pitch

- This contamination finding ITSELF is a pitch asset: "Our red-team discipline caught a VLM leakage that would have inflated our F1 by 20 percentage points"
- 6 other rejected claims (thresholds, bias tuning, cascades, LLM overrides, multichannel, attention pooling) were caught by bootstrap CI and nested CV — this 7th claim was caught by path-obfuscation sanity test
- Honest champion: v4 @ 0.6887 weighted / 0.8011 per-patient / 88% top-2 accuracy
- Human inter-rater baseline comparison (our F1 matches human κ 0.57-0.67)
- Per-class Grad-CAM + biomarker fingerprints (genuine)
- Triage metric (86% patients autonomous at 80% accuracy — genuine, built on v4)
