> [!WARNING]
> **CONTAMINATED — DO NOT CITE.** This report used `cache/vlm_few_shot_collages/<CLASS>__<scan>.png` paths whose filename leaked the class label to the VLM. Caught by red-team audit `reports/RED_TEAM_SONNET_0_8873.md` on 2026-04-18.
> Honest replacement: `reports/VLM_SONNET_HONEST.md` (Sonnet honest wF1 = 0.3424, inflation +0.545).
> Leakage prevention infra: `teardrop/safe_paths.py` + `reports/LEAKAGE_PREVENTION.md`.

---

# VLM Model Comparison: Haiku 4.5 vs Sonnet 4.6 vs Opus 4.7

Task: direct-image AFM tear-droplet classification (5 classes). Prompt and rendering identical to `scripts/vlm_direct_classify.py` baseline. Only variable is the `--model` flag passed to `claude -p`.

Subset: stratified, person-disjoint (seed=42, per_class=5). Given the current dataset the subset yields **21 scans** (not 25) because SucheOko has only 2 unique persons and Diabetes has 4.

## Summary

| Model | n | Acc | F1 macro | Mean conf | Mean cost/scan | Mean lat (s) | Subset cost | Full-set projection (240) |
|---|---|---|---|---|---|---|---|---|
| haiku (claude-haiku-4-5) | 21 | 0.952 | 0.960 | 0.82 | $0.0161 | 16.7 | $0.338 | $3.86 |
| sonnet (claude-sonnet-4-6) | 21 | 0.952 | 0.915 | 0.75 | $0.0305 | 10.7 | $0.639 | $7.31 |
| opus (claude-opus-4-7) | 21 | 0.810 | 0.777 | 0.69 | $0.0628 | 11.3 | $1.319 | $15.07 |

## Per-class F1

| Class | Haiku F1 | Sonnet F1 | Opus F1 |
|---|---|---|---|
| ZdraviLudia | 0.909 | 0.909 | 0.833 |
| Diabetes | 1.000 | 1.000 | 1.000 |
| PGOV_Glaukom | 1.000 | 1.000 | 0.750 |
| SklerozaMultiplex | 0.889 | 1.000 | 0.800 |
| SucheOko | 1.000 | 0.667 | 0.500 |

### Confusion matrix — haiku

(rows = true, cols = predicted)

| true \ pred | Zdravi | Diabet | PGOV_G | Sklero | SucheO |
|---|---|---|---|---|---|
| Zdravi | 5 | 0 | 0 | 0 | 0 |
| Diabet | 0 | 4 | 0 | 0 | 0 |
| PGOV_G | 0 | 0 | 5 | 0 | 0 |
| Sklero | 1 | 0 | 0 | 4 | 0 |
| SucheO | 0 | 0 | 0 | 0 | 2 |

### Confusion matrix — sonnet

(rows = true, cols = predicted)

| true \ pred | Zdravi | Diabet | PGOV_G | Sklero | SucheO |
|---|---|---|---|---|---|
| Zdravi | 5 | 0 | 0 | 0 | 0 |
| Diabet | 0 | 4 | 0 | 0 | 0 |
| PGOV_G | 0 | 0 | 5 | 0 | 0 |
| Sklero | 0 | 0 | 0 | 5 | 0 |
| SucheO | 1 | 0 | 0 | 0 | 1 |

### Confusion matrix — opus

(rows = true, cols = predicted)

| true \ pred | Zdravi | Diabet | PGOV_G | Sklero | SucheO |
|---|---|---|---|---|---|
| Zdravi | 5 | 0 | 0 | 0 | 0 |
| Diabet | 0 | 4 | 0 | 0 | 0 |
| PGOV_G | 0 | 0 | 3 | 1 | 1 |
| Sklero | 1 | 0 | 0 | 4 | 0 |
| SucheO | 1 | 0 | 0 | 0 | 1 |

## Pairwise agreement

| Pair | n | agree | rate |
|---|---|---|---|
| haiku vs sonnet | 21 | 19 | 90.48% |
| haiku vs opus | 21 | 18 | 85.71% |
| sonnet vs opus | 21 | 18 | 85.71% |

## Majority-vote ensemble (all 3 models)

- n = 21  acc = 0.905  F1 macro = 0.878
- unanimous (all 3 agree): 17/21  split: 4

| Class | Ensemble F1 |
|---|---|
| ZdraviLudia | 0.833 |
| Diabetes | 1.000 |
| PGOV_Glaukom | 1.000 |
| SklerozaMultiplex | 0.889 |
| SucheOko | 0.667 |

## Qualitative reasoning samples

### haiku

**Top-confidence CORRECT**:
- `TRAIN_SET/ZdraviLudia/6L.000` (ZdraviLudia, conf 0.88): Dense dendritic ferning with uniform, fine branching throughout the field of view; regular crystalline organization with minimal fragmentation or granularity; high fractal complexity consistent with healthy tear electrolyte composition (D ~1.75).
- `TRAIN_SET/ZdraviLudia/8L.000` (ZdraviLudia, conf 0.88): Dense, hierarchical dendritic ferning with uniform branching throughout—characteristic of healthy tear morphology. Clear fractal structure with fine sub-branches on main dendrites, no fragmentation or chaotic granularity.
- `TRAIN_SET/ZdraviLudia/48.001` (ZdraviLudia, conf 0.88): Dense, uniform dendritic ferning with consistent fine branching throughout the field, characteristic of Masmali grade 0-1 healthy morphology. High organizational symmetry and estimated fractal dimension ~1.77 consistent with healthy tear crystallization.

**Top-confidence WRONG**:
- `TRAIN_SET/SklerozaMultiplex/20_LM_SM-SS.001` true=SklerozaMultiplex pred=ZdraviLudia (conf 0.72): Dense, well-organized dendritic ferning with uniform hierarchical branching and fine crystalline structures throughout, consistent with Masmali grade 0–1 healthy tear morphology. Lacks the fragmentation of dry eye, granularity of glaucoma, or distinct heterogeneous coarse/fine regions characteristic of MS.

### sonnet

**Top-confidence CORRECT**:
- `TRAIN_SET/ZdraviLudia/6L.000` (ZdraviLudia, conf 0.91): The scan shows dense, highly ordered dendritic ferning with uniform, fine branching throughout the entire field of view — consistent with Masmali grade 0–1 and fractal D ~1.75–1.85. There are no granular, amorphous, or fragmented regions that would suggest pathology.
- `TRAIN_SET/ZdraviLudia/7.000` (ZdraviLudia, conf 0.91): The scan shows dense, radially symmetric dendritic ferning with fine, uniform branching throughout the entire field of view — a classic Masmali grade 0–1 pattern consistent with healthy tear crystallization. No fragmentation, granularity, loops, or coarse rods are present.
- `TRAIN_SET/ZdraviLudia/8L.000` (ZdraviLudia, conf 0.91): The scan shows dense, well-developed dendritic ferning with uniform hierarchical branching and fine secondary/tertiary dendrites throughout the entire field of view, consistent with Masmali grade 0-1 and fractal D ~1.75-1.85. No fragmentation, amorphous regions, granular loops, or coarsening indicative of pathology is present.

**Top-confidence WRONG**:
- `TRAIN_SET/SucheOko/29_PM_suche_oko.000` true=SucheOko pred=ZdraviLudia (conf 0.62): The scan shows well-developed dendritic ferning with dense, uniform branching radiating from a central nucleation point, consistent with Masmali grade 0-1 healthy tear morphology. The network is continuous and structured rather than fragmented or granular, arguing against dry eye or glaucoma; however, the bright central overexposure makes fractal dimension estimation uncertain.

### opus

**Top-confidence CORRECT**:
- `TRAIN_SET/ZdraviLudia/7.000` (ZdraviLudia, conf 0.90): Dense, uniform radial dendritic ferning emanating from a central nucleation point with fine, regular branching and no fragmented or granular regions — consistent with healthy tear crystallization (Masmali grade 0-1, high fractal dimension).
- `TRAIN_SET/ZdraviLudia/6L.000` (ZdraviLudia, conf 0.88): Dense, uniform dendritic ferning with fine symmetrical branching and high fractal complexity across the field; no fragmentation, loops, or coarse heterogeneity — consistent with healthy Masmali grade 0-1.
- `TRAIN_SET/ZdraviLudia/8L.000` (ZdraviLudia, conf 0.82): Dense, well-organized dendritic ferning with uniform primary trunks and regular secondary branching filling the field of view; no fragmentation or amorphous voids, consistent with healthy Masmali grade 0-1 and high fractal dimension.

**Top-confidence WRONG**:
- `TRAIN_SET/PGOV_Glaukom/21_LV_PGOV+SII.000` true=PGOV_Glaukom pred=SucheOko (conf 0.62): Scan shows fragmented, isolated bright blobs on a largely empty amorphous background with no connected dendritic network — consistent with Masmali grade 3-4 sparse ferning typical of dry eye. Low fractal dimension, no coherent branching.
- `TRAIN_SET/SucheOko/29_PM_suche_oko.000` true=SucheOko pred=ZdraviLudia (conf 0.60): Dense radial dendritic ferning with uniform fine branching emanating from a bright central nucleation point; branches appear thin and well-developed without fragmentation or amorphous gaps, consistent with Masmali grade 0-1 and high fractal dimension typical of healthy tears.
- `TRAIN_SET/PGOV_Glaukom/23_LV_PGOV.000` true=PGOV_Glaukom pred=SklerozaMultiplex (conf 0.55): Heterogeneous morphology within the scan: coarse elongated bright ridges coexist with dense fine granular speckle in adjacent regions, indicating high intra-sample variance typical of MS rather than uniform ferning or fragmented dry-eye network.

## Cost analysis

Full-dataset projection uses the observed mean cost/scan for each model × 240 (approx size of TRAIN_SET scan count). Claude CLI adds a ~$0.04-0.15 per-invocation harness cost (visible in `total_cost_usd`) regardless of model, so real API costs are lower than the numbers below if a proper batched API call were used instead.

## 2-model ensemble (Haiku + Sonnet only, drops Opus)

- n = 21  acc = 0.952  F1 macro = 0.960
- unanimous: 19/21  split: 2

## Recommendation

- Haiku acc = 0.952 F1 = 0.960 @ $0.0161/scan
- Sonnet acc = 0.952 F1 = 0.915 @ $0.0305/scan   (Δ acc = +0.000)
- Opus   acc = 0.810 F1 = 0.777 @ $0.0628/scan   (Δ acc = -0.143)
- 3-way ensemble acc = 0.905

**Verdict: Haiku wins. Do NOT scale up.** The stronger models regress on this task: Opus -0.143, Sonnet +0.000. Haiku already sits at the ceiling for this prompt + fixed 512 px central-tile render. Stronger models are more skeptical of the class-biology cues in the prompt (Opus in particular flags 'overexposed centers' and calls out uncertainty instead of committing), which looks like better calibration but costs accuracy here. Worst regressor: opus (-0.143).

**Ensemble note:** the 3-model majority vote (0.905) does NOT beat the best single model (0.952). Opus drags the ensemble down: its errors inject disagreement on cases Haiku and Sonnet got right. A 2-model (Haiku + Sonnet) ensemble would likely be better, but adds 2x the cost of Haiku alone for no acc gain over Haiku solo.
