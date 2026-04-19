> [!WARNING]
> **CONTAMINATED — DO NOT CITE.** This report used `cache/vlm_few_shot_collages/<CLASS>__<scan>.png` paths whose filename leaked the class label to the VLM. Caught by red-team audit `reports/RED_TEAM_SONNET_0_8873.md` on 2026-04-18.
> Honest replacement: `reports/VLM_SONNET_HONEST.md` (Sonnet honest wF1 = 0.3424, inflation +0.545).
> Leakage prevention infra: `teardrop/safe_paths.py` + `reports/LEAKAGE_PREVENTION.md`.

---

# VLM Few-Shot on FULL 240 Scans (Wave 7 candidate)

Generated: 2026-04-18T21:22:34

## Setup

- Retrieval: DINOv2-B (`cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`), cosine sim, **person-disjoint** anchors.
- k=2 anchors per class -> 10-anchor collage + QUERY tile.
- VLM: `claude-haiku-4-5`, `claude -p --output-format json` via CLI subprocess.
- Parallelism: 8 worker threads.
- Scans scored: 240 / 240.
- Total API cost: $3.966
- Wall-clock time: 43.9 s

## Scan-level metrics

- Accuracy: **0.6667**
- **Weighted F1: 0.6755**
- Macro F1: 0.5925
- Mean confidence: 0.816

## Per-class F1

| Class | P | R | F1 | Support |
|---|---|---|---|---|
| ZdraviLudia | 0.663 | 0.929 | 0.774 | 70 |
| Diabetes | 0.364 | 0.640 | 0.464 | 25 |
| PGOV_Glaukom | 0.769 | 0.556 | 0.645 | 36 |
| SklerozaMultiplex | 1.000 | 0.558 | 0.716 | 95 |
| SucheOko | 0.316 | 0.429 | 0.364 | 14 |

## Confusion matrix (rows=true, cols=pred)

| true\pred | ZdraviLudi | Diabetes | PGOV_Glauk | SklerozaMu | SucheOko |
|---|---|---|---|---|---|
| ZdraviLudi | 65 | 5 | 0 | 0 | 0 |
| Diabetes | 9 | 16 | 0 | 0 | 0 |
| PGOV_Glauk | 6 | 8 | 20 | 0 | 2 |
| SklerozaMu | 13 | 12 | 6 | 53 | 11 |
| SucheOko | 5 | 3 | 0 | 0 | 6 |

## Per-person aggregation (majority vote across scans of same person)

- N persons: 35
- Accuracy: 0.7429
- Weighted F1: 0.7466
- Macro F1: 0.6998

## Per-patient-eye aggregation (majority vote across scans of same eye)

- N patient-eyes: 44
- Accuracy: 0.7955
- Weighted F1: 0.7984
- Macro F1: 0.7225

## Bootstrap 1000x vs v4 baseline (weighted F1 = 0.6887)

- VLM weighted F1 mean: 0.6753  (95% CI [0.6159, 0.7360])
- Delta (VLM - v4) mean: **-0.0134**  (95% CI [-0.0728, +0.0473])
- **P(Delta > 0) = 0.325**

## Comparison vs 40-scan subset (`reports/VLM_FEW_SHOT_RESULTS.md`)

| Run | N | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|---|
| 40-scan subset (stratified 8/class, first-scan-per-person) | 40 | 0.8000 | ~0.80 (implied) | 0.7974 |
| **Full 240** | 240 | **0.6667** | **0.6755** | **0.5925** |

### Why the drop (macro F1 0.797 -> 0.592)?

The 40-scan subset was **easy by construction**: `stratified_person_disjoint` with `per_class=8` picks a balanced 8 scans per class drawn round-robin from distinct persons, typically the *first* scan per person. Those are "canonical" scans — well-scanned, representative exemplars.

The full 240 run exposes:

- **Within-person variance.** Single persons contribute many scans (e.g. `1-SM-LM-18` contributes ~15 SM scans, `Sklo-No2` a long series). The nearest DINOv2 anchors for these *from a different person* are often visually less similar, so the VLM falls back to more generic "fine ferning -> ZdraviLudia" or "coarse -> Diabetes" priors.
- **Class imbalance (7:1 SM:SucheOko).** Weighted F1 is now dominated by SM (95 scans) and ZdraviLudia (70); SucheOko (14) contributes little weight, so we don't get the 40-scan subset's per-class parity.
- **SucheOko anchor pool collapse.** With only **2 SucheOko persons** and the strict person-disjoint rule, every SucheOko query sees the SAME single other person as its two anchors. The VLM has effectively one exemplar for that class, badly hurting retrieval-based recall.

### Reasoning coherence

- `most_similar_anchor_class` matches `predicted_class` in **100%** of calls (structured prompt honored).
- But `most_similar_anchor_class == true_class` only **55 - 93%** of the time:
  | Class | Anchor-class-picked-is-true |
  |---|---|
  | ZdraviLudia | 92.9% (65/70) |
  | Diabetes | 64.0% (16/25) |
  | PGOV_Glaukom | 55.6% (20/36) |
  | SklerozaMultiplex | 55.8% (53/95) |
  | SucheOko | 42.9% (6/14) |

This is consistent with the DINOv2 top-2 nearest-neighbor actually being from the wrong class for harder query scans, **not** the VLM misreading the collage.

### Confidence vs correctness

- Low-conf (<0.75) wrong: **21**
- High-conf (>=0.85) wrong: **21**

The model is *not* well-calibrated: overconfident errors are as common as low-conf errors. This is actually useful for ensemble fusion — raw softmaxes from v4 should be temperature-scaled and mixed with binary VLM votes rather than VLM confidences used as weights.

## Next steps for ensemble fusion

The per-query confidences and `anchors` metadata are saved in `cache/vlm_few_shot_full_predictions.json`. Ready to:

1. Project VLM predictions into one-hot or soft (confidence * one-hot + (1-confidence)/5) probability vectors.
2. Fuse with v4 softmaxes via logistic-regression stacker or simple geometric mean.
3. Because per-person / per-eye aggregation already lifts us to **weighted F1 0.7466 / 0.7984**, a simple "majority vote at person level + v4 fallback for SucheOko" rule could push scan-level F1 north of v4 alone.
