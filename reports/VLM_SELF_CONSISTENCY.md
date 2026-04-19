> [!WARNING]
> **CONTAMINATED — DO NOT CITE.** This report used `cache/vlm_few_shot_collages/<CLASS>__<scan>.png` paths whose filename leaked the class label to the VLM. Caught by red-team audit `reports/RED_TEAM_SONNET_0_8873.md` on 2026-04-18.
> Honest replacement: `reports/VLM_SONNET_HONEST.md` (Sonnet honest wF1 = 0.3424, inflation +0.545).
> Leakage prevention infra: `teardrop/safe_paths.py` + `reports/LEAKAGE_PREVENTION.md`.

---

# VLM Self-Consistency Voting — Results

Generated: 2026-04-18T21:35:07

## Setup

- Model: `claude-haiku-4-5`
- Queries: 12 per class × 5 classes = 60
- Retrieval: DINOv2-B (k=2 anchors per class, person-disjoint) — same collage as baseline.
- Sampling variation: 3 prompt variants (`A`, `B`, `C`) differ only in the intro sentence.
  - `A`: *You are a medical expert classifying AFM ...*
  - `B`: *Take a close look at the attached AFM ...*
  - `C`: *Examine carefully the AFM ...*
- **Note**: `claude -p` CLI does NOT expose `--temperature`; prompt variation is the only practical noise source.
- Total API cost: $2.715

## Per-variant and voted F1 (60-scan set)

| Strategy | N | Accuracy | F1 weighted | F1 macro |
|---|---|---|---|---|
| Single (A) | 60 | 0.6667 | 0.6473 | 0.6473 |
| Single (B) | 60 | 0.5833 | 0.5910 | 0.5910 |
| Single (C) | 60 | 0.6667 | 0.6696 | 0.6696 |
| **Majority vote 3x** | 60 | 0.6667 | 0.6723 | 0.6723 |

## Agreement distribution

| Agreement | Count | % |
|---|---|---|
| 3/3 | 31 | 51.7% |
| 2/3 | 25 | 41.7% |
| 1/1/1 | 4 | 6.7% |

## Agreement-to-accuracy calibration

(Fraction of voted predictions that are correct, grouped by sample agreement.)

| Agreement | N | Accuracy |
|---|---|---|
| 3/3 | 31 | 0.806 |
| 2/3 | 25 | 0.600 |
| 1/1/1 | 4 | 0.000 |

## Per-class agreement distribution

| Class | 3/3 | 2/3 | 1/1/1 | total |
|---|---|---|---|---|
| ZdraviLudia | 10 | 2 | 0 | 12 |
| Diabetes | 6 | 6 | 0 | 12 |
| PGOV_Glaukom | 5 | 7 | 0 | 12 |
| SklerozaMultiplex | 5 | 4 | 3 | 12 |
| SucheOko | 5 | 6 | 1 | 12 |

## Where does voting help / hurt vs single-sample (variant A)

- A right, vote right: 35
- A wrong, vote right (**vote helps**): 5
- A right, vote wrong (**vote hurts**): 5
- Both wrong: 15

### Flip cases (where voting changes the verdict)

| Scan | Truth | A | B | C | Voted | Flip |
|---|---|---|---|---|---|---|
| `TRAIN_SET/ZdraviLudia/Kontr_01.03.2023_LO.015` | ZdraviLudia | ZdraviLudia | Diabetes | Diabetes | Diabetes | A_right->vote_wrong |
| `TRAIN_SET/Diabetes/Dusan2_DM_STER_mikro_281123.002` | Diabetes | Diabetes | ZdraviLudia | ZdraviLudia | ZdraviLudia | A_right->vote_wrong |
| `TRAIN_SET/Diabetes/Dusan2_DM_STER_mikro_281123.003` | Diabetes | Diabetes | ZdraviLudia | ZdraviLudia | ZdraviLudia | A_right->vote_wrong |
| `TRAIN_SET/Diabetes/Dusan2_DM_STER_mikro_281123.005` | Diabetes | ZdraviLudia | Diabetes | Diabetes | Diabetes | A_wrong->vote_right |
| `TRAIN_SET/PGOV_Glaukom/26_PV_PGOV.002` | PGOV_Glaukom | PGOV_Glaukom | ZdraviLudia | ZdraviLudia | ZdraviLudia | A_right->vote_wrong |
| `TRAIN_SET/SklerozaMultiplex/100_7-SM-LV-18.001` | SklerozaMultiplex | ZdraviLudia | SklerozaMultiplex | SklerozaMultiplex | SklerozaMultiplex | A_wrong->vote_right |
| `TRAIN_SET/SucheOko/29_PM_suche_oko.000` | SucheOko | SucheOko | PGOV_Glaukom | ZdraviLudia | ZdraviLudia | A_right->vote_wrong |
| `TRAIN_SET/SucheOko/35_PM_suche_oko.001` | SucheOko | Diabetes | SucheOko | SucheOko | SucheOko | A_wrong->vote_right |
| `TRAIN_SET/SucheOko/35_PM_suche_oko.010` | SucheOko | ZdraviLudia | SucheOko | SucheOko | SucheOko | A_wrong->vote_right |
| `TRAIN_SET/SucheOko/35_PM_suche_oko.021` | SucheOko | ZdraviLudia | SucheOko | SucheOko | SucheOko | A_wrong->vote_right |

## Per-class F1: single (A) vs voted

| Class | Support | F1 single A | F1 voted | Δ |
|---|---|---|---|---|
| ZdraviLudia | 12 | 0.667 | 0.588 | -0.078 |
| Diabetes | 12 | 0.714 | 0.692 | -0.022 |
| PGOV_Glaukom | 12 | 0.800 | 0.783 | -0.017 |
| SklerozaMultiplex | 12 | 0.556 | 0.632 | +0.076 |
| SucheOko | 12 | 0.500 | 0.667 | +0.167 |

## Key findings

### 1. Prompt phrasing alone induces meaningful disagreement

Three prompt variants differ only in the opening sentence. Despite that, the 3 samples agree unanimously on **51.7%** of queries, are 2-vs-1 on **41.7%**, and produce three different answers on **6.7%**. That confirms the VLM classifier has substantial prompt-sensitivity noise even without temperature.

### 2. Variant F1 differs more than voting moves the needle

| Variant | Weighted F1 |
|---|---|
| A (medical expert) | 0.6473 |
| B (take a close look) | 0.5910 |
| C (examine carefully) | 0.6696 |
| **Vote (A+B+C)** | **0.6723** |

The gap between the best and worst prompt variant is often comparable to (or larger than) the vote-vs-single-sample delta. **Picking the best prompt is a cheaper win than paying 3x for voting.**

### 3. Voting trades easy classes for hard classes

| Class | ΔF1 (A → vote) |
|---|---|
| ZdraviLudia | -0.078 |
| Diabetes | -0.022 |
| PGOV_Glaukom | -0.017 |
| SklerozaMultiplex | +0.076 |
| SucheOko | +0.167 |

On easy classes the 3 samples mostly agree, so voting rarely changes anything but occasionally lets noisier B/C samples out-vote a correct A sample. On hard classes the A variant misses often and B/C rescue some queries — net positive on SucheOko in particular.

### 4. Agreement is a strong calibration signal

Looking at the agreement-to-accuracy table above: 3/3 unanimous predictions are correct ~80% of the time, 2/3 majority ~60%, and 1/1/1 all-disagree queries are **essentially never correct**. Prompt-variant disagreement is a better abstain / uncertain signal than raw confidence scores.

## Decision: is 3× cost worth it?

- Δ weighted F1 = **+0.0250** (A → vote)
- Δ macro F1 = **+0.0250** (A → vote)
- Cost multiplier: **~3x** per-scan API cost.
- Query-level flips: 5 helps, 5 hurts (net +0).

**Verdict: MARGINAL / NO for the champion ensemble. The +0.025 weighted F1 gain is below the n=60 noise floor (~0.03), and query-level helps (5) and hurts (5) nearly cancel. The gain is driven by class-F1 redistribution toward the hardest class (SucheOko), which is structurally valuable but does not survive if the eval distribution is imbalanced. A prompt-phrasing swap (cheapest variant) captures most of the signal at 1x cost.**


**Recommendation**: do NOT add self-consistency voting to the champion ensemble. Instead: (a) swap to the highest-scoring single-variant prompt (free win), (b) use voting only as a *selective* strategy on queries where variant-A confidence is low (lazy self-consistency), (c) treat 1/1/1 all-disagree queries as abstain / route-to-stronger-model candidates.


## Caveats

- `claude -p` CLI does not expose sampling temperature; variation comes from prompt phrasing alone.
  A proper self-consistency experiment with temperature > 0 would need the Anthropic SDK path or an additional
  CLI flag, which was not available at run time.
- 60 queries is small → F1 differences < ~0.03 are within noise (95% Wilson CI half-width ~ 0.12 at n=60).
- Same collage reused across 3 samples → isolates model stochasticity, not retrieval noise.