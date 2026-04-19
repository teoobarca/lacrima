# Multi-Agent Orchestration — Full Documentation

**Project:** Hack Košice 2026 — UPJŠ Tear Disease Classification
**Methodology:** Karpathy autoresearch lifted one abstraction higher (orchestrator + specialists + red-team)
**Final shipped model:** `models/ensemble_v4_multiscale/` — weighted F1 0.6887 (person-LOPO honest)

## By the numbers

| | |
|---|---|
| **Wall-clock duration** | 18.2 hours (single Claude Code session: 2026-04-18 11:27 UTC → 04-19 05:39 UTC) |
| **Sub-agents launched** | 218 across 21 waves |
| **Cumulative sub-agent compute** | 20.2 hours (parallelised into 18.2h wall-clock — 5–8 concurrent agents during peak waves) |
| **Orchestrator messages** | 2,366 (Claude Opus 4.7) |
| **Sub-agent messages** | 6,563 (mixed Haiku 4.5 / Sonnet 4.6 / Opus 4.7) |
| **Total tokens** | ~1.49 billion (959 M orchestrator + 530 M sub-agents, predominantly via prompt caching) |
| **Honest experiments completed** | 30+ |
| **Red-team contamination catches** | 9 |
| **Code shipped** | 53,223 lines of Python |
| **Documentation shipped** | 14,811 lines of Markdown across 50+ reports |
| **Pitch deck** | 1,277-line single-file HTML/CSS/JS |

---

## Methodology overview

### Inspiration

Andrej Karpathy's **autoresearch** demonstrated a single LLM in a self-improvement loop. We took the same idea **one abstraction higher**: an **orchestrator** Claude Opus that dispatches **specialist sub-agents**, with a human (the orchestrator's principal) brainstorming strategy and reviewing decisions.

### Roles

| Role | What it does | Examples |
|---|---|---|
| **Orchestrator** | Maintains state, dispatches specialists, makes strategic decisions | Wave planning, ledger updates, agent coordination |
| **Researcher** | Surveys literature via Perplexity (multi-query parallel) | EXTERNAL_DATA_SURVEY, THEORETICAL_CEILING, PITCH_INSPIRATION |
| **Implementer** | Writes code, runs experiments, reports honest metrics | TDA features, MAE pretrain, foundation zoo, LoRA |
| **Validator (Red-Team)** | Tries to break a claim — bootstrap CI, leakage audit | 9 contamination catches, including our own 0.8873 leak |
| **Synthesizer** | Integrates findings from multiple implementers into deliverable | FINAL_REPORT, PITCH_NARRATIVE |
| **Specialist** | Dives deep on one problem | SucheOko rescue, clinical reasoning, Grad-CAM |

### Wave-based execution

Agents are dispatched in **waves** (parallel batches). Each wave: 3-8 agents, ~15-30 min wall-clock per wave. Outputs land in `cache/` (data) and `reports/` (markdown).

---

## Wave-by-wave history

### Wave 1 — Data audit + baselines (4 agents)
- Data audit (240 scans, 35 persons, 7:1 imbalance)
- Handcrafted features baseline (XGBoost 0.50)
- Validator caught L/R eye image-level grouping bug → fixed via `person_id()`
- DINOv2-B + LR baseline: **0.615**

### Wave 2 — Probability ensembles + CGNN (4 agents)
- Prob-average ensemble: claimed 0.67, **red-team rejected** (threshold leak), honest 0.6528
- CGNN on MPS hung → CPU retry: 0.365 (kept for interpretability only)
- Cascade specialists: hurts (-0.048)

### Wave 3 — Cascade soft-blend, LLM reasoning (4 agents)
- Cascade-stacker soft-blend: +0.01 (small, kept)
- LLM-gated reasoner CLI: F1 override hurts (-0.012), reasoning text salvaged

### Wave 4 — TTA, packaging, synthesis (3 agents)
- v1 TTA shipped: 0.6458
- Pitch visualizations rendered

### Wave 5 — Autoresearch (10 hypotheses), multichannel, SSL (6 agents)
- **Autoresearch agent discovered the v2 recipe**: L2-norm + geom-mean of softmaxes → +0.011
- v2 TTA shipped: **0.6562** (champion at this point)
- SSL SupCon: marginal, kept embeddings for ensemble
- Advanced features (multifractal, lacunarity): no breakthrough alone

### Wave 6 — Multichannel + red-team (2 agents)
- Multichannel E7 (Height + Amplitude + Phase): claimed +0.008
- **Red-team rejected**: bootstrap CI [-0.04, +0.05] crosses 0 (P(gain>0) = 0.598 = coin flip)

### Wave 7 — Multi-scale + TTA + red-team (3 agents)
- Multi-scale (90 + 45 nm) + TTA: claimed 0.6887
- **Red-team approved**: bootstrap P(Δ>0) = 0.999 strict
- v4 multiscale shipped: **0.6887 ← CURRENT CHAMPION**

### Wave 8 — V4 packaging, BMP fallback, open-set (3 agents)
- v4 bundle saved to `models/ensemble_v4_multiscale/`
- BMP raster fallback: 0.66 (degradation 2 pp)
- Open-set detection AUROC 0.62

### Wave 9 — Literature benchmark + per-patient + first VLM (4 agents)
- Literature benchmark: human inter-rater κ 0.57-0.75 → ML matches/exceeds
- **First VLM filename leak caught**: `vlm_tiles/Diabetes__37_DM.png` → 88% leaky → 28% honest
- Per-patient F1: 0.8011 (per-scan 0.6887)

### Wave 10 — Per-patient metrics, external data, Grad-CAM, synthetic (4 agents)
- External AFM survey: QUAM-AFM, Dryad polymer, VOPTICAL — none ready-to-use
- Grad-CAM per-class attribution maps generated
- Synthetic augmentation tested, no F1 gain

### Wave 11 — DANN, hierarchical, foundation zoo, physics, meta-ensemble, paper, clinical (8 agents)
- DANN domain-adversarial: max 0.6111 (below baseline)
- Hierarchical: -0.034
- Physics-informed simulation: insufficient signal alone
- Clinical reasoning template (VLM-generated narrative scaffold)

### Wave 12 — 5 new techniques in parallel (5 agents)
- Active learning: kept for triage
- Class-specific ensemble: -0.005
- Open-set OOD detector: usable for triage at AUROC 0.62

### Wave 13 — Expert Council + 5 ablations (5 agents)
- Expert Council Haiku judge: **0.591** (HURTS v4 by -0.097)
- Pure k-NN baseline: 0.612 (clean ablation, v4 wins by +21 pp on overlap)
- Self-consistency TTA: noise floor
- Sonnet 4.6 60-subset: 0.845 *(later proven contaminated)*
- Opus 4.7 zero-shot: 0.277 (anchors essential)

### Wave 14 — Red-team Sonnet 0.8873 (1 agent)
- Sonnet full-240 few-shot: claimed **0.8873** (above literature ceiling 0.78-0.84)
- **Red-team caught second filename leak**: `vlm_few_shot_collages/<CLASS>__scan.png`
- 20 paths leaky → 20/20 correct; same 20 obfuscated → 5/19 correct
- VLM path **dead**: honest wF1 = 0.34

### Wave 15 — Honest Sonnet rerun + leakage prevention (4 agents)
- Honest Sonnet rerun (obfuscated paths): wF1 0.3424 (inflation +0.545!)
- Fusion ensemble (v4 + k-NN + XGB): max 0.6613 (< v4)
- Threshold calibration nested LOPO: +0.005 (noise)
- **Leakage prevention infra deployed**: `teardrop/safe_paths.py` + 16 scripts retrofitted + 12 unit tests

### Wave 16 — VLM re-ranker + TDA + augmented head (3 agents)
- VLM binary re-ranker on abstain (with safe paths): 35% acc on binary (worse than random 50%) → -0.021
- TDA persistent homology fusion: -0.064 (errors correlate with DINOv2)
- Augmented head D4 expansion: -0.037

### Wave 17 — Hierarchical + LLM numeric reasoner + MAE (3 agents)
- Hierarchical 2-stage: -0.034 (healthy "relief valve" lost)
- LLM numeric reasoner (Sonnet on quantitative features only, no image): wF1 **0.126** (mode collapse to ZdraviLudia)
- MAE in-domain pretraining (ViT-Tiny, 17k patches): -0.117

### Wave 18 — LoRA + ProtoNet + 4 more (6 agents)
- **LoRA DINOv2-B fine-tune**: -0.041 (overfit on 240 samples)
- Prototypical Networks: -0.169, SucheOko 0.000 → 0.113 (still under threshold)
- Foundation zoo (DINOv2-L, SigLIP-SO400M, EVA-02, OpenCLIP-L, PubMedCLIP): all under DINOv2-B baseline
- LR hparam sweep nested CV: -0.053
- Embedding Mixup (4 variants): -0.032 to -0.035
- Patient-level classifier: 0.8177 *(red-team caveat: apples-to-oranges, real gain only +0.044 on fair baseline)*

### Wave 19 — Hybrid Re-ID (1 agent)
- Hybrid Re-ID classifier: +0.002 (noise, but SAFE — fallback to v4 if re-id doesn't fire)
- Re-ID fire rate 7%, fire accuracy 94% (genuine signal when fires)
- AUC same-person detection 0.485 (DINOv2 is class-discriminator, not patient-discriminator)

### Wave 20 LOCAL — Multi-seed + bagging + multichannel + tree heads + weighted geomean (5 experiments)
*(Credit limit hit, ran via local Bash python — $0 API)*
- Multi-seed v4 ensemble (5 seeds): 0.0000 Δ (LR is deterministic with lbfgs solver)
- Bagging + multi-C (25 members per encoder): -0.017 (SucheOko underrepresented in bootstrap bags)
- Multichannel revisit: -0.016
- Tree heads (ExtraTrees / RandomForest): -8.6 / -9.6 (overfit on 768-dim features with 240 samples)
- Weighted geomean (inner-F1 weights): -0.003

### Wave 21 — Final shipping (in progress)
- v5 adaptive shipped (calibration, triage, hybrid Re-ID safety net)
- Documentation consolidated
- Pitch materials finalized

---

## Red-team contamination catches (9 total)

| # | Wave | Caught | Inflation | Fix |
|---|---|---|---|---|
| 1 | 1 | L/R eye image-level grouping (44 eyes vs 35 persons) | +1-2 pp | `person_id()` collapses L/R |
| 2 | 2 | Threshold tuning + subset selection on same OOF | +0.017 | Nested CV mandatory |
| 3 | 3 | Cascade gating threshold tuned on eval | rejected | Soft-blend instead |
| 4 | 3 | Specialist threshold tuned on eval | rejected | Stacker pattern |
| 5 | 5 | Production-Optimizer 4-component bias tuning | +0.06 | Reverted to 2-comp |
| 6 | 6 | Multichannel E7 +0.008 claim | within noise | Bootstrap CI rejected |
| 7 | 9 | **VLM filename leak** `vlm_tiles/<CLASS>__scan.png` | +0.60 | Honest rerun: 0.88 → 0.28 |
| 8 | 14 | **VLM filename leak #2** `vlm_few_shot_collages/<CLASS>__scan.png` | +0.55 | Built `safe_paths.py` runtime guard |
| 9 | 18 | Patient-level "0.8177" used apples-to-oranges baseline | +0.085 | Real gain +0.044 only under per-patient regime |

**After catch #8**, we built `teardrop/safe_paths.py` — a runtime guard that **physically prevents** filename leakage:
- `safe_tile_path()` is the only way to construct VLM image paths
- `assert_prompt_safe(prompt)` raises `PromptLeakError` if prompt contains class names
- 16 VLM scripts retrofitted, 12 unit tests passing
- AST-based lint forbids `f"{cls}__..."` patterns
- See `reports/LEAKAGE_PREVENTION.md`

---

## Negative results (honest)

These were tried and don't work — kept as evidence of honest exploration:

| Direction | Result | Why it failed |
|---|---|---|
| LoRA fine-tune | -4.1 pp | 240 samples too few, even 0.5% trainable params overfit |
| MAE pretraining (ViT-Tiny) | -11.7 pp | 17k patches is 100× smaller than published MAE corpora |
| Foundation zoo (5 encoders) | -0.5 to -2.6 pp | All worse than DINOv2-B; greedy selection terminates immediately |
| Hierarchical 2-stage | -3.4 pp | Healthy "relief valve" for ambiguous diseased scans is lost |
| Embedding Mixup | -3.2 pp | DINOv2 embeddings already linearly separable; mixup blurs boundaries |
| Augmented head D4 | -3.7 pp | 8 near-collinear samples per scan dilute LR balance |
| Tree heads (ET/RF) | -8.6 / -9.6 pp | 768-dim features × 240 samples → overfit |
| Multichannel (Wave 6 revisit) | -1.6 pp | RGB drag down |
| TDA persistent homology | -6.4 pp fusion | Errors correlate with DINOv2 (not orthogonal) |
| Bagging + multi-C | -1.7 pp | SucheOko underrepresented in bootstrap |
| Threshold calibration nested | +0.5 (noise) | LR balanced is already optimal |
| Multi-seed v4 ensemble | 0.0 Δ | LR with lbfgs is deterministic |
| Weighted geomean (inner F1) | -0.3 (noise) | Uniform weights are optimum |
| ProtoNet ensemble | -1.9 pp | SucheOko 1-person prototype regime brutal |
| DANN (Wave 11) | 0.611 standalone | Domain-invariant features hurt class signal |
| Self-consistency TTA voting | +0.003 (noise) | Voting cost not justified |
| Sonnet 4.6 zero-shot | -41 pp | AFM out-of-distribution for web-trained VLMs |
| Opus 4.7 zero-shot | -41 pp | Same |
| Sonnet 4.6 few-shot honest | -34 pp | Same; anchors don't bridge OOD gap |
| Opus 4.7 few-shot | -4.8 pp vs Sonnet | 5× cost, marginally worse |
| LLM numeric reasoner (text-only) | -56 pp | Mode collapse to majority class |
| Expert Council (LLM judge) | -9.7 pp | Haiku judge inherits AFM OOD problem, over-predicts ZdraviLudia |
| Patient-level broadcast | unusable | Not valid under patient-disjoint test split (organizers' regime) |

---

## Confirmed working components (in shipped v4)

- **DINOv2-B @ 90 nm/px** — overall fractal structure
- **DINOv2-B @ 45 nm/px** — fine crystal edges
- **BiomedCLIP @ 90 nm/px + D4 TTA** — medical prior with rotation invariance
- **L2-normalize → StandardScaler → Logistic Regression** per encoder (frozen backbone)
- **Geometric mean** of 3 softmaxes (penalizes disagreement)
- **Person-LOPO** validation (35 folds, 240 honest predictions)

---

## v5 production layers (additive on top of v4)

- **Layer 1: Hybrid Re-ID** — adaptive blend with nearest-neighbor label if cosine sim > 0.94 (worst case = v4)
- **Layer 2: Temperature scaling** — T = 2.97, ECE 0.21 → 0.08 (60% better calibration)
- **Layer 3: Triage abstain** — flag scans with margin < 0.10 for human review (92% autonomous)

---

## Final state

- **Shipped model**: `models/ensemble_v4_multiscale/`
- **Production wrapper**: `models/ensemble_v5_adaptive/`
- **Honest weighted F1**: **0.6887** (person-LOPO, patient-disjoint)
- **Realistic test F1 estimate**: 0.69-0.73 (full-train bonus over LOPO)
- **Pitch evidence**: 218 agents, 30+ honest experiments, 9 red-team catches, runtime leakage prevention deployed

---

## Why the orchestration pattern matters at 240 samples

At this data scale, the orchestration pattern **does not** push F1 past the data ceiling — that ceiling is set by:
- Number of patients (35), especially SucheOko's 2-person structural limit
- Patient-disjoint regime requirements
- Inter-rater ambiguity in manual class definitions

What the pattern **does** do:
1. **Find the true ceiling honestly** (vs over-claiming via leakage)
2. **Rule out clever-looking ideas** that don't generalize (30+ negative results documented)
3. **Document a credible research process** end-to-end (every report git-tracked)
4. **Produce comprehensive artifact set** that one researcher couldn't in the same time

The +0.07 F1 improvement (0.615 → 0.6887) was obtained by recognizing **which dimensions were worth pushing** (multi-scale, encoder diversity), dispatching specialists, and **rigorously rejecting inflated claims**. Any one dimension alone would have plateaued earlier.
