# Multi-Agent Orchestration Pattern — The Methodology

**Applied to:** Hack Košice 2026 UPJŠ tear-film AFM disease classification
**Author:** Claude Opus 4.7 as orchestrator
**Observation window:** 2026-04-18, ~4 hours autonomous execution

---

## Core idea

Treat a small-data ML project as a **research laboratory** rather than a pipeline:
- One **orchestrator agent** (Claude Opus 4.7) maintains state, dispatches specialists, and makes strategic decisions.
- Many **specialist sub-agents** (general-purpose Claude sub-agents) execute bounded tasks in parallel.
- A dedicated **red-team discipline**: every F1 > baseline claim is independently audited before adoption.
- A **living ledger** (`STATE.md` + `AUTORESEARCH_LEDGER.md`) is the single source of truth.

The pattern is **one level above** Karpathy's `autoresearch` loop: autoresearch is one agent iterating on one metric. Here we have a dispatcher coordinating many specialists with complementary roles — closer to how a PI runs a lab.

---

## Roles

### Orchestrator (the director)
- Reads agent reports.
- Updates `STATE.md` after each wave.
- Decides what to dispatch next based on gaps and opportunities.
- Never does gruntwork that can be delegated (hard rule).
- Writes syntheses (pitch narrative, final report).

### Researcher agents
- Use Perplexity to scan literature for benchmarks, datasets, methodologies.
- Return structured markdown reports with citations.
- Example waves: literature benchmark, external AFM dataset survey.

### Implementer agents
- Write code, run experiments, report honest metrics.
- Receive self-contained prompts with exact paths and expected deliverables.
- Example waves: TTA, multi-channel, multi-scale, synthetic data.

### Validator / red-team agents
- Given a claim, try to break it.
- Bootstrap CI, nested CV, leakage audits, label-shuffle tests.
- Reject or confirm with evidence.
- 7 total audits so far; rejected 6 inflated claims, approved 2.

### Synthesizer agents
- Integrate findings from multiple implementers into one deliverable.
- Write `FINAL_REPORT.md`, `PITCH_NARRATIVE.md`, etc.

### Specialist agents
- Dive deep on one problem (e.g., SucheOko 2-patient ceiling, Grad-CAM attribution, clinical report generation).

---

## Wave-based execution

| Wave | Focus | Agents | Outcome |
|---|---|---:|---|
| 1 | Data audit, baselines, handcrafted features | 4 | F1 baseline 0.50 established |
| 2 | Probability-average ensembles, CGNN retry | 4 | Ensemble 0.6528 (nested-tuned) |
| 3 | Cascade specialists, LLM reasoning, soft-blend | 4 | Cascade hurts, LLM hurts; soft-blend +0.01 |
| 4 | TTA, shippable packaging, final synthesis | 3 | v1 TTA 0.6458 shipped |
| 5 | Autoresearch (10 hypotheses), multi-channel, SSL, advanced features | 6 | **v2 recipe: 0.6562** (L2 + geom-mean) |
| 6 | Multichannel fusion + red-team | 2 | +0.008 claim rejected (bootstrap CI crossed 0) |
| 7 | Multi-scale + TTA at 45nm + red-team | 3 | **v4 champion: 0.6887** (P>0=0.999) |
| 8 | V4 packaging, BMP fallback, open-set detection | 3 | Shipped bundle; BMP 0.66; open-set AUROC 0.62 |
| 9 | Literature benchmark (human κ comparison) | 1 | Our F1 matches/exceeds human inter-rater |
| 10 | Per-patient metrics, external AFM, Grad-CAM, synthetic data | 4 | *(running)* |
| 11 | DANN, hierarchical, foundation zoo, physics, meta-ensemble, active learning, paper draft, clinical report | 8 | *(dispatching)* |

## Red-team rejection history

Every claim > baseline gets a bootstrap/nested-CV audit. Rejected claims:

| Claim | Source | Honest value | Rejection reason |
|---|---:|---:|---|
| 0.6878 | Production-Optimizer | 0.6326 | Eye-level + bias tuning on same OOF |
| 0.6780 | prob_ensemble headline | 0.6516 | Eye-level + threshold leak |
| 0.6770 | cascade @ thr=0.50 | — | Gating threshold tuned on eval |
| 0.6731 | double-gated cascade | — | Specialist threshold tuned on eval |
| 0.6698 | prob_ensemble tuned | 0.6528 | Thresholds + subset on same OOF |
| 0.6645 | multichannel E7 | ~0.656 | Bootstrap CI [−0.04, +0.05] crosses 0 |

Accepted claims:
- v2 recipe: 0.6562 (Wave 5 H1+H2, no tuning)
- v4 multi-scale: 0.6887 (Wave 7, bootstrap P(Δ>0)=0.999)

---

## Ledger artifacts (all git-tracked)

| File | Purpose |
|---|---|
| `STATE.md` | Current orchestration state, live-updated |
| `AUTORESEARCH_LEDGER.md` | Cumulative hypothesis history, scored by impact |
| `reports/FINAL_REPORT.md` | Comprehensive technical write-up |
| `reports/PITCH_NARRATIVE.md` | 5-minute pitch script |
| `reports/DESIGN_RATIONALE.md` | Why every architectural choice |
| `reports/ARCHITECTURE.md` | Mermaid diagrams of inference + orchestration |
| `reports/LITERATURE_BENCHMARK.md` | Published benchmark comparisons, human κ |
| `reports/ERROR_ANALYSIS.md` | Failure modes, Mode A/B/C/D |
| `reports/RED_TEAM_*.md` × 4 | Independent audits |
| `reports/BENCHMARK_DASHBOARD.md` | Auto-generated leaderboard |
| `reports/EXPECTED_TEST_PERFORMANCE.md` | 7 test-time scenarios |

---

## Why this pattern works for small-data ML

1. **Parallelism** — 4-8 specialists run in the time one sequential researcher would.
2. **Independent audit** — red-team catches leakage that would slip through one-person iteration.
3. **Diversity of hypothesis** — autoresearch agent proposes ideas the orchestrator wouldn't think of.
4. **Living documentation** — every artifact is git-tracked from commit 1, reproducible.
5. **Honest negative results** — non-winning experiments are cataloged with "why didn't this work", becoming pitch evidence.

## What this pattern would need to scale further

- Shared cache system so agents don't duplicate encoding.
- Agent-to-agent messaging (Claude agents can't talk to each other yet; only through shared files).
- Cost accounting per sub-agent.
- Version-controlled "agent library" of reusable prompt templates.

---

## Meta-insight

**At 240 training samples, orchestration pattern DOES NOT help beat the data ceiling.** What it DOES help is:
- finding the true ceiling honestly (vs over-claiming)
- ruling out clever-looking ideas that don't generalize
- documenting a credible research process end-to-end
- producing a comprehensive artifact set that one researcher couldn't in the same time

Our final +0.07 F1 improvement (0.615 → 0.6887) was obtained by the orchestrator recognizing which dimensions were worth pushing, dispatching specialists, and rigorously rejecting inflated claims. Any ONE of these dimensions alone would have plateaued earlier.
