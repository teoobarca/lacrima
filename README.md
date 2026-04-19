# Teardrop Challenge — Hack Košice 2026 / UPJŠ

**Disease classification from tear-film AFM micrographs** — orchestrated multi-agent ML research pipeline.

> **Champion shipped model:** v4 multi-scale TTA ensemble (DINOv2-B at 90 nm/px + 45 nm/px + BiomedCLIP, D4 test-time augmentation, L2-normalized embeddings, geometric-mean softmax combination).
> **Honest F1:** 0.6887 weighted (macro 0.554), person-level Leave-One-Patient-Out (35 persons, 240 scans, 5 classes). Red-team bootstrap 95% CI strictly > 0 (P = 0.999).

## Quick start (inference)

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python predict_cli.py \
    --model models/ensemble_v2_tta \
    --input /path/to/TEST_SET \
    --output submission.csv
```

## Interactive demo

```bash
.venv/bin/python app.py        # opens on http://localhost:7860
```

## Documentation

| File | Purpose |
|---|---|
| [`SUBMISSION.md`](SUBMISSION.md) | Organizer-facing handoff — how to predict on new data |
| [`REPRODUCE.md`](REPRODUCE.md) | Full reproduction guide (fresh machine → shipped model) |
| [`STATE.md`](STATE.md) | Orchestration ledger — live state of experiments |
| [`CLAUDE.md`](CLAUDE.md) | Project context + conventions (for AI agents) |
| [`pitch_slides.md`](pitch_slides.md) | 10-slide pitch outline |
| [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) | Comprehensive technical report |
| [`reports/PITCH_NARRATIVE.md`](reports/PITCH_NARRATIVE.md) | 5-minute pitch script |
| [`reports/ABSTRACT.md`](reports/ABSTRACT.md) | Research-paper-style single-page summary |
| [`reports/ARCHITECTURE.md`](reports/ARCHITECTURE.md) | System diagrams (Mermaid) |
| [`reports/DATA_AUDIT.md`](reports/DATA_AUDIT.md) | Raw data audit (pre-training) |
| [`reports/ERROR_ANALYSIS.md`](reports/ERROR_ANALYSIS.md) | Failure-mode deep-dive |
| [`reports/BENCHMARK_DASHBOARD.md`](reports/BENCHMARK_DASHBOARD.md) | Canonical leaderboard (auto-generated) |
| [`reports/pitch/`](reports/pitch/) | Pitch visualizations (6 PNGs) |

## Headline results

| Model | Weighted F1 | Macro F1 |
|---|---:|---:|
| **★ v4 multi-scale TTA (shipped, 90+45nm+BiomedCLIP)** | **0.6887** | **0.5541** |
| v2 TTA (superseded champ, single-scale) | 0.6562 | 0.5382 |
| v1 TTA ensemble | 0.6458 | 0.5154 |
| Non-TTA ensemble | 0.6346 | 0.4934 |
| DINOv2-B 45 nm/px single | 0.6433 | 0.5038 |
| DINOv2-B 90 nm/px single | 0.6150 | 0.4910 |
| Handcrafted (94 feat) + LR | 0.4882 | 0.3707 |
| Label-shuffle null | 0.276 ± 0.042 | — |

Dataset: **240 scans × 35 persons × 5 classes**, imbalance 7:1 (SM vs SucheOko).

## Methodology in one paragraph

Raw Bruker SPM → preprocess (plane-level + resample to 90 nm/px + robust-normalize) → tile into 9 × 512² patches → D4-group test-time augmentation (72 tile views per scan) → encode with two frozen foundation models (DINOv2-B ViT-B/14 and BiomedCLIP ViT-B/16) → mean-pool tile embeddings to one scan-level vector → per-encoder StandardScaler + LogisticRegression with balanced class weights → arithmetic mean of softmaxes → argmax.

## Orchestration pattern

This project was built by an LLM-orchestrated research pipeline:
- **Orchestrator** (Claude Opus) maintains `STATE.md` and dispatches sub-agents in parallel rounds
- **Specialist agents** (general-purpose Claude sub-agents) do focused work: implement experiments, validate, synthesize, build demos
- **Red-team agents** audit every F1 > baseline claim — rejected 3 inflated claims (0.67–0.69) before adoption
- **~20 sub-agents across 5 waves** produced features, embeddings, ensembles, specialists, LLM-reasoning layers, interpretability artifacts, and a working Gradio demo

## Honest negatives

- Hard-override cascade: −0.048 (low-confidence ≠ wrong)
- LLM prediction override: −0.012
- 4-component concat + bias tuning (nested): 0.633 (regression)
- Crystal Graph NN alone: 0.365 (data-starved)
- SupCon SSL projection: 0.612 (below baseline, expected for small data)

## Key limitation

**SucheOko has only 2 unique persons in the training set.** F1 = 0.00 is a ceiling set by the data, not the model. 57% of non-near-miss errors come from this single class.

## License / contact

Hackathon artifact. Contact via project repo issues.
