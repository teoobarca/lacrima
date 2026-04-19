# Lacrima — Disease detection from a single tear

> **Hack Košice 2026 · UPJŠ Tear Challenge** — chronic disease classification from atomic-force microscopy scans of dried tear droplets.

Built through **multi-agent LLM orchestration**: Karpathy's autoresearch lifted one abstraction higher — an orchestrator dispatching specialist sub-agents (researcher · implementer · red-team · synthesizer), with a human-in-the-loop directing strategy.

---

## Headline

| Metric | Value |
|---|---|
| **Weighted F1** (official metric, person-LOPO honest) | **0.6887** |
| Macro F1 | 0.5541 |
| Per-patient F1 (majority vote across patient's scans) | 0.8011 |
| Top-2 accuracy | 88 % |
| Bootstrap 95 % CI | [0.5952 — 0.7931] |
| Signal vs random null | **15.7 σ above baseline** |
| Dataset | 240 scans · 35 patients · 5 classes |
| Process | 218 sub-agents · 21 waves · 30+ honest experiments · 9 contaminations red-teamed |

---

## Quickstart (inference)

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python predict_cli.py \
    --input  /path/to/TEST_SET \
    --output submission.csv
```

Default model is `models/ensemble_v4_multiscale/` (the shipped champion).

## Interactive demo

```bash
.venv/bin/python app.py        # opens on http://localhost:7860
```

## Pitch deck

```bash
open pitch_deck.html           # right-click = next, left-click = back, F = fullscreen
```

---

## Architecture (v4 multi-scale ensemble)

Three frozen foundation encoders, geometric mean of softmaxes:

```
AFM scan (.spm)
  ├──▶ DINOv2-B @ 90 nm/px  ──▶ L2 → StandardScaler → LR head ──┐
  ├──▶ DINOv2-B @ 45 nm/px  ──▶ L2 → StandardScaler → LR head ──┤── geomean ──▶ argmax
  └──▶ BiomedCLIP @ 90 nm/px (D4 TTA) → L2 → … → LR head      ──┘
```

- **DINOv2** — Meta · 142M images · self-supervised. Universal visual encoder.
- **BiomedCLIP** — Microsoft · 15M PubMed images. Medical prior orthogonal to DINOv2.
- **Multi-scale**: 90 nm captures whole fractal · 45 nm captures fine crystal edges.
- **Geometric mean** penalises encoder disagreement → robust ensemble.
- **Frozen backbones** — 240 scans is too few to fine-tune (LoRA tested → −4 pp F1).

Full diagram: [`reports/ARCHITECTURE.md`](reports/ARCHITECTURE.md).

---

## Documentation

### Start here
| File | Purpose |
|---|---|
| [`pitch_deck.html`](pitch_deck.html) | 6-slide deck (open in browser, F = fullscreen) |
| [`reports/AGENTS_DOCUMENTATION.md`](reports/AGENTS_DOCUMENTATION.md) | Wave-by-wave agent log (218 agents, 21 waves) |
| [`reports/V4_FINAL_AUDIT.md`](reports/V4_FINAL_AUDIT.md) | Pre-submission integrity audit (6 rounds, all pass) |
| [`reports/THEORETICAL_CEILING.md`](reports/THEORETICAL_CEILING.md) | Where the F1 ceiling lives, literature-informed |
| [`reports/LEAKAGE_PREVENTION.md`](reports/LEAKAGE_PREVENTION.md) | Runtime-enforced leakage guards (`teardrop/safe_paths.py`) |

### Deep dives
| File | Purpose |
|---|---|
| [`STATE.md`](STATE.md) | Live orchestration ledger |
| [`ORCHESTRATION.md`](ORCHESTRATION.md) | Multi-agent methodology write-up |
| [`reports/ARCHITECTURE.md`](reports/ARCHITECTURE.md) | System diagrams + bundle layout |
| [`reports/DATA_AUDIT.md`](reports/DATA_AUDIT.md) | Raw data audit |
| [`reports/ERROR_ANALYSIS.md`](reports/ERROR_ANALYSIS.md) | Failure-mode deep-dive |
| [`reports/BENCHMARK_DASHBOARD.md`](reports/BENCHMARK_DASHBOARD.md) | Canonical leaderboard |
| [`reports/RED_TEAM_*.md`](reports/) | Independent audits of championship claims |
| [`reports/VLM_CONTAMINATION_FINDING.md`](reports/VLM_CONTAMINATION_FINDING.md) | Filename-leak post-mortem |
| [`SUBMISSION.md`](SUBMISSION.md) | Organizer-facing handoff |
| [`REPRODUCE.md`](REPRODUCE.md) | Fresh-machine reproduction guide |

---

## Honest negative results

We tried 30+ directions; most lost. Documented in [`reports/AGENTS_DOCUMENTATION.md`](reports/AGENTS_DOCUMENTATION.md). Highlights of what didn't work:

| Direction | Result | Why |
|---|---|---|
| LoRA fine-tuning of DINOv2-B | −4.1 pp wF1 | 240 scans too few for any backbone training |
| MAE in-domain pretraining (ViT-Tiny) | −11.7 pp | 17k patches is 100× smaller than published MAE corpora |
| Foundation-model zoo (5 encoders) | All under DINOv2-B baseline | Sweet spot is 3-encoder geomean |
| Hierarchical 2-stage classifier | −3.4 pp | Healthy class is a relief valve, not a wall |
| TDA persistent-homology fusion | −6.4 pp | Errors correlate with DINOv2, not orthogonal |
| VLM (Haiku / Sonnet / Opus, all variants) | up to −56 pp | AFM is out-of-distribution for web-trained VLMs |

---

## Red-team discipline

Every score above baseline is independently audited by a red-team sub-agent (bootstrap CI + leakage scan + nested-CV recheck). **Nine contaminations were caught before any went live**, including:

- Image-level vs person-level eye grouping (Wave 1)
- OOF threshold/bias tuning leakage (Waves 2–6)
- VLM filename leak via `.png` paths — twice in different scripts (Waves 9 + 14)
- Patient-level "0.8177" using apples-to-oranges baseline (Wave 18)

After the second filename-leak we built [`teardrop/safe_paths.py`](teardrop/safe_paths.py) — a runtime guard that physically prevents class names from appearing in prompt paths, with 12 unit tests and AST-based lint enforcement.

---

## Key limitation

**SucheOko has only 2 unique patients in the dataset.** Per-class F1 is structurally bounded — no model can recover this without more samples. Per-class F1 = 0.00 is a *data-collection problem*, not a model problem.

---

## License

Hackathon artifact. MIT.

## Contact

Built for Hack Košice 2026. Contact via repo issues.
