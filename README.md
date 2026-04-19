# Lacrima

**Chronic disease classification from atomic-force microscopy scans of dried tear droplets.**

Built for [Hack Košice 2026](https://www.hackkosice.com/) · [UPJŠ Faculty of Science](https://www.upjs.sk/en/faculty-of-science/) Tear Challenge · April 2026.

The dataset is 240 raw Bruker SPM scans from 35 unique patients across 5 diagnostic classes
(`ZdraviLudia` healthy, `Diabetes`, `PGOV_Glaukom` glaucoma, `SklerozaMultiplex`, `SucheOko` dry-eye).
Class imbalance is 7 : 1 (SM 95 vs SucheOko 14). The official metric is **weighted F1**.

---

## Headline metrics

All metrics are person-disjoint Leave-One-Patient-Out cross-validation (35 outer folds, every scan predicted exactly once by a model that never saw any other scan from the same patient).

| Metric | Value | Notes |
|---|---|---|
| **Weighted F1** | **0.6887** | Official challenge metric |
| Macro F1 | 0.5541 | SucheOko F1 = 0.000 (2-patient structural ceiling) |
| Accuracy | 0.6958 | — |
| Top-2 accuracy | 0.8792 | True class is in top-2 predictions 88 % of the time |
| Per-patient F1 (majority vote) | 0.8011 | Aggregating scans of the same patient |
| Bootstrap 95 % CI on weighted F1 | [0.5952, 0.7931] | 1000 person-resamples, paired |
| Signal vs label-shuffle null | **15.7 σ** above baseline | Null = 0.281 ± 0.026 over 100 shuffles |

Per-class F1 (person-LOPO):
`ZdraviLudia 0.917 · SklerozaMultiplex 0.691 · Diabetes 0.583 · PGOV_Glaukom 0.579 · SucheOko 0.000`

Confusion matrix and full breakdown: [`reports/V4_FINAL_AUDIT.md`](reports/V4_FINAL_AUDIT.md).

---

## Approach

The shipped model (`models/ensemble_v4_multiscale/`) is a 3-component **frozen-encoder ensemble**:

```
                                                            ┌──────────────────────┐
AFM scan (.spm)  →  preprocess  →  9 tiles × 512²  →  ┬──→ │ DINOv2-B @ 90 nm/px  │ → L2 → StandardScaler → LR → softmax₁
                                                      ├──→ │ DINOv2-B @ 45 nm/px  │ → L2 → StandardScaler → LR → softmax₂
                                                      └──→ │ BiomedCLIP @ 90 nm   │ → L2 → StandardScaler → LR → softmax₃
                                                            └─────────┬────────────┘
                                                                      │  geometric mean
                                                                      ▼
                                                                   argmax  →  predicted class
```

### Component decisions (each empirically validated)

| Decision | Empirical evidence |
|---|---|
| **Resample to 90 nm/px** | 78 % of native scans are 92.5 µm range at 1024² → 90 nm is no-resample; 50 µm scans get a clean 7.5× downsample. See `reports/DATA_AUDIT.md`. |
| **Add a second DINOv2 stream at 45 nm/px** | Wave 7 systematic scale grid: 90+45 → 0.6887, single 90 → 0.6562, single 45 → ~0.62, 90+45+180 → 0.68. Three scales saturate. |
| **BiomedCLIP at 90 nm/px only (no 45 nm)** | BiomedCLIP-B/16 native input is 224². At 45 nm/px the context window crops too tight; F1 drops 0.5 pp. |
| **D4 TTA on BiomedCLIP, none on DINOv2** | TTA on DINOv2 multi-scale regresses F1 by 0.022 — multi-scale already provides view diversity. TTA on BiomedCLIP medical features adds +1 pp. |
| **Frozen backbones** | LoRA fine-tune (r=8, α=16, attention layers, ~600k trainable) tested in Wave 18: −4.1 pp F1. 240 scans is too few even for adapter training. |
| **Logistic Regression heads** | Tree heads (ExtraTrees, RandomForest) overfit 768-dim features on 240 samples (−8.6 to −9.6 pp). LR with L2 + class-balanced weights is optimum. |
| **Geometric mean of softmaxes** | Empirically +1 pp over arithmetic mean (Wave 5 autoresearch discovery). Penalises encoder disagreement → robust. |
| **No fancy fusion** | Logistic-regression stacker (nested LOPO) tested in Wave 15: −2.7 pp. Inner CV is noisier than the base ensemble at n=240. |

Architecture diagrams (Mermaid + bundle layout): [`reports/ARCHITECTURE.md`](reports/ARCHITECTURE.md).

---

## Evaluation protocol — person-LOPO

For each of 35 unique patients (collapsed across L/R eyes, see `teardrop.data.person_id`):

1. Hold out all scans of patient `P`.
2. Fit `StandardScaler` and Logistic-Regression head per encoder on the remaining 34 patients' scans.
3. Predict softmax probabilities for `P`'s scans.
4. Repeat for all 35 patients → 240 honest out-of-fold predictions.
5. Compute weighted F1 on the concatenated predictions.

This **directly simulates the test scenario** confirmed by organisers: per-image evaluation,
patient info stripped, **patient-disjoint train/test split**.

Bootstrap CI: 1000 person-level resamples (with replacement) → recompute F1 each time.

Label-shuffle null baseline: shuffle ground-truth labels 100 times → compute F1 against
unchanged predictions. Signal strength is `(F1_real − mean(F1_null)) / std(F1_null)` = **15.7 σ**.

---

## Reproduction

### Environment

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Tested on Python 3.13, macOS 15 (Apple Silicon, MPS) and Linux x86_64 (CUDA optional).

### Inference on a test directory

```bash
.venv/bin/python predict_cli.py \
    --input  /path/to/TEST_SET \
    --output submission.csv
```

Default model is `models/ensemble_v4_multiscale/`. Cold start ~17 s/scan (cached embeddings ~3 s/scan).

Output CSV format:

```
scan_id, predicted_class, p_ZdraviLudia, p_Diabetes, p_PGOV_Glaukom, p_SklerozaMultiplex, p_SucheOko
```

### Re-train the shipped model from scratch

```bash
# (1) Encode all scans through both backbones + cache embeddings
.venv/bin/python scripts/encode_dinov2_tta.py    # 90 nm/px
.venv/bin/python scripts/encode_dinov2_45nm.py   # 45 nm/px
.venv/bin/python scripts/encode_biomedclip_tta.py

# (2) Fit + save the v4 ensemble
.venv/bin/python scripts/train_ensemble_v4_multiscale.py
```

Full reproduction guide (fresh machine → shipped bundle): [`REPRODUCE.md`](REPRODUCE.md).

### Demo

```bash
.venv/bin/python app.py        # http://localhost:7860
```

Gradio 4-tab demo: classify a scan, view confidence + Grad-CAM, generate clinical reasoning narrative.

---

## Methodology — multi-agent orchestration

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/nanochat) — a single LLM in a self-improvement loop.

This project takes the same idea **one abstraction higher**:

* an **orchestrator** (Claude Opus) maintains state and dispatches specialist sub-agents in parallel waves
* **specialist sub-agents** (Claude general-purpose) execute focused tasks: literature research,
  experiment implementation, red-team auditing, result synthesis, deep specialist work
* a **human-in-the-loop** brainstorms strategy with the orchestrator (PI ↔ postdoc dynamic)

Total: **218 sub-agents across 21 waves**, producing 30+ honest experiments and the documentation in [`reports/`](reports/).

Methodology write-up: [`ORCHESTRATION.md`](ORCHESTRATION.md). Full agent log:
[`reports/AGENTS_DOCUMENTATION.md`](reports/AGENTS_DOCUMENTATION.md).

---

## Red-team discipline

Every score above the running baseline is independently audited by a red-team sub-agent
(bootstrap CI · leakage scan · nested-CV recheck). **Nine contaminations were caught
before any went live**:

| # | Wave | Catch | Inflation |
|---|---|---|---|
| 1 | 1 | Image-level vs person-level eye grouping (44 → 35 persons) | +1–2 pp |
| 2 | 2 | Threshold tuning + subset selection on same OOF | +0.017 |
| 3 | 3 | Cascade gating threshold tuned on eval set | rejected |
| 4 | 3 | Specialist threshold tuned on eval set | rejected |
| 5 | 5 | Production-Optimizer 4-component bias tuning | +0.06 |
| 6 | 6 | Multichannel E7 +0.008 claim | within bootstrap noise |
| 7 | 9 | **VLM filename leak** via `vlm_tiles/<CLASS>__scan.png` paths | **88 % → 28 %** honest |
| 8 | 14 | **VLM filename leak #2** in `vlm_few_shot_collages/` paths | **88.7 % → 34 %** honest |
| 9 | 18 | Patient-level "0.8177" used apples-to-oranges baseline | real gain only +0.044 under per-patient regime |

After catch #8 we built [`teardrop/safe_paths.py`](teardrop/safe_paths.py) — a runtime guard
that physically prevents class names from appearing in prompt paths:

* `safe_tile_path(index, subdir)` returns class-free hash-obfuscated paths
* `assert_prompt_safe(prompt)` raises `PromptLeakError` if class names slip into a prompt
* 16 VLM scripts retrofitted, 12 unit tests + AST-based static lint

A third occurrence of this bug is now structurally impossible.

Full post-mortem: [`reports/VLM_CONTAMINATION_FINDING.md`](reports/VLM_CONTAMINATION_FINDING.md)
and [`reports/LEAKAGE_PREVENTION.md`](reports/LEAKAGE_PREVENTION.md).

---

## Honest negatives

We tried 30+ directions; most lost. Each documented as evidence of honest exploration.

| Direction | Δ vs v4 | Why it failed |
|---|---|---|
| LoRA fine-tuning of DINOv2-B (r=8, attn) | **−4.1 pp** | 240 scans too few even for ~600k adapter parameters |
| MAE in-domain pretraining (ViT-Tiny, 17k patches) | **−11.7 pp** | 100× smaller corpus than published MAE training sets |
| Foundation-model zoo (DINOv2-L, SigLIP-SO400M, EVA-02, OpenCLIP-L, PubMedCLIP) | All under DINOv2-B alone | Greedy forward selection terminates at step 1 |
| TDA persistent-homology fusion (cripser sublevel + landscape) | **−6.4 pp** | Errors correlate with DINOv2, not orthogonal |
| Hierarchical 2-stage (healthy vs disease, then 4-way) | **−3.4 pp** | Healthy class is a relief valve for ambiguous diseased scans |
| Embedding Mixup / CutMix (4 variants) | **−3.2 pp** | DINOv2 embeddings already linearly separable |
| Augmented head (D4 expanded train, noise injection) | **−3.7 pp** | Near-collinear copies dilute LR class balance |
| ProtoNet ensemble + per-class gating | **−1.9 pp** | SucheOko's 1-person LOPO regime is brutal |
| LR head hyperparameter nested-CV sweep | **−5.3 pp** | Nested CV is noisier than flat at n=240 |
| Multi-seed v4 ensemble (5 seeds) | 0.0 pp | LR with lbfgs is deterministic |
| Bagging + multi-C (25 members per encoder) | **−1.7 pp** | SucheOko underrepresented in bootstrap bags |
| Tree heads (ExtraTrees, RandomForest) standalone | **−8.6 / −9.6 pp** | Overfit 768-dim features on 240 samples |
| Multichannel RGB (Height + Amplitude + Phase) fusion | **−1.6 pp** | Auxiliary channels drag down DINOv2 priors |
| Patient-level classifier with attention pooling | unusable | Requires patient ID at test time → invalid under patient-disjoint split |
| Hybrid Re-ID classifier (k-NN fallback to v4) | +0.002 pp | Below noise floor under patient-disjoint regime, but SAFE — never harms |
| Threshold calibration (nested LOPO) | +0.005 pp | Within bootstrap noise, defaults already optimum |
| VLM zero-shot (Haiku 4.5 / Sonnet 4.6 / Opus 4.7) | up to **−41 pp** | AFM is out-of-distribution for web-trained VLMs |
| VLM few-shot honest (Sonnet 4.6 on full 240) | **−34 pp** | Anchor exemplars don't bridge the OOD gap |
| Expert Council (Haiku judge over base models) | **−9.7 pp** | Judge inherits the AFM OOD problem |
| LLM numeric reasoner (Sonnet on quantitative features only) | **−56 pp** | Mode-collapse to the majority class |

These are real bootstrap-validated deltas under the same person-LOPO protocol as the champion.

---

## Repository layout

```
.
├── pitch_deck.html                  Single-file HTML pitch deck (no build step)
├── predict_cli.py                   Inference CLI (default: v4 ensemble)
├── app.py                           Gradio 4-tab demo
├── requirements.txt
│
├── teardrop/                        Core library
│   ├── data.py                      enumerate_samples, person_id, patient_id, CLASSES
│   ├── cv.py                        leave_one_patient_out, patient_stratified_kfold
│   ├── encoders.py                  load_dinov2, load_biomedclip
│   ├── infer.py                     TearClassifier, EnsembleClassifierBundle, TTAPredictor
│   ├── safe_paths.py                Runtime leakage guards (assert_prompt_safe)
│   └── … (features, topology, clinical_report, …)
│
├── models/
│   ├── ensemble_v4_multiscale/      Shipped champion (wF1 0.6887)
│   ├── ensemble_v5_adaptive/        Production wrapper (calibration + triage)
│   ├── ensemble_v2_tta/             Earlier 2-component champion (wF1 0.6562)
│   ├── ensemble_v1_tta/             v1 TTA ensemble (wF1 0.6458)
│   └── … (single-encoder baselines)
│
├── scripts/                         70+ experiment scripts (each documented)
│   ├── train_ensemble_v4_multiscale.py
│   ├── encode_dinov2_tta.py
│   ├── encode_biomedclip_tta.py
│   ├── multiscale_experiment.py     Wave 7 scale-grid that found 90+45 sweet spot
│   └── …
│
├── tests/
│   └── test_no_leak.py              12 unit tests for leakage-prevention infra
│
└── reports/                         50+ markdown reports
    ├── AGENTS_DOCUMENTATION.md
    ├── V4_FINAL_AUDIT.md
    ├── ARCHITECTURE.md
    ├── DATA_AUDIT.md
    ├── ERROR_ANALYSIS.md
    ├── THEORETICAL_CEILING.md
    ├── LEAKAGE_PREVENTION.md
    ├── BENCHMARK_DASHBOARD.md
    ├── RED_TEAM_*.md
    └── … (every experiment + audit)
```

---

## Documentation map

### For replication and verification
* [`reports/V4_FINAL_AUDIT.md`](reports/V4_FINAL_AUDIT.md) — pre-submission integrity audit, 6 rounds, all pass
* [`REPRODUCE.md`](REPRODUCE.md) — fresh-machine reproduction guide
* [`reports/DATA_AUDIT.md`](reports/DATA_AUDIT.md) — raw dataset inspection
* [`reports/BENCHMARK_DASHBOARD.md`](reports/BENCHMARK_DASHBOARD.md) — canonical leaderboard

### For methodology and rationale
* [`reports/ARCHITECTURE.md`](reports/ARCHITECTURE.md) — full system diagrams + bundle layout
* [`reports/THEORETICAL_CEILING.md`](reports/THEORETICAL_CEILING.md) — literature-informed F1 ceiling for this regime
* [`reports/ERROR_ANALYSIS.md`](reports/ERROR_ANALYSIS.md) — failure modes A/B/C/D
* [`reports/EXTERNAL_DATA_SURVEY.md`](reports/EXTERNAL_DATA_SURVEY.md) — survey of external AFM corpora

### For process transparency
* [`reports/AGENTS_DOCUMENTATION.md`](reports/AGENTS_DOCUMENTATION.md) — wave-by-wave agent log (218 agents)
* [`STATE.md`](STATE.md) — live orchestration ledger
* [`ORCHESTRATION.md`](ORCHESTRATION.md) — methodology write-up
* [`reports/VLM_CONTAMINATION_FINDING.md`](reports/VLM_CONTAMINATION_FINDING.md) — filename-leak post-mortem
* [`reports/LEAKAGE_PREVENTION.md`](reports/LEAKAGE_PREVENTION.md) — `safe_paths.py` infrastructure

### Organizer-facing
* [`SUBMISSION.md`](SUBMISSION.md) — handoff doc

---

## Key limitation

**SucheOko (dry-eye) has only 2 unique patients in the entire dataset.** Per-class F1 = 0.000
is a *data-collection limit*, not a model limit. Person-LOPO holds out one patient at a time;
for SucheOko this means training on a single remaining patient — no method survives this
regime. ProtoNet partial rescue achieves F1 = 0.113 (still below ship threshold).

The realistic ceiling for this dataset (35 patients, 5 classes, severe imbalance, frozen
foundation-model regime) is **0.78–0.84 weighted F1** based on literature for comparable
small-medical-imaging benchmarks. Closing the remaining 0.10–0.15 F1 gap requires either
more SucheOko patients or full-backbone fine-tuning with much heavier regularisation
than 240 scans support. Detail: [`reports/THEORETICAL_CEILING.md`](reports/THEORETICAL_CEILING.md).

---

## License

[MIT](LICENSE).

## Citation

If this work is useful in academic context:

```
Lacrima — chronic disease classification from AFM tear-droplet morphology
via multi-agent LLM-orchestrated frozen-encoder ensembles.
Hack Košice 2026, UPJŠ Faculty of Science.
https://github.com/teoobarca/lacrima
```
