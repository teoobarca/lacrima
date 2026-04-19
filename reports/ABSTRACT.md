# Abstract

**Title:** Multi-agent orchestrated machine learning for tear-film AFM disease classification under extreme-data-scarcity

**Problem.** Pavol Jozef Šafárik University (UPJŠ) asks whether chronic disease (Diabetes, Sclerosis Multiplex, Glaucoma, Dry-Eye) can be diagnosed from a single drop of tear via Atomic Force Microscopy. The provided training set contains 240 Bruker-Nanoscope SPM scans across 5 classes, collected from only 35 unique persons — class imbalance 7:1 (SM vs SucheOko), with Dry-Eye represented by just 2 persons.

**Approach.** We constructed a LLM-orchestrated research pipeline in which an orchestrating agent (Claude Opus) dispatched 15+ specialist sub-agents across 5 waves, covering data audit, feature engineering, foundation-model transfer learning, topological data analysis, graph neural networks, probability-averaging ensembles, cascade classifiers, LLM-reasoning second opinions, and test-time augmentation. A discipline of independent red-team agents audited every headline claim for threshold-tuning and subset-selection leakage. Our final classifier is the softmax-average of two frozen foundation-model encoders (DINOv2-B ViT-B/14 and BiomedCLIP) applied to 9 tiles per scan with D4-group test-time augmentation (72 tile views per scan), followed by per-encoder balanced Logistic Regression heads.

**Validation discipline.** All evaluation is person-level Leave-One-Patient-Out CV (35 groups) after a validator agent discovered that the naive patient-ID parser was treating left- and right-eye scans of the same person as distinct "patients" (44 eye-level vs 35 person-level). Any method involving tuning is evaluated via nested CV; bare OOF-tuning is rejected as leaky.

**Results.** Honest weighted F1 of the shipped TTA ensemble is **0.6458** under person-LOPO (macro F1 = 0.516). Single-encoder DINOv2-B baseline is 0.615; random-label null is 0.28 ± 0.04 (~8σ). TTA adds +0.011 at ensemble level and +0.028 / +0.029 per encoder — a small but systematic improvement. A threshold-tuned reference variant reaches 0.6528 via nested CV but is more fragile and not shipped. Per-class F1 ranges from 0.86 (healthy) to 0.00 (Dry-Eye — fundamentally limited by 2-patient training set).

**Key negative results.** Hard-override cascade of binary specialists hurt overall F1 (−0.048); LLM-prediction override hurt (−0.012); 4-component concat with log-probability bias tuning regressed to 0.633 under nested CV; a Crystal-Graph-Neural-Network over skeletonized dendrites reached only 0.37 F1 alone. Three initially-reported claims (0.67–0.69) all reduced to ≤0.65 once red-team nested CV was applied.

**Contribution.** (i) An open-source orchestrated-agent pipeline for rapid ML-research on small medical datasets; (ii) an honest performance ceiling for tear-AFM disease classification at 240 scans / 35 persons; (iii) an interpretability layer via LLM-reasoning and retrieval-augmented prompts that issues clinical rationales on uncertain cases without additional API cost.

**Limitations.** 240 scans is a hard data ceiling — classes with 2–4 unique persons (SucheOko, Diabetes) cannot reach satisfying F1 regardless of model complexity. Classifier is 5-class only; if the hidden test set contains 9 classes from the original challenge brief (Alzheimer, Bipolar, Panic, Cataract, Pigment Dispersion), an open-set strategy would be required.
