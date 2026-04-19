# Confident-Triage Metrics — Hospital Deployment Framing

Re-frames the champion v4 ensemble's raw 0.69 weighted F1 as a triage curve for hospital deployment: **if the model is confident above threshold T, it handles the case autonomously; otherwise the case is routed to a specialist for review.**  This script operates entirely on Platt-calibrated, person-LOPO OOF predictions — no retraining, pure post-hoc analysis.

## Methodology

- **Source:** `cache/v4_oof_predictions.npz` (240 scans × 5 classes, person-LOPO OOF softmaxes from the v4 ensemble = DINOv2-B @ 90 nm/px + DINOv2-B @ 45 nm/px + BiomedCLIP TTA).
- **Calibration:** per-class one-vs-rest Platt (logistic on logit-raw softmax) fit with the **same** person-LOPO protocol used in `METRICS_UPGRADE.md`.  No leakage between scans of the same person.
- **Per-patient aggregation:** softmaxes averaged over each patient's scans; threshold applied to the aggregated top-1 confidence.
- **Triage rule:** accept prediction iff `max softmax >= T`.  Accepted → model handles autonomously.  Rejected → refer to specialist.

## 1. Headline operating points — hospital deployment

The **patient level** is the clinically correct reporting level (one prediction per patient, not per AFM frame), so we pitch those numbers first:

| Target accuracy | Coverage (auto) | Realized accuracy | Threshold T | Referred to specialist |
|---|---|---|---|---|
| **80%** (patient) | **85.7%** (30/35) | **80.0%** | 0.491 | 5/35 (14.3%) |
| **90%** (patient) | **8.6%** (3/35) | **100.0%** | 0.791 | 32/35 (91.4%) |
| **95%** (patient) | **8.6%** (3/35) | **100.0%** | 0.791 | 32/35 (91.4%) |

Per-scan view (each of 240 AFM frames treated as an independent case):

| Target accuracy | Coverage (auto) | Realized accuracy | Threshold T | Referred to specialist |
|---|---|---|---|---|
| 80% (scan, Platt) | **unfeasible** | — | — | 240/240 |
| 90% (scan, Platt) | **unfeasible** | — | — | 240/240 |
| 95% (scan, Platt) | **unfeasible** | — | — | 240/240 |

**Baseline (no triage, T=0):** per-scan accuracy = 68.3%, weighted F1 = 0.655; per-patient accuracy = 77.1%, weighted F1 = 0.741.  These are the 'status quo' numbers from METRICS_UPGRADE.md.

**Honest note on per-scan Platt results:** Platt calibration smooths over-confident extremes (by design — that's what makes probabilities trustworthy) but erases the small slice of ultra-confident per-scan predictions the model gets right.  As a result, per-scan Platt accuracy tops out at ~73% (its running-accuracy curve never reaches 80%).  See Section 6 for raw-softmax triage, which is less well-calibrated but offers a cleaner accuracy-coverage envelope at the very top.

## 2. Confidence-vs-accuracy sweep (per-scan)

| Threshold T | Coverage | n accepted | Accuracy | Weighted F1 | Macro F1 | Rejected accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 100.0% | 240/240 | 68.3% | 0.655 | 0.503 | — |
| 0.40 | 96.2% | 231/240 | 69.3% | 0.665 | 0.510 | 44.4% |
| 0.50 | 81.2% | 195/240 | 70.8% | 0.676 | 0.477 | 57.8% |
| 0.60 | 66.7% | 160/240 | 73.1% | 0.698 | 0.466 | 58.8% |
| 0.70 | 49.2% | 118/240 | 72.9% | 0.680 | 0.385 | 63.9% |
| 0.80 | 27.1% | 65/240 | 66.2% | 0.608 | 0.350 | 69.1% |
| 0.90 | 6.7% | 16/240 | 62.5% | 0.625 | 0.350 | 68.8% |

*'Rejected accuracy' = raw argmax accuracy on cases below threshold (i.e. what the doctor would see from the model as a 'low-confidence' suggestion).*

### Per-class coverage × accuracy per threshold (per-scan)

| T | Healthy cov / acc | Diabetes cov / acc | Glaucoma cov / acc | Multiple Sclerosis cov / acc | Dry Eye cov / acc |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 100.0% / 94.3% | 100.0% / 44.0% | 100.0% / 38.9% | 100.0% / 76.8% | 100.0% / 0.0% |
| 0.40 | 97.1% / 95.6% | 76.0% / 47.4% | 100.0% / 38.9% | 98.9% / 76.6% | 100.0% / 0.0% |
| 0.50 | 88.6% / 95.2% | 48.0% / 25.0% | 83.3% / 40.0% | 83.2% / 81.0% | 85.7% / 0.0% |
| 0.60 | 77.1% / 96.3% | 36.0% / 22.2% | 55.6% / 30.0% | 71.6% / 83.8% | 64.3% / 0.0% |
| 0.70 | 65.7% / 100.0% | 24.0% / 0.0% | 38.9% / 21.4% | 47.4% / 82.2% | 50.0% / 0.0% |
| 0.80 | 31.4% / 100.0% | 12.0% / 0.0% | 25.0% / 11.1% | 27.4% / 76.9% | 35.7% / 0.0% |
| 0.90 | 1.4% / 100.0% | 0.0% / — | 5.6% / 0.0% | 12.6% / 75.0% | 7.1% / 0.0% |

## 3. Per-patient sweep (n=35)

| Threshold T | Patients accepted | Coverage | Accuracy | Weighted F1 |
|---:|---:|---:|---:|---:|
| 0.30 | 35/35 | 100.0% | 77.1% | 0.741 |
| 0.40 | 34/35 | 97.1% | 76.5% | 0.729 |
| 0.50 | 29/35 | 82.9% | 82.8% | 0.788 |
| 0.60 | 23/35 | 65.7% | 78.3% | 0.737 |
| 0.70 | 12/35 | 34.3% | 75.0% | 0.726 |
| 0.80 | 2/35 | 5.7% | 100.0% | 1.000 |
| 0.90 | 1/35 | 2.9% | 100.0% | 1.000 |

**Per-patient baseline (T=0):** accuracy = 77.1%, weighted F1 = 0.741.

### Per-patient operating points for target accuracies

| Target | Coverage | Accuracy | Threshold T | Patients accepted |
|---:|---:|---:|---:|---:|
| 80% | 85.7% | 80.0% | 0.491 | 30/35 |
| 90% | 8.6% | 100.0% | 0.791 | 3/35 |
| 95% | 8.6% | 100.0% | 0.791 | 3/35 |

## 4. Per-class calibration tendency (scan level)

Who is overconfident?  'Calibration gap' = mean top-1 confidence when the model predicts this class, minus precision on that prediction.  Positive gap = overconfident (model is wrong more often than it thinks).

| Class | n predicted | Precision | Mean conf on pred | Gap (conf-prec) | Tendency |
|---|---:|---:|---:|---:|---|
| Healthy | 78 | 0.846 | 0.696 | -0.151 | under-confident |
| Diabetes | 20 | 0.550 | 0.490 | -0.060 | under-confident |
| Glaucoma | 30 | 0.467 | 0.660 | +0.193 | over-confident |
| Multiple Sclerosis | 111 | 0.658 | 0.700 | +0.042 | well-calibrated |
| Dry Eye | 1 | 0.000 | 0.439 | +0.439 | over-confident |

## 4b. Per-class calibration tendency (patient level)

| Class | n predicted | Precision | Mean conf on pred | Gap | Tendency |
|---|---:|---:|---:|---:|---|
| Healthy | 16 | 0.938 | 0.647 | -0.290 | under-confident |
| Diabetes | 2 | 1.000 | 0.400 | -0.600 | under-confident |
| Glaucoma | 3 | 0.667 | 0.631 | -0.035 | well-calibrated |
| Multiple Sclerosis | 14 | 0.571 | 0.657 | +0.086 | over-confident |
| Dry Eye | 0 | — | — | — | never-predicted |

## 5. Hospital deployment narrative

**Status quo:** ophthalmologist manually inspects every AFM scan (100% human review).  Time cost = `N × t_scan` per patient.

**Triage deployment:** the v4 model runs on every scan; only cases below a confidence threshold T are routed to the specialist.  Confident cases get an autonomous prediction + calibrated probability.

### Target-accuracy operating envelope (per-scan)

- **80% target accuracy: NOT achievable** on any non-empty coverage slice — the calibrated OOF predictions never cross this accuracy floor.  (The per-scan ceiling is ~68.3% with full coverage.)

- **90% target accuracy: NOT achievable** on any non-empty coverage slice — the calibrated OOF predictions never cross this accuracy floor.  (The per-scan ceiling is ~68.3% with full coverage.)

- **95% target accuracy: NOT achievable** on any non-empty coverage slice — the calibrated OOF predictions never cross this accuracy floor.  (The per-scan ceiling is ~68.3% with full coverage.)


### Patient-level deployment envelope

- **80% target (patient):** T ≈ 0.491 → 85.7% of patients (30/35) handled autonomously at 80.0% accuracy; 5/35 referred.

- **90% target (patient):** T ≈ 0.791 → 8.6% of patients (3/35) handled autonomously at 100.0% accuracy; 32/35 referred.

- **95% target (patient):** T ≈ 0.791 → 8.6% of patients (3/35) handled autonomously at 100.0% accuracy; 32/35 referred.


## 6. Raw-softmax operating points (for comparison)

Platt calibration makes the PROBABILITIES trustworthy but SMOOTHS the top of the confidence distribution (the hottest confidences 0.99+ get pulled toward the empirical bin accuracy). That smoothing can erase the separation between 'confidently correct' and 'confidently wrong' at the very top of the curve (see the reliability diagram).  For a triage rule that leverages extreme confidence values, the RAW softmax often offers a better accuracy-coverage envelope, at the cost of probabilities that are no longer well-calibrated in the Brier/ECE sense.  We report both:

### Raw per-scan sweep

| T | Coverage | Accuracy | Weighted F1 |
|---:|---:|---:|---:|
| 0.30 | 100.0% | 69.6% | 0.689 |
| 0.40 | 100.0% | 69.6% | 0.689 |
| 0.50 | 99.2% | 70.2% | 0.693 |
| 0.60 | 94.2% | 70.8% | 0.698 |
| 0.70 | 85.8% | 71.4% | 0.705 |
| 0.80 | 79.2% | 72.6% | 0.716 |
| 0.90 | 65.0% | 75.0% | 0.741 |

### Raw per-scan target-accuracy operating points

| Target | Coverage | Accuracy | Threshold T | n accepted |
|---:|---:|---:|---:|---:|
| 80% | 16.7% | 80.0% | 0.999 | 40/240 |
| 90% | 0.4% | 100.0% | 1.000 | 1/240 |
| 95% | 0.4% | 100.0% | 1.000 | 1/240 |

### Raw per-patient target-accuracy operating points

| Target | Coverage | Accuracy | Threshold T | n accepted |
|---:|---:|---:|---:|---:|
| 80% | 100.0% | 82.9% | 0.538 | 35/35 |
| 90% | 42.9% | 93.3% | 0.810 | 15/35 |
| 95% | 31.4% | 100.0% | 0.882 | 11/35 |

## 7. Pitch one-liners

- **"At a patient-level confidence of 0.49, the model classifies **30/35 patients (86% of the cohort) autonomously at 80% accuracy**; the remaining 5/35 low-confidence cases are routed to a specialist."**

- **"At the 90% target the model is autonomous on 3/35 patients (9% of the cohort) with 100% observed accuracy — a small but fully-autonomous 'green-light' slice."**

- **"Raw-softmax per-patient triage: **93% accuracy on 43% of patients (15/35)** autonomously, confidence threshold T ≥ 0.81."**

- **"At 95% target accuracy (patient-level, raw softmax), the model is autonomous on **11/35 patients (31%) with 100% observed accuracy** — a high-confidence green-light slice."**

- **"Using raw-softmax confidence (T ≥ 1.00), the model handles **40/240 scans (17%) autonomously at 80% accuracy**, referring the remaining 200 ambiguous scans (83%) to the specialist."**

- **"Scan-level honesty note: after Platt calibration, no confidence threshold cleanly gives 80% accuracy — the per-scan ceiling is ~73%.  Triage works best at the patient level, where per-patient softmax averaging denoises the signal."**

- **"Status quo = doctor inspects **all 35 patients**. Triage at 80% accuracy removes **30/35 patients (86% of workload)** from the queue; specialist time is focused on the 5 ambiguous cases."**


## 8. Figures

- `reports/pitch/12_triage_curves.png` — accuracy-coverage curves (per-scan + per-patient, Platt-calibrated with raw per-scan dashed for comparison), reliability diagram with triage zones, deployment-narrative operating table, and threshold-sweep bar chart.
