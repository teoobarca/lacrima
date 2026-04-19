# Patient-Level Classifier

## Motivation

Train a classifier where **each person is one sample** (35 samples, 34-dim-LOPO-friendly).
All scans of a person inherit its predicted label at inference. This directly matches
an evaluation regime in which the organizers grade per-patient. If they grade per-scan
with potentially diverse labels per patient (not our case in TRAIN_SET), this approach
would hurt — measured and reported here.

## Setup

- Encoders: DINOv2-B @ 90 nm/px + DINOv2-B @ 45 nm/px + BiomedCLIP D4 TTA.
- Per scan: L2-normalize 768/768/512-D embedding.
- Aggregate to one vector per person with three variants (mean / max / attention).
- Attention weights = v4 scan-level top-1 probability (softmax-normalized within person).
- Concatenate the three encoders per person → 2048-D.
- LOPO Logistic Regression (class_weight balanced, C=1, L2-norm + StandardScaler).
- 35 train/predict rounds, then broadcast to 240 scans for fair comparison with v4.

Reference: v4 scan-level wF1 = **0.6887**, macroF1 = 0.5541.

## Results

| Variant | Person wF1 | Person macroF1 | Scan wF1 (broadcast) | Scan macroF1 | Δ vs v4 | P(Δ>0) |
|---------|-----------:|---------------:|---------------------:|-------------:|--------:|-------:|
| mean | 0.8180 | 0.6781 | 0.8127 | 0.6844 | +0.1243 | 1.000 |
| max  | 0.7577 | 0.5799 | 0.7506 | 0.5918 | +0.0617 | 0.971 |
| attn | 0.8367 | 0.6838 | 0.8177 | 0.6902 | +0.1293 | 1.000 |

## Per-class F1 (best variant: `attn`)

| Class | F1 |
|-------|---:|
| ZdraviLudia | 1.0000 |
| Diabetes | 0.9362 |
| PGOV_Glaukom | 0.6957 |
| SklerozaMultiplex | 0.8190 |
| SucheOko | 0.0000 |

## Bootstrap vs v4 (1000 resamples, best variant)

- mean Δ = **+0.1293**
- P(Δ > 0) = **1.000**
- 95% CI = [+0.0815, +0.1765]

## Verdict

CHAMPION CANDIDATE (scan-level wF1 exceeds v4)

## Caveats

- Broadcasting the person-level prediction to every scan is fair **only if the
  organizers evaluate per-patient**. In our TRAIN_SET every scan of a given person
  has the same class anyway, so this is not a lossy assumption on our data, but the
  hidden test set may differ.
- Person-level F1 is on 35 samples, which gives a coarse signal (one misclassified
  person moves person-wF1 by ~3 points). The scan-broadcast metric above is the
  figure to compare against v4.