# Topological + Morphology Feature Track (Wave-6.5)
## Goal
Provide a genuinely orthogonal signal to the v4 DINOv2/BiomedCLIP champion. TDA persistent homology reads connectivity of the height field across all thresholds; morphology (multifractal, lacunarity, succolarity) reads self-similarity and directional open-path structure. Neither shares a backbone with v4.
## Feature Composition
- Persistent homology (sublevel+superlevel, two scales, H0+H1, PI + landscape + stats + Betti curves) — 1015 dims, cached from `teardrop.topology.persistence_features`.
- Multifractal D(q) spectrum (q ∈ {-5..10}, 5 scales) — 13 dims.
- Gliding-box lacunarity (10 box sizes + slope + mean) — 12 dims.
- Directional succolarity (4 thresholds × 4 dirs + 4 anisotropy) — 20 dims.
- **Total: 1060 features per scan.**

## Results (person-LOPO via StratifiedGroupKFold k=5, groups=person)
| Model | Weighted F1 | Macro F1 | Notes |
|---|---|---|---|
| v4 multiscale (baseline) | 0.6887 | 0.5541 | DINOv2@90nm + DINOv2@45nm + BiomedCLIP-TTA |
| TDA+morphology XGB | 0.5310 | 0.3744 | standalone, 1060-d features |
| **v4 ⊙ TDA (geom-mean)** | **0.6262** | **0.4915** | equal weight, no stacker |
| Δ vs v4 | -0.0624 | -0.0625 |  |

## Per-class Weighted F1 (fusion vs v4)
| Class | v4 | TDA | v4 ⊙ TDA | Δ |
|---|---|---|---|---|
| ZdraviLudia | 0.917 | 0.752 | 0.887 | -0.029 |
| Diabetes | 0.583 | 0.105 | 0.450 | -0.133 |
| PGOV_Glaukom | 0.579 | 0.411 | 0.500 | -0.079 |
| SklerozaMultiplex | 0.691 | 0.604 | 0.620 | -0.071 |
| SucheOko | 0.000 | 0.000 | 0.000 | +0.000 |

## Per-fold (5-fold SGKF) wF1
| Fold | TDA wF1 |
|---|---|
| 0 | 0.5486 |
| 1 | 0.4976 |
| 2 | 0.5992 |
| 3 | 0.4678 |
| 4 | 0.5411 |

## Paired Bootstrap (1000×) — fused vs v4
- Δ wF1 mean = **-0.0639**  CI95 = [-0.1111, -0.0230]
- P(Δ > 0) = **0.001**
- Fused wF1 CI95 = [0.5572, 0.6909]
- v4 wF1    CI95 = [0.6308, 0.7525]

## Error Orthogonality
| Case | Count | % |
|---|---|---|
| both correct | 111 | 46.2% |
| v4-only correct | 56 | 23.3% |
| TDA-only correct | 20 | 8.3% |
| neither correct | 53 | 22.1% |

Expected both-correct under independence = 91.2, observed = 111. Observed > expected → models make **correlated** errors (both tend to fail on the same "hard" scans, e.g. SucheOko). Only 20 scans would benefit from TDA's unique signal, and even the equal-weight geometric mean averages down v4's stronger vote on the other 167 scans where v4 is right.

## Ablation: Tempered Fusion Variants
Weight the TDA softmax lower to check if any mixture beats v4 alone. Weighted geometric mean `v4^a * tda^b` normalized:

| a | b | Fused wF1 |
|---|---|---|
| 1.00 | 0.50 | 0.6590 |
| 1.00 | 0.30 | 0.6607 |
| 1.00 | 0.20 | 0.6698 |
| 1.00 | 0.10 | 0.6854 |
| 1.00 | 0.00 | 0.6887 (= v4 alone) |

Weighted arithmetic (sum-rule):

| w_v4 | Fused wF1 |
|---|---|
| 0.50 | 0.6192 |
| 0.70 | 0.6607 |
| 0.80 | 0.6798 |
| 0.90 | 0.6868 |
| 0.95 | 0.6887 (= v4 alone within rounding) |

**No mixture exceeds v4 alone.** TDA is 15 wF1 points below v4 in absolute terms, and its unique-correct set (20 scans) is too small to compensate.

## Verdict
**No fusion benefit.** Fused wF1=0.6262 < 0.68 ceiling for v4 alone, and P(Δ>0)=0.001. Single-track v4 remains champion. TDA retained as **pitch-level diversity asset**: its value is interpretability (H_1 loop statistics are physically meaningful — closed dendrite cells — and class-discriminative signatures exist especially for PGOV_Glaukom where sparse high-persistence loops appear). This is NOT an F1 win, but it is honest evidence that the v4 champion captures most of the signal available in 240 scans and that adding a truly-orthogonal weak learner does not help with only ~8% unique-correct contribution.

## Artifacts
- `scripts/tda_features.py` — this script
- `teardrop/topology.py` — persistent-homology extractor
- `cache/tda_features.npz` — 240 × 1060 float32 matrix aligned to v4 order
- `cache/features_tda.parquet` — cached PH features
- `cache/features_morphology.parquet` — cached morphology features
- `cache/tda_predictions.json` — standalone XGB OOF softmax
- `cache/tda_fusion_predictions.json` — v4 ⊙ TDA fused OOF softmax
