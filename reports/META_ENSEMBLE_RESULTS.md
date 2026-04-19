# Meta-Ensemble Voting — Results

**Question:** does ANY combination of previously-evaluated OOF predictions beat the v4 multi-scale champion (person-LOPO weighted F1 = **0.6887**)?

## Inputs (aligned on 240 scans, 35 persons)

| Source | Model key | Standalone W-F1 | Standalone M-F1 |
|---|---|---:|---:|
| `cache/...` | `v4_multiscale` | 0.6887 | 0.5541 |
| `cache/...` | `v2_2comp` | 0.6346 | 0.4934 |
| `cache/...` | `cascade_stage1` | 0.6346 | 0.4934 |
| `cache/...` | `cascade_stacker_blend` | 0.6451 | 0.5033 |
| `cache/...` | `mcv2_dinov2_height` | 0.6463 | 0.5373 |
| `cache/...` | `mcv2_dinov2_rgb` | 0.6192 | 0.5179 |
| `cache/...` | `mcv2_biomedclip_tta` | 0.6220 | 0.4915 |

## Strategies

| # | Strategy | W-F1 | M-F1 | Δ vs v4 (0.6887) | Leakage risk |
|---|---|---:|---:|---:|---|
| S1 | **S1_uniform_mean** | 0.6626 | 0.5309 | -0.0261 | none — simple arithmetic mean of softmaxes |
| S2 | **S2_geometric_mean** | 0.6685 | 0.5376 | -0.0202 | none — log-space mean of softmaxes |
| S3 | **S3_f1_weighted_mean** | 0.6672 | 0.5336 | -0.0215 | mild — weights use full-OOF F1 (global scalar per model) |
| S4 | **S4_rank_vote** | 0.6529 | 0.5112 | -0.0358 | none — argmax majority vote |
| S5 | **S5_meta_lr_nested_probs** | 0.6132 | 0.5378 | -0.0755 | controlled — nested person-LOPO meta-LR on concat probas (N*5 feats) |
| S5 | **S5_meta_lr_INSAMPLE_leaky** | 0.7317 | 0.6942 | +0.0430 | HIGH — same LR fit and scored in-sample (shown ONLY to expose optimism gap) |
| S6 | **S6_meta_lr_nested_onehot** | 0.5724 | 0.5002 | -0.1163 | controlled — nested person-LOPO meta-LR on one-hot argmax (N*5 feats) |

## Per-class F1

| Strategy | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|:---:|:---:|:---:|:---:|:---:|
| **S1_uniform_mean** | 0.8859 | 0.5652 | 0.5333 | 0.6703 | 0.0000 |
| **S2_geometric_mean** | 0.9007 | 0.5909 | 0.5263 | 0.6703 | 0.0000 |
| **S3_f1_weighted_mean** | 0.8919 | 0.5652 | 0.5333 | 0.6774 | 0.0000 |
| **S4_rank_vote** | 0.8859 | 0.4545 | 0.5455 | 0.6703 | 0.0000 |
| **S5_meta_lr_nested_probs** | 0.8611 | 0.6275 | 0.5581 | 0.5200 | 0.1224 |
| **S5_meta_lr_INSAMPLE_leaky** | 0.9130 | 0.7692 | 0.6265 | 0.6623 | 0.5000 |
| **S6_meta_lr_nested_onehot** | 0.8707 | 0.5333 | 0.5385 | 0.4430 | 0.1154 |

## Red-team analysis

- **S1, S2, S4** have no tuning: they combine cached OOFs with a deterministic rule. Their weighted-F1 is a direct read of the underlying predictions' agreement structure, not an inflated in-sample fit.
- **S3** uses each model's full-OOF weighted-F1 as a global scalar weight. Those weights are scalars computed from the full 240-row OOF set; they do NOT depend on any held-out person's labels in a sample-specific way, but the same full OOFs appear in both the weight estimate and the combination. Treat the number as mildly optimistic (< 0.002 typical gap vs a nested-weight variant; prior red-team in `reports/RED_TEAM_ENSEMBLE_AUDIT.md` shows this).
- **S5, S6** would be MAJOR leak offenders if naively fit on the full OOF and then scored on it. We instead refit the meta-LR for each held-out person on the other persons' OOF rows only — a true nested person-LOPO. The leaky ref row (`S5_..._INSAMPLE_leaky`) is included to show the size of the optimism gap when the nesting is skipped.

## Verdict

- Best HONEST strategy: **S2_geometric_mean** with W-F1 = 0.6685 (Δ vs v4 = -0.0202).
- **No honest strategy beats v4.** The meta-ensemble approach over these cached OOFs does not surface a new champion. v4 (multi-scale DINOv2-B 90+45 nm + BiomedCLIP-TTA, geom-mean) remains the best honest model at 0.6887 W-F1 / 0.5541 M-F1.

- **Optimism gap for S5 (leaky vs nested):** 0.7317 − 0.6132 = **+0.1185**. Any stacker score reported without nesting would be inflated by roughly this much — a textbook leakage trap on a 240-row OOF.
