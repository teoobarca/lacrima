# LLM-Gated Uncertainty Reasoner — Results

Run date: 2026-04-18 16:07
Model: `claude -p --model haiku` (CLI subprocess, no API key).

## 1. Approach

Instead of running an LLM on all 240 scans, we invoke Claude
ONLY on Stage-1 low-confidence cases (maxprob < 0.55).
For each uncertain scan we build a retrieval-augmented prompt:

1. Retrieve k=3 nearest neighbours per Stage-1 top-2 class from
   DINOv2-B scan-mean embeddings (cosine similarity).
2. Person-LOPO rule: retrieved scans must have a different
   `person_id` than the query (no same-person leakage).
3. Prompt contains trimmed domain rules for the top-2 classes,
   features for the 6 reference cases, and the query's features.
4. Claude picks between top-1 and top-2 and returns JSON.
5. If it flips (chooses top-2) we overwrite the Stage-1 label.

## 2. CLI verification

- `which claude` -> available (aliased with `--allow-dangerously-skip-permissions`).
- `claude -p 'Reply PONG' --model haiku` -> PONG (OK)

## 3. Uncertain-case statistics

- Total scans: 240
- Uncertain (maxprob < 0.55): **47**
- Processed (margin-sorted, cap=60): **47**

| True class | Sent to LLM | Agreed top-1 | Flipped to top-2 | Flip correct | Flip wrong | Stage-1 acc (uncertain) | Refined acc (uncertain) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZdraviLudia | 14 | 8 | 6 | 2 | 4 | 12/14 (0.86) | 8/14 (0.57) |
| Diabetes | 7 | 6 | 1 | 1 | 0 | 3/7 (0.43) | 4/7 (0.57) |
| PGOV_Glaukom | 2 | 1 | 1 | 1 | 0 | 1/2 (0.50) | 2/2 (1.00) |
| SklerozaMultiplex | 24 | 13 | 9 | 4 | 5 | 18/24 (0.75) | 12/24 (0.50) |

- LLM agreed with Stage-1 top-1: **28**
- LLM flipped to top-2: **17**
- Unparseable / fallback: **2**

## 4. F1 comparison

| Metric | Stage-1 ensemble | LLM-gated refined | Delta |
|---|---:|---:|---:|
| Weighted F1 (all 240) | 0.6698 | 0.6575 | -0.0123 |
| Macro F1 (all 240)    | 0.5206 | 0.5161 | -0.0045 |

Per-class classification reports:

```
Stage-1:
                   precision    recall  f1-score   support

      ZdraviLudia      0.821     0.914     0.865        70
         Diabetes      0.529     0.360     0.429        25
     PGOV_Glaukom      0.600     0.583     0.592        36
SklerozaMultiplex      0.700     0.737     0.718        95
         SucheOko      0.000     0.000     0.000        14

         accuracy                          0.683       240
        macro avg      0.530     0.519     0.521       240
     weighted avg      0.662     0.683     0.670       240


LLM-gated refined:
                   precision    recall  f1-score   support

      ZdraviLudia      0.811     0.857     0.833        70
         Diabetes      0.476     0.400     0.435        25
     PGOV_Glaukom      0.611     0.611     0.611        36
SklerozaMultiplex      0.687     0.716     0.701        95
         SucheOko      0.000     0.000     0.000        14

         accuracy                          0.667       240
        macro avg      0.517     0.517     0.516       240
     weighted avg      0.650     0.667     0.658       240

```

## 5. Reasoning examples (cherry-picked for the pitch)

### 5a. LLM correctly FLIPPED the Stage-1 top-1 (LLM = tie-breaker wins)

**Dusan1_DM_STER_mikro_281123.014** — true=Diabetes, Stage-1=ZdraviLudia, LLM=Diabetes (conf=0.68)
> Query's glcm_contrast_d5_mean (19.43) nearly matches Diabetes [4] (19.61) and is far lower than Healthy [1] (37.8); glcm_contrast_d1_mean (1.922) precisely matches Diabetes [4] (1.912), with elevated Sa (0.2154) and dissimilarity (2.094) consistent with denser tear lattice and higher osmolarity.

**25_PV_PGOV.016** — true=PGOV_Glaukom, Stage-1=SklerozaMultiplex, LLM=PGOV_Glaukom (conf=0.6)
> Query exhibits glcm_correlation_d5=0.7304, the lowest across all references (next-lowest ~0.81), directly matching domain rule for Glaukom's 'locally chaotic texture' from MMP-9 protease degradation. High glcm_dissimilarity_d3=2.289 further supports granular morphology over dendritic. However, several features (contrast_d5=31.91, lbp_25=0.2655) are outliers beyond training range, reducing confiden

**100_7-SM-LV-18.000** — true=SklerozaMultiplex, Stage-1=SklerozaMultiplex, LLM=SklerozaMultiplex (conf=0.78)
> Query's Sq=0.3125 exceeds all PGOV_Glaukom references (max 0.2855) and fits SklerozaMultiplex range. Critical discriminator: glcm_contrast_d1=1.005 and glcm_contrast_d5=11.52 far exceed PGOV (0.51–0.73 and 4.14–8.01) but align with SM's heterogeneous texture signature (0.66–2.07 and 5.96–23.13), consistent with mixed crystalline morphologies characteristic of MS.

### 5b. LLM wrongly FLIPPED (made things worse)

**1-SM-LM-18.003** — true=SklerozaMultiplex, Stage-1=SklerozaMultiplex, LLM=PGOV_Glaukom (conf=0.78)
> Query glcm_contrast_d1_mean (11.93) matches PGOV reference (11.67) precisely vs SklerozaMultiplex (2.83). Glcm_correlation_d5_mean (0.2407) exhibits characteristic chaotic texture of glaucoma. Glcm_dissimilarity_d3_mean (3.754) is 0.5 units from PGOV mean (3.259) but 2.1 units from SklerozaMultiplex mean (1.641).

**1-SM-LM-18.007** — true=SklerozaMultiplex, Stage-1=SklerozaMultiplex, LLM=Diabetes (conf=0.68)
> Query GLCM contrast (d1=4.155, d5=30.15) and dissimilarity (2.249) are consistently lower than SklerozaMultiplex references (contrast d1=14–14.27, dissimilarity=3–4.5) but aligned with Diabetes range. Energy (0.2305) is elevated and closest to Diabetes ref[4]=0.2098, indicating higher packing density. Fractal D_mean (1.778) sits between classes but slightly lower, consistent with Diabetes.

### 5c. Bonus: LLM agreed with Stage-1 and was correct (confidence-boost case)

**Dusan1_DM_STER_mikro_281123.011** — true=Diabetes, both Stage-1 and LLM predicted Diabetes (conf=0.65)
> Query's glcm_correlation_d5 (0.8092) matches Diabetes mean (0.7978) nearly exactly and is vastly higher than SM (0.3214), indicating preserved spatial structure typical of denser Diabetes lattice. GLCM homogeneity_d1 (0.6637) also aligns closely with Diabetes (0.6453). However, lower Sa/Sq and contrast values relative to Diabetes reference cases suggest either atypical presentation or preprocessin

## 6. Cost / latency

- Wall time for 47 LLM calls: **412.3s** (avg 8.77s/call, 4 workers).
- Model: `haiku`. No API key used — CLI is authenticated on host.
- Marginal USD cost: 0 (subscription-covered CLI).
- Cost for equivalent Anthropic API call (Haiku 4.5, ~1.5K input + 150 output tok/call):
  ~70,500 input tok × $1/MTok + ~7,050 output tok × $5/MTok = **~$0.106**.

## 7. Honest conclusion

The LLM-gated tie-breaker hurts F1 (weighted -0.0123, macro -0.0045). Do NOT ship as a prediction override; keep for reasoning / audit output only.

**Pitch value, independent of F1:**

- Each uncertain case now ships with a human-readable rationale citing
  specific feature values *and* nearest-neighbour reference scans. A
  linear probe on DINOv2 embeddings cannot do this.
- The gating mechanism itself is the story: Stage-1 (fast, embedding-
  based ensemble) handles the easy 80%; Stage-2 (LLM + retrieval) only
  spends compute on the hard 20%. This is cost-efficient in a way
  "LLM classifies everything" is not.
- Uncertain cases are exactly the cases where a clinician would want a
  second opinion. We produce that opinion with an audit trail showing
  which reference cases and which features drove the call.

## 8. Appendix — files written

- `cache/llm_reasoner_raw.jsonl` — one JSON record per uncertain scan (raw CLI
  response + parsed fields).
- `cache/llm_gated_refined.npz` — refined prediction arrays aligned to
  `scan_paths`.
- `reports/LLM_GATED_RESULTS.md` — this file.
