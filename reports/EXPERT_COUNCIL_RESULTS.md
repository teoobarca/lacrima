# Expert Council — multi-classifier + LLM judge

Run: 2026-04-18 21:33   Model: `claude-haiku-4-5`   Workers: 8

## 1. Architecture

Each of the 240 AFM tear scans is processed person-LOPO by three
independent experts plus a quantitative morphology summary. The LLM
judge receives all of the above along with the anonymised query image
and outputs a single predicted class.

| Expert | Signal | Training | Output |
|---|---|---|---|
| 1. Vision ensemble (v4) | DINOv2-B TTA @ 90 nm + DINOv2-B @ 45 nm + BiomedCLIP-TTA, L2-normed, geomean of softmaxes | person-LOPO 5-fold CV | 5-way softmax |
| 2. k-NN retrieval | DINOv2-B scan embeddings, cosine | top-5 nearest w/ SAME person excluded | similarity-weighted votes |
| 3. XGBoost morphology | 440 handcrafted features (GLCM×4 scales, LBP, fractal D, multifractal, lacunarity, gabor, wavelet packet, HOG, Hurst/DFA, surface stats) | person-LOPO StratifiedGroupKFold(5) | 5-way softmax |
| 4. Morphology panel | fractal D, Sq, Ssk, GLCM d1, Masmali grade heuristic, HOG mean, lacunarity slope | — | quantitative summary in prompt |
| Judge | Claude Haiku 4.5 (CLI, no API key) | — | JSON {predicted_class, confidence, reasoning} |

Key safeguards:
- Images referenced ONLY by `scan_XXXX.png` (obfuscated); true class
  never appears in the prompt or path.
- k-NN neighbour set explicitly excludes scans from the query's person
  (person-LOPO retrieval).
- XGBoost OOF uses `StratifiedGroupKFold(group=person_id)` — no
  same-person leakage.
- v4 softmaxes are pre-computed OOF; each scan's probability comes from
  the fold where its person was the held-out group.

## 2. Run statistics

- Total scans: **240**
- Judge calls that returned parseable JSON: **240**
- Wall time (LLM): **755.0s**
- Estimated API cost (Haiku 4.5, ~3.0k input + 200 out / call, $1/$5 per MTok): **~$0.960**

## 3. Headline F1 comparison

| Expert | Weighted F1 | Macro F1 |
|---|---:|---:|
| Expert 1 — v4 vision ensemble | 0.6887 | 0.5541 |
| Expert 2 — k-NN retrieval | 0.5859 | 0.4396 |
| Expert 3 — XGBoost morphology | 0.5458 | 0.3856 |
| **Judge (Haiku 4.5)** | **0.5912** | **0.4308** |
| Delta (judge - v4) | -0.0975 | -0.1233 |

Person-LOPO reported F1 for v4 baseline was 0.6887 (doc). On this
run, v4 weighted F1 = 0.6887 (sanity check).

## 4. Bootstrap 1000x vs v4

Paired bootstrap on weighted F1 (resample 240 scans w/ replacement):

- Mean Delta F1w (judge - v4) = **-0.0974**
- 95% CI: [-0.1519, -0.0472]
- P(Delta > 0) = **0.000**

**Verdict:** judge probably hurts F1. Keep for reasoning/audit only; do NOT override vision ensemble.

## 5. Per-class F1 (Judge)

| Class | F1 |
|---|---:|
| ZdraviLudia | 0.8140 |
| Diabetes | 0.2424 |
| PGOV_Glaukom | 0.4308 |
| SklerozaMultiplex | 0.6667 |
| SucheOko | 0.0000 |

```
                   precision    recall  f1-score   support

      ZdraviLudia      0.686     1.000     0.814        70
         Diabetes      0.500     0.160     0.242        25
     PGOV_Glaukom      0.483     0.389     0.431        36
SklerozaMultiplex      0.660     0.674     0.667        95
         SucheOko      0.000     0.000     0.000        14

         accuracy                          0.633       240
        macro avg      0.466     0.445     0.431       240
     weighted avg      0.586     0.633     0.591       240

```

## 6. Confusion matrix (Judge)

| true \ pred | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|---:|---:|---:|---:|---:|
| ZdraviLudia | 70 | 0 | 0 | 0 | 0 |
| Diabetes | 19 | 4 | 0 | 2 | 0 |
| PGOV_Glaukom | 1 | 0 | 14 | 21 | 0 |
| SklerozaMultiplex | 8 | 4 | 15 | 64 | 4 |
| SucheOko | 4 | 0 | 0 | 10 | 0 |

## 7. Per-person F1 (Judge, weighted)

- Mean: 0.7318   Median: 0.9333   Min: 0.0000   Max: 1.0000
- Fraction of persons with F1 >= 0.8: 62.86%
- Fraction of persons with F1 == 0.0: 14.29%

## 8. Error analysis

### 8a. Judge disagrees with ALL three experts (3 cases)

Of these, **0** are judge-correct (rescue) and **3** are judge-wrong (overrule experts unsuccessfully).

| key | true | v4 | kNN | XGB | judge | correct? | reasoning |
|---|---|---|---|---|---|:-:|---|
| scan_0009 | PGOV_Glaukom | PGOV_Glaukom | PGOV_Glaukom | PGOV_Glaukom | ZdraviLudia | NO | All three experts converge on PGOV_Glaukom as top-1 pick (avg confidence 0.708), but quantitative morphology directly contradicts this diagnosis. Fractal D=1.773 falls squarely wit |
| scan_0186 | SklerozaMultiplex | SucheOko | SucheOko | SklerozaMultiplex | ZdraviLudia | NO | Vision experts (1 & 2) strongly favor SucheOko (0.935 and 0.599), but quantitative morphology powerfully contradicts: Fractal D 1.827 falls perfectly within healthy range (1.70–1.8 |
| scan_0122 | SklerozaMultiplex | SucheOko | SucheOko | SklerozaMultiplex | ZdraviLudia | NO | Vision experts (1 & 2) strongly converge on SucheOko (0.976, 0.602), but quantitative morphology contradicts this: fractal D=1.827 is HIGH and within the healthy range (1.70-1.85), |

### 8b. Judge CORRECTED a wrong majority (4 cases)

Cases where >=2 of 3 experts agreed on a wrong answer but the judge
pulled the prediction back to the true class.

| key | true | majority-wrong | judge | reasoning |
|---|---|---|---|---|
| scan_0147 | SklerozaMultiplex | PGOV_Glaukom | SklerozaMultiplex | Expert 1 (VLM) strongly votes PGOV_Glaukom (0.992), backed by Expert 2 (k-NN, 0.606). However, Expert 3 (XGBoost morphology) confidently picks SklerozaMultiplex (0.649) and cites direct quantitative f |
| scan_0174 | SklerozaMultiplex | PGOV_Glaukom | SklerozaMultiplex | Vision experts converge on PGOV_Glaukom (95.4%, 61.1%), but XGBoost morphology (68.9%) contradicts this. Critical evidence: GLCM correlation 0.999 is incompatible with Glaukom's expected LOW correlati |
| scan_0206 | SklerozaMultiplex | PGOV_Glaukom | SklerozaMultiplex | Expert disagreement is stark: Expert 1 (DINOv2+BiomedCLIP) strongly favors MS (0.930) while Experts 2 and 3 favor glaucoma (0.58 and 0.902). However, the quantitative morphology reveals a critical inc |
| scan_0103 | ZdraviLudia | Diabetes | ZdraviLudia | Experts 1 & 2 (vision-based) favor Diabetes, but Expert 3 (XGBoost morphology) very strongly predicts ZdraviLudia (p=0.949). Quantitative support: Fractal D=1.751 sits at the midpoint of healthy range |

### 8c. Judge OVERRULED a correct majority (4 cases)

Cases where >=2 of 3 experts agreed on the correct answer but the
judge flipped to a wrong class — these are the failure mode.

| key | true | majority-right | judge | reasoning |
|---|---|---|---|---|
| scan_0226 | Diabetes | Diabetes | ZdraviLudia | Visual inspection reveals dense, highly-branched dendritic ferning with organized fine structure throughout the field, consistent with healthy tear ferning. Quantitative support: fractal dimension 1.7 |
| scan_0190 | Diabetes | Diabetes | ZdraviLudia | Experts 1 and 2 (vision-based) favor Diabetes, but Expert 3 (morphology) strongly picks ZdraviLudia (0.861), a judgment supported by quantitative morphology: fractal D=1.774 is at the high end of heal |
| scan_0009 | PGOV_Glaukom | PGOV_Glaukom | ZdraviLudia | All three experts converge on PGOV_Glaukom as top-1 pick (avg confidence 0.708), but quantitative morphology directly contradicts this diagnosis. Fractal D=1.773 falls squarely within healthy range (1 |
| scan_0046 | SklerozaMultiplex | SklerozaMultiplex | SucheOko | Experts 1–2 (vision-based DINOv2/BiomedCLIP, k-NN) unanimously vote SklerozaMultiplex (p=1.0), but Expert 3 (XGBoost morphology) conservatively picks SucheOko (p=0.658) with SklerozaMultiplex at 0.330 |

## 9. Files written

- `cache/expert_council_predictions.json` — one record per scan
  (all expert outputs, morphology, judge response, reasoning).
- `reports/EXPERT_COUNCIL_RESULTS.md` — this file.
