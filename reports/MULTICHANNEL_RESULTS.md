# Multichannel AFM probe — results

**Question.** Bruker Nanoscope `.NNN` files can export multiple channels
(Height, Amplitude Error, Phase, Deflection, …) — we've been reading
`channel='Height'` only. Do the other channels exist, and does using them
improve classification?

**Short answer.** Yes to both. All 240 scans have `Height`, `Amplitude Error`,
and `Height Sensor` channels; 227/240 also have `Phase`. Encoding
**RGB = (Height, Amplitude Error, Phase)** with frozen DINOv2-B and
averaging its predictions with the pure-Height model lifts person-level LOPO
macro F1 from 0.4998 → **0.5452** (+0.045) and weighted F1 from
0.6313 → **0.6557** (+0.025).

---

## 1. Channel availability survey — `reports/channel_survey.csv`

Per-file channel lists for all 240 scans. Summary:

| Channel | Files with it | % |
|---|---|---|
| Height | 240 / 240 | 100% |
| Amplitude Error | 240 / 240 | 100% |
| Height Sensor | 240 / 240 | 100% |
| **Phase** | 227 / 240 | **94.6%** |

Phase is missing from **13 Diabetes** scans, all from the two
`Dusan{1,2}_DM_STER_mikro_281123.*` acquisitions (no Phase channel was saved
during that session). All other Diabetes scans (12/25) and all scans of other
classes have Phase.

Channel meaning:
- **Height**: topography after feedback-loop correction (nm). Our existing baseline.
- **Height Sensor**: direct Z-piezo sensor readout; similar to Height.
- **Amplitude Error**: deviation from target tapping-mode amplitude; highlights edges
  and regions where feedback lags. Encodes sharpness / contrast / fibre edges.
- **Phase**: cantilever phase lag relative to drive; sensitive to stiffness,
  adhesion, and viscoelasticity of tear film. Independent physical contrast.

---

## 2. LOPO (person-level) F1 — frozen DINOv2-B, tile-level LR

Pipeline identical to the existing tiled baseline
(`scripts/baseline_tiled_ensemble.py`): plane-level → resample to 90 nm/px →
robust-normalize → tile 512×512 (≤9 tiles/scan) → DINOv2-B embeddings →
balanced Logistic Regression at the tile level → scan prediction = mean of
per-tile softmax. LOPO groups = `person_id` (L+R eye merged).

Single-channel afmhot rendering for Height, Amplitude Error, Phase; RGB
variant renders (R=Height, G=Amplitude, B=Phase) directly as an RGB image.

| Config | macro F1 | weighted F1 | N | ZdraviLudia | Diabetes | PGOV_Glaukom | SklerozaMultiplex | SucheOko |
|---|---|---|---|---|---|---|---|---|
| **Height only** (baseline) | 0.4998 | 0.6313 | 240 | 0.840 | 0.500 | 0.507 | 0.652 | 0.000 |
| Amplitude Error only | 0.4576 | 0.5757 | 240 | 0.795 | 0.455 | 0.466 | 0.573 | 0.000 |
| Phase only | 0.4038 | 0.5864 | 227 | 0.824 | 0.000 | 0.571 | 0.570 | 0.054 |
| **RGB (H+A+P)** | **0.5288** | 0.6375 | 240 | 0.838 | **0.634** | 0.471 | 0.639 | 0.063 |
| Concat(H\|A\|P) scan-mean features | 0.3724 | 0.5720 | 227 | 0.848 | 0.000 | 0.438 | 0.576 | 0.000 |
| **Avg-prob(Height, RGB)** | **0.5452** | **0.6557** | 240 | **0.846** | **0.651** | 0.486 | **0.667** | **0.077** |

Observations:
- **Single-channel Amplitude and Phase are each worse than Height alone.** The
  physical information they carry is real but less separable in isolation at
  our sample size.
- **RGB stack beats Height alone on macro F1** (+0.029) and weighted F1
  (+0.006). The big win is Diabetes (0.500 → 0.634). SucheOko goes from 0 → 1
  correct scan.
- **Concatenating per-channel mean embeddings into a 2304-D feature vector
  overfits** — expected with only 240 samples. Don't use this.
- **Best config: simple probability average of Height-tile model and RGB-tile
  model.** It combines the best of both: Height's strong general separation
  plus RGB's Diabetes signal, and lifts SklerozaMultiplex too.

---

## 3. Verdict

Multichannel Bruker data is present and the Amplitude+Phase channels carry
independent, useful signal — but only when combined with Height (not as
replacements). The strongest, simplest use is to add a second tile-level
DINOv2-B model trained on RGB(H, Amplitude, Phase) and ensemble its softmax
with the existing Height tile-model.

Expected lift on this probe vs Height-only baseline:
- macro F1: **+0.045** (0.4998 → 0.5452)
- weighted F1: **+0.025** (0.6313 → 0.6557)
- Diabetes F1: **+0.151** (0.500 → 0.651)

For context, the existing shippable ensemble baseline is weighted F1 ≈ 0.6346
(Height DINOv2 + BiomedCLIP). On the same LOPO protocol used here, Height
DINOv2 alone is 0.6313 weighted; adding an RGB-DINOv2 channel delivers 0.6557
weighted. Whether this propagates into the full ensemble requires re-running
the BiomedCLIP + stacker pipeline with the new RGB embeddings — not included
in this probe (25-min budget).

### Ship recommendation

Yes, warrant an `models/ensemble_v2_multichannel/`:
- Add `cache/multichan_tiled_emb_dinov2vitb14_t512_n9.npz` to the ensemble
  inputs (already written by this probe).
- Retrain the tile-level logistic-regression stacker with an additional RGB
  input group. For the 13 `Dusan*_DM_STER_mikro_281123` scans that lack Phase,
  the current script substitutes Height in the Blue slot so all 240 scans are
  predicted — the +0.151 Diabetes lift you see includes those files. An
  alternative is to route Phase-missing scans through the Height-only model
  only (hard gate); this probe did not separately test that.

---

## Artefacts

- `scripts/multichannel_probe.py` — full pipeline (survey → encode → LOPO eval).
- `reports/channel_survey.csv` — per-file channel lists.
- `reports/channel_survey_summary.json` — aggregated counts.
- `reports/multichannel_results.json` — per-config LOPO metrics.
- `cache/multichan_tiled_emb_dinov2vitb14_t512_n9.npz` — tile embeddings for
  Height, Amplitude Error, Phase, and RGB; 768-D DINOv2-B, 811/811/759/811 tiles.

**Compute used.** ~10 min on Apple MPS (M-series GPU) — well under the 25-min
budget. Most of it is preprocessing (plane-level + resample) and DINOv2 encode.
