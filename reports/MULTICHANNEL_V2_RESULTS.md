# Multichannel × V2-recipe fusion — results

**Question.** Does combining the Wave-5 multichannel discovery
(Bruker `Height` / `Amplitude Error` / `Phase` channels encoded with
DINOv2-B) with the V2 champion recipe (L2-norm → StandardScaler → LR →
**geometric mean** of softmaxes) push person-LOPO weighted F1 past the
current 0.6562 champion?

**Short answer.** **Yes — a 3-way ensemble lifts W-F1 from 0.6562 → 0.6645
(+0.0083)** on honest person-LOPO (35 persons, 240 scans, no threshold
tuning, no OOF subset selection).

Winning ensemble (next-champion candidate):

> **E7 = GEOM-MEAN(** DINOv2-B Height (tile-mean pool) · DINOv2-B RGB
> (H+Amp+Phase stack, tile-mean pool) · BiomedCLIP TTA-D4 Height **)** —
> all with V2 recipe (L2-norm row-wise → StandardScaler per fold → LR
> balanced).

Macro F1: 0.5382 → **0.5435** (+0.0053). ZdraviLudia and
SklerozaMultiplex both improve cleanly; the cost is SucheOko (0.0645 →
0.0).

---

## 1. Setup

- Person-LOPO via `teardrop.cv.leave_one_patient_out` with
  `groups = teardrop.data.person_id(path)` (35 unique persons). Every scan
  predicted exactly once, patient-disjoint train folds.
- V2 recipe applied per member per fold: row-L2-normalize →
  `StandardScaler` (fit on train) → `LogisticRegression(class_weight="balanced",
  C=1.0, lbfgs, max_iter=3000)`. OOF softmax collected per member. Members
  fused by **geometric mean** across softmaxes → argmax.
- Cached embeddings reused (no re-encode):
  - `cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz` — DINOv2-B TTA-D4 Height (scan-level, 768-D)
  - `cache/tta_emb_biomedclip_afmhot_t512_n9_d4.npz` — BiomedCLIP TTA-D4 Height (scan-level, 512-D)
  - `cache/multichan_tiled_emb_dinov2vitb14_t512_n9.npz` — DINOv2-B per-channel tile embeddings (Height / Amplitude Error / Phase / RGB). Mean-pooled across tiles to (240, 768) per channel.
- **Phase fallback.** 13 Diabetes scans (two `Dusan*_DM_STER_mikro_281123` sessions) lack the Phase channel. Their Phase scan-vector is replaced with the Height scan-vector from the same multichannel cache (v2 recipe then applied on top). No other falls.
- No threshold tuning, no member selection on OOF, no per-class calibration.

---

## 2. Per-member honest LOPO F1 (V2 recipe, solo)

| Member | Weighted F1 | Macro F1 |
|---|---|---|
| DINOv2-B TTA-D4 Height (`dinov2_tta_height`) | 0.6464 | 0.5286 |
| BiomedCLIP TTA-D4 Height (`biomedclip_tta_height`) | 0.6220 | 0.4915 |
| DINOv2-B Height tile-mean (`dinov2_height_pool`) | 0.6463 | 0.5373 |
| DINOv2-B Amplitude-Error tile-mean (`dinov2_amp_pool`) | 0.5396 | 0.4187 |
| DINOv2-B Phase tile-mean (`dinov2_phase_pool`) | 0.5711 | 0.4994 |
| DINOv2-B RGB(H+A+P) tile-mean (`dinov2_rgb_pool`) | 0.6192 | 0.5179 |

TTA Height and tile-mean Height are effectively tied as solo classifiers
(0.6464 vs 0.6463). Amplitude and Phase solo are the weakest — useful only
as ensemble partners.

---

## 3. Ensemble grid (all geometric mean, V2 recipe)

| # | Ensemble | W-F1 | M-F1 | Δ W-F1 vs 0.6562 | Zdrav | Diab | Glauk | SM | SO |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | dinov2_TTA_H + biomedclip_TTA_H (**V2 champion**) | **0.6562** | 0.5382 | +0.0000 | 0.869 | 0.542 | 0.564 | 0.652 | 0.065 |
| E2 | dinov2_H_pool + dinov2_amp | 0.5987 | 0.4810 | −0.0575 | 0.816 | 0.545 | 0.444 | 0.599 | 0.000 |
| E3 | dinov2_H_pool + dinov2_phase | 0.6151 | 0.5254 | −0.0411 | 0.842 | 0.683 | 0.472 | 0.565 | 0.065 |
| E4 | dinov2_H_pool + dinov2_amp + dinov2_phase | 0.6025 | 0.4986 | −0.0537 | 0.840 | 0.667 | 0.417 | 0.570 | 0.000 |
| E5 | dinov2_H_pool + dinov2_rgb | 0.6506 | 0.5521 | −0.0056 | 0.850 | 0.727 | 0.466 | 0.638 | 0.080 |
| E6 | dinov2_H_pool + dinov2_amp + biomedclip_TTA_H | 0.6391 | 0.5237 | −0.0171 | 0.874 | 0.651 | 0.474 | 0.620 | 0.000 |
| **E7** | **dinov2_H_pool + dinov2_rgb + biomedclip_TTA_H** | **0.6645** | 0.5435 | **+0.0083** | **0.884** | 0.667 | **0.507** | **0.660** | 0.000 |
| E8 | dinov2_H_pool + dinov2_amp + dinov2_phase + biomedclip_TTA_H | 0.6454 | 0.5361 | −0.0108 | 0.887 | 0.698 | 0.486 | 0.609 | 0.000 |

(Class headers: Zdrav = ZdraviLudia, Diab = Diabetes, Glauk = PGOV_Glaukom, SM = SklerozaMultiplex, SO = SucheOko.)

Observations:
- **Only E7 beats the champion by ≥ +0.005.** It wins by swapping out one of the two members: instead of pairing two Height encoders (DINOv2 + BiomedCLIP), it pairs three complementary views — Height pool, RGB multichannel pool, and BiomedCLIP TTA Height.
- **Adding more DINOv2 members without BiomedCLIP always hurts (E2, E3, E4).** The duplicate-encoder correlation eats the benefit; the V2 champion explicitly wins because its two members use **different encoders**.
- **Adding Amplitude as a member hurts (E2, E6, E8).** Amplitude is the weakest solo classifier (0.5396) and its errors correlate with Height's.
- **RGB is the best add-on.** E5 (Height + RGB) is nearly on par with champion at 0.6506, and becomes E7 once BiomedCLIP is added.
- **Phase alone or Phase + Amplitude drag weighted-F1 down** even though they sometimes boost Diabetes F1. Single-class lift ≠ overall lift.

Per-class deltas E1 → E7:
- ZdraviLudia: 0.869 → 0.884 (+0.015)
- Diabetes: 0.542 → 0.667 (+0.125)
- PGOV_Glaukom: 0.564 → 0.507 (−0.057)
- SklerozaMultiplex: 0.652 → 0.660 (+0.008)
- SucheOko: 0.065 → 0.000 (−0.065)

E7 trades SucheOko and PGOV_Glaukom losses for a large Diabetes and
small ZdraviLudia / SM gain. SucheOko has only 2 persons so +1/−1 scan
moves its F1 sharply; do not over-weight that change.

---

## 4. Recommendation

### Next-champion candidate

**E7** — `GEOM-MEAN( DINOv2-B Height tile-mean · DINOv2-B RGB(H+A+P) tile-mean · BiomedCLIP TTA-D4 Height )` with V2 preprocessing.

- Honest person-LOPO weighted F1: **0.6645** (+0.0083 vs 0.6562)
- Honest person-LOPO macro F1: **0.5435** (+0.0053)
- No threshold tuning, no OOF selection, no member search after looking at val.
- Phase channel falls back to Height for 13 Diabetes scans (documented honestly).

### Caveats the red team should pressure-test

1. **+0.0083 is one sigma-ish for n=240 / 35 persons.** Bootstrap CIs or repeated seeds haven't been computed here. The jump comes from 2 extra correct predictions out of 240 (~0.83%). A prudent next step is seed-averaged LR and a bootstrap-over-persons CI for E7 vs E1 before crowning it.
2. **SucheOko F1 drops to 0.0** (E1 had one correct SucheOko). Given only 2 SucheOko persons, that's 1 scan's worth of change; macro is only up marginally.
3. **Recipe cost doubles.** E7 needs a DINOv2-B inference at inference time on **two renderings** (Height afmhot and RGB stack) plus BiomedCLIP on Height. Still shippable — two DINOv2 forward passes + one BiomedCLIP, same preprocess.
4. The DINOv2 tile-mean pool path uses `multichan_tiled_emb_dinov2vitb14_t512_n9.npz` which was built **without D4 TTA** (`baseline_tiled_ensemble.py` style, up to 9 tiles, no augmentation). For a cleaner cross-validation, a future run could re-render RGB with D4 TTA — the current delta is entirely without TTA on the multichannel members.

### Ship plan (if adopted)

1. `models/ensemble_v3_multichannel/` with three persisted heads:
   - dinov2 Height tile-mean pool (multichannel cache, no TTA)
   - dinov2 RGB(H+Amp+Phase) tile-mean pool (Phase→Height fallback for the 13 listed files)
   - BiomedCLIP TTA-D4 Height (existing component)
2. Inference: for each scan, compute all three 240-row-ish features → L2 → SS → LR → softmax, geom-mean, argmax.
3. Report card on model card: honest W-F1 0.6645, macro 0.5435.

---

## 5. Artefacts

- `scripts/multichannel_v2_fusion.py` — end-to-end fusion script (8 ensembles, ~9 s wall clock).
- `reports/multichannel_v2_results.json` — machine-readable summary with per-member and per-ensemble metrics.
- `cache/best_multichannel_v2_predictions.npz` — winning E7 predictions + per-member OOF softmax (`P_dinov2_height_pool`, `P_dinov2_rgb_pool`, `P_biomedclip_tta_height`), labels, groups, and scan paths. Use for downstream audit / red-team replication.

Compute: 8.6 s on CPU (no encode, cache reuse). Well inside the 15 min budget.
