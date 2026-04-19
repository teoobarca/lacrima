# Independent Validation Audit

Author: independent reviewer (no knowledge of team's claimed numbers).
Embedding source: `cache/tiled_emb_dinov2_vits14_afmhot_t512_n9.npz`.

---

## Task 1 — Patient ID parsing audit

### Function under test
```python
def patient_id(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".bmp"):
        name = name[:-4]
        if name.endswith("_1"):
            name = name[:-2]
    name = _SCAN_INDEX_RE.sub("", name)  # strips trailing .NNN
    return name
```

### Dataset stats (computed)

- Total raw SPM scans detected: **240**
- Total unique patient IDs (per parser): **44**

Per-class:

| class | scans | patients (unique IDs) |
|---|---:|---:|
| ZdraviLudia | 70 | 21 |
| Diabetes | 25 | 4 |
| PGOV_Glaukom | 36 | 5 |
| SklerozaMultiplex | 95 | 12 |
| SucheOko | 14 | 2 |

### 30-filename sample

| filename | parsed_patient_id | class | look_correct | reason |
|---|---|---|---|---|
| `48.003` | `48` | ZdraviLudia | Y | OK |
| `7.001` | `7` | ZdraviLudia | Y | OK |
| `5P.001` | `5P` | ZdraviLudia | Y | OK |
| `48.007` | `48` | ZdraviLudia | Y | OK |
| `1L_M.002` | `1L_M` | ZdraviLudia | Y | OK |
| `9L.001` | `9L` | ZdraviLudia | Y | OK |
| `Dusan1_DM_STER_mikro_281123.012` | `Dusan1_DM_STER_mikro_281123` | Diabetes | Y | OK |
| `DM_01.03.2024_LO.008` | `DM_01.03.2024_LO` | Diabetes | Y | dots in patient id (date kept) - one session per patient assumed |
| `Dusan1_DM_STER_mikro_281123.013` | `Dusan1_DM_STER_mikro_281123` | Diabetes | Y | OK |
| `Dusan1_DM_STER_mikro_281123.010` | `Dusan1_DM_STER_mikro_281123` | Diabetes | Y | OK |
| `Dusan1_DM_STER_mikro_281123.008` | `Dusan1_DM_STER_mikro_281123` | Diabetes | Y | OK |
| `Dusan2_DM_STER_mikro_281123.005` | `Dusan2_DM_STER_mikro_281123` | Diabetes | Y | OK |
| `21_LV_PGOV+SII.000` | `21_LV_PGOV+SII` | PGOV_Glaukom | Y | OK |
| `23_LV_PGOV.001` | `23_LV_PGOV` | PGOV_Glaukom | Y | OK |
| `25_PV_PGOV.015` | `25_PV_PGOV` | PGOV_Glaukom | Y | OK |
| `21_LV_PGOV+SII.001` | `21_LV_PGOV+SII` | PGOV_Glaukom | Y | OK |
| `26_PV_PGOV.007` | `26_PV_PGOV` | PGOV_Glaukom | Y | OK |
| `25_PV_PGOV.014` | `25_PV_PGOV` | PGOV_Glaukom | Y | OK |
| `1-SM-LM-18.002` | `1-SM-LM-18` | SklerozaMultiplex | Y | OK |
| `100_8-SM-PV-18.002` | `100_8-SM-PV-18` | SklerozaMultiplex | Y | OK |
| `1-SM-LM-18.007` | `1-SM-LM-18` | SklerozaMultiplex | Y | OK |
| `22_PV_SM.003` | `22_PV_SM` | SklerozaMultiplex | Y | OK |
| `19_PM_SM.070` | `19_PM_SM` | SklerozaMultiplex | Y | OK |
| `19_SM_MK.001` | `19_SM_MK` | SklerozaMultiplex | Y | OK |
| `29_PM_suche_oko.000` | `29_PM_suche_oko` | SucheOko | Y | OK |
| `29_PM_suche_oko.009` | `29_PM_suche_oko` | SucheOko | Y | OK |
| `29_PM_suche_oko.007` | `29_PM_suche_oko` | SucheOko | Y | OK |
| `35_PM_suche_oko.000` | `35_PM_suche_oko` | SucheOko | Y | OK |
| `29_PM_suche_oko.008` | `29_PM_suche_oko` | SucheOko | Y | OK |
| `29_PM_suche_oko.005` | `29_PM_suche_oko` | SucheOko | Y | OK |

### Failure-mode probes

**Are there filenames where two genuinely different patients get the same ID?**  
No collisions across class folders. Every patient ID belongs to exactly one class. Good.

**Are one patient's scans split across multiple IDs (eye sides treated as separate patients)?**  
This is a serious risk — the parser treats the LM/PM/LV/PV/LO/PO eye-side suffix as part of the patient ID.
So scans from the **same person** but **different eyes** get separate IDs — i.e. the same person is in train AND val.

Suspected single-person-split-into-two-IDs (eye-side pairs found in dataset):
- `1-SM-LM-18`  <->  `1-SM-PM-18`

ZdraviLudia uses bare `<n>L` / `<n>P` IDs — same-person/different-eye pairs:
- `2L`  <->  `2P`
- `5L`  <->  `5P`
- `6L`  <->  `6P`
- `8L`  <->  `8P`
- `9L`  <->  `9P`

**Cross-reference probe — does `DM_01.03.2024_LO.001`..`.009` look like 9 scans of one session, or 9 patients?**  
The parser collapses them into a single ID `DM_01.03.2024_LO` (verified: 9 files share that ID). 
This matches my reading: a date + eye code (`LO`=ľavé oko) is one session of one patient. Good.

### Verdict on Task 1

- The **trailing `.NNN` collapse is correct** — sequential scans of one session map to one patient ID.
- **BUT the eye-side suffix (LM/PM/LV/PV/LO/PO and trailing L/P in ZdraviLudia) is NOT collapsed**, so left and right eye of the same physical person become separate "patients". This is leakage by any clinical definition: the two eyes of one human are not statistically independent — they share genetics, systemic disease, age, hydration, diet, sample-prep batch, scan day. LOPO with the current parser is really leave-one-eye-out, not leave-one-person-out.
- The team's claim of "~44 unique patients" is inflated by this. Many of the 44 IDs are pairs of eyes from the same person (e.g. `1-SM-LM-18`/`1-SM-PM-18`, `100_7-SM-LV-18`/`100_8-SM-PV-18`, `1L_M`/`1P`, etc.). True patient count is likely closer to 20–30.

**Recommended fix:** Strip eye-side suffix tokens before grouping. Concretely, after dropping `.NNN`, also strip a trailing `_(LM|PM|LV|PV|LO|PO)` token, and for ZdraviLudia strip a trailing single `L`/`P` after a numeric ID. Then re-run LOPO. If F1 drops noticeably, the previous score was leakage-inflated. If it stays the same, the model was already eye-agnostic.

---

## Task 2 — Label-shuffle null baseline

- Embedding cache: tile-level X = (811, 384), scan-level after mean-pool = (240, 384).
- Number of classes: 5, naïve uniform-random baseline ≈ 0.200.
- Classifier: StandardScaler + LogisticRegression(class_weight='balanced', max_iter=2000), LOPO.

**TRUE-label LOPO weighted F1: `0.5959`**

**Shuffled-label LOPO (5 seeds, scan_y permuted, scan_groups intact):**

| seed | weighted F1 |
|---:|---:|
| 0 | 0.2696 |
| 1 | 0.3003 |
| 2 | 0.2645 |
| 3 | 0.3353 |
| 4 | 0.2098 |
| **mean ± std** | **0.2759 ± 0.0416** |

**Gap (true − shuffled): `0.3200` → PASS — true F1 exceeds shuffled by 0.320, well above noise. The classifier is learning real label-correlated structure.**

---

## Task 3 — CV variance estimation (RepeatedStratifiedGroupKFold)

- 5-fold × 5 repeats = 25 fold scores.
- Patient-grouped, class-stratified.

| repeat | fold | n_train | n_val | weighted F1 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 191 | 49 | 0.7742 |
| 0 | 1 | 195 | 45 | 0.6440 |
| 0 | 2 | 193 | 47 | 0.6213 |
| 0 | 3 | 189 | 51 | 0.6273 |
| 0 | 4 | 192 | 48 | 0.6104 |
| 1 | 0 | 188 | 52 | 0.7816 |
| 1 | 1 | 195 | 45 | 0.6440 |
| 1 | 2 | 193 | 47 | 0.6033 |
| 1 | 3 | 193 | 47 | 0.6075 |
| 1 | 4 | 191 | 49 | 0.5625 |
| 2 | 0 | 192 | 48 | 0.7567 |
| 2 | 1 | 193 | 47 | 0.6437 |
| 2 | 2 | 191 | 49 | 0.4948 |
| 2 | 3 | 192 | 48 | 0.5925 |
| 2 | 4 | 192 | 48 | 0.5050 |
| 3 | 0 | 188 | 52 | 0.5653 |
| 3 | 1 | 197 | 43 | 0.7630 |
| 3 | 2 | 196 | 44 | 0.6863 |
| 3 | 3 | 188 | 52 | 0.7124 |
| 3 | 4 | 191 | 49 | 0.5514 |
| 4 | 0 | 191 | 49 | 0.7298 |
| 4 | 1 | 192 | 48 | 0.6815 |
| 4 | 2 | 193 | 47 | 0.6274 |
| 4 | 3 | 192 | 48 | 0.5551 |
| 4 | 4 | 192 | 48 | 0.5966 |

**Mean ± std weighted F1: `0.6375 ± 0.0795`**
  - min: `0.4948`, max: `0.7816`
  - LOPO single number from Task 2: `0.5959`

Distribution histogram (10 bins, 0.0..1.0):

```
  [0.00, 0.10)    0  
  [0.10, 0.20)    0  
  [0.20, 0.30)    0  
  [0.30, 0.40)    0  
  [0.40, 0.50)    1  #
  [0.50, 0.60)    7  #######
  [0.60, 0.70)   11  ###########
  [0.70, 0.80)    6  ######
  [0.80, 0.90)    0  
  [0.90, 1.00)    0  
```

**Variance verdict:** Moderate variance — single-seed claims are misleading without error bars.

---

## Final overall verdict

Results have **real signal** above shuffled-label noise, with reasonable cross-fold stability.

**However, the patient-id parser silently treats the two eyes of one person as different patients** (see Task 1).
LOPO and "patient-grouped" KFold are therefore really *leave-one-eye-out* and *eye-grouped* — same human can sit on both sides of the split. 
This is data leakage and most likely *inflates* every reported number, including the ones above. 
Until that is fixed, the magnitudes of the F1 numbers should not be quoted as honest patient-level performance.

### Recommendations to the team

1. **Fix `patient_id`** to also strip eye-side tokens (`LM`, `PM`, `LV`, `PV`, `LO`, `PO`, and trailing `L`/`P` after numeric IDs).
2. **Re-run LOPO with the fixed parser** and compare F1. The drop is your true leakage estimate. Report both numbers.
3. **Always report mean±std over ≥5 seeds** of repeated K-fold; do not quote single-LOPO numbers without a variance estimate.
4. **Always include a label-shuffle null baseline** in the same table as real F1 — anything within ~2σ of shuffled is not a result.
5. **Manually map filenames to people** (a human-readable spreadsheet of `filename | person | eye | session_date | class`). For 240 files / ~25 people this is one afternoon and removes ALL guessing.
6. **Be skeptical of weighted-F1 with imbalanced classes** — SucheOko has only 2 putative patients; LOPO removing one of them leaves the model with a single positive example and the score is dominated by majority classes. Report per-class F1 too.
