# Data Audit — TRAIN_SET

Dátum: 2026-04-18. Zdroj: `TRAIN_SET.zip` (3.2 GiB) z `https://temp.kotol.cloud/?c=7IDU`.

## TL;DR (kľúčové zistenia)

1. **Len 5 tried** (nie 9 ako v PDF slides): Diabetes, PGOV_Glaukom, SklerozaMultiplex, SucheOko, ZdraviLudia.
2. **240 raw AFM skenov** (Bruker Nanoscope SPM formát) + 240 renderovaných BMP preview.
3. **Iba 44 unikátnych pacientov** — SucheOko má len 2, Diabetes 4. **Patient-level CV split je nevyhnutnosť.**
4. **BMP preview má vpálené popisky** (axis `0.0 / 92.5 μm`) → CNN sa môže naučiť watermark namiesto biológie.
5. **Silne nevyvážené triedy:** SM 95 vs. SucheOko 14 (7:1).
6. **Nejednotné scan parametre:** rozlíšenie 256² až 4096², scan range 10–92.5 μm, obdĺžnikové skeny tiež existujú.

## 1. Štruktúra dát

```
TRAIN_SET/
├── Diabetes/           25 raw .NNN + 27 BMP
├── PGOV_Glaukom/       36 raw .NNN + 35 BMP
├── SklerozaMultiplex/  95 raw .NNN + 95 BMP
├── SucheOko/           14 raw .NNN + 13 BMP
└── ZdraviLudia/        70 raw .NNN + 70 BMP
```

- **Raw AFM:** Bruker Nanoscope formát (`\*File list\\Version: 0x08150200`). Extensions `.001`, `.002`, ... kde číslo je scan index v rámci session. Čítateľné Python lib `AFMReader.spm.load_spm()`.
- **BMP:** 704×575 RGB uint8, rendered preview so scale bar a axis labels priamo v obrázku (*potenciálna data leakage*).

## 2. Distribúcia tried

| Trieda | Scany | % | BMP |
|---|---:|---:|---:|
| SklerozaMultiplex | 95 | 40 % | 95 |
| ZdraviLudia | 70 | 29 % | 70 |
| PGOV_Glaukom | 36 | 15 % | 35 |
| Diabetes | 25 | 11 % | 27 |
| SucheOko | 14 | 6 % | 13 |

Imbalance ratio 7:1 → **focal loss + class weights** (gamma=2.0, class_weight=balanced).

## 3. Patient-level štruktúra (kritické!)

| Trieda | Pacienti | Scany | Scany/pac |
|---|---:|---:|---:|
| ZdraviLudia | 21 | 70 | 3.3 |
| SklerozaMultiplex | 12 | 95 | 7.9 |
| PGOV_Glaukom | 5 | 36 | 7.2 |
| Diabetes | 4 | 25 | 6.2 |
| **SucheOko** | **2** | **14** | **7.0** |
| **Spolu** | **44** | **240** | **5.5** |

**SucheOko má LEN 2 pacientov!** V `StratifiedGroupKFold(5)` to znamená, že pri fold-splitte nebude mať niektorá fold žiadny SucheOko pacienta. Budeme musieť alebo:
- zmenšiť k na 2, alebo
- použiť **leave-one-patient-out CV** pre malé triedy, alebo
- considér každý scan od pacienta ako augmentation (patient-level split ale image-level training).

Jeden pacient má ~8 scanov z tej istej relácie (napr. `DM_01.03.2024_LO.001` až `.009`) — to sú temporálne blízke skeny toho istého vzorku, takže **silne korelované**. Naivný per-image split by viedol k ~20-30 % nafúknutej F1 na validácii.

## 4. Rozmery a scan parametre

### Rozlíšenia (H×W)
| Shape | n |
|---|---:|
| 1024×1024 | 117 |
| 512×512 | 72 |
| 256×256 | 9 |
| 2944×2944 | 9 |
| 3840×3840 | 5 |
| 2560×2560 | 2 |
| 3456×3456 | 2 |
| obdĺžnikové (432×1024, 3722×7040, ...) | 16 |
| ostatné štvorcové | 8 |

### Scan range (fyzikálna veľkosť)
| Scan μm | n | % |
|---|---:|---:|
| 92.5 μm | 187 | 78 % |
| 50.0 μm | 32 | 13 % |
| 20.0 μm | 8 | 3 % |
| 10.0 μm | 3 | 1 % |
| ostatné (38–80 μm) | 10 | 4 % |

**Implikácia:** 10× rozpätie scan sizes znamená, že 1 pixel pri 10 μm skene = 40 nm, pri 92.5 μm = 360 nm. Textúrové features (GLCM pri `distances=[1,3]`) sa vtedy porovnávajú úplne odlišné fyzikálne mierky.

**Fix:** normalizuj na konštantný pixel-to-nm (napr. 90 nm/px) resample pred feature extraction, alebo multi-scale architektúra/ensemble per scale.

### Výška (Z, nm)
Float64, typicky range desiatky stoviek nm (`±300–700 nm` je bežné). Niektoré skeny majú scanner tilt → **plane leveling (1st-order fit)** je povinný preprocessing step.

## 5. Vizuálna inšpekcia

Renderované vzorky: `reports/samples/*.png` (BMP grid), `reports/raw_samples/*.png` (raw height + BMP side-by-side).

**Morfologické pozorovania (kvalitatívne):**

- **ZdraviLudia:** klasické husté dendritické "fern-like" siete, radiálne star-burst vzory.
- **SucheOko:** fragmentovanejšie, menej husté, kratšie vetvy (konzistentné s Masmali grade 3–4).
- **Diabetes:** dendritické, ale hrubšie vetvy, hustejšie packing, viac "solid" kryštalické oblasti.
- **Glaukom:** odlišné — granulárne, roztrúsené čiastočky, menej klasického dendritu (MMP-9 efekt).
- **SklerozaMultiplex:** veľmi heterogénne vnútri triedy — občas hrubé lineárne kryštály (rods), občas jemné granule.

Zrakom sú triedy naozaj odlišné → ML úloha je realistická.

## 6. Riziká a mitigácie

| Riziko | Dopad | Mitigácia |
|---|---|---|
| **Patient leakage** vo val splite | Nafúknutá F1 o 20–30 % | `GroupKFold` / `StratifiedGroupKFold` s `group=patient_id` |
| **BMP watermark** (axis labels) | CNN učí rozpoznať scan range miesto biológie | Používať raw SPM alebo crop centrálnej oblasti BMP |
| **Scan-size confound** | Textúrové features porovnávajú rôzne mierky | Resample na konštantné px/μm pred feature extraction |
| **Scanner tilt** (neskorigovaný plane) | Globálny gradient prebíja lokálny detail | 1st-order polynomial plane leveling |
| **Class imbalance 7:1** | Model ignoruje minoritné triedy | Class weights + focal loss + stratified sampling |
| **Within-class variabilita SM** | Potenciálne multi-subtypes | Embedding-space cluster analýza — prípadne sub-klasifikácia |
| **Len 2 pacienti pre SucheOko** | Test F1 pre SucheOko môže divergovať | Silná augmentácia, prípadne exclude/merge s inou triedou ak pravidlá dovolia |

## 7. Odporúčaný preprocessing pipeline

```python
def preprocess(spm_path) -> np.ndarray:
    arr, px_nm = load_spm(spm_path, channel='Height')
    arr = level_plane_1st_order(arr)          # remove scanner tilt
    arr = resample_to_pixel_size(arr, px_nm,  # unify physical scale
                                  target_nm_per_px=90.0)
    arr = robust_normalize(arr, p_low=2, p_high=98)
    arr = center_crop_or_tile(arr, size=512)
    return arr
```

Pre CNN vstup potom multi-channel:
```python
x = stack([arr, sobel_x(arr), sobel_y(arr), |grad|, laplacian(arr)])
```

## 8. Otázky pre organizátorov (priorita poradie)

1. **Je hidden test set patient-disjoint od TRAIN_SETu?** (rozhoduje medzi patient-level vs. image-level split)
2. Je úloha **5-class** alebo sa v hidden sete objavia aj ďalšie choroby z PDF (Alzheimer, bipolárka, panika, cataract, PDS)?
3. Aký je **formát hidden input** — raw SPM alebo BMP? (rozhoduje či sa oplatí raw pipeline)
4. Single-label alebo multi-label (komorbidity)?
5. Je povolené používať externé pretrénované modely (BiomedCLIP, DINOv2)?
6. Musí klasifikátor bežať **plne offline** na Windows, alebo môže volať API?
7. Submission format — skript, Docker, notebook, .exe?

## 9. Next actions

- [ ] Implementovať `patient_id(path)` utility a uložiť `patient_id` column v metadata table.
- [ ] Baseline: **BiomedCLIP linear probe** na centrálnych 512×512 croppoch raw SPM height mapy.
- [ ] Parallel: **GLCM + LBP + fractal dimension + Ra/Rq → XGBoost** na tých istých 512² cropoch.
- [ ] Patient-level `StratifiedGroupKFold(5)` — alebo LOPO (leave-one-patient-out) pre SucheOko.
- [ ] Prototype **Crystal Graph Neural Network** na skeletonizovanom AFM height map.
