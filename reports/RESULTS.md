# Výsledky — Hack Košice 2026 / UPJŠ Tear Challenge

Aktualizované: 2026-04-18.

## TL;DR

**⚠ CORRECTION 2026-04-18:** Validator agent našiel bug v `patient_id` parseri — L/P oči toho istého človeka boli grupované ako rôzni pacienti. Opravené cez `person_id`. Reálne osoby: **35 (nie 44)**. Nové *person-level* LOPO F1 sú o ~1 % nižšie než predchádzajúce eye-level — signál je teda reálny, ale všetky staré čísla sú mierne nadhodnotené.

### Person-level LOPO (HONEST, 35 persons)

| Model | Weighted F1 | Macro F1 | Notes |
|---|---:|---:|---|
| **DINOv2-B tiled + LR** | **0.615** | **0.491** | 9 tiles/scan, mean-pool |
| DINOv2-S tiled + LR | 0.593 | 0.478 | 9 tiles/scan |
| BiomedCLIP tiled + LR | 0.580 | 0.434 | 9 tiles/scan |

### Eye-level LOPO (staré, nadhodnotené, zachované pre porovnanie)

| Model | Weighted F1 | Macro F1 | Notes |
|---|---:|---:|---|
| Handcrafted (94 feat) → XGBoost | 0.502 | 0.344 | GLCM+LBP+fractal+roughness |
| DINOv2-S linear probe | 0.582 | 0.451 | single 512² crop |
| DINOv2-B linear probe | 0.581 | 0.430 | single 512² crop |
| BiomedCLIP linear probe | 0.577 | 0.441 | single 512² crop |
| DINOv2-S tiled (tile-level train) | 0.628 | 0.501 | aggregation by scan-level proba mean |
| DINOv2-B tiled (tile-level train) | 0.600 | 0.439 | |
| BiomedCLIP tiled (tile-level train) | 0.566 | 0.411 | |

**Best honest F1: 0.615 (DINOv2-B tiled, scan-level mean pool, person-level LOPO).**

## Per-class F1 (LOPO, best model = DINOv2-S TILED)

| Trieda | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ZdraviLudia | 0.77 | 0.89 | **0.82** | 70 |
| SklerozaMultiplex | 0.66 | 0.65 | **0.66** | 95 |
| PGOV_Glaukom | 0.54 | 0.56 | **0.55** | 36 |
| Diabetes | 0.57 | 0.32 | **0.41** | 25 |
| SucheOko | 0.07 | 0.07 | **0.07** | 14 |

## Confusion matrix (LOPO, DINOv2-S TILED)

```
                      Zdraví Diab Glauk SM  Such
ZdraviLudia              62    3    0    3    2
Diabetes                 14    8    0    3    0
PGOV_Glaukom              0    0   20   15    1
SklerozaMultiplex         3    3   17   62   10
SucheOko                  2    0    0   11    1
```

**Najčastejšie zámeny:**
- SM ↔ Glaukom (17+15 = 32 misclassifications) — morfologicky podobné dendrity
- Diabetes → Zdraví (14) — model "pochybuje" pri Diabetes a defaultuje na majoritnú triedu
- SucheOko → SM (11/14) — tiež morfologicky podobné

## Vizualizácie a artefakty

- `reports/DATA_AUDIT.md` — komplet audit (44 pacientov, scan parametre, riziká)
- `reports/raw_afm_probe.csv` — metadata všetkých 240 SPM
- `reports/samples/*.png` — BMP grids per class
- `reports/raw_samples/*.png` — raw height + BMP side-by-side
- `cache/features_handcrafted.parquet` — 94 features × 240 scans
- `cache/emb_*.npz` — embeddings cached (DINOv2-S, DINOv2-B, BiomedCLIP × {single, tiled})

## Kľúčové zistenia

1. **Tiling >> single crop.** +5 % weighted F1, +5 % macro F1 zadarmo. Namiesto centra 512² používame 9 nepretínajúcich sa dlaždíc 512² → 9× viac efektívnych training samples + scan-level mean-pool aggregácia pri inferencii.
2. **Veľkosť foundation modelu je takmer irelevantná.** DINOv2-S (384-dim, 22M params) prekonáva DINOv2-B (768-dim, 86M). Pri 240 vzorkách väčší model = viac overfittingu, nie viac signálu.
3. **BiomedCLIP nie je výrazne lepší** než general DINOv2 napriek medical pretraining-u. Tear AFM nie je v jeho training distribution.
4. **SucheOko je fundamentálne limitované** — len 2 unikátni pacienti. Žiadny pretraining ani classifier head to neopraví. Treba buď:
   - (a) augmentation explosion (rotácie, fliplyt, scale jitter — efektívne 2 → 50+ "pseudo-pacientov"),
   - (b) synthetic data cez diffusion model conditioned na SucheOko vzore,
   - (c) one-vs-rest binary anomaly detection,
   - (d) priznať že táto trieda potiahne F1 dole.
5. **Diabetes sa dá vytiahnuť** — z 0.05 (handcrafted) na 0.41 (DINOv2 tiled). Ďalší priestor pre zlepšenie cez ensemble + thresholding.
6. **Glaukom ↔ SM zámeny** sú dominantné. Pravdepodobne nie sú opraviteľné single-image pristúpom; treba patient-level metadata alebo hierarchical klasifikáciu.

## Čo NIE JE ešte hotové (queued experiments)

### Rýchle (každé < 1 h)
- [ ] Concat ensemble: DINOv2-S + DINOv2-B + BiomedCLIP + handcrafted → XGBoost
- [ ] TTA pri inferencii — 8× rotácie/flipy, mean of predictions
- [ ] Per-class threshold tuning na OOF predictions
- [ ] MLP head namiesto LogisticRegression
- [ ] Weak augmentation per tile pri trénovaní (scan-time augmentation)
- [ ] Konfidenčný "abstain" mechanizmus pre SucheOko (predikuj len ak istý)

### Stredné (1–4 h)
- [ ] DINOv2-L (1024-dim) tiled — overiť či máme priestor
- [ ] Render mode experiment — RGB grayscale vs `afmhot` vs gradient overlay
- [ ] 5-channel input (height + ∂x + ∂y + |∇| + Δ) cez tenkú CNN klasifikačnú hlavu
- [ ] Strong tile-level augmentation pri tréningu (rotácia + flip + crop scale)
- [ ] Repeated StratifiedGroupKFold s 5 seedmi — variance reduction validačnej metriky

### Ambiciózne (novel tracky, 4–12 h)
- [ ] **Crystal Graph NN** — skeletonizácia AFM výškovej mapy → graf → GIN klasifikácia (interpretabilný, novel)
- [ ] **TDA / persistent homology** — `giotto-tda` features → ensemble member
- [ ] **Multifractal spectrum + lacunarity** — full f(α) signature místo single fractal D
- [ ] **LLM reasoning layer** — extrahované features + domain knowledge → Claude vracia interpretované klasifikácie s zdôvodnením (pitch killer)
- [ ] **Self-supervised pretraining** — DINO pretraining na vlastnom datasete pred linear probe
- [ ] **Synthetic SucheOko via diffusion** — conditional Stable Diffusion finetuning

## Implementačná infraštruktúra (hotová)

```
teardrop-challenge/
├── teardrop/                    # core modul
│   ├── data.py                 # SPM load, plane level, resample, normalize, tile, patient_id
│   ├── cv.py                   # StratifiedGroupKFold, leave-one-patient-out
│   ├── encoders.py             # DINOv2, BiomedCLIP, OpenCLIP wrappers
│   └── features.py             # GLCM, LBP, fractal, roughness, HOG
├── scripts/
│   ├── data_audit.py
│   ├── probe_raw_afm.py
│   ├── visualize_samples.py
│   ├── visualize_raw_afm.py
│   ├── baseline_handcrafted_xgb.py
│   ├── baseline_foundation_probe.py
│   ├── baseline_tiled_ensemble.py
│   └── baseline_concat_ensemble.py
├── cache/                       # cached embeddings + features
└── reports/                     # vizualizácie, audit, výsledky
```

## Risk register (aktualizovaný)

| Riziko | Status | Mitigácia |
|---|---|---|
| Patient leakage | ✓ vyriešené | LOPO + StratifiedGroupKFold |
| BMP watermark | ✓ vyriešené | používame raw SPM |
| Scan-size confound | ✓ vyriešené | resample na 90 nm/px |
| Class imbalance | ⚠ stále aktívne | class_weight='balanced', threshold tuning TODO |
| SucheOko 2-patient limit | ✗ neriešené | augmentation explosion alebo synthetic data |
| Overfit to LOPO | ⚠ riziko | po finalizácii zachovať untouched test set z 5 patient-leave-out |
| Hidden test má 9 tried | ✗ neznáme | pripraviť open-set scenár (anomaly detection) |
| Hidden test má nových pacientov | ✗ neznáme | LOPO eval je proxy ale nie garancia |

## Ďalšie strategické úvahy

- **Submission strategy**: pripraviť 2-3 model varianty (najsilnejší ensemble, najinterpretabilnejší, najrobustnejší) a vybrať podľa toho čo organizátori vyhodnotia.
- **Pitch strategy**: každopádne mať pripravený CGNN alebo TDA novel track na demonštráciu — aj keby F1 bola nižšia, judges to ohodnotia.
- **Time budget na hackathone**: zo 48h máme 240-vzoriek dataset perfektne preskúmaný + baselines pripravené. Vlastný hackathon teda začne s F1 = 0.628 ako baseline a 24+ hodín na zlepšenia.
