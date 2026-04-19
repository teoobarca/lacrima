# Hack Košice 2026 — UPJŠ Tear Challenge

Projekt na hackathon challenge od Prírodovedeckej fakulty UPJŠ: **klasifikácia chronických ochorení z AFM mikroskopických snímok vysušených sĺz**.

## Úloha v skratke

- **Vstup:** AFM (atomic force microscopy) snímky vysušenej kvapky slzy — výškové mapy, scan range ~50–92 μm, nm-úroveň vertikálneho rozlíšenia.
- **Výstup:** klasifikátor, ktorý vráti pravdepodobnosť pre každú z ~9 tried.
- **Triedy v PDF slides (9):** healthy, diabetes, multiple sclerosis, glaukóm, Alzheimer, bipolárka, panika, cataract, PDS, dry eye.
- **Triedy v skutočnom TRAIN_SET (len 5):** `Diabetes`, `PGOV_Glaukom`, `SklerozaMultiplex`, `SucheOko`, `ZdraviLudia`.
- **Platforma:** Windows (podmienka organizátorov).
- **Metrika:** **weighted F1-score** na skrytom datasete.
- **Dataset:** stiahnutý z `https://temp.kotol.cloud/?c=7IDU` (3.2 GiB ZIP, expiruje ~21.04.2026).
- **Hint nástroj:** [Gwyddion](http://gwyddion.net/) — open-source SPM analýza.

## Aktuálny shipped model

**Champion (Wave 5):** `models/ensemble_v2_tta/` — DINOv2-B + BiomedCLIP TTA (D4), L2-normalized embeddings, geometric-mean softmaxes. **Honest F1 = 0.6562 weighted, 0.5382 macro (person-LOPO).**

**Candidate (Wave 6, pending red-team):** 3-way fusion E7 with DINOv2-B RGB multichannel (Height + Amplitude + Phase), **0.6645 F1**.

Benchmark dashboard: `reports/BENCHMARK_DASHBOARD.md`. Design rationale: `reports/DESIGN_RATIONALE.md`. Architecture diagrams: `reports/ARCHITECTURE.md`.

## Aktuálny stav datasetu (po audite)

- **240 raw AFM skenov** (Bruker Nanoscope SPM) + **240 BMP preview** (704×575 RGB s vpálenými axis labels).
- **35 unikátnych osôb** (44 eyes — L/P eye collapsed). SucheOko má **2 osoby**, Diabetes 4, Glaukom 5, SM 9, Healthy 15. *(Validator audit 2026-04-18 odhalil že L/P oči neboli zlúčené — opravené cez `person_id()`, `patient_id()` zachovaný pre eye-level.)*
- **Class imbalance 7:1** (SM 95 : SucheOko 14).
- **Rozlíšenia rôznorodé** (256² až 4096², aj obdĺžnikové).
- **Scan range:** 78 % je 92.5 μm, 13 % je 50 μm, zvyšok 10–80 μm.
- Detail: `reports/DATA_AUDIT.md`, vizualizácie: `reports/samples/` a `reports/raw_samples/`.

## KRITICKÉ pravidlá (z auditu)

1. **Patient-level CV split je MUST** (`StratifiedGroupKFold` s `group=patient_id`). Image-level split nafúkne F1 o 20–30 %.
2. **Nepoužívaj BMP priamo bez croppingu** — má vpálené axis labels = data leakage.
3. **Plane leveling (1st-order polynomial)** je povinný preprocessing krok kvôli scanner tilt.
4. **Resample na konštantné px/μm** pred texture feature extraction (kvôli 10× variabilite scan size).

## Biofyzikálny princíp

Slza = elektrolytový koktail (NaCl, KCl, NaHCO₃) + proteíny (laktoferín, lyzozým, albumín, IgG) + mucíny + lipidy + glukóza + cytokíny. Pri desikácii sa rozpustené látky **samoorganizujú do fraktálnych dendritických kryštálov** (*tear ferning*). Chemické zloženie → tvar kryštálu. Choroba mení zloženie → mení vzor.

**Dôkazová báza podľa choroby:**
- **Silná:** dry eye (Masmali škála I–IV), diabetes, glaukóm (MMP-9), MS (AFM + IR štúdie).
- **Stredná:** Alzheimer (amyloid-β, tau v slze), cataracts (metabolity).
- **Exploratívna:** bipolárka, panika (katecholamíny, kortizol) — sparse literatúra, pravdepodobne najťažšie triedy.

## Kľúčové zdroje

### Dokumenty v repe
- `UPJS.pdf` — oficiálne challenge slides.
- `UPJS_HK_2026_presentation vk la.pptx` — rozšírená prezentácia (obsahuje detailné zadanie, hypotézu, odkazy na dataset).

### Literatúra (na začiatok)
- Sensors (MDPI) 2023, PMID 37299978 — MS tear fluid AFM.
- MDPI Diagnostics 5(4):48 (2025) — AFM+XRD+EDX cez DED/glaukóm/MS.
- PubMed 39788273 (2025) — AFM-IR na diabetic tear desiccates.
- PMC10582561 — Deep learning na malom AFM datasete (DINO SSL, 88–100 % accuracy).
- ScienceDirect 2026 — TearNET: CNN ferning grading.
- Gwyddion docs — statistical-analysis, fractal-analysis.

## Stratégia riešenia

### Safety-net baseline (musí byť hotový do 12 hodín)
1. **BiomedCLIP / DINOv2 frozen encoder + MLP/XGBoost head** — pre <500 vzoriek najsilnejší „istý" prístup. BiomedCLIP bol pretrénovaný na PubMed obrázkoch, pravdepodobne videl tear ferning.
2. **Handcrafted features → XGBoost:** GLCM (Haralick) + LBP + fraktálny rozmer + Ra/Rq/Ssk/Sku z výškovej mapy + HOG.
3. **StratifiedKFold(5)** od prvej minúty. Weighted CE / Focal loss + class weights.

### Novel smery (pitch killers)
- **Crystal Graph Neural Network (CGNN):** skeletonizácia → graf → GIN/GAT. Attention na hranách = interpretabilná „ktorá vetva rozhodla".
- **Topological Data Analysis (TDA):** persistent homology (`giotto-tda`) na filtrovanej výške. H₀/H₁ perzistenčné diagramy ako feature.
- **Multifractal + lacunarity + succolarity signature.**
- **LLM reasoning layer:** kvantitatívne features + domain knowledge → Claude / GPT-4 vracia `{probabilities, reasoning}` v JSON. Interpretabilný výstup (clinical-style).
- **Physics-informed inverzná úloha:** reaction-diffusion simulácia (Cahn-Hilliard) ako prior, odhad chemických parametrov z obrázka → klasifikácia z chémie.

### Ensemble plán
CNN/VLM track + handcrafted/XGBoost track + jeden z novel trackov (CGNN alebo TDA) → stacking na validation out-of-fold predikciách.

## Stack (Windows)

```
Python 3.10+  (Anaconda odporúčaná)
PyTorch 2.x + torchvision + torchgeometric  (CUDA ak je GPU)
scikit-learn 1.4+, scikit-image 0.21+
OpenCV-Python
XGBoost 2.x
imbalanced-learn
PyWavelets
pytorch-grad-cam
AFMReader           # Python 3 náhrada za pygwy
giotto-tda          # persistent homology
sknw / skan         # skeletonization → graph
open_clip_torch     # BiomedCLIP, SigLIP, DINOv2
transformers        # HuggingFace modely
Gwyddion (GUI)      # ručná exploration AFM súborov
```

## Riziká (risk register)

| Riziko | Pravd. | Dopad | Mitigácia |
|---|---|---|---|
| Overfitting (malý dataset) | Vysoká | Vysoký | Frozen VLM backbone, aggressive augmentácia, label smoothing, early stopping na val F1 |
| Class imbalance | Vysoká | Vysoký | Focal loss + class weights, stratified K-fold, per-class threshold tuning |
| Train/test distribution shift | Stredná | Vysoký | Grad-CAM audit (netreba aby model chytal pozadie), konzistentný preprocessing |
| AFM formát nečitateľný v Pythone | Stredná | Stredný | AFMReader fallback; Gwyddion batch export do CSV/PNG |
| Psychiatrické triedy ambigue | Vysoká | Stredný | Embedding space clustering analysis; prípadne „uncertain" bucket |
| API inferencia nepovolená | Neznáme | Stredný | Overiť pravidlá; self-hosted Llama/Qwen alebo čisto offline VLM path |

## Otázky, ktoré treba zistiť na hackathone

1. Koľko vzoriek per trieda? (ovplyvní voľbu SSL pretraining vs. frozen VLM)
2. Formát súborov — `.gwy`, `.spm`, `.tiff`, `.png`? Raw výšková matica alebo už renderované?
3. Môže finálna inferencia volať externé API, alebo musí bežať plne offline na Windows?
4. Dostaneme explicit train/val split, alebo si máme robiť vlastný?
5. Sú na jedného pacienta viaceré scany? (ovplyvní splitting — treba *patient-level* split, nie *image-level*).
6. Je evaluácia single-label alebo multi-label (komorbidity)?

## Workflow konvencie

- **Jazyk:** Slovenčina pre komunikáciu, poznámky, commit messages — **vždy s diakritikou** (háčky, mäkčene, dĺžne). Technické termíny ponechať anglicky (*focal loss*, *fractal dimension*, atď.).
- **Nové ML experimenty:** každý v samostatnom notebooku / skripte s jasným názvom (`01_data_audit.ipynb`, `02_baseline_efficientnet.ipynb`, ...).
- **Reprodukovateľnosť:** každý tréning má seed, logujú sa hyperparametre.
- **Validácia:** **nikdy** nereporter accuracy bez weighted F1 + per-class F1 + confusion matrix.

## Stav projektu

- [x] Preštudované zadanie (PDF + PPTX).
- [x] Rešerš literatúry a novel prístupov.
- [ ] Prístup k datasetu.
- [ ] Data audit.
- [ ] Baseline pipeline.
- [ ] Novel track(y).
- [ ] Ensemble + pitch materiály.
