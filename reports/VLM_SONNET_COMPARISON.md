> [!WARNING]
> **CONTAMINATED — DO NOT CITE.** This report used `cache/vlm_few_shot_collages/<CLASS>__<scan>.png` paths whose filename leaked the class label to the VLM. Caught by red-team audit `reports/RED_TEAM_SONNET_0_8873.md` on 2026-04-18.
> Honest replacement: `reports/VLM_SONNET_HONEST.md` (Sonnet honest wF1 = 0.3424, inflation +0.545).
> Leakage prevention infra: `teardrop/safe_paths.py` + `reports/LEAKAGE_PREVENTION.md`.

---

# VLM upgrade test: Haiku 4.5 vs Sonnet 4.6 (few-shot, k=2)

Dátum: 2026-04-18
Subset: 60-scan person-stratified (12 per class, `seed=123`), person-LOPO (anchor z tej istej osoby vylúčený).
Prompt / collage / retrieval identické — mení sa len model.

## TL;DR — Sonnet vyhráva jednoznačne

| Metrika | Haiku 4.5 | Sonnet 4.6 | Delta |
|---|---|---|---|
| Accuracy | 0.7333 | **0.8500** | **+0.1167** |
| Weighted F1 | 0.7351 | **0.8454** | **+0.1103** |
| Macro F1 | 0.7351 | **0.8454** | **+0.1103** |
| Avg cost / prediction | $0.0167 | $0.0340 | **2.03×** |
| Avg latency | 18.6 s | 13.0 s | 0.70× (rýchlejší) |
| Total cost (60 scans) | $1.00 | $2.04 | +$1.04 |
| Mean confidence | 0.809 | 0.780 | −0.03 |

**Decision rule**: threshold bol >5 pp F1 → **Sonnet prekračuje +11 pp a odporúča sa škálovanie na plný 240-scan dataset.**

Cost pre full 240: ~**$8.15** (Sonnet) vs ~$4.00 (Haiku). Rozdiel $4.15 je bezproblémový voči rozpočtu.

Bonus: Sonnet má **kratšiu latenciu** (13s vs 18.6s) aj napriek väčšiemu modelu — pravdepodobne menej tool-use loopovania (Read volá raz a ide rovno na odpoveď).

## Per-class F1

| Class | Haiku | Sonnet | Δ |
|---|---|---|---|
| ZdraviLudia | 0.688 | **0.889** | **+0.201** |
| Diabetes | 0.727 | **0.917** | **+0.189** |
| PGOV_Glaukom | 0.720 | **0.828** | +0.108 |
| SklerozaMultiplex | **0.909** | 0.857 | −0.052 |
| SucheOko | 0.632 | **0.737** | +0.105 |

- **Najväčší zisk: ZdraviLudia + Diabetes** (Haiku ich nepretržite mieša medzi sebou — *"ferning vs coarser lattice"* rozlíšenie je jemnejšie).
- **Strata: SklerozaMultiplex** (Sonnet 3× povie PGOV_Glaukom pre SM queries ktoré majú horizontálne banding/striping artefakty — nízka ale nenulová regresia).
- **SucheOko zostáva najťažšia pre oba modely** (silný class imbalance + 2 osoby celkovo = len 1 extraperson anchor pre LOPO setup).

## Confusion matrix

### Haiku 4.5
```
                 Zdravi  Diabet  PGOV_G  Sklero  SucheO
  ZdraviLudia       11       1       0       0       0
  Diabetes           3       8       0       0       1
  PGOV_Glaukom       2       1       9       0       0
  SklerozaMultiplex  1       0       1      10       0
  SucheOko           3       0       3       0       6
```

### Sonnet 4.6
```
                 Zdravi  Diabet  PGOV_G  Sklero  SucheO
  ZdraviLudia       12       0       0       0       0   <- perfect recall
  Diabetes           0      11       1       0       0   <- 0 confusion with healthy
  PGOV_Glaukom       0       0      12       0       0   <- perfect recall
  SklerozaMultiplex  0       0       3       9       0   <- striping artefakty vedú na Glaukom
  SucheOko           3       1       1       0       7   <- jediná slabá trieda
```

Key observations:
- Sonnet úplne eliminuje `Diabetes → ZdraviLudia` zámeny (Haiku 3 zle, Sonnet 0) a `Glaukom → Zdraví` (Haiku 2 zle, Sonnet 0).
- Sonnet má stále `SucheOko → Zdravi` leak (3 scans). Toto je pravdepodobne štruktúrne: dry-eye pacient č. 29 má centrálne nucleated ferning, ktorý vyzerá „zdravo".
- Nová Sonnet-špecifická chyba: `SklerozaMultiplex → PGOV_Glaukom` (3 scans). Sonnet reaguje na horizontálne striping patterns a klasifikuje ich ako Glaukomové „MMP-9 degradation".

## Reasoning quality — inspekcia 5 disagreements

Celkovo **18 disagreements** (Sonnet 13 × správne, Haiku 5 × správne). 5 reprezentatívnych:

### 1. `ZdraviLudia/29.001` — truth = ZdraviLudia, Sonnet wins

- **Haiku** (conf 0.72 → **Diabetes** ❌):
  *"thicker, elevated crystal structures with robust diagonal branching and dense packing, closely resembling both Diabetes anchors"*
- **Sonnet** (conf 0.88 → **ZdraviLudia** ✓):
  *"dense, fine dendritic branching with uniform fern-like crystal networks closely matching both ZdraviLudia anchors (#1 RL_001 and #2 48_004), exhibiting the hallmark high fractal dimension branching pattern"*

Sonnet correctly reads *fine dendritic branching* = healthy; Haiku misreads brightness as thickness. Sonnet tiež správne cituje konkrétne anchor IDs.

### 2. `Diabetes/DM_01.03.2024_LO.003` — truth = Diabetes, Sonnet wins

- **Haiku** (conf 0.95 → **ZdraviLudia** ❌): overconfident false claim of *"high fractal dimension"*.
- **Sonnet** (conf 0.72 → **Diabetes** ✓): *"thick, densely packed crystalline branches radiating from a bright central point... coarser lattice structure with elevated roughness"*.

Sonnet kalibráciou lepšie — Haiku sa chopil 0.95 confidence na nesprávnu odpoveď.

### 3. `PGOV_Glaukom/27PV_PGOV_PEX.000` — truth = Glaukom, Sonnet wins

- **Haiku** → Diabetes (conf 0.72). Zameniť *bright orange* za *glycated proteins*.
- **Sonnet** → Glaukom (conf 0.82): cituje konkrétne **"MMP-9 degradation"** patologický mechanizmus a identifikuje **"plate-like structures and disordered texture"**.

Sonnet má hlbšie domain knowledge.

### 4. `SklerozaMultiplex/Sklo-No2.041` — truth = SM, **Haiku wins**

- **Haiku** → SM (správne, conf 0.72): všimne si *"heterogeneous, granular texture with mixed bright and dark regions and chaotic organization"*.
- **Sonnet** → **PGOV_Glaukom** ❌ (conf 0.72): redukuje horizontálne stripe artefakty na Glaukomové MMP-9.

Príklad kde je Sonnet „too confident v patologickom reasoning" — vidí striping a hneď ho kauzálne pripíše MMP-9. Haiku tu je opatrnejšie a všimne si mixed granular texture.

### 5. `SucheOko/35_PM_suche_oko.010` — truth = SucheOko, **Haiku wins**

- **Haiku** → SucheOko (správne, conf 0.70): *"organized fine branches but with conspicuous empty (dark) regions between primary branches"*.
- **Sonnet** → **PGOV_Glaukom** ❌ (conf 0.62): *"discrete granular ring-like/looping bright clusters"*.

Sonnet preferuje rings/loops nad empty regions. Nízka Sonnet confidence (0.62) je korektný kalibračný signál — jedno z mála disagreements kde mu 0.62 reflektuje neistotu.

### Kvalitatívny súhrn reasoning

| Dimenzia | Haiku 4.5 | Sonnet 4.6 |
|---|---|---|
| Cituje konkrétne anchor IDs? | Väčšinou nie | **Áno** (*„#1 RL_001 and #2 48_004"*) |
| Používa patofyziologický jazyk? | Príležitostne | **Konzistentne** (*MMP-9, high fractal D, glycated proteins*) |
| Calibration (confidence ↔ správnosť) | Slabá (0.95 na wrong answer) | **Lepšia** (nižšie conf pri neistote) |
| Tendencia k ZdraviLudia bias | **Vysoká** (14× pred Zdraví — 9× wrong) | **Nízka** (12× pred Zdraví — 3× wrong) |
| Latency | 18.6 s (často retry Read) | 13.0 s (jeden pass) |

Sonnetov reasoning je merateľne presnejší: zachytí rozdiel medzi *fine* a *coarse* branching, uvádza anchor identifikátory a zreteľnejšie spája vizuálne crty s domain literatúrou (MMP-9, fractal D).

## Verdikt

**Sonnet 4.6 výrazne (+11 pp F1) poráža Haiku 4.5 pri 2× nákladoch a 0.7× latencii. Odporúčam:**

1. **Škálovať Sonnet na full 240-scan dataset** (~$8 cost, <60 min wall). Očakávaná honest person-LOPO F1 pri extrapolácii: **0.82–0.87** (podľa stability cez 60-subset).
2. Nepoužívať Haiku pre production inferenciu — úspora $4 nestojí za stratu 11 pp F1.
3. **Retiny concern for SklerozaMultiplex**: Sonnet má tendenciu confúzne priradiť striping → Glaukom. Ak to bude problém v ensemble, fallback na Haiku pre SM cases alebo k=3 anchors.

## Artefakty

- `scripts/vlm_few_shot_sonnet.py` — Sonnet variant skriptu.
- `cache/vlm_sonnet_predictions.json` — raw Sonnet predikcie (60 scans).
- `cache/vlm_few_shot_predictions.json` — Haiku predikcie (now 62 scans; 60 overlapuje s Sonnet).
- `cache/vlm_sonnet_vs_haiku_summary.json` — machine-readable summary.
- `cache/vlm_sonnet_vs_haiku_disagreements.json` — full 18-disagreement reasoning dump.
- `cache/vlm_few_shot_collages/` — zdieľané collages (identické pre oba modely).
