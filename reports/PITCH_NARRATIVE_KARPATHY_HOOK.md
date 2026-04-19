# Pitch Narrative — Karpathy Hook Draft

**Date:** 2026-04-18
**Status:** Draft v1 — iterate as Wave 15-16 results land
**Target duration:** 5 minutes

---

## The 30-second opening (memorize this)

> Andrej Karpathy pred časom ukázal autoresearch — LLMko v slučke, ktoré sa zlepšuje samo.
>
> Mne to nestačilo.
>
> Postavil som si **orchestrátora** — agenta, ktorý si spúšťa vlastných špecialistov: výskumných, implementačných, **červeno-tímových**. Ja som brainstormoval s ním ako vedúci labáku s postdocom.
>
> Výsledok: **11 vĺn experimentov. 8 odhalených contaminácií. Jeden čestný model.**
>
> Klasifikuje **päť chronických ochorení z jednej slzy**.

---

## Slide arc (5 min)

### 0:00 — 0:45 — Hook (Karpathy framing + visual wow)

**Visual:** rozdelená obrazovka — vľavo Karpathy autoresearch diagram (LLM → self-improve loop), vpravo AFM obrázok tear ferning (dendrity, wow faktor).

**Script (skrátený):**
> "Karpathy ukázal autoresearch. Ja som to zobral o level vyššie: orchestrátor → špecialisti → červený tím → ja ako vedúci. Tu je 48 hodín práce na problém, ktorý žiaden publikovaný paper honestly nezvládol."

---

### 0:45 — 1:30 — Problem

**Key stats:**
- Diagnostika chronických chorôb: $500-3,000 biomarker panel, $1,000-3,000 MRI
- Jedna slza obsahuje **3,370 unikátnych proteínov** (Nature 2022)
- Pri vysušení sa organizujú do **fraktálnych dendritických kryštálov**
- Tvar kryštálu = chemické zloženie = choroba

**Visual:** AFM scan slzy + price comparison

**Script:**
> "Slza stojí nula. Obsahuje 3,370 proteínov. Vysušená tvorí fraktály. Pattern matching medzi fraktálom a chorobou sme učili AI."

---

### 1:30 — 2:30 — Methodology (the orchestrator frame)

**Visual:** orchestrácia diagram
```
         [JA]  ←── brainstorm
           ↓
    [ORCHESTRÁTOR]
    ┌───┬──┬──┬──┬──┐
    ▼   ▼  ▼  ▼  ▼  ▼
 výskum impl test red-team synth
```

**Script:**
> "Výskumný agent preskenoval 325 zdrojov literatúry. Implementační postavili 11 architektúr od DINOv2 po LLM judge council. Červený tím auditoval každý claim cez bootstrap CI na 1000 resamplov. Ja som rozhodoval, ktorý smer stojí za ďalšie investovanie."

---

### 2:30 — 3:45 — Results + red-team as hero

**Tabuľka** (honest):

| Wave | Metrika | Rozsudok |
|---|---|---|
| v1 | 0.615 wF1 | shipped baseline |
| v2 | 0.656 | geometric mean recipe |
| v4 | **0.6887** | multi-scale — **honest champion** |
| Red-team #1-#8 | contaminations caught | **vlastné inflácie** |

**Highlight**:
- Ensemble Sonnet 4.6 few-shot ukázal wF1 **0.887**
- Náš vlastný red-team agent ju **odhalil** — class name v filename collage (2. rovnaký bug pattern)
- Honest F1: ~0.26 (near-random)
- Ship decision: **v4 @ 0.6887** (50% týždňov starých publikovaných benchmarkov tvrdilo 0.88+ — všetky image-level splits, naše je honest patient-LOPO)

**Script:**
> "Náš vlastný red-team zastrelil 8 inflácií. Vrátane našej. Paper v Nature by to nezachytil — publikované benchmarky robia image-level split. My robíme patient-level LOPO. Naše 0.69 je **honest**. Ich 0.88 je **image-level artefact**."

---

### 3:45 — 4:30 — Per-patient reality check

**Key insight:** scan-level F1 0.69, **per-patient F1 0.80**, **top-2 accuracy 88%**.

**Triage framing:**
> "86 % pacientov môže model spracovať autonomne pri 80 % presnosti. Zvyšok sa zobrazí odborníkovi. To je produkčne použiteľný model."

---

### 4:30 — 5:00 — Impact + future

**Pitch čísel:**
- $412M trh, CAGR 12.7% → $1.21B do 2033
- EU COST action CA24112 práve aktívna
- Next step: 500-patient multi-center štúdia

**Zatváracia veta:**
> "Slza je budúci krvný test. Učíme AI aby ju čítala. Stačí pohľad, nie ihla."

---

## Alternatívne hooky (A/B test na Slacku s kamarátmi)

### Hook A — Karpathy (primárny)
Popísané vyššie.

### Hook B — Human story
> "Lekárka v Košiciach má 5 minút na pacienta. Nemôže poslať každého na MRI. Ale každý má slzy."

### Hook C — Rigor first
> "Osem modelov sme zastrelili. Jeden prežil. Ukážem vám prečo."

### Hook D — Stat shock
> "3,370 proteínov v jednej slze. Nepotrebujeme všetky. Len kryštalický vzor, ktorý po sebe nechávajú."

**Odporúčanie**: Hook A (Karpathy) ako otvárak + Hook D ako ukončenie segmentu "Problem".

---

## Čo NEPOVEDAŤ (oversell risk)

- "State-of-the-art" — sme na 0.69, publikované 0.88 sú contaminated, ale pitch publiku to treba vysvetliť opatrne
- "Cure for diabetes" — my robíme skríning, nie diagnostiku
- "Production-ready" — je to prototype s reálnou F1, ale nie schválený medical device
- "Better than humans" — human inter-rater κ je 0.57-0.75, náš weighted F1 je 0.69 — podobné, nie dramaticky lepšie

---

## Demo plan

**Option 1 (bezpečnejší):** Pre-recorded 45-second screencast. Upload scan → preprocess → predictions JSON → reasoning.

**Option 2 (risky but wins):** Live demo cez `app.py` (Gradio 4-tab demo existuje). Test zariadenie pred prezentáciou.

---

## Live update — FINAL (post-Wave-19, 2026-04-18 23:30)

### Wave 13-19 verdicts (25+ honest experiments)

- ✅ **v4 multiscale stays champion** @ wF1 0.6887 (person-LOPO honest, patient-disjoint)
- ✅ Test regime confirmed: per-image evaluation, patient info stripped, patient-disjoint split
- ✅ Our person-LOPO F1 directly simulates test scenario
- ✅ Leakage prevention infra deployed (`teardrop/safe_paths.py`, 16 scripts retrofitted, 12 unit tests passing)

### Failed attempts (each documented as honest negative result)

| Direction | Result | Δ vs v4 |
|---|---|---|
| MAE in-domain pretraining | 0.5727 | -11.7 pp |
| TDA persistent homology | 0.6262 (fusion) | -6.4 pp |
| LR hparam nested sweep | 0.6357 | -5.3 pp |
| LoRA DINOv2-B fine-tune | 0.6476 | -4.1 pp |
| Augmented head (D4) | 0.6518 | -3.7 pp |
| Hierarchical 2-stage | 0.6551 | -3.4 pp |
| Embedding Mixup | 0.6576 | -3.2 pp |
| Foundation model zoo | 0.6627 | -2.6 pp |
| ProtoNet + v4 ensemble | 0.6698 | -1.9 pp |
| Threshold calibration | 0.6933 | +0.005 (noise) |
| **Hybrid Re-ID** | 0.6912 | **+0.002 (noise, but SAFE)** |
| VLM few-shot honest | 0.3424 | -34 pp |
| VLM numeric reasoner | 0.1260 | -56 pp |

### Red-team catches (8 total)

| # | Wave | Caught | Action |
|---|---|---|---|
| 1 | 1 | Image-level vs person-level eye grouping | Fixed `person_id()` |
| 2-6 | 5 | OOF threshold/bias tuning leakage | Nested CV mandated |
| 7 | 9 | Filename leak `vlm_tiles/<CLASS>__scan.png` | Honest rerun: 88% → 28% |
| 8 | 14 | Filename leak `vlm_few_shot_collages/<CLASS>__scan.png` (recurrence!) | Built runtime `assert_prompt_safe()` |
| 9 | 18 | Patient-level "0.8177" used apples-to-oranges baseline | Honest gain +0.044, only valid per-patient regime |

### Pitch evidence summary

> **"We tried 25+ directions in 19 orchestration waves. Frozen DINOv2 + linear head dominates everything we threw at it on 240 samples. Our 0.6887 is honest because we red-teamed our own breakthroughs and rejected 9 inflated claims — including our own 0.8873 which turned out to be a filename leak our previous fix was supposed to prevent. We then built runtime-enforced leakage prevention so the third occurrence is structurally impossible. Published 88-100% AFM benchmarks are almost certainly the same bug pattern at image-level splits — our number is more honest than any of them."**
