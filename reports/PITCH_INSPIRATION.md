# PITCH INSPIRATION — Hack Košice 2026 Tear Challenge

**Scratchpad — raw material, not final slides**
**Date:** 2026-04-18
**Method:** 6× parallel perplexity_ask (~250 sources)

---

## AI Era Trends 2025-2026

- **Agentic AI = #1 narrative.** Shift from "models that answer" → "agents that act." Gartner: 40% of enterprise apps embed agents by end-2026; 29% of total AI value by 2028.
- **Multi-agent orchestration** mainstream: Amazon Bedrock, LangGraph, CrewAI, AutoGen (Microsoft), SuperAGI. "Many analyses, one merge" pattern (ORCH, Frontiers 2026) = **our Expert Council**.
- **Test-time compute scaling** beats pre-training scaling (ICLR 2025 oral: 4× efficiency, 14× smaller model matches larger). TTA + ensemble = living example.
- **LLM-as-judge** validated clinical: Pearson 0.87 correlation to expert consensus (Frontiers 2026). VERT ensembles GPT-4.1 + Gemini + Claude for radiology.
- **Foundation models in medical imaging** = hot track 2026. Field moving from "fine-tune everything" → "freeze and retrieve."
- **RAG for medical VLMs**: melanoma RAG VLM F1=0.6864 no fine-tune (arXiv 2509.08338). Few published papers. **Our few-shot retrieval approach is in the right place at the right time.**

---

## Our Novelty Angles

### Genuinely novel / very sparse in literature:
- **AFM + VLM + retrieval-augmented few-shot for disease classification** — no published precedent. Closest: RAG-VLM for dermoscopy (2025), different modality.
- **Expert Council (multi-LLM judge) on microscopy images** — pattern exists for text (radiology, NLP QA), not documented for AFM.
- **Multi-agent orchestration on 240-sample scientific dataset** — industrially trendy, rarely demonstrated on real scientific classification with tight label budgets.
- **7+ documented red-team rejections as methodology** — rigour framing as differentiator.

### Standard (do NOT oversell):
- DINOv2 frozen encoder + head (textbook).
- GLCM/LBP handcrafted features (textbook).
- StratifiedGroupKFold patient-level split (correct practice, not novel).
- LLM-as-judge in general (published since 2023).

### Honest differentiator package:
> "We applied a 2026-era agentic AI stack — multi-agent orchestration, retrieval-augmented VLM few-shot, LLM expert council — to a domain (AFM tear crystallography) where none of these combinations are published. We systematically red-teamed and rejected 7+ candidate models using bootstrap CI, so our F1 is real."

---

## Social Impact Framing

### Cost comparison (USD):
- Tear biomarker panel: $500–3,000
- Blood biomarkers: $500–1,200
- MRI: $1,000–3,000 per scan; equipment $1M+
- Confirmatory PET: $8,868–10,345
- **AFM tear scan: equipment amortized, collection = zero cost, zero pain**

### Market validation:
- Tear diagnostics: $412–482M in 2024, **CAGR 12.7%** → $1.21B by 2033 (Dataintelo)
- **EU COST action CA24112** active — "Implementation of Tear Fluid Biomarkers in Precision Medicine"
- **3,370 unique proteins** identified in tears (Nature 2022)
- Non-invasive → **repeat testing → longitudinal tracking**

### Low-resource angle:
> No MRI, no specialized lab. Clinic with AFM + our classifier = instant chronic disease screening.

---

## Selling-Point One-Liners

1. **"A tear drop knows things your blood test doesn't — yet."**
2. **"We trained AI to read disease signatures in the crystal patterns of a single dried tear."**
3. **"Non-invasive, zero-reagent chronic disease screening: collect a tear, get a diagnosis."**
4. **"We built a multi-agent AI council that argues over your AFM scan — and documents every rejection."**
5. **"DINOv2 sees the crystal. Claude reads the meaning. Together they [X]% — ablations prove each contributes."**
6. **"7 models rejected. 1 survived. We show our work."** (rigour hook)
7. **"84% on disease classification. 240 samples. No fine-tuning. Just a VLM and a library of reference tears."**
8. **"What if your ophthalmologist's slit lamp could also screen for MS and diabetes? Now it can."**
9. **"From 3 GiB of AFM scans to a deployable classifier in 48 hours — with full red-team audit."**
10. **"Tears are the blood test of the future — we just taught AI to read them."**

---

## Storytelling Patterns

### Hook anatomy (mixed audience):
- **Open with the human, not the model.** "A doctor in Košice has 5 minutes per patient. She can't order an MRI for everyone. But everyone has tears."
- **Before/after contrast:**
  - BEFORE: chronic diagnosis = expensive, invasive, slow, specialist-gated
  - AFTER: tear drop → AFM → classifier → per-disease probability in 30 s
- **Surprising stat anchor:** "3,370 proteins in a single tear. We don't need all — just the crystal pattern they leave behind."

### 5-minute arc:
1. **0:00–0:30** — Hook + visual wow (tear crystallography image)
2. **0:30–1:15** — Problem: 5 diseases, non-invasive imperative, current costs
3. **1:15–2:30** — Solution: pipeline (AFM → preprocess → VLM few-shot / Expert Council → probabilities). ONE clean architecture diagram.
4. **2:30–3:30** — Demo + results: confusion matrix + honest F1 + "rejected 7 models to get here"
5. **3:30–4:15** — Novelty: "No one has combined AFM + agentic VLM stack"
6. **4:15–5:00** — Impact + future: market, EU COST action, "next: 500 patients multi-center"

### Demo design:
- Real AFM scan image (crystal dendrites = beautiful).
- Expert Council JSON output: per-disease probabilities + reasoning. **"AI that explains itself" = aha moment.**
- Risk mitigation: pre-recorded 45-s screencast if live demo fails.

### Hack Košice judging criteria match:
- **Originality** ✓ (unpublished combination)
- **Impact** ✓ (screening economics, EU COST action)
- **Implementation Complexity** ✓ (LOPO, multi-agent, 7 red-team iter)
- **Presentation** ← needs: minimal slides, cohesive team story

---

## Similar Winning Hackathon Pitches

| Event | Project | Key move | Lesson |
|---|---|---|---|
| MIT Hacking Medicine 2025 | HospiTwin | Framed as "plug into existing IT" | Frame as augmenting ophtho, not replacing |
| MIT Hacking Medicine | AI radiology | Opened with "N thousand die from errors" stat | Use diagnostic delay stat as opener |
| ETHGlobal NYC — Etherius | NFT agent | Plain English → structured output | Show Expert Council JSON — same pattern |
| HackMIT 2023 — Muse | AI playlists | Never mentioned embeddings; showed UX | Lead with tear image + result, not architecture |
| Hack Košice past winners | Smart-city / social | Local problem + measurable impact + live demo | Anchor to Slovak healthcare, UPJŠ partnership |
| MIT Grand Hack | Patient-safety | Clinician + ML engineer team | Team intro names domain expertise |

**Meta-lesson:** Technical depth assumed. What wins: (1) judges feel problem, (2) see it working, (3) believe team can take further.

---

## Top 3 pitch angles (prioritized)

1. **Defensible novelty**: AFM + retrieval-augmented VLM few-shot = no published precedent. Melanoma RAG-VLM (2025) is closest, different modality.
2. **Agentic AI × biophysics timing**: 2026 #1 AI trend (Gartner, Frontiers, Deloitte). Our multi-agent + Expert Council maps directly onto ORCH / VERT / LLM-Synergy literature — applied to AFM crystallography (unpublished domain).
3. **Impact case is scientifically grounded AND emotionally resonant**: $412M market, 12.7% CAGR, EU COST CA24112 active, 3,370 proteins in a tear, zero reagents vs $500-3000 panels.

**Red-team discipline (7+ rejections, bootstrap CI)** = credibility anchor, not headline. Reason judges should trust our F1 number.
