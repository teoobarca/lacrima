# RED-TEAM AUDIT: Sonnet Few-Shot wF1 = 0.8873 on Full 240

**Date:** 2026-04-18
**Claim under audit:** `scripts/vlm_few_shot_sonnet_full_240.py` reports **weighted F1 = 0.8873** on all 240 scans with person-LOPO anchors (DINOv2-B retrieval + k=2 anchors/class + collage + Sonnet 4.6).
**Auditor verdict: REJECT — inflated by filename leakage, identical bug-class as Wave-1 `vlm_direct_classify.py`.**

---

## TL;DR

The 0.8873 wF1 is almost entirely explained by **class-name leakage through the collage filename**. The collage is written to
`cache/vlm_few_shot_collages/<CLASS_NAME>__<scan_id>.png`, and that path is passed to Sonnet in the prompt ("Read the image at {img_path}"). Sonnet reads the class out of the filename, then produces plausible post-hoc visual reasoning.

**Empirical proof (this audit):**
- 20 scans Sonnet got 100% correct with the leaky path
- Re-run with path obfuscated to `/tmp/vlm_obf_.../scan_XXXX.png` (no class anywhere)
- **Honest accuracy: 5/19 = 26.3 %** (1 errored). Near-random (5-class baseline = 20 %).
- Model collapsed to predicting `SklerozaMultiplex` for 14/19 queries.

Honest wF1 on full 240 will **not** be 0.8873. Extrapolating from the 26.3 % accuracy on previously-correct queries, the honest wF1 is plausibly in the 0.15-0.35 range — worse than v4's 0.6887, consistent with the Wave-1 `VLM_CONTAMINATION_FINDING.md` result (honest F1 ~ 0.14-0.20).

Evidence script: `/tmp/vlm_sonnet_obfuscated_test.py`; results: `/tmp/vlm_obfuscated_results.json`.

---

## Audit checklist

### 1. Person-LOPO integrity — PASS (clean)
Verified on `cache/vlm_sonnet_full_predictions.json`:
- 2400 anchors across 240 queries.
- `anchor.person == query.person`: **0**
- `patient_id(anchor) == patient_id(query)` (eye-level): **0**
- `anchor.path == query.path`: **0**

`vlm_few_shot.retrieve_anchors_per_class` uses `emb["groups"]` which is built from `s.person` (i.e. `person_id()` with L/R collapsed). Embedding cache `cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz` has 35 unique groups (matches person count, not 44 eye count). Hard assert in `prepare_job` (line 101) enforces `a_sample.person != sample.person`. **Not the bug.**

### 2. Collage image content — PASS
Inspected `cache/vlm_few_shot_collages/Diabetes__37_DM_010.png`. Class names appear only as column headers (legitimate anchor-column labels). QUERY tile is unlabeled. No EXIF, no visible filename inside the rendered image. Image content is clean.

### 3. Prompt filename leakage — FAIL (critical)
`scripts/vlm_few_shot.py:499`, inherited by `vlm_few_shot_sonnet_full_240.py:120`:
```python
collage_path = COLLAGE_DIR / f"{s.cls}__{s.raw_path.name.replace('.', '_')}.png"
```
then:
```python
prompt = PROMPT_TEMPLATE.format(img_path=str(img_path))  # "Read the image at {img_path}"
```
Sonnet sees the absolute path literally, e.g.
`/Users/rafael/.../vlm_few_shot_collages/SucheOko__29_PM_suche_oko_005.png`.
Every collage filename is prefixed with its class. This is the **same bug** caught in Wave-1 (`vlm_direct_classify.py`, `RED_TEAM_VLM.md`, `VLM_CONTAMINATION_FINDING.md`).

### 4. Self-anchor leak — PASS
Confirmed 0 cases where a query's own scan appears in its anchor set.

### 5. SucheOko paradox — RESOLVED (structurally OK)
2 SucheOko persons: `29EYE_suche_oko`, `35EYE_suche_oko`. When person 29 is queried, SucheOko anchors are only from person 35, and vice versa. Person-LOPO is satisfied and retrieval still succeeds — not a bug. 9/14 correct is plausible *if the classification were genuine*, but given leak in (3), correct SucheOko predictions are almost certainly filename-driven.

### 6. Reproduce with obfuscated paths — FAIL (confirms leak)
Ran Sonnet on 20 previously-perfectly-classified collages copied to `/tmp/vlm_obf_*/scan_XXXX.png`:
- Accuracy: **5/19 = 0.263** (1 parse error). Was 20/20 with leak.
- Per-class: Diabetes 0/4, Glaukom 0/4, SucheOko 0/3, ZdraviLudia 1/4, SM 4/4 (all SM predictions by mode-collapse).
- Mean confidence dropped slightly (0.52 typical) and reasoning became generic ("granular texture resembles the SklerozaMultiplex anchors").

This is the decisive test. The leak explanation accounts for **essentially all** of the accuracy delta between honest (~0.26) and leaky (1.00) on the same queries.

### 7. Reasoning sanity — AMBIGUOUS (post-hoc)
Correct-prediction reasoning strings reference anchor IDs and morphology plausibly ("query matches Diabetes anchor #1 with thick coarse branches"). Looks genuine in isolation. But given (6), Sonnet is clearly capable of generating plausible-looking reasoning *after* reading the class from the filename. The reasoning is confabulated post-hoc.

### 8. DINOv2 embedding drift — PASS
Embedding cache `tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz` is the same one used by v4 multiscale champion (`scripts/train_ensemble_v4_multiscale.py:96`). 240 scans × 768 dim, 35 person groups. Not a source of leak.

### 9. McNemar / paired bootstrap vs v4 — MOOT
Script reports P(delta > 0 vs v4) = 1.000 under bootstrap, but the delta itself is driven by leakage. Bootstrap does not detect leakage.

---

## Final Verdict: **REJECT**

Do NOT promote to champion. Do NOT include in Expert Council. Do NOT cite in pitch.

### Required fixes before any further VLM claims are credible
1. Rename collage files to content-agnostic IDs (`collage_<sha1>.png` or `collage_<query_idx:04d>.png`). No class string anywhere in path.
2. Write collages to an isolated temp dir whose path contains no project/class keywords.
3. Strip `img_path` from the prompt entirely — pass the image as an attachment only (the `claude -p` `--files` flag or equivalent), not as a text reference.
4. Re-run on 240 with the fixed pipeline. Compare honest wF1 to v4 champion via paired bootstrap.
5. Add a standing regression test: any new VLM script must pass a `/tmp/scan_0000.png` sanity run with >= 0.50 accuracy before the full-240 run is allowed.

### Expected honest number
Based on 26.3 % accuracy on previously-correct queries (these are the "easy" ones — true honest accuracy on the full 240 may be lower), and consistent with the Wave-1 honest rerun (25-scan obfuscated: 28 % accuracy, 0.20 wF1), **honest Sonnet few-shot wF1 is expected in the 0.15 - 0.35 range**. This is well below v4's 0.6887 and does not merit ensembling without substantial redesign (e.g. LLM-reason layer over handcrafted features only, no VLM-on-image in an identifiable path).

### Historical note
This is the **8th** inflated claim rejected on this codebase and the **2nd** instance of the exact same filename-leak bug (first: `vlm_direct_classify.py` -> 88% → 28% honest). The few-shot scripts inherited the pattern from the direct-classify script when it was forked. All VLM scripts under `scripts/vlm_*` that construct collage paths via `COLLAGE_DIR / f"{s.cls}__..."` are suspect and should be re-audited:
- `vlm_few_shot.py:499`
- `vlm_few_shot_sonnet.py:499`
- `vlm_few_shot_full_240.py:133`
- `vlm_few_shot_sonnet_full_240.py:120`
- `vlm_few_shot_k3.py:472`
- `vlm_few_shot_k2_extend.py:111`
- `vlm_few_shot_opus.py:126`
- `vlm_self_consistency.py:377`

All prior Sonnet/Haiku few-shot result tables (including the 60-subset "0.8454" in `VLM_FEW_SHOT_RESULTS.md` and the k3 comparisons) are **retroactively suspect** and should be marked CONTAMINATED pending re-run with obfuscated paths.
