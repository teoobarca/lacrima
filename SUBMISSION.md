# Submission Handoff — UPJŠ Teardrop Challenge (Hack Košice 2026)

**Model bundle:** `models/ensemble_v1/`
**Task:** 5-class tear-film AFM disease classification
**Shipped by:** Hack Košice 2026 team

---

## 1. Quick start

```python
from teardrop.infer import TearClassifier

# Load the shippable ensemble bundle
clf = TearClassifier.load('models/ensemble_v1')

# Predict a single scan (accepts Bruker Nanoscope .NNN or .spm):
pred_class, probs = clf.predict_scan('path/to/scan.015')

# Predict an entire directory (walks recursively for raw SPM files)
# Returns a pandas DataFrame: columns = filename, pred_class, probabilities per class
df = clf.predict_directory('/path/to/TEST_SET')
df.to_csv('submission.csv', index=False)
```

Output classes (alphabetical index used by the LR heads):

```
0 — Diabetes
1 — PGOV_Glaukom
2 — SklerozaMultiplex
3 — SucheOko
4 — ZdraviLudia
```

---

## 2. Environment

- Python **3.13**
- Virtual env: `.venv/bin/python` (see `.venv` at project root)
- Install: `pip install -r requirements.txt`

### Key dependencies (from `requirements.txt`)

```
numpy
pillow
matplotlib
scikit-image
scikit-learn
opencv-python
pandas
torch
torchvision
xgboost
imbalanced-learn
AFMReader
pywavelets
giotto-tda
sknw
open_clip_torch
transformers
tqdm
```

On first load the classifier downloads DINOv2-B (~350 MB via `torch.hub`) and BiomedCLIP ViT-B/16 (via HuggingFace hub). Subsequent runs are cached locally (~/.cache).

Hardware: CPU works; GPU (CUDA or MPS) recommended for the encoder forward pass on large batches.

---

## 3. Model architecture (one paragraph)

Two frozen ViT encoders — **DINOv2-B ViT-B/14** (768-dim) and **BiomedCLIP ViT-B/16** (512-dim) — each followed by its own `StandardScaler → LogisticRegression(class_weight='balanced', C=1.0, max_iter=3000)` head. Per scan: load SPM → 1st-order plane level → resample to 90 nm/px → 2–98 percentile robust normalize → up to 9 non-overlapping 512² tiles → render each as `afmhot` RGB → encode tiles → mean-pool tile embeddings → standardize → LR logits → softmax. The two softmax outputs are **uniformly averaged**, argmax = predicted class. No thresholds, no calibration, no stacking.

---

## 4. Expected performance

Numbers come from honest **person-level Leave-One-Patient-Out CV** on the 240-scan TRAIN_SET (35 persons after collapsing L/R eyes).

| Metric | Value | Protocol |
|---|---:|---|
| Weighted F1 (raw argmax, shipped) | **0.6346** | person-LOPO, 35 folds |
| Weighted F1 (+ nested-CV thresholds, ref.) | **0.6528** | person-LOPO, nested |
| Macro F1 | **0.4934** | person-LOPO raw argmax |
| Baseline (DINOv2-B alone) | 0.615 | single-model LR |
| Null baseline (label-shuffle ×5 seeds) | 0.276 ± 0.042 | signal sanity |

### Per-class F1 (person-LOPO, champion)

| Class | F1 |
|---|---:|
| ZdraviLudia | ~0.82 |
| SklerozaMultiplex | ~0.72 |
| PGOV_Glaukom | ~0.59 |
| Diabetes | ~0.43 |
| SucheOko | ~0.00 |

See `reports/FINAL_REPORT.md` for full leaderboard and red-team history.

---

## 5. Known limitations

1. **SucheOko F1 ≈ 0.** The TRAIN_SET has only **2 distinct persons** (14 scans) labelled SucheOko. Any person-disjoint evaluation that holds one of them out has 0 or 1 SucheOko training persons remaining. We report this honestly — it is a data-acquisition ceiling, not a modelling failure. If the hidden test set contains new SucheOko patients, expect low recall on that class specifically.
2. **5-class only.** The UPJŠ PDF references additional diseases (Alzheimer, bipolar, panic, cataract, PDS). Our TRAIN_SET contains only 5 classes and our classifier is closed-set — unseen classes will be mis-classified with high confidence. No open-set / OOD detector is shipped.
3. **Patient is a dominant latent variable.** At 240 scans / 35 persons, a UMAP of the embeddings clusters by `person_id` more tightly than by class. Expect some performance drop on truly unseen persons vs LOPO numbers.
4. **Input format assumed to be raw Bruker Nanoscope SPM.** We do *not* accept the BMP previews — they contain axis labels / scale bars that constitute watermark leakage. If the hidden test set is BMP-only, contact us; a BMP fallback path exists but is not shipped.
5. **Training-set F1 = 1.0.** That is expected (LR fits perfectly in these dimensions on 240 samples) and is *not* a validation signal. Only quote the LOPO numbers in section 4.
6. **No per-class thresholds in the shipped bundle.** Threshold tuning on the 240-row OOF inflated our headline by +0.035 and collapsed under nested CV. We shipped raw argmax specifically to avoid that failure mode. The 0.6528 reference variant requires re-introducing nested-CV thresholds.

---

## 6. Bundle contents

```
models/ensemble_v1/
├── dinov2b/
│   ├── classifier.npz       # scaler mean/scale + LR coef/intercept
│   └── meta.json            # encoder_name = dinov2_vitb14
├── biomedclip/
│   ├── classifier.npz
│   └── meta.json            # encoder_name = biomedclip
├── meta.json                # kind='ensemble', components=['dinov2b','biomedclip']
├── train_predictions.csv    # 240 training-scan predictions (sanity only)
└── README.md                # detailed bundle documentation
```

An alternative single-encoder bundle is also available: `models/dinov2b_tiled_v1/` (expected LOPO F1 ≈ 0.615).

---

## 7. Reproducibility

Retrain from cached embeddings (no re-encoding needed):

```bash
.venv/bin/python scripts/train_ensemble_model.py
```

Required cache files (produced by `scripts/baseline_tiled_ensemble.py`):

- `cache/tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz`
- `cache/tiled_emb_biomedclip_afmhot_t512_n9.npz`

Re-derive all honest LOPO numbers:

```bash
.venv/bin/python scripts/prob_ensemble.py        # ensemble leaderboard → cache/best_ensemble_predictions.npz
.venv/bin/python scripts/redteam_v2.py           # nested-CV audit → cache/red_team_audit_v2.npz
```

---

## 8. Supporting documents

| Document | Description |
|---|---|
| `reports/FINAL_REPORT.md` | Comprehensive technical report, full leaderboard, negative results, limitations |
| `reports/PITCH_NARRATIVE.md` | 5-minute pitch script |
| `reports/DATA_AUDIT.md` | Dataset audit (scan parameters, class balance, preprocessing) |
| `reports/VALIDATION_AUDIT.md` | Independent validation (patient-ID parsing, null baseline, CV variance) |
| `reports/RED_TEAM_ENSEMBLE_AUDIT.md` | Nested-CV audit of ensemble threshold tuning |
| `reports/RED_TEAM_ENSEMBLE_V2_AUDIT.md` | Nested-CV audit of 4-component concat ensemble |
| `reports/TDA_RESULTS.md` | Topological data analysis track |
| `reports/CGNN_CPU_RESULTS.md` | Crystal graph neural network track |
| `reports/CASCADE_RESULTS.md` | Binary-specialist cascade (negative result) |
| `reports/CASCADE_STACKER_RESULTS.md` | Soft-blend stacker (small honest gain) |
| `reports/LLM_GATED_RESULTS.md` | LLM reasoning layer for uncertain cases |
| `reports/pitch/INDEX.md` | Pitch figure catalogue |

---

## 9. Contact

*(placeholder — Hack Košice 2026 team contact details)*

Team lead: **TBD**
Email: **TBD**
GitHub: **TBD**

For questions about the model, evaluation protocol, or reproducibility, see the corresponding `reports/*.md` files first; all decisions and their red-team audits are logged there.
