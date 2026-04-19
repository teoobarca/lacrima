# ensemble_v2_tta — L2-norm + geometric-mean TTA ensemble

**Honest person-LOPO weighted F1: 0.6562** (+0.0104 over v1_tta 0.6458).
**Macro F1: 0.5382** (+0.0228 over v1_tta 0.5154).

Same two frozen encoders and D4 test-time augmentation as `ensemble_v1_tta`,
but:
- Embeddings are **L2-normalized row-wise** before StandardScaler
- Per-encoder softmaxes are combined by **geometric mean** (log-space arithmetic)
  instead of arithmetic mean

Discovered by the autoresearch hypothesis-generating agent in Wave 5 — both
changes are honest (no tuning, no OOF peeking, no threshold search). See
`reports/AUTORESEARCH_WAVE5_RESULTS.md` for the ablation.

## Per-class F1
- ZdraviLudia: 0.87
- Diabetes: **0.54** (+0.11 over v1_tta)
- PGOV_Glaukom: 0.56 (−0.03)
- SklerozaMultiplex: 0.65 (−0.07)
- SucheOko: **0.06** (non-zero for the first time)

Gains concentrate on the minority classes (Diabetes, SucheOko). The majority
classes drop slightly — net weighted F1 is +0.0104, net macro F1 is +0.023.

## Inference

To reuse at inference, apply the SAME preprocessing:
1. Preprocess SPM → 9 tiles → D4 TTA → encode → mean-pool to scan embedding
2. L2-normalize each scan embedding (axis=1)
3. StandardScaler.transform with the saved means/scales
4. LR predict_proba per component
5. Geometric mean of the two softmaxes → argmax

A reference inference helper lives in `predict_v2.py` (same folder).

## Training data
240 scans / 35 persons / 5 classes. Provenance: autoresearch Wave 5 H2.
