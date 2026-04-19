# ensemble_v4_multiscale — SHIPPED CHAMPION

**Honest person-LOPO weighted F1: 0.6887** (+0.0325 over v2, +0.0429 over v1 TTA).
**Macro F1: 0.5541** (+0.0159 over v2).

Recipe: 3-way ensemble
- DINOv2-B at 90 nm/px (non-TTA, 9 tiles)
- DINOv2-B at 45 nm/px (non-TTA, 9 tiles) — finer resolution captures lipid/mucin texture
- BiomedCLIP at 90 nm/px with D4 TTA (72 views per scan)

Each component: L2-normalize → StandardScaler → LR(balanced). Ensemble: geometric mean of 3 softmaxes.

Why no TTA on DINOv2? Empirically, D4 TTA at 45 nm/px REGRESSES F1 by 0.022 — at that zoom-in level, the class-distinguishing fine-scale features are orientation-specific, so averaging over rotations washes out signal. See `reports/MULTISCALE_TTA_RESULTS.md`.

Red-team bootstrap 95% CI for ΔF1 vs v2 fair apples-to-apples: [+0.010, +0.069], P(Δ>0) = 0.999.

## Per-class F1
- ZdraviLudia: ~0.87
- Diabetes: ~0.58
- PGOV_Glaukom: ~0.58
- SklerozaMultiplex: ~0.69
- SucheOko: ~0.00 (2-patient ceiling)

## Inference

See `predict.py` + `predict_cli.py` at project root:
    .venv/bin/python predict_cli.py --model models/ensemble_v4_multiscale --input /path/to/TEST_SET

Inference time: ~50s/scan (dominated by 72-view BiomedCLIP TTA). Acceptable for hackathon batch prediction.
