# Autoresearch Ledger — cumulative hypothesis history

Running log of all autoresearch-style hypotheses tested across waves. Sorted by impact.

## Waves 1–4 (pre-autoresearch): manual orchestration

- Tile → mean-pool → LR: **+0.05 F1** over single-crop
- Frozen DINOv2-B → BiomedCLIP ensemble: **+0.02 F1** over single encoder
- Test-time D4 augmentation: **+0.011 F1** on ensemble

## Wave 5 autoresearch — 10 hypotheses scored, top 5 tested

| Rank | ID | Hypothesis | Result | Impact |
|---|---|---|---|---|
| ★1 | H1 | Geometric mean of softmaxes (vs arithmetic) | **+0.0075 F1** | **ADOPTED** |
| ★2 | H2 | L2-normalize embeddings before StandardScaler | **+0.0029 F1** (on top of H1) | **ADOPTED** |
| 3 | H4 | Concat all embeddings + nested-C LR | 0.6412 < 0.6562 | REJECTED (LR-C unstable) |
| 4 | H3 | Project out top-k person-ID directions | −0.06 F1 | REJECTED (patient = class confound) |
| 5 | H8 | Add TDA/handcrafted as 3rd ensemble component | −0.03 to −0.05 | REJECTED (weak components dilute geom-mean) |
| — | H5 | K-NN classifier on embeddings | not tested | lower priority |
| — | H6 | Ensemble of 5 encoders (+OpenCLIP, +SigLIP) | not tested | compute-heavy |
| — | H7 | Per-patient embedding normalization | not tested | risky — similar to H3 |
| — | H9 | Larger tile (768) vs smaller (384) | not tested | encoding cost |
| — | H10 | Grayscale render vs afmhot | not tested | likely lower signal |

## Wave 5 multichannel probe (parallel)

| Config | Honest F1 | Notes |
|---|---:|---|
| Bruker Height alone | 0.6313 | baseline for probe |
| Height + Amplitude + Phase as RGB | 0.6375 | modest gain alone |
| Height-DINOv2 + RGB-DINOv2 (arith mean) | **0.6557** | competitive with TTA v1 |

## Wave 5 other agents

| Approach | Honest F1 | Status |
|---|---:|---|
| SupCon SSL projection head | 0.6120 | below baseline (expected small-data SSL) |
| Advanced features (443-dim: multifractal + lacunarity + succolarity + wavelet-packet + multi-scale HOG) | 0.4707 | below handcrafted baseline alone |
| Error analysis Mode B deep-dive | N/A | confirms data ceiling; not a model problem |

## Wave 8 (v4 packaging, in progress)

- v4 multi-scale champion packaging (`models/ensemble_v4_multiscale/`) from red-team-approved Config D
- Honest F1 0.6887 (+0.0325 over v2 champion)
- Inference pipeline: DINOv2-B 90nm (no TTA) + DINOv2-B 45nm (no TTA) + BiomedCLIP-TTA → v2 recipe

## Wave 7b (multi-scale TTA test — NEGATIVE, informative)

- **D-TTA** (add D4 to 90nm + 45nm DINOv2 branches) REGRESSES: 0.6887 → 0.6666 (-0.022)
- Per-member: DINOv2-B 45nm **non-TTA = 0.6544** vs **TTA = 0.6255**. TTA HURTS at 45 nm/px.
- Interpretation: at zoom-in 45 nm/px, class-distinguishing features (lipid/mucin texture) are orientation-specific. D4 averaging washes out signal.
- Insight: TTA helpfulness depends on scale — at coarse 90 nm/px it helps, at fine 45 nm/px it hurts.
- Config D (non-TTA DINOv2 branches, TTA only on BiomedCLIP) stays honest champion at 0.6887.

## Wave 7 (multi-scale tile experiment — RED-TEAM APPROVED, NEW CHAMPION)

- **Config D**: DINOv2-B 90 nm/px (non-TTA) + DINOv2-B 45 nm/px (non-TTA) + BiomedCLIP-TTA → v2 recipe → geom-mean = **0.6887** (macro 0.5541).
- **+0.0325 over v2 champion** — 4× larger than the rejected E7 delta (+0.008).
- Per-class lifts are BROAD (red-team-measured): Healthy +0.048, Diabetes +0.042, SM +0.040, Glaukom +0.015, SucheOko -0.065 (1-scan swing). 4/5 classes improve.
- 45 nm/px ALONE is a stronger single-scale signal than 90 nm/px (0.6544 vs 0.6162 with DINOv2-B).
- **Red-team bootstrap B=1000**: Δ vs v2-noTTA (fair) = +0.039, CI [+0.010, +0.069], **P(Δ>0) = 0.999** strictly positive.
- **VERDICT: SHIP as v4 champion.** Report: `reports/RED_TEAM_MULTISCALE.md`.

## Wave 6 (complete — REJECTED by red-team)

- **Multichannel × v2 recipe fusion E7:** DINOv2-B Height + DINOv2-B RGB(H+Amp+Phase) + BiomedCLIP-H-TTA = 0.6645 (point estimate).
- **Red-team bootstrap audit**: 95% CI for ΔF1(E7−E1) = [−0.04, +0.05], P(gain>0) = 0.598 — within noise.
- **Per-class breakdown exposed fragility**: 158% of +0.0083 gain comes from Diabetes (36 scans); PGOV_Glaukom −0.057, SucheOko 1→0 correct.
- **Verdict: DO NOT SHIP. Keep v2 (0.6562) as champion.** Red-team discipline saved us from publishing noise. See `reports/RED_TEAM_E7_BOOTSTRAP.md`.

## Next-wave candidates (Wave 7 if time allows)

| Hypothesis | Rationale | Estimated EV |
|---|---|---:|
| Per-tile LR + v2 recipe aggregation | Currently scan-mean; per-tile might help Diabetes majority-single-patient | +0.005–0.01 |
| Temperature scaling before geom-mean | Calibrate soft assignments | +0.002–0.005 |
| 3-encoder ensemble (+ DINOv2-S via TTA) | More diversity | +0.005–0.01 |
| Multi-scale tiles (45 nm/px + 90 nm/px) | Cross-scale texture | +0.005–0.015 |
| DINOv2-L (1024-dim, big) | Bigger model | +0.005 but fragile |
| OpenCLIP ViT-L/14 | Alternative foundation | +0.005 |

## Data ceiling

Per `reports/ERROR_ANALYSIS.md`: realistic remaining headroom is **+0.005 to +0.015 F1** from any single intervention. SucheOko 2-patient problem is a HARD ceiling.
