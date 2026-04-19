# Physics-informed simulation experiment

_Cahn-Hilliard spinodal decomposition → DINOv2-B embedding space_

Compute time: **364.6 s** total (simulation + encoding + UMAP)

## Motivation

Tear ferning is classically modeled as a mix of spinodal decomposition (salt-water phase separation) and dendritic solidification. We use the simplest physical pretext — the 2D Cahn-Hilliard equation — with class-specific parameter tweaks to generate synthetic tear textures, then ask whether a frozen DINOv2-B encoder places them near the intended real-class centroids.

## PDE and class parameter mapping

```
∂u/∂t = M ∇² (f'(u) − ε² ∇² u),   f(u) = (well_depth/4)(u²−1)²
```

| Class | u₀_mean | u₀_amp | ε² | M | well_depth | steps | biological rationale |
|---|---|---|---|---|---|---|---|
| Healthy | +0.00 | 0.05 | 1.50 | 1.00 | 1.00 | 600 | baseline balanced spinodal |
| Diabetes | +0.18 | 0.05 | 2.50 | 1.00 | 1.00 | 600 | hyperglycaemia → u₀ bias, higher ε² (thicker interfaces) |
| Glaucoma | +0.00 | 0.08 | 0.60 | 1.00 | 1.00 | 600 | MMP-9 disrupts mucin matrix → low ε² (fine branches) |
| Multiple Sclerosis | -0.05 | 0.06 | 1.10 | 1.60 | 0.70 | 600 | altered protein diffusivity → higher M, shallower well |
| Dry Eye | -0.12 | 0.18 | 2.00 | 0.35 | 1.00 | 300 | low aqueous fraction → low M, halted evolution, high initial noise |

30 simulations per class with 15% multiplicative jitter on mobility / ε² / initial-noise amplitude and independent RNG seeds.

## Nearest-centroid diagnostic

For each simulation we compute its cosine similarity to the five real-class centroids (in the DINOv2-B embedding space, fit only on real scans) and ask whether the top-1 / top-3 nearest centroid matches the intended class.

**Overall top-1: 21.3%** (chance = 20%)
**Overall top-3: 43.3%**

| Class | top-1 | top-2 | top-3 | mean cos-sim to own centroid |
|---|---|---|---|---|
| Healthy | 97% | 100% | 100% | +0.087 |
| Diabetes | 0% | 0% | 3% | -0.067 |
| Glaucoma | 10% | 80% | 100% | +0.005 |
| Multiple Sclerosis | 0% | 3% | 13% | -0.043 |
| Dry Eye | 0% | 0% | 0% | -0.146 |

## Augmentation experiment (LOPO)

We add the simulated embeddings into each LOPO training fold with their intended labels and compare scan-level weighted-F1 against the real-only baseline.

- Real only: **0.621**
- Real + sim: **0.622**
- Delta: **+0.002**

**Verdict:** No meaningful effect — the simulations are far enough from the real manifold that the logistic probe ignores them.

## Honest limitations

- Pure Cahn-Hilliard gives **isotropic labyrinthine** phase separation. Real tear scans show directional dendritic ferning driven by directional drying and anisotropic surface energy, which CH alone cannot reproduce.
- The class-parameter mapping is a cartoon; we are not claiming each term corresponds quantitatively to a measured biological quantity.
- Even if simulations do not land on real centroids, the exercise still documents the physics → embedding gap, which is useful for future directions (phase-field crystal, Kobayashi-Warren anisotropic solidification, or DDPM conditioned on real scans).

## Artefacts

- `teardrop/physics_sim.py` — solver + class presets.
- `scripts/physics_pretraining_experiment.py` — this experiment.
- `reports/pitch/10_physics_simulation.png` — real vs sim grid + UMAP.
- `cache/physics_sim_dinov2b_embeddings.npz` — simulated embeddings (reused on re-run).
