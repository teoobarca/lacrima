"""Physics-informed simulation of tear ferning via Cahn-Hilliard phase-field PDE.

This module provides a pure-NumPy 2D Cahn-Hilliard solver used to generate
synthetic "ferning" height maps per tear-disease class.

PDE (dimensionless form):

    ∂u/∂t = M ∇² μ
    μ     = df/du - ε² ∇² u
    f(u)  = (1/4)(u² - 1)²      # symmetric double-well, minima at u=±1

Numerics: Eyre-style semi-implicit spectral scheme on periodic boundary
conditions. The linear fourth-order term `M ε² ∇⁴ u` is treated implicitly
in Fourier space so we get unconditional linear stability; the nonlinear
`M ∇² f'(u)` term is treated explicitly. This lets us use large dt and
reach the phase-separated state in a few hundred FFT steps.

Disease hypothesis → PDE parameter mapping
==========================================

The 5 TRAIN_SET classes correspond to physiologically-motivated tweaks of
the default spinodal decomposition:

  ZdraviLudia (healthy): balanced composition, moderate mobility, well-
    developed dendrites — baseline.
  Diabetes: hyperglycaemia increases tear osmolarity and solute load; we
    model this as a positive initial-concentration bias (u₀ > 0) and slightly
    higher ε² (thicker interfaces, chunkier crystals).
  PGOV_Glaukom: MMP-9 activity disrupts the mucin matrix — surface tension
    drops, so ε² is reduced → thin fragmented branches.
  SklerozaMultiplex: altered protein composition changes solute diffusivity;
    we tune the mobility M and free-energy depth so branching is elongated
    but less dense.
  SucheOko (dry eye): lipid layer disruption + very low aqueous fraction →
    much lower mobility, larger initial noise amplitude, incomplete ferning
    with isolated droplet-like blobs.

These are cartoons of real biology — the purpose is to generate visually
distinct synthetic textures that *might* help foundation-model embeddings
separate the disease classes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# PDE parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CHParams:
    """Parameters for the dimensionless Cahn-Hilliard solver."""

    grid: int = 256            # side length of the square grid
    dx: float = 1.0            # grid spacing
    dt: float = 1.0            # time step (semi-implicit — can be large)
    steps: int = 600           # number of time steps
    mobility: float = 1.0      # M (diffusion prefactor)
    epsilon2: float = 1.5      # ε² (square of interface width)
    u0_mean: float = 0.0       # average composition (−1..+1)
    u0_amp: float = 0.05       # noise amplitude in the initial field
    well_depth: float = 1.0    # multiplier on f(u) = (well_depth/4)(u²−1)²
    seed: int = 0              # RNG seed


# Per-class parameter presets (base). Per-sample variation is added on top.
CLASS_PRESETS: dict[str, CHParams] = {
    # Healthy: balanced spinodal, moderate mobility, good dendrite branching.
    "ZdraviLudia": CHParams(
        grid=256, dt=1.0, steps=600,
        mobility=1.0, epsilon2=1.5,
        u0_mean=0.0, u0_amp=0.05,
        well_depth=1.0, seed=0,
    ),
    # Diabetes: higher solute concentration (u0 bias), slightly thicker
    # interfaces → chunkier branches.
    "Diabetes": CHParams(
        grid=256, dt=1.0, steps=600,
        mobility=1.0, epsilon2=2.5,
        u0_mean=0.18, u0_amp=0.05,
        well_depth=1.0, seed=0,
    ),
    # Glaucoma: MMP-9 drops surface tension → low ε², thin fragmented
    # branches, elevated noise.
    "PGOV_Glaukom": CHParams(
        grid=256, dt=1.0, steps=600,
        mobility=1.0, epsilon2=0.6,
        u0_mean=0.0, u0_amp=0.08,
        well_depth=1.0, seed=0,
    ),
    # Multiple Sclerosis: altered protein diffusivity — elongated branches,
    # shallower well.
    "SklerozaMultiplex": CHParams(
        grid=256, dt=1.0, steps=600,
        mobility=1.6, epsilon2=1.1,
        u0_mean=-0.05, u0_amp=0.06,
        well_depth=0.7, seed=0,
    ),
    # Dry Eye: very low mobility + high initial heterogeneity → incomplete
    # ferning with isolated blobs.
    "SucheOko": CHParams(
        grid=256, dt=1.0, steps=300,
        mobility=0.35, epsilon2=2.0,
        u0_mean=-0.12, u0_amp=0.18,
        well_depth=1.0, seed=0,
    ),
}


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _init_field(p: CHParams) -> np.ndarray:
    rng = np.random.default_rng(p.seed)
    u = p.u0_mean + p.u0_amp * rng.standard_normal((p.grid, p.grid)).astype(np.float32)
    return u


def _k_squared(grid: int, dx: float) -> np.ndarray:
    """Return |k|² on the FFT grid (real Laplacian eigenvalue magnitude)."""
    k = 2.0 * np.pi * np.fft.fftfreq(grid, d=dx).astype(np.float64)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return (kx * kx + ky * ky)


def simulate(p: CHParams, return_trace: bool = False) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
    """Run a Cahn-Hilliard simulation. Returns final u field shape (grid, grid).

    Uses a semi-implicit Fourier scheme:

        (1 + dt M ε² k⁴) û_{n+1} = û_n − dt M k² · F[ f'(u_n) ]

    The implicit treatment of the ∇⁴ term removes the severe dt ∝ dx⁴
    restriction of an explicit Euler scheme.

    If `return_trace` is True, also returns a list of ~10 snapshots at log-
    spaced time points (useful for visualising dendrite growth).
    """
    u = _init_field(p).astype(np.float64)
    trace: list[np.ndarray] = []
    snap_steps: set[int] = set()
    if return_trace:
        snap_steps = set(np.unique(np.round(
            np.geomspace(1, max(p.steps, 2), 10)
        ).astype(int)).tolist())

    k2 = _k_squared(p.grid, p.dx)
    # Denominator for implicit step: 1 + dt * M * ε² * k⁴
    denom = 1.0 + p.dt * p.mobility * p.epsilon2 * (k2 * k2)

    for step in range(p.steps):
        # f'(u) = well_depth * u * (u² − 1)
        dfdu = p.well_depth * u * (u * u - 1.0)
        # û_{n+1} = [ û_n − dt*M*k²*F[f'(u_n)] ] / (1 + dt*M*ε²*k⁴)
        u_hat = np.fft.fft2(u)
        f_hat = np.fft.fft2(dfdu)
        u_hat_new = (u_hat - p.dt * p.mobility * k2 * f_hat) / denom
        u = np.real(np.fft.ifft2(u_hat_new))
        if not np.isfinite(u).all():
            break
        if return_trace and (step + 1) in snap_steps:
            trace.append(u.astype(np.float32).copy())

    u = u.astype(np.float32)
    if return_trace:
        return u, trace
    return u


# ---------------------------------------------------------------------------
# Post-processing (height map for DINOv2)
# ---------------------------------------------------------------------------

def field_to_height(u: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Map a Cahn-Hilliard field u ∈ [−1, +1] to a [0, 1] height map.

    Uses the same robust percentile normalisation as real AFM scans (see
    teardrop.data.robust_normalize), so simulated and real images receive
    identical downstream preprocessing.
    """
    lo, hi = np.percentile(u, [p_low, p_high])
    if hi - lo < 1e-6:
        return np.zeros_like(u, dtype=np.float32)
    h = np.clip((u - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return h


def sample_params(
    base: CHParams,
    rng: np.random.Generator,
    *,
    jitter: float = 0.15,
) -> CHParams:
    """Return a CHParams near `base` with small multiplicative jitter.

    `jitter` is the 1-sigma relative jitter on mobility, epsilon2 and
    u0_amp. u0_mean gets absolute additive jitter (±jitter*0.1).
    """
    def j(x: float) -> float:
        return float(x * (1.0 + jitter * rng.standard_normal()))

    return replace(
        base,
        mobility=max(0.05, j(base.mobility)),
        epsilon2=max(0.1, j(base.epsilon2)),
        u0_amp=max(0.01, j(base.u0_amp)),
        u0_mean=float(base.u0_mean + 0.1 * jitter * rng.standard_normal()),
        seed=int(rng.integers(0, 2**31 - 1)),
    )


def simulate_class(
    cls: str,
    n: int,
    *,
    jitter: float = 0.15,
    master_seed: int = 42,
    steps: Optional[int] = None,
) -> list[np.ndarray]:
    """Generate `n` simulated height maps for the given class.

    Returns a list of 2D float32 arrays in [0, 1] (shape = grid×grid).
    """
    if cls not in CLASS_PRESETS:
        raise KeyError(f"Unknown class: {cls}")
    base = CLASS_PRESETS[cls]
    if steps is not None:
        base = replace(base, steps=steps)
    rng = np.random.default_rng(master_seed + hash(cls) % 10_000)
    out: list[np.ndarray] = []
    for _ in range(n):
        p = sample_params(base, rng, jitter=jitter)
        u = simulate(p)
        out.append(field_to_height(u))
    return out
