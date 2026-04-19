"""Topological Data Analysis features for AFM tear-desiccate height maps.

Pipeline:
    height map h in [0,1] of shape (H, W)
        --> SUBLEVEL cubical persistent homology (cripser) for H_0 + H_1
        --> persistence diagrams for each dim
        --> vectorize: persistence images + statistics + landscapes
        --> optionally repeat at downsampled scale for multi-scale topology

Why TDA?
    Dendritic crystallization in tear droplets is fundamentally topological:
    - H_0 captures connected components born at low height thresholds
        (number of separate dendrite branches as we sweep up the height filtration)
    - H_1 captures loops / enclosed cells (mesh density, dendritic branching pattern)
    Persistence (death - birth) measures how "robust" each feature is, which is
    invariant to small noise but sensitive to true geometric structure.

Output: ~150 floats per image, suitable for sklearn / XGBoost.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Persistent homology backends
# ---------------------------------------------------------------------------

def _compute_cubical_ph(h: np.ndarray, maxdim: int = 1) -> np.ndarray:
    """Sublevel cubical persistent homology via cripser.

    Returns array of shape (n_features, 9) with columns
        [dim, birth, death, x1, y1, z1, x2, y2, z2].
    """
    import cripser
    return cripser.computePH(h.astype(np.float32), maxdim=maxdim)


def _diag_for_dim(ph: np.ndarray, dim: int, max_val: float = 1.0) -> np.ndarray:
    """Extract (birth, death) pairs for a given homology dimension.

    Replaces +inf deaths (essential classes) with `max_val` so persistence is finite.
    Drops zero-persistence pairs (b == d).
    """
    rows = ph[ph[:, 0] == dim]
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    births = rows[:, 1].astype(np.float64)
    deaths_raw = rows[:, 2].astype(np.float64)
    deaths_raw = np.where(np.isfinite(deaths_raw), deaths_raw, max_val)
    deaths = np.minimum(deaths_raw, max_val).astype(np.float32)
    births = births.astype(np.float32)
    diag = np.stack([births, deaths], axis=1)
    pers = diag[:, 1] - diag[:, 0]
    diag = diag[pers > 1e-6]
    return diag


# ---------------------------------------------------------------------------
# Vectorizers
# ---------------------------------------------------------------------------

def _persistence_stats(diag: np.ndarray, prefix: str) -> dict[str, float]:
    """Summary statistics of a persistence diagram (b, d pairs)."""
    out: dict[str, float] = {}
    if diag.shape[0] == 0:
        for k in ("n", "n_significant", "total_pers", "max_pers", "mean_pers",
                  "std_pers", "median_pers", "p90_pers", "p99_pers",
                  "entropy", "mean_birth", "mean_midlife", "std_midlife"):
            out[f"{prefix}_{k}"] = 0.0
        return out

    pers = diag[:, 1] - diag[:, 0]
    midlife = (diag[:, 0] + diag[:, 1]) / 2
    out[f"{prefix}_n"] = float(diag.shape[0])
    out[f"{prefix}_n_significant"] = float((pers > 0.05).sum())
    out[f"{prefix}_total_pers"] = float(pers.sum())
    out[f"{prefix}_max_pers"] = float(pers.max())
    out[f"{prefix}_mean_pers"] = float(pers.mean())
    out[f"{prefix}_std_pers"] = float(pers.std())
    out[f"{prefix}_median_pers"] = float(np.median(pers))
    out[f"{prefix}_p90_pers"] = float(np.percentile(pers, 90))
    out[f"{prefix}_p99_pers"] = float(np.percentile(pers, 99))
    # Persistence entropy (Atienza et al.)
    p = pers / max(pers.sum(), 1e-12)
    p = p[p > 0]
    out[f"{prefix}_entropy"] = float(-(p * np.log(p)).sum()) if p.size else 0.0
    out[f"{prefix}_mean_birth"] = float(diag[:, 0].mean())
    out[f"{prefix}_mean_midlife"] = float(midlife.mean())
    out[f"{prefix}_std_midlife"] = float(midlife.std())
    return out


def _persistence_image(
    diag: np.ndarray,
    prefix: str,
    grid: int = 8,
    sigma: float = 0.05,
) -> dict[str, float]:
    """Hand-rolled persistence image on a grid x grid lattice.

    Uses the (birth, persistence) representation. Each (b, p) point is
    weighted by a linear ramp w(p) = p (so longer-lived features dominate)
    and smeared with an isotropic Gaussian of stddev sigma.
    Result is a fixed-length feature vector of length grid*grid.
    """
    feats: dict[str, float] = {}
    img = np.zeros((grid, grid), dtype=np.float32)
    if diag.shape[0] == 0:
        for i in range(grid * grid):
            feats[f"{prefix}_pi{i:02d}"] = 0.0
        return feats

    bx = np.linspace(0, 1, grid)  # birth axis [0,1]
    py = np.linspace(0, 1, grid)  # persistence axis [0,1]
    BX, PY = np.meshgrid(bx, py, indexing="xy")

    births = diag[:, 0]
    pers = diag[:, 1] - diag[:, 0]
    weights = pers  # linear ramp weight
    inv2s2 = 1.0 / (2 * sigma * sigma)

    for b, p, w in zip(births, pers, weights):
        # Gaussian centered at (b, p)
        gauss = np.exp(-((BX - b) ** 2 + (PY - p) ** 2) * inv2s2)
        img += (w * gauss).astype(np.float32)

    flat = img.ravel()
    for i, v in enumerate(flat):
        feats[f"{prefix}_pi{i:02d}"] = float(v)
    return feats


def _persistence_landscape(
    diag: np.ndarray,
    prefix: str,
    n_levels: int = 3,
    n_samples: int = 16,
) -> dict[str, float]:
    """Sampled persistence landscape: top n_levels functions sampled at n_samples points.

    Output dim = n_levels * n_samples per diagram.
    """
    feats: dict[str, float] = {}
    xs = np.linspace(0, 1, n_samples)
    if diag.shape[0] == 0:
        for li in range(n_levels):
            for i in range(n_samples):
                feats[f"{prefix}_ls{li}_{i:02d}"] = 0.0
        return feats

    births = diag[:, 0]
    deaths = diag[:, 1]
    # Per-feature tent function: max(0, min(x-b, d-x)).
    # Stack -> shape (n_features, n_samples)
    tents = np.maximum(
        0.0,
        np.minimum(xs[None, :] - births[:, None], deaths[:, None] - xs[None, :]),
    )
    # Sort each column descending; take top n_levels.
    tents_sorted = -np.sort(-tents, axis=0)
    n_avail = tents_sorted.shape[0]
    for li in range(n_levels):
        if li < n_avail:
            row = tents_sorted[li]
        else:
            row = np.zeros(n_samples, dtype=np.float32)
        for i, v in enumerate(row):
            feats[f"{prefix}_ls{li}_{i:02d}"] = float(v)
    return feats


# ---------------------------------------------------------------------------
# Top-level extractor
# ---------------------------------------------------------------------------

def _maybe_downsample(h: np.ndarray, max_side: int) -> np.ndarray:
    """Box-average to <= max_side on the longer edge (preserves topology better than NN)."""
    H, W = h.shape
    side = max(H, W)
    if side <= max_side:
        return h
    factor = int(np.ceil(side / max_side))
    new_H = (H // factor) * factor
    new_W = (W // factor) * factor
    h = h[:new_H, :new_W]
    h = h.reshape(new_H // factor, factor, new_W // factor, factor).mean(axis=(1, 3))
    return h.astype(np.float32)


def persistence_features(
    h: np.ndarray,
    max_side: int = 256,
    multiscale_factors: Iterable[int] = (1, 2),
    pi_grid: int = 8,
    pi_sigma: float = 0.05,
    landscape_levels: int = 3,
    landscape_samples: int = 16,
) -> dict[str, float]:
    """Extract a fixed-length topological feature vector from a height map.

    Parameters
    ----------
    h : np.ndarray
        Height map in [0,1], shape (H, W).
    max_side : int
        Downsample to at most this many pixels per side before persistence
        (cubical PH is O(N) but quadratic memory; 256 is plenty for AFM topology).
    multiscale_factors : iterable of int
        For each factor f, also compute features on h downsampled by another f
        on top of max_side. Default (1, 2) gives base + 2x-coarse view.
    pi_grid : int
        Persistence image grid size (PI dim per diagram = pi_grid**2).
    pi_sigma : float
        Persistence image Gaussian bandwidth.
    landscape_levels, landscape_samples : int
        Persistence landscape parameters (per-diagram dim = levels*samples).

    Returns
    -------
    dict[str, float]
        Feature dictionary; size depends on the parameters but is fixed
        across all calls with the same arguments.
    """
    # Downsample once to base resolution
    h0 = _maybe_downsample(h, max_side)

    feats: dict[str, float] = {}

    for factor in multiscale_factors:
        if factor == 1:
            hf = h0
        else:
            # average-pool by factor
            new_H = (h0.shape[0] // factor) * factor
            new_W = (h0.shape[1] // factor) * factor
            hf = h0[:new_H, :new_W]
            hf = hf.reshape(new_H // factor, factor, new_W // factor, factor).mean(axis=(1, 3))
            hf = hf.astype(np.float32)

        # Compute sublevel and superlevel (= sublevel of -h shifted to [0,1]).
        # Superlevel persistence captures "high-elevation" topology = the
        # bright dendrite ridges themselves rather than dark valleys.
        for orient, hh in (("sub", hf), ("sup", 1.0 - hf)):
            try:
                ph = _compute_cubical_ph(hh, maxdim=1)
            except Exception:
                ph = np.zeros((0, 9), dtype=np.float32)

            for dim in (0, 1):
                diag = _diag_for_dim(ph, dim, max_val=1.0)
                prefix = f"f{factor}_{orient}_h{dim}"
                feats.update(_persistence_stats(diag, prefix))
                feats.update(_persistence_image(
                    diag, prefix, grid=pi_grid, sigma=pi_sigma,
                ))
                feats.update(_persistence_landscape(
                    diag, prefix,
                    n_levels=landscape_levels,
                    n_samples=landscape_samples,
                ))

    # Threshold-based topology counts (Betti curves at fixed thresholds).
    # Quick proxy: number of connected components / loops in binarized image.
    feats.update(_betti_curve_features(h0, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)))

    return feats


def _betti_curve_features(
    h: np.ndarray,
    thresholds: Iterable[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
) -> dict[str, float]:
    """Number of CCs / holes in binarized image at several thresholds.

    A cheap surrogate for Betti curves. Uses scipy.ndimage.
    """
    from scipy import ndimage

    out: dict[str, float] = {}
    structure = np.ones((3, 3), dtype=np.uint8)
    H, W = h.shape
    area = float(H * W)
    for t in thresholds:
        bw = (h >= t).astype(np.uint8)
        _, n_cc = ndimage.label(bw, structure=structure)
        # Holes: CC count of background restricted to non-border holes.
        bg = 1 - bw
        lbl, n_bg = ndimage.label(bg, structure=structure)
        # A hole = bg-component that doesn't touch the image border.
        border_labels = set(lbl[0, :].tolist() + lbl[-1, :].tolist()
                            + lbl[:, 0].tolist() + lbl[:, -1].tolist())
        border_labels.discard(0)
        n_holes = max(0, n_bg - len(border_labels))
        coverage = float(bw.sum()) / area
        tt = f"{int(t*100):02d}"
        out[f"betti_b0_t{tt}"] = float(n_cc)
        out[f"betti_b1_t{tt}"] = float(n_holes)
        out[f"betti_cov_t{tt}"] = coverage
    return out


def feature_dim(
    pi_grid: int = 8,
    landscape_levels: int = 3,
    landscape_samples: int = 16,
    multiscale_factors: Iterable[int] = (1, 2),
    n_thresholds: int = 5,
) -> int:
    """Return the expected number of features for given parameters."""
    n_stats = 13
    per_diag = n_stats + pi_grid * pi_grid + landscape_levels * landscape_samples
    n_diags = len(list(multiscale_factors)) * 2 * 2  # factors x (sub/sup) x (H0/H1)
    return n_diags * per_diag + 3 * n_thresholds
