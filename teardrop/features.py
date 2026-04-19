"""Handcrafted feature extraction for AFM tear-desiccate height maps.

Features (per image):
- Surface roughness: Sa, Sq, Ssk, Sku, Sz_p2p, range
- GLCM Haralick (4 distances × 4 angles): contrast, correlation, energy, homogeneity, ASM, dissimilarity
- LBP (Local Binary Patterns): rotation-invariant histogram (24 bins)
- Fractal dimension (box counting on binarized image)
- HOG: histogram of oriented gradients (low-dim summary)
- Histogram statistics: percentiles + spread
"""
from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

# ---------------------------------------------------------------------------
# Roughness (height-domain statistics)
# ---------------------------------------------------------------------------

def roughness_features(h: np.ndarray) -> dict[str, float]:
    """Standard ISO surface roughness metrics. Input expected normalized to [0,1]."""
    flat = h.flatten()
    mean = float(flat.mean())
    centered = flat - mean
    Sa = float(np.abs(centered).mean())
    Sq = float(np.sqrt((centered ** 2).mean()))
    Sz = float(flat.max() - flat.min())
    if Sq < 1e-9:
        Ssk = 0.0
        Sku = 3.0
    else:
        Ssk = float((centered ** 3).mean() / (Sq ** 3))
        Sku = float((centered ** 4).mean() / (Sq ** 4))
    p1, p5, p50, p95, p99 = np.percentile(flat, [1, 5, 50, 95, 99])
    return {
        "Sa": Sa, "Sq": Sq, "Sz": Sz, "Ssk": Ssk, "Sku": Sku,
        "p1": float(p1), "p5": float(p5), "p50": float(p50),
        "p95": float(p95), "p99": float(p99),
        "iqr": float(p95 - p5),
    }


# ---------------------------------------------------------------------------
# GLCM (Gray-Level Co-occurrence Matrix)
# ---------------------------------------------------------------------------

def glcm_features(
    h: np.ndarray,
    levels: int = 32,
    distances: tuple[int, ...] = (1, 3, 5, 9),
    angles: tuple[float, ...] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> dict[str, float]:
    """GLCM Haralick features over multiple distances + angles."""
    img = (h * (levels - 1)).astype(np.uint8)
    glcm = graycomatrix(img, distances=list(distances), angles=list(angles),
                        levels=levels, symmetric=True, normed=True)
    feats = {}
    for prop in ("contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"):
        vals = graycoprops(glcm, prop=prop)  # shape (n_dist, n_ang)
        for di, d in enumerate(distances):
            feats[f"glcm_{prop}_d{d}_mean"] = float(vals[di].mean())
            feats[f"glcm_{prop}_d{d}_std"] = float(vals[di].std())
    return feats


# ---------------------------------------------------------------------------
# LBP (Local Binary Patterns)
# ---------------------------------------------------------------------------

def lbp_features(
    h: np.ndarray,
    radius: int = 3,
    n_points: int = 24,
) -> dict[str, float]:
    """Rotation-invariant uniform LBP histogram (n_points + 2 bins)."""
    img = (h * 255).astype(np.uint8)
    lbp = local_binary_pattern(img, P=n_points, R=radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return {f"lbp_{i}": float(hist[i]) for i in range(n_bins)}


# ---------------------------------------------------------------------------
# Fractal dimension (box counting)
# ---------------------------------------------------------------------------

def fractal_dimension(h: np.ndarray, n_thresholds: int = 5) -> dict[str, float]:
    """Mean box-counting fractal dimension over multiple thresholds.

    Binarize at multiple percentile thresholds, compute box-counting D for each,
    return mean + std. More stable than single-threshold.
    """
    out = {}
    Ds = []
    for pct in np.linspace(40, 80, n_thresholds):
        thresh = np.percentile(h, pct)
        binary = h > thresh
        D = _boxcount_dim(binary)
        Ds.append(D)
    out["fractal_D_mean"] = float(np.mean(Ds))
    out["fractal_D_std"] = float(np.std(Ds))
    return out


def _boxcount_dim(binary: np.ndarray) -> float:
    """Box-counting dimension on a binary image."""
    sizes = []
    counts = []
    n = min(binary.shape)
    s = 2
    while s <= n // 4:
        # Reshape & count boxes that contain at least one True pixel
        H, W = binary.shape
        H2 = (H // s) * s
        W2 = (W // s) * s
        b = binary[:H2, :W2]
        b = b.reshape(H2 // s, s, W2 // s, s).any(axis=(1, 3))
        sizes.append(s)
        counts.append(int(b.sum()))
        s *= 2
    if len(sizes) < 2:
        return 0.0
    log_s = np.log(1.0 / np.array(sizes, dtype=float))
    log_n = np.log(np.array(counts, dtype=float) + 1e-9)
    coef = np.polyfit(log_s, log_n, 1)
    return float(coef[0])


# ---------------------------------------------------------------------------
# HOG (Histogram of Oriented Gradients) summary
# ---------------------------------------------------------------------------

def hog_features(h: np.ndarray) -> dict[str, float]:
    """HOG with coarse cells; return summary stats (mean, std, min, max, percentiles)."""
    img = (h * 255).astype(np.uint8)
    descriptor = hog(
        img,
        orientations=9,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        feature_vector=True,
        block_norm="L2-Hys",
    )
    p25, p50, p75 = np.percentile(descriptor, [25, 50, 75])
    return {
        "hog_mean": float(descriptor.mean()),
        "hog_std": float(descriptor.std()),
        "hog_min": float(descriptor.min()),
        "hog_max": float(descriptor.max()),
        "hog_p25": float(p25), "hog_p50": float(p50), "hog_p75": float(p75),
    }


# ---------------------------------------------------------------------------
# Combined extractor
# ---------------------------------------------------------------------------

def extract_all_features(h: np.ndarray) -> dict[str, float]:
    """Extract all handcrafted features. Input: 2D float array in [0,1]."""
    feats: dict[str, float] = {}
    feats.update(roughness_features(h))
    feats.update(glcm_features(h))
    feats.update(lbp_features(h))
    feats.update(fractal_dimension(h))
    feats.update(hog_features(h))
    return feats


def feature_names() -> list[str]:
    """Return ordered list of feature names produced by extract_all_features."""
    dummy = np.random.rand(64, 64).astype(np.float32)
    return list(extract_all_features(dummy).keys())
