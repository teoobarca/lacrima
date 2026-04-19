"""Advanced texture-feature extractors for AFM tear-desiccate height maps.

Each function takes ``h: np.ndarray[float32] in [0, 1]`` (2-D) and returns a
``dict[str, float]``. The combined :func:`extract_all_advanced_features`
bundles all of them together (~250 features).

Physically motivated extractors:

1. Multifractal spectrum f(alpha) via box-counting q-moments.
2. Lacunarity at multiple scales (gliding box).
3. Succolarity — directional permeability (4 directions).
4. Wavelet-packet tree energies (db4, 3 levels).
5. Hurst exponent + DFA + trend CV.
6. Extended GLCM (many distances x angles x 6 props).
7. Gabor filter bank responses (8 orientations x 4 frequencies).
8. Multi-scale HOG (cells 8, 16, 32 px).

Only numpy / scipy / skimage / pywt are used — no heavy extras.
"""
from __future__ import annotations

import math
import warnings
from typing import Iterable

import numpy as np
import pywt
from scipy import ndimage
from scipy.signal import fftconvolve
from skimage.feature import graycomatrix, graycoprops, hog
from skimage.filters import gabor_kernel

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 1. Multifractal spectrum f(alpha) via box-counting q-moments
# ---------------------------------------------------------------------------

_MF_Q_VALUES: tuple[float, ...] = (-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0)


def multifractal_features(
    h: np.ndarray,
    q_values: Iterable[float] = _MF_Q_VALUES,
    box_sizes: Iterable[int] = (4, 8, 16, 32, 64, 128),
) -> dict[str, float]:
    """Multifractal spectrum via box-counting q-moments.

    For each scale s and moment q, compute the partition sum
    Z(q, s) = sum_i p_i(s)^q   where p_i = box-mass / total-mass.
    Fit log Z(q, s) ~ tau(q) log s, then
        alpha(q)   = d tau / dq       (numerical derivative)
        f(alpha) = q * alpha(q) - tau(q)     (Legendre transform).

    Returns ``f(alpha)`` at each q, ``alpha(q)`` at each q,
    alpha_min / max, width Delta_alpha, asymmetry.
    """
    q_values = tuple(float(q) for q in q_values)
    total = float(h.sum())
    if total < 1e-12:
        # Degenerate flat image — return zeros.
        return _mf_zeros(q_values)

    # Partition sums for each (q, s).
    log_s: list[float] = []
    Zqs: dict[float, list[float]] = {q: [] for q in q_values}

    H, W = h.shape
    for s in box_sizes:
        if s > min(H, W):
            continue
        H2 = (H // s) * s
        W2 = (W // s) * s
        if H2 == 0 or W2 == 0:
            continue
        b = h[:H2, :W2]
        # Sum of pixel values inside each s x s box -> mass.
        boxes = b.reshape(H2 // s, s, W2 // s, s).sum(axis=(1, 3))
        mass = boxes.ravel() / total
        # Floor tiny / empty boxes so negative-q moments stay finite.
        # Any box with < 1e-6 of total mass is lifted to that floor; keeps the
        # multifractal slope well-defined for the heavy-tail q<0 side.
        mass = np.maximum(mass, 1e-6)
        mass = mass / mass.sum()  # renormalize
        log_s.append(math.log(1.0 / s))
        for q in q_values:
            if abs(q - 1.0) < 1e-9:
                # Special case q = 1: use Shannon form so derivative is smooth.
                Z = float(np.sum(mass * np.log(mass)))
                # tau(1) = 0 identically; we need mu(s) = Z which defines
                # alpha(1) directly via slope of Z(s) vs log s.
                Zqs[q].append(Z)
            else:
                Z = float(np.sum(mass ** q))
                # log with a floor keeps this finite even if Z is tiny/huge.
                Zqs[q].append(math.log(max(Z, 1e-300)))

    if len(log_s) < 2:
        return _mf_zeros(q_values)

    log_s_arr = np.asarray(log_s, dtype=np.float64)

    # tau(q) = slope of log Z(q, s) vs log s  (for q != 1).
    tau: dict[float, float] = {}
    for q in q_values:
        y = np.asarray(Zqs[q], dtype=np.float64)
        if len(y) < 2:
            tau[q] = 0.0
            continue
        if abs(q - 1.0) < 1e-9:
            # For q=1, alpha(1) = d Z / d log s where Z = sum p log p.
            # We'll store the slope as tau_1; true tau(1)=0 though.
            slope, _ = np.polyfit(log_s_arr, y, 1)
            tau[q] = float(slope)
        else:
            slope, _ = np.polyfit(log_s_arr, y, 1)
            tau[q] = float(slope)

    # alpha(q) = d tau / d q via central differences over the provided q grid.
    qs = np.asarray(q_values, dtype=np.float64)
    tau_arr = np.asarray([tau[q] for q in q_values], dtype=np.float64)
    alpha = np.gradient(tau_arr, qs)

    # f(alpha) = q * alpha - tau.  For q=1 we set tau(1)=0 (theoretical value).
    f_alpha = np.zeros_like(alpha)
    for i, q in enumerate(q_values):
        if abs(q - 1.0) < 1e-9:
            f_alpha[i] = 1.0 * alpha[i] - 0.0
        else:
            f_alpha[i] = q * alpha[i] - tau_arr[i]

    feats: dict[str, float] = {}
    for q, a, f in zip(q_values, alpha, f_alpha):
        feats[f"mf_alpha_q{q:+.0f}"] = float(a)
        feats[f"mf_falpha_q{q:+.0f}"] = float(f)
    feats["mf_alpha_min"] = float(np.min(alpha))
    feats["mf_alpha_max"] = float(np.max(alpha))
    feats["mf_alpha_width"] = float(np.max(alpha) - np.min(alpha))
    # Asymmetry: distance from alpha(q=0) to each extreme.
    i0 = int(np.argmin(np.abs(qs)))  # index closest to q=0
    a0 = float(alpha[i0])
    feats["mf_alpha_asym"] = float(
        ((np.max(alpha) - a0) - (a0 - np.min(alpha)))
        / max(feats["mf_alpha_width"], 1e-9)
    )
    # Spectrum peak height -> capacity dimension D0 (at q=0).
    feats["mf_D0"] = float(f_alpha[i0])
    # Final safety: replace any non-finite with 0 so downstream never sees NaN.
    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0
    return feats


def _mf_zeros(q_values: Iterable[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for q in q_values:
        out[f"mf_alpha_q{q:+.0f}"] = 0.0
        out[f"mf_falpha_q{q:+.0f}"] = 0.0
    out["mf_alpha_min"] = 0.0
    out["mf_alpha_max"] = 0.0
    out["mf_alpha_width"] = 0.0
    out["mf_alpha_asym"] = 0.0
    out["mf_D0"] = 0.0
    return out


# ---------------------------------------------------------------------------
# 2. Lacunarity at multiple scales (gliding box, Allain-Cloitre)
# ---------------------------------------------------------------------------

def lacunarity_features(
    h: np.ndarray,
    box_sizes: Iterable[int] = (4, 8, 16, 32, 64),
) -> dict[str, float]:
    """Gliding-box lacunarity on a continuous-mass image.

    For each box of size s, compute mass M = sum of pixel values in box.
    L(s) = <M^2> / <M>^2 = 1 + Var(M) / Mean(M)^2.
    Low L -> homogeneous; high L -> gappy/clumpy.
    """
    box_sizes = tuple(int(s) for s in box_sizes)
    feats: dict[str, float] = {}
    lac_vals: list[float] = []
    log_s: list[float] = []
    H, W = h.shape
    # Integral image lets us compute box sums in O(1) each.
    S = np.zeros((H + 1, W + 1), dtype=np.float64)
    S[1:, 1:] = np.cumsum(np.cumsum(h.astype(np.float64), axis=0), axis=1)

    for s in box_sizes:
        if s >= H or s >= W:
            feats[f"lac_s{s}"] = 0.0
            continue
        # Sum over every s x s box (gliding stride=1).
        M = (
            S[s:H + 1, s:W + 1]
            - S[0:H - s + 1, s:W + 1]
            - S[s:H + 1, 0:W - s + 1]
            + S[0:H - s + 1, 0:W - s + 1]
        )
        m = float(M.mean())
        v = float(M.var())
        lac = 1.0 + v / (m * m + 1e-12)
        feats[f"lac_s{s}"] = lac
        lac_vals.append(lac)
        log_s.append(math.log(s))

    # Slope of log L(s) vs log s — often a single heterogeneity descriptor.
    if len(lac_vals) >= 2:
        slope, _ = np.polyfit(np.asarray(log_s), np.log(np.asarray(lac_vals) + 1e-12), 1)
        feats["lac_slope"] = float(slope)
        feats["lac_mean"] = float(np.mean(lac_vals))
        feats["lac_std"] = float(np.std(lac_vals))
        feats["lac_ratio_small_to_large"] = float(lac_vals[0] / (lac_vals[-1] + 1e-9))
    else:
        feats["lac_slope"] = 0.0
        feats["lac_mean"] = 0.0
        feats["lac_std"] = 0.0
        feats["lac_ratio_small_to_large"] = 0.0
    return feats


# ---------------------------------------------------------------------------
# 3. Succolarity (directional permeability)
# ---------------------------------------------------------------------------

def succolarity_features(
    h: np.ndarray,
    threshold_pct: float = 50.0,
    box_sizes: Iterable[int] = (8, 16, 32, 64),
) -> dict[str, float]:
    """Succolarity: box-counting estimate of directional connectivity.

    Algorithm (de Melo & Conci, 2013, simplified):
        1. Binarize (above/below median by default) -> 0 permeable, 1 blocked.
        2. For each direction, "flood" permeable pixels starting from one
           side edge and compute the fraction filled at each box size,
           weighted by box position. Higher -> more permeable in that direction.
    """
    # 0 = permeable (low height = valley), 1 = blocked (high height).
    thresh = np.percentile(h, threshold_pct)
    blocked = (h >= thresh).astype(np.uint8)
    permeable = 1 - blocked

    feats: dict[str, float] = {}
    for d_name, flood in (
        ("left_right", _flood_from_edge(permeable, axis=1, reverse=False)),
        ("right_left", _flood_from_edge(permeable, axis=1, reverse=True)),
        ("top_bottom", _flood_from_edge(permeable, axis=0, reverse=False)),
        ("bottom_top", _flood_from_edge(permeable, axis=0, reverse=True)),
    ):
        vals: list[float] = []
        for s in box_sizes:
            if s > min(flood.shape):
                continue
            H2 = (flood.shape[0] // s) * s
            W2 = (flood.shape[1] // s) * s
            b = flood[:H2, :W2]
            boxes = b.reshape(H2 // s, s, W2 // s, s).mean(axis=(1, 3))
            vals.append(float(boxes.mean()))
        feats[f"suc_{d_name}"] = float(np.mean(vals)) if vals else 0.0
    # Overall + anisotropy.
    vals = [feats[f"suc_{d}"] for d in ("left_right", "right_left", "top_bottom", "bottom_top")]
    feats["suc_mean"] = float(np.mean(vals))
    feats["suc_anisotropy"] = float(np.std(vals) / (np.mean(vals) + 1e-9))
    return feats


def _flood_from_edge(permeable: np.ndarray, axis: int, reverse: bool) -> np.ndarray:
    """Binary flood: a permeable pixel keeps its value only if the run of
    permeable pixels continues uninterrupted from the starting edge along the
    given axis. Returns a float32 array of the flooded mask (0/1)."""
    p = permeable
    if reverse:
        p = np.flip(p, axis=axis)
    # Cumulative AND along the axis: first blocking pixel stops the flood.
    if axis == 0:
        # For each column, True until first 0.
        # cummin of binary gives "still 1?" while cumulative product does the same.
        flooded = np.cumprod(p, axis=0)
    else:
        flooded = np.cumprod(p, axis=1)
    if reverse:
        flooded = np.flip(flooded, axis=axis)
    return flooded.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Wavelet-packet tree energies
# ---------------------------------------------------------------------------

def wavelet_packet_features(
    h: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> dict[str, float]:
    """Energy per subband of a 2-D wavelet packet decomposition (db4, 3 levels).

    A 3-level 2-D WP decomposition gives 4^3 = 64 subbands. For each subband
    we return the L2 energy; we also return global statistics.
    """
    wp = pywt.WaveletPacket2D(
        data=h.astype(np.float64),
        wavelet=wavelet,
        mode="symmetric",
        maxlevel=level,
    )
    nodes = sorted(
        (n for n in wp.get_level(level, order="natural")),
        key=lambda node: node.path,
    )
    feats: dict[str, float] = {}
    energies: list[float] = []
    for node in nodes:
        coef = node.data
        e = float(np.sum(coef * coef))
        feats[f"wp_{node.path}_energy"] = e
        energies.append(e)
    energies_arr = np.asarray(energies, dtype=np.float64)
    total = float(energies_arr.sum()) + 1e-12
    # Normalized-energy summary (position-invariant spectral shape).
    p = energies_arr / total
    feats["wp_energy_total"] = total
    feats["wp_energy_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))
    feats["wp_energy_max_frac"] = float(p.max())
    feats["wp_energy_top5_frac"] = float(np.sort(p)[-5:].sum())
    feats["wp_energy_std"] = float(energies_arr.std())
    return feats


# ---------------------------------------------------------------------------
# 5. Hurst exponent + DFA + trend CV
# ---------------------------------------------------------------------------

def hurst_dfa_features(h: np.ndarray) -> dict[str, float]:
    """Long-range correlations: R/S Hurst and DFA slope on row+col signals.

    We concatenate averaged row/col profiles and a flattened 1D traversal.
    """
    row_mean = h.mean(axis=1)
    col_mean = h.mean(axis=0)
    # Zig-zag scan -> long 1D series to capture global correlations.
    flat = h.ravel().astype(np.float64)

    feats: dict[str, float] = {}
    for name, series in (("row", row_mean), ("col", col_mean), ("flat", flat)):
        H = _hurst_rs(series)
        dfa = _dfa_slope(series)
        feats[f"hurst_{name}"] = H
        feats[f"dfa_{name}"] = dfa
    # CV of row trends (how much each row tilts) — proxy for scanner-like drift.
    row_trends = []
    xs = np.arange(h.shape[1])
    for row in h:
        slope, _ = np.polyfit(xs, row, 1)
        row_trends.append(slope)
    row_trends = np.asarray(row_trends)
    m = float(np.abs(row_trends).mean())
    feats["trend_row_cv"] = float(row_trends.std() / (m + 1e-9))
    feats["trend_row_abs_mean"] = m
    return feats


def _hurst_rs(x: np.ndarray) -> float:
    """Rescaled-range (R/S) Hurst estimate."""
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if N < 16:
        return 0.5
    # Use dyadic window sizes.
    lags = []
    rs = []
    n = 8
    while n <= N // 2:
        chunks = N // n
        segs = x[:chunks * n].reshape(chunks, n)
        seg_means = segs.mean(axis=1, keepdims=True)
        y = segs - seg_means
        Z = np.cumsum(y, axis=1)
        R = Z.max(axis=1) - Z.min(axis=1)
        S = segs.std(axis=1, ddof=0)
        valid = S > 1e-12
        if valid.sum() > 0:
            lags.append(n)
            rs.append(float((R[valid] / S[valid]).mean()))
        n *= 2
    if len(lags) < 2:
        return 0.5
    slope, _ = np.polyfit(np.log(lags), np.log(np.asarray(rs) + 1e-12), 1)
    return float(slope)


def _dfa_slope(x: np.ndarray) -> float:
    """Detrended-fluctuation-analysis exponent."""
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if N < 32:
        return 0.5
    y = np.cumsum(x - x.mean())
    # Window sizes (dyadic).
    F = []
    lags = []
    n = 8
    while n <= N // 4:
        segs = y[: (N // n) * n].reshape(-1, n)
        xs = np.arange(n)
        # Fit+detrend per segment (vectorized over segments).
        A = np.vstack([xs, np.ones_like(xs)]).T
        coef, *_ = np.linalg.lstsq(A, segs.T, rcond=None)
        fit = (A @ coef).T  # (n_segs, n)
        rms = np.sqrt(((segs - fit) ** 2).mean(axis=1))
        F.append(float(rms.mean()))
        lags.append(n)
        n *= 2
    if len(lags) < 2:
        return 0.5
    slope, _ = np.polyfit(np.log(lags), np.log(np.asarray(F) + 1e-12), 1)
    return float(slope)


# ---------------------------------------------------------------------------
# 6. Extended GLCM (many distances x angles x 6 props)
# ---------------------------------------------------------------------------

def extended_glcm_features(
    h: np.ndarray,
    levels: int = 32,
    distances: tuple[int, ...] = (1, 2, 4, 8, 12, 16, 24, 32),
    n_angles: int = 8,
) -> dict[str, float]:
    """Coarse-grained summary of a much larger GLCM sweep.

    Full grid would be 8 dist x 8 angle x 6 prop = 384 raw values.
    We keep per-distance mean+std across angles for each property -> 8x6x2 = 96,
    then reduce further with global stats -> ~50 features per spec.
    """
    img = (h * (levels - 1)).astype(np.uint8)
    angles = tuple(np.pi * k / n_angles for k in range(n_angles))
    glcm = graycomatrix(
        img,
        distances=list(distances),
        angles=list(angles),
        levels=levels,
        symmetric=True,
        normed=True,
    )
    feats: dict[str, float] = {}
    for prop in ("contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"):
        vals = graycoprops(glcm, prop=prop)  # shape (n_dist, n_ang)
        # Per-distance angle-averaged -> n_dist features.
        per_d_mean = vals.mean(axis=1)
        per_d_std = vals.std(axis=1)
        for di, d in enumerate(distances):
            feats[f"xglcm_{prop}_d{d}"] = float(per_d_mean[di])
        # Global rollups.
        feats[f"xglcm_{prop}_all_mean"] = float(vals.mean())
        feats[f"xglcm_{prop}_all_std"] = float(vals.std())
        feats[f"xglcm_{prop}_aniso"] = float(per_d_std.mean())
        # Slope vs distance (how fast correlation decays with scale).
        if len(distances) >= 2:
            slope, _ = np.polyfit(np.log(list(distances)), per_d_mean, 1)
            feats[f"xglcm_{prop}_slope"] = float(slope)
    return feats


# ---------------------------------------------------------------------------
# 7. Gabor filter bank responses
# ---------------------------------------------------------------------------

def gabor_features(
    h: np.ndarray,
    n_orient: int = 8,
    frequencies: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4),
) -> dict[str, float]:
    """Gabor bank: n_orient orientations x len(frequencies) frequencies.

    For each filter we compute mean, std, energy, entropy of magnitude response
    -> 4 stats. Plus bank-level anisotropy / scale-sensitivity summaries.
    """
    feats: dict[str, float] = {}
    mags_by_freq: dict[float, list[float]] = {f: [] for f in frequencies}
    # Pre-mean-subtract so DC doesn't dominate low-frequency responses.
    hc = h - float(h.mean())
    for fi, f in enumerate(frequencies):
        # Cap bandwidth so low-frequency kernels don't blow up in size.
        # sigma_x ~ 1/f with bandwidth=1 gives typical kernel ~ 5 sigma.
        sigma_x = min(1.0 / (f * 2.0 * np.pi) * 1.5, 6.0)
        sigma_y = sigma_x
        for oi in range(n_orient):
            theta = np.pi * oi / n_orient
            kernel = gabor_kernel(
                frequency=f, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y
            )
            kr = np.real(kernel).astype(np.float32)
            ki = np.imag(kernel).astype(np.float32)
            # fftconvolve is O(N log N) vs O(N*K^2) for large kernels.
            if kr.size > 361:   # > 19x19 -> use FFT
                r = fftconvolve(hc, kr, mode="same")
                i = fftconvolve(hc, ki, mode="same")
            else:
                r = ndimage.convolve(hc, kr, mode="reflect")
                i = ndimage.convolve(hc, ki, mode="reflect")
            mag = np.sqrt(r * r + i * i)
            m = float(mag.mean())
            s = float(mag.std())
            e = float((mag * mag).mean())
            p = mag / (mag.sum() + 1e-12)
            ent = float(-(p * np.log(p + 1e-12)).sum())
            pref = f"gabor_f{fi}_o{oi}"
            feats[f"{pref}_mean"] = m
            feats[f"{pref}_std"] = s
            feats[f"{pref}_energy"] = e
            feats[f"{pref}_entropy"] = ent
            mags_by_freq[f].append(m)
    # Per-frequency aniso + tuning sharpness.
    for fi, f in enumerate(frequencies):
        vals = np.asarray(mags_by_freq[f])
        feats[f"gabor_f{fi}_aniso"] = float(vals.std() / (vals.mean() + 1e-9))
        feats[f"gabor_f{fi}_max_orient"] = float(np.argmax(vals) * np.pi / n_orient)
    return feats


# ---------------------------------------------------------------------------
# 8. Multi-scale HOG
# ---------------------------------------------------------------------------

def multiscale_hog_features(
    h: np.ndarray,
    cell_sizes: tuple[int, ...] = (8, 16, 32),
    orientations: int = 9,
) -> dict[str, float]:
    """HOG at multiple cell sizes -> histogram statistics per scale."""
    img = (h * 255).astype(np.uint8)
    feats: dict[str, float] = {}
    for cs in cell_sizes:
        try:
            desc = hog(
                img,
                orientations=orientations,
                pixels_per_cell=(cs, cs),
                cells_per_block=(2, 2),
                feature_vector=True,
                block_norm="L2-Hys",
            )
        except Exception:
            desc = np.zeros(1, dtype=np.float32)
        p10, p25, p50, p75, p90 = np.percentile(desc, [10, 25, 50, 75, 90])
        feats[f"mhog_c{cs}_mean"] = float(desc.mean())
        feats[f"mhog_c{cs}_std"] = float(desc.std())
        feats[f"mhog_c{cs}_min"] = float(desc.min())
        feats[f"mhog_c{cs}_max"] = float(desc.max())
        feats[f"mhog_c{cs}_p10"] = float(p10)
        feats[f"mhog_c{cs}_p25"] = float(p25)
        feats[f"mhog_c{cs}_p50"] = float(p50)
        feats[f"mhog_c{cs}_p75"] = float(p75)
        feats[f"mhog_c{cs}_p90"] = float(p90)
        # Entropy of the (renormalized) descriptor -> orientation dispersion.
        p = desc / (desc.sum() + 1e-12)
        feats[f"mhog_c{cs}_entropy"] = float(-(p * np.log(p + 1e-12)).sum())
    return feats


# ---------------------------------------------------------------------------
# Combined advanced extractor
# ---------------------------------------------------------------------------

def extract_all_advanced_features(h: np.ndarray) -> dict[str, float]:
    """Extract every advanced feature bundle. Expects float32 in [0, 1]."""
    feats: dict[str, float] = {}
    feats.update(multifractal_features(h))
    feats.update(lacunarity_features(h))
    feats.update(succolarity_features(h))
    feats.update(wavelet_packet_features(h))
    feats.update(hurst_dfa_features(h))
    feats.update(extended_glcm_features(h))
    feats.update(gabor_features(h))
    feats.update(multiscale_hog_features(h))
    # Final safety: scrub any non-finite (log-of-zero, 0/0, etc.) -> 0.
    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0
    return feats


def advanced_feature_names() -> list[str]:
    rng = np.random.default_rng(0)
    dummy = rng.random((128, 128)).astype(np.float32)
    return list(extract_all_advanced_features(dummy).keys())
