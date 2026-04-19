"""Biomarker fingerprint per class — handcrafted-feature attribution.

For each of the 5 classes, compute the mean profile of handcrafted AFM
features and compare to the population mean. Rank features by effect size
(Cohen's d = (mu_class - mu_pop) / sigma_pop). Display the top discriminative
features as a grouped bar chart — one "fingerprint" row per class.

Selected clinical-interpretable features (roughness / texture / fractal):
    Sa, Sq, Ssk, Sku
    fractal_D_mean, fractal_D_std
    glcm_contrast_d1_mean, glcm_homogeneity_d1_mean,
    glcm_energy_d1_mean, glcm_correlation_d1_mean,
    glcm_dissimilarity_d1_mean
    iqr (p95 - p5 spread)

Outputs:
    reports/pitch/09_biomarker_fingerprint.png
    reports/pitch/09_biomarker_table.csv  (per-class z-scores for the 12 biomarkers)

Usage:
    python -m scripts.biomarker_attribution
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


CLASSES = ["ZdraviLudia", "Diabetes", "PGOV_Glaukom", "SklerozaMultiplex", "SucheOko"]
DISPLAY = {
    "ZdraviLudia": "Healthy",
    "Diabetes": "Diabetes",
    "PGOV_Glaukom": "Glaucoma",
    "SklerozaMultiplex": "Mult. Sclerosis",
    "SucheOko": "Dry Eye",
}
CLASS_COLOR = {
    "ZdraviLudia": "#2ca02c",
    "Diabetes": "#d62728",
    "PGOV_Glaukom": "#9467bd",
    "SklerozaMultiplex": "#1f77b4",
    "SucheOko": "#ff7f0e",
}

# Curated interpretable feature set (12 features, clinical meaning)
BIOMARKERS = [
    ("Sa",                            "Roughness Sa (avg absolute deviation)"),
    ("Sq",                            "Roughness Sq (RMS height)"),
    ("Ssk",                           "Skewness (peak/valley asymmetry)"),
    ("Sku",                           "Kurtosis (tail / spike-iness)"),
    ("iqr",                           "IQR p95-p5 (height dynamic range)"),
    ("fractal_D_mean",                "Fractal dim (ferning complexity)"),
    ("fractal_D_std",                 "Fractal dim std"),
    ("glcm_contrast_d1_mean",         "GLCM contrast (local variation)"),
    ("glcm_homogeneity_d1_mean",      "GLCM homogeneity"),
    ("glcm_energy_d1_mean",           "GLCM energy (uniformity)"),
    ("glcm_correlation_d1_mean",      "GLCM correlation (directionality)"),
    ("glcm_dissimilarity_d1_mean",    "GLCM dissimilarity"),
]


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(_ROOT / "cache" / "features_handcrafted.parquet")
    return df


def compute_fingerprints(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return (n_classes, n_features) DataFrame of z-scores (class mean vs pop)."""
    pop_mean = df[feature_cols].mean()
    pop_std = df[feature_cols].std().replace(0, 1.0)
    rows = []
    for cls in CLASSES:
        sub = df[df.cls == cls]
        z = (sub[feature_cols].mean() - pop_mean) / pop_std
        rows.append(z.rename(cls))
    return pd.concat(rows, axis=1).T  # rows=class, cols=feature


def top_k_per_class(z: pd.DataFrame, k: int = 5) -> dict[str, list[tuple[str, float]]]:
    """Return top-k features per class ranked by |z|."""
    out = {}
    for cls in CLASSES:
        zc = z.loc[cls].abs().sort_values(ascending=False).head(k)
        out[cls] = [(feat, float(z.loc[cls, feat])) for feat in zc.index]
    return out


def make_fingerprint_figure(z: pd.DataFrame, out_path: Path,
                            feature_cols: list[str]):
    """Grouped bar chart — one row per class, x=feature, y=z-score."""
    n_features = len(feature_cols)
    short_labels = [lbl for _, lbl in BIOMARKERS]
    fig, axes = plt.subplots(len(CLASSES), 1, figsize=(13, 2.0 * len(CLASSES)), sharex=True)

    vmax = float(np.abs(z.values).max()) * 1.05

    for r, cls in enumerate(CLASSES):
        ax = axes[r]
        vals = z.loc[cls, feature_cols].values
        colors = [CLASS_COLOR[cls] if v >= 0 else "#333333" for v in vals]
        bars = ax.bar(range(n_features), vals, color=colors, edgecolor="black",
                      linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylim(-vmax, vmax)
        ax.set_ylabel(f"{DISPLAY[cls]}\n(z-score)", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        # Annotate top-3 |z| per class
        topk = np.argsort(np.abs(vals))[::-1][:3]
        for idx in topk:
            ax.text(idx, vals[idx] + (0.15 if vals[idx] >= 0 else -0.3),
                    f"{vals[idx]:+.1f}σ", ha="center", fontsize=7.5, fontweight="bold")
        # Subtle frame
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xticks(range(n_features))
    axes[-1].set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8)
    fig.suptitle(
        "Biomarker fingerprint per class — handcrafted AFM features\n"
        "Bars show how each class's mean deviates from the population mean "
        "(Cohen's-d-like z-score, σ = population std).\n"
        "Positive = higher than average; negative = lower. Top-3 |σ| annotated per class.",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[saved] {out_path}")


def main():
    df = load_features()
    print(f"Loaded {len(df)} scans across {df.cls.nunique()} classes.")
    feature_cols = [name for name, _ in BIOMARKERS]

    z = compute_fingerprints(df, feature_cols)
    print("\nZ-score table (class mean vs population mean):")
    print(z.round(2).to_string())

    top = top_k_per_class(z, k=5)
    print("\nTop-5 discriminative features per class (|z|):")
    for cls, lst in top.items():
        print(f"  {DISPLAY[cls]:16s}: " + ", ".join(
            f"{f}({v:+.2f})" for f, v in lst))

    out_fig = _ROOT / "reports" / "pitch" / "09_biomarker_fingerprint.png"
    make_fingerprint_figure(z, out_fig, feature_cols)

    out_csv = _ROOT / "reports" / "pitch" / "09_biomarker_table.csv"
    z.to_csv(out_csv, float_format="%.4f")
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()
