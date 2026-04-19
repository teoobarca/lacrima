"""Generate presentation-quality figures for the Hack Košice 2026 pitch.

Outputs (all PNG, 150 dpi, tight layout) to reports/pitch/:
  01_class_distribution.png
  02_class_morphology_grid.png
  03_umap_embedding.png
  04_confusion_matrix.png
  05_per_class_metrics.png
  06_morphology_comparison.png
  INDEX.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, Normalize

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path("/Users/rafael/Programming/teardrop-challenge")
sys.path.insert(0, str(ROOT))

from teardrop.data import (  # noqa: E402
    enumerate_samples,
    preprocess_spm,
    load_height,
    CLASSES,
    CLASS_TO_IDX,
)
from teardrop.cv import leave_one_patient_out  # noqa: E402

OUT = ROOT / "reports" / "pitch"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

sns.set_theme(style="darkgrid", context="talk")
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "font.size": 12,
})

# Pretty class display names
PRETTY = {
    "ZdraviLudia": "Healthy",
    "Diabetes": "Diabetes",
    "PGOV_Glaukom": "Glaucoma",
    "SklerozaMultiplex": "Multiple\nSclerosis",
    "SucheOko": "Dry Eye",
}
PRETTY_FLAT = {k: v.replace("\n", " ") for k, v in PRETTY.items()}

# Distinct colors for the 5 classes (consistent across figures)
CLASS_COLORS = {
    "ZdraviLudia": "#2ecc71",       # green - healthy
    "Diabetes": "#f39c12",          # orange
    "PGOV_Glaukom": "#3498db",      # blue
    "SklerozaMultiplex": "#9b59b6", # purple
    "SucheOko": "#e74c3c",          # red
}


def banner(s):
    print(f"\n{'='*60}\n{s}\n{'='*60}")


# ---------------------------------------------------------------------------
# Figure 1: Class distribution
# ---------------------------------------------------------------------------

def fig01_class_distribution(samples):
    banner("Figure 1: class distribution")
    cls_counts = Counter(s.cls for s in samples)
    pat_per_cls = {}
    for s in samples:
        pat_per_cls.setdefault(s.cls, set()).add(s.patient)
    pat_counts = {c: len(p) for c, p in pat_per_cls.items()}

    order = CLASSES
    scans = [cls_counts[c] for c in order]
    pats = [pat_counts[c] for c in order]
    labels = [PRETTY[c] for c in order]
    colors = [CLASS_COLORS[c] for c in order]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: scans per class
    ax = axes[0]
    bars = ax.bar(labels, scans, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, scans):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, str(v),
                ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_title("Scans per class", fontsize=15)
    ax.set_ylabel("Number of scans")
    ax.set_ylim(0, max(scans) * 1.15)

    # Right: patients per class with red highlight on SucheOko
    ax = axes[1]
    pat_colors = []
    for c in order:
        if c == "SucheOko":
            pat_colors.append("#e74c3c")
        else:
            pat_colors.append(CLASS_COLORS[c])
    bars = ax.bar(labels, pats, color=pat_colors, edgecolor="black", linewidth=0.6)
    for b, v, c in zip(bars, pats, order):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, str(v),
                ha="center", va="bottom", fontsize=13, fontweight="bold",
                color="#c0392b" if c == "SucheOko" else "black")
    ax.set_title("Unique patients per class", fontsize=15)
    ax.set_ylabel("Number of patients")
    ax.set_ylim(0, max(pats) * 1.2)
    # callout for SucheOko
    ax.annotate(
        "Dry Eye: only 2 patients\n(fundamental data limit)",
        xy=(4, pats[4] + 0.3),
        xytext=(1.8, max(pats) * 0.7),
        fontsize=12, color="#c0392b", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.8),
        bbox=dict(boxstyle="round,pad=0.4", fc="#fdecea", ec="#e74c3c", lw=1.2),
    )

    fig.suptitle("Dataset class distribution (n=240 scans, 44 patients)",
                 fontsize=17, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "01_class_distribution.png")
    plt.close(fig)
    print("  -> saved 01_class_distribution.png")


# ---------------------------------------------------------------------------
# Figure 2: 5-class morphology grid (3 samples each)
# ---------------------------------------------------------------------------

def fig02_class_morphology_grid(samples):
    banner("Figure 2: morphology grid")
    rng = np.random.default_rng(42)

    # Group samples by class
    by_cls = {c: [] for c in CLASSES}
    for s in samples:
        by_cls[s.cls].append(s)

    n_per = 3
    chosen = {}
    for c in CLASSES:
        idx = rng.choice(len(by_cls[c]), size=min(n_per, len(by_cls[c])), replace=False)
        chosen[c] = [by_cls[c][i] for i in idx]

    fig, axes = plt.subplots(n_per, len(CLASSES), figsize=(20, 12))
    for col, c in enumerate(CLASSES):
        for row in range(n_per):
            ax = axes[row, col]
            s = chosen[c][row] if row < len(chosen[c]) else None
            if s is None:
                ax.axis("off")
                continue
            try:
                img = preprocess_spm(s.raw_path)
                hm = load_height(s.raw_path)
                H, W = hm.height.shape
                um = H * hm.pixel_nm / 1000.0
                ax.imshow(img, cmap="afmhot")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(PRETTY_FLAT[c],
                                 fontsize=18, fontweight="bold",
                                 color=CLASS_COLORS[c], pad=12)
                ax.set_xlabel(f"{um:.0f} \u00b5m", fontsize=10)
            except Exception as e:
                print(f"  fail {s.raw_path.name}: {e}")
                ax.axis("off")

    fig.suptitle("AFM morphology — three samples per class (afmhot, plane-leveled)",
                 fontsize=18, fontweight="bold", y=1.0)
    plt.tight_layout()
    fig.savefig(OUT / "02_class_morphology_grid.png")
    plt.close(fig)
    print("  -> saved 02_class_morphology_grid.png")


# ---------------------------------------------------------------------------
# Figure 3: UMAP of DINOv2 scan embeddings
# ---------------------------------------------------------------------------

def load_scan_embeddings():
    """Mean-pool tile embeddings per scan."""
    cache = ROOT / "cache" / "tiled_emb_dinov2_vits14_afmhot_t512_n9.npz"
    d = np.load(cache, allow_pickle=True)
    X_tiles = d["X"]                    # (n_tiles, 384)
    tile_to_scan = d["tile_to_scan"]    # (n_tiles,)
    scan_y = d["scan_y"]                # (n_scans,)
    scan_groups = d["scan_groups"]      # (n_scans,) patient ids
    scan_paths = d["scan_paths"]        # (n_scans,)

    n_scans = len(scan_y)
    dim = X_tiles.shape[1]
    X_scan = np.zeros((n_scans, dim), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int64)
    for i, s in enumerate(tile_to_scan):
        X_scan[s] += X_tiles[i]
        counts[s] += 1
    X_scan /= np.maximum(counts[:, None], 1)
    return X_scan, scan_y, scan_groups, scan_paths


def fig03_umap_embedding():
    banner("Figure 3: UMAP")
    X, y, groups, paths = load_scan_embeddings()
    print(f"  scan-level X: {X.shape}, classes: {np.unique(y)}, patients: {len(np.unique(groups))}")

    Xs = StandardScaler().fit_transform(X)

    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        Z = reducer.fit_transform(Xs)
        proj_name = "UMAP"
    except Exception as e:
        print(f"  UMAP failed ({e}), falling back to PCA")
        from sklearn.decomposition import PCA
        Z = PCA(n_components=2, random_state=42).fit_transform(Xs)
        proj_name = "PCA"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: by class
    ax = axes[0]
    for cls_name, cls_idx in CLASS_TO_IDX.items():
        m = y == cls_idx
        ax.scatter(Z[m, 0], Z[m, 1],
                   color=CLASS_COLORS[cls_name],
                   label=PRETTY_FLAT[cls_name],
                   s=70, alpha=0.78, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Coloured by CLASS", fontsize=15)
    ax.set_xlabel(f"{proj_name} 1")
    ax.set_ylabel(f"{proj_name} 2")
    ax.legend(loc="best", frameon=True, fontsize=10)

    # Right: by patient
    ax = axes[1]
    unique_pat = sorted(set(groups.tolist()))
    n_pat = len(unique_pat)
    cmap = mpl.colormaps.get_cmap("tab20").resampled(max(20, n_pat))
    pat_to_color = {p: cmap(i % cmap.N) for i, p in enumerate(unique_pat)}
    for p in unique_pat:
        m = groups == p
        ax.scatter(Z[m, 0], Z[m, 1],
                   color=pat_to_color[p],
                   label=p if n_pat < 30 else None,
                   s=70, alpha=0.78, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Coloured by PATIENT (n={n_pat})", fontsize=15)
    ax.set_xlabel(f"{proj_name} 1")
    ax.set_ylabel(f"{proj_name} 2")
    if n_pat < 30:
        ax.legend(loc="best", fontsize=8, ncol=2)

    fig.suptitle(f"DINOv2 embedding space — class structure vs. patient structure ({proj_name})",
                 fontsize=17, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "03_umap_embedding.png")
    plt.close(fig)
    print("  -> saved 03_umap_embedding.png")


# ---------------------------------------------------------------------------
# Figures 4 & 5: LOPO confusion matrix + per-class metrics
# ---------------------------------------------------------------------------

def lopo_predictions():
    """Run LOPO on scan-level DINOv2-S embeddings with LogisticRegression."""
    X, y, groups, paths = load_scan_embeddings()
    n = len(y)
    y_pred = np.full(n, -1, dtype=np.int64)
    y_proba = np.zeros((n, len(CLASSES)), dtype=np.float64)

    for tr, va in leave_one_patient_out(groups):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xva = scaler.transform(X[va])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000,
                                  solver="lbfgs")
        clf.fit(Xtr, y[tr])
        # Map class -> column
        col_to_cls = {i: c for i, c in enumerate(clf.classes_)}
        proba = clf.predict_proba(Xva)
        for col, c in col_to_cls.items():
            y_proba[va, c] = proba[:, col]
        y_pred[va] = clf.predict(Xva)
    return y, y_pred, y_proba


def fig04_confusion_matrix(y_true, y_pred):
    banner("Figure 4: confusion matrix")
    labels = list(range(len(CLASSES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    n = len(CLASSES)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.set_facecolor("white")

    # Build per-cell colors: green diag, red off-diag scaled by value
    img = np.zeros((n, n, 4))
    green = np.array([46, 204, 113]) / 255.0
    red = np.array([231, 76, 60]) / 255.0
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if i == j:
                # green intensity proportional to recall
                alpha = 0.25 + 0.75 * v
                img[i, j, :3] = green
                img[i, j, 3] = alpha
            else:
                alpha = 0.05 + 0.85 * v
                img[i, j, :3] = red
                img[i, j, 3] = alpha
    ax.imshow(img, aspect="equal")

    # Annotate
    for i in range(n):
        for j in range(n):
            pct = cm_norm[i, j] * 100
            count = cm[i, j]
            txt = f"{pct:.0f}%\n({count})"
            color = "black" if cm_norm[i, j] < 0.55 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=12, color=color, fontweight="bold")

    pretty = [PRETTY_FLAT[c] for c in CLASSES]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(pretty, rotation=30, ha="right")
    ax.set_yticklabels(pretty)
    ax.set_xlabel("Predicted class", fontsize=13, fontweight="bold")
    ax.set_ylabel("True class", fontsize=13, fontweight="bold")
    ax.grid(False)
    ax.set_title("LOPO confusion matrix — DINOv2-S linear probe\n(row-normalized; each row sums to 100%)",
                 fontsize=15, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(OUT / "04_confusion_matrix.png")
    plt.close(fig)
    print("  -> saved 04_confusion_matrix.png")


def fig05_per_class_metrics(y_true, y_pred):
    banner("Figure 5: per-class metrics")
    labels = list(range(len(CLASSES)))
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    pretty = [PRETTY_FLAT[c] for c in CLASSES]
    n = len(CLASSES)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 7))
    b1 = ax.bar(x - width, p, width, label="Precision", color="#3498db",
                edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x,         r, width, label="Recall",    color="#2ecc71",
                edgecolor="black", linewidth=0.5)
    b3 = ax.bar(x + width, f, width, label="F1",        color="#9b59b6",
                edgecolor="black", linewidth=0.5)

    # value labels
    for bars in (b1, b2, b3):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
    # support counts above each group
    ymax = max(max(p), max(r), max(f), 1.0) * 1.05
    for xi, s, c in zip(x, sup, CLASSES):
        ax.text(xi, ymax + 0.02, f"n={s}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=CLASS_COLORS[c])

    ax.axhline(weighted_f1, color="#c0392b", linestyle="--", linewidth=2,
               label=f"weighted-F1 = {weighted_f1:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(pretty, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_ylim(0, ymax + 0.18)
    ax.set_title(f"Per-class performance (LOPO eval, DINOv2-S linear probe)\n"
                 f"weighted-F1={weighted_f1:.3f}   macro-F1={macro_f1:.3f}",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    fig.savefig(OUT / "05_per_class_metrics.png")
    plt.close(fig)
    print(f"  -> saved 05_per_class_metrics.png  (weighted-F1={weighted_f1:.3f})")


# ---------------------------------------------------------------------------
# Figure 6: morphology comparison with biological annotations
# ---------------------------------------------------------------------------

ANNOTATIONS = {
    "ZdraviLudia":      "Dense dendritic ferning network\nMasmali grade I",
    "Diabetes":         "Thickened branches, dense packing\nhigh osmolarity",
    "PGOV_Glaukom":     "Granular structure\nMMP-9 mediated matrix degradation",
    "SklerozaMultiplex":"Heterogeneous crystallization\nprotein/lipid alteration",
    "SucheOko":         "Fragmented, sparse network\nMasmali grade III\u2013IV",
}


def pick_iconic(samples):
    """Pick the most 'iconic' sample per class.

    Heuristic: choose the sample whose preprocessed image has the highest
    standard deviation (= most contrast / structure). To stay fast we evaluate
    at most 6 candidates per class, picked deterministically with seed=42.
    """
    rng = np.random.default_rng(42)
    by_cls = {c: [] for c in CLASSES}
    for s in samples:
        by_cls[s.cls].append(s)
    chosen = {}
    for c in CLASSES:
        cands = by_cls[c]
        idx = rng.choice(len(cands), size=min(6, len(cands)), replace=False)
        best = None
        best_score = -1.0
        for i in idx:
            try:
                img = preprocess_spm(cands[i].raw_path)
                score = float(img.std())
                if score > best_score:
                    best_score = score
                    best = (cands[i], img)
            except Exception as e:
                print(f"  skip {cands[i].raw_path.name}: {e}")
        if best is None:
            # fallback: first that loads
            for s in cands:
                try:
                    img = preprocess_spm(s.raw_path)
                    best = (s, img)
                    break
                except Exception:
                    continue
        chosen[c] = best
    return chosen


def fig06_morphology_comparison(samples):
    banner("Figure 6: iconic morphology comparison")
    chosen = pick_iconic(samples)
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(25, 7))
    for ax, c in zip(axes, CLASSES):
        s, img = chosen[c]
        ax.imshow(img, cmap="afmhot")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(PRETTY_FLAT[c], fontsize=20, fontweight="bold",
                     color=CLASS_COLORS[c], pad=10)
        ax.set_xlabel(ANNOTATIONS[c], fontsize=12, labelpad=10)
        # add scan_um info below
        try:
            hm = load_height(s.raw_path)
            H = hm.height.shape[0]
            um = H * hm.pixel_nm / 1000.0
            ax.text(0.98, 0.02, f"{um:.0f} \u00b5m",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=11, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="white", alpha=0.55))
        except Exception:
            pass

    fig.suptitle("Tear-ferning morphology by disease class",
                 fontsize=20, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "06_morphology_comparison.png")
    plt.close(fig)
    print("  -> saved 06_morphology_comparison.png")


# ---------------------------------------------------------------------------
# INDEX.md
# ---------------------------------------------------------------------------

INDEX_BODY = """# Pitch figures — Hack Košice 2026

| File | Description |
|------|-------------|
| `01_class_distribution.png` | Bar charts: scans per class + unique patients per class (highlighting SucheOko's 2-patient limit). |
| `02_class_morphology_grid.png` | 5×3 grid of preprocessed AFM height maps (afmhot), three samples per class, deterministic seed=42. |
| `03_umap_embedding.png` | UMAP of mean-pooled DINOv2-S scan embeddings, coloured by class (left) and by patient (right). |
| `04_confusion_matrix.png` | Row-normalized LOPO confusion matrix for the DINOv2-S linear probe; diagonal cells in green, errors in red. |
| `05_per_class_metrics.png` | Per-class precision / recall / F1 from the LOPO probe; support counts above each group, weighted-F1 baseline as red dashed line. |
| `06_morphology_comparison.png` | Iconic single-scan morphology per class with biological / Masmali annotations for the pitch slide. |

All figures rendered at 150 dpi with seaborn `darkgrid` theme.
"""


def write_index():
    (OUT / "INDEX.md").write_text(INDEX_BODY)
    print("  -> wrote INDEX.md")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    print(f"loaded {len(samples)} samples")

    # Run figures; isolate failures so a single broken one doesn't kill the run
    steps = [
        ("fig01", lambda: fig01_class_distribution(samples)),
        ("fig02", lambda: fig02_class_morphology_grid(samples)),
        ("fig03", lambda: fig03_umap_embedding()),
    ]

    # 4 & 5 share the LOPO predictions -> compute once
    def run_lopo():
        y_true, y_pred, _ = lopo_predictions()
        fig04_confusion_matrix(y_true, y_pred)
        fig05_per_class_metrics(y_true, y_pred)

    steps.append(("fig04+05", run_lopo))
    steps.append(("fig06", lambda: fig06_morphology_comparison(samples)))

    for name, fn in steps:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"!! {name} FAILED: {e}")
            traceback.print_exc()

    write_index()
    print(f"\nTotal elapsed: {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
