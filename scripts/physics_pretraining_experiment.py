"""Physics-informed pretext experiment for tear ferning classification.

Pipeline
========
1. Cahn-Hilliard spinodal decomposition simulations with class-specific
   parameters (see teardrop.physics_sim).
2. Encode simulated + real AFM scans with the same DINOv2-B pipeline
   (afmhot RGB render at 512×512).
3. UMAP projection: do simulated textures cluster with their intended
   class centroids?
4. (Optional) augmentation experiment: add the simulated embeddings to the
   LOPO training folds and see whether weighted-F1 improves.

Outputs
-------
- reports/pitch/10_physics_simulation.png (real vs sim side-by-side)
- reports/PHYSICS_SIMULATION_RESULTS.md
- cache/physics_sim_dinov2b_embeddings.npz
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

ROOT = Path("/Users/rafael/Programming/teardrop-challenge")
sys.path.insert(0, str(ROOT))

from teardrop.physics_sim import (  # noqa: E402
    CLASS_PRESETS,
    simulate,
    sample_params,
    field_to_height,
)
from teardrop.data import CLASSES  # noqa: E402

OUT_PITCH = ROOT / "reports" / "pitch"
OUT_REPORT = ROOT / "reports" / "PHYSICS_SIMULATION_RESULTS.md"
CACHE = ROOT / "cache" / "physics_sim_dinov2b_embeddings.npz"
OUT_PITCH.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Style (matches other pitch figures)
# ---------------------------------------------------------------------------

sns.set_theme(style="darkgrid", context="talk")
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

PRETTY = {
    "ZdraviLudia": "Healthy",
    "Diabetes": "Diabetes",
    "PGOV_Glaukom": "Glaucoma",
    "SklerozaMultiplex": "Multiple Sclerosis",
    "SucheOko": "Dry Eye",
}
CLASS_COLORS = {
    "ZdraviLudia": "#2ecc71",
    "Diabetes": "#f39c12",
    "PGOV_Glaukom": "#3498db",
    "SklerozaMultiplex": "#9b59b6",
    "SucheOko": "#e74c3c",
}


def banner(s):
    print(f"\n{'='*60}\n{s}\n{'='*60}")


# ---------------------------------------------------------------------------
# Step 1: generate simulation dataset
# ---------------------------------------------------------------------------

def generate_simulations(n_per_class: int = 30, master_seed: int = 42):
    banner(f"Step 1: Cahn-Hilliard simulations ({n_per_class} per class)")
    heights: list[np.ndarray] = []
    y: list[int] = []
    cls_names: list[str] = []
    sim_idx: list[int] = []

    for cls_idx, cls in enumerate(CLASSES):
        base = CLASS_PRESETS[cls]
        rng = np.random.default_rng(master_seed + cls_idx * 997)
        t0 = time.perf_counter()
        for i in range(n_per_class):
            p = sample_params(base, rng, jitter=0.15)
            u = simulate(p)
            h = field_to_height(u)
            heights.append(h)
            y.append(cls_idx)
            cls_names.append(cls)
            sim_idx.append(i)
        dt = time.perf_counter() - t0
        print(f"  {cls:22s} {n_per_class} sims in {dt:.1f}s ({dt / n_per_class:.2f}s/sim)")

    return heights, np.array(y, dtype=np.int64), cls_names, sim_idx


# ---------------------------------------------------------------------------
# Step 2: encode with DINOv2-B (afmhot RGB render)
# ---------------------------------------------------------------------------

def heights_to_pil(heights: list[np.ndarray], size: int = 512) -> list[Image.Image]:
    import matplotlib.cm as cm
    out = []
    for h in heights:
        rgba = (cm.afmhot(h) * 255).astype(np.uint8)[..., :3]
        im = Image.fromarray(rgba, mode="RGB").resize(
            (size, size), Image.Resampling.BICUBIC
        )
        out.append(im)
    return out


def encode_simulations(heights: list[np.ndarray]) -> np.ndarray:
    banner("Step 2: encode simulations with DINOv2-B")
    from teardrop.encoders import load_dinov2
    enc = load_dinov2("vitb14")
    print(f"  device: {enc.device}")
    pils = heights_to_pil(heights, size=512)
    t0 = time.perf_counter()
    X = enc.encode(pils, batch_size=16)
    print(f"  encoded {len(pils)} images in {time.perf_counter() - t0:.1f}s  shape={X.shape}")
    return X


# ---------------------------------------------------------------------------
# Step 3: load real embeddings, joint UMAP
# ---------------------------------------------------------------------------

def load_real_embeddings():
    """Load cached DINOv2-B scan-level embeddings used by the main pipeline.

    We use the tile-average variant if available (tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz),
    otherwise fall back to the scan-level emb file.
    """
    tiled = ROOT / "cache" / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz"
    scan = ROOT / "cache" / "emb_dinov2_vitb14_afmhot.npz"
    if tiled.exists():
        d = np.load(tiled, allow_pickle=True)
        X_tiles = d["X"]
        tile_to_scan = d["tile_to_scan"]
        scan_y = d["scan_y"]
        scan_groups = d["scan_groups"]
        scan_paths = d["scan_paths"]
        n_scans = len(scan_y)
        X_scan = np.zeros((n_scans, X_tiles.shape[1]), dtype=np.float32)
        cnt = np.zeros(n_scans, dtype=np.int64)
        for i, s in enumerate(tile_to_scan):
            X_scan[s] += X_tiles[i]
            cnt[s] += 1
        X_scan /= np.maximum(cnt[:, None], 1)
        print(f"  loaded REAL tiled-avg embeddings: {X_scan.shape}")
        return X_scan, scan_y, scan_groups, scan_paths
    d = np.load(scan, allow_pickle=True)
    print(f"  loaded REAL scan embeddings: {d['X'].shape}")
    return d["X"], d["y"], d["groups"], d["paths"]


def joint_umap(X_real: np.ndarray, X_sim: np.ndarray):
    """Fit UMAP on combined real+sim, return 2D projections for each."""
    from sklearn.preprocessing import StandardScaler

    X_all = np.vstack([X_real, X_sim])
    Xs = StandardScaler().fit_transform(X_all)
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
    Z_real = Z[: len(X_real)]
    Z_sim = Z[len(X_real):]
    return Z_real, Z_sim, proj_name


# ---------------------------------------------------------------------------
# Step 4: cluster-agreement diagnostic
# ---------------------------------------------------------------------------

def cluster_agreement(X_real: np.ndarray, y_real: np.ndarray,
                       X_sim: np.ndarray, y_sim: np.ndarray) -> dict:
    """For each simulated sample, find the nearest real class centroid in
    embedding space. Report: (a) per-class top-1 agreement rate (nearest
    centroid == intended class), and (b) top-3 hit rate.
    """
    # Centroids of real classes in the *original* embedding space
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_real)
    Xr = scaler.transform(X_real)
    Xs = scaler.transform(X_sim)
    n_cls = len(CLASSES)
    centroids = np.zeros((n_cls, Xr.shape[1]), dtype=np.float64)
    for c in range(n_cls):
        m = y_real == c
        if m.sum() > 0:
            centroids[c] = Xr[m].mean(axis=0)
    # Cosine distance to each centroid
    from sklearn.preprocessing import normalize
    Cn = normalize(centroids, axis=1)
    Sn = normalize(Xs, axis=1)
    sims = Sn @ Cn.T  # (n_sim, n_cls)  higher = more similar
    ranks = np.argsort(-sims, axis=1)  # best class first

    per_class = {}
    for c in range(n_cls):
        mask = y_sim == c
        if mask.sum() == 0:
            continue
        top1 = (ranks[mask, 0] == c).mean()
        top3 = np.any(ranks[mask, :3] == c, axis=1).mean()
        top2 = np.any(ranks[mask, :2] == c, axis=1).mean()
        per_class[CLASSES[c]] = {
            "top1": float(top1),
            "top2": float(top2),
            "top3": float(top3),
            "n_sim": int(mask.sum()),
            "mean_sim_to_own_centroid": float(sims[mask, c].mean()),
        }
    overall_top1 = float((ranks[:, 0] == y_sim).mean())
    overall_top3 = float(np.any(ranks[:, :3] == y_sim[:, None], axis=1).mean())
    return {
        "overall_top1": overall_top1,
        "overall_top3": overall_top3,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Step 5: optional augmentation experiment
# ---------------------------------------------------------------------------

def augmentation_experiment(X_real, y_real, groups_real, X_sim, y_sim):
    """Compare LOPO weighted-F1 with and without simulated embeddings added
    to each training fold. SucheOko is excluded from val folds because it
    only has 2 patients — so we run full leave-one-patient-out.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    from teardrop.cv import leave_one_patient_out

    def run(augment: bool):
        n = len(y_real)
        y_pred = np.full(n, -1, dtype=np.int64)
        for tr, va in leave_one_patient_out(groups_real):
            Xtr = X_real[tr]
            ytr = y_real[tr]
            if augment:
                Xtr = np.vstack([Xtr, X_sim])
                ytr = np.concatenate([ytr, y_sim])
            scaler = StandardScaler().fit(Xtr)
            Xtr_s = scaler.transform(Xtr)
            Xva_s = scaler.transform(X_real[va])
            clf = LogisticRegression(class_weight="balanced", max_iter=2000,
                                      solver="lbfgs")
            clf.fit(Xtr_s, ytr)
            y_pred[va] = clf.predict(Xva_s)
        return f1_score(y_real, y_pred, average="weighted"), y_pred

    banner("Step 5: augmentation experiment")
    f1_base, _ = run(False)
    f1_aug, _ = run(True)
    print(f"  LOPO weighted-F1 (real only):       {f1_base:.3f}")
    print(f"  LOPO weighted-F1 (real + sim aug):  {f1_aug:.3f}")
    print(f"  delta:                              {f1_aug - f1_base:+.3f}")
    return {
        "f1_real_only": float(f1_base),
        "f1_real_plus_sim": float(f1_aug),
        "delta": float(f1_aug - f1_base),
    }


# ---------------------------------------------------------------------------
# Figure: real vs sim side-by-side + UMAP
# ---------------------------------------------------------------------------

def make_pitch_figure(
    heights_sim: list[np.ndarray],
    y_sim: np.ndarray,
    Z_real: np.ndarray,
    y_real: np.ndarray,
    Z_sim: np.ndarray,
    proj_name: str,
    cluster_stats: dict,
    real_scan_examples: dict[str, np.ndarray],
):
    banner("Figure: 10_physics_simulation.png")
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 2.2])

    # Top row: real AFM example per class
    for i, cls in enumerate(CLASSES):
        ax = fig.add_subplot(gs[0, i])
        h = real_scan_examples.get(cls)
        if h is not None:
            ax.imshow(h, cmap="afmhot", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"REAL — {PRETTY[cls]}", fontsize=12,
                     color=CLASS_COLORS[cls], fontweight="bold")

    # Bottom row: simulated example per class (pick first sim per class)
    for i, cls_name in enumerate(CLASSES):
        ax = fig.add_subplot(gs[1, i])
        cls_idx = CLASSES.index(cls_name)
        sel = np.where(y_sim == cls_idx)[0]
        if len(sel) > 0:
            ax.imshow(heights_sim[sel[0]], cmap="afmhot", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        top1 = cluster_stats["per_class"].get(cls_name, {}).get("top1", float("nan"))
        ax.set_title(
            f"SIM — {PRETTY[cls_name]}\nnearest-centroid top-1: {top1*100:.0f}%",
            fontsize=10, color=CLASS_COLORS[cls_name], fontweight="bold",
        )

    # Right column: joint UMAP
    ax = fig.add_subplot(gs[:, 5])
    # Real: filled circles
    for cls_name, cls_idx in zip(CLASSES, range(len(CLASSES))):
        m = y_real == cls_idx
        ax.scatter(
            Z_real[m, 0], Z_real[m, 1],
            s=55, alpha=0.85, edgecolor="white", linewidth=0.5,
            color=CLASS_COLORS[cls_name], label=f"Real {PRETTY[cls_name]}",
        )
    # Sim: hollow markers (different marker shape)
    for cls_name, cls_idx in zip(CLASSES, range(len(CLASSES))):
        m = y_sim == cls_idx
        ax.scatter(
            Z_sim[m, 0], Z_sim[m, 1],
            s=45, alpha=0.95,
            facecolor="none", edgecolor=CLASS_COLORS[cls_name], linewidth=1.6,
            marker="D", label=f"Sim {PRETTY[cls_name]}",
        )
    ax.set_title(
        f"Joint {proj_name} — real (filled) vs Cahn-Hilliard sim (hollow diamonds)\n"
        f"nearest-centroid top-1 overall: {cluster_stats['overall_top1']*100:.1f}%  "
        f"top-3: {cluster_stats['overall_top3']*100:.1f}%",
        fontsize=13,
    )
    ax.set_xlabel(f"{proj_name} 1"); ax.set_ylabel(f"{proj_name} 2")
    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.85)

    fig.suptitle(
        "Physics-informed pretext: Cahn-Hilliard spinodal decomposition vs real tear AFM",
        fontsize=17, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = OUT_PITCH / "10_physics_simulation.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> saved {out_path}")
    return out_path


def load_real_scan_examples() -> dict[str, np.ndarray]:
    """Load one preprocessed real scan per class for the figure."""
    from teardrop.data import enumerate_samples, preprocess_spm
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    out: dict[str, np.ndarray] = {}
    for cls in CLASSES:
        for s in samples:
            if s.cls == cls:
                try:
                    h = preprocess_spm(s.raw_path, target_nm_per_px=90.0, crop_size=512)
                    out[cls] = h
                    break
                except Exception as e:
                    print(f"  skip {s.raw_path.name}: {e}")
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    n_per_class: int,
    cluster_stats: dict,
    aug_stats: dict | None,
    compute_time_s: float,
):
    lines: list[str] = []
    lines.append("# Physics-informed simulation experiment\n")
    lines.append(f"_Cahn-Hilliard spinodal decomposition → DINOv2-B embedding space_\n")
    lines.append(f"Compute time: **{compute_time_s:.1f} s** total (simulation + encoding + UMAP)\n")
    lines.append("## Motivation\n")
    lines.append(
        "Tear ferning is classically modeled as a mix of spinodal decomposition "
        "(salt-water phase separation) and dendritic solidification. We use the "
        "simplest physical pretext — the 2D Cahn-Hilliard equation — with class-"
        "specific parameter tweaks to generate synthetic tear textures, then ask "
        "whether a frozen DINOv2-B encoder places them near the intended real-"
        "class centroids.\n"
    )
    lines.append("## PDE and class parameter mapping\n")
    lines.append(
        "```\n∂u/∂t = M ∇² (f'(u) − ε² ∇² u),   f(u) = (well_depth/4)(u²−1)²\n```\n"
    )
    lines.append("| Class | u₀_mean | u₀_amp | ε² | M | well_depth | steps | biological rationale |")
    lines.append("|---|---|---|---|---|---|---|---|")
    from teardrop.physics_sim import CLASS_PRESETS as CP
    rationales = {
        "ZdraviLudia": "baseline balanced spinodal",
        "Diabetes": "hyperglycaemia → u₀ bias, higher ε² (thicker interfaces)",
        "PGOV_Glaukom": "MMP-9 disrupts mucin matrix → low ε² (fine branches)",
        "SklerozaMultiplex": "altered protein diffusivity → higher M, shallower well",
        "SucheOko": "low aqueous fraction → low M, halted evolution, high initial noise",
    }
    for cls in CLASSES:
        p = CP[cls]
        lines.append(
            f"| {PRETTY[cls]} | {p.u0_mean:+.2f} | {p.u0_amp:.2f} | {p.epsilon2:.2f} "
            f"| {p.mobility:.2f} | {p.well_depth:.2f} | {p.steps} | {rationales[cls]} |"
        )
    lines.append("")
    lines.append(f"{n_per_class} simulations per class with 15% multiplicative jitter on "
                 "mobility / ε² / initial-noise amplitude and independent RNG seeds.\n")

    lines.append("## Nearest-centroid diagnostic\n")
    lines.append(
        "For each simulation we compute its cosine similarity to the five real-class "
        "centroids (in the DINOv2-B embedding space, fit only on real scans) and ask "
        "whether the top-1 / top-3 nearest centroid matches the intended class.\n"
    )
    lines.append(f"**Overall top-1: {cluster_stats['overall_top1']*100:.1f}%** "
                 f"(chance = {100/len(CLASSES):.0f}%)")
    lines.append(f"**Overall top-3: {cluster_stats['overall_top3']*100:.1f}%**\n")
    lines.append("| Class | top-1 | top-2 | top-3 | mean cos-sim to own centroid |")
    lines.append("|---|---|---|---|---|")
    for cls in CLASSES:
        st = cluster_stats["per_class"].get(cls, {})
        if not st:
            continue
        lines.append(
            f"| {PRETTY[cls]} | {st['top1']*100:.0f}% | {st['top2']*100:.0f}% "
            f"| {st['top3']*100:.0f}% | {st['mean_sim_to_own_centroid']:+.3f} |"
        )
    lines.append("")

    if aug_stats is not None:
        lines.append("## Augmentation experiment (LOPO)\n")
        lines.append(
            "We add the simulated embeddings into each LOPO training fold with "
            "their intended labels and compare scan-level weighted-F1 against the "
            "real-only baseline.\n"
        )
        lines.append(f"- Real only: **{aug_stats['f1_real_only']:.3f}**")
        lines.append(f"- Real + sim: **{aug_stats['f1_real_plus_sim']:.3f}**")
        lines.append(f"- Delta: **{aug_stats['delta']:+.3f}**\n")
        if aug_stats["delta"] > 0.005:
            verdict = ("Synthetic Cahn-Hilliard textures provide a small positive signal "
                       "when mixed into the training set — keep as augmentation.")
        elif aug_stats["delta"] < -0.005:
            verdict = ("Adding simulated textures hurts — the DINOv2-B encoder places "
                       "them close enough to real scans to introduce label noise.")
        else:
            verdict = ("No meaningful effect — the simulations are far enough from the "
                       "real manifold that the logistic probe ignores them.")
        lines.append(f"**Verdict:** {verdict}\n")

    lines.append("## Honest limitations\n")
    lines.append(
        "- Pure Cahn-Hilliard gives **isotropic labyrinthine** phase separation. Real "
        "tear scans show directional dendritic ferning driven by directional drying "
        "and anisotropic surface energy, which CH alone cannot reproduce.\n"
        "- The class-parameter mapping is a cartoon; we are not claiming each term "
        "corresponds quantitatively to a measured biological quantity.\n"
        "- Even if simulations do not land on real centroids, the exercise still "
        "documents the physics → embedding gap, which is useful for future directions "
        "(phase-field crystal, Kobayashi-Warren anisotropic solidification, or DDPM "
        "conditioned on real scans).\n"
    )
    lines.append("## Artefacts\n")
    lines.append("- `teardrop/physics_sim.py` — solver + class presets.")
    lines.append("- `scripts/physics_pretraining_experiment.py` — this experiment.")
    lines.append("- `reports/pitch/10_physics_simulation.png` — real vs sim grid + UMAP.")
    lines.append(f"- `cache/{CACHE.name}` — simulated embeddings (reused on re-run).\n")
    OUT_REPORT.write_text("\n".join(lines))
    print(f"  -> wrote {OUT_REPORT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()

    n_per_class = 30

    # 1. simulate
    heights_sim, y_sim, cls_names, sim_idx = generate_simulations(n_per_class=n_per_class)

    # 2. encode (use cache if present and correct size)
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        if d["X"].shape[0] == len(heights_sim):
            X_sim = d["X"]
            print(f"  reusing cached embeddings: {X_sim.shape}")
        else:
            X_sim = encode_simulations(heights_sim)
            np.savez_compressed(CACHE, X=X_sim, y=y_sim,
                                 cls_names=np.array(cls_names),
                                 sim_idx=np.array(sim_idx))
    else:
        X_sim = encode_simulations(heights_sim)
        np.savez_compressed(CACHE, X=X_sim, y=y_sim,
                             cls_names=np.array(cls_names),
                             sim_idx=np.array(sim_idx))

    # 3. load real
    banner("Step 3: load real DINOv2-B embeddings")
    X_real, y_real, groups_real, paths_real = load_real_embeddings()

    # 4. cluster-agreement diagnostic
    banner("Step 4: nearest-centroid diagnostic")
    cluster_stats = cluster_agreement(X_real, y_real, X_sim, y_sim)
    print(f"  overall top-1: {cluster_stats['overall_top1']*100:.1f}%  "
          f"top-3: {cluster_stats['overall_top3']*100:.1f}%")
    for cls, st in cluster_stats["per_class"].items():
        print(f"    {cls:22s} top-1 {st['top1']*100:5.1f}%  "
              f"top-3 {st['top3']*100:5.1f}%  "
              f"cos-sim {st['mean_sim_to_own_centroid']:+.3f}")

    # 5. joint UMAP
    banner("Step 5: joint UMAP")
    Z_real, Z_sim, proj_name = joint_umap(X_real, X_sim)
    print(f"  projection: {proj_name}")

    # 6. augmentation
    aug_stats = augmentation_experiment(X_real, y_real, groups_real, X_sim, y_sim)

    # 7. real-scan examples for figure
    banner("Step 7: pick real-scan examples for figure")
    real_examples = load_real_scan_examples()
    print(f"  got {len(real_examples)} real examples")

    # 8. figure
    make_pitch_figure(
        heights_sim, y_sim, Z_real, y_real, Z_sim, proj_name,
        cluster_stats, real_examples,
    )

    # 9. report
    elapsed = time.perf_counter() - t_start
    write_report(n_per_class, cluster_stats, aug_stats, elapsed)
    print(f"\nDone in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
