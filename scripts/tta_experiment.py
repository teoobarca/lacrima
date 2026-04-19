"""Test-Time Augmentation (TTA) experiment for the teardrop AFM classifier.

Pipeline
--------
For each of 240 scans:
    1. Preprocess → up to 9 non-overlapping 512x512 tiles (same as baseline).
    2. For each tile, apply the dihedral group D4 (8 symmetries):
       {original, rot90, rot180, rot270, flipLR, flipLR+rot90,
        flipLR+rot180, flipLR+rot270}
    3. Render each augmented tile with the `afmhot` colormap -> PIL RGB.
    4. Encode all 72 PIL images with each encoder (DINOv2-B, BiomedCLIP).
    5. Mean-pool across the 72 embeddings -> (1, D) scan-level embedding.

Evaluation
----------
Person-level LOPO with `LogisticRegression(class_weight='balanced',
max_iter=3000, C=1.0) + StandardScaler` per fold. Three configs:

    A. TTA DINOv2-B alone
    B. TTA BiomedCLIP alone
    C. TTA proba-avg ensemble (DINOv2-B + BiomedCLIP)

Both Raw-argmax weighted F1 and macro F1 are reported and compared against
non-TTA cached baselines.

Caches
------
cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz
cache/tta_emb_biomedclip_afmhot_t512_n9_d4.npz
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import (  # noqa: E402
    CLASSES,
    enumerate_samples,
    load_height,
    plane_level,
    resample_to_pixel_size,
    robust_normalize,
    tile,
)
from teardrop.encoders import (  # noqa: E402
    EncoderBundle,
    height_to_pil,
    load_biomedclip,
    load_dinov2,
)

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
N_CLASSES = len(CLASSES)


# ---------------------------------------------------------------------------
# Tile generation (same logic as scripts/baseline_tiled_ensemble.py)
# ---------------------------------------------------------------------------

def preprocess_to_tiles(
    raw_path: Path,
    target_nm_per_px: float = 90.0,
    tile_size: int = 512,
    max_tiles: int = 9,
) -> list[np.ndarray]:
    hm = load_height(raw_path)
    h = plane_level(hm.height)
    h = resample_to_pixel_size(h, hm.pixel_nm, target_nm_per_px)
    h = robust_normalize(h)
    if h.shape[0] < tile_size or h.shape[1] < tile_size:
        pad_h = max(0, tile_size - h.shape[0])
        pad_w = max(0, tile_size - h.shape[1])
        h = np.pad(
            h,
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
            mode="reflect",
        )
    tiles = tile(h, tile_size, stride=tile_size)
    if not tiles:
        return [h[:tile_size, :tile_size]]
    if len(tiles) > max_tiles:
        idx = np.linspace(0, len(tiles) - 1, max_tiles).astype(int)
        tiles = [tiles[i] for i in idx]
    return tiles


# ---------------------------------------------------------------------------
# D4 augmentations
# ---------------------------------------------------------------------------

def d4_augmentations(arr: np.ndarray) -> list[np.ndarray]:
    """Return the 8 elements of the dihedral group D4 applied to a 2D array."""
    rots = [np.rot90(arr, k=k) for k in range(4)]
    flipped = np.fliplr(arr)
    rots_flip = [np.rot90(flipped, k=k) for k in range(4)]
    return rots + rots_flip


# ---------------------------------------------------------------------------
# Build TTA embeddings
# ---------------------------------------------------------------------------

def build_tta_embeddings(
    samples,
    encoder: EncoderBundle,
    render_mode: str = "afmhot",
    tile_size: int = 512,
    max_tiles: int = 9,
    batch_size: int = 16,
):
    """TTA encode: 9 tiles * 8 D4 augs per scan, mean-pool per scan.

    Returns:
        X_scan: (n_scans, D)
        scan_y: (n_scans,)
        scan_groups: (n_scans,) -- person ID
        scan_paths: list[str]
    """
    cache_path = (
        CACHE
        / f"tta_emb_{encoder.name}_{render_mode}_t{tile_size}_n{max_tiles}_d4.npz"
    )
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        print(f"[cache] {cache_path}")
        return (
            z["X_scan"],
            z["scan_y"],
            z["scan_groups"],
            z["scan_paths"].tolist(),
        )

    print(
        f"TTA encode {encoder.name} (render={render_mode}, tile={tile_size}, "
        f"max_tiles={max_tiles}, |D4|=8)"
    )

    # Build list of all PIL images in order, remembering per-scan slice indices.
    all_pil = []
    scan_slices = []  # list[(start, end)] into all_pil
    scan_y = []
    scan_groups = []
    scan_paths = []

    t0 = time.time()
    for si, s in enumerate(samples):
        try:
            tiles = preprocess_to_tiles(
                s.raw_path,
                target_nm_per_px=90.0,
                tile_size=tile_size,
                max_tiles=max_tiles,
            )
            start = len(all_pil)
            for t in tiles:
                for aug in d4_augmentations(t):
                    # np.ascontiguousarray because np.rot90 returns a view which
                    # can confuse downstream libs.
                    all_pil.append(
                        height_to_pil(np.ascontiguousarray(aug), mode=render_mode)
                    )
            end = len(all_pil)
            scan_slices.append((start, end))
            scan_y.append(s.label)
            scan_groups.append(s.person)
            scan_paths.append(str(Path(s.raw_path).resolve()))
        except Exception as e:
            print(f"  [err] {s.raw_path.name}: {e}")
            scan_slices.append((len(all_pil), len(all_pil)))
            scan_y.append(s.label)
            scan_groups.append(s.person)
            scan_paths.append(str(Path(s.raw_path).resolve()))
        if (si + 1) % 40 == 0:
            print(
                f"  preproc [{si+1}/{len(samples)}] "
                f"{len(all_pil)} images  {time.time()-t0:.1f}s"
            )

    print(
        f"  total {len(all_pil)} PIL images from {len(samples)} scans in "
        f"{time.time()-t0:.1f}s"
    )

    print("  encoding...")
    t1 = time.time()
    X = encoder.encode(all_pil, batch_size=batch_size)  # (N_total, D)
    enc_time = time.time() - t1
    print(f"  encoded {X.shape} in {enc_time:.1f}s")

    # Mean-pool per scan.
    n_scans = len(scan_slices)
    X_scan = np.zeros((n_scans, X.shape[1]), dtype=np.float32)
    for si, (a, b) in enumerate(scan_slices):
        if b > a:
            X_scan[si] = X[a:b].mean(axis=0)

    out = dict(
        X_scan=X_scan,
        scan_y=np.array(scan_y, dtype=int),
        scan_groups=np.array(scan_groups),
        scan_paths=np.array(scan_paths),
        encode_time_s=np.array(enc_time),
    )
    np.savez(cache_path, **out)
    print(f"[saved] {cache_path}")
    return out["X_scan"], out["scan_y"], out["scan_groups"], scan_paths


# ---------------------------------------------------------------------------
# Person-LOPO predict_proba
# ---------------------------------------------------------------------------

def lopo_predict_proba(X, y, groups) -> np.ndarray:
    """Return (n, N_CLASSES) OOF predict_proba via person-level LOPO."""
    n = len(y)
    P = np.zeros((n, N_CLASSES), dtype=np.float64)
    var = X.var(axis=0)
    keep = var > 1e-12
    X_k = X[:, keep]
    for tr, va in leave_one_patient_out(groups):
        sc = StandardScaler()
        Xt = sc.fit_transform(X_k[tr])
        Xv = sc.transform(X_k[va])
        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
        clf = LogisticRegression(
            class_weight="balanced", max_iter=3000, C=1.0,
            solver="lbfgs", n_jobs=4,
        )
        clf.fit(Xt, y[tr])
        proba = clf.predict_proba(Xv)
        full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
        for ci, cls in enumerate(clf.classes_):
            full[:, cls] = proba[:, ci]
        P[va] = full
    return P


def f1_of(P, y):
    pred = P.argmax(axis=1)
    return (
        float(f1_score(y, pred, average="weighted", zero_division=0)),
        float(f1_score(y, pred, average="macro", zero_division=0)),
    )


def align_rows(X_a, groups_a, y_a, paths_a, X_b, groups_b, y_b, paths_b):
    """Reorder (X_b, ...) to match the row ordering of (X_a, ...) via scan_paths."""
    path_to_i = {p: i for i, p in enumerate(paths_a)}
    order = np.array([path_to_i[p] for p in paths_b])
    # invert: we want X_b_aligned[i_in_a] = X_b[j where paths_b[j]==paths_a[i_in_a]]
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return X_b[inv], np.array(groups_b)[inv], np.array(y_b)[inv], [paths_b[i] for i in inv]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("TTA experiment (D4 per tile, mean-pool over 72 embeddings per scan)")
    print("=" * 72)

    samples = enumerate_samples(ROOT / "TRAIN_SET")
    print(f"Enumerated {len(samples)} samples")
    persons = sorted({s.person for s in samples})
    print(f"Unique persons: {len(persons)}")

    t_total = time.time()

    # --- Encode with DINOv2-B ---
    print("\n--- Encoder: DINOv2-B ---")
    t_dino0 = time.time()
    enc_d = load_dinov2("vitb14")
    Xd, yd, gd, paths_d = build_tta_embeddings(
        samples, enc_d, render_mode="afmhot", tile_size=512, max_tiles=9,
        batch_size=16,
    )
    t_dino = time.time() - t_dino0
    del enc_d
    import gc; gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    # --- Encode with BiomedCLIP ---
    print("\n--- Encoder: BiomedCLIP ---")
    t_bc0 = time.time()
    enc_b = load_biomedclip()
    Xb, yb, gb, paths_b = build_tta_embeddings(
        samples, enc_b, render_mode="afmhot", tile_size=512, max_tiles=9,
        batch_size=16,
    )
    t_bc = time.time() - t_bc0
    del enc_b
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    # Realign rows via scan_paths (enumerate_samples is deterministic, but be safe)
    Xb2, gb2, yb2, _ = align_rows(Xd, gd, yd, paths_d, Xb, gb, yb, paths_b)
    assert np.array_equal(yd, yb2), "label mismatch after alignment"
    assert np.array_equal(gd, gb2), "group mismatch after alignment"
    y = yd
    groups = gd

    # --- Eval ---
    print("\n--- Eval: person-LOPO LR ---")
    t_eval0 = time.time()

    print("  DINOv2-B alone (TTA)...")
    P_d = lopo_predict_proba(Xd, y, groups)
    f1w_d, f1m_d = f1_of(P_d, y)
    print(f"    weighted F1 = {f1w_d:.4f}   macro F1 = {f1m_d:.4f}")

    print("  BiomedCLIP alone (TTA)...")
    P_b = lopo_predict_proba(Xb2, y, groups)
    f1w_b, f1m_b = f1_of(P_b, y)
    print(f"    weighted F1 = {f1w_b:.4f}   macro F1 = {f1m_b:.4f}")

    print("  Ensemble (DINOv2-B + BiomedCLIP), proba-avg (TTA+TTA)...")
    P_ens = 0.5 * (P_d + P_b)
    f1w_e, f1m_e = f1_of(P_ens, y)
    print(f"    weighted F1 = {f1w_e:.4f}   macro F1 = {f1m_e:.4f}")

    t_eval = time.time() - t_eval0

    # --- Non-TTA baselines (cached) for comparison ---
    print("\n--- Non-TTA baselines (cached, person-LOPO proba-avg w/ mean-pool) ---")

    def load_mean_pool_from_tiled(npz_name: str):
        z = np.load(CACHE / npz_name, allow_pickle=True)
        X_tiles = z["X"]
        t2s = z["tile_to_scan"]
        n_scans = len(z["scan_y"])
        out = np.zeros((n_scans, X_tiles.shape[1]), dtype=np.float32)
        counts = np.zeros(n_scans, dtype=np.int32)
        for ti, si in enumerate(t2s):
            out[si] += X_tiles[ti]
            counts[si] += 1
        out /= np.maximum(counts, 1)[:, None]
        paths = [str(Path(p).resolve()) for p in z["scan_paths"]]
        return out, np.asarray(z["scan_y"]), paths

    # Non-TTA DINOv2-B
    Xd_nt, yd_nt, paths_d_nt = load_mean_pool_from_tiled(
        "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz"
    )
    # Reorder to canonical (TTA) order
    pmap = {p: i for i, p in enumerate(paths_d_nt)}
    order = np.array([pmap[p] for p in paths_d])
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    Xd_nt_al = Xd_nt[inv]
    yd_nt_al = yd_nt[inv]
    assert np.array_equal(yd_nt_al, y), "non-TTA DINOv2 label mismatch after align"

    P_d_nt = lopo_predict_proba(Xd_nt_al, y, groups)
    f1w_d_nt, f1m_d_nt = f1_of(P_d_nt, y)
    print(f"  DINOv2-B scan-mean (non-TTA): weighted F1 = {f1w_d_nt:.4f}   macro F1 = {f1m_d_nt:.4f}")

    # Non-TTA BiomedCLIP
    Xb_nt, yb_nt, paths_b_nt = load_mean_pool_from_tiled(
        "tiled_emb_biomedclip_afmhot_t512_n9.npz"
    )
    pmap = {p: i for i, p in enumerate(paths_b_nt)}
    order = np.array([pmap[p] for p in paths_d])
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    Xb_nt_al = Xb_nt[inv]
    yb_nt_al = yb_nt[inv]
    assert np.array_equal(yb_nt_al, y), "non-TTA BiomedCLIP label mismatch after align"

    P_b_nt = lopo_predict_proba(Xb_nt_al, y, groups)
    f1w_b_nt, f1m_b_nt = f1_of(P_b_nt, y)
    print(f"  BiomedCLIP scan-mean (non-TTA): weighted F1 = {f1w_b_nt:.4f}   macro F1 = {f1m_b_nt:.4f}")

    P_ens_nt = 0.5 * (P_d_nt + P_b_nt)
    f1w_e_nt, f1m_e_nt = f1_of(P_ens_nt, y)
    print(f"  Ensemble (non-TTA), proba-avg: weighted F1 = {f1w_e_nt:.4f}   macro F1 = {f1m_e_nt:.4f}")

    # --- Summary table ---
    print("\n" + "=" * 72)
    print("SUMMARY — person-LOPO, raw argmax")
    print("=" * 72)
    print(f"{'model':40s} {'non-TTA':>10s} {'TTA':>10s} {'Delta':>10s}")
    rows = [
        ("DINOv2-B scan-mean", f1w_d_nt, f1w_d),
        ("BiomedCLIP scan-mean", f1w_b_nt, f1w_b),
        ("Ensemble proba-avg (raw argmax)", f1w_e_nt, f1w_e),
    ]
    for name, non_tta, tta in rows:
        delta = tta - non_tta
        print(f"{name:40s} {non_tta:>10.4f} {tta:>10.4f} {delta:>+10.4f}")

    # Macro F1 table
    print()
    print(f"{'model':40s} {'non-TTA macro':>15s} {'TTA macro':>10s} {'Delta':>10s}")
    rows_m = [
        ("DINOv2-B scan-mean", f1m_d_nt, f1m_d),
        ("BiomedCLIP scan-mean", f1m_b_nt, f1m_b),
        ("Ensemble proba-avg (raw argmax)", f1m_e_nt, f1m_e),
    ]
    for name, non_tta, tta in rows_m:
        delta = tta - non_tta
        print(f"{name:40s} {non_tta:>15.4f} {tta:>10.4f} {delta:>+10.4f}")

    # --- Verdict ---
    delta_ens = f1w_e - f1w_e_nt
    if delta_ens >= 0.01:
        verdict = "SHIP: TTA gives meaningful lift (>= +0.01) on ensemble"
    elif delta_ens >= 0.005:
        verdict = "MARGINAL: +0.005..+0.01 -- within noise for 240 scans"
    elif delta_ens >= -0.005:
        verdict = "NOISE: |Delta| < +0.005 -- keep non-TTA champion"
    else:
        verdict = "NEGATIVE: TTA regresses -- keep non-TTA champion"
    print("\nVerdict:", verdict)

    t_wall = time.time() - t_total
    print(f"\nTotal wall time: {t_wall:.1f}s (DINOv2-B: {t_dino:.1f}s, "
          f"BiomedCLIP: {t_bc:.1f}s, LOPO eval: {t_eval:.1f}s)")

    # --- Write report ---
    write_report(
        f1w_d_nt=f1w_d_nt, f1m_d_nt=f1m_d_nt,
        f1w_b_nt=f1w_b_nt, f1m_b_nt=f1m_b_nt,
        f1w_e_nt=f1w_e_nt, f1m_e_nt=f1m_e_nt,
        f1w_d=f1w_d, f1m_d=f1m_d,
        f1w_b=f1w_b, f1m_b=f1m_b,
        f1w_e=f1w_e, f1m_e=f1m_e,
        verdict=verdict,
        t_dino=t_dino, t_bc=t_bc, t_eval=t_eval, t_wall=t_wall,
    )


def write_report(
    f1w_d_nt, f1m_d_nt, f1w_b_nt, f1m_b_nt, f1w_e_nt, f1m_e_nt,
    f1w_d, f1m_d, f1w_b, f1m_b, f1w_e, f1m_e,
    verdict, t_dino, t_bc, t_eval, t_wall,
):
    out = REPORTS / "TTA_RESULTS.md"
    lines: list[str] = []
    lines.append("# Test-Time Augmentation (TTA) Results")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(
        "Current honest champion: DINOv2-B + BiomedCLIP proba-avg, "
        "raw-argmax weighted F1 = 0.6346 (person-LOPO, 240 scans, 35 persons)."
    )
    lines.append(
        "Does adding D4 test-time augmentation to the encoded tiles close the "
        "gap to the threshold-tuned 0.6528 without any tuning?"
    )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("For each of the 240 scans:")
    lines.append("")
    lines.append(
        "1. Load via `AFMReader.spm`, plane-level, resample to 90 nm/px, "
        "robust-normalize (2..98 percentile clip) -- identical to the tiled baseline."
    )
    lines.append("2. Split into up to 9 non-overlapping 512x512 tiles.")
    lines.append("")
    lines.append(
        "3. For each tile, apply the dihedral group D4 = "
        "{id, r90, r180, r270, flipLR, flipLR+r90, flipLR+r180, flipLR+r270}. "
        "This yields 8 augmented views per tile."
    )
    lines.append(
        "4. Render each augmented tile with the `afmhot` colormap (matplotlib) "
        "-> PIL RGB."
    )
    lines.append(
        "5. Feed all 72 PIL images to the encoder in batches of 16; "
        "mean-pool over the 72 embeddings to a single (1, D) scan vector."
    )
    lines.append("")
    lines.append("Encoders: DINOv2-B (ViT-B/14, D=768) and BiomedCLIP (ViT-B/16, D=512).")
    lines.append("")
    lines.append(
        "Evaluation: person-level LOPO (via `teardrop.data.person_id` -> 35 groups). "
        "Per fold: `StandardScaler` -> `LogisticRegression(class_weight='balanced', "
        "max_iter=3000, C=1.0)`. No threshold tuning, no nested CV. Raw argmax."
    )
    lines.append("")
    lines.append(
        "Non-TTA baselines are re-computed with the exact same LOPO loop using "
        "the cached `tiled_emb_*_t512_n9.npz` embeddings mean-pooled to scan level "
        "(1 tile * 1 aug = 9 embeddings per scan -> mean)."
    )
    lines.append("")
    lines.append("## Results (person-LOPO, raw argmax)")
    lines.append("")
    lines.append("### Weighted F1")
    lines.append("")
    lines.append("| model | non-TTA | TTA (D4) | Delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| DINOv2-B scan-mean | {f1w_d_nt:.4f} | {f1w_d:.4f} | {f1w_d - f1w_d_nt:+.4f} |")
    lines.append(f"| BiomedCLIP scan-mean | {f1w_b_nt:.4f} | {f1w_b:.4f} | {f1w_b - f1w_b_nt:+.4f} |")
    lines.append(f"| **Ensemble proba-avg (raw argmax)** | **{f1w_e_nt:.4f}** | **{f1w_e:.4f}** | **{f1w_e - f1w_e_nt:+.4f}** |")
    lines.append("")
    lines.append("### Macro F1")
    lines.append("")
    lines.append("| model | non-TTA | TTA (D4) | Delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| DINOv2-B scan-mean | {f1m_d_nt:.4f} | {f1m_d:.4f} | {f1m_d - f1m_d_nt:+.4f} |")
    lines.append(f"| BiomedCLIP scan-mean | {f1m_b_nt:.4f} | {f1m_b:.4f} | {f1m_b - f1m_b_nt:+.4f} |")
    lines.append(f"| Ensemble proba-avg (raw argmax) | {f1m_e_nt:.4f} | {f1m_e:.4f} | {f1m_e - f1m_e_nt:+.4f} |")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append(
        "Decision rule (pre-registered): Ensemble Delta-weighted-F1 >= +0.01 -> "
        "ship TTA'd model as new champion and save to `models/ensemble_v1_tta/`. "
        "+0.005..+0.01 -> marginal, report but stay with non-TTA. "
        "|Delta| < +0.005 -> noise, stay. Delta < -0.005 -> negative, stay."
    )
    lines.append("")
    delta_ens = f1w_e - f1w_e_nt
    lines.append("Observed ensemble Delta = {:+.4f} weighted F1.".format(delta_ens))
    lines.append("")
    lines.append("## Compute cost (wall clock)")
    lines.append("")
    lines.append(f"- DINOv2-B TTA encode (240 scans * up to 72 images): {t_dino:.1f} s")
    lines.append(f"- BiomedCLIP TTA encode: {t_bc:.1f} s")
    lines.append(f"- LOPO eval (5 LR runs over 35 persons): {t_eval:.1f} s")
    lines.append(f"- Total wall: {t_wall:.1f} s")
    lines.append("")
    lines.append("Device: MPS (Apple Silicon). Batch size 16.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Mean-pooling over the D4 orbit approximately enforces dihedral invariance "
        "on the final scan embedding. For AFM scans, the physical sample has no "
        "preferred orientation (the scanner head / sample rotation is arbitrary), "
        "so the ground-truth class is D4-invariant. Any variance in the encoder's "
        "output under D4 is therefore noise, and averaging over the orbit should "
        "reduce it -- provided the encoder is not already nearly-equivariant."
    )
    lines.append("")
    lines.append(
        "DINOv2 and BiomedCLIP are trained on natural/medical images with a "
        "*canonical* orientation (faces up, horizons level, etc.), so they are "
        "probably NOT orientation-invariant by default. This is what gives TTA "
        "a theoretical foothold."
    )
    lines.append("")
    lines.append(
        "With only 240 scans the sampling noise of a single weighted-F1 point is "
        "roughly +/- 0.02 (back-of-envelope: one class flip changes F1 by 1/240 in a "
        "uniform ideal, practically more due to class imbalance). So a gain has to "
        "clear +0.01 to even be distinguishable from bootstrap variation, "
        "and +0.02+ to be confidently real."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`")
    lines.append("- `cache/tta_emb_biomedclip_afmhot_t512_n9_d4.npz`")
    lines.append("- `scripts/tta_experiment.py`")
    lines.append("- `reports/TTA_RESULTS.md` (this file)")
    lines.append("")
    out.write_text("\n".join(lines))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
