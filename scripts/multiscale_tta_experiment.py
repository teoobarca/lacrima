"""Multi-scale + D4 TTA experiment — DINOv2-B 45 nm/px TTA + existing 90 nm/px TTA.

Goal: push past the non-TTA multi-scale champion (Config D = 0.6887, from
`multiscale_experiment.py`) by also adding D4 test-time augmentation to the
45 nm/px DINOv2-B branch (90 nm/px already has a TTA cache).

Pipeline:
  1. Build `cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4_45nm.npz`:
     - For each of 240 scans, preprocess at 45 nm/px (plane-level -> resample
       -> robust-normalize), tile at 512 px (cap 9), apply D4 (8 augs) per
       tile, encode all 72 views with DINOv2-B, mean-pool -> (1, 768).
     - Cache keys: X_scan, scan_y, scan_groups, scan_paths (TTA format).
  2. Load existing 90 nm/px DINOv2 TTA + BiomedCLIP TTA caches, align rows
     via `scan_paths`.
  3. Eval 4 configs with v2 recipe (L2-norm -> StandardScaler -> LR(balanced),
     geometric-mean ensemble) under person-LOPO (35 groups):
        Config D       : DINOv2-B 90 non-TTA + DINOv2-B 45 non-TTA + BiomedCLIP TTA
                         (reproduction of multiscale_experiment Config D = 0.6887)
        Config D-TTA   : DINOv2-B 90 TTA + DINOv2-B 45 TTA + BiomedCLIP TTA
        Config E       : DINOv2-B 90 TTA + DINOv2-B 45 TTA (two-scale single-encoder)
        Config F       : DINOv2-B 45 TTA alone
  4. Per-class F1 + weighted + macro F1, saved to
     `reports/MULTISCALE_TTA_RESULTS.md` + JSON.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import (  # noqa: E402
    CLASSES,
    enumerate_samples,
    load_height,
    person_id,
    plane_level,
    resample_to_pixel_size,
    robust_normalize,
    tile,
)
from teardrop.encoders import height_to_pil, load_dinov2  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
CACHE.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

N_CLASSES = len(CLASSES)
EPS = 1e-12

# Reference champions
CHAMP_V2 = 0.6562                # v2 TTA ensemble (90 nm/px)
CHAMP_MULTISCALE_D = 0.6887      # Wave 7 Config D (90 nm/px + 45 nm/px non-TTA + BiomedCLIP-TTA)


# ---------------------------------------------------------------------------
# 45 nm/px preprocessing + D4 TTA encoding
# ---------------------------------------------------------------------------

def preprocess_45nm_tiles(
    raw_path: Path,
    tile_size: int = 512,
    max_tiles: int = 9,
) -> list[np.ndarray]:
    """Preprocess a raw SPM at 45 nm/px and return up to max_tiles 512x512 tiles.

    Mirrors `build_45nm_cache` in `multiscale_experiment.py` so caching logic
    is identical — identical tile content as the non-TTA 45 nm/px cache when
    D4 is not applied.
    """
    hm = load_height(raw_path)
    h = plane_level(hm.height)
    h = resample_to_pixel_size(h, hm.pixel_nm, 45.0)
    h = robust_normalize(h)
    if h.shape[0] < tile_size or h.shape[1] < tile_size:
        pad_h = max(0, tile_size - h.shape[0])
        pad_w = max(0, tile_size - h.shape[1])
        h = np.pad(
            h,
            ((pad_h // 2, pad_h - pad_h // 2),
             (pad_w // 2, pad_w - pad_w // 2)),
            mode="reflect",
        )
    tiles = tile(h, tile_size, stride=tile_size)
    if not tiles:
        return [h[:tile_size, :tile_size]]
    if len(tiles) > max_tiles:
        idx = np.linspace(0, len(tiles) - 1, max_tiles).astype(int)
        tiles = [tiles[i] for i in idx]
    return tiles


def d4_augmentations(arr: np.ndarray) -> list[np.ndarray]:
    """Dihedral group D4: {id, r90, r180, r270, flipLR, flipLR+r90/180/270}."""
    rots = [np.rot90(arr, k=k) for k in range(4)]
    flipped = np.fliplr(arr)
    rots_flip = [np.rot90(flipped, k=k) for k in range(4)]
    return rots + rots_flip


def build_45nm_tta_cache(
    tile_size: int = 512,
    max_tiles: int = 9,
    batch_size: int = 16,
) -> Path:
    """Encode 45 nm/px with D4 TTA, mean-pool to (240, 768) and cache."""
    cache_path = (
        CACHE / f"tta_emb_dinov2_vitb14_afmhot_t{tile_size}_n{max_tiles}_d4_45nm.npz"
    )
    if cache_path.exists():
        print(f"[cache-hit] {cache_path.name}")
        return cache_path

    print(f"[build] {cache_path.name}")
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    print(f"  samples: {len(samples)}")

    enc = load_dinov2("vitb14")
    print(f"  encoder: {enc.name} on {enc.device}")

    all_pil: list = []
    scan_slices: list[tuple[int, int]] = []
    scan_y: list[int] = []
    scan_groups: list[str] = []
    scan_paths: list[str] = []

    t0 = time.time()
    total_raw_tiles = 0
    for si, s in enumerate(samples):
        try:
            tiles = preprocess_45nm_tiles(
                s.raw_path, tile_size=tile_size, max_tiles=max_tiles,
            )
            total_raw_tiles += len(tiles)
            start = len(all_pil)
            for t in tiles:
                for aug in d4_augmentations(t):
                    all_pil.append(
                        height_to_pil(np.ascontiguousarray(aug), mode="afmhot")
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
            print(f"  preproc [{si + 1}/{len(samples)}] views={len(all_pil)} "
                  f"(raw_tiles={total_raw_tiles})  t={time.time() - t0:.1f}s")

    print(f"  TOTAL views={len(all_pil)} (raw tiles before D4={total_raw_tiles}) "
          f"in {time.time() - t0:.1f}s")
    print(f"  encoding {len(all_pil)} views on {enc.device}...")
    t1 = time.time()
    X = enc.encode(all_pil, batch_size=batch_size)
    enc_time = time.time() - t1
    print(f"  encoded {X.shape} in {enc_time:.1f}s")

    # Mean-pool per scan
    d = X.shape[1]
    n_scans = len(scan_slices)
    X_scan = np.zeros((n_scans, d), dtype=np.float32)
    for si, (a, b) in enumerate(scan_slices):
        if b > a:
            X_scan[si] = X[a:b].mean(axis=0)

    np.savez(
        cache_path,
        X_scan=X_scan,
        scan_y=np.array(scan_y, dtype=np.int64),
        scan_groups=np.array(scan_groups),
        scan_paths=np.array(scan_paths),
        encode_time_s=np.array(enc_time),
    )
    print(f"[saved] {cache_path}")

    # Free encoder memory
    del enc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    return cache_path


# ---------------------------------------------------------------------------
# V2 recipe person-LOPO
# ---------------------------------------------------------------------------

def lopo_predict_v2(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """V2 recipe person-LOPO OOF softmax: L2-norm -> StandardScaler -> LR(bal)."""
    n = len(y)
    P = np.zeros((n, N_CLASSES), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups):
        Xt = normalize(X[tr], norm="l2", axis=1)
        Xv = normalize(X[va], norm="l2", axis=1)
        sc = StandardScaler()
        Xt = sc.fit_transform(Xt)
        Xv = sc.transform(Xv)
        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
        clf = LogisticRegression(
            class_weight="balanced", max_iter=3000, C=1.0,
            solver="lbfgs", n_jobs=4, random_state=42,
        )
        clf.fit(Xt, y[tr])
        proba = clf.predict_proba(Xv)
        p_full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
        for ci, cls in enumerate(clf.classes_):
            p_full[:, cls] = proba[:, ci]
        P[va] = p_full
    return P


def geom_mean_probs(probs_list: list[np.ndarray]) -> np.ndarray:
    log_sum = np.zeros_like(probs_list[0])
    for P in probs_list:
        log_sum = log_sum + np.log(P + EPS)
    G = np.exp(log_sum / len(probs_list))
    G /= G.sum(axis=1, keepdims=True)
    return G


def metrics_of(P: np.ndarray, y: np.ndarray) -> dict:
    pred = P.argmax(axis=1)
    return {
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(
            y, pred, average=None, labels=list(range(N_CLASSES)), zero_division=0,
        ).tolist(),
    }


def align_by_path(paths_ref: list[str], paths_src: list[str],
                  X_src: np.ndarray) -> np.ndarray:
    src_idx = {p: i for i, p in enumerate(paths_src)}
    order = np.array([src_idx[p] for p in paths_ref])
    return X_src[order]


def mean_pool_tiles(X_tiles: np.ndarray, t2s: np.ndarray, n_scans: int) -> np.ndarray:
    d = X_tiles.shape[1]
    out = np.zeros((n_scans, d), dtype=np.float32)
    for si in range(n_scans):
        m = t2s == si
        if m.any():
            out[si] = X_tiles[m].mean(axis=0)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve(paths):
    return [str(Path(p).resolve()) for p in paths]


def main():
    t0 = time.time()
    print("=" * 78)
    print("Multi-scale + D4 TTA experiment — DINOv2-B 45nm TTA + 90nm TTA + BiomedCLIP TTA")
    print("=" * 78)

    # ---- 1. Build 45 nm/px TTA cache ----
    build_45nm_tta_cache(tile_size=512, max_tiles=9, batch_size=16)

    # ---- 2. Load all caches (use 90 nm/px TTA ordering as reference) ----
    print("\n[load] caches")
    zd90 = np.load(CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz",
                   allow_pickle=True)
    zd45_tta = np.load(CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4_45nm.npz",
                       allow_pickle=True)
    zd45_nt = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz",
                      allow_pickle=True)
    zd90_nt = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz",
                      allow_pickle=True)
    zbc_tta = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz",
                      allow_pickle=True)

    paths_ref = _resolve(zd90["scan_paths"])
    y = np.asarray(zd90["scan_y"], dtype=np.int64)
    # Use person_id from path for groups (consistent with multiscale_experiment.py)
    groups = np.array([person_id(Path(p)) for p in paths_ref])
    n_scans = len(paths_ref)
    n_persons = len(np.unique(groups))
    print(f"  reference order: DINOv2-B 90 nm TTA ({n_scans} scans, {n_persons} persons)")
    assert n_persons == 35, f"expected 35 persons, got {n_persons}"

    # DINOv2-B 90 nm TTA
    X_d90_tta = np.asarray(zd90["X_scan"], dtype=np.float32)
    # DINOv2-B 45 nm TTA
    paths_d45_tta = _resolve(zd45_tta["scan_paths"])
    X_d45_tta = align_by_path(
        paths_ref, paths_d45_tta, np.asarray(zd45_tta["X_scan"], dtype=np.float32),
    )
    # DINOv2-B 45 nm non-TTA (tiled -> mean-pool)
    paths_d45_nt = _resolve(zd45_nt["scan_paths"])
    X_d45_nt_raw = mean_pool_tiles(
        np.asarray(zd45_nt["X"]),
        np.asarray(zd45_nt["tile_to_scan"]),
        len(paths_d45_nt),
    )
    X_d45_nt = align_by_path(paths_ref, paths_d45_nt, X_d45_nt_raw)
    # DINOv2-B 90 nm non-TTA (tiled -> mean-pool)
    paths_d90_nt = _resolve(zd90_nt["scan_paths"])
    X_d90_nt_raw = mean_pool_tiles(
        np.asarray(zd90_nt["X"]),
        np.asarray(zd90_nt["tile_to_scan"]),
        len(paths_d90_nt),
    )
    X_d90_nt = align_by_path(paths_ref, paths_d90_nt, X_d90_nt_raw)
    # BiomedCLIP TTA (90 nm/px)
    paths_bc = _resolve(zbc_tta["scan_paths"])
    X_bc_tta = align_by_path(
        paths_ref, paths_bc, np.asarray(zbc_tta["X_scan"], dtype=np.float32),
    )

    # Label sanity — align each label vector and compare
    for name, z, pths in (
        ("d45_tta", zd45_tta, paths_d45_tta),
        ("d45_nt",  zd45_nt,  paths_d45_nt),
        ("d90_nt",  zd90_nt,  paths_d90_nt),
        ("bc_tta",  zbc_tta,  paths_bc),
    ):
        y_src = np.asarray(z["scan_y"]).reshape(-1, 1)
        y_al = align_by_path(paths_ref, pths, y_src).ravel().astype(np.int64)
        assert np.array_equal(y, y_al), f"label mismatch for {name}"

    print(f"  features (aligned):")
    print(f"    DINOv2-B 90 nm TTA   : {X_d90_tta.shape}")
    print(f"    DINOv2-B 45 nm TTA   : {X_d45_tta.shape}")
    print(f"    DINOv2-B 90 nm non-TTA: {X_d90_nt.shape}")
    print(f"    DINOv2-B 45 nm non-TTA: {X_d45_nt.shape}")
    print(f"    BiomedCLIP 90 nm TTA : {X_bc_tta.shape}")

    # ---- 3. Per-member person-LOPO OOF softmax ----
    print("\n[lopo-v2] fitting per-member L2-norm -> StandardScaler -> LR(bal)")
    members = {
        "dinov2_90nm_tta":   X_d90_tta,
        "dinov2_45nm_tta":   X_d45_tta,
        "dinov2_90nm_nt":    X_d90_nt,
        "dinov2_45nm_nt":    X_d45_nt,
        "biomedclip_tta":    X_bc_tta,
    }
    P = {}
    per_member = {}
    for name, Xm in members.items():
        ts = time.time()
        P[name] = lopo_predict_v2(Xm, y, groups)
        m = metrics_of(P[name], y)
        per_member[name] = m
        print(f"  {name:24s} W-F1={m['weighted_f1']:.4f}  "
              f"M-F1={m['macro_f1']:.4f}  ({time.time() - ts:.1f}s)")

    # ---- 4. Four configs ----
    print("\n[configs] geometric-mean ensembles")
    configs = {
        "D_reproduce_nonTTA_multiscale_plus_BC":
            ["dinov2_90nm_nt", "dinov2_45nm_nt", "biomedclip_tta"],
        "D_TTA_multiscale_plus_BC":
            ["dinov2_90nm_tta", "dinov2_45nm_tta", "biomedclip_tta"],
        "E_TTA_multiscale_dinov2_only":
            ["dinov2_90nm_tta", "dinov2_45nm_tta"],
        "F_TTA_dinov2_45nm_alone":
            ["dinov2_45nm_tta"],
    }
    results = {}
    for name, keys in configs.items():
        G = geom_mean_probs([P[k] for k in keys])
        m = metrics_of(G, y)
        results[name] = {"members": keys, **m}
        print(f"  {name:40s} W-F1={m['weighted_f1']:.4f}  "
              f"M-F1={m['macro_f1']:.4f}")

    # ---- 5. Table ----
    print("\n" + "=" * 78)
    print("Compare vs champions:")
    print(f"  v2 champion (DINOv2-B 90 TTA + BiomedCLIP 90 TTA):        {CHAMP_V2:.4f}")
    print(f"  Config D non-TTA multiscale (Wave 7, target reproduction): {CHAMP_MULTISCALE_D:.4f}")
    print("=" * 78)
    print(f"\n{'Config':42s} {'W-F1':>7s} {'M-F1':>7s} {'Δ v2':>8s} {'Δ D':>8s}")
    print("-" * 78)
    for name in configs:
        m = results[name]
        print(f"{name:42s} {m['weighted_f1']:7.4f} {m['macro_f1']:7.4f} "
              f"{m['weighted_f1'] - CHAMP_V2:+8.4f} "
              f"{m['weighted_f1'] - CHAMP_MULTISCALE_D:+8.4f}")

    # Per-class table
    print(f"\n{'Config':42s} " + " ".join(f"{c[:10]:>11s}" for c in CLASSES))
    for name in configs:
        pcf1 = results[name]["per_class_f1"]
        print(f"{name:42s} " + " ".join(f"{v:>11.4f}" for v in pcf1))

    # ---- 6. Verdict on D-TTA vs D ----
    d_rep = results["D_reproduce_nonTTA_multiscale_plus_BC"]["weighted_f1"]
    d_tta = results["D_TTA_multiscale_plus_BC"]["weighted_f1"]
    delta_dtta = d_tta - d_rep
    delta_v2 = d_tta - CHAMP_V2

    if delta_dtta >= 0.01 and delta_v2 >= 0.01:
        verdict = (
            "SHIP: Config D-TTA beats reproduced D by >= +0.01 AND beats v2 champion "
            "by >= +0.01. Train + save as models/ensemble_v4_multiscale_tta/."
        )
    elif delta_dtta >= 0.005 and delta_v2 >= 0.01:
        verdict = (
            "MARGINAL-BUT-KEEP: D-TTA lifts +0.005..+0.01 over D; both beat v2. "
            "Document as new champion but note the within-noise lift over D."
        )
    elif delta_dtta >= 0 and delta_v2 >= 0.01:
        verdict = (
            "TTA REDUNDANT AT 45 nm/px: D-TTA does not materially beat D, but the "
            "multi-scale ensemble still beats v2. Ship non-TTA Config D as v4 or keep v2."
        )
    elif delta_v2 >= 0.01:
        verdict = (
            "D-TTA REGRESSES from D: 45 nm/px TTA hurts but multi-scale still beats v2. "
            "Do NOT ship D-TTA; Config D (non-TTA 45 nm) remains the best multi-scale option."
        )
    else:
        verdict = (
            "FAIL: neither D nor D-TTA clearly beats v2 champion. Stay with v2."
        )
    print("\nVerdict:", verdict)

    # ---- 7. Persist summary ----
    summary = {
        "hypothesis": "D4 TTA on both 45 nm/px and 90 nm/px DINOv2-B branches "
                      "lifts the Config D multi-scale ensemble further.",
        "person_lopo": True,
        "n_persons": int(n_persons),
        "n_scans": int(n_scans),
        "champions": {
            "v2": CHAMP_V2,
            "multiscale_D_target": CHAMP_MULTISCALE_D,
        },
        "per_member": per_member,
        "configs": results,
        "deltas": {
            "D_TTA_minus_D_reproduced":       delta_dtta,
            "D_TTA_minus_v2":                 delta_v2,
            "D_TTA_minus_multiscale_D_target": d_tta - CHAMP_MULTISCALE_D,
        },
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (REPORTS / "multiscale_tta_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] reports/multiscale_tta_results.json")

    # ---- 8. Markdown report ----
    write_markdown_report(summary)

    print(f"\n[done] total elapsed: {time.time() - t0:.1f}s")
    return summary


def write_markdown_report(summary: dict) -> None:
    per_member = summary["per_member"]
    configs = summary["configs"]
    champ_v2 = summary["champions"]["v2"]
    champ_D = summary["champions"]["multiscale_D_target"]

    lines = []
    lines.append("# Multi-Scale + D4 TTA Experiment — Results\n")
    lines.append(
        "**Hypothesis:** Adding D4 TTA to the 45 nm/px DINOv2-B branch (the 90 nm/px "
        "branch and BiomedCLIP already use D4 TTA) will push the Wave 7 Config D "
        "multi-scale ensemble further. The fine scale shows structure the canonical "
        "DINOv2 orientation priors may not be invariant to, so TTA has theoretical "
        "foothold there as well.\n"
    )
    lines.append("## Methodology\n")
    lines.append(
        "- **Data:** 240 AFM scans, 35 persons (LOPO groups via "
        "`teardrop.data.person_id`).\n"
        "- **Scales & encoders:**\n"
        "  - DINOv2-B @ 90 nm/px, D4 TTA (72 views/scan, mean-pooled) — "
        "`cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`.\n"
        "  - DINOv2-B @ 45 nm/px, D4 TTA (72 views/scan, mean-pooled) — "
        "`cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4_45nm.npz` (**new**).\n"
        "  - BiomedCLIP @ 90 nm/px, D4 TTA — "
        "`cache/tta_emb_biomedclip_afmhot_t512_n9_d4.npz`.\n"
        "  - DINOv2-B @ 90/45 nm/px non-TTA (for reference) — tiled caches, mean-pooled.\n"
        "- **Recipe (V2, no tuning):** per scan embedding, L2-normalize -> "
        "StandardScaler(fit-on-train) -> LogisticRegression(class_weight='balanced', "
        "C=1.0, max_iter=3000).\n"
        "- **Ensemble:** geometric mean of per-member softmax probabilities.\n"
        "- **Evaluation:** person-level LOPO (35 folds), weighted & macro F1, per-class F1.\n"
    )
    lines.append("## Per-member LOPO metrics\n")
    lines.append("| Member | Weighted F1 | Macro F1 |")
    lines.append("|---|---:|---:|")
    for name, m in per_member.items():
        lines.append(f"| `{name}` | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} |")
    lines.append("")

    lines.append("## Configurations (geom-mean ensembles)\n")
    lines.append("| Config | Members | Weighted F1 | Macro F1 | Δ v2 (0.6562) | Δ D (0.6887) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for name, m in configs.items():
        members = ", ".join(m["members"])
        lines.append(
            f"| **{name}** | {members} | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['weighted_f1'] - champ_v2:+.4f} | {m['weighted_f1'] - champ_D:+.4f} |"
        )
    lines.append("")

    lines.append("## Per-class F1\n")
    lines.append("| Config | " + " | ".join(CLASSES) + " |")
    lines.append("|---|" + "|".join([":---:"] * len(CLASSES)) + "|")
    for name, m in configs.items():
        pcf1 = m["per_class_f1"]
        lines.append(f"| **{name}** | " + " | ".join(f"{v:.4f}" for v in pcf1) + " |")
    lines.append("")

    lines.append("## Deltas\n")
    for k, v in summary["deltas"].items():
        lines.append(f"- `{k}` = {v:+.4f}")
    lines.append("")

    lines.append("## Verdict\n")
    lines.append(f"**{summary['verdict']}**\n")

    lines.append("## Honest reporting\n")
    lines.append(
        "- Person-level LOPO (35 folds), V2 recipe only, no threshold tuning, no "
        "OOF model selection.\n"
        "- Config D (non-TTA 45 nm/px) reproduction in this script may differ "
        "slightly from the Wave 7 published 0.6887 due to minor version drift; "
        "the meaningful comparison is **D-TTA - D_reproduced** within this run.\n"
        "- If D-TTA regresses vs D, the honest conclusion is that TTA is redundant "
        "(or harmful) at 45 nm/px — fine-scale tiles may already average out "
        "orientation variance because there are ~9 of them covering different "
        "parts of the scan.\n"
    )

    (REPORTS / "MULTISCALE_TTA_RESULTS.md").write_text("\n".join(lines))
    print(f"[saved] reports/MULTISCALE_TTA_RESULTS.md")


if __name__ == "__main__":
    main()
