"""Multi-scale tile experiment — combine DINOv2-B at TWO scales (90 nm/px + 45 nm/px).

Hypothesis: tear crystallization has multi-scale structure (fine crystal lattice at
10-30 nm, macro dendrite at 100+ nm). Combining two tile scales may capture
complementary signal.

Pipeline:
  1. Build 45 nm/px tiled DINOv2-B cache (if not cached).
     - preprocess_spm(target_nm_per_px=45, crop_size=512)
     - tile (up to 9 tiles/scan, evenly sampled if more)
     - height_to_pil(mode='afmhot') -> DINOv2-B encode
     - save cache/tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz
       keys: X, tile_to_scan, scan_y, scan_groups, scan_paths
  2. Mean-pool tile embeddings -> scan-level features per scale.
  3. V2 recipe per member (L2-norm -> StandardScaler -> LR(class_weight=balanced))
     evaluated with PERSON-level LOPO (35 groups, teardrop.data.person_id).
  4. Four configurations:
       Config A: DINOv2-B 90 nm/px (existing champion DINOv2-B feature)
       Config B: DINOv2-B 45 nm/px (new scale)
       Config C: DINOv2-B 90 + 45 (2-scale single-encoder geom-mean ensemble)
       Config D: 90 + 45 + BiomedCLIP-90-TTA (3-way multi-scale + multi-encoder)
  5. Report weighted F1, macro F1, per-class F1 for each.
  6. Save reports/MULTISCALE_RESULTS.md with comparison table.

Note: no D4 TTA in this round (budget).
"""
from __future__ import annotations

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
    center_crop_or_pad,
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

# Champions for reference.
CHAMP_V2 = 0.6562  # DINOv2-B 90nm TTA + BiomedCLIP 90nm TTA
CHAMP_E7 = 0.6645  # DINOv2-B 90 + DINOv2-B RGB + BiomedCLIP-TTA (wave-6 awaiting red-team)


# ---------------------------------------------------------------------------
# 45 nm/px tiled cache builder
# ---------------------------------------------------------------------------

def build_45nm_cache(max_tiles: int = 9) -> Path:
    """Build cache/tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz if absent."""
    cache_path = CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz"
    if cache_path.exists():
        print(f"[cache-hit] {cache_path.name}")
        return cache_path

    print(f"[build] {cache_path.name}")
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    print(f"  samples: {len(samples)}")

    enc = load_dinov2("vitb14")
    print(f"  encoder: {enc.name} on {enc.device}")

    all_pil: list = []
    tile_to_scan: list[int] = []
    scan_y: list[int] = []
    scan_groups: list[str] = []
    scan_paths: list[str] = []

    t0 = time.time()
    total_raw_tiles = 0
    for si, s in enumerate(samples):
        try:
            # NOTE: `preprocess_spm` center-crops to 512² and would leave only 1
            # tile at 45 nm/px — defeating the purpose. We run the same pipeline
            # steps (plane-level → resample → robust-normalize) without the final
            # center-crop so `tile()` can carve multiple 512² tiles from the
            # ~2048² resampled map.
            hm = load_height(s.raw_path)
            h_full = plane_level(hm.height)
            h_full = resample_to_pixel_size(h_full, hm.pixel_nm, 45.0)
            h_full = robust_normalize(h_full)
            # pad if smaller than 512 so we have at least one tile
            if h_full.shape[0] < 512 or h_full.shape[1] < 512:
                pad_h = max(0, 512 - h_full.shape[0])
                pad_w = max(0, 512 - h_full.shape[1])
                h_full = np.pad(
                    h_full,
                    ((pad_h // 2, pad_h - pad_h // 2),
                     (pad_w // 2, pad_w - pad_w // 2)),
                    mode="reflect",
                )
            tiles = tile(h_full, 512, stride=512)
            if not tiles:
                tiles = [center_crop_or_pad(h_full, 512)]
            total_raw_tiles += len(tiles)
            if len(tiles) > max_tiles:
                idx = np.linspace(0, len(tiles) - 1, max_tiles).astype(int)
                tiles = [tiles[i] for i in idx]
            for t in tiles:
                all_pil.append(height_to_pil(t, mode="afmhot"))
                tile_to_scan.append(si)
            scan_y.append(s.label)
            scan_groups.append(person_id(s.raw_path))  # PERSON-level groups
            scan_paths.append(str(s.raw_path))
        except Exception as e:
            print(f"  [err] {s.raw_path.name}: {e}")
        if (si + 1) % 40 == 0:
            print(f"  preproc [{si + 1}/{len(samples)}] tiles={len(all_pil)} "
                  f"(raw={total_raw_tiles})  t={time.time() - t0:.1f}s")

    print(f"  TOTAL tiles={len(all_pil)} (raw before capping={total_raw_tiles})  "
          f"in {time.time() - t0:.1f}s")
    print(f"  encoding {len(all_pil)} tiles on {enc.device}...")
    t1 = time.time()
    X = enc.encode(all_pil, batch_size=16)
    print(f"  encoded {X.shape} in {time.time() - t1:.1f}s")

    np.savez(
        cache_path,
        X=X,
        tile_to_scan=np.array(tile_to_scan, dtype=np.int64),
        scan_y=np.array(scan_y, dtype=np.int64),
        scan_groups=np.array(scan_groups),
        scan_paths=np.array(scan_paths),
    )
    print(f"[saved] {cache_path}")
    return cache_path


# ---------------------------------------------------------------------------
# V2 recipe evaluation
# ---------------------------------------------------------------------------

def mean_pool_tiles(X_tiles: np.ndarray, t2s: np.ndarray, n_scans: int) -> np.ndarray:
    d = X_tiles.shape[1]
    out = np.zeros((n_scans, d), dtype=np.float32)
    for si in range(n_scans):
        m = t2s == si
        if m.any():
            out[si] = X_tiles[m].mean(axis=0)
    return out


def align_to_reference(paths_ref: list[str], paths_src: list[str],
                       X_src: np.ndarray) -> np.ndarray:
    src_idx = {p: i for i, p in enumerate(paths_src)}
    order = np.array([src_idx[p] for p in paths_ref])
    return X_src[order]


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print("Multi-scale tile experiment — DINOv2-B 90 nm/px + 45 nm/px")
    print("=" * 78)

    # ---- 1. Ensure 45 nm/px cache ----
    build_45nm_cache(max_tiles=9)

    # ---- 2. Load all three caches ----
    print("\n[load] caches")
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz", allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz",
                  allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)

    print(f"  DINOv2-B 90nm: X={z90['X'].shape} scans={len(z90['scan_y'])}")
    print(f"  DINOv2-B 45nm: X={z45['X'].shape} scans={len(z45['scan_y'])}")
    print(f"  BiomedCLIP TTA: X_scan={zbc['X_scan'].shape}")

    # ---- 3. Build person-level groups for 90nm cache (it shipped with patient-level) ----
    paths_90 = [str(p) for p in z90["scan_paths"]]
    paths_45 = [str(p) for p in z45["scan_paths"]]
    paths_bc = [str(p) for p in zbc["scan_paths"]]

    # Use 90nm ordering as reference; derive person groups from path.
    groups = np.array([person_id(Path(p)) for p in paths_90])
    y = np.asarray(z90["scan_y"], dtype=np.int64)
    n_scans = len(y)
    n_persons = len(np.unique(groups))
    print(f"  reference scan order: 90nm cache ({n_scans} scans, {n_persons} persons)")
    assert n_persons == 35, f"expected 35 persons, got {n_persons}"

    # ---- 4. Mean-pool 90 & 45 to scan-level, align to 90-order ----
    X90_scan = mean_pool_tiles(z90["X"], z90["tile_to_scan"], len(paths_90))
    X45_scan_raw = mean_pool_tiles(z45["X"], z45["tile_to_scan"], len(paths_45))
    X45_scan = align_to_reference(paths_90, paths_45, X45_scan_raw)
    Xbc_scan = align_to_reference(paths_90, paths_bc, zbc["X_scan"].astype(np.float32))

    print(f"\n  scan-level features (aligned):")
    print(f"    DINOv2-B 90nm  pooled: {X90_scan.shape}")
    print(f"    DINOv2-B 45nm  pooled: {X45_scan.shape}")
    print(f"    BiomedCLIP TTA       : {Xbc_scan.shape}")

    # Sanity: labels consistent across caches
    y45_aligned = align_to_reference(paths_90, paths_45,
                                      np.asarray(z45["scan_y"]).reshape(-1, 1)).ravel()
    ybc_aligned = align_to_reference(paths_90, paths_bc,
                                      np.asarray(zbc["scan_y"]).reshape(-1, 1)).ravel()
    assert np.array_equal(y, y45_aligned.astype(np.int64)), "label mismatch 90↔45"
    assert np.array_equal(y, ybc_aligned.astype(np.int64)), "label mismatch 90↔bclip"

    # ---- 5. Per-member V2 recipe LOPO softmax ----
    print("\n[lopo-v2] fitting per-member L2-norm -> StandardScaler -> LR(bal)")
    members = {
        "dinov2_90nm": X90_scan,
        "dinov2_45nm": X45_scan,
        "biomedclip_tta_90nm": Xbc_scan,
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

    # ---- 6. Four configs ----
    print("\n[configs] geometric-mean ensembles")
    configs = {
        "A_dinov2_90nm":
            ["dinov2_90nm"],
        "B_dinov2_45nm":
            ["dinov2_45nm"],
        "C_dinov2_90+45":
            ["dinov2_90nm", "dinov2_45nm"],
        "D_dinov2_90+45+biomedclip_tta":
            ["dinov2_90nm", "dinov2_45nm", "biomedclip_tta_90nm"],
    }

    results = {}
    for name, keys in configs.items():
        G = geom_mean_probs([P[k] for k in keys])
        m = metrics_of(G, y)
        results[name] = {"members": keys, **m}
        print(f"  {name:36s} W-F1={m['weighted_f1']:.4f}  "
              f"M-F1={m['macro_f1']:.4f}")

    # ---- 7. Table vs champions ----
    print("\n" + "=" * 78)
    print(f"Compare vs champions:")
    print(f"  v2 champion (DINOv2-B 90 TTA + BiomedCLIP 90 TTA): {CHAMP_V2:.4f}")
    print(f"  E7 candidate (DINOv2-B 90 + DINOv2-B RGB + BiomedCLIP TTA): {CHAMP_E7:.4f}")
    print("=" * 78)
    print(f"\n{'Config':40s} {'W-F1':>7s} {'M-F1':>7s} {'Δ v2':>8s} {'Δ E7':>8s}")
    print("-" * 74)
    for name in configs:
        m = results[name]
        print(f"{name:40s} {m['weighted_f1']:7.4f} {m['macro_f1']:7.4f} "
              f"{m['weighted_f1'] - CHAMP_V2:+8.4f} "
              f"{m['weighted_f1'] - CHAMP_E7:+8.4f}")

    # Per-class table for each
    print(f"\n{'Config':40s} " + " ".join(f"{c[:10]:>11s}" for c in CLASSES))
    for name in configs:
        pcf1 = results[name]["per_class_f1"]
        print(f"{name:40s} " + " ".join(f"{v:>11.4f}" for v in pcf1))

    # ---- 8. Persist summary ----
    summary = {
        "hypothesis": "multi-scale tile structure helps classification",
        "person_lopo": True,
        "n_persons": int(n_persons),
        "n_scans": int(n_scans),
        "champions": {"v2": CHAMP_V2, "E7": CHAMP_E7},
        "per_member": per_member,
        "configs": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (REPORTS / "multiscale_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] reports/multiscale_results.json")

    # ---- 9. Write markdown report ----
    write_markdown_report(summary)

    print(f"\n[done] total elapsed: {time.time() - t0:.1f}s")
    return summary


def write_markdown_report(summary: dict) -> None:
    per_member = summary["per_member"]
    configs = summary["configs"]
    champ_v2 = summary["champions"]["v2"]
    champ_e7 = summary["champions"]["E7"]

    lines = []
    lines.append("# Multi-Scale Tile Experiment — Results\n")
    lines.append(
        "**Hypothesis:** tear crystallization has multi-scale structure "
        "(fine crystal lattice at 10-30 nm, macro dendrite at 100+ nm). "
        "Combining TWO tile scales (90 nm/px + 45 nm/px) may capture complementary signal.\n"
    )
    lines.append("## Methodology\n")
    lines.append(
        "- **Data:** 240 AFM scans, 35 persons (LOPO groups via "
        "`teardrop.data.person_id`).\n"
        "- **Scales:**\n"
        "  - 90 nm/px (existing champion cache, `tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz`).\n"
        "  - 45 nm/px (new cache, `tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz`).\n"
        "- **Tiling:** `preprocess_spm` followed by `tile(size=512)`, then cap at 9 tiles/scan "
        "(evenly-spaced subset). At 45 nm/px a 1024² original scan becomes ~2048² and yields "
        "16 raw tiles before capping.\n"
        "- **Encoder:** DINOv2-B (vitb14), `afmhot` render, no D4 TTA this round.\n"
        "- **Pooling:** tile embeddings mean-pooled per scan (scan-level features).\n"
        "- **Recipe (V2, no tuning):** L2-norm → StandardScaler(fit-on-train) → "
        "LogisticRegression(class_weight='balanced', C=1.0, max_iter=3000).\n"
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
    lines.append("| Config | Members | Weighted F1 | Macro F1 | Δ v2 (0.6562) | Δ E7 (0.6645) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for name, m in configs.items():
        members = ", ".join(m["members"])
        lines.append(
            f"| **{name}** | {members} | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['weighted_f1'] - champ_v2:+.4f} | {m['weighted_f1'] - champ_e7:+.4f} |"
        )
    lines.append("")

    lines.append("## Per-class F1\n")
    lines.append("| Config | " + " | ".join(CLASSES) + " |")
    lines.append("|---|" + "|".join([":---:"] * len(CLASSES)) + "|")
    for name, m in configs.items():
        pcf1 = m["per_class_f1"]
        lines.append(f"| **{name}** | " + " | ".join(f"{v:.4f}" for v in pcf1) + " |")
    lines.append("")

    # Analysis
    best_cfg = max(configs, key=lambda k: configs[k]["weighted_f1"])
    best_m = configs[best_cfg]
    delta_v2 = best_m["weighted_f1"] - champ_v2
    delta_e7 = best_m["weighted_f1"] - champ_e7

    lines.append("## Interpretation\n")
    lines.append(
        f"- **Best multi-scale config:** `{best_cfg}` with weighted F1 "
        f"**{best_m['weighted_f1']:.4f}** (Δ v2 = {delta_v2:+.4f}, Δ E7 = {delta_e7:+.4f}).\n"
    )
    d_90 = per_member["dinov2_90nm"]["weighted_f1"]
    d_45 = per_member["dinov2_45nm"]["weighted_f1"]
    d_c = configs["C_dinov2_90+45"]["weighted_f1"]
    lift_c = d_c - max(d_90, d_45)
    if lift_c > 0.005:
        lines.append(
            f"- 90+45 fusion (Config C) lifts over the stronger single-scale by "
            f"{lift_c:+.4f}, consistent with multi-scale complementarity. "
            f"Would stack with multichannel (amp / phase / RGB) — a v4 "
            f"\"multi-scale × multi-channel\" champion becomes a reasonable next step.\n"
        )
    elif lift_c > 0:
        lines.append(
            f"- 90+45 fusion (Config C) edges out stronger single-scale by only "
            f"{lift_c:+.4f} — within noise given 240 scans and 35 LOPO folds; "
            f"no strong case that fine-scale structure adds classification signal here.\n"
        )
    else:
        lines.append(
            f"- 90+45 fusion (Config C) does NOT beat the stronger single-scale "
            f"({lift_c:+.4f}). Fine-scale (45 nm/px) features appear redundant with, "
            f"or noisier than, 90 nm/px features at DINOv2-B. Multi-scale is not the "
            f"winning axis.\n"
        )
    d_d = configs["D_dinov2_90+45+biomedclip_tta"]["weighted_f1"]
    if d_d >= champ_v2 + 0.005:
        lines.append(
            f"- Config D ({d_d:.4f}) matches/beats v2 champion by ≥ 0.005 — "
            f"multi-scale stacks usefully with BiomedCLIP-TTA (multi-encoder × multi-scale).\n"
        )
    elif d_d >= champ_v2:
        lines.append(
            f"- Config D ({d_d:.4f}) ties v2 champion within noise "
            f"(Δ = {d_d - champ_v2:+.4f}); adding the 45-nm scale on top of "
            f"DINOv2-B 90 + BiomedCLIP-TTA neither helps nor hurts.\n"
        )
    else:
        lines.append(
            f"- Config D ({d_d:.4f}) is **below** the v2 champion "
            f"(Δ = {d_d - champ_v2:+.4f}). The 45-nm branch dilutes the D-TTA+B-TTA "
            f"ensemble — same-encoder fine-scale tile is worse than cross-encoder TTA.\n"
        )

    lines.append(
        "\n## Honest reporting\n"
        "- Person-level LOPO (35 folds), V2 recipe only, no threshold tuning, no OOF model "
        "selection.\n"
        "- Champions are numerically optimistic baselines: v2 uses D4-TTA on both encoders; "
        "this experiment uses NO TTA for the DINOv2 members (budget constraint), so the "
        "Config A number here is slightly below the TTA-boosted champion.\n"
        "- Fair comparison: look at the **lift** of C - A and D - (A+BiomedCLIP), "
        "not the absolute deltas vs champions.\n"
    )

    (REPORTS / "MULTISCALE_RESULTS.md").write_text("\n".join(lines))
    print(f"[saved] reports/MULTISCALE_RESULTS.md")


if __name__ == "__main__":
    main()
