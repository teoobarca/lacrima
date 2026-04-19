"""Multichannel x V2-recipe fusion — does combining multichannel encoding with
the L2-norm + GEOMETRIC mean recipe push past the 0.6562 champion?

V2 recipe (champion):
  1. Load scan-level embedding per encoder
  2. L2-normalize each row
  3. StandardScaler (fit on train fold, transform val)
  4. LogisticRegression(class_weight='balanced') predict_proba
  5. GEOMETRIC mean of softmaxes across members -> argmax

Multichannel inputs tested (DINOv2-B, mean-pooled from tiled cache):
  - X_height       scan-level (240, 768)   [tiles: 811]
  - X_amp          scan-level (240, 768)   [tiles: 811]
  - X_phase        scan-level (240, 768)   -- 13 scans fall back to X_height
  - X_rgb          scan-level (240, 768)   [tiles: 811]
  - X_bclip        scan-level (240, 512)   [from TTA-D4 cache, Height channel]

All eight ensembles use person-LOPO groups (teardrop.data.person_id). No
threshold tuning, no OOF selection: each ensemble's label prediction is just
argmax of the geometric mean of its members' softmaxes across all 240 scans.
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
from teardrop.data import CLASSES  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
N_CLASSES = len(CLASSES)
EPS = 1e-12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean_pool_tiles(X_tiles: np.ndarray, t2s: np.ndarray, n_scans: int) -> np.ndarray:
    """Mean-pool tile embeddings to scan-level features."""
    d = X_tiles.shape[1]
    out = np.zeros((n_scans, d), dtype=np.float32)
    for si in range(n_scans):
        m = t2s == si
        if m.any():
            out[si] = X_tiles[m].mean(axis=0)
    return out


def align_to_reference(paths_ref: list[str], paths_src: list[str],
                       X_src: np.ndarray) -> np.ndarray:
    """Return X_src reindexed so row i matches paths_ref[i]."""
    src_idx = {p: i for i, p in enumerate(paths_src)}
    order = np.array([src_idx[p] for p in paths_ref])
    return X_src[order]


def lopo_predict_v2(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """V2 recipe person-LOPO OOF softmax predictions.

    Pipeline per fold: L2-normalize -> StandardScaler -> LR(class_weight=bal).
    Returns (n, n_classes) of out-of-fold softmax probs aligned to class order.
    """
    n = len(y)
    P = np.zeros((n, N_CLASSES), dtype=np.float64)
    # Drop zero-variance columns globally (fit-on-train in recipe uses StandardScaler
    # which handles it; we keep all columns for consistency with the v2 script).
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
        "pred": pred.tolist(),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print("Multichannel x V2-recipe fusion — 8 ensembles vs 0.6562 champion")
    print("=" * 78)

    # --- Load TTA caches (reference alignment = DINOv2-B TTA Height scan order) ---
    print("\n[load] TTA DINOv2-B (Height, D4)  &  TTA BiomedCLIP (Height, D4)")
    zd = np.load(CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz", allow_pickle=True)
    zb = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)
    Xd_tta = zd["X_scan"].astype(np.float32)
    y = zd["scan_y"].astype(np.int64)
    groups = zd["scan_groups"].astype(str)
    tta_paths = [str(Path(p)) for p in zd["scan_paths"]]
    Xb_tta = align_to_reference(tta_paths,
                                 [str(Path(p)) for p in zb["scan_paths"]],
                                 zb["X_scan"].astype(np.float32))
    print(f"  TTA-DINOv2-B Height: {Xd_tta.shape}")
    print(f"  TTA-BiomedCLIP Height: {Xb_tta.shape}")
    print(f"  persons: {len(np.unique(groups))}  classes: {N_CLASSES}")

    # --- Load multichannel tiled cache, mean-pool to scan-level per channel ---
    print("\n[load] multichannel tiled cache -> scan-level mean-pool")
    zm = np.load(CACHE / "multichan_tiled_emb_dinov2vitb14_t512_n9.npz", allow_pickle=True)

    def scan_from_channel(key: str):
        Xt = zm[f"X_{key}"].astype(np.float32)
        t2s = zm[f"t2s_{key}"].astype(np.int64)
        paths = [str(Path(p)) for p in zm[f"paths_{key}"]]
        n_scans = len(paths)
        X_scan_local = mean_pool_tiles(Xt, t2s, n_scans)
        return X_scan_local, paths

    X_h_scan, h_paths = scan_from_channel("height")
    X_a_scan, a_paths = scan_from_channel("amplitude")
    X_p_scan, p_paths = scan_from_channel("phase")
    X_r_scan, r_paths = scan_from_channel("rgb")

    # Align height/amp/rgb to TTA scan order
    X_h = align_to_reference(tta_paths, h_paths, X_h_scan)
    X_a = align_to_reference(tta_paths, a_paths, X_a_scan)
    X_r = align_to_reference(tta_paths, r_paths, X_r_scan)

    # Phase: 227 scans present; for 13 missing scans, fall back to Height
    p_idx = {p: i for i, p in enumerate(p_paths)}
    X_ph = np.zeros_like(X_h)
    n_fallback = 0
    for i, p in enumerate(tta_paths):
        if p in p_idx:
            X_ph[i] = X_p_scan[p_idx[p]]
        else:
            X_ph[i] = X_h[i]  # Height fallback (v2 recipe applied after)
            n_fallback += 1
    print(f"  X_height: {X_h.shape}  X_amp: {X_a.shape}  "
          f"X_phase: {X_ph.shape} (fallback->Height for {n_fallback} scans)  "
          f"X_rgb: {X_r.shape}")
    assert n_fallback == 13, f"expected 13 missing-phase scans, got {n_fallback}"

    # --- Compute per-member OOF softmax via v2 recipe ---
    print("\n[lopo] fitting V2 recipe on each member (L2-norm -> SS -> LR)")
    members = {
        "dinov2_tta_height": (Xd_tta, y, groups),
        "biomedclip_tta_height": (Xb_tta, y, groups),
        "dinov2_height_pool": (X_h, y, groups),
        "dinov2_amp_pool": (X_a, y, groups),
        "dinov2_phase_pool": (X_ph, y, groups),
        "dinov2_rgb_pool": (X_r, y, groups),
    }
    P = {}
    for name, (X, y_, g_) in members.items():
        ts = time.time()
        P[name] = lopo_predict_v2(X, y_, g_)
        m = metrics_of(P[name], y_)
        print(f"  {name:28s} W-F1={m['weighted_f1']:.4f}  "
              f"M-F1={m['macro_f1']:.4f}  ({time.time() - ts:.1f}s)")

    # --- Experiment grid: geometric mean ensembles ---
    print("\n[ensemble] geometric-mean combinations")
    configs = [
        ("E1_champion_v2: dinov2_tta_H + biomedclip_tta_H",
         ["dinov2_tta_height", "biomedclip_tta_height"]),
        ("E2: dinov2_H_pool + dinov2_amp",
         ["dinov2_height_pool", "dinov2_amp_pool"]),
        ("E3: dinov2_H_pool + dinov2_phase",
         ["dinov2_height_pool", "dinov2_phase_pool"]),
        ("E4: dinov2_H_pool + dinov2_amp + dinov2_phase",
         ["dinov2_height_pool", "dinov2_amp_pool", "dinov2_phase_pool"]),
        ("E5: dinov2_H_pool + dinov2_rgb",
         ["dinov2_height_pool", "dinov2_rgb_pool"]),
        ("E6: dinov2_H_pool + dinov2_amp + biomedclip_tta_H",
         ["dinov2_height_pool", "dinov2_amp_pool", "biomedclip_tta_height"]),
        ("E7: dinov2_H_pool + dinov2_rgb + biomedclip_tta_H",
         ["dinov2_height_pool", "dinov2_rgb_pool", "biomedclip_tta_height"]),
        ("E8: dinov2_H_pool + dinov2_amp + dinov2_phase + biomedclip_tta_H",
         ["dinov2_height_pool", "dinov2_amp_pool", "dinov2_phase_pool",
          "biomedclip_tta_height"]),
        # Additional: also try the v2 champion using TTA DINOv2-B + TTA BiomedCLIP
        # via our own fold loop to sanity-check we reproduce 0.6562.
    ]

    CHAMP = 0.6562
    print(f"  champion (ensemble_v2_tta): {CHAMP:.4f}")
    results = []
    for label, member_keys in configs:
        G = geom_mean_probs([P[k] for k in member_keys])
        m = metrics_of(G, y)
        delta = m["weighted_f1"] - CHAMP
        marker = "  <-- CHAMP candidate" if m["weighted_f1"] >= CHAMP + 0.005 else ""
        print(f"  {label}")
        print(f"    W-F1={m['weighted_f1']:.4f} (Δ={delta:+.4f})  "
              f"M-F1={m['macro_f1']:.4f}{marker}")
        for ci, cname in enumerate(CLASSES):
            print(f"      {cname:22s} f1={m['per_class_f1'][ci]:.4f}")
        results.append({
            "label": label,
            "members": member_keys,
            "weighted_f1": m["weighted_f1"],
            "macro_f1": m["macro_f1"],
            "per_class_f1": m["per_class_f1"],
            "delta_vs_champ": delta,
            "pred": m["pred"],
        })

    # --- Recommendation ---
    ranked = sorted(results, key=lambda r: r["weighted_f1"], reverse=True)
    best = ranked[0]
    print("\n[recommend]")
    print(f"  best ensemble: {best['label']}")
    print(f"    W-F1={best['weighted_f1']:.4f}  M-F1={best['macro_f1']:.4f}  "
          f"Δ={best['delta_vs_champ']:+.4f}")

    next_champ_candidate = best["delta_vs_champ"] >= 0.005

    # --- Persist ---
    summary = {
        "champion_v2_reported": CHAMP,
        "person_lopo": True,
        "n_persons": int(len(np.unique(groups))),
        "n_scans": int(len(y)),
        "phase_fallback_height_count": int(n_fallback),
        "member_metrics": {
            name: {
                "weighted_f1": metrics_of(P[name], y)["weighted_f1"],
                "macro_f1": metrics_of(P[name], y)["macro_f1"],
                "per_class_f1": metrics_of(P[name], y)["per_class_f1"],
            }
            for name in members
        },
        "ensembles": [
            {k: v for k, v in r.items() if k != "pred"} for r in results
        ],
        "best": {k: v for k, v in best.items() if k != "pred"},
        "next_champion_candidate": bool(next_champ_candidate),
        "elapsed_s": round(time.time() - t0, 1),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "multichannel_v2_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] reports/multichannel_v2_results.json")

    # Save best predictions if it beat champion by >= 0.005
    if next_champ_candidate:
        member_probs = {k: P[k] for k in best["members"]}
        np.savez(
            CACHE / "best_multichannel_v2_predictions.npz",
            y=y, groups=groups, tta_paths=np.array(tta_paths),
            pred=np.array(best["pred"]),
            ensemble_label=best["label"],
            members=np.array(best["members"]),
            **{f"P_{k}": v for k, v in member_probs.items()},
        )
        print(f"[saved] cache/best_multichannel_v2_predictions.npz "
              f"(Δ={best['delta_vs_champ']:+.4f} beats champion)")
    else:
        print("[info] no ensemble beat champion by >= 0.005; "
              "not saving best_multichannel_v2_predictions.npz")

    print(f"\n[done] total elapsed: {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
