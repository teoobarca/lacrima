"""Bonus experiment: does MAE-pretrained ViT-Tiny add complementary signal to v4?

Individually MAE underperforms (see MAE_PRETRAINING_RESULTS.md), but low individual
F1 CAN still help when combined -- if MAE features capture structure the other encoders
miss. This script evaluates MAE as a 4th member of the v4 recipe (geomean ensemble of
softmaxes, L2-norm + StandardScaler + LR per member, person-LOPO).

Members:
  A) DINOv2-B 90 nm/px tiles (mean-pool per scan)
  B) DINOv2-B 45 nm/px tiles (mean-pool per scan)
  C) BiomedCLIP 90 nm/px TTA D4 (already scan-level)
  D) MAE ViT-Tiny CLS features (mean-pool per scan)

Compare:
  - v4 baseline (A+B+C)
  - v4 + MAE (A+B+C+D)
  - MAE alone (D)

Uses PERSON-level LOPO.
"""
from __future__ import annotations

import sys
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
from teardrop.data import CLASSES, person_id  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
SEED = 42
EPS = 1e-9


def _mean_pool(X_tiles, t2s, n):
    D = X_tiles.shape[1]
    out = np.zeros((n, D), dtype=np.float32)
    cnt = np.zeros(n, dtype=np.int32)
    for i, s in enumerate(t2s):
        out[s] += X_tiles[i]
        cnt[s] += 1
    cnt = np.maximum(cnt, 1)
    return out / cnt[:, None]


def _load_scan_features(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_scan, y, groups, path list)."""
    if name == "dino90":
        z = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz", allow_pickle=True)
        X = _mean_pool(z["X"].astype(np.float32), z["tile_to_scan"].astype(int), len(z["scan_y"]))
    elif name == "dino45":
        z = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz", allow_pickle=True)
        X = _mean_pool(z["X"].astype(np.float32), z["tile_to_scan"].astype(int), len(z["scan_y"]))
    elif name == "bmc_tta":
        z = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)
        # tta_emb has scan-level features already (averaged over TTA views)
        X = z["X_scan"].astype(np.float32) if "X_scan" in z.files else None
        if X is None:
            # Fall back to tiled-TTA -> mean pool
            X = _mean_pool(z["X"].astype(np.float32), z["tile_to_scan"].astype(int), len(z["scan_y"]))
    elif name == "mae":
        z = np.load(CACHE / "mae_emb_tear_tiny.npz", allow_pickle=True)
        X = z["X_scan"].astype(np.float32)
    else:
        raise ValueError(name)
    y = z["scan_y"].astype(int)
    paths = z["scan_paths"].tolist()
    persons = np.array([person_id(Path(p)) for p in paths])
    return X, y, persons, paths


def _fit_softmax_lopo(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Honest LOPO: return (n_scans, n_classes) softmax probs."""
    n = len(y)
    probs = np.zeros((n, len(CLASSES)), dtype=np.float32)
    for tr, va in leave_one_patient_out(groups):
        Xn = normalize(X[tr], norm="l2", axis=1)
        sc = StandardScaler().fit(Xn)
        Xt = sc.transform(Xn)
        Xv = sc.transform(normalize(X[va], norm="l2", axis=1))
        clf = LogisticRegression(
            class_weight="balanced", max_iter=3000, C=1.0,
            solver="lbfgs", n_jobs=4, random_state=SEED,
        )
        clf.fit(Xt, y[tr])
        probs[va] = clf.predict_proba(Xv)
    return probs


def _geomean(probs_list: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(probs_list, axis=0)  # (M, N, C)
    log_p = np.log(np.clip(stack, EPS, 1.0)).mean(axis=0)
    p = np.exp(log_p)
    p = p / p.sum(axis=1, keepdims=True).clip(EPS)
    return p


def main():
    print("=" * 72)
    print("MAE ensemble check: does MAE add complementary signal to v4?")
    print("=" * 72)

    # Load everything.
    X_dino90, y_d90, g_d90, paths_d90 = _load_scan_features("dino90")
    X_dino45, y_d45, g_d45, _ = _load_scan_features("dino45")
    X_bmc_tta, y_b, g_b, _ = _load_scan_features("bmc_tta")
    X_mae, y_m, g_m, _ = _load_scan_features("mae")

    # Sanity: all should be length 240 with same y / groups.
    for name, arr in [("y", [y_d90, y_d45, y_b, y_m]), ("g", [g_d90, g_d45, g_b, g_m])]:
        same = all(np.array_equal(arr[0], a) for a in arr[1:])
        print(f"  {name} consistent: {same}  shapes={[a.shape for a in arr]}")

    y = y_d90
    groups = g_d90

    print(f"\nFeature shapes:")
    print(f"  dino90      {X_dino90.shape}")
    print(f"  dino45      {X_dino45.shape}")
    print(f"  bmc_tta     {X_bmc_tta.shape}")
    print(f"  mae         {X_mae.shape}")

    # Per-member LOPO probs + F1.
    members = {
        "dino90": X_dino90,
        "dino45": X_dino45,
        "bmc_tta": X_bmc_tta,
        "mae": X_mae,
    }
    probs = {}
    print("\nPer-member LOPO F1 (weighted / macro):")
    for name, X in members.items():
        p = _fit_softmax_lopo(X, y, groups)
        probs[name] = p
        pred = p.argmax(axis=1)
        f1w = f1_score(y, pred, average="weighted")
        f1m = f1_score(y, pred, average="macro")
        print(f"  {name:10s}  F1w={f1w:.4f}  F1m={f1m:.4f}")

    # Ensembles.
    print("\nEnsembles (geomean of softmaxes):")
    combos = {
        "v4 (dino90 + dino45 + bmc_tta)": ["dino90", "dino45", "bmc_tta"],
        "v4 + MAE (4-way)": ["dino90", "dino45", "bmc_tta", "mae"],
        "dino90 + MAE": ["dino90", "mae"],
        "dino45 + MAE": ["dino45", "mae"],
        "bmc_tta + MAE": ["bmc_tta", "mae"],
    }
    results = {}
    for label, names in combos.items():
        p = _geomean([probs[n] for n in names])
        pred = p.argmax(axis=1)
        f1w = f1_score(y, pred, average="weighted")
        f1m = f1_score(y, pred, average="macro")
        results[label] = (f1w, f1m, pred)
        print(f"  {label:40s}  F1w={f1w:.4f}  F1m={f1m:.4f}")

    # Summary line for report append.
    f1_v4 = results["v4 (dino90 + dino45 + bmc_tta)"][0]
    f1_v4_mae = results["v4 + MAE (4-way)"][0]
    delta = f1_v4_mae - f1_v4

    print("\n" + "=" * 72)
    print(f"v4 baseline:         {f1_v4:.4f}  weighted F1  (expected ~0.6887 per v4 report)")
    print(f"v4 + MAE (4-way):    {f1_v4_mae:.4f}  weighted F1")
    print(f"Delta MAE adds to v4: {delta:+.4f}")
    if delta > 0.005:
        print("VERDICT: MAE adds complementary signal! Consider integrating as 4th encoder.")
    elif delta < -0.005:
        print("VERDICT: MAE HURTS v4 ensemble. Do not integrate.")
    else:
        print("VERDICT: MAE neutral to v4 (within noise).")

    # Append to MAE report.
    append_path = REPORTS / "MAE_PRETRAINING_RESULTS.md"
    if append_path.exists():
        with open(append_path, "a") as f:
            f.write("\n## Bonus: MAE features as 4th member of v4 ensemble\n\n")
            f.write("Does MAE add complementary signal despite lower individual F1? Quick ensemble test:\n\n")
            f.write("| Configuration | Weighted F1 | Macro F1 |\n")
            f.write("|---|---:|---:|\n")
            for label, (f1w, f1m, _) in results.items():
                f.write(f"| {label} | {f1w:.4f} | {f1m:.4f} |\n")
            f.write("\n")
            f.write(f"**Delta MAE adds to v4 baseline: {delta:+.4f} weighted F1.**\n\n")
            if delta > 0.005:
                f.write("Even though MAE scores lower individually (0.556 vs DINOv2-B 0.615), it ")
                f.write("captures complementary structure that improves the geomean ensemble.\n")
            elif delta < -0.005:
                f.write("MAE features hurt the ensemble -- they share failure modes with the other ")
                f.write("members without adding complementary signal. Do not integrate.\n")
            else:
                f.write("MAE features are neutral to the v4 ensemble -- within sampling noise. ")
                f.write("Integration would add complexity without clear gain.\n")
            f.write("\n")
        print(f"  appended ensemble results to {append_path}")


if __name__ == "__main__":
    main()
