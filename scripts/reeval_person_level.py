"""Re-evaluate best models with PERSON-LEVEL (not eye-level) grouping.

Validator found that patient_id doesn't merge L/P eyes of same person.
With person_id, 44 "patients" → 35 persons. This is the honest LOPO.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.data import CLASSES, enumerate_samples, person_id
from teardrop.cv import leave_one_patient_out, patient_stratified_kfold

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"


def load_tiled(name: str):
    path = CACHE / f"tiled_emb_{name}_afmhot_t512_n9.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return {
        "X": z["X"],
        "tile_to_scan": z["tile_to_scan"],
        "scan_y": z["scan_y"],
        "scan_groups_eye": z["scan_groups"],  # this is patient_id (eye-level)
        "scan_paths": z["scan_paths"],
    }


def person_groups_from_paths(paths: list[str]) -> np.ndarray:
    from pathlib import Path as P
    return np.array([person_id(P(p)) for p in paths])


def aggregate_tiles(X_tiles, tile_to_scan, n_scans):
    out = np.zeros((n_scans, X_tiles.shape[1]), dtype=np.float32)
    cnt = np.zeros(n_scans, dtype=np.int32)
    for ti, si in enumerate(tile_to_scan):
        out[si] += X_tiles[ti]
        cnt[si] += 1
    cnt = np.maximum(cnt, 1)
    return out / cnt[:, None]


def eval_lr(X, y, groups, label: str):
    print(f"\n=== {label}  ({len(np.unique(groups))} groups) ===")
    lopo_preds = np.full(len(y), -1, dtype=int)
    for tr, va in leave_one_patient_out(groups):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr])
        Xv = scaler.transform(X[va])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0,
                                 solver="lbfgs", n_jobs=4)
        clf.fit(Xt, y[tr])
        lopo_preds[va] = clf.predict(Xv)
    f1w = f1_score(y, lopo_preds, average="weighted")
    f1m = f1_score(y, lopo_preds, average="macro")
    print(f"  LOPO weighted F1: {f1w:.4f}  macro F1: {f1m:.4f}")
    print(classification_report(y, lopo_preds, target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(y, lopo_preds, labels=list(range(len(CLASSES))))
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
    return f1w, f1m


def main():
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    n_scans = len(samples)

    # Rebuild person groups by path matching
    paths_to_si = {str(s.raw_path): i for i, s in enumerate(samples)}
    person_by_si = np.array([s.person for s in samples])
    eye_by_si = np.array([s.patient for s in samples])

    # Models to test
    for name in ["dinov2_vits14", "dinov2_vitb14", "biomedclip"]:
        d = load_tiled(name)
        if d is None:
            continue
        X_scan = aggregate_tiles(d["X"], d["tile_to_scan"], n_scans)
        y = d["scan_y"]

        # sanity: align order (cached scan_y is in same order as enumerate_samples since builder used that)
        print(f"\n\n############## {name} ##############")
        # Eye-level (old)
        print(f"--- eye-level (old 'patient' group: 44) ---")
        eval_lr(X_scan, y, eye_by_si, f"{name} EYE-LEVEL")

        # Person-level (new)
        print(f"\n--- PERSON-level (merged 35) ---")
        eval_lr(X_scan, y, person_by_si, f"{name} PERSON-LEVEL")


if __name__ == "__main__":
    main()
