"""Red-team v2 audit — honest nested LOPO for optimize_ensemble.py claim.

Checks:
 - Grouping level (eye vs person)
 - Bias-tuning leakage on same OOF → nested alternative
 - Honest single number for the same subset+bias method
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out
from teardrop.data import CLASSES, enumerate_samples, person_id

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
N_CLASSES = len(CLASSES)


# ---------------------------------------------------------------------------

def load_tiled(name: str):
    path = CACHE / f"tiled_emb_{name}_afmhot_t512_n9.npz"
    z = np.load(path, allow_pickle=True)
    return (z["X"], z["tile_to_scan"], z["scan_y"],
            z["scan_groups"], z["scan_paths"].tolist())


def aggregate_tiles_to_scan(X_tiles, tile_to_scan, n_scans):
    d = X_tiles.shape[1]
    out = np.zeros((n_scans, d), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int32)
    for ti, si in enumerate(tile_to_scan):
        out[si] += X_tiles[ti]
        counts[si] += 1
    counts = np.maximum(counts, 1)
    out /= counts[:, None]
    return out


def load_handcrafted(samples):
    hc = pd.read_parquet(CACHE / "features_handcrafted.parquet")
    path_to_idx = {str(s.raw_path): i for i, s in enumerate(samples)}
    hc["si"] = hc["raw"].map(path_to_idx)
    hc = hc.dropna(subset=["si"]).sort_values("si")
    hc_cols = [c for c in hc.columns if c not in ("raw", "cls", "label", "patient", "si")]
    Xh = np.zeros((len(samples), len(hc_cols)), dtype=np.float32)
    for _, row in hc.iterrows():
        Xh[int(row["si"])] = row[hc_cols].values.astype(np.float32)
    return np.nan_to_num(Xh, nan=0.0, posinf=0.0, neginf=0.0)


def l2norm(A):
    n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    return A / n


def lopo_lr(X, y, groups, C=1.0):
    n = len(y)
    oof = np.zeros((n, N_CLASSES), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr])
        Xv = scaler.transform(X[va])
        clf = LogisticRegression(class_weight="balanced", max_iter=4000, C=C,
                                 solver="lbfgs", n_jobs=4)
        clf.fit(Xt, y[tr])
        probs = clf.predict_proba(Xv)
        full = np.zeros((len(va), N_CLASSES))
        for i, c in enumerate(clf.classes_):
            full[:, c] = probs[:, i]
        oof[va] = full
    return oof


def tune_biases(probs, y, thr_grid=None, passes=3):
    """Coordinate-ascent bias search — same logic as the script under audit."""
    if thr_grid is None:
        thr_grid = np.linspace(-3.0, 3.0, 25)
    log_p = np.log(np.clip(probs, 1e-9, 1.0))
    biases = np.zeros(N_CLASSES)

    def preds_with(b):
        return (log_p + b[None, :]).argmax(axis=1)

    base_f1 = f1_score(y, preds_with(biases), average="weighted")
    for _ in range(passes):
        improved = False
        for c in range(N_CLASSES):
            best = biases[c]
            best_f1 = base_f1
            for b in thr_grid:
                trial = biases.copy()
                trial[c] = b
                f = f1_score(y, preds_with(trial), average="weighted")
                if f > best_f1 + 1e-6:
                    best_f1 = f
                    best = b
            if best != biases[c]:
                biases[c] = best
                base_f1 = best_f1
                improved = True
        if not improved:
            break
    return biases


def apply_biases(probs, biases):
    log_p = np.log(np.clip(probs, 1e-9, 1.0))
    return (log_p + biases[None, :]).argmax(axis=1)


# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("RED-TEAM V2 — optimize_ensemble.py audit")
    print("=" * 72)

    samples = enumerate_samples(ROOT / "TRAIN_SET")
    n_scans = len(samples)

    s = load_tiled("dinov2_vits14")
    scan_y = s[2]
    scan_groups_eye = s[3]  # patient_id — eye-level
    Xs = aggregate_tiles_to_scan(s[0], s[1], n_scans)

    b = load_tiled("dinov2_vitb14")
    Xb = aggregate_tiles_to_scan(b[0], b[1], n_scans)

    bm = load_tiled("biomedclip")
    Xbm = aggregate_tiles_to_scan(bm[0], bm[1], n_scans)

    Xhc = load_handcrafted(samples)
    y = scan_y.astype(int)

    # Build champion feature: concat all 4, L2-norm the embedding parts
    Xs_n = l2norm(Xs)
    Xb_n = l2norm(Xb)
    Xbm_n = l2norm(Xbm)
    X_champ = np.concatenate([Xs_n, Xb_n, Xbm_n, Xhc], axis=1)
    print(f"Champion feature matrix: {X_champ.shape}")

    # Build person-level groups aligned to samples (scan_paths order)
    scan_paths = s[4]  # list of paths
    raw_to_sample = {str(sp.raw_path): sp for sp in samples}
    person_groups = np.array([raw_to_sample[str(p)].person for p in scan_paths])
    print(f"Eye-level groups: {len(np.unique(scan_groups_eye))} unique")
    print(f"Person-level groups: {len(np.unique(person_groups))} unique")

    # Sanity: confirm saved parquet reproduces as claimed
    saved = pd.read_parquet(REPORTS / "best_oof_predictions.parquet")
    print(f"\nSaved OOF parquet unique patients: {saved['patient'].nunique()}")
    print(f"Saved weighted F1: {f1_score(saved['true_label'], saved['pred_label'], average='weighted'):.4f}")

    # ========================================================================
    # Part A — Reproduce the champion number at eye-level (leaky bias tune)
    # ========================================================================
    print("\n" + "=" * 72)
    print("PART A — eye-level LOPO (44 groups) w/ LEAKY bias tuning (= claim)")
    print("=" * 72)
    t0 = time.time()
    oof_eye = lopo_lr(X_champ, y, scan_groups_eye)
    raw_pred_eye = oof_eye.argmax(axis=1)
    wf_raw_eye = f1_score(y, raw_pred_eye, average="weighted")
    mf_raw_eye = f1_score(y, raw_pred_eye, average="macro")
    print(f"  raw argmax (no bias):  wF1={wf_raw_eye:.4f}  mF1={mf_raw_eye:.4f}  ({time.time()-t0:.1f}s)")

    biases_leaky_eye = tune_biases(oof_eye, y)
    leaky_pred_eye = apply_biases(oof_eye, biases_leaky_eye)
    wf_leaky_eye = f1_score(y, leaky_pred_eye, average="weighted")
    mf_leaky_eye = f1_score(y, leaky_pred_eye, average="macro")
    pc_leaky_eye = f1_score(y, leaky_pred_eye, average=None,
                            labels=list(range(N_CLASSES)), zero_division=0)
    print(f"  leaky bias-tuned:      wF1={wf_leaky_eye:.4f}  mF1={mf_leaky_eye:.4f}")
    print(f"  biases: {biases_leaky_eye}")
    print(f"  per-class F1: " + " | ".join(f"{c[:5]}={f:.3f}" for c, f in zip(CLASSES, pc_leaky_eye)))

    # ========================================================================
    # Part B — eye-level, NESTED bias tuning
    # ========================================================================
    print("\n" + "=" * 72)
    print("PART B — eye-level LOPO, NESTED bias tuning")
    print("=" * 72)
    # For each outer fold (1 patient), tune biases on the remaining 43 patients'
    # OOF predictions (oof_eye with that patient excluded), then apply to the
    # held-out patient.
    honest_pred_eye = np.full(n_scans, -1, dtype=int)
    biases_per_fold_eye = []
    t0 = time.time()
    for tr, va in leave_one_patient_out(scan_groups_eye):
        inner_biases = tune_biases(oof_eye[tr], y[tr])
        honest_pred_eye[va] = apply_biases(oof_eye[va], inner_biases)
        biases_per_fold_eye.append(inner_biases)
    wf_honest_eye = f1_score(y, honest_pred_eye, average="weighted")
    mf_honest_eye = f1_score(y, honest_pred_eye, average="macro")
    pc_honest_eye = f1_score(y, honest_pred_eye, average=None,
                             labels=list(range(N_CLASSES)), zero_division=0)
    print(f"  NESTED eye:  wF1={wf_honest_eye:.4f}  mF1={mf_honest_eye:.4f}  ({time.time()-t0:.1f}s)")
    print(f"  per-class F1: " + " | ".join(f"{c[:5]}={f:.3f}" for c, f in zip(CLASSES, pc_honest_eye)))
    median_b = np.median(np.array(biases_per_fold_eye), axis=0)
    print(f"  median biases across 44 folds: {median_b}")

    # ========================================================================
    # Part C — person-level LOPO, RAW argmax + leaky + nested
    # ========================================================================
    print("\n" + "=" * 72)
    print("PART C — person-level LOPO (35 groups)")
    print("=" * 72)
    t0 = time.time()
    oof_pers = lopo_lr(X_champ, y, person_groups)
    raw_pred_pers = oof_pers.argmax(axis=1)
    wf_raw_pers = f1_score(y, raw_pred_pers, average="weighted")
    mf_raw_pers = f1_score(y, raw_pred_pers, average="macro")
    print(f"  raw argmax:    wF1={wf_raw_pers:.4f}  mF1={mf_raw_pers:.4f}  ({time.time()-t0:.1f}s)")

    biases_leaky_pers = tune_biases(oof_pers, y)
    leaky_pred_pers = apply_biases(oof_pers, biases_leaky_pers)
    wf_leaky_pers = f1_score(y, leaky_pred_pers, average="weighted")
    mf_leaky_pers = f1_score(y, leaky_pred_pers, average="macro")
    pc_leaky_pers = f1_score(y, leaky_pred_pers, average=None,
                             labels=list(range(N_CLASSES)), zero_division=0)
    print(f"  leaky bias:    wF1={wf_leaky_pers:.4f}  mF1={mf_leaky_pers:.4f}")
    print(f"  per-class F1: " + " | ".join(f"{c[:5]}={f:.3f}" for c, f in zip(CLASSES, pc_leaky_pers)))
    print(f"  leaky biases: {biases_leaky_pers}")

    honest_pred_pers = np.full(n_scans, -1, dtype=int)
    biases_per_fold_pers = []
    t0 = time.time()
    for tr, va in leave_one_patient_out(person_groups):
        inner_biases = tune_biases(oof_pers[tr], y[tr])
        honest_pred_pers[va] = apply_biases(oof_pers[va], inner_biases)
        biases_per_fold_pers.append(inner_biases)
    wf_honest_pers = f1_score(y, honest_pred_pers, average="weighted")
    mf_honest_pers = f1_score(y, honest_pred_pers, average="macro")
    pc_honest_pers = f1_score(y, honest_pred_pers, average=None,
                              labels=list(range(N_CLASSES)), zero_division=0)
    print(f"  NESTED person: wF1={wf_honest_pers:.4f}  mF1={mf_honest_pers:.4f}  ({time.time()-t0:.1f}s)")
    print(f"  per-class F1: " + " | ".join(f"{c[:5]}={f:.3f}" for c, f in zip(CLASSES, pc_honest_pers)))
    median_bp = np.median(np.array(biases_per_fold_pers), axis=0)
    print(f"  median biases across {len(np.unique(person_groups))} folds: {median_bp}")

    # ========================================================================
    # Summary table
    # ========================================================================
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    header = f"{'setting':45s}  {'wF1':>7s}  {'mF1':>7s}"
    print(header)
    print("-" * len(header))
    rows = [
        ("Claim (eye-level leaky bias tune)",       wf_leaky_eye, mf_leaky_eye),
        ("eye-level raw argmax",                    wf_raw_eye,   mf_raw_eye),
        ("eye-level NESTED biases",                 wf_honest_eye, mf_honest_eye),
        ("person-level leaky bias tune",            wf_leaky_pers, mf_leaky_pers),
        ("person-level raw argmax",                 wf_raw_pers,   mf_raw_pers),
        ("person-level NESTED biases",              wf_honest_pers, mf_honest_pers),
    ]
    for name, wf, mf in rows:
        print(f"{name:45s}  {wf:7.4f}  {mf:7.4f}")

    # Confusion matrices for nested person-level (the recommended honest number)
    cm_honest_pers = confusion_matrix(y, honest_pred_pers, labels=list(range(N_CLASSES)))
    print("\nConfusion matrix — person-level NESTED:")
    print(pd.DataFrame(cm_honest_pers, index=CLASSES, columns=CLASSES).to_string())

    # Save artifact
    np.savez(CACHE / "red_team_audit_v2.npz",
             y=y,
             person_groups=person_groups,
             eye_groups=scan_groups_eye,
             oof_eye=oof_eye,
             oof_pers=oof_pers,
             leaky_pred_eye=leaky_pred_eye,
             honest_pred_eye=honest_pred_eye,
             leaky_pred_pers=leaky_pred_pers,
             honest_pred_pers=honest_pred_pers,
             biases_leaky_eye=biases_leaky_eye,
             biases_leaky_pers=biases_leaky_pers,
             biases_per_fold_eye=np.array(biases_per_fold_eye),
             biases_per_fold_pers=np.array(biases_per_fold_pers),
             )
    print(f"\n[saved] {CACHE / 'red_team_audit_v2.npz'}")


if __name__ == "__main__":
    main()
