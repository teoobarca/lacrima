"""Failure-mode analysis for the champion TTA ensemble.

1. Reproduce OOF predictions under person-level LOPO with 2-encoder LR ensemble
   (DINOv2-B TTA + BiomedCLIP TTA), averaged softmax, raw argmax.
2. Cluster misclassifications into Modes A/B/C/D by confidence and rank of
   true class.
3. Deep-dive Mode B: nearest-neighbour retrieval in DINOv2-B tiled (non-TTA)
   scan-mean space; handcrafted-feature distance to class means.
4. Per-class error patterns.

Writes:
  reports/error_cases.csv
  reports/ERROR_ANALYSIS.md  (driven by this script's printed artefacts; the
                              human-written prose lives in the .md itself)
  reports/error_analysis.json  (raw numbers for the .md)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import CLASSES  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
N_CLASSES = len(CLASSES)


# ---------------------------------------------------------------------------
# Person-LOPO predict_proba (same LR hyper-params as tta_experiment.py)
# ---------------------------------------------------------------------------

def lopo_predict_proba(X, y, groups):
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


def align_to(Xd, paths_d, Xb, paths_b):
    """Align Xb rows to match paths_d ordering."""
    path_to_i = {p: i for i, p in enumerate(paths_b)}
    order = np.array([path_to_i[p] for p in paths_d])
    return Xb[order]


def main():
    print("=" * 72)
    print("Failure-mode analysis for the shipped TTA ensemble")
    print("=" * 72)

    # --- Load TTA embeddings --------------------------------------------------
    dino = np.load(CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz",
                   allow_pickle=True)
    bmc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz",
                  allow_pickle=True)

    Xd = dino["X_scan"]
    y = dino["scan_y"].astype(int)
    groups = dino["scan_groups"].astype(str)
    paths = np.array([str(p) for p in dino["scan_paths"]])

    Xb = align_to(Xd, list(paths),
                  bmc["X_scan"], [str(p) for p in bmc["scan_paths"]])
    # Sanity: aligned labels must match
    y_b_aligned = bmc["scan_y"].astype(int)
    # align y_b to paths ordering
    path_to_i_b = {str(p): i for i, p in enumerate(bmc["scan_paths"])}
    order = np.array([path_to_i_b[p] for p in paths])
    assert np.array_equal(y, y_b_aligned[order]), "label mismatch"

    print(f"Loaded: DINOv2-B TTA {Xd.shape}, BiomedCLIP TTA {Xb.shape}, "
          f"{len(np.unique(groups))} persons.")

    # --- Reproduce LOPO -------------------------------------------------------
    print("\n[1/4] Reproducing 2-encoder TTA ensemble (person-LOPO)...")
    P_d = lopo_predict_proba(Xd, y, groups)
    P_b = lopo_predict_proba(Xb, y, groups)
    P_ens = 0.5 * (P_d + P_b)
    pred = P_ens.argmax(axis=1)

    f1w = f1_score(y, pred, average="weighted", zero_division=0)
    f1m = f1_score(y, pred, average="macro", zero_division=0)
    print(f"  weighted F1 = {f1w:.4f}  (target ~0.6458)")
    print(f"  macro F1    = {f1m:.4f}")
    print(classification_report(y, pred, target_names=CLASSES, zero_division=0))

    # --- Per-scan error frame -------------------------------------------------
    print("\n[2/4] Per-scan confidence + failure-mode assignment...")
    maxprob = P_ens.max(axis=1)
    # other-class probability = max over all classes != true class
    other_prob = np.zeros_like(maxprob)
    for i in range(len(y)):
        p = P_ens[i].copy()
        p[y[i]] = -1.0
        other_prob[i] = p.max()

    # rank of true class (1 = top-1 = correct, ..., 5 = last)
    ranks = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        order_i = np.argsort(-P_ens[i])
        ranks[i] = int(np.where(order_i == y[i])[0][0]) + 1

    rows = []
    for i in range(len(y)):
        correct = int(pred[i] == y[i])
        # Classify failure mode for wrong scans only
        if correct:
            mode = "OK"
        else:
            mp = maxprob[i]
            r = ranks[i]
            if r == 2:
                mode = "C_near_miss"  # top-2 contains truth
            elif r >= 4:
                mode = "D_catastrophic"  # truth outside top-3
            elif mp < 0.40:
                mode = "A_low_conf"
            elif mp > 0.70:
                mode = "B_high_conf"
            else:
                mode = "E_mid_conf"  # not A or B
        rows.append(dict(
            path=str(paths[i]),
            scan_name=Path(paths[i]).name,
            person=groups[i],
            true_class=CLASSES[y[i]],
            pred_class=CLASSES[pred[i]],
            correct=correct,
            maxprob=float(maxprob[i]),
            true_class_prob=float(P_ens[i, y[i]]),
            other_class_prob=float(other_prob[i]),
            second_class=CLASSES[int(np.argsort(-P_ens[i])[1])],
            second_prob=float(np.sort(-P_ens[i])[1] * -1),
            rank_of_true=int(ranks[i]),
            mode=mode,
            p_ZdraviLudia=float(P_ens[i, 0]),
            p_Diabetes=float(P_ens[i, 1]),
            p_PGOV_Glaukom=float(P_ens[i, 2]),
            p_SklerozaMultiplex=float(P_ens[i, 3]),
            p_SucheOko=float(P_ens[i, 4]),
        ))
    df = pd.DataFrame(rows)
    out_csv = REPORTS / "error_cases.csv"
    df.to_csv(out_csv, index=False)
    print(f"  -> {out_csv}")

    # Mode counts
    mode_counts = df["mode"].value_counts().to_dict()
    print("  mode counts:", mode_counts)

    # --- Confusion matrix -----------------------------------------------------
    cm = confusion_matrix(y, pred, labels=list(range(N_CLASSES)))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("                  " + " ".join(f"{c[:5]:>5}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c:17s}" + " ".join(f"{cm[i, j]:>5d}" for j in range(N_CLASSES)))

    # --- [3/4] Mode B deep-dive: nearest neighbours in DINOv2-B tiled space --
    print("\n[3/4] Mode B deep-dive: k-NN in DINOv2-B tiled (non-TTA) scan-mean...")
    tiled = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz",
                    allow_pickle=True)
    X_tiles = tiled["X"]
    tile_to_scan = tiled["tile_to_scan"]
    t_paths = [str(p) for p in tiled["scan_paths"]]
    # Build scan-mean embedding from tiles
    n_scans = len(t_paths)
    D = X_tiles.shape[1]
    X_scan_tiled = np.zeros((n_scans, D), dtype=np.float32)
    for si in range(n_scans):
        msk = tile_to_scan == si
        if msk.any():
            X_scan_tiled[si] = X_tiles[msk].mean(axis=0)

    # Reorder to paths-of-interest ordering
    path_to_t = {p: i for i, p in enumerate(t_paths)}
    order_t = np.array([path_to_t[p] for p in paths])
    X_scan_tiled = X_scan_tiled[order_t]

    # Load handcrafted features in same order
    hcdf = pd.read_parquet(CACHE / "features_handcrafted.parquet")
    hcdf["raw"] = hcdf["raw"].astype(str)
    hc_idx = {r: i for i, r in enumerate(hcdf["raw"].tolist())}
    hc_order = np.array([hc_idx[p] for p in paths])
    feat_cols = [c for c in hcdf.columns if c not in ("raw", "cls", "label", "patient")]
    Xhc = hcdf[feat_cols].to_numpy(dtype=np.float64)[hc_order]

    # Compute class means (on Xhc)
    class_means = np.zeros((N_CLASSES, Xhc.shape[1]))
    class_stds = np.zeros((N_CLASSES, Xhc.shape[1]))
    for c in range(N_CLASSES):
        class_means[c] = Xhc[y == c].mean(axis=0)
        class_stds[c] = Xhc[y == c].std(axis=0) + 1e-9

    # Normalize Xhc for distance comparison (robust Z via global mean/std)
    gmean = Xhc.mean(axis=0)
    gstd = Xhc.std(axis=0) + 1e-9
    Xhc_z = (Xhc - gmean) / gstd
    class_means_z = (class_means - gmean) / gstd

    # L2-normalise DINO scan-mean for cosine dist
    def l2n(M):
        nrm = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
        return M / nrm
    Xn = l2n(X_scan_tiled)

    # Mode B cases
    mode_b_idx = df.index[df["mode"] == "B_high_conf"].tolist()
    mode_b_idx = sorted(mode_b_idx, key=lambda i: -df.loc[i, "maxprob"])
    print(f"  {len(mode_b_idx)} Mode B scans")

    mode_b_details = []
    for i in mode_b_idx:
        # Nearest 5 scans from OTHER persons
        sim = Xn @ Xn[i]
        # Mask: different person
        diff_person = groups != groups[i]
        cand = np.where(diff_person)[0]
        top = cand[np.argsort(-sim[cand])[:5]]
        nn_classes = [CLASSES[y[j]] for j in top]
        nn_sims = [float(sim[j]) for j in top]
        nn_persons = [groups[j] for j in top]
        nn_paths = [Path(paths[j]).name for j in top]

        # Handcrafted distance to class means (scaled Euclidean in z-space)
        def d_to(c):
            return float(np.linalg.norm(Xhc_z[i] - class_means_z[c]))
        d_true = d_to(y[i])
        d_pred = d_to(pred[i])
        true_cls = CLASSES[y[i]]
        pred_cls = CLASSES[pred[i]]

        # Majority class among NN
        from collections import Counter
        maj = Counter(nn_classes).most_common(1)[0]
        maj_cls, maj_count = maj

        # Why is it a mode B error?
        nn_matches_pred = sum(1 for c in nn_classes if c == pred_cls)
        nn_matches_true = sum(1 for c in nn_classes if c == true_cls)

        mode_b_details.append(dict(
            i=int(i),
            scan=Path(paths[i]).name,
            person=str(groups[i]),
            true=true_cls,
            pred=pred_cls,
            maxprob=float(df.loc[i, "maxprob"]),
            true_prob=float(df.loc[i, "true_class_prob"]),
            rank_true=int(df.loc[i, "rank_of_true"]),
            nn_classes=nn_classes,
            nn_sims=nn_sims,
            nn_persons=nn_persons,
            nn_scans=nn_paths,
            nn_majority=maj_cls,
            nn_maj_count=int(maj_count),
            nn_matches_pred=int(nn_matches_pred),
            nn_matches_true=int(nn_matches_true),
            hc_dist_to_true_mean=d_true,
            hc_dist_to_pred_mean=d_pred,
            hc_closer_to_pred=bool(d_pred < d_true),
        ))

    # --- [4/4] Per-class patterns --------------------------------------------
    print("\n[4/4] Per-class error patterns...")
    per_class = []
    for c, name in enumerate(CLASSES):
        mask = y == c
        n = int(mask.sum())
        correct = int(((pred == y) & mask).sum())
        errs = df[(df["true_class"] == name) & (df["correct"] == 0)]
        if len(errs) == 0:
            most_common = ("", 0, 0.0)
            conf_break = {}
        else:
            target_counts = errs["pred_class"].value_counts()
            tgt = target_counts.index[0]
            cnt = int(target_counts.iloc[0])
            most_common = (tgt, cnt, float(cnt / len(errs)))
            conf_break = target_counts.to_dict()
        per_class.append(dict(
            class_name=name,
            n_total=n,
            n_correct=correct,
            n_errors=int(n - correct),
            recall=float(correct / n) if n else 0.0,
            most_confused_with=most_common[0],
            most_confused_count=most_common[1],
            most_confused_share=most_common[2],
            error_breakdown=conf_break,
        ))

    # Summaries
    summary = dict(
        weighted_f1=float(f1w),
        macro_f1=float(f1m),
        mode_counts=mode_counts,
        confusion_matrix=cm.tolist(),
        classes=CLASSES,
        per_class=per_class,
        mode_b_details=mode_b_details,
    )
    out_json = REPORTS / "error_analysis.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {out_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
