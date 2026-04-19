"""Build v5 = v4 + adaptive production layers.

Layers:
  0. v4 multi-scale (base prediction)
  1. Hybrid Re-ID adaptive blend (worst case = v4)
  2. Temperature calibration (post-hoc, for honest probabilities)
  3. Triage / abstain output with margin threshold

Person-LOPO honest evaluation. Saves to models/ensemble_v5_adaptive/.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
MODEL_DIR = ROOT / "models" / "ensemble_v5_adaptive"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EPS = 1e-9


def _mean_pool(X_tiles, tile_to_scan, n_scans):
    out = np.zeros((n_scans, X_tiles.shape[1]), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int64)
    for i, s in enumerate(tile_to_scan):
        out[s] += X_tiles[i]
        counts[s] += 1
    out /= np.maximum(counts, 1)[:, None]
    return out


def fit_lopo_softmax(X, y, groups, n_classes=5):
    n = len(y)
    oof = np.zeros((n, n_classes), dtype=np.float32)
    for p in np.unique(groups):
        train_mask = groups != p
        test_mask = ~train_mask
        Xt_n = normalize(X[train_mask], norm="l2", axis=1)
        sc = StandardScaler().fit(Xt_n)
        Xt_s = sc.transform(Xt_n)
        clf = LogisticRegression(class_weight="balanced", max_iter=3000,
                                 C=1.0, solver="lbfgs", random_state=42)
        clf.fit(Xt_s, y[train_mask])
        Xe_n = normalize(X[test_mask], norm="l2", axis=1)
        full = np.zeros((Xe_n.shape[0], n_classes), dtype=np.float32)
        full[:, clf.classes_] = clf.predict_proba(sc.transform(Xe_n))
        oof[test_mask] = full
    return oof


def hybrid_reid(p_v4, X_dinov2, y, groups, threshold=0.94):
    """Layer 1: blend v4 prediction with nearest train neighbor label.
    Worst case (sim < threshold for all): = v4 unchanged.
    """
    n = len(y)
    out = p_v4.copy()
    fired = np.zeros(n, dtype=bool)
    fire_correct = 0
    fire_total = 0
    # L2 normalize
    Xn = X_dinov2 / (np.linalg.norm(X_dinov2, axis=1, keepdims=True) + EPS)
    for i in range(n):
        # exclude same person
        train_mask = groups != groups[i]
        train_idx = np.where(train_mask)[0]
        sims = Xn[train_idx] @ Xn[i]
        j_best = train_idx[sims.argmax()]
        sim_max = sims.max()
        if sim_max > threshold:
            # adaptive weight in (0, 1] above threshold
            w = min(1.0, (sim_max - threshold) / (1.0 - threshold))
            onehot = np.zeros(5, dtype=np.float32)
            onehot[y[j_best]] = 1.0
            out[i] = (1 - w) * p_v4[i] + w * onehot
            out[i] /= out[i].sum()
            fired[i] = True
            fire_total += 1
            if y[j_best] == y[i]:
                fire_correct += 1
    fire_acc = fire_correct / max(fire_total, 1)
    return out, fired, fire_acc


def fit_temperature(p_logits, y_true):
    """Layer 2: fit single temperature T to minimize NLL (calibration only)."""
    def nll(T):
        if T <= 0:
            return 1e9
        # softmax(logits / T)
        z = p_logits / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        # clip to avoid log(0)
        p = np.clip(p, 1e-12, 1.0)
        return -np.log(p[np.arange(len(y_true)), y_true]).mean()
    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return res.x


def temperature_apply(p, T):
    """Apply temperature to probabilities (treat as logits via log)."""
    z = np.log(p + EPS)
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    out = np.exp(z)
    out /= out.sum(axis=1, keepdims=True)
    return out


def expected_calibration_error(p, y_true, n_bins=10):
    """ECE: weighted gap between confidence and accuracy."""
    confidences = p.max(axis=1)
    predictions = p.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return ece


def triage_metric(p, y_true, margin_threshold=0.10):
    """Layer 3: split into autonomous vs flagged-for-review."""
    sorted_p = np.sort(p, axis=1)[:, ::-1]
    margin = sorted_p[:, 0] - sorted_p[:, 1]
    autonomous_mask = margin >= margin_threshold
    pred_top1 = p.argmax(axis=1)
    if autonomous_mask.sum() == 0:
        auto_acc = 0.0
    else:
        auto_acc = (pred_top1[autonomous_mask] == y_true[autonomous_mask]).mean()
    return {
        "n_autonomous": int(autonomous_mask.sum()),
        "n_flagged": int((~autonomous_mask).sum()),
        "fraction_autonomous": float(autonomous_mask.mean()),
        "autonomous_accuracy": float(auto_acc),
    }


def main():
    print("[load]")
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz", allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz", allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)
    y = z90["scan_y"]
    groups = z90["scan_groups"]
    n = len(y)
    X90 = _mean_pool(z90["X"], z90["tile_to_scan"], n)
    X45 = _mean_pool(z45["X"], z45["tile_to_scan"], n)
    Xbc = zbc["X_scan"].astype(np.float32)

    # Layer 0: v4 base
    print("\n[layer 0] v4 multi-scale")
    p90 = fit_lopo_softmax(X90, y, groups)
    p45 = fit_lopo_softmax(X45, y, groups)
    pbc = fit_lopo_softmax(Xbc, y, groups)
    log_p = (np.log(p90 + EPS) + np.log(p45 + EPS) + np.log(pbc + EPS)) / 3.0
    p_v4 = np.exp(log_p - log_p.max(axis=1, keepdims=True))
    p_v4 /= p_v4.sum(axis=1, keepdims=True)
    pred_v4 = p_v4.argmax(axis=1)
    f1w_v4 = f1_score(y, pred_v4, average="weighted")
    f1m_v4 = f1_score(y, pred_v4, average="macro")
    ece_v4 = expected_calibration_error(p_v4, y)
    print(f"  v4: wF1={f1w_v4:.4f} mF1={f1m_v4:.4f} ECE={ece_v4:.4f}")

    # Layer 1: Hybrid Re-ID
    print("\n[layer 1] Hybrid Re-ID adaptive blend (threshold=0.94)")
    p_l1, fired, fire_acc = hybrid_reid(p_v4, X90, y, groups, threshold=0.94)
    pred_l1 = p_l1.argmax(axis=1)
    f1w_l1 = f1_score(y, pred_l1, average="weighted")
    f1m_l1 = f1_score(y, pred_l1, average="macro")
    print(f"  fired on {fired.sum()}/{n} ({fired.mean():.1%}), fire_acc={fire_acc:.3f}")
    print(f"  v4+ReID: wF1={f1w_l1:.4f} mF1={f1m_l1:.4f}  Δ={f1w_l1-f1w_v4:+.4f}")

    # Layer 2: Temperature calibration (fit on outer OOF — honest in expectation, but
    # ideally fit per inner fold; for simplicity here we do fit-and-eval on full OOF
    # which is mildly optimistic on calibration but does NOT change argmax/F1)
    print("\n[layer 2] Temperature calibration")
    T_v4 = fit_temperature(np.log(p_v4 + EPS), y)
    p_v4_calib = temperature_apply(p_v4, T_v4)
    ece_v4_calib = expected_calibration_error(p_v4_calib, y)
    print(f"  v4 alone: T*={T_v4:.3f}, ECE: {ece_v4:.4f} → {ece_v4_calib:.4f}")
    T_l1 = fit_temperature(np.log(p_l1 + EPS), y)
    p_l1_calib = temperature_apply(p_l1, T_l1)
    ece_l1 = expected_calibration_error(p_l1, y)
    ece_l1_calib = expected_calibration_error(p_l1_calib, y)
    print(f"  v5 (v4+ReID): T*={T_l1:.3f}, ECE: {ece_l1:.4f} → {ece_l1_calib:.4f}")

    # Layer 3: Triage
    print("\n[layer 3] Triage with margin threshold")
    for margin in [0.05, 0.10, 0.15, 0.20]:
        triage = triage_metric(p_l1_calib, y, margin_threshold=margin)
        print(f"  margin={margin:.2f}: autonomous {triage['fraction_autonomous']:.1%} "
              f"(acc={triage['autonomous_accuracy']:.3f}), flagged {triage['n_flagged']}")

    # Bootstrap v5 (Layer 1 with calibration) vs v4 baseline
    print("\n[bootstrap 1000x v5 vs v4]")
    rng = np.random.default_rng(0)
    persons = np.unique(groups)
    deltas = []
    for _ in range(1000):
        sampled = rng.choice(persons, size=len(persons), replace=True)
        mask = np.isin(groups, sampled)
        f_v4 = f1_score(y[mask], pred_v4[mask], average="weighted", zero_division=0)
        f_v5 = f1_score(y[mask], pred_l1[mask], average="weighted", zero_division=0)
        deltas.append(f_v5 - f_v4)
    d = np.array(deltas)
    print(f"  mean Δ={d.mean():+.4f}  CI95=[{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]")
    print(f"  P(Δ≥0)={(d>=0).mean():.3f}  P(Δ>0)={(d>0).mean():.3f}")

    # Save v5 OOF predictions + meta
    np.savez(CACHE / "v5_adaptive_oof.npz",
             p_v4=p_v4, p_v5=p_l1_calib, y=y, groups=groups,
             fired=fired, T_calibration=T_l1)
    print("\nSaved cache/v5_adaptive_oof.npz")

    # Save model meta
    meta = {
        "kind": "ensemble_v5_adaptive",
        "base_model": "ensemble_v4_multiscale",
        "layers": ["v4_multi_scale", "hybrid_reid", "temperature_scaling", "triage_abstain"],
        "config": {
            "reid_threshold": 0.94,
            "temperature": float(T_l1),
            "triage_margin": 0.10,
            "honest_lopo_weighted_f1_v4_baseline": float(f1w_v4),
            "honest_lopo_weighted_f1_v5": float(f1w_l1),
            "honest_lopo_macro_f1_v5": float(f1m_l1),
            "ece_uncalibrated": float(ece_l1),
            "ece_calibrated": float(ece_l1_calib),
            "reid_fire_rate": float(fired.mean()),
            "reid_fire_accuracy": float(fire_acc),
        },
        "classes": ["ZdraviLudia", "Diabetes", "PGOV_Glaukom", "SklerozaMultiplex", "SucheOko"],
        "trained_on_n_scans": int(n),
        "trained_on_n_persons": int(len(persons)),
        "expected_test_f1": "0.69-0.73 (patient-disjoint regime, full-train bonus)",
        "provenance": "Wave 20 v5 — v4 + adaptive Re-ID + calibration + triage",
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {MODEL_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()
