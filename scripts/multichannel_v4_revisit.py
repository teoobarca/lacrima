"""Revisit multichannel (Height+Amplitude+Phase) DINOv2 RGB encoding as v4 ensemble member.

Wave 6 E7 was rejected because of wide bootstrap CI. Here we honestly fuse
DINOv2-RGB (multichannel) softmax with v4 baseline via geometric mean and
verify with bootstrap.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

CACHE = "/Users/rafael/Programming/teardrop-challenge/cache"
EPS = 1e-9


def main():
    v4 = np.load(f"{CACHE}/v4_oof_predictions.npz", allow_pickle=True)
    mc = np.load(f"{CACHE}/best_multichannel_v2_predictions.npz", allow_pickle=True)

    # v4 OOF
    p_v4 = v4["proba"]  # (240, 5)
    y_v4 = v4["y"]
    persons_v4 = v4["persons"]
    paths_v4 = v4["scan_paths"]

    # multichannel components
    p_dh = mc["P_dinov2_height_pool"]
    p_drgb = mc["P_dinov2_rgb_pool"]  # multichannel
    p_bc = mc["P_biomedclip_tta_height"]
    y_mc = mc["y"]
    groups_mc = mc["groups"]
    paths_mc = mc["tta_paths"]

    # Align via paths
    print(f"v4 n={len(p_v4)}, mc n={len(p_drgb)}")
    if len(p_v4) != len(p_drgb):
        # use intersection
        common = np.intersect1d(paths_v4, paths_mc)
        print(f"  common scans: {len(common)}")
        v4_idx = {p: i for i, p in enumerate(paths_v4)}
        mc_idx = {p: i for i, p in enumerate(paths_mc)}
        idx_v4 = [v4_idx[p] for p in common]
        idx_mc = [mc_idx[p] for p in common]
        p_v4 = p_v4[idx_v4]
        y_v4 = y_v4[idx_v4]
        persons_v4 = persons_v4[idx_v4]
        p_drgb = p_drgb[idx_mc]
    else:
        # check label alignment
        if not np.array_equal(y_v4, y_mc):
            print("  WARNING: label mismatch — try align by paths")

    # Sanity: v4 baseline
    pred_v4 = p_v4.argmax(axis=1)
    f1w_b = f1_score(y_v4, pred_v4, average="weighted")
    f1m_b = f1_score(y_v4, pred_v4, average="macro")
    print(f"\n[baseline v4] wF1={f1w_b:.4f} mF1={f1m_b:.4f}")

    # fusion variants
    variants = []
    for name, weight_rgb in [("v4_geom_rgb_w0.3", 0.3),
                             ("v4_geom_rgb_w0.5", 0.5),
                             ("v4_geom_rgb_w0.7", 0.7),
                             ("v4_x_rgb_50_50", 0.5)]:
        # geometric weighted mean
        log_p = (1 - weight_rgb) * np.log(p_v4 + EPS) + weight_rgb * np.log(p_drgb + EPS)
        p_fused = np.exp(log_p - log_p.max(axis=1, keepdims=True))
        p_fused /= p_fused.sum(axis=1, keepdims=True)
        pred_f = p_fused.argmax(axis=1)
        f1w = f1_score(y_v4, pred_f, average="weighted")
        f1m = f1_score(y_v4, pred_f, average="macro")
        variants.append((name, pred_f, f1w, f1m))
        print(f"[{name}] wF1={f1w:.4f} mF1={f1m:.4f}  Δ={f1w-f1w_b:+.4f}")

    # bootstrap best
    best_name, best_pred, best_w, _ = max(variants, key=lambda x: x[2])
    print(f"\n[best: {best_name}] bootstrap 1000x vs v4 baseline")
    rng = np.random.default_rng(0)
    persons = np.unique(persons_v4)
    deltas = []
    for _ in range(1000):
        sampled = rng.choice(persons, size=len(persons), replace=True)
        mask = np.isin(persons_v4, sampled)
        f_b = f1_score(y_v4[mask], pred_v4[mask], average="weighted", zero_division=0)
        f_v = f1_score(y_v4[mask], best_pred[mask], average="weighted", zero_division=0)
        deltas.append(f_v - f_b)
    d = np.array(deltas)
    print(f"  mean Δ={d.mean():+.4f}  CI95=[{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]")
    print(f"  P(Δ>0)={(d>0).mean():.3f}")


if __name__ == "__main__":
    main()
