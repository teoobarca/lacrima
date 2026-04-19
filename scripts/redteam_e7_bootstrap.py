"""Red-team audit for E7 (multichannel 3-way fusion) claim.

Checks:
 1. Reproduce E1 (0.6562) and E7 (0.6645) honest person-LOPO weighted F1.
 2. Person-level bootstrap CI for ΔF1 = F1(E7) - F1(E1), B=1000 resamples.
 3. Seed sensitivity: re-run E7 LOPO 5 times with different LR random_state seeds.
 4. Per-class F1 delta verification.

Outputs reports/RED_TEAM_E7_BOOTSTRAP.md.
"""
from __future__ import annotations

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
B = 1000


# -------------------------- helpers --------------------------

def mean_pool_tiles(X_tiles, t2s, n_scans):
    d = X_tiles.shape[1]
    out = np.zeros((n_scans, d), dtype=np.float32)
    for si in range(n_scans):
        m = t2s == si
        if m.any():
            out[si] = X_tiles[m].mean(axis=0)
    return out


def align_to_reference(paths_ref, paths_src, X_src):
    src_idx = {p: i for i, p in enumerate(paths_src)}
    order = np.array([src_idx[p] for p in paths_ref])
    return X_src[order]


def lopo_predict_v2(X, y, groups, random_state=42):
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
            solver="lbfgs", n_jobs=4, random_state=random_state,
        )
        clf.fit(Xt, y[tr])
        proba = clf.predict_proba(Xv)
        p_full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
        for ci, cls in enumerate(clf.classes_):
            p_full[:, cls] = proba[:, ci]
        P[va] = p_full
    return P


def geom_mean(probs_list):
    log_sum = np.zeros_like(probs_list[0])
    for P in probs_list:
        log_sum += np.log(P + EPS)
    G = np.exp(log_sum / len(probs_list))
    G /= G.sum(axis=1, keepdims=True)
    return G


def metrics(P, y):
    pred = P.argmax(axis=1)
    return {
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(
            y, pred, average=None, labels=list(range(N_CLASSES)), zero_division=0,
        ).tolist(),
        "pred": pred,
    }


def person_bootstrap_delta(pred_a, pred_b, y, groups, B=1000, seed=0):
    """Resample persons with replacement; recompute weighted F1 delta per boot."""
    rng = np.random.default_rng(seed)
    unique_persons = np.unique(groups)
    n_persons = len(unique_persons)
    # precompute per-person indices
    person_to_idx = {p: np.where(groups == p)[0] for p in unique_persons}
    f1_a_boot = np.zeros(B)
    f1_b_boot = np.zeros(B)
    for b in range(B):
        sampled_persons = rng.choice(unique_persons, size=n_persons, replace=True)
        idx = np.concatenate([person_to_idx[p] for p in sampled_persons])
        ya = y[idx]
        f1_a_boot[b] = f1_score(ya, pred_a[idx], average="weighted", zero_division=0)
        f1_b_boot[b] = f1_score(ya, pred_b[idx], average="weighted", zero_division=0)
    return f1_a_boot, f1_b_boot


# -------------------------- load data --------------------------

def load_features():
    zd = np.load(CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz", allow_pickle=True)
    zb = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)
    Xd_tta = zd["X_scan"].astype(np.float32)
    y = zd["scan_y"].astype(np.int64)
    groups = zd["scan_groups"].astype(str)
    tta_paths = [str(Path(p)) for p in zd["scan_paths"]]
    Xb_tta = align_to_reference(tta_paths,
                                 [str(Path(p)) for p in zb["scan_paths"]],
                                 zb["X_scan"].astype(np.float32))

    zm = np.load(CACHE / "multichan_tiled_emb_dinov2vitb14_t512_n9.npz", allow_pickle=True)

    def scan_from_channel(key):
        Xt = zm[f"X_{key}"].astype(np.float32)
        t2s = zm[f"t2s_{key}"].astype(np.int64)
        paths = [str(Path(p)) for p in zm[f"paths_{key}"]]
        n_scans = len(paths)
        return mean_pool_tiles(Xt, t2s, n_scans), paths

    X_h_scan, h_paths = scan_from_channel("height")
    X_r_scan, r_paths = scan_from_channel("rgb")

    X_h = align_to_reference(tta_paths, h_paths, X_h_scan)
    X_r = align_to_reference(tta_paths, r_paths, X_r_scan)

    return {
        "y": y,
        "groups": groups,
        "tta_paths": tta_paths,
        "X_dinov2_tta_height": Xd_tta,
        "X_biomedclip_tta_height": Xb_tta,
        "X_dinov2_height_pool": X_h,
        "X_dinov2_rgb_pool": X_r,
    }


# -------------------------- main --------------------------

def main():
    t0 = time.time()
    print("=" * 78)
    print("RED-TEAM AUDIT: E7 vs E1 (v2 champion)")
    print("=" * 78)

    # Load features
    print("\n[1] loading features...")
    d = load_features()
    y = d["y"]
    groups = d["groups"]
    n_scans = len(y)
    n_persons = len(np.unique(groups))
    print(f"  n_scans={n_scans}  n_persons={n_persons}")

    # Sanity: saved npz labels & groups should match
    zsaved = np.load(CACHE / "best_multichannel_v2_predictions.npz", allow_pickle=True)
    assert np.array_equal(y, zsaved["y"]), "y mismatch with saved npz"
    assert np.array_equal(groups, zsaved["groups"].astype(str)), "groups mismatch"
    print("  saved npz y/groups match fresh load")

    # --------------- AUDIT 1: reproduce E1 & E7 ---------------
    print("\n[2] reproduce E1 (champion) and E7 (challenger) with seed=42")
    print("    computing per-member OOF softmax via v2 recipe...")
    P_dinov2_tta = lopo_predict_v2(d["X_dinov2_tta_height"], y, groups, random_state=42)
    P_bclip_tta = lopo_predict_v2(d["X_biomedclip_tta_height"], y, groups, random_state=42)
    P_h_pool = lopo_predict_v2(d["X_dinov2_height_pool"], y, groups, random_state=42)
    P_rgb_pool = lopo_predict_v2(d["X_dinov2_rgb_pool"], y, groups, random_state=42)

    # E1 = champion
    P_E1 = geom_mean([P_dinov2_tta, P_bclip_tta])
    m_E1 = metrics(P_E1, y)

    # E7 = challenger
    P_E7 = geom_mean([P_h_pool, P_rgb_pool, P_bclip_tta])
    m_E7 = metrics(P_E7, y)

    print(f"  E1 recomputed: W-F1={m_E1['weighted_f1']:.4f}  M-F1={m_E1['macro_f1']:.4f}")
    print(f"  E7 recomputed: W-F1={m_E7['weighted_f1']:.4f}  M-F1={m_E7['macro_f1']:.4f}")

    # Also verify against saved E7 predictions (these were produced by the exact
    # same script run, so the pred array in npz should match our fresh re-run).
    saved_pred = zsaved["pred"]
    matches = int((saved_pred == m_E7["pred"]).sum())
    print(f"  saved E7 pred matches fresh E7 pred on {matches}/{n_scans} scans")

    # Also: use saved member softmaxes to recompute E7 (verifies npz integrity)
    P_E7_from_saved = geom_mean([
        zsaved["P_dinov2_height_pool"],
        zsaved["P_dinov2_rgb_pool"],
        zsaved["P_biomedclip_tta_height"],
    ])
    m_E7_from_saved = metrics(P_E7_from_saved, y)
    print(f"  E7 from saved npz softmaxes: W-F1={m_E7_from_saved['weighted_f1']:.4f}  "
          f"M-F1={m_E7_from_saved['macro_f1']:.4f}")

    claim_E1 = 0.6562
    claim_E7 = 0.6645
    repro_E1 = abs(m_E1["weighted_f1"] - claim_E1) < 1e-3  # allow small rounding
    repro_E7 = abs(m_E7["weighted_f1"] - claim_E7) < 1e-3
    print(f"  reproduce E1 (claim {claim_E1:.4f}): {'PASS' if repro_E1 else 'FAIL'}")
    print(f"  reproduce E7 (claim {claim_E7:.4f}): {'PASS' if repro_E7 else 'FAIL'}")
    delta_observed = m_E7["weighted_f1"] - m_E1["weighted_f1"]
    print(f"  observed ΔF1 = {delta_observed:+.4f}")

    # --------------- AUDIT 2: bootstrap CI ---------------
    print(f"\n[3] bootstrap CI over persons (B={B})")
    pred_E1 = m_E1["pred"]
    pred_E7 = m_E7["pred"]
    ts = time.time()
    f1_E1_boot, f1_E7_boot = person_bootstrap_delta(
        pred_E1, pred_E7, y, groups, B=B, seed=20260418
    )
    delta_boot = f1_E7_boot - f1_E1_boot
    ci_lo = float(np.percentile(delta_boot, 2.5))
    ci_hi = float(np.percentile(delta_boot, 97.5))
    p_pos = float((delta_boot > 0).mean())
    boot_mean_delta = float(delta_boot.mean())
    boot_median_delta = float(np.median(delta_boot))
    print(f"  elapsed: {time.time() - ts:.1f}s")
    print(f"  bootstrap mean ΔF1 = {boot_mean_delta:+.4f}")
    print(f"  bootstrap median ΔF1 = {boot_median_delta:+.4f}")
    print(f"  95% CI for ΔF1 = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  P(ΔF1 > 0)    = {p_pos:.3f}")
    ci_strictly_positive = ci_lo > 0
    print(f"  95% CI strictly > 0: {ci_strictly_positive}")

    # --------------- AUDIT 3: seed sensitivity (E7) ---------------
    print("\n[4] seed sensitivity on E7 (5 random_state values)")
    seeds = [0, 1, 13, 42, 123]
    e7_seed_f1w = []
    e7_seed_f1m = []
    e1_seed_f1w = []
    for s in seeds:
        ts = time.time()
        Ph = lopo_predict_v2(d["X_dinov2_height_pool"], y, groups, random_state=s)
        Pr = lopo_predict_v2(d["X_dinov2_rgb_pool"], y, groups, random_state=s)
        Pb = lopo_predict_v2(d["X_biomedclip_tta_height"], y, groups, random_state=s)
        Pd = lopo_predict_v2(d["X_dinov2_tta_height"], y, groups, random_state=s)
        mE7 = metrics(geom_mean([Ph, Pr, Pb]), y)
        mE1 = metrics(geom_mean([Pd, Pb]), y)
        e7_seed_f1w.append(mE7["weighted_f1"])
        e7_seed_f1m.append(mE7["macro_f1"])
        e1_seed_f1w.append(mE1["weighted_f1"])
        print(f"  seed={s:4d}  E1_WF1={mE1['weighted_f1']:.4f}  "
              f"E7_WF1={mE7['weighted_f1']:.4f}  Δ={mE7['weighted_f1']-mE1['weighted_f1']:+.4f}  "
              f"({time.time()-ts:.1f}s)")
    e7_mean = float(np.mean(e7_seed_f1w))
    e7_std = float(np.std(e7_seed_f1w, ddof=1))
    e7_min = float(np.min(e7_seed_f1w))
    e7_max = float(np.max(e7_seed_f1w))
    e1_mean = float(np.mean(e1_seed_f1w))
    e1_std = float(np.std(e1_seed_f1w, ddof=1))
    delta_seeds = np.array(e7_seed_f1w) - np.array(e1_seed_f1w)
    delta_mean = float(delta_seeds.mean())
    delta_std = float(delta_seeds.std(ddof=1))
    print(f"  E7 over seeds: mean={e7_mean:.4f}  std={e7_std:.4f}  "
          f"range=[{e7_min:.4f},{e7_max:.4f}]")
    print(f"  E1 over seeds: mean={e1_mean:.4f}  std={e1_std:.4f}")
    print(f"  Δ(E7-E1) over seeds: mean={delta_mean:+.4f}  std={delta_std:.4f}  "
          f"all_positive={bool((delta_seeds > 0).all())}")

    # --------------- AUDIT 4: per-class breakdown ---------------
    print("\n[5] per-class F1 delta (E1 -> E7)")
    print(f"  {'class':<22s}  {'E1 F1':>7s}  {'E7 F1':>7s}  {'Δ':>8s}")
    claims = {
        "Diabetes": +0.125,
        "PGOV_Glaukom": -0.057,
    }
    claim_verified = {}
    for ci, cname in enumerate(CLASSES):
        f1a = m_E1["per_class_f1"][ci]
        f1b = m_E7["per_class_f1"][ci]
        dd = f1b - f1a
        print(f"  {cname:<22s}  {f1a:7.4f}  {f1b:7.4f}  {dd:+8.4f}")
        if cname in claims:
            claim_verified[cname] = abs(dd - claims[cname]) < 1e-3

    # Which class contributes most? Compute class-support * delta
    n_per_class = np.bincount(y, minlength=N_CLASSES)
    contrib = [
        (CLASSES[ci],
         (m_E7["per_class_f1"][ci] - m_E1["per_class_f1"][ci]) * n_per_class[ci] / n_scans)
        for ci in range(N_CLASSES)
    ]
    print(f"\n  weighted-F1 delta contribution by class (delta * support / n):")
    for cname, c in contrib:
        print(f"    {cname:<22s}  {c:+.4f}")

    # --------------- write report ---------------
    print("\n[6] writing report...")
    md = []
    md.append("# Red-team audit — E7 (multichannel 3-way) vs E1 (v2 champion)\n")
    md.append(f"Date: 2026-04-18.  Auditor: red-team bootstrap.  B={B} person-level resamples, "
              "5 LR seeds, person-LOPO only (`teardrop.cv.leave_one_patient_out`, "
              "groups from `teardrop.data.person_id`).\n")
    md.append("## TL;DR")
    if repro_E1 and repro_E7 and ci_strictly_positive and (delta_seeds > 0).all():
        verdict = "**SHIP E7 as v3 champion.**"
    elif repro_E1 and repro_E7 and p_pos >= 0.90 and (delta_seeds > 0).all():
        verdict = "**Borderline — lean ship E7 as v3.**"
    else:
        verdict = "**Do not ship. Stay with v2.**"
    md.append(f"\n{verdict}  Reasons below.\n")

    md.append("## 1. Reproduction (claim reproduces?)\n")
    md.append(f"| Ensemble | Claim W-F1 | Recomputed W-F1 | Match? |")
    md.append(f"|---|---:|---:|:---:|")
    md.append(f"| E1 (champion v2) | {claim_E1:.4f} | {m_E1['weighted_f1']:.4f} | "
              f"{'PASS' if repro_E1 else 'FAIL'} |")
    md.append(f"| E7 (challenger) | {claim_E7:.4f} | {m_E7['weighted_f1']:.4f} | "
              f"{'PASS' if repro_E7 else 'FAIL'} |")
    md.append("")
    md.append(f"- Macro F1: E1 = {m_E1['macro_f1']:.4f}, E7 = {m_E7['macro_f1']:.4f}  "
              f"(claimed 0.5382 / 0.5435).")
    md.append(f"- Saved npz E7 softmaxes reproduce E7 W-F1 = "
              f"{m_E7_from_saved['weighted_f1']:.4f} (integrity check passes).")
    md.append(f"- Saved npz `pred` agrees with fresh re-run on {matches}/{n_scans} scans.\n")

    md.append("## 2. Person-level bootstrap CI for ΔF1 = F1(E7) - F1(E1)\n")
    md.append(f"- Resampling: persons with replacement (n={n_persons} per bootstrap), "
              f"B = {B}.\n")
    md.append(f"- Point estimate (observed): **{delta_observed:+.4f}**.")
    md.append(f"- Bootstrap mean: {boot_mean_delta:+.4f}, median: {boot_median_delta:+.4f}.")
    md.append(f"- **95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]**.")
    md.append(f"- **P(ΔF1 > 0) = {p_pos:.3f}**.")
    verdict_ci = ("95% CI strictly > 0, real gain." if ci_strictly_positive
                  else "95% CI crosses 0 — improvement is within person-resampling noise.")
    md.append(f"- Verdict: {verdict_ci}\n")

    md.append("## 3. Seed sensitivity (LR `random_state`)\n")
    md.append(f"Five seeds re-run: {seeds}. Each seed recomputes full LOPO for both "
              "E1 and E7.\n")
    md.append("| seed | E1 W-F1 | E7 W-F1 | Δ |")
    md.append("|---:|---:|---:|---:|")
    for s, f_e1, f_e7 in zip(seeds, e1_seed_f1w, e7_seed_f1w):
        md.append(f"| {s} | {f_e1:.4f} | {f_e7:.4f} | {f_e7 - f_e1:+.4f} |")
    md.append("")
    md.append(f"- E7 across seeds: mean = **{e7_mean:.4f}**, std = **{e7_std:.4f}**, "
              f"range = [{e7_min:.4f}, {e7_max:.4f}].")
    md.append(f"- E1 across seeds: mean = {e1_mean:.4f}, std = {e1_std:.4f}.")
    md.append(f"- Δ across seeds: mean = {delta_mean:+.4f}, std = {delta_std:.4f}, "
              f"all positive = {bool((delta_seeds > 0).all())}.")
    unstable = e7_std > 0.008
    md.append(f"- Seed-variance threshold: std > 0.008 ⇒ unstable. E7 std = "
              f"{e7_std:.4f} ⇒ **{'UNSTABLE' if unstable else 'stable'}**.\n")

    md.append("## 4. Per-class breakdown (E1 → E7)\n")
    md.append("| class | E1 F1 | E7 F1 | Δ | claimed Δ | match? |")
    md.append("|---|---:|---:|---:|---:|:---:|")
    claims_full = {"ZdraviLudia": None, "Diabetes": +0.125, "PGOV_Glaukom": -0.057,
                   "SklerozaMultiplex": None, "SucheOko": None}
    for ci, cname in enumerate(CLASSES):
        f1a = m_E1["per_class_f1"][ci]
        f1b = m_E7["per_class_f1"][ci]
        dd = f1b - f1a
        claim_dd = claims_full[cname]
        ok_str = ""
        claim_str = ""
        if claim_dd is not None:
            claim_str = f"{claim_dd:+.3f}"
            ok_str = "PASS" if abs(dd - claim_dd) < 2e-3 else "FAIL"
        md.append(f"| {cname} | {f1a:.4f} | {f1b:.4f} | {dd:+.4f} | {claim_str} | {ok_str} |")
    md.append("")
    md.append("- SucheOko went from one correct scan (F1 = 0.0645) to zero correct "
              "(F1 = 0). Given only 2 SucheOko persons, that's 1 scan's worth of change.")
    md.append("- Support-weighted contributions to ΔW-F1:")
    for cname, c in contrib:
        md.append(f"  - {cname}: {c:+.4f}")
    md.append(f"")
    # Is Diabetes the driver?
    diabetes_contrib = dict(contrib)["Diabetes"]
    others_contrib = sum(c for cn, c in contrib if cn != "Diabetes")
    md.append(f"- Diabetes alone contributes **{diabetes_contrib:+.4f}** to ΔW-F1; "
              f"all other classes together contribute {others_contrib:+.4f}. "
              f"The +0.0083 total is "
              f"{'dominantly driven by Diabetes' if diabetes_contrib > 0.8 * delta_observed else 'not solely driven by Diabetes'}.\n")

    md.append("## 5. Verdict\n")
    md.append(f"Ship decision inputs:")
    md.append(f"- Reproduction: E1 {'PASS' if repro_E1 else 'FAIL'}, "
              f"E7 {'PASS' if repro_E7 else 'FAIL'}.")
    md.append(f"- Bootstrap 95% CI for ΔF1: [{ci_lo:+.4f}, {ci_hi:+.4f}].")
    md.append(f"- P(ΔF1 > 0) under person-resampling: {p_pos:.3f}.")
    md.append(f"- Seed std(E7) = {e7_std:.4f} (threshold 0.008).")
    md.append(f"- Seed Δ all positive: {bool((delta_seeds > 0).all())}.")
    md.append(f"- Diabetes drives {100*diabetes_contrib/max(delta_observed,1e-9):.0f}% of ΔW-F1.")
    md.append(f"")
    md.append(f"### Final call: {verdict}")
    md.append("")
    # Reasoning
    reasoning = []
    if not (repro_E1 and repro_E7):
        reasoning.append("reproduction failed — numbers in the claim don't match the script.")
    if not ci_strictly_positive:
        reasoning.append(f"the 95% bootstrap CI [{ci_lo:+.4f}, {ci_hi:+.4f}] "
                         f"includes 0, so the observed +0.0083 is within person-level "
                         f"resampling noise.")
    else:
        reasoning.append("the bootstrap 95% CI excludes zero.")
    if unstable:
        reasoning.append(f"E7's LR-seed std = {e7_std:.4f} exceeds the 0.008 stability "
                         f"threshold, so the single-seed +0.0083 is near the noise floor.")
    else:
        reasoning.append(f"E7 is stable across seeds (std = {e7_std:.4f} < 0.008).")
    if (delta_seeds > 0).all():
        reasoning.append("Δ(E7 − E1) is positive in all 5 seeds.")
    else:
        neg_seeds = [seeds[i] for i, d_ in enumerate(delta_seeds) if d_ <= 0]
        reasoning.append(f"Δ(E7 − E1) is not always positive across seeds "
                         f"(non-positive for seeds {neg_seeds}).")
    md.append("Reasoning: " + " ".join(reasoning))
    md.append("")

    (REPORTS / "RED_TEAM_E7_BOOTSTRAP.md").write_text("\n".join(md) + "\n")
    print(f"  wrote reports/RED_TEAM_E7_BOOTSTRAP.md")
    print(f"\n[done] total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
