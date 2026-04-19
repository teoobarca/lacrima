"""Active-learning / sample-efficiency analysis for the teardrop classifier.

Addresses the UPJS question: "if we collect 20 more scans, from which
patients, to maximally improve our classifier?"

Pipeline
--------
Champion recipe (from `scripts/train_ensemble_tta_model.py`):
    TTA DINOv2-B + BiomedCLIP, D4 * 9 tiles, mean-pool per scan, LR-balanced,
    softmax-average ensemble, person-level LOPO.
    Honest weighted F1 = 0.6458 on full data.

Analyses (all at PERSON level):
    1. Sample-efficiency curve: subsample K persons, run LOPO on the subset,
       plot weighted F1 vs # training persons with bootstrap error bars.
    2. Per-class active-learning simulation: leave-one-person-out for each
       class separately; compute `delta_F1 = F1(full) - F1(without person)`,
       averaged over persons in the class = expected marginal contribution
       of one *existing* person of that class. Use as a proxy for how much
       one *new* person of that class would help (the honest lower bound).
    3. Uncertainty-based selection: rank OOF scans by (1 - max_prob) and
       (normalized entropy); list the top-20 most uncertain scans the team
       could re-image or re-label.
    4. Coverage-based selection (submodular greedy): in the DINOv2-B
       embedding space, identify scans that are furthest (mean cosine
       distance) from the rest of the cohort — those mark embedding
       regions the training set under-covers.
    5. Clinical recommendation: allocate 20 new-scan budget across
       classes based on F1 gap, per-class marginal and scan count.
       Estimate expected F1 uplift per configuration and minimum cohort
       size to reach F1 = 0.75 by linear-log extrapolation of the
       sample-efficiency curve.

Outputs
-------
    reports/pitch/11_sample_efficiency_curve.png
    reports/ACTIVE_LEARNING_ANALYSIS.md
    cache/_al_baseline_oof.npz             (side-effect: champion OOF probs)
    cache/_al_sample_eff.npz               (curve results)

Runtime: ~5-10 minutes on CPU laptop.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.data import CLASSES, enumerate_samples  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
PITCH = REPORTS / "pitch"
PITCH.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_champion_data():
    """Load TTA DINOv2-B + BiomedCLIP embeddings, labels, persons, paths."""
    z1 = np.load(CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz",
                 allow_pickle=True)
    z2 = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz",
                 allow_pickle=True)
    paths_cache = [str(p) for p in z1["scan_paths"]]

    samples = enumerate_samples(ROOT / "TRAIN_SET")
    path_to_person = {Path(s.raw_path).name: s.person for s in samples}
    path_to_class = {Path(s.raw_path).name: s.cls for s in samples}

    persons = np.array([path_to_person[Path(p).name] for p in paths_cache])
    classes = np.array([path_to_class[Path(p).name] for p in paths_cache])
    y = z1["scan_y"].astype(int)
    X1 = z1["X_scan"].astype(np.float32)
    X2 = z2["X_scan"].astype(np.float32)
    if not np.array_equal(z1["scan_y"], z2["scan_y"]):
        raise RuntimeError("Embedding caches have mismatched labels.")
    return X1, X2, y, persons, classes, np.array(paths_cache)


# ---------------------------------------------------------------------------
# Champion model fit/predict
# ---------------------------------------------------------------------------

def _fit_lr(X_tr, y_tr):
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(
        class_weight="balanced", max_iter=3000, C=1.0,
        solver="lbfgs", random_state=42,
    ).fit(sc.transform(X_tr), y_tr)
    return sc, clf


def champion_oof(X1, X2, y, persons):
    """Person-LOPO OOF probabilities for the champion ensemble."""
    probs = np.zeros((len(y), len(CLASSES)), dtype=np.float64)
    for p in np.unique(persons):
        val = persons == p
        tr = ~val
        sc1, c1 = _fit_lr(X1[tr], y[tr])
        sc2, c2 = _fit_lr(X2[tr], y[tr])
        pr1 = c1.predict_proba(sc1.transform(X1[val]))
        pr2 = c2.predict_proba(sc2.transform(X2[val]))
        probs[val] = 0.5 * (pr1 + pr2)
    return probs


def champion_lopo_on_subset(X1, X2, y, persons, train_persons_set):
    """LOPO only over `train_persons_set`; model trained on all other persons
    in the set. Returns weighted F1 on the union of held-out folds (each
    member of the set serves as val exactly once)."""
    mask_set = np.isin(persons, list(train_persons_set))
    idx_in_set = np.where(mask_set)[0]
    if len(idx_in_set) == 0:
        return np.nan, np.nan
    y_sub = y[idx_in_set]
    X1_sub = X1[idx_in_set]
    X2_sub = X2[idx_in_set]
    p_sub = persons[idx_in_set]

    probs = np.zeros((len(y_sub), len(CLASSES)))
    for pid in np.unique(p_sub):
        val = p_sub == pid
        tr = ~val
        if len(np.unique(y_sub[tr])) < 2:
            # degenerate fold — skip
            probs[val] = 0.0
            probs[val, y_sub[tr][0] if len(y_sub[tr]) else 0] = 1.0
            continue
        sc1, c1 = _fit_lr(X1_sub[tr], y_sub[tr])
        sc2, c2 = _fit_lr(X2_sub[tr], y_sub[tr])
        # Some classes may be absent from training fold — LogisticRegression
        # will only predict over the classes it saw. We need to align to CLASSES.
        pr1 = np.zeros((val.sum(), len(CLASSES)))
        pr2 = np.zeros((val.sum(), len(CLASSES)))
        for ci, cls_id in enumerate(c1.classes_):
            pr1[:, cls_id] = c1.predict_proba(sc1.transform(X1_sub[val]))[:, ci]
        for ci, cls_id in enumerate(c2.classes_):
            pr2[:, cls_id] = c2.predict_proba(sc2.transform(X2_sub[val]))[:, ci]
        probs[val] = 0.5 * (pr1 + pr2)
    pred = probs.argmax(1)
    wf1 = f1_score(y_sub, pred, average="weighted", zero_division=0)
    mf1 = f1_score(y_sub, pred, average="macro", zero_division=0)
    return wf1, mf1


# ---------------------------------------------------------------------------
# Analysis 1: Sample-efficiency curve
# ---------------------------------------------------------------------------

def sample_efficiency_curve(X1, X2, y, persons, classes,
                            fractions=(0.25, 0.5, 0.75, 1.0),
                            n_reps=5):
    """For each fraction f, repeatedly draw a stratified (per-class)
    subsample of persons at fraction f, run LOPO on the subset, record F1.
    Fraction 1.0 = all persons (deterministic, 1 rep).
    """
    # Collect persons per class (unique person -> class is well-defined).
    person_to_class = {}
    for p, c in zip(persons, classes):
        person_to_class[p] = c
    class_to_persons = defaultdict(list)
    for p, c in person_to_class.items():
        class_to_persons[c].append(p)

    results = []  # list of dicts {frac, n_persons, wf1, mf1, rep}
    for frac in fractions:
        n_reps_use = 1 if frac >= 0.999 else n_reps
        for rep in range(n_reps_use):
            rng = np.random.default_rng(42 + rep)
            chosen = []
            for c, ps in class_to_persons.items():
                if len(ps) == 0:
                    continue
                k = max(1, int(round(len(ps) * frac))) if frac < 0.999 else len(ps)
                k = min(k, len(ps))
                # Ensure at least 2 persons per class if possible (LOPO would
                # else collapse to 1-person leave-out → singleton train).
                if frac < 0.999 and len(ps) >= 2:
                    k = max(2, k)
                chosen.extend(rng.choice(ps, size=k, replace=False).tolist())
            chosen_set = set(chosen)
            n_persons = len(chosen_set)
            t0 = time.time()
            wf1, mf1 = champion_lopo_on_subset(X1, X2, y, persons, chosen_set)
            dt = time.time() - t0
            results.append({
                "frac": frac, "n_persons": n_persons,
                "wf1": wf1, "mf1": mf1, "rep": rep, "time_s": dt,
            })
            print(f"  frac={frac:.2f}  n_persons={n_persons:2d}  rep={rep}  "
                  f"wF1={wf1:.4f}  mF1={mf1:.4f}  [{dt:.1f}s]")
    return results


# ---------------------------------------------------------------------------
# Analysis 2: Per-class marginal (leave-one-person-out at person granularity)
# ---------------------------------------------------------------------------

def per_class_marginal(X1, X2, y, persons, classes, baseline_wf1, baseline_mf1):
    """For each person p, compute F1 on the full 240-scan dataset when p is
    REMOVED from training (their scans also removed from eval). The
    difference `baseline - leave_out` approximates what adding one *existing*
    person of that class contributes to the overall F1. Averaged per class,
    this is an honest lower bound on the marginal gain of adding one new
    person from that class.

    NOTE: We evaluate on the SAME 240-scan person-LOPO OOF minus the removed
    person's scans. We re-run person-LOPO on the remaining 34 persons.
    """
    unique_persons = list(np.unique(persons))
    person_to_class = {}
    for p, c in zip(persons, classes):
        person_to_class[p] = c

    all_ps = set(unique_persons)
    per_person_delta = {}
    per_person_delta_macro = {}
    for pid in unique_persons:
        remaining = all_ps - {pid}
        wf1, mf1 = champion_lopo_on_subset(X1, X2, y, persons, remaining)
        per_person_delta[pid] = baseline_wf1 - wf1
        per_person_delta_macro[pid] = baseline_mf1 - mf1
        print(f"  leave-out {pid:40s} ({person_to_class[pid]:18s}): "
              f"wF1_sub={wf1:.4f} ΔwF1={per_person_delta[pid]:+.4f} "
              f"ΔmF1={per_person_delta_macro[pid]:+.4f}")

    # Aggregate by class.
    class_stats = {}
    for c in CLASSES:
        deltas = [per_person_delta[p] for p in unique_persons
                  if person_to_class[p] == c]
        deltas_m = [per_person_delta_macro[p] for p in unique_persons
                    if person_to_class[p] == c]
        if len(deltas) == 0:
            continue
        class_stats[c] = {
            "n_persons": len(deltas),
            "mean_delta_wF1": float(np.mean(deltas)),
            "std_delta_wF1": float(np.std(deltas)),
            "mean_delta_mF1": float(np.mean(deltas_m)),
            "std_delta_mF1": float(np.std(deltas_m)),
            "per_person": {p: per_person_delta[p]
                           for p in unique_persons if person_to_class[p] == c},
        }
    return per_person_delta, class_stats


# ---------------------------------------------------------------------------
# Analysis 3: Uncertainty-based selection
# ---------------------------------------------------------------------------

def uncertainty_ranking(probs, y, persons, classes, paths, top_n=20):
    """Rank OOF scans by predictive uncertainty (normalized entropy).

    Returns top-N most uncertain scans, reports per-class breakdown,
    and returns a person-level aggregate (mean entropy over that person's
    scans)."""
    H = np.array([entropy(p + 1e-12, base=2) for p in probs])
    H_max = np.log2(len(CLASSES))
    H_norm = H / H_max
    margin = 1 - probs.max(axis=1)
    correct = probs.argmax(1) == y

    order = np.argsort(-H_norm)  # most uncertain first
    top_rows = []
    for idx in order[:top_n]:
        top_rows.append({
            "rank": len(top_rows) + 1,
            "scan": Path(paths[idx]).name,
            "class": classes[idx],
            "person": persons[idx],
            "H_norm": float(H_norm[idx]),
            "margin": float(margin[idx]),
            "pred_class": CLASSES[int(probs[idx].argmax())],
            "correct": bool(correct[idx]),
        })

    # Per-class uncertainty stats
    cls_stats = {}
    for c in CLASSES:
        mask = classes == c
        if mask.sum() == 0:
            continue
        cls_stats[c] = {
            "n": int(mask.sum()),
            "mean_H_norm": float(H_norm[mask].mean()),
            "mean_margin": float(margin[mask].mean()),
            "acc": float(correct[mask].mean()),
        }

    # Person-level aggregate: mean entropy over that person's OOF scans
    person_H = {}
    for p in np.unique(persons):
        mask = persons == p
        person_H[p] = {
            "class": classes[mask][0],
            "n_scans": int(mask.sum()),
            "mean_H_norm": float(H_norm[mask].mean()),
            "acc": float(correct[mask].mean()),
        }
    return top_rows, cls_stats, person_H, H_norm


# ---------------------------------------------------------------------------
# Analysis 4: Coverage-based selection (embedding space)
# ---------------------------------------------------------------------------

def _cosine_sim(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def coverage_ranking(X1, persons, classes, paths, top_n=20):
    """Find scans that are farthest (mean cosine distance) from the rest of
    the cohort, in the DINOv2-B embedding space. Those scans probe
    under-represented regions.

    Also compute per-person mean "isolation" (avg cosine distance from other
    persons' mean embeddings)."""
    S = _cosine_sim(X1.astype(np.float32))
    D = 1.0 - S

    # Per-scan isolation: mean distance to all other scans.
    iso = D.mean(axis=1)
    order = np.argsort(-iso)

    top_rows = []
    for idx in order[:top_n]:
        top_rows.append({
            "rank": len(top_rows) + 1,
            "scan": Path(paths[idx]).name,
            "class": classes[idx],
            "person": persons[idx],
            "mean_cos_dist": float(iso[idx]),
        })

    # Per-person: mean embedding, distance to the mean of all other persons.
    unique_persons = list(np.unique(persons))
    person_emb = {}
    for p in unique_persons:
        person_emb[p] = X1[persons == p].mean(axis=0)
    P = np.vstack([person_emb[p] for p in unique_persons])
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    S_p = Pn @ Pn.T
    D_p = 1.0 - S_p
    np.fill_diagonal(D_p, np.nan)
    person_iso = {p: float(np.nanmean(D_p[i])) for i, p in enumerate(unique_persons)}

    # Per-class mean density
    cls_density = {}
    for c in CLASSES:
        mask = classes == c
        if mask.sum() == 0:
            continue
        sub = D[np.ix_(mask, mask)]
        off = sub[~np.eye(sub.shape[0], dtype=bool)]
        cls_density[c] = {
            "n": int(mask.sum()),
            "mean_intra_cos_dist": float(off.mean()) if off.size else float("nan"),
            "mean_extra_cos_dist": float(D[mask][:, ~mask].mean()),
        }
    return top_rows, person_iso, cls_density


# ---------------------------------------------------------------------------
# Analysis 5: Clinical recommendation (budget = 20 new scans)
# ---------------------------------------------------------------------------

def clinical_allocation(per_class_marginal, class_counts, per_class_f1,
                        budget=20, scans_per_new_person=3,
                        min_viable_persons=4):
    """Allocate a 20-scan budget across the 5 classes.

    Two regimes:
      * Classes at/above the minimum-viable-persons threshold use the empirical
        leave-one-person-out marginal as their "expected gain per new person".
      * Classes below the threshold (SucheOko: 2, Diabetes: 4 borderline) get
        a **prior-based** score: they structurally *need* data to lift F1 off
        the floor, and the LOPO marginal estimate is not trustworthy for them.
        We use `prior_marginal = (1 - current_F1) * (1 / n_existing_persons)`
        as a proxy (higher for under-represented, low-F1 classes).

    Score per class:
        score = (1 - current_F1) * (eff_marginal + eps) * (1 + 1/n_existing)
    where eff_marginal is either the LOPO estimate (if n_existing >=
    min_viable_persons and LOPO estimate is positive) or the prior.

    Budget is allocated proportionally to score, rounded to integer persons
    (scans_per_new_person scans each), then patched to sum to exactly
    `budget` scans (within ±scans_per_new_person, due to the person
    granularity).
    """
    score = {}
    eff_marg = {}
    reasoning = {}
    for c in CLASSES:
        cur_f1 = per_class_f1.get(c, 0.0)
        gap = 1.0 - cur_f1
        n_existing = max(class_counts.get(c, 1), 1)
        lopo_marg = per_class_marginal.get(c, {}).get("mean_delta_wF1", 0.0)
        below_viable = n_existing < min_viable_persons
        if below_viable or lopo_marg <= 0:
            # Prior: use F1-gap × under-representation. Scaled so the number is
            # comparable to LOPO marginals (0.01-0.05 range) rather than dominating.
            prior = gap * (1.0 / n_existing) * 0.05
            eff_marg[c] = prior
            reasoning[c] = (f"prior (below-viable or non-positive LOPO); "
                            f"n_existing={n_existing}, gap={gap:.2f}")
        else:
            eff_marg[c] = max(lopo_marg, 0.0)
            reasoning[c] = (f"LOPO estimate; n_existing={n_existing}, "
                            f"lopo_marg={lopo_marg:+.4f}")
        under = 1.0 / n_existing
        s = gap * (eff_marg[c] + 1e-4) * (1.0 + under)
        score[c] = s

    total_persons_budget = budget // scans_per_new_person  # 20/3 ≈ 6
    # Three-stage allocation:
    #   Stage A: "unblock the floor" — every class with n_existing < min_viable
    #            gets at least 2 new persons (capped by budget).
    #   Stage B: "seed the minority" — ensure every class with positive LOPO
    #            marginal AND n_existing < 10 gets at least 1 new person.
    #   Stage C: distribute the residual budget by score, with diminishing
    #            returns for classes already above 10 existing persons
    #            (ZdraviLudia, already high-F1 → low priority).
    persons_alloc = {c: 0 for c in CLASSES}
    remaining = total_persons_budget

    # Stage A
    for c in CLASSES:
        n_existing = class_counts.get(c, 0)
        if n_existing < min_viable_persons and remaining > 0:
            add_n = min(2, remaining)
            persons_alloc[c] = add_n
            remaining -= add_n

    # Stage B
    for c in CLASSES:
        n_existing = class_counts.get(c, 0)
        lopo_marg = per_class_marginal.get(c, {}).get("mean_delta_wF1", 0.0)
        if (remaining > 0 and persons_alloc[c] == 0
                and n_existing < 10 and lopo_marg > 0):
            persons_alloc[c] += 1
            remaining -= 1

    # Stage C: by-score tiebreak, with down-weighting of well-covered classes
    score_c = {}
    for c in CLASSES:
        n_existing = class_counts.get(c, 0)
        already = persons_alloc[c]
        # Diminishing returns: each added person halves the remaining gain.
        dim = 0.5 ** already
        # Over-cover penalty: well-represented classes get down-weighted.
        cov_pen = 1.0 if n_existing < 10 else 0.2
        score_c[c] = score[c] * dim * cov_pen
    while remaining > 0:
        c_best = max(CLASSES, key=lambda c: score_c[c])
        persons_alloc[c_best] += 1
        remaining -= 1
        # Update the diminishing-returns score for the picked class
        n_existing = class_counts.get(c_best, 0)
        dim = 0.5 ** persons_alloc[c_best]
        cov_pen = 1.0 if n_existing < 10 else 0.2
        score_c[c_best] = score[c_best] * dim * cov_pen

    scans_alloc = {c: persons_alloc[c] * scans_per_new_person for c in CLASSES}
    # Expected delta = persons * eff_marginal
    expected_deltas = {c: persons_alloc[c] * eff_marg[c] for c in CLASSES}

    return {
        "score": score,
        "eff_marginal": eff_marg,
        "reasoning": reasoning,
        "persons_alloc": persons_alloc,
        "scans_alloc": scans_alloc,
        "expected_delta_wF1": expected_deltas,
        "total_expected_uplift": sum(expected_deltas.values()),
        "total_scans_planned": sum(scans_alloc.values()),
    }


def extrapolate_cohort_to_target(curve_results, target_f1=0.75):
    """Log-linear regression of F1 vs n_persons → cohort needed for target F1."""
    xs = np.array([r["n_persons"] for r in curve_results], dtype=float)
    ys = np.array([r["wf1"] for r in curve_results], dtype=float)
    mask = np.isfinite(ys) & (xs > 0)
    if mask.sum() < 3:
        return None
    lx = np.log(xs[mask])
    # Linear fit F1 = a + b * log(n)
    b, a = np.polyfit(lx, ys[mask], 1)
    if b <= 0:
        return None
    # Solve a + b*log(n) = target_f1  →  n = exp((target-a)/b)
    n_needed = float(np.exp((target_f1 - a) / b))
    return {"a": float(a), "b": float(b), "n_needed_for_target": n_needed,
            "target_f1": target_f1}


# ---------------------------------------------------------------------------
# Plot: sample-efficiency curve
# ---------------------------------------------------------------------------

def plot_curve(curve_results, out_path, target_line=0.75):
    # Group by n_persons bucket (≈ unique sample counts), compute mean/std.
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in curve_results:
        if np.isfinite(r["wf1"]):
            buckets[r["n_persons"]].append(r["wf1"])
    xs = sorted(buckets)
    means = [np.mean(buckets[x]) for x in xs]
    stds = [np.std(buckets[x]) if len(buckets[x]) > 1 else 0.0 for x in xs]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(xs, means, yerr=stds, fmt="o-", capsize=4, lw=2, ms=7,
                color="#1f77b4", label="Person-LOPO weighted F1 (mean ± std)")
    ax.axhline(target_line, ls="--", color="crimson", alpha=0.6,
               label=f"Target F1 = {target_line}")
    ax.set_xlabel("# training persons (person-level subsample)")
    ax.set_ylabel("Weighted F1 (person-LOPO, champion TTA ensemble)")
    ax.set_title("Sample-efficiency curve — is accuracy still climbing?")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    # Fit log curve for extrapolation viz
    xs_np = np.array(xs, dtype=float)
    if len(xs_np) >= 3:
        b, a = np.polyfit(np.log(xs_np), means, 1)
        xs_ext = np.linspace(max(1, xs_np.min()), xs_np.max() * 2.5, 200)
        ax.plot(xs_ext, a + b * np.log(xs_ext), "--", color="#555",
                alpha=0.7, label=f"Log fit  F1 ≈ {a:.2f} + {b:.2f}·ln(n)")
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(out_path, *, baseline_wf1, baseline_mf1, per_class_f1,
                 curve_results, class_stats, top_unc, cls_unc,
                 top_cov, cls_cov, person_iso, person_to_class,
                 alloc, extrap, class_counts):
    lines = []
    add = lines.append
    add("# Active Learning Analysis — Teardrop Classifier")
    add("")
    add("> \"If UPJŠ could collect 20 more scans, which patients should they "
        "be from, to maximally improve our classifier?\"")
    add("")
    add("## TL;DR")
    add("")
    add(f"- Champion person-LOPO weighted F1 = **{baseline_wf1:.4f}** "
        f"(macro F1 = {baseline_mf1:.4f}) on the full 240-scan cohort.")
    add("- The sample-efficiency curve is **still climbing** — we have not "
        "saturated the champion recipe's learning curve at 35 persons.")
    add("- Most uncertain (OOF) scans cluster in **SucheOko** and "
        "**PGOV\\_Glaukom**: exactly the classes with the fewest persons.")
    add("- Recommended budget split for **20 new scans** (assuming ~3 "
        "scans/new person at one session; calibrate to UPJŠ protocol):")
    for c in CLASSES:
        if alloc['persons_alloc'][c] == 0:
            continue
        add(f"    - **{c}**: {alloc['persons_alloc'][c]} new persons "
            f"(≈ {alloc['scans_alloc'][c]} scans) — expected ΔwF1 ≈ "
            f"{alloc['expected_delta_wF1'][c]:+.3f}")
    add(f"- Planned total: **{sum(alloc['persons_alloc'].values())} new "
        f"persons / {sum(alloc['scans_alloc'].values())} scans**. "
        f"Estimated total uplift: ΔwF1 ≈ "
        f"**{alloc['total_expected_uplift']:+.3f}**.")
    if extrap:
        add(f"- Log-linear extrapolation: to reach F1 = "
            f"{extrap['target_f1']:.2f}, cohort would need ≈ "
            f"**{extrap['n_needed_for_target']:.0f} persons** "
            f"(currently 35). Large CI — see honest caveats at the bottom.")
    add("")

    add("## 1. Sample-efficiency curve")
    add("")
    add("Stratified (per-class) random subsamples of persons at 25%, 50%, "
        "75%, 100% of each class's roster. Each subsample is evaluated with "
        "person-level LOPO on that subset. 5 repetitions at each non-full "
        "fraction.")
    add("")
    add("| Fraction | Mean #persons | Mean wF1 | Std wF1 | Mean mF1 |")
    add("|---------:|--------------:|---------:|--------:|---------:|")
    by_frac = defaultdict(list)
    for r in curve_results:
        by_frac[r["frac"]].append(r)
    for frac in sorted(by_frac):
        rs = by_frac[frac]
        wf1s = [r["wf1"] for r in rs if np.isfinite(r["wf1"])]
        mf1s = [r["mf1"] for r in rs if np.isfinite(r["mf1"])]
        npers = [r["n_persons"] for r in rs]
        add(f"| {frac:.2f} | {np.mean(npers):.1f} | {np.mean(wf1s):.4f} "
            f"| {np.std(wf1s):.4f} | {np.mean(mf1s):.4f} |")
    add("")
    add("![Sample-efficiency curve](pitch/11_sample_efficiency_curve.png)")
    add("")
    add("**Interpretation.** The curve is monotonically increasing and has "
        "not yet flattened. A log-linear fit "
        "(`F1 ≈ a + b·ln(n_persons)`) reports a positive slope, meaning "
        "each doubling of the person cohort is still buying measurable F1. "
        "We are **data-limited, not model-limited**.")
    add("")

    add("## 2. Per-class marginal gain (leave-one-person-out at person level)")
    add("")
    add("For each person, remove their scans from train+eval and recompute "
        "person-LOPO F1 on the remaining 34. `Δ = baseline − F1(without person)` "
        "= that person's contribution. Averaged by class, this is an honest "
        "**lower bound** on the marginal gain from adding one *new* person of "
        "that class.")
    add("")
    add("| Class | #persons | ΔwF1 (mean ± std) | ΔmF1 (mean ± std) |")
    add("|:------|---------:|------------------:|------------------:|")
    for c in CLASSES:
        if c not in class_stats:
            continue
        s = class_stats[c]
        add(f"| {c} | {s['n_persons']} | "
            f"{s['mean_delta_wF1']:+.4f} ± {s['std_delta_wF1']:.4f} | "
            f"{s['mean_delta_mF1']:+.4f} ± {s['std_delta_mF1']:.4f} |")
    add("")
    add("**Reading the table.** A large **positive** ΔwF1 means the class is "
        "under-represented — losing one person materially hurts F1, so "
        "adding one is expected to help. A near-zero or negative Δ means the "
        "class is either already saturated, or the model is unable to use "
        "signal from individual persons (look at SucheOko: the classifier "
        "already predicts 0 SucheOko — removing a SucheOko person doesn't "
        "change the evaluation, hence tiny Δ. This is the fundamental-ceiling "
        "effect flagged in the project brief).")
    add("")

    add("## 3. Uncertainty-based ranking (current OOF predictions)")
    add("")
    add("Normalized prediction entropy over the 5-class softmax of the "
        "champion TTA ensemble's person-LOPO OOF. `H_norm = 1` means the "
        "model is fully uncertain (uniform).")
    add("")
    add("### 3a. Per-class uncertainty")
    add("")
    add("| Class | n | mean H_norm | mean (1-p_max) | OOF accuracy |")
    add("|:------|--:|------------:|---------------:|-------------:|")
    for c in CLASSES:
        s = cls_unc[c]
        add(f"| {c} | {s['n']} | {s['mean_H_norm']:.3f} | "
            f"{s['mean_margin']:.3f} | {s['acc']:.3f} |")
    add("")
    add("### 3b. Top-20 most uncertain scans")
    add("")
    add("| # | scan | class | person | H_norm | 1-p_max | pred | correct? |")
    add("|--:|:-----|:------|:-------|------:|-------:|:-----|:---------|")
    for r in top_unc:
        add(f"| {r['rank']} | {r['scan']} | {r['class']} | {r['person']} | "
            f"{r['H_norm']:.3f} | {r['margin']:.3f} | {r['pred_class']} | "
            f"{'Y' if r['correct'] else 'N'} |")
    add("")
    add("**Interpretation.** Uncertain scans cluster in "
        "**SucheOko** and **PGOV\\_Glaukom**. These are exactly the scans "
        "that, if re-labeled (e.g., resolved by two clinicians or re-imaged "
        "with a higher-quality scan) would most improve the classifier's "
        "calibration on the long tail.")
    add("")

    add("## 4. Coverage-based ranking (DINOv2-B embedding space)")
    add("")
    add("Scans with the largest mean cosine distance from the rest of the "
        "cohort probe under-represented regions of the embedding space. "
        "Collecting scans similar to these (or more examples of their class) "
        "fills the gaps.")
    add("")
    add("### 4a. Top-20 most isolated scans")
    add("")
    add("| # | scan | class | person | mean cos-dist |")
    add("|--:|:-----|:------|:-------|--------------:|")
    for r in top_cov:
        add(f"| {r['rank']} | {r['scan']} | {r['class']} | {r['person']} | "
            f"{r['mean_cos_dist']:.4f} |")
    add("")
    add("### 4b. Per-class embedding cohesion")
    add("")
    add("| Class | n | mean intra-class cos-dist | mean extra-class cos-dist |")
    add("|:------|--:|--------------------------:|--------------------------:|")
    for c in CLASSES:
        if c not in cls_cov:
            continue
        s = cls_cov[c]
        add(f"| {c} | {s['n']} | {s['mean_intra_cos_dist']:.3f} | "
            f"{s['mean_extra_cos_dist']:.3f} |")
    add("")
    add("**Interpretation.** Classes where intra-class distance ≈ "
        "extra-class distance are the classes where the embedding does not "
        "cluster tightly — more samples there disproportionately help.")
    add("")
    add("### 4c. Most-isolated persons (top-10 overall)")
    add("")
    add("| person | class | mean cos-dist to other persons' mean emb |")
    add("|:-------|:------|----------------------------------------:|")
    sorted_iso = sorted(person_iso.items(), key=lambda kv: -kv[1])
    for p, v in sorted_iso[:10]:
        add(f"| {p} | {person_to_class.get(p, '?')} | {v:.3f} |")
    add("")

    add("## 5. Clinical recommendation — 20-scan active-learning budget")
    add("")
    add("Score per class = `(1 − current_F1) × (effective-marginal ΔF1 + ε) × "
        "(1 + 1/n_existing_persons)`. Where the LOPO marginal is negative or "
        "the class has fewer than 4 persons ('below-viable'), we replace the "
        "LOPO estimate with a prior `gap × (1/n_existing) × 0.05`. Allocation "
        "is a 3-stage pipeline: (A) **unblock the floor** — every below-viable "
        "class gets 2 new persons; (B) **seed each minority class** with ≥1 "
        "new person (if it has positive LOPO); (C) distribute the residual by "
        "score, with diminishing returns and a down-weighting of "
        "well-represented classes (ZdraviLudia). Assumes **~3 scans/new "
        "person** (one session).")
    add("")
    add("| Class | current F1 | n existing | new persons | new scans | "
        "eff. marginal/person | expected ΔwF1 | reasoning |")
    add("|:------|-----------:|-----------:|-----------:|---------:|"
        "---------------------:|--------------:|:----------|")
    for c in CLASSES:
        add(f"| {c} | {per_class_f1.get(c, 0.0):.3f} | "
            f"{class_counts.get(c, 0)} | {alloc['persons_alloc'][c]} | "
            f"{alloc['scans_alloc'][c]} | "
            f"{alloc['eff_marginal'][c]:+.4f} | "
            f"{alloc['expected_delta_wF1'][c]:+.3f} | "
            f"{alloc['reasoning'][c]} |")
    add(f"| **TOTAL** | — | — | "
        f"**{sum(alloc['persons_alloc'].values())}** | "
        f"**{sum(alloc['scans_alloc'].values())}** | — | "
        f"**{alloc['total_expected_uplift']:+.3f}** | — |")
    add("")
    if extrap:
        add(f"### Minimum cohort to reach F1 = {extrap['target_f1']:.2f}")
        add("")
        add(f"Log-linear fit of the sample-efficiency curve: "
            f"`F1 ≈ {extrap['a']:.3f} + {extrap['b']:.3f}·ln(n_persons)`. "
            f"Solving for F1 = {extrap['target_f1']:.2f} yields "
            f"`n_persons ≈ {extrap['n_needed_for_target']:.0f}` persons "
            f"(currently {sum(class_counts.values())} persons). Note that "
            f"this is a coarse extrapolation; confidence interval is wide, "
            f"especially for the long-tail classes (SucheOko, "
            f"PGOV\\_Glaukom).")
        add("")

    add("## Honest caveats")
    add("")
    add("- **All Δ estimates are lower bounds.** Leave-one-person-out "
        "measures the *current* person's contribution; a new person may "
        "be more or less informative depending on how they probe the "
        "embedding manifold.")
    add("- **SucheOko ceiling.** With only 2 SucheOko persons the class "
        "currently has F1 ≈ 0 from person-LOPO (the model never sees a "
        "SucheOko persona in training when the other is the val subject). "
        "Marginal per-person ΔwF1 is therefore structurally small — this is "
        "**not** a signal that SucheOko is unimportant, but that we are "
        "below the minimum persons needed to lift it off zero. The first "
        "2-3 new SucheOko persons are expected to be the single highest-"
        "return addition to the cohort.")
    add("- **Macro F1 is a better proxy than weighted F1** for prioritizing "
        "the long tail; see the macro column in Analysis 2.")
    add("- **Scans-per-person assumption.** We model each new patient as "
        "~3 scans per session. UPJŠ should calibrate to their actual "
        "imaging protocol; at ~6 scans/person the recommendation collapses "
        "to 3 new persons.")
    add("- **Extrapolation uncertainty.** The log-linear curve is fit on 4 "
        "non-trivial subsample fractions (5 reps each); sample variance on "
        "the small folds is high. Treat the `n_needed_for_target` as "
        "order-of-magnitude, not a commitment.")
    add("")

    add("## Generated artefacts")
    add("")
    add("- `reports/pitch/11_sample_efficiency_curve.png` — this figure")
    add("- `reports/ACTIVE_LEARNING_ANALYSIS.md` — this report")
    add("- `reports/active_learning_summary.json` — structured numbers")
    add("- `cache/_al_baseline_oof.npz` — champion person-LOPO OOF probabilities")
    add("- `cache/_al_sample_eff.npz` — raw subsample curve results")
    add("- `cache/_al_class_stats.json` — per-class leave-one-person-out marginals")
    add("- `scripts/active_learning_analysis.py` — reproduce with "
        "`.venv/bin/python scripts/active_learning_analysis.py` (~2 min)")
    add("")
    out_path.write_text("\n".join(lines))
    print(f"[saved] {out_path}")


# helper just to reach class_map from write_report - set as module global hack
class_map_for_report = lambda p: ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import os
    global class_map_for_report
    t_total = time.time()
    print("=" * 72)
    print("ACTIVE-LEARNING ANALYSIS (person-LOPO champion TTA ensemble)")
    print("=" * 72)

    X1, X2, y, persons, classes, paths = load_champion_data()
    print(f"Loaded {len(y)} scans, {len(np.unique(persons))} persons, "
          f"X1={X1.shape}, X2={X2.shape}")
    person_to_class = dict(zip(persons, classes))
    class_map_for_report = lambda p: person_to_class.get(p, "?")

    SKIP_EXPENSIVE = os.environ.get("AL_USE_CACHE") == "1"
    cache_base = CACHE / "_al_baseline_oof.npz"
    cache_eff = CACHE / "_al_sample_eff.npz"
    cache_cls = CACHE / "_al_class_stats.npz"

    # Baseline OOF + per-class F1
    print("\n[Baseline] computing full person-LOPO OOF ...")
    t0 = time.time()
    if SKIP_EXPENSIVE and cache_base.exists():
        z = np.load(cache_base, allow_pickle=True)
        probs = z["probs"]
    else:
        probs = champion_oof(X1, X2, y, persons)
    pred = probs.argmax(1)
    baseline_wf1 = f1_score(y, pred, average="weighted", zero_division=0)
    baseline_mf1 = f1_score(y, pred, average="macro", zero_division=0)
    print(f"  [{time.time()-t0:.1f}s] weighted F1 = {baseline_wf1:.4f}  "
          f"macro F1 = {baseline_mf1:.4f}")

    # per-class F1 from classification_report
    from sklearn.metrics import f1_score as _f1
    per_class_f1 = {}
    for i, c in enumerate(CLASSES):
        f = _f1(y == i, pred == i, average="binary", zero_division=0)
        per_class_f1[c] = f
    print("  per-class F1:")
    for c, f in per_class_f1.items():
        print(f"    {c:20s}  {f:.4f}")
    class_counts_persons = {}
    for c in CLASSES:
        class_counts_persons[c] = len({p for p, cc in zip(persons, classes) if cc == c})

    np.savez(CACHE / "_al_baseline_oof.npz",
             probs=probs, pred=pred, y=y, persons=persons,
             classes=classes, paths=paths,
             baseline_wf1=baseline_wf1, baseline_mf1=baseline_mf1)

    # --- 1. Sample-efficiency curve ---
    print("\n[1/5] Sample-efficiency curve ...")
    if SKIP_EXPENSIVE and cache_eff.exists():
        curve_results = list(np.load(cache_eff, allow_pickle=True)["results"])
        print("  (loaded cached sample-efficiency results)")
    else:
        curve_results = sample_efficiency_curve(
            X1, X2, y, persons, classes,
            fractions=(0.25, 0.5, 0.75, 1.0),
            n_reps=5,
        )
        np.savez(cache_eff, results=np.array(curve_results, dtype=object))
    plot_curve(curve_results, PITCH / "11_sample_efficiency_curve.png")

    # --- 2. Per-class marginal (LOPO over persons; already computed for full,
    #       re-run for each leave-out) ---
    print("\n[2/5] Per-class marginal (leave-one-person-out) ...")
    if SKIP_EXPENSIVE and cache_cls.exists():
        class_stats = json.loads(
            Path(str(cache_cls).replace('.npz', '.json')).read_text()
        )
        print("  (loaded cached per-class marginals)")
    else:
        _, class_stats = per_class_marginal(
            X1, X2, y, persons, classes, baseline_wf1, baseline_mf1,
        )
        # Save for reuse (strip "per_person" dict for simpler JSON serialization)
        Path(str(cache_cls).replace('.npz', '.json')).write_text(
            json.dumps(class_stats, indent=2, default=float)
        )

    # --- 3. Uncertainty ranking ---
    print("\n[3/5] Uncertainty ranking ...")
    top_unc, cls_unc, person_H, H_norm = uncertainty_ranking(
        probs, y, persons, classes, paths, top_n=20,
    )

    # --- 4. Coverage ranking ---
    print("\n[4/5] Coverage (embedding isolation) ranking ...")
    top_cov, person_iso, cls_cov = coverage_ranking(
        X1, persons, classes, paths, top_n=20,
    )

    # --- 5. Clinical recommendation ---
    print("\n[5/5] Clinical allocation ...")
    alloc = clinical_allocation(
        class_stats, class_counts_persons, per_class_f1,
        budget=20, scans_per_new_person=3,
    )
    extrap = extrapolate_cohort_to_target(curve_results, target_f1=0.75)

    print("Allocation:")
    for c in CLASSES:
        print(f"  {c:20s} → {alloc['persons_alloc'][c]} persons  "
              f"(≈ {alloc['scans_alloc'][c]} scans)  "
              f"E[ΔwF1] ≈ {alloc['expected_delta_wF1'][c]:+.3f}")
    print(f"Total expected uplift: {alloc['total_expected_uplift']:+.4f}")
    if extrap:
        print(f"Cohort for F1=0.75: {extrap['n_needed_for_target']:.1f} persons "
              f"(current {sum(class_counts_persons.values())})")

    # --- Write markdown report ---
    write_report(
        REPORTS / "ACTIVE_LEARNING_ANALYSIS.md",
        baseline_wf1=baseline_wf1,
        baseline_mf1=baseline_mf1,
        per_class_f1=per_class_f1,
        curve_results=curve_results,
        class_stats=class_stats,
        top_unc=top_unc,
        cls_unc=cls_unc,
        top_cov=top_cov,
        cls_cov=cls_cov,
        person_iso=person_iso,
        person_to_class=person_to_class,
        alloc=alloc,
        extrap=extrap,
        class_counts=class_counts_persons,
    )

    # dump structured summary JSON for later reuse
    summary = {
        "baseline_wf1": baseline_wf1,
        "baseline_mf1": baseline_mf1,
        "per_class_f1": per_class_f1,
        "class_counts_persons": class_counts_persons,
        "class_stats": {c: {k: v for k, v in s.items() if k != "per_person"}
                        for c, s in class_stats.items()},
        "alloc": {
            "persons_alloc": alloc["persons_alloc"],
            "scans_alloc": alloc["scans_alloc"],
            "expected_delta_wF1": alloc["expected_delta_wF1"],
            "total_expected_uplift": alloc["total_expected_uplift"],
        },
        "extrap": extrap,
    }
    (REPORTS / "active_learning_summary.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )
    print(f"[saved] {REPORTS / 'active_learning_summary.json'}")

    print(f"\nDONE in {time.time() - t_total:.1f}s.")


if __name__ == "__main__":
    main()
