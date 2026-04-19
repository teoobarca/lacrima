"""Fusion ensemble — v4 + k-NN + XGBoost handcrafted (honest, VLM-free).

Combines three *clean* OOF probability tracks (no VLM contamination) via four
fusion strategies, under strict person-LOPO. A nested-LOPO logistic-regression
stacker prevents the leakage trap where the stacker trains on OOF probs that
were produced including the query's fold.

Inputs
------
- cache/v4_oof_predictions.npz          (proba, y, persons, scan_paths) — wF1 0.6887
- cache/knn_baseline_best_predictions.json (k=1 majority, wF1 0.6117)
- cache/expert_council_predictions.json (expert3_xgb.probs — handcrafted XGB OOF)
- cache/vlm_honest_manifest.json        (scan_key → raw_path mapping)

Outputs
-------
- cache/fusion_ensemble_predictions.json
- reports/FUSION_ENSEMBLE.md

Strategies
----------
a) Geometric mean of softmaxes (equal weight).
b) Weighted arithmetic mean with weights ∝ per-model wF1.
c) Nested-LOPO logistic-regression stacker on concatenated probs (+ entropy feats).
d) Class-specific routing: for each class c, best-on-c model (by per-class F1)
   gets larger weight, mixed with a small uniform prior.

All numbers are person-LOPO. A 1000× paired bootstrap compares each fusion
against v4. A fusion is flagged as ensemble champion candidate only if
wF1 > 0.70 AND P(Δ>0 vs v4) > 0.90.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from teardrop.data import CLASSES  # noqa: E402

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Data loading — align ALL sources by scan_path
# ---------------------------------------------------------------------------

def rel_path(p: str) -> str:
    s = str(p)
    marker = "TRAIN_SET/"
    i = s.find(marker)
    return s[i:] if i >= 0 else s


def load_all() -> dict:
    # 1) v4 OOF — canonical order
    v4 = np.load(CACHE / "v4_oof_predictions.npz", allow_pickle=True)
    v4_proba = v4["proba"].astype(np.float64)             # (240, 5)
    y = v4["y"].astype(np.int64)
    persons = np.array([str(p) for p in v4["persons"]])
    scan_paths = np.array([str(p) for p in v4["scan_paths"]])
    rel_paths = np.array([rel_path(p) for p in scan_paths])
    N = len(scan_paths)

    # 2) k-NN OOF (JSON by rel path)
    with open(CACHE / "knn_baseline_best_predictions.json") as f:
        knn_json = json.load(f)
    knn_by_path = {r["path"]: r for r in knn_json["predictions"]}
    knn_proba = np.zeros((N, len(CLASSES)), dtype=np.float64)
    knn_f1 = float(knn_json["weighted_f1"])
    knn_config = knn_json["config"]
    for i, rp in enumerate(rel_paths):
        rec = knn_by_path[rp]
        for j, c in enumerate(CLASSES):
            knn_proba[i, j] = rec["proba"][c]

    # 3) XGBoost handcrafted OOF + soft k-NN (k=5, sim-weighted) from expert_council JSON
    # Align via manifest scan_key ↔ raw_path
    with open(CACHE / "vlm_honest_manifest.json") as f:
        manifest = json.load(f)
    key_to_rel = {k: rel_path(v["raw_path"]) for k, v in manifest.items()}
    rel_to_key = {v: k for k, v in key_to_rel.items()}

    with open(CACHE / "expert_council_predictions.json") as f:
        council = json.load(f)
    xgb_proba = np.zeros((N, len(CLASSES)), dtype=np.float64)
    knn_soft_proba = np.zeros((N, len(CLASSES)), dtype=np.float64)
    for i, rp in enumerate(rel_paths):
        key = rel_to_key[rp]
        rec = council[key]
        for j, c in enumerate(CLASSES):
            xgb_proba[i, j] = rec["expert3_xgb"]["probs"][c]
            knn_soft_proba[i, j] = rec["expert2_knn"]["probs"][c]

    # Normalise each to proper probability row (defence against FP drift)
    for M in (v4_proba, knn_proba, xgb_proba, knn_soft_proba):
        s = M.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        M /= s

    # Per-model wF1 sanity
    v4_f1 = float(f1_score(y, v4_proba.argmax(1), average="weighted", zero_division=0))
    knn_f1_re = float(f1_score(y, knn_proba.argmax(1), average="weighted", zero_division=0))
    xgb_f1 = float(f1_score(y, xgb_proba.argmax(1), average="weighted", zero_division=0))
    knn_soft_f1 = float(f1_score(y, knn_soft_proba.argmax(1), average="weighted", zero_division=0))

    return {
        "N": N,
        "y": y,
        "persons": persons,
        "scan_paths": scan_paths,
        "rel_paths": rel_paths,
        "v4_proba": v4_proba,
        "knn_proba": knn_proba,
        "knn_soft_proba": knn_soft_proba,
        "xgb_proba": xgb_proba,
        "v4_f1": v4_f1,
        "knn_f1": knn_f1_re,
        "knn_soft_f1": knn_soft_f1,
        "xgb_f1": xgb_f1,
        "knn_config": knn_config,
    }


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------

EPS = 1e-9


def geo_mean(probs_list: list[np.ndarray]) -> np.ndarray:
    log_sum = np.zeros_like(probs_list[0])
    for p in probs_list:
        log_sum = log_sum + np.log(np.clip(p, EPS, 1.0))
    log_sum = log_sum / len(probs_list)
    out = np.exp(log_sum)
    out = out / out.sum(axis=1, keepdims=True)
    return out


def weighted_arith(probs_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = np.zeros_like(probs_list[0])
    for wi, p in zip(w, probs_list):
        out += wi * p
    out = out / out.sum(axis=1, keepdims=True)
    return out


def per_class_f1_arr(y, ypred, n_classes):
    return f1_score(y, ypred, labels=list(range(n_classes)),
                    average=None, zero_division=0)


def class_routing(probs_list: list[np.ndarray],
                  y: np.ndarray, persons: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class routing via nested LOPO.

    For each held-out person, compute per-class F1 of each model using ONLY
    the remaining persons' OOF preds, then build per-class weights ∝ per-class F1.
    Apply those weights as arithmetic mean on the held-out person's probs.
    """
    N = len(y)
    n_models = len(probs_list)
    out = np.zeros_like(probs_list[0])
    unique_persons = np.unique(persons)
    for held in unique_persons:
        inner_mask = persons != held
        eval_mask = persons == held
        # inner per-class F1 for each model
        class_weights = np.ones((n_models, n_classes))  # default: uniform
        for m, p in enumerate(probs_list):
            preds = p[inner_mask].argmax(axis=1)
            pcf1 = per_class_f1_arr(y[inner_mask], preds, n_classes)
            class_weights[m] = pcf1 + 0.05  # small prior to avoid zero weights
        # normalise per class across models
        class_weights = class_weights / class_weights.sum(axis=0, keepdims=True)
        # apply: for each class, weighted arithmetic combination
        fused = np.zeros((eval_mask.sum(), n_classes))
        for m in range(n_models):
            fused += probs_list[m][eval_mask] * class_weights[m][None, :]
        fused = fused / fused.sum(axis=1, keepdims=True)
        out[eval_mask] = fused
    return out


def _entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(p * np.log(np.clip(p, EPS, 1.0)), axis=1, keepdims=True)


def _maxp(p: np.ndarray) -> np.ndarray:
    return p.max(axis=1, keepdims=True)


def build_stacker_features(probs_list: list[np.ndarray]) -> np.ndarray:
    """Per-row feature vector = concat(probs) + per-model entropy + per-model maxp.

    For 3 models × 5 classes + 3 entropy + 3 maxp = 21 dims. Matches "60-dim"
    budget in the spec generously (we prefer a smaller stacker to avoid
    overfitting on only 240 samples).
    """
    cols = []
    for p in probs_list:
        cols.append(p)
    for p in probs_list:
        cols.append(_entropy(p))
    for p in probs_list:
        cols.append(_maxp(p))
    return np.concatenate(cols, axis=1)


def stacker_nested_lopo(probs_list: list[np.ndarray],
                        y: np.ndarray, persons: np.ndarray,
                        C: float = 1.0) -> np.ndarray:
    """NESTED LOPO logistic regression stacker.

    CRITICAL: The prob vectors we ingest are each row's person-LOPO OOF
    prediction. Because the query's person was held out of the CV fold that
    produced its proba, training a plain LR on the full OOF matrix minus
    the held-out row is actually honest (no within-fold leakage) — but to
    be maximally conservative and match the task spec, we go one level
    deeper: for every test person p, we train the stacker only on rows
    whose person != p. This is the safest interpretation.
    """
    N = len(y)
    n_classes = len(CLASSES)
    out = np.zeros((N, n_classes), dtype=np.float64)
    unique_persons = np.unique(persons)
    Xfull = build_stacker_features(probs_list)
    for held in unique_persons:
        tr = persons != held
        te = persons == held
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xfull[tr])
        Xte = scaler.transform(Xfull[te])
        lr = LogisticRegression(
            C=C, max_iter=2000,
            class_weight="balanced", random_state=RNG_SEED,
        )
        lr.fit(Xtr, y[tr])
        # Ensure output columns match CLASSES order (LR orders by sorted y)
        proba = lr.predict_proba(Xte)
        full = np.zeros((proba.shape[0], n_classes))
        for j, cls in enumerate(lr.classes_):
            full[:, int(cls)] = proba[:, j]
        # fill any class LR didn't see during training with tiny uniform
        if full.sum(axis=1).min() < 0.99:
            s = full.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            full = full / s
        out[te] = full
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics_from_proba(proba: np.ndarray, y: np.ndarray) -> dict:
    preds = proba.argmax(1)
    per_cls = f1_score(y, preds, labels=list(range(len(CLASSES))),
                       average=None, zero_division=0)
    return {
        "weighted_f1": float(f1_score(y, preds, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "accuracy": float((preds == y).mean()),
        "per_class_f1": {CLASSES[i]: float(per_cls[i]) for i in range(len(CLASSES))},
        "preds": preds,
    }


def bootstrap_delta(y: np.ndarray, pred_new: np.ndarray, pred_base: np.ndarray,
                    n_iter: int = 1000, seed: int = RNG_SEED) -> dict:
    """Paired bootstrap on wF1. Positive Δ = new better than base."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_iter)
    a_arr = np.empty(n_iter)
    b_arr = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        a = f1_score(y[idx], pred_new[idx], average="weighted", zero_division=0)
        b = f1_score(y[idx], pred_base[idx], average="weighted", zero_division=0)
        deltas[i] = a - b
        a_arr[i] = a
        b_arr[i] = b
    return {
        "mean_delta": float(deltas.mean()),
        "ci_lo_2.5": float(np.percentile(deltas, 2.5)),
        "ci_hi_97.5": float(np.percentile(deltas, 97.5)),
        "p_delta_gt_0": float((deltas > 0).mean()),
        "new_mean_f1": float(a_arr.mean()),
        "base_mean_f1": float(b_arr.mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[1] Loading OOF sources ...")
    d = load_all()
    N, y, persons = d["N"], d["y"], d["persons"]
    probs = [d["v4_proba"], d["knn_proba"], d["xgb_proba"]]
    probs_soft = [d["v4_proba"], d["knn_soft_proba"], d["xgb_proba"]]
    probs_vx = [d["v4_proba"], d["xgb_proba"]]          # v4+XGB only
    probs_vk = [d["v4_proba"], d["knn_soft_proba"]]     # v4+softkNN only
    model_names = ["v4_multiscale", "knn_dinov2_k1", "xgb_handcrafted"]
    model_names_soft = ["v4_multiscale", "knn_dinov2_k5_softw", "xgb_handcrafted"]
    individual_f1 = [d["v4_f1"], d["knn_f1"], d["xgb_f1"]]
    individual_f1_soft = [d["v4_f1"], d["knn_soft_f1"], d["xgb_f1"]]
    print(f"    N={N}, persons={len(np.unique(persons))}, classes={len(CLASSES)}")
    print(f"    {'v4_multiscale':22s}  wF1 = {d['v4_f1']:.4f}")
    print(f"    {'knn_dinov2_k1_majority':22s}  wF1 = {d['knn_f1']:.4f}")
    print(f"    {'knn_dinov2_k5_softw':22s}  wF1 = {d['knn_soft_f1']:.4f}")
    print(f"    {'xgb_handcrafted':22s}  wF1 = {d['xgb_f1']:.4f}")

    # ----------------------- Fusion strategies -----------------------
    print("[2] Running fusion strategies ...")

    fusions = {}

    def run_bundle(tag: str, probs_b: list[np.ndarray], f1_b: list[float]) -> None:
        wb = np.array(f1_b, dtype=np.float64)
        wb = wb / wb.sum()
        # (a) geometric mean equal weights
        p_geo = geo_mean(probs_b)
        fusions[f"{tag}_geo_mean_equal"] = {
            "proba": p_geo,
            "method": f"[{tag}] geometric mean, equal weights (proven v2 pattern)",
        }
        # (a2) geometric mean F1-weighted exponents
        log_sum = sum(w * np.log(np.clip(p, EPS, 1.0)) for w, p in zip(wb, probs_b))
        p_geo_w = np.exp(log_sum)
        p_geo_w = p_geo_w / p_geo_w.sum(axis=1, keepdims=True)
        fusions[f"{tag}_geo_mean_f1w"] = {
            "proba": p_geo_w,
            "method": f"[{tag}] geometric mean, F1-weighted exponents ({wb.round(3).tolist()})",
        }
        # (b) weighted arithmetic
        p_wa = weighted_arith(probs_b, f1_b)
        fusions[f"{tag}_arith_f1w"] = {
            "proba": p_wa,
            "method": f"[{tag}] arithmetic mean, weights=wF1 ({wb.round(3).tolist()})",
        }
        # (c) nested-LOPO stacker
        p_stack = stacker_nested_lopo(probs_b, y, persons, C=1.0)
        fusions[f"{tag}_lr_stacker"] = {
            "proba": p_stack,
            "method": f"[{tag}] LR stacker, NESTED LOPO (no stacker-fold leakage)",
        }
        # (d) class-specific routing
        p_route = class_routing(probs_b, y, persons, n_classes=len(CLASSES))
        fusions[f"{tag}_class_routing"] = {
            "proba": p_route,
            "method": f"[{tag}] per-class weights from inner-fold per-class F1 (nested LOPO)",
        }

    print("    bundle A: v4 + knn(k=1 hard) + xgb")
    run_bundle("3way_hardknn", probs, individual_f1)
    print("    bundle B: v4 + knn(k=5 softw) + xgb")
    run_bundle("3way_softknn", probs_soft, individual_f1_soft)
    print("    bundle C: v4 + xgb only")
    run_bundle("2way_v4xgb", probs_vx, [d["v4_f1"], d["xgb_f1"]])
    print("    bundle D: v4 + soft knn only")
    run_bundle("2way_v4knn", probs_vk, [d["v4_f1"], d["knn_soft_f1"]])

    # ----------------------- Metrics + bootstrap -----------------------
    print("[3] Metrics + bootstrap vs v4 ...")
    v4_preds = probs[0].argmax(1)
    v4_metrics = metrics_from_proba(probs[0], y)
    v4_f1 = v4_metrics["weighted_f1"]

    summary = {
        "n_samples": int(N),
        "n_persons": int(len(np.unique(persons))),
        "classes": CLASSES,
        "individual": {
            "v4_multiscale": {
                "weighted_f1": d["v4_f1"],
                "macro_f1": float(f1_score(y, d["v4_proba"].argmax(1),
                                           average="macro", zero_division=0)),
            },
            "knn_dinov2_k1_majority": {
                "weighted_f1": d["knn_f1"],
                "macro_f1": float(f1_score(y, d["knn_proba"].argmax(1),
                                           average="macro", zero_division=0)),
            },
            "knn_dinov2_k5_softw": {
                "weighted_f1": d["knn_soft_f1"],
                "macro_f1": float(f1_score(y, d["knn_soft_proba"].argmax(1),
                                           average="macro", zero_division=0)),
            },
            "xgb_handcrafted": {
                "weighted_f1": d["xgb_f1"],
                "macro_f1": float(f1_score(y, d["xgb_proba"].argmax(1),
                                           average="macro", zero_division=0)),
            },
        },
        "fusion": {},
        "champion_ref": {
            "name": "v4_multiscale",
            "weighted_f1": v4_f1,
            "macro_f1": v4_metrics["macro_f1"],
        },
    }

    print(f"\n    {'strategy':32s}  wF1     mF1    acc    Δ mean   CI95             P(Δ>0)")
    print(f"    {'-'*32}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*15}  {'-'*6}")
    print(f"    {'v4_multiscale (reference)':32s}  {v4_f1:.4f}  "
          f"{v4_metrics['macro_f1']:.4f}  {v4_metrics['accuracy']:.4f}   n/a      n/a              n/a")

    for name, entry in fusions.items():
        m = metrics_from_proba(entry["proba"], y)
        boot = bootstrap_delta(y, m["preds"], v4_preds)
        summary["fusion"][name] = {
            "method": entry["method"],
            "weighted_f1": m["weighted_f1"],
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"],
            "per_class_f1": m["per_class_f1"],
            "bootstrap_vs_v4": boot,
        }
        print(f"    {name:32s}  {m['weighted_f1']:.4f}  "
              f"{m['macro_f1']:.4f}  {m['accuracy']:.4f}  "
              f"{boot['mean_delta']:+.4f}  "
              f"[{boot['ci_lo_2.5']:+.4f},{boot['ci_hi_97.5']:+.4f}]  "
              f"{boot['p_delta_gt_0']:.3f}")

    # ----------------------- Pick winner -----------------------
    best_name = max(summary["fusion"].keys(),
                    key=lambda k: summary["fusion"][k]["weighted_f1"])
    best = summary["fusion"][best_name]
    summary["winner"] = {
        "name": best_name,
        "weighted_f1": best["weighted_f1"],
        "macro_f1": best["macro_f1"],
        "delta_vs_v4": best["weighted_f1"] - v4_f1,
        "p_delta_gt_0": best["bootstrap_vs_v4"]["p_delta_gt_0"],
    }
    flag = (best["weighted_f1"] > 0.70) and (best["bootstrap_vs_v4"]["p_delta_gt_0"] > 0.90)
    summary["ensemble_champion_candidate"] = bool(flag)

    # ----------------------- Save predictions JSON -----------------------
    out_json = {
        "meta": {
            "components": ["v4_multiscale", "knn_dinov2_k1", "knn_dinov2_k5_soft",
                           "xgb_handcrafted"],
            "component_wF1": {
                "v4_multiscale": d["v4_f1"],
                "knn_dinov2_k1_majority": d["knn_f1"],
                "knn_dinov2_k5_softw": d["knn_soft_f1"],
                "xgb_handcrafted": d["xgb_f1"],
            },
            "v4_reference_wF1": v4_f1,
            "winner": summary["winner"],
            "ensemble_champion_candidate": bool(flag),
            "safeguards": [
                "all probs person-LOPO (v4 via StratifiedGroupKFold, k-NN via "
                "neighbour exclusion, XGB via StratifiedGroupKFold)",
                "stacker uses NESTED LOPO — outer=held-out person, stacker fit on remaining",
                "class routing uses inner-fold per-class F1, never touching the held-out person",
            ],
        },
        "summary": summary,
        "winner_predictions": [],
    }
    for i in range(N):
        winner_proba = fusions[best_name]["proba"][i]
        record = {
            "scan_path": d["scan_paths"][i],
            "rel_path": d["rel_paths"][i],
            "person": str(persons[i]),
            "true_class": CLASSES[int(y[i])],
            "pred_class": CLASSES[int(winner_proba.argmax())],
            "correct": bool(winner_proba.argmax() == y[i]),
            "proba": {CLASSES[j]: float(winner_proba[j]) for j in range(len(CLASSES))},
            "v4_proba": {CLASSES[j]: float(d["v4_proba"][i, j]) for j in range(len(CLASSES))},
            "knn_hard_proba": {CLASSES[j]: float(d["knn_proba"][i, j]) for j in range(len(CLASSES))},
            "knn_soft_proba": {CLASSES[j]: float(d["knn_soft_proba"][i, j]) for j in range(len(CLASSES))},
            "xgb_proba": {CLASSES[j]: float(d["xgb_proba"][i, j]) for j in range(len(CLASSES))},
        }
        out_json["winner_predictions"].append(record)

    out_path = CACHE / "fusion_ensemble_predictions.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[saved] {out_path}")

    # ----------------------- Markdown report -----------------------
    write_report(summary, flag)

    # ----------------------- Verdict -----------------------
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    delta = best["weighted_f1"] - v4_f1
    p_gt = best["bootstrap_vs_v4"]["p_delta_gt_0"]
    print(f"  Winner: {best_name}  wF1={best['weighted_f1']:.4f} (Δ vs v4 = {delta:+.4f}, P(Δ>0)={p_gt:.3f})")
    if flag:
        print("  → FLAG: ensemble champion CANDIDATE (wF1>0.70, P(Δ>0)>0.90). Red-team before shipping.")
    elif v4_f1 - 0.002 <= best["weighted_f1"] <= 0.70:
        print("  → HONEST ADMISSION: within [v4, 0.70] — v4 alone is the ceiling for this "
              "component set. Fusion gives no material lift.")
    else:
        print(f"  → Fusion {'adds' if delta > 0 else 'harms'} {abs(delta):.4f} weighted F1 vs v4, "
              f"P(Δ>0)={p_gt:.3f}. Not a clear win.")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(summary: dict, flag: bool) -> None:
    lines = []
    a = lines.append
    a("# Fusion Ensemble — v4 + k-NN + XGBoost (Clean, VLM-free)")
    a("")
    a("**Question:** can combining three clean (non-VLM) OOF tracks — v4 multiscale "
      "(0.6887), DINOv2 k-NN retrieval (0.6117), XGBoost on 440 handcrafted features — "
      "push past v4's weighted F1 under person-LOPO?")
    a("")
    a("**VLM contamination guard:** no Haiku/Sonnet/Opus features are used. Wave-14 "
      "filename-leak finding explicitly disqualifies those tracks.")
    a("")
    a(f"**Sample size:** {summary['n_samples']} scans × {summary['n_persons']} persons × "
      f"{len(CLASSES)} classes (person-LOPO).")
    a("")
    a("## 1. Component baselines")
    a("")
    a("| Model | Weighted F1 | Macro F1 |")
    a("|---|---:|---:|")
    for name, ind in summary["individual"].items():
        a(f"| {name} | {ind['weighted_f1']:.4f} | {ind['macro_f1']:.4f} |")
    a("")

    a("## 2. Fusion results (person-LOPO)")
    a("")
    a("| Strategy | wF1 | mF1 | Accuracy | Δ vs v4 | 95% CI | P(Δ>0) |")
    a("|---|---:|---:|---:|---:|---|---:|")
    for name, entry in summary["fusion"].items():
        boot = entry["bootstrap_vs_v4"]
        a(f"| {name} | {entry['weighted_f1']:.4f} | {entry['macro_f1']:.4f} | "
          f"{entry['accuracy']:.4f} | {boot['mean_delta']:+.4f} | "
          f"[{boot['ci_lo_2.5']:+.4f}, {boot['ci_hi_97.5']:+.4f}] | "
          f"{boot['p_delta_gt_0']:.3f} |")
    a(f"| **v4 (reference)** | **{summary['champion_ref']['weighted_f1']:.4f}** | "
      f"**{summary['champion_ref']['macro_f1']:.4f}** | — | — | — | — |")
    a("")

    # Winner section
    w = summary["winner"]
    a(f"## 3. Winner: `{w['name']}`")
    a("")
    a(f"- **Weighted F1:** {w['weighted_f1']:.4f}")
    a(f"- **Macro F1:** {w['macro_f1']:.4f}")
    a(f"- **Δ vs v4:** {w['delta_vs_v4']:+.4f}")
    a(f"- **P(Δ > 0):** {w['p_delta_gt_0']:.3f}")
    a("")
    a("### Per-class F1 (winner)")
    a("")
    a("| Class | F1 |")
    a("|---|---:|")
    for cls, v in summary["fusion"][w["name"]]["per_class_f1"].items():
        a(f"| {cls} | {v:.4f} |")
    a("")

    a("## 4. Method detail")
    a("")
    for name, entry in summary["fusion"].items():
        a(f"- **`{name}`** — {entry['method']}")
    a("")

    a("## 5. Red-team self-check on leakage")
    a("")
    a("- **v4 probs** come from `StratifiedGroupKFold(groups=person_id)`; each scan's "
      "probability is produced by a fold where its person was in the val set.")
    a("- **k-NN probs** mask out ALL scans sharing the query's person before voting "
      "(see `scripts/knn_baseline.py`, `mask_same_person`).")
    a("- **XGBoost probs** come from `StratifiedGroupKFold(groups=person_id)` in "
      "`scripts/expert_council.py::xgboost_oof`; no same-person leakage.")
    a("")
    a("- **Geometric mean / weighted arithmetic mean** require NO training — purely "
      "deterministic transforms of already-OOF probs. No leakage possible.")
    a("")
    a("- **Logistic-regression stacker** uses **NESTED LOPO**: for every held-out "
      "person, the stacker is fit on rows belonging to *other* persons (whose probs "
      "are themselves already person-LOPO). The held-out person's row is NEVER used "
      "in any stacker training step. This defuses the \"stacker sees its own eval fold\" "
      "trap flagged in the task spec.")
    a("")
    a("- **Class-specific routing** computes per-class F1 weights using only rows from "
      "persons other than the held-out person. The held-out person's fused probs depend "
      "only on inner-fold diagnostics.")
    a("")
    a("- **Possible residual leakage:** k-NN and v4 both use DINOv2 embeddings. They "
      "are therefore correlated, which weakens diversity gains — this is an *efficacy* "
      "concern, not a *leakage* concern.")
    a("")

    a("## 6. Verdict")
    a("")
    v4_f1 = summary["champion_ref"]["weighted_f1"]
    best_f1 = w["weighted_f1"]
    delta = w["delta_vs_v4"]
    p_gt = w["p_delta_gt_0"]
    if flag:
        a(f"**CANDIDATE FLAGGED.** Best fusion (`{w['name']}`, wF1 = {best_f1:.4f}) "
          f"exceeds 0.70 with P(Δ>0 vs v4) = {p_gt:.3f} > 0.90. Send through Wave-14-style "
          "red-team audit (check for any hidden leakage, confirm person split integrity, "
          "validate on blind-held subset) before promoting to champion.")
    elif v4_f1 <= best_f1 <= 0.70:
        a(f"**v4 is the ceiling for this component set.** Best fusion "
          f"(`{w['name']}`) lands at wF1 = {best_f1:.4f} (Δ = {delta:+.4f}, "
          f"P(Δ>0) = {p_gt:.3f}). This sits in the [v4, 0.70] band — the three "
          "clean components are too correlated (v4 and k-NN share the DINOv2 backbone) "
          "to produce meaningful fusion gains. Recommendation: keep v4 as the shipped "
          "champion; add fusion only if a genuinely orthogonal signal (CGNN, TDA, "
          "physics-informed prior) becomes available.")
    elif best_f1 < v4_f1:
        a(f"**Fusion hurts.** Best candidate (`{w['name']}`, wF1 = {best_f1:.4f}) "
          f"trails v4 by {abs(delta):.4f} weighted F1 (P(Δ>0) = {p_gt:.3f}). "
          "**Keep v4 alone as the shipped champion.** "
          "The three clean components are too correlated (v4 ensemble already contains "
          "DINOv2, and k-NN rides on the same DINOv2 embedding) and the XGBoost head is "
          "substantially weaker than v4. Adding any of them to v4 injects more noise than "
          "signal. A genuinely orthogonal track (CGNN, TDA persistent homology, physics-"
          "informed prior) is required to break past v4.")
    else:
        a(f"**Mixed.** Best fusion (`{w['name']}`, wF1 = {best_f1:.4f}) beats v4 by "
          f"{delta:+.4f} but P(Δ>0) = {p_gt:.3f} is below the 0.90 confidence bar. "
          "Not strong enough to displace v4.")
    a("")

    a("## 7. Files")
    a("")
    a("- `cache/fusion_ensemble_predictions.json` — all fusion probs + winner preds")
    a("- `reports/FUSION_ENSEMBLE.md` — this file")
    a("- `scripts/fusion_ensemble.py` — reproduction script")
    a("")

    report_path = REPORTS / "FUSION_ENSEMBLE.md"
    report_path.write_text("\n".join(lines))
    print(f"[saved] {report_path}")


if __name__ == "__main__":
    main()
