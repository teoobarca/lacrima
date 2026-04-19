"""Evaluate advanced features with person-level LOPO.

Three experiments per feature set:
    A. XGBoost on all features (handcrafted + advanced, ~350-dim)
    B. XGBoost on top-50 features selected inside each training fold
       (univariate F-test, XGB importance, L1-LR — all nested).
    C. LR + StandardScaler on top-50 F-test features (nested).

Bonus: concat of top-50 advanced + DINOv2-B (768-dim) -> XGBoost.

Usage:
    .venv/bin/python scripts/eval_advanced_features.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.data import CLASSES, enumerate_samples, person_id  # noqa: E402
from teardrop.cv import leave_one_patient_out  # noqa: E402

CACHE = ROOT / "cache" / "features_advanced.parquet"
DINOV2_CACHE = ROOT / "cache" / "_scan_mean_emb_dinov2b.npz"
REPORT = ROOT / "reports" / "ADVANCED_FEATURES_RESULTS.md"

# The advanced module defines exactly these groups by prefix.
ADVANCED_PREFIXES = (
    "mf_", "lac_", "suc_", "wp_", "hurst_", "dfa_", "trend_",
    "xglcm_", "gabor_", "mhog_",
)
META_COLS = ("raw", "cls", "label", "patient", "person")


def _xgb_params(n_classes: int) -> dict:
    return dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=1.5,
        reg_alpha=0.5,
        random_state=42,
        n_jobs=4,
        objective="multi:softprob",
        num_class=n_classes,
        tree_method="hist",
    )


def _sample_weights(y: np.ndarray) -> np.ndarray:
    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
    lookup = {c: w for c, w in zip(np.unique(y), cw)}
    return np.asarray([lookup[label] for label in y])


def person_lopo_eval(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_fn,
) -> dict:
    """Run leave-one-PERSON-out. model_fn(X_tr, y_tr, w_tr) -> (pred_fn, info).
    pred_fn(X_va) -> class predictions (int array)."""
    w = _sample_weights(y)
    n = len(y)
    preds = np.full(n, -1, dtype=int)
    for tr, va in leave_one_patient_out(groups):
        pred_fn, _ = model_fn(X[tr], y[tr], w[tr])
        preds[va] = pred_fn(X[va])
    f1w = f1_score(y, preds, average="weighted")
    f1m = f1_score(y, preds, average="macro")
    return dict(
        preds=preds,
        weighted_f1=float(f1w),
        macro_f1=float(f1m),
        report=classification_report(y, preds, target_names=CLASSES, zero_division=0),
        cm=confusion_matrix(y, preds, labels=list(range(len(CLASSES)))),
    )


# ---- model factories -------------------------------------------------------

def make_xgb_full():
    def fn(X_tr, y_tr, w_tr):
        clf = XGBClassifier(**_xgb_params(len(CLASSES)))
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        return clf.predict, dict()
    return fn


def make_xgb_topk_ftest(k: int):
    """Select top-k by univariate F-test on TRAIN only, fit XGB on reduced X."""
    def fn(X_tr, y_tr, w_tr):
        k_eff = min(k, X_tr.shape[1])
        selector = SelectKBest(f_classif, k=k_eff)
        Xs = selector.fit_transform(X_tr, y_tr)
        clf = XGBClassifier(**_xgb_params(len(CLASSES)))
        clf.fit(Xs, y_tr, sample_weight=w_tr)

        def predict(X_va: np.ndarray) -> np.ndarray:
            return clf.predict(selector.transform(X_va))

        return predict, dict(selected=selector.get_support(indices=True))
    return fn


def make_xgb_topk_importance(k: int):
    """Two-stage XGB: fit on all, keep top-k by gain, re-fit on those."""
    def fn(X_tr, y_tr, w_tr):
        pre = XGBClassifier(**_xgb_params(len(CLASSES)))
        pre.fit(X_tr, y_tr, sample_weight=w_tr)
        imp = pre.feature_importances_
        k_eff = min(k, X_tr.shape[1])
        idx = np.argsort(imp)[::-1][:k_eff]
        clf = XGBClassifier(**_xgb_params(len(CLASSES)))
        clf.fit(X_tr[:, idx], y_tr, sample_weight=w_tr)

        def predict(X_va: np.ndarray) -> np.ndarray:
            return clf.predict(X_va[:, idx])

        return predict, dict(selected=idx)
    return fn


def make_lr_topk(k: int):
    def fn(X_tr, y_tr, w_tr):
        k_eff = min(k, X_tr.shape[1])
        selector = SelectKBest(f_classif, k=k_eff)
        Xs = selector.fit_transform(X_tr, y_tr)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                multi_class="multinomial",
                random_state=42,
            )),
        ])
        pipe.fit(Xs, y_tr)

        def predict(X_va: np.ndarray) -> np.ndarray:
            return pipe.predict(selector.transform(X_va))

        return predict, dict(selected=selector.get_support(indices=True))
    return fn


# ---- importance aggregation ------------------------------------------------

def global_xgb_importance(
    X: np.ndarray,
    y: np.ndarray,
    feat_names: list[str],
    top: int = 30,
) -> list[tuple[str, float]]:
    w = _sample_weights(y)
    clf = XGBClassifier(**_xgb_params(len(CLASSES)))
    clf.fit(X, y, sample_weight=w)
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:top]
    return [(feat_names[i], float(imp[i])) for i in order]


# ---- main ------------------------------------------------------------------

def main() -> None:  # noqa: C901
    print(f"Loading {CACHE}")
    df = pd.read_parquet(CACHE)
    # Ensure person column: earlier cache may not have it.
    if "person" not in df.columns:
        df["person"] = df["raw"].map(lambda p: person_id(Path(p)))

    feature_cols = [c for c in df.columns if c not in META_COLS]
    adv_cols = [c for c in feature_cols if c.startswith(ADVANCED_PREFIXES)]
    basic_cols = [c for c in feature_cols if c not in adv_cols]
    print(
        f"  feature counts: basic={len(basic_cols)}, advanced={len(adv_cols)}, "
        f"total={len(feature_cols)}"
    )

    y = df["label"].values.astype(int)
    groups = df["person"].values
    n_persons = len(np.unique(groups))
    print(f"  n_samples={len(y)}, n_persons={n_persons}")

    all_results: dict[str, dict] = {}

    def evaluate(name: str, Xcols: list[str], model_fn, store_sel=False):
        print(f"\n=== {name}   (d={len(Xcols)}) ===")
        X = df[Xcols].values.astype(np.float64)
        # Replace any residual NaN/inf just in case.
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        t0 = time.time()
        res = person_lopo_eval(X, y, groups, model_fn)
        print(f"  weighted F1 = {res['weighted_f1']:.4f}")
        print(f"  macro    F1 = {res['macro_f1']:.4f}")
        print(f"  wall = {time.time()-t0:.1f}s")
        all_results[name] = {
            "weighted_f1": res["weighted_f1"],
            "macro_f1": res["macro_f1"],
            "n_features": len(Xcols),
        }
        return res

    # Experiment 1: handcrafted only (sanity re-check)
    evaluate("A1_handcrafted94_XGB_full", basic_cols, make_xgb_full())

    # Experiment 2: advanced only - full ~256
    evaluate("A2_advanced_XGB_full", adv_cols, make_xgb_full())

    # Experiment 3: combined ~350 full
    evaluate("A3_combined_XGB_full", feature_cols, make_xgb_full())

    # Experiment 4: combined, top-50 by F-test nested
    evaluate("B1_combined_top50_ftest_XGB", feature_cols, make_xgb_topk_ftest(50))

    # Experiment 5: combined, top-50 by XGB importance nested
    evaluate("B2_combined_top50_xgbimp_XGB", feature_cols, make_xgb_topk_importance(50))

    # Experiment 6: combined, top-50 F-test + LR+scaler
    evaluate("C1_combined_top50_ftest_LR", feature_cols, make_lr_topk(50))

    # Experiment 7: advanced only, top-50 F-test nested
    evaluate("B3_advanced_top50_ftest_XGB", adv_cols, make_xgb_topk_ftest(50))

    # Bonus: top-50 advanced + DINOv2-B concat
    if DINOV2_CACHE.exists():
        print("\n=== Bonus: top-50 advanced (F-test) + DINOv2-B concat ===")
        dz = np.load(DINOV2_CACHE, allow_pickle=True)
        emb = dz["emb"].astype(np.float32)       # (240, 768)
        emb_paths = [str(p) for p in dz["scan_paths"]]
        # Align rows by raw path.
        idx_by_path = {p: i for i, p in enumerate(emb_paths)}
        order = np.asarray([idx_by_path[p] for p in df["raw"].tolist()])
        emb_aligned = emb[order]
        # Advanced-only feature block -> nested top-50 F-test selection.
        Xadv_all = df[adv_cols].values.astype(np.float64)
        Xadv_all = np.nan_to_num(Xadv_all, nan=0.0, posinf=0.0, neginf=0.0)

        def concat_topk_fn(X_tr, y_tr, w_tr):
            # X_tr is the full concat [adv | emb]; we only select from adv.
            n_adv = Xadv_all.shape[1]
            k = min(50, n_adv)
            sel = SelectKBest(f_classif, k=k)
            sel.fit(X_tr[:, :n_adv], y_tr)
            sel_idx_adv = sel.get_support(indices=True)
            full_idx = np.concatenate([sel_idx_adv, np.arange(n_adv, X_tr.shape[1])])
            clf = XGBClassifier(**_xgb_params(len(CLASSES)))
            clf.fit(X_tr[:, full_idx], y_tr, sample_weight=w_tr)

            def predict(X_va: np.ndarray) -> np.ndarray:
                return clf.predict(X_va[:, full_idx])
            return predict, dict(selected=full_idx)

        Xcat = np.concatenate([Xadv_all, emb_aligned.astype(np.float64)], axis=1)
        w = _sample_weights(y)
        preds = np.full(len(y), -1, dtype=int)
        for tr, va in leave_one_patient_out(groups):
            pfn, _ = concat_topk_fn(Xcat[tr], y[tr], w[tr])
            preds[va] = pfn(Xcat[va])
        f1w = f1_score(y, preds, average="weighted")
        f1m = f1_score(y, preds, average="macro")
        print(f"  weighted F1 = {f1w:.4f}")
        print(f"  macro    F1 = {f1m:.4f}")
        all_results["D1_top50adv_plus_DINOv2B_XGB"] = {
            "weighted_f1": float(f1w),
            "macro_f1": float(f1m),
            "n_features": Xcat.shape[1],
        }

        # Plain DINOv2-B-alone baseline re-check with XGBoost on this config
        w = _sample_weights(y)
        preds = np.full(len(y), -1, dtype=int)
        for tr, va in leave_one_patient_out(groups):
            clf = XGBClassifier(**_xgb_params(len(CLASSES)))
            clf.fit(emb_aligned[tr], y[tr], sample_weight=w[tr])
            preds[va] = clf.predict(emb_aligned[va])
        f1w = f1_score(y, preds, average="weighted")
        f1m = f1_score(y, preds, average="macro")
        print(f"  DINOv2-B alone (XGB) w-F1={f1w:.4f}  m-F1={f1m:.4f}")
        all_results["D0_DINOv2B_only_XGB"] = {
            "weighted_f1": float(f1w),
            "macro_f1": float(f1m),
            "n_features": 768,
        }
    else:
        print(f"[warn] {DINOV2_CACHE} not found -> skipping bonus.")

    # ---- Global importance on top ----------------------------------------
    Xall = df[feature_cols].values.astype(np.float64)
    Xall = np.nan_to_num(Xall)
    global_imp = global_xgb_importance(Xall, y, feature_cols, top=30)
    print("\n=== Top-30 global XGB importance (full-fit, all samples) ===")
    for name, v in global_imp:
        print(f"  {name:40s} {v:.4f}")

    # Write report.
    write_report(all_results, basic_cols, adv_cols, feature_cols, global_imp)


def write_report(
    results: dict,
    basic_cols: list[str],
    adv_cols: list[str],
    all_cols: list[str],
    global_imp: list[tuple[str, float]],
) -> None:
    lines: list[str] = []
    a = lines.append
    a("# Advanced Texture Features — Results\n")
    a("_Person-level LOPO on TRAIN_SET (n=240 scans, 35 persons, 5 classes)._\n")
    a("")
    a("## Methodology")
    a("")
    a("We engineered ~256 advanced texture features targeted at tear-crystallization")
    a("physics and concatenated them with the existing 94-dim handcrafted set to give")
    a("a combined feature bank of 349 dims. All evaluations are person-level")
    a("leave-one-out (35 persons → 35 folds).")
    a("")
    a(f"- Handcrafted features (reused from `teardrop/features.py`): **{len(basic_cols)}**")
    a(f"- Advanced features (`teardrop/features_advanced.py`): **{len(adv_cols)}**")
    a(f"- Combined: **{len(all_cols)}**")
    a("")
    a("Feature families implemented:")
    a("1. Multifractal spectrum f(α) via box-counting q-moments (q ∈ {-5,-3,-1,0,1,3,5})")
    a("2. Lacunarity (gliding box, Allain-Cloitre) at scales {4,8,16,32,64}")
    a("3. Succolarity (4 directional flood-permeabilities + summary)")
    a("4. Wavelet-packet tree energies (db4, level=3 → 64 subbands)")
    a("5. Hurst exponent (R/S) + DFA on row/col/flat signals + row-trend CV")
    a("6. Extended GLCM (8 distances × 8 angles × 6 Haralick props, summarized)")
    a("7. Gabor bank (8 orientations × 4 frequencies, mean/std/energy/entropy)")
    a("8. Multi-scale HOG (cells ∈ {8,16,32}, 9 orientations)")
    a("")
    a("Feature selection is **nested** inside each LOPO fold (F-test on TRAIN only,")
    a("never on OOF). Top-50 is always selected.")
    a("")
    a("## Results summary (weighted F1, person-LOPO)")
    a("")
    a("| Experiment | n features | Weighted F1 | Macro F1 |")
    a("|---|---:|---:|---:|")
    for name, r in results.items():
        a(f"| {name} | {r['n_features']} | {r['weighted_f1']:.4f} | {r['macro_f1']:.4f} |")
    a("")
    # Baseline comparison.
    champ_w = 0.6458
    champ_label = "DINOv2-B+BiomedCLIP TTA ensemble (person-LOPO)"
    dino_alone = 0.615
    hand_alone = 0.49
    best = max(results.items(), key=lambda kv: kv[1]["weighted_f1"])
    a("## Comparison to baselines")
    a("")
    a("| Baseline | Weighted F1 |")
    a("|---|---:|")
    a(f"| Handcrafted 94 (prior) | {hand_alone:.3f} |")
    a(f"| DINOv2-B alone (prior) | {dino_alone:.3f} |")
    a(f"| {champ_label} | {champ_w:.4f} |")
    a(f"| **Best advanced experiment ({best[0]})** | **{best[1]['weighted_f1']:.4f}** |")
    a("")
    delta_hand = best[1]["weighted_f1"] - hand_alone
    delta_dino = best[1]["weighted_f1"] - dino_alone
    delta_tta = best[1]["weighted_f1"] - champ_w
    a(f"Δ vs handcrafted-only: **{delta_hand:+.3f}**")
    a(f"Δ vs DINOv2-B-alone: **{delta_dino:+.3f}**")
    a(f"Δ vs TTA ensemble (shipped): **{delta_tta:+.3f}**")
    a("")
    a("## Top-30 global XGB feature importances (fit on full 240, all 349 features)")
    a("")
    a("| Feature | Importance |")
    a("|---|---:|")
    for name, v in global_imp:
        a(f"| {name} | {v:.4f} |")
    a("")
    a("_Important note_: these are fit on all data, so they only indicate which")
    a("features carry the most usable signal globally — they are NOT the selected")
    a("features during nested LOPO evaluation.")
    a("")
    a("## Honest commentary")
    a("")
    if delta_hand > 0.0:
        a(f"- Advanced features **do beat handcrafted-alone** (+{delta_hand:.3f}).")
    else:
        a(f"- Advanced features **do not beat handcrafted-alone** ({delta_hand:+.3f}).")
    if delta_dino > 0.0:
        a(f"- Advanced features **do beat DINOv2-B-alone** (+{delta_dino:.3f}).")
    else:
        a(f"- Advanced features **do not beat DINOv2-B-alone** ({delta_dino:+.3f}).")
    if delta_tta > 0.0:
        a(f"- Advanced features **do beat the shipped TTA ensemble** (+{delta_tta:.3f}).")
    else:
        a(f"- Advanced features **do not beat the shipped TTA ensemble** ({delta_tta:+.3f}).")
    a("")
    a("## Files")
    a("- `teardrop/features_advanced.py` — extractor module (8 families, ~256 feats)")
    a("- `scripts/extract_advanced_features.py` — cache builder")
    a("- `scripts/eval_advanced_features.py` — this evaluator")
    a("- `cache/features_advanced.parquet` — 240 × (349 + meta) cache")

    REPORT.write_text("\n".join(lines))
    print(f"\n[report] wrote {REPORT}")

    # Also dump raw JSON.
    jpath = ROOT / "reports" / "advanced_features_results.json"
    jpath.write_text(json.dumps(results, indent=2))
    print(f"[report] wrote {jpath}")


if __name__ == "__main__":
    main()
