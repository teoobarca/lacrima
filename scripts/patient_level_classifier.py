"""Patient-level classifier: aggregate per-scan embeddings to per-person embeddings, then LR.

Rationale
---------
Organizers may evaluate per-patient (all scans of one person get one label),
in which case training at person granularity is better than training per-scan
and voting. This script produces 35 person-level embeddings per encoder, runs
leave-one-person-out CV on them, maps the predicted class back to every scan
of the held-out person, and compares the resulting scan-level wF1 against v4
(0.6887).

Encoders (all from existing caches):
    A) DINOv2-B 90 nm/px tiled (mean-pooled to scan first, then person)
    B) DINOv2-B 45 nm/px tiled (ditto)
    C) BiomedCLIP D4 TTA (already per-scan)

Pooling variants per person:
    v_mean : mean of L2-normalized scan embeddings  (core approach)
    v_max  : per-dim max of L2-normalized scan embeddings
    v_attn : softmax-weighted mean, weights = v4 scan-level top-1 prob

For each variant we concatenate the 3 encoders (L2-normalized person vectors)
and run LOPO-LR (class_weight balanced). Predictions are broadcast back to
scan-level and compared against v4 OOF predictions via a paired bootstrap.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.data import CLASSES, person_id  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
EPS = 1e-9
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _mean_pool_tiles(X_tiles: np.ndarray, tile_to_scan: np.ndarray, n_scans: int) -> np.ndarray:
    D = X_tiles.shape[1]
    out = np.zeros((n_scans, D), dtype=np.float32)
    cnt = np.zeros(n_scans, dtype=np.int64)
    for i, s in enumerate(tile_to_scan):
        out[s] += X_tiles[i]
        cnt[s] += 1
    cnt = np.maximum(cnt, 1)
    return out / cnt[:, None]


def load_scan_embeddings():
    """Returns (paths, y, persons, {'dino90':X, 'dino45':X, 'biomedclip':X})."""
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz", allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz", allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)

    paths = z90["scan_paths"]
    y = z90["scan_y"].astype(int)
    if not np.array_equal(z45["scan_paths"], paths):
        raise RuntimeError("45nm paths misaligned")
    if not np.array_equal(zbc["scan_paths"], paths):
        raise RuntimeError("biomedclip paths misaligned")

    n_scans = len(paths)
    X90 = _mean_pool_tiles(z90["X"], z90["tile_to_scan"], n_scans)
    X45 = _mean_pool_tiles(z45["X"], z45["tile_to_scan"], n_scans)
    Xbc = zbc["X_scan"].astype(np.float32)

    persons = np.array([person_id(Path(str(p))) for p in paths])
    return paths, y, persons, {"dino90": X90, "dino45": X45, "biomedclip": Xbc}


def load_v4_oof():
    z = np.load(CACHE / "v4_oof_predictions.npz", allow_pickle=True)
    return {
        "proba": z["proba"],
        "y": z["y"].astype(int),
        "persons": z["persons"],
        "scan_paths": z["scan_paths"],
    }


# ---------------------------------------------------------------------------
# Aggregation variants
# ---------------------------------------------------------------------------

def _l2(x: np.ndarray) -> np.ndarray:
    return normalize(x, norm="l2", axis=1)


def aggregate_person(
    X_scan: np.ndarray,
    persons: np.ndarray,
    variant: str,
    scan_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse (n_scan, D) -> (n_person, D). Returns (X_person, person_order).

    variant: 'mean' | 'max' | 'attn' (requires scan_weights)
    """
    X_norm = _l2(X_scan)
    order = sorted(np.unique(persons).tolist())
    D = X_norm.shape[1]
    X_person = np.zeros((len(order), D), dtype=np.float32)
    for pi, pid in enumerate(order):
        m = persons == pid
        if not m.any():
            continue
        group = X_norm[m]
        if variant == "mean":
            v = group.mean(axis=0)
        elif variant == "max":
            v = group.max(axis=0)
        elif variant == "attn":
            if scan_weights is None:
                raise ValueError("attn needs scan_weights")
            w = scan_weights[m]
            w = np.exp(w - w.max())
            w /= w.sum() + EPS
            v = (group * w[:, None]).sum(axis=0)
        else:
            raise ValueError(variant)
        X_person[pi] = v
    return X_person, np.array(order)


def person_labels(persons: np.ndarray, y: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Majority label per person (they are almost always constant anyway)."""
    out = np.zeros(len(order), dtype=int)
    for pi, pid in enumerate(order):
        labels = y[persons == pid]
        out[pi] = Counter(labels.tolist()).most_common(1)[0][0]
    return out


# ---------------------------------------------------------------------------
# LOPO classifier on person level
# ---------------------------------------------------------------------------

def lopo_person_predict(X_person: np.ndarray, y_person: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LOPO LR on person-level dataset. Returns (preds, proba)."""
    n = len(y_person)
    preds = np.full(n, -1, dtype=int)
    proba = np.zeros((n, len(CLASSES)), dtype=np.float32)
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        Xtr = normalize(X_person[tr], norm="l2", axis=1)
        Xva = normalize(X_person[i:i + 1], norm="l2", axis=1)
        sc = StandardScaler().fit(Xtr)
        Xtr_s = sc.transform(Xtr)
        Xva_s = sc.transform(Xva)
        clf = LogisticRegression(
            class_weight="balanced", max_iter=3000, C=1.0,
            solver="lbfgs", n_jobs=1, random_state=42,
        )
        clf.fit(Xtr_s, y_person[tr])
        preds[i] = clf.predict(Xva_s)[0]
        p = clf.predict_proba(Xva_s)[0]
        # Map class indices (LR may have missing classes in train if every sample is dropped,
        # but with 34 train persons and 5 classes balanced this is safe — still guard).
        full = np.zeros(len(CLASSES), dtype=np.float32)
        for ci, cls_id in enumerate(clf.classes_):
            full[cls_id] = p[ci]
        proba[i] = full
    return preds, proba


def broadcast_to_scan(
    person_preds: np.ndarray,
    person_proba: np.ndarray,
    person_order: np.ndarray,
    scan_persons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pid_to_idx = {pid: i for i, pid in enumerate(person_order)}
    scan_preds = np.array([person_preds[pid_to_idx[p]] for p in scan_persons])
    scan_proba = np.stack([person_proba[pid_to_idx[p]] for p in scan_persons])
    return scan_preds, scan_proba


# ---------------------------------------------------------------------------
# Bootstrap comparison
# ---------------------------------------------------------------------------

def paired_bootstrap(y: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray,
                     n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.zeros(n_boot, dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fa = f1_score(y[idx], preds_a[idx], average="weighted")
        fb = f1_score(y[idx], preds_b[idx], average="weighted")
        deltas[b] = fa - fb
    return {
        "mean_delta": float(deltas.mean()),
        "p_gt_zero": float((deltas > 0).mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_concat_person(
    scan_embs: dict,
    persons: np.ndarray,
    variant: str,
    scan_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a concatenated person-level embedding across 3 encoders."""
    parts = []
    order_ref = None
    for key in ("dino90", "dino45", "biomedclip"):
        Xp, order = aggregate_person(scan_embs[key], persons, variant, scan_weights)
        if order_ref is None:
            order_ref = order
        else:
            if not np.array_equal(order_ref, order):
                raise RuntimeError("person order mismatch")
        # L2-norm the person vector so the 3 encoders contribute on equal footing.
        Xp = _l2(Xp)
        parts.append(Xp)
    X_person = np.concatenate(parts, axis=1).astype(np.float32)
    return X_person, order_ref


def main():
    print("[load] scan-level embeddings and v4 OOF")
    paths, y, persons, scan_embs = load_scan_embeddings()
    v4 = load_v4_oof()
    # Align v4 to our path order (sanity — should already match)
    path_to_idx = {str(p): i for i, p in enumerate(paths)}
    v4_order = np.array([path_to_idx[str(p)] for p in v4["scan_paths"]])
    v4_proba = np.zeros_like(v4["proba"])
    v4_proba[v4_order] = v4["proba"]
    v4_preds = v4_proba.argmax(axis=1)

    v4_wf1 = f1_score(y, v4_preds, average="weighted")
    v4_mf1 = f1_score(y, v4_preds, average="macro")
    print(f"  v4 scan-level wF1={v4_wf1:.4f}  macroF1={v4_mf1:.4f}")

    # Scan-level confidence (top-1 prob) used for attention pooling
    scan_top1 = v4_proba.max(axis=1)

    # Person-level ground truth (using majority label per person)
    order_tmp = np.array(sorted(np.unique(persons).tolist()))
    y_person = person_labels(persons, y, order_tmp)
    print(f"  n_persons={len(order_tmp)}  class counts: "
          f"{dict(Counter(y_person.tolist()))}")

    variants = {}
    for variant in ("mean", "max", "attn"):
        print(f"\n[variant] {variant}")
        weights = scan_top1 if variant == "attn" else None
        X_person, order = build_concat_person(scan_embs, persons, variant, weights)
        print(f"  X_person shape: {X_person.shape}")

        # Align y_person to this order (should equal order_tmp)
        yp = person_labels(persons, y, order)

        preds_p, proba_p = lopo_person_predict(X_person, yp)
        person_f1w = f1_score(yp, preds_p, average="weighted")
        person_f1m = f1_score(yp, preds_p, average="macro")
        print(f"  person-level LOPO wF1={person_f1w:.4f}  macroF1={person_f1m:.4f}")

        # Broadcast to scan-level
        scan_preds, scan_proba = broadcast_to_scan(preds_p, proba_p, order, persons)
        scan_f1w = f1_score(y, scan_preds, average="weighted")
        scan_f1m = f1_score(y, scan_preds, average="macro")
        per_class = f1_score(y, scan_preds, average=None, labels=list(range(len(CLASSES))))
        print(f"  scan-level (broadcast) wF1={scan_f1w:.4f}  macroF1={scan_f1m:.4f}")
        for c, f in zip(CLASSES, per_class):
            print(f"    {c:20s} F1={f:.4f}")

        # Bootstrap vs v4
        bs = paired_bootstrap(y, scan_preds, v4_preds, n_boot=1000, seed=42)
        print(f"  bootstrap vs v4: mean Δ={bs['mean_delta']:+.4f}  "
              f"P(Δ>0)={bs['p_gt_zero']:.3f}  CI95=[{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]")

        variants[variant] = {
            "person_wf1": float(person_f1w),
            "person_macrof1": float(person_f1m),
            "scan_wf1": float(scan_f1w),
            "scan_macrof1": float(scan_f1m),
            "per_class_f1": {c: float(f) for c, f in zip(CLASSES, per_class)},
            "bootstrap_vs_v4": bs,
            "person_preds": preds_p.tolist(),
            "person_proba": proba_p.tolist(),
            "scan_preds": scan_preds.tolist(),
        }

    # -----------------------------------------------------------------------
    # Persist artefacts
    # -----------------------------------------------------------------------
    out_json = {
        "v4_scan_wf1": float(v4_wf1),
        "v4_scan_macrof1": float(v4_mf1),
        "n_persons": int(len(order_tmp)),
        "person_order": order_tmp.tolist(),
        "person_labels": y_person.tolist(),
        "scan_paths": [str(p) for p in paths],
        "scan_y": y.tolist(),
        "variants": variants,
    }
    out_path = CACHE / "patient_classifier_predictions.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[saved] {out_path}")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    best_variant = max(variants, key=lambda v: variants[v]["scan_wf1"])
    best = variants[best_variant]
    verdict = (
        "CHAMPION CANDIDATE (scan-level wF1 exceeds v4)"
        if best["scan_wf1"] > v4_wf1
        else "Does not beat v4 at scan level — keep as pitch angle for per-patient evaluation regime"
    )

    md = [
        "# Patient-Level Classifier",
        "",
        "## Motivation",
        "",
        "Train a classifier where **each person is one sample** (35 samples, 34-dim-LOPO-friendly).",
        "All scans of a person inherit its predicted label at inference. This directly matches",
        "an evaluation regime in which the organizers grade per-patient. If they grade per-scan",
        "with potentially diverse labels per patient (not our case in TRAIN_SET), this approach",
        "would hurt — measured and reported here.",
        "",
        "## Setup",
        "",
        "- Encoders: DINOv2-B @ 90 nm/px + DINOv2-B @ 45 nm/px + BiomedCLIP D4 TTA.",
        "- Per scan: L2-normalize 768/768/512-D embedding.",
        "- Aggregate to one vector per person with three variants (mean / max / attention).",
        "- Attention weights = v4 scan-level top-1 probability (softmax-normalized within person).",
        "- Concatenate the three encoders per person → 2048-D.",
        "- LOPO Logistic Regression (class_weight balanced, C=1, L2-norm + StandardScaler).",
        "- 35 train/predict rounds, then broadcast to 240 scans for fair comparison with v4.",
        "",
        f"Reference: v4 scan-level wF1 = **{v4_wf1:.4f}**, macroF1 = {v4_mf1:.4f}.",
        "",
        "## Results",
        "",
        "| Variant | Person wF1 | Person macroF1 | Scan wF1 (broadcast) | Scan macroF1 | Δ vs v4 | P(Δ>0) |",
        "|---------|-----------:|---------------:|---------------------:|-------------:|--------:|-------:|",
    ]
    for v in ("mean", "max", "attn"):
        r = variants[v]
        d = r["bootstrap_vs_v4"]
        md.append(
            f"| {v:4s} | {r['person_wf1']:.4f} | {r['person_macrof1']:.4f} | "
            f"{r['scan_wf1']:.4f} | {r['scan_macrof1']:.4f} | "
            f"{d['mean_delta']:+.4f} | {d['p_gt_zero']:.3f} |"
        )

    md += [
        "",
        "## Per-class F1 (best variant: `" + best_variant + "`)",
        "",
        "| Class | F1 |",
        "|-------|---:|",
    ]
    for c, f in best["per_class_f1"].items():
        md.append(f"| {c} | {f:.4f} |")

    bs = best["bootstrap_vs_v4"]
    md += [
        "",
        "## Bootstrap vs v4 (1000 resamples, best variant)",
        "",
        f"- mean Δ = **{bs['mean_delta']:+.4f}**",
        f"- P(Δ > 0) = **{bs['p_gt_zero']:.3f}**",
        f"- 95% CI = [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Caveats",
        "",
        "- Broadcasting the person-level prediction to every scan is fair **only if the",
        "  organizers evaluate per-patient**. In our TRAIN_SET every scan of a given person",
        "  has the same class anyway, so this is not a lossy assumption on our data, but the",
        "  hidden test set may differ.",
        "- Person-level F1 is on 35 samples, which gives a coarse signal (one misclassified",
        "  person moves person-wF1 by ~3 points). The scan-broadcast metric above is the",
        "  figure to compare against v4.",
    ]
    md_path = REPORTS / "PATIENT_LEVEL_CLASSIFIER.md"
    md_path.write_text("\n".join(md))
    print(f"[saved] {md_path}")
    print(f"\n[verdict] best variant = {best_variant} ({verdict})")


if __name__ == "__main__":
    main()
