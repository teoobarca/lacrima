"""Honest evaluation of the BMP fallback path on TRAIN_SET.

We have BMP files for 238/240 scans and known ground truth.  We run the v4
multi-scale ensemble through the BMP preprocessing pipeline instead of the
raw-SPM pipeline and compute person-LOPO F1 honestly (retrain LR heads per
fold on top of the BMP embeddings — this is the fair comparison to the raw
SPM 0.6887 baseline).

Approach (to stay within the 25-min budget):

    1. Encode all 238 BMPs through BOTH DINOv2-B and BiomedCLIP *once*
       (9 tiles x 238 = 2142 tiles per encoder; fast on MPS).
    2. Mean-pool tiles to scan-level embeddings (per encoder).
    3. Person-LOPO with the v2 recipe:
            L2-normalize -> StandardScaler -> LR(balanced)
       Each component's softmax is computed out-of-fold; geom-mean gives the
       ensemble prediction.
    4. Compare per-class F1 and weighted F1 to the 0.6887 raw-SPM reference.

Note: the BMP path cannot naturally produce two different pixel sizes (90
vs 45 nm/px) because the PNG is already rasterized, so we re-use the same
DINOv2-B embedding for the "90nm" and "45nm" slots of the ensemble — giving
two independently-trained linear heads over the SAME features.  In practice
that is a weak ensemble (the two heads mostly agree), so we report both:
    * "BMP 2-head" (DINOv2 90+45 heads + BiomedCLIP head)
    * "BMP 1-DINOv2" (DINOv2 90nm head + BiomedCLIP head only)
Either result is a fair BMP-path number.

Expected honest F1: ~0.50-0.60 (watermark removed by crop, but 704x575 RGB
losses a lot vs. the 1024x1024+ raw height map).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.bmp_infer import preprocess_bmp  # noqa: E402
from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import CLASSES, enumerate_samples, person_id  # noqa: E402
from teardrop.encoders import load_biomedclip, load_dinov2  # noqa: E402

DATA_ROOT = ROOT / "TRAIN_SET"
CACHE_DIR = ROOT / "cache"
REPORTS = ROOT / "reports"
N_CLASSES = len(CLASSES)
EPS = 1e-12


def encode_all_bmps(samples, encoder, cache_path: Path) -> np.ndarray:
    """Encode all BMPs (9 tiles each) and return mean-pooled (N, D) array."""
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if cached["n"] == len(samples):
            print(f"  [cache] {cache_path.name} hit — n={len(samples)}")
            return cached["X_scan"].astype(np.float32)

    print(f"  [encode] n={len(samples)} via {encoder.name}")
    t0 = time.time()
    scan_embs = []
    for i, s in enumerate(samples):
        tiles = preprocess_bmp(s.bmp_path)
        emb = encoder.encode(tiles, batch_size=len(tiles))
        scan_embs.append(emb.mean(axis=0))
        if (i + 1) % 25 == 0:
            print(f"    [{i + 1}/{len(samples)}] t={time.time() - t0:.0f}s")
    X_scan = np.stack(scan_embs).astype(np.float32)
    np.savez(cache_path, X_scan=X_scan, n=len(samples),
             encoder_name=encoder.name)
    print(f"  [saved] {cache_path.name} shape={X_scan.shape}")
    return X_scan


def lopo_softmax(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    P = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups):
        Xt = normalize(X[tr], norm="l2", axis=1)
        Xv = normalize(X[va], norm="l2", axis=1)
        sc = StandardScaler().fit(Xt)
        Xt = np.nan_to_num(sc.transform(Xt), nan=0.0)
        Xv = np.nan_to_num(sc.transform(Xv), nan=0.0)
        clf = LogisticRegression(class_weight="balanced", max_iter=3000,
                                 C=1.0, solver="lbfgs", n_jobs=4,
                                 random_state=42)
        clf.fit(Xt, y[tr])
        proba = clf.predict_proba(Xv)
        p_full = np.zeros((len(va), N_CLASSES), dtype=np.float64)
        for ci, cls in enumerate(clf.classes_):
            p_full[:, cls] = proba[:, ci]
        P[va] = p_full
    return P


def geom_mean(probs_list: list[np.ndarray]) -> np.ndarray:
    log_sum = np.zeros_like(probs_list[0])
    for P in probs_list:
        log_sum += np.log(P + EPS)
    G = np.exp(log_sum / len(probs_list))
    G /= G.sum(axis=1, keepdims=True)
    return G


def _f1_report(P: np.ndarray, y: np.ndarray, name: str):
    pred = P.argmax(axis=1)
    fw = f1_score(y, pred, average="weighted", zero_division=0)
    fm = f1_score(y, pred, average="macro", zero_division=0)
    per = f1_score(y, pred, average=None, labels=list(range(N_CLASSES)),
                   zero_division=0)
    print(f"  {name:30s} weighted_f1={fw:.4f}  macro_f1={fm:.4f}")
    for ci, cls in enumerate(CLASSES):
        print(f"    {cls:20s} f1={per[ci]:.3f}")
    return fw, fm, per


def main():
    print("=" * 78)
    print("BMP fallback eval (v4 recipe, person-LOPO, honest)")
    print("=" * 78)

    # Gather samples that have BMPs
    print("\n[1] enumerate samples with BMP")
    all_samples = enumerate_samples(DATA_ROOT)
    samples = [s for s in all_samples if s.bmp_path is not None]
    print(f"  total={len(all_samples)}  with BMP={len(samples)}")
    y = np.array([s.label for s in samples], dtype=np.int64)
    groups = np.array([s.person for s in samples])
    n_persons = len(np.unique(groups))
    print(f"  n_persons={n_persons}")

    # Encode via DINOv2-B + BiomedCLIP
    print("\n[2] encode BMPs through both encoders (cached)")
    CACHE_DIR.mkdir(exist_ok=True)
    dinov2 = load_dinov2("vitb14")
    X_dinov2 = encode_all_bmps(samples, dinov2,
                               CACHE_DIR / "bmp_scan_emb_dinov2_vitb14.npz")
    del dinov2

    biomed = load_biomedclip()
    X_biomed = encode_all_bmps(samples, biomed,
                               CACHE_DIR / "bmp_scan_emb_biomedclip.npz")
    del biomed

    # LOPO per component
    print("\n[3] person-LOPO v2-recipe softmax per component")
    t0 = time.time()
    P_d = lopo_softmax(X_dinov2, y, groups)
    P_b = lopo_softmax(X_biomed, y, groups)
    print(f"  done in {time.time() - t0:.1f}s")

    # Two evaluations: "3-head" mimicking v4 (D90+D45+Bc) but with same D emb,
    # and "2-head" (D + Bc).
    print("\n[4] results")
    _f1_report(P_d, y, "DINOv2-B (BMP) only")
    _f1_report(P_b, y, "BiomedCLIP (BMP) only")
    P_geo2 = geom_mean([P_d, P_b])
    _f1_report(P_geo2, y, "geom-mean(D + Bc)")
    # v4-style 3-head: same DINOv2 emb gets 2 LR heads (90nm slot + 45nm slot),
    # both trained the same way -> heads are identical, so this equals 2-head.
    # Report it explicitly anyway for completeness.
    P_geo3 = geom_mean([P_d, P_d, P_b])
    _f1_report(P_geo3, y, "geom-mean(D + D + Bc)  [v4-style]")

    # save arrays for further inspection
    out = CACHE_DIR / "bmp_fallback_oof.npz"
    np.savez(out, P_dinov2=P_d, P_biomed=P_b,
             P_geom2=P_geo2, P_geom3=P_geo3,
             y=y, groups=groups,
             scan_paths=np.array([str(s.raw_path) for s in samples]))
    print(f"\n[saved] {out}")

    # Write a small report
    REPORTS.mkdir(exist_ok=True)
    out_md = REPORTS / "BMP_FALLBACK_EVAL.md"
    pred2 = P_geo2.argmax(axis=1)
    pred3 = P_geo3.argmax(axis=1)
    with open(out_md, "w") as f:
        f.write("# BMP fallback evaluation — person-LOPO F1\n\n")
        f.write(f"n_samples = {len(samples)} (of 240 BMP-linked)\n")
        f.write(f"n_persons = {n_persons}\n\n")
        for name, P in [("DINOv2-B only", P_d),
                        ("BiomedCLIP only", P_b),
                        ("geom-mean(D + Bc)", P_geo2),
                        ("geom-mean(D + D + Bc) [v4-style]", P_geo3)]:
            pred = P.argmax(axis=1)
            fw = f1_score(y, pred, average="weighted", zero_division=0)
            fm = f1_score(y, pred, average="macro", zero_division=0)
            f.write(f"## {name}\n")
            f.write(f"- weighted_f1 = {fw:.4f}\n")
            f.write(f"- macro_f1    = {fm:.4f}\n\n")
        f.write("## Per-class F1 (best ensemble: geom-mean(D + Bc))\n\n")
        per = f1_score(y, pred2, average=None,
                       labels=list(range(N_CLASSES)), zero_division=0)
        for ci, cls in enumerate(CLASSES):
            f.write(f"- {cls}: {per[ci]:.3f}\n")
        f.write("\n## Ref\n")
        f.write("Raw-SPM v4 champion: weighted_f1=0.6887, macro_f1=0.5541\n")
    print(f"[report] {out_md}")


if __name__ == "__main__":
    main()
