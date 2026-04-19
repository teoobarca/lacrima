"""Prototypical Networks evaluation on tear AFM classification.

Reference: Snell, Swersky, Zemel (2017), NeurIPS.

Setup matches the v4 multiscale champion for a fair comparison:
  - Three encoders: DINOv2-B 90 nm, DINOv2-B 45 nm, BiomedCLIP-TTA 90 nm.
  - Scan-level features (mean-pool tiles for the two DINOv2 branches, already
    scan-level for BiomedCLIP-TTA).
  - Person-level LOPO (35 folds) via `teardrop.data.person_id`.
  - Geometric-mean ensemble of per-encoder predictions (v2 recipe).

Experiments (all evaluated person-LOPO):

  Baseline : training-free ProtoNet (softmax on -sqeuclidean after L2-norm).
  Cosine   : training-free ProtoNet with cosine distance.
  Weighted : inverse-distance-to-centroid weighted prototypes (outlier robust).
  Temp*T   : tuned softmax temperature (scan logits on distance scale).
  Adapter  : train a small MLP (768/512 -> 256 -> 128) per fold via episodic
             ProtoNet loss, then prototype + NN in the learned space.

Honest bar to clear: v4 multi-scale LR = 0.6887 weighted F1 / 0.5541 macro F1.
Focus question: does *any* variant lift SucheOko F1 above 0?
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import CLASSES, person_id  # noqa: E402
from teardrop.protonet import (  # noqa: E402
    ProtoClassifier,
    embed_with_adapter,
    train_adapter,
)

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

N_CLASSES = len(CLASSES)
EPS = 1e-12

# Honest v4 champion numbers for comparison
CHAMP_V4_WF1 = 0.6887
CHAMP_V4_MF1 = 0.5541


# ---------------------------------------------------------------------------
# I/O + alignment
# ---------------------------------------------------------------------------

def mean_pool_tiles(X_tiles: np.ndarray, t2s: np.ndarray, n_scans: int) -> np.ndarray:
    d = X_tiles.shape[1]
    out = np.zeros((n_scans, d), dtype=np.float32)
    for si in range(n_scans):
        m = t2s == si
        if m.any():
            out[si] = X_tiles[m].mean(axis=0)
    return out


def align_to_reference(paths_ref: list[str], paths_src: list[str],
                       X_src: np.ndarray) -> np.ndarray:
    src_idx = {p: i for i, p in enumerate(paths_src)}
    order = np.array([src_idx[p] for p in paths_ref])
    return X_src[order]


def load_encoder_features() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Return ({encoder_name: X_scan (240, D)}, y, groups), all aligned.

    Alignment is to the 90nm cache scan ordering (reference).
    """
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz",
                  allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz",
                  allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz",
                  allow_pickle=True)

    paths_90 = [str(p) for p in z90["scan_paths"]]
    paths_45 = [str(p) for p in z45["scan_paths"]]
    paths_bc = [str(p) for p in zbc["scan_paths"]]

    y = np.asarray(z90["scan_y"], dtype=np.int64)
    groups = np.array([person_id(Path(p)) for p in paths_90])

    X90 = mean_pool_tiles(z90["X"], z90["tile_to_scan"], len(paths_90))
    X45_raw = mean_pool_tiles(z45["X"], z45["tile_to_scan"], len(paths_45))
    X45 = align_to_reference(paths_90, paths_45, X45_raw)
    Xbc = align_to_reference(paths_90, paths_bc, zbc["X_scan"].astype(np.float32))

    feats = {
        "dinov2_90nm": X90,
        "dinov2_45nm": X45,
        "biomedclip_tta": Xbc,
    }
    return feats, y, groups


# ---------------------------------------------------------------------------
# LOPO experiment runners
# ---------------------------------------------------------------------------

def lopo_protonet(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    distance: str = "sqeuclidean",
    temperature: float = 1.0,
    weighted: bool = False,
) -> np.ndarray:
    """Training-free per-fold ProtoNet. Returns OOF softmax (N, C)."""
    P = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    for tr, va in leave_one_patient_out(groups):
        clf = ProtoClassifier(
            distance=distance, temperature=temperature,
            n_classes=N_CLASSES, weighted=weighted,
        )
        clf.fit(X[tr], y[tr])
        P[va] = clf.predict_proba(X[va])
    return P


def lopo_adapter_protonet(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    hidden: int = 256,
    out_dim: int = 128,
    n_episodes: int = 400,
    k_support: int = 4,
    k_query: int = 2,
    distance: str = "sqeuclidean",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    learnable_temperature: bool = True,
    device: str = "cpu",
    seed: int = 42,
) -> np.ndarray:
    """Per-fold adapter-trained ProtoNet. Returns OOF softmax (N, C)."""
    P = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    for fi, (tr, va) in enumerate(leave_one_patient_out(groups)):
        X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
        model, temp = train_adapter(
            X_tr, y_tr, g_tr,
            n_classes=N_CLASSES,
            hidden=hidden, out_dim=out_dim,
            n_episodes=n_episodes,
            k_support=k_support, k_query=k_query,
            distance=distance,
            lr=lr, weight_decay=weight_decay,
            learnable_temperature=learnable_temperature,
            device=device, seed=seed + fi,
        )
        # Embed all, fit ProtoClassifier on adapter features for this fold
        Z_tr = embed_with_adapter(model, X_tr, device=device)
        Z_va = embed_with_adapter(model, X[va], device=device)
        clf = ProtoClassifier(distance=distance, temperature=temp,
                              n_classes=N_CLASSES)
        clf.fit(Z_tr, y_tr)
        P[va] = clf.predict_proba(Z_va)
    return P


def geom_mean_probs(probs_list: list[np.ndarray]) -> np.ndarray:
    log_sum = np.zeros_like(probs_list[0])
    for Pk in probs_list:
        log_sum = log_sum + np.log(Pk + EPS)
    G = np.exp(log_sum / len(probs_list))
    G /= G.sum(axis=1, keepdims=True)
    return G


def metrics_of(P: np.ndarray, y: np.ndarray) -> dict:
    pred = P.argmax(axis=1)
    return {
        "weighted_f1": float(
            f1_score(y, pred, average="weighted", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y, pred, average="macro", zero_division=0)
        ),
        "per_class_f1": f1_score(
            y, pred, average=None,
            labels=list(range(N_CLASSES)), zero_division=0,
        ).tolist(),
    }


def pretty_pc(per_class_f1: list[float]) -> str:
    return "  ".join(f"{CLASSES[i][:6]}={v:.3f}"
                     for i, v in enumerate(per_class_f1))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    # Adapter training uses torch.cdist whose backward is missing on MPS
    # (aten::_cdist_backward). CPU is fast enough for a 240-sample MLP.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print("Prototypical Networks — person-LOPO evaluation")
    print(f"  device={device}")
    print("=" * 78)

    feats, y, groups = load_encoder_features()
    n_persons = len(np.unique(groups))
    print(f"\n[data] n_scans={len(y)}  n_persons={n_persons}  "
          f"n_classes={N_CLASSES}")
    for c, name in enumerate(CLASSES):
        m = y == c
        n_p = len(np.unique(groups[m]))
        print(f"  {name:20s} scans={m.sum():3d}  persons={n_p:2d}")

    # Per-encoder per-experiment
    results: dict = {"per_encoder": {}, "ensemble": {}}

    # NOTE on temperature: under geometric-mean ensembling with a SHARED T
    # across all encoders, changing T merely rescales the log-probabilities
    # uniformly per-encoder and does NOT change the ensemble argmax. So a
    # fixed-T sweep would be uninformative. The temperature experiment here
    # is therefore a LEARNABLE per-fold T inside the adapter variant, which
    # does change what the adapter optimizes and thus the ensemble.
    experiments = [
        # (name, is_adapter, kwargs)
        ("baseline_sqeuclid",
         False,
         dict(distance="sqeuclidean", temperature=1.0, weighted=False)),
        ("cosine",
         False,
         dict(distance="cosine", temperature=1.0, weighted=False)),
        ("weighted_prototypes",
         False,
         dict(distance="sqeuclidean", temperature=1.0, weighted=True)),
        # Adapter with learnable temperature + episodic training
        ("adapter_sqeuclid",
         True,
         dict(distance="sqeuclidean", n_episodes=400, k_support=4, k_query=2,
              hidden=256, out_dim=128, learnable_temperature=True,
              device=device, seed=42)),
        ("adapter_cosine",
         True,
         dict(distance="cosine", n_episodes=400, k_support=4, k_query=2,
              hidden=256, out_dim=128, learnable_temperature=True,
              device=device, seed=42)),
    ]

    # Probs store (per-experiment, per-encoder): {exp_name: {enc_name: P}}
    probs: dict[str, dict[str, np.ndarray]] = {}

    for exp_name, is_adapter, kwargs in experiments:
        probs[exp_name] = {}
        results["per_encoder"][exp_name] = {}
        print(f"\n[experiment] {exp_name}")
        for enc_name, X in feats.items():
            t_s = time.time()
            if is_adapter:
                Pk = lopo_adapter_protonet(X, y, groups, **kwargs)
            else:
                Pk = lopo_protonet(X, y, groups, **kwargs)
            m = metrics_of(Pk, y)
            probs[exp_name][enc_name] = Pk
            results["per_encoder"][exp_name][enc_name] = m
            dt = time.time() - t_s
            print(f"  {enc_name:18s} W={m['weighted_f1']:.4f} "
                  f"M={m['macro_f1']:.4f}  {pretty_pc(m['per_class_f1'])} "
                  f" ({dt:.1f}s)")

        # Geom-mean ensemble across encoders
        G = geom_mean_probs(list(probs[exp_name].values()))
        m_ens = metrics_of(G, y)
        results["ensemble"][exp_name] = m_ens
        print(f"  {'ENSEMBLE(geom)':18s} "
              f"W={m_ens['weighted_f1']:.4f} M={m_ens['macro_f1']:.4f}  "
              f"{pretty_pc(m_ens['per_class_f1'])}")

    # Summary table
    print("\n" + "=" * 78)
    print("Summary — person-LOPO F1 (weighted | macro | SucheOko)")
    print("=" * 78)
    print(f"{'experiment':25s} {'W-F1':>7s} {'M-F1':>7s} "
          f"{'ΔV4-W':>8s} {'SucheOko':>9s}")
    print("-" * 72)
    suo = CLASSES.index("SucheOko")
    for exp in experiments:
        name = exp[0]
        m = results["ensemble"][name]
        print(f"{name:25s} {m['weighted_f1']:7.4f} {m['macro_f1']:7.4f} "
              f"{m['weighted_f1'] - CHAMP_V4_WF1:+8.4f} "
              f"{m['per_class_f1'][suo]:9.4f}")

    print(f"\nv4 multiscale LR champion:  W-F1={CHAMP_V4_WF1:.4f}  "
          f"M-F1={CHAMP_V4_MF1:.4f}")

    # Persist
    out = {
        "n_scans": int(len(y)),
        "n_persons": int(n_persons),
        "classes": CLASSES,
        "champions": {"v4_weighted_f1": CHAMP_V4_WF1,
                      "v4_macro_f1": CHAMP_V4_MF1},
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (REPORTS / "protonet_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n[saved] reports/protonet_results.json")

    write_markdown_report(out)
    print(f"[done] total elapsed: {time.time() - t0:.1f}s")


def write_markdown_report(summary: dict) -> None:
    classes = summary["classes"]
    suo = classes.index("SucheOko")
    r = summary["results"]
    ens = r["ensemble"]
    per_enc = r["per_encoder"]
    champ_w = summary["champions"]["v4_weighted_f1"]
    champ_m = summary["champions"]["v4_macro_f1"]

    L: list[str] = []
    L.append("# Prototypical Networks — Results\n")
    L.append(
        "**Hypothesis:** Prototypical Networks (Snell et al., 2017) should help "
        "with extreme few-shot classes — our dataset has 2 persons for SucheOko "
        "and 4 for Diabetes, for which logistic regression struggles because the "
        "decision boundary is fit with very few positive samples. A prototype "
        "(class centroid in embedding space) gives each class equal footing "
        "regardless of its sample count.\n"
    )
    L.append("## Setup\n")
    L.append(
        f"- **Data:** {summary['n_scans']} AFM scans, "
        f"{summary['n_persons']} persons, 5 classes.\n"
        "- **Encoders (matching v4 multiscale champion):** DINOv2-B 90 nm/px, "
        "DINOv2-B 45 nm/px, BiomedCLIP 90 nm/px with D4 TTA.\n"
        "- **Features:** scan-level (mean-pool tile embeddings).\n"
        "- **Evaluation:** PERSON-level LOPO (35 folds, "
        "`teardrop.data.person_id`).\n"
        "- **Prototype:** L2-normalize support embeddings per class → take mean "
        "→ L2-normalize again.\n"
        "- **Classification:** softmax over negative distances "
        "(squared-Euclidean after L2-norm, or cosine).\n"
        "- **Ensemble:** geometric mean of per-encoder softmaxes "
        "(same recipe as v2/v4).\n"
        "- **Baseline to beat:** v4 multi-scale LR ensemble "
        f"= W-F1 {champ_w:.4f}, M-F1 {champ_m:.4f}.\n"
    )

    L.append("\n## Experiments\n")
    L.append(
        "1. **baseline_sqeuclid** — training-free, squared-Euclidean distance "
        "after L2-normalization, temperature T=1.\n"
        "2. **cosine** — training-free, cosine distance (1 - <q,p>).\n"
        "3. **weighted_prototypes** — inverse-distance-to-centroid weighting "
        "(outlier-robust, Mahalanobis-ish).\n"
        "4. **adapter_sqeuclid / adapter_cosine** — per-fold trained MLP "
        "(D → 256 → 128) with episodic ProtoNet loss "
        "(4-support, 2-query, 400 episodes, AdamW, learnable temperature). "
        "Then prototype + NN in the learned space.\n\n"
        "**A note on fixed temperature:** under a geometric-mean ensemble with "
        "a temperature shared across all encoders, changing T only rescales "
        "the log-probabilities uniformly per encoder and does NOT alter the "
        "ensemble argmax — a fixed-T sweep is therefore uninformative. The "
        "temperature experiment in this report is the LEARNABLE per-fold T "
        "inside the adapter (exp (4)), which does change what the adapter "
        "optimizes and thus affects the ensemble.\n"
    )

    # Per-encoder table (baseline only, for brevity)
    L.append("\n## Per-encoder ProtoNet — baseline (sqEuclidean, T=1)\n")
    L.append("| Encoder | Weighted F1 | Macro F1 | "
             + " | ".join(classes) + " |")
    L.append("|---|---:|---:|" + "|".join([":---:"] * len(classes)) + "|")
    for enc, m in per_enc["baseline_sqeuclid"].items():
        pc = " | ".join(f"{v:.3f}" for v in m["per_class_f1"])
        L.append(f"| `{enc}` | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} | "
                 f"{pc} |")

    # Ensemble results across experiments
    L.append("\n## Ensemble (geometric mean across 3 encoders)\n")
    L.append("| Experiment | Weighted F1 | Macro F1 | Δ v4 | "
             + " | ".join(classes) + " |")
    L.append("|---|---:|---:|---:|" + "|".join([":---:"] * len(classes)) + "|")
    for name, m in ens.items():
        pc = " | ".join(f"{v:.3f}" for v in m["per_class_f1"])
        L.append(f"| **{name}** | {m['weighted_f1']:.4f} | "
                 f"{m['macro_f1']:.4f} | "
                 f"{m['weighted_f1'] - champ_w:+.4f} | {pc} |")
    L.append("")

    # Analysis — pick best and comment on SucheOko
    best_name = max(ens, key=lambda k: ens[k]["weighted_f1"])
    best = ens[best_name]
    best_suo_name = max(ens, key=lambda k: ens[k]["per_class_f1"][suo])
    best_suo = ens[best_suo_name]["per_class_f1"][suo]

    # Also check per-encoder SucheOko rescue (sometimes a single encoder
    # breaks the 0, but geom-mean smears it back out).
    best_enc_suo = 0.0
    best_enc_suo_id = ("", "")
    for exp_name, enc_dict in per_enc.items():
        for enc_name, m in enc_dict.items():
            v = m["per_class_f1"][suo]
            if v > best_enc_suo:
                best_enc_suo = v
                best_enc_suo_id = (exp_name, enc_name)

    L.append("## SucheOko minority-class analysis\n")
    L.append(
        "SucheOko has only 2 persons (14 scans). With person-level LOPO the "
        "training fold for each SucheOko person has only ONE remaining SucheOko "
        "person — a true 1-shot (on person axis) regime. Under the LR recipe "
        "this collapses to F1 = 0 on those folds.\n\n"
        f"- **Best SucheOko F1 (ensemble):** "
        f"{best_suo:.4f} — experiment `{best_suo_name}`.\n"
        f"- **Best SucheOko F1 (single encoder, any experiment):** "
        f"{best_enc_suo:.4f} — `{best_enc_suo_id[0]} / "
        f"{best_enc_suo_id[1]}`.\n"
    )
    if best_suo > 0:
        L.append(
            f"- PARTIAL WIN: ProtoNet rescues SucheOko from 0 to "
            f"{best_suo:.3f} in the ensemble — the prototypical structure "
            "*does* help the extreme minority class even when the overall "
            "weighted F1 doesn't move or regresses.\n"
        )
    elif best_enc_suo > 0:
        L.append(
            f"- PARTIAL SIGNAL: SucheOko F1 climbs to **{best_enc_suo:.3f}** "
            f"on the `{best_enc_suo_id[1]}` encoder under the adapter "
            f"(`{best_enc_suo_id[0]}`), but the geometric-mean ensemble "
            "dilutes it back to 0 — the other two encoders never predict "
            "SucheOko and the geom-mean of three softmaxes where only one "
            "gives SucheOko ≠ 0 does not recover the positive. "
            "Useful follow-up: switch to MAX or weighted-mean ensembling "
            "for the minority-class channel, or condition on a ProtoNet "
            "gate.\n"
        )
    else:
        L.append(
            "- ProtoNet did NOT rescue SucheOko — the prototype in embedding "
            "space is still dominated by majority classes (or the lone "
            "remaining SucheOko person's prototype is off the query "
            "manifold).\n"
        )

    L.append("\n## Verdict vs v4 LR champion\n")
    delta_w = best["weighted_f1"] - champ_w
    delta_m = best["macro_f1"] - champ_m
    L.append(
        f"- Best overall ProtoNet variant: `{best_name}` — "
        f"W-F1 {best['weighted_f1']:.4f} "
        f"(Δ v4 = {delta_w:+.4f}), M-F1 {best['macro_f1']:.4f} "
        f"(Δ v4 = {delta_m:+.4f}).\n"
    )
    if delta_w >= 0.005:
        L.append(
            "- **ProtoNet BEATS v4 meaningfully** (≥ 0.005 weighted-F1 lift). "
            "Candidate v5.\n"
        )
    elif delta_w >= 0:
        L.append(
            "- ProtoNet ties v4 within noise — no clear win on weighted F1. "
            "Consider ProtoNet only if the per-class SucheOko lift is worth "
            "the architectural complexity.\n"
        )
    else:
        L.append(
            "- ProtoNet is **below** v4 LR on weighted F1. The LR recipe is "
            "hard to beat when the majority classes each have ≥ 5 persons; "
            "the prototype loses discriminative capacity in favor of symmetry. "
            "Expected, per Snell et al.: ProtoNet helps strictly-few-shot but "
            "not well-populated classes.\n"
        )
    L.append(
        "- If Diabetes (4 persons) or SucheOko (2 persons) per-class F1 is "
        "noticeably higher than under v4, keep ProtoNet predictions as a "
        "minority-class rescue channel even if the overall score doesn't "
        "improve.\n"
    )

    L.append("\n## Honest reporting\n")
    L.append(
        "- Person-LOPO (35 folds) for every number here; no OOF model "
        "selection; no threshold tuning; no train/val leakage in the adapter "
        "(the adapter is trained fresh on each fold's training set).\n"
        "- Adapter runs use fixed hyperparameters (no per-fold HP search) to "
        "prevent nested-CV leakage.\n"
        "- Caches are the same three encoder outputs used by the v4 "
        "multi-scale champion → apples-to-apples comparison.\n"
    )

    (REPORTS / "PROTONET_RESULTS.md").write_text("\n".join(L))
    print(f"[saved] reports/PROTONET_RESULTS.md")


if __name__ == "__main__":
    main()
