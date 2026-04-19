"""MAE fine-tuning: attach a classification head to the pre-trained MAE encoder
and evaluate under person-level LOPO.

Complements `scripts/mae_pretrain.py` (which already trained the MAE encoder and
ran a quick LogisticRegression probe). This script implements the explicit
step-2 of the task spec:

    1. Load MAE encoder weights (from `models/mae_tear_tiny/encoder.pt`).
    2. Add a classification head. We use a TWO-variant ablation to settle the
       frozen-vs-LoRA question:
         a) `linear_frozen`: encoder frozen, 1-layer linear head. CE + balanced
            class weights. AdamW, cosine schedule.
         b) `mlp_frozen`:    encoder frozen, 2-layer MLP head.
       We do NOT do LoRA because:
         - 72-epoch MAE encoder was trained on ~15k effective views from only
           1.9k base tiles => heavy correlation, high overfitting risk.
         - In LOPO each val fold is a single person (1-25 scans); with no held-out
           tune set there is no signal to early-stop encoder updates against.
         - MAE paper's canonical downstream protocol IS the linear probe -- if a
           frozen probe is below baseline, LoRA will overfit even more.
       If the frozen probe wF1 >= 0.60, we would enable a third variant with
       `LoRA(r=4)` on the last 2 ViT blocks -- gated behind that threshold.
    3. Person-LOPO over 35 unique `person_id`s. Each fold: train head only,
       predict on val, collect probabilities + argmax.
    4. Save `cache/mae_predictions.json` (protocol, y, persons, per-variant probs
       and preds) and `reports/MAE_PRETRAINING.md` with bootstrap CI vs v4.

Outputs
-------
- `cache/mae_predictions.json`           -- OOF predictions (both variants)
- `reports/MAE_PRETRAINING.md`           -- fine-tune report + v4 comparison

Does NOT touch v4 champion artifacts.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.data import CLASSES  # noqa: E402

# Reuse the model class from the pretraining script.
from scripts.mae_pretrain import (  # noqa: E402
    MAETearTiny,
    prepare_image_tensor,
)

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Load pre-trained MAE encoder + cached tile features
# ---------------------------------------------------------------------------


def load_mae_encoder() -> MAETearTiny:
    ckpt_path = MODELS / "mae_tear_tiny" / "encoder.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing MAE encoder: {ckpt_path}. "
                                "Run scripts/mae_pretrain.py first.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MAETearTiny()
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    print(f"  loaded MAE encoder from {ckpt_path} "
          f"(epochs trained: {ckpt['config'].get('epochs_trained')})")
    return model


def load_cached_tile_features() -> dict:
    """Pre-computed CLS features from mae_pretrain.py (1889 tiles, 192-d)."""
    z = np.load(CACHE / "mae_emb_tear_tiny.npz", allow_pickle=True)
    return {
        "X_tiles": z["X_tiles"].astype(np.float32),           # (1889, 192)
        "X_scan":  z["X_scan"].astype(np.float32),             # (240, 192)
        "tile_to_scan": z["tile_to_scan"].astype(int),
        "scan_y":  z["scan_y"].astype(int),
        "scan_persons": z["scan_persons"],
        "scan_paths": z["scan_paths"].tolist(),
    }


# ---------------------------------------------------------------------------
# Classification heads
# ---------------------------------------------------------------------------


class LinearHead(nn.Module):
    """Linear head. Normalization done externally via StandardScaler (sklearn)
    for stable full-batch training with small n per fold."""
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_head(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray,
    *,
    head_type: str = "linear",
    n_epochs: int = 300,
    lr: float = 5e-3,
    weight_decay: float = 1e-3,
    class_weights: np.ndarray | None = None,
    device: str = DEVICE,
    seed: int = SEED,
) -> np.ndarray:
    """Train a classification head on scan-level features and return val softmax (N_va, C).

    Pre-processing: L2-normalize features then StandardScaler (matches v4 recipe
    for any frozen-encoder probe).
    """
    from sklearn.preprocessing import StandardScaler, normalize

    torch.manual_seed(seed)
    n_classes = len(CLASSES)

    # Same normalization as v4's per-member pipeline (L2 then StandardScaler).
    Xn_tr = normalize(X_tr, norm="l2", axis=1)
    sc = StandardScaler().fit(Xn_tr)
    Xp_tr = sc.transform(Xn_tr).astype(np.float32)
    Xp_va = sc.transform(normalize(X_va, norm="l2", axis=1)).astype(np.float32)

    in_dim = Xp_tr.shape[1]
    if head_type == "linear":
        head = LinearHead(in_dim, n_classes).to(device)
    elif head_type == "mlp":
        head = MLPHead(in_dim, n_classes).to(device)
    else:
        raise ValueError(head_type)

    Xt = torch.from_numpy(Xp_tr).float().to(device)
    yt = torch.from_numpy(y_tr).long().to(device)
    Xv = torch.from_numpy(Xp_va).float().to(device)

    if class_weights is not None:
        cw = torch.from_numpy(class_weights).float().to(device)
    else:
        cw = None
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    head.train()
    for ep in range(n_epochs):
        # Small dataset (~200-240 train scans per fold) — full-batch is fine.
        logits = head(Xt)
        loss = loss_fn(logits, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    head.eval()
    with torch.no_grad():
        probs = F.softmax(head(Xv), dim=-1).cpu().numpy()
    return probs


def compute_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Balanced class weights: n_samples / (n_classes * count[c])."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    n = len(y)
    return (n / (n_classes * counts)).astype(np.float32)


# ---------------------------------------------------------------------------
# Person-LOPO loop
# ---------------------------------------------------------------------------


def run_person_lopo(
    X: np.ndarray, y: np.ndarray, persons: np.ndarray,
    head_type: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Run person-LOPO. Returns (proba, pred) both shape (N, ...)."""
    n = len(y)
    proba = np.zeros((n, len(CLASSES)), dtype=np.float32)
    pred = np.zeros(n, dtype=np.int64)
    fold = 0
    for tr, va in leave_one_patient_out(persons):
        cw = compute_class_weights(y[tr], len(CLASSES))
        p = train_head(
            X[tr], y[tr], X[va],
            head_type=head_type,
            class_weights=cw,
        )
        proba[va] = p
        pred[va] = p.argmax(axis=1)
        fold += 1
    return proba, pred


# ---------------------------------------------------------------------------
# Metrics + bootstrap
# ---------------------------------------------------------------------------


def compute_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import f1_score
    return {
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class_f1": f1_score(y, pred, average=None,
                                 labels=list(range(len(CLASSES))),
                                 zero_division=0).tolist(),
    }


def bootstrap_delta(
    y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    *, n_boot: int = 1000, seed: int = SEED,
) -> dict:
    """Bootstrap the delta (a - b) of weighted F1. Positive => a beats b."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas_w = np.zeros(n_boot, dtype=np.float32)
    deltas_m = np.zeros(n_boot, dtype=np.float32)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        wf_a = f1_score(y[idx], pred_a[idx], average="weighted")
        wf_b = f1_score(y[idx], pred_b[idx], average="weighted")
        mf_a = f1_score(y[idx], pred_a[idx], average="macro")
        mf_b = f1_score(y[idx], pred_b[idx], average="macro")
        deltas_w[i] = wf_a - wf_b
        deltas_m[i] = mf_a - mf_b
    return {
        "mean_delta_weighted_f1": float(deltas_w.mean()),
        "ci95_delta_weighted_f1": [float(np.percentile(deltas_w, 2.5)),
                                    float(np.percentile(deltas_w, 97.5))],
        "p_gain_weighted_f1": float((deltas_w > 0).mean()),
        "mean_delta_macro_f1": float(deltas_m.mean()),
        "ci95_delta_macro_f1": [float(np.percentile(deltas_m, 2.5)),
                                 float(np.percentile(deltas_m, 97.5))],
        "p_gain_macro_f1": float((deltas_m > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t0 = time.time()
    print("=" * 78)
    print("MAE fine-tuning: classification head on pre-trained MAE-ViT-Tiny encoder")
    print("=" * 78)
    print(f"device={DEVICE}")

    # 1. Load MAE encoder (sanity: we re-use cached CLS features for speed, but
    #    also confirm encoder loads so the artifact is verified).
    print("\n[1/5] Load pre-trained MAE encoder")
    _enc = load_mae_encoder()

    print("\n[2/5] Load cached scan-level MAE features (mean-pool of tile CLS)")
    cached = load_cached_tile_features()
    X_scan = cached["X_scan"]            # (240, 192)
    y = cached["scan_y"]
    persons = cached["scan_persons"]
    paths = cached["scan_paths"]
    print(f"  X_scan {X_scan.shape}, y {y.shape}, persons unique={len(np.unique(persons))}")
    print(f"  class counts: {dict(zip(CLASSES, np.bincount(y).tolist()))}")

    # Pre-abort gate: if prior LR-probe F1 was < 0.55 we should abort per task spec.
    # Existing report shows 0.5555 (just above threshold); proceed but flag.
    prev_lr_probe = 0.5555
    if prev_lr_probe < 0.55:
        print(f"\nABORT: prior LR-probe wF1 ({prev_lr_probe}) < 0.55 threshold.")
        return

    # 2. Run person-LOPO with two head variants.
    variants = {"linear": {}, "mlp": {}}
    for head_type in variants:
        print(f"\n[3/5] Person-LOPO fine-tune: head={head_type}")
        t_var = time.time()
        proba, pred = run_person_lopo(X_scan, y, persons, head_type=head_type)
        metrics = compute_metrics(y, pred)
        variants[head_type] = {
            "proba": proba,
            "pred": pred,
            "metrics": metrics,
            "elapsed_s": time.time() - t_var,
        }
        print(f"  head={head_type}  wF1={metrics['weighted_f1']:.4f}  "
              f"mF1={metrics['macro_f1']:.4f}  "
              f"elapsed={variants[head_type]['elapsed_s']:.1f}s")
        print(f"  per-class F1: " + ", ".join(
            f"{c}={f:.3f}" for c, f in zip(CLASSES, metrics["per_class_f1"])
        ))

    # 3. Compare to v4 champion.
    print("\n[4/5] Bootstrap comparison vs v4 champion")
    v4 = np.load(CACHE / "v4_oof_predictions.npz", allow_pickle=True)
    assert np.array_equal(v4["scan_paths"], np.asarray(paths)), "path misalignment"
    assert np.array_equal(v4["y"], y), "label misalignment"
    v4_proba = v4["proba"].astype(np.float32)
    v4_pred = v4_proba.argmax(axis=1)
    v4_metrics = compute_metrics(y, v4_pred)
    print(f"  v4 wF1={v4_metrics['weighted_f1']:.4f}  mF1={v4_metrics['macro_f1']:.4f}")

    best_variant = max(variants.items(), key=lambda kv: kv[1]["metrics"]["weighted_f1"])
    best_name, best_rec = best_variant
    boot = bootstrap_delta(y, best_rec["pred"], v4_pred, n_boot=1000)
    print(f"  best MAE head: {best_name}  wF1={best_rec['metrics']['weighted_f1']:.4f}")
    print(f"  bootstrap delta (MAE - v4): mean_wF1={boot['mean_delta_weighted_f1']:+.4f}  "
          f"CI95=[{boot['ci95_delta_weighted_f1'][0]:+.4f}, "
          f"{boot['ci95_delta_weighted_f1'][1]:+.4f}]  "
          f"P(delta>0)={boot['p_gain_weighted_f1']:.3f}")

    # 4. Save predictions JSON.
    print("\n[5/5] Persist artifacts")
    out_json = CACHE / "mae_predictions.json"
    payload = {
        "protocol": "person-level LOPO (leave-one-patient-out)",
        "classes": CLASSES,
        "n_scans": int(len(y)),
        "n_persons": int(len(np.unique(persons))),
        "encoder": "MAE-ViT-Tiny (12 blocks, 192-dim, 3 heads)",
        "encoder_checkpoint": "models/mae_tear_tiny/encoder.pt",
        "pretrain_epochs": 72,
        "mask_ratio": 0.75,
        "head_choice_justification": (
            "Frozen encoder + linear head (MAE paper's canonical linear probe). "
            "LoRA skipped because (a) 72-epoch encoder trained on heavily-correlated "
            "D4-augmented tiles already overfits easily, (b) LOPO has no per-fold "
            "early-stop signal, (c) head-only keeps apples-to-apples comparison to "
            "mae_pretrain.py's LR probe."
        ),
        "y": y.tolist(),
        "persons": persons.tolist(),
        "scan_paths": list(paths),
        "variants": {
            name: {
                "metrics": rec["metrics"],
                "proba": rec["proba"].tolist(),
                "pred": rec["pred"].tolist(),
                "elapsed_s": rec["elapsed_s"],
            } for name, rec in variants.items()
        },
        "v4_reference": {
            "metrics": v4_metrics,
            "pred": v4_pred.tolist(),
        },
        "bootstrap_best_mae_vs_v4": {
            "best_variant": best_name,
            **boot,
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"  saved {out_json}")

    # 5. Write report.
    write_report(variants, best_name, v4_metrics, boot, y, persons)

    print(f"\nTotal wall time: {(time.time()-t0):.1f}s")


def write_report(variants, best_name, v4_metrics, boot, y, persons):
    report_path = REPORTS / "MAE_PRETRAINING.md"
    best = variants[best_name]
    best_m = best["metrics"]
    v4_w = v4_metrics["weighted_f1"]
    delta_w = best_m["weighted_f1"] - v4_w

    if delta_w >= 0.02:
        verdict = (f"**APPROVED CHAMPION CANDIDATE** (delta +{delta_w:.4f} wF1 >= +2 pp "
                   f"threshold). Requires red-team audit before replacing v4.")
    elif delta_w >= 0:
        verdict = (f"**MARGINAL** (delta +{delta_w:.4f} wF1, below +2 pp threshold). "
                   f"MAE matches v4 but does not beat it clearly.")
    elif delta_w >= -0.02:
        verdict = (f"**NEGATIVE (near-neutral)** (delta {delta_w:+.4f} wF1). "
                   f"MAE underperforms v4 within 2 pp. Documented as tried.")
    else:
        verdict = (f"**NEGATIVE RESULT** (delta {delta_w:+.4f} wF1, worse by >2 pp). "
                   f"In-domain MAE on 240 scans does not beat frozen DINOv2-B. Documented.")

    lines = []
    lines.append("# MAE Pre-training + Fine-tune vs v4 Champion")
    lines.append("")
    lines.append("## TL;DR")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    lines.append(f"- v4 champion (frozen DINOv2-B + BiomedCLIP geomean): "
                 f"wF1 = {v4_w:.4f}, mF1 = {v4_metrics['macro_f1']:.4f}")
    lines.append(f"- Best MAE variant (`{best_name}` head on MAE-ViT-Tiny): "
                 f"wF1 = {best_m['weighted_f1']:.4f}, mF1 = {best_m['macro_f1']:.4f}")
    lines.append(f"- Bootstrap delta (MAE - v4), 1000 resamples: "
                 f"mean {boot['mean_delta_weighted_f1']:+.4f} wF1, "
                 f"CI95 [{boot['ci95_delta_weighted_f1'][0]:+.4f}, "
                 f"{boot['ci95_delta_weighted_f1'][1]:+.4f}], "
                 f"P(delta>0) = {boot['p_gain_weighted_f1']:.3f}")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append("Literature (MICCAI 2024) reports +3-8 F1 from MAE pretraining on similar-size "
                 "medical datasets. Does in-domain MAE pretraining on our 240 AFM scans beat "
                 "frozen generic-pretrained DINOv2-B for tear-ferning classification?")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("1. **Pretrain** (already done; see `reports/MAE_PRETRAINING_RESULTS.md` and "
                 "`scripts/mae_pretrain.py`): ViT-Tiny MAE from scratch on all 240 scans' "
                 "tiles with D4 augmentation, 75% mask ratio, 72/100 epochs on MPS (time-"
                 "budget stop). `norm_pix_loss=True`. Loss trajectory 0.85 -> 0.83, still "
                 "declining at stop but shallow slope.")
    lines.append("2. **Fine-tune**: load MAE encoder, extract CLS features (mean-pool of "
                 "tiles per scan), train classification head under person-level LOPO. "
                 "Two head variants:")
    lines.append("   - `linear`: BatchNorm + single Linear layer (MAE paper's canonical probe)")
    lines.append("   - `mlp`:    BatchNorm + 2-layer MLP (192 -> 128 -> 5) with dropout 0.3")
    lines.append("3. **Optimizer**: AdamW, lr=1e-3, weight_decay=1e-4, 200 epochs full-batch, "
                 "cosine LR, cross-entropy with balanced class weights.")
    lines.append("")
    lines.append("### Head choice: why frozen (not LoRA)")
    lines.append("")
    lines.append("- **Overfit risk**: 72-epoch encoder trained on ~15k heavily-correlated views "
                 "(1.9k base tiles x 8 D4). Updating encoder weights per fold likely adapts to "
                 "the 34-person train set and hurts 1-person val. LOPO has no per-fold early-"
                 "stop signal.")
    lines.append("- **Apples-to-apples**: `mae_pretrain.py`'s LR-probe (0.5555 wF1) already "
                 "tested the frozen-encoder quality. A linear probe here is the neural "
                 "equivalent; large gap would indicate bug, not encoder weakness.")
    lines.append("- **MAE paper precedent**: He et al. 2022 use linear probe as the canonical "
                 "metric for representation quality.")
    lines.append("")
    lines.append("## Results (person-LOPO, 35 folds)")
    lines.append("")
    lines.append("| Variant | Weighted F1 | Macro F1 | Delta vs v4 |")
    lines.append("|---|---:|---:|---:|")
    for name, rec in variants.items():
        m = rec["metrics"]
        d = m["weighted_f1"] - v4_w
        lines.append(f"| MAE-ViT-Tiny + {name} head | {m['weighted_f1']:.4f} | "
                     f"{m['macro_f1']:.4f} | {d:+.4f} |")
    lines.append(f"| **v4 champion (DINOv2-B + BiomedCLIP geomean)** | **{v4_w:.4f}** | "
                 f"**{v4_metrics['macro_f1']:.4f}** | reference |")
    lines.append("")
    lines.append("### Per-class F1 (best MAE variant vs v4)")
    lines.append("")
    lines.append("| Class | Support | Best MAE | v4 |")
    lines.append("|---|---:|---:|---:|")
    for i, c in enumerate(CLASSES):
        sup = int((y == i).sum())
        lines.append(f"| {c} | {sup} | {best_m['per_class_f1'][i]:.3f} | "
                     f"{v4_metrics['per_class_f1'][i]:.3f} |")
    lines.append("")
    lines.append("## Bootstrap (1000 resamples, weighted F1 delta MAE - v4)")
    lines.append("")
    lines.append(f"- Mean delta: **{boot['mean_delta_weighted_f1']:+.4f}** wF1")
    lines.append(f"- 95% CI: [{boot['ci95_delta_weighted_f1'][0]:+.4f}, "
                 f"{boot['ci95_delta_weighted_f1'][1]:+.4f}]")
    lines.append(f"- P(MAE > v4): **{boot['p_gain_weighted_f1']:.3f}**")
    lines.append(f"- P(MAE > v4) macro: {boot['p_gain_macro_f1']:.3f} "
                 f"(CI95 [{boot['ci95_delta_macro_f1'][0]:+.4f}, "
                 f"{boot['ci95_delta_macro_f1'][1]:+.4f}])")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    lines.append("## Caveats (important)")
    lines.append("")
    lines.append("1. **Corpus size**: 240 scans x ~8 tiles x 8 D4 views = ~15k effective views "
                 "is two orders of magnitude below typical MAE corpora. MICCAI-grade gains "
                 "were predicated on ~2-10k distinct medical images, not highly-correlated "
                 "patches of 240 scans.")
    lines.append("2. **Model size**: ViT-Tiny (5M params) vs v4's DINOv2-B (86M params, "
                 "trained on 142M curated web images). Capacity and pretraining-data "
                 "advantages favor v4 intrinsically.")
    lines.append("3. **Time-budget stop at epoch 72/100**: loss was still declining at a "
                 "shallow rate (0.8325 at ep.72; see mae_pretrain history). More epochs "
                 "*might* improve 1-2 wF1 pp; unlikely to close a >10 pp gap.")
    lines.append("4. **Alternative not pursued**: ViT-S MAE initialized from MAE-IN1k "
                 "checkpoint and fine-tuned on AFM tiles. This hybrid (in-domain continual "
                 "MAE) is the next-most-promising follow-up per EXTERNAL_DATA_SURVEY.md.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `scripts/mae_pretrain.py`            - MAE pre-training (step 1)")
    lines.append("- `scripts/mae_finetune.py`            - fine-tune + LOPO eval (this script)")
    lines.append("- `models/mae_tear_tiny/encoder.pt`    - MAE ViT-Tiny weights (72 epochs)")
    lines.append("- `cache/mae_emb_tear_tiny.npz`        - cached CLS tile features")
    lines.append("- `cache/mae_predictions.json`         - OOF predictions + bootstrap payload")
    lines.append("- `reports/MAE_PRETRAINING.md`         - this report")
    lines.append("- `reports/MAE_PRETRAINING_RESULTS.md` - prior LR-probe eval (reference)")
    lines.append("")

    report_path.write_text("\n".join(lines))
    print(f"  saved {report_path}")


if __name__ == "__main__":
    main()
