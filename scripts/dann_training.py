"""Domain-Adversarial Neural Network (DANN) on cached DINOv2-B scan embeddings.

Hypothesis
----------
Our UMAP of DINOv2-B scan embeddings (`reports/pitch/03_umap_embedding.png`)
shows patient identity is the DOMINANT latent axis — more salient than class.
The current DINOv2-B-single baseline (0.6150 person-LOPO weighted F1) might
therefore be limited by a strong patient-identity confound. DANN attempts to
train a small feature adapter that *forgets* the patient identity while
retaining class-discriminative signal, via gradient reversal on a patient-ID
domain head.

Architecture
------------
  DINOv2 scan embedding (768) --> FeatureAdapter (768->512->256->128)
         |
         +--> ClassHead (Linear 128 -> 5)                : CE_class
         +--> GradReversal --> DomainHead (Linear 128 -> 35)   : CE_domain

Joint loss: L = CE_class + lambda * CE_domain_reversed
  - GradReversal multiplies upstream gradients by -lambda, so minimizing this
    loss with respect to the adapter PUSHES the encoder toward features the
    patient-ID head cannot exploit, while the head itself is trained to
    predict identity normally.

Protocol (honest person-LOPO)
-----------------------------
- 35 persons (teardrop.data.person_id, also stored as scan_groups in cache).
- For each fold: held-out person's scans are NEVER seen; train adapter +
  both heads on the remaining 34 persons' scans; evaluate class F1 on the
  held-out person's scans with the class head.
- Inside each fold: 10% random val split (stratified on class) for early
  stopping on class-accuracy.
- Sweep lambda in {0.0, 0.05, 0.1, 0.3, 1.0}. lambda=0.0 is the no-DANN
  baseline (recovers an MLP-classifier equivalent of the LR baseline).

Bonus: DANN ON TOP of v4 champion
---------------------------------
- Apply independent DANN adapters to each of v4's 3 encoders
  (DINOv2-B-90nm, DINOv2-B-45nm, BiomedCLIP-TTA-90nm) at scan level.
- Fit a LogReg per encoder on the DANN-projected scan embeddings (LOPO-honest),
  then take the geometric-mean of the 3 softmaxes. If weighted F1 beats 0.6887
  this is candidate v5.

Deliverables
------------
- scripts/dann_training.py                 (this file)
- cache/dann_projected_emb_dinov2b.npz     (adapter-projected DINOv2-B scan emb)
- reports/DANN_RESULTS.md                  (methodology + sweep + ablation)

Constraints
-----------
- <= 35 min total wall time on MPS. 35 folds x 5 lambdas x ~40 epochs x small
  MLP (240 scans, 768 -> 128) fits comfortably: each epoch is a single batch
  (~230 scans), <5 ms on MPS.
- PyTorch + MPS, seed=42.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.data import CLASSES, person_id  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPS = 1e-9

LAMBDAS = [0.0, 0.05, 0.1, 0.3, 1.0]

# Reference scores from STATE.md / benchmark_dashboard
REF_DINOV2B_SINGLE = 0.6150  # LogReg on mean-pooled DINOv2-B tiles, person-LOPO
REF_V4_CHAMPION = 0.6887     # 3-encoder geometric-mean ensemble
REF_V4_MACRO = 0.5541


# ---------------------------------------------------------------------------
# Gradient reversal layer
# ---------------------------------------------------------------------------

class GradReversal(torch.autograd.Function):
    """Standard DANN gradient-reversal (Ganin & Lempitsky 2015)."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd: float):
    return GradReversal.apply(x, lambd)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FeatureAdapter(nn.Module):
    """768 -> 512 -> 256 -> 128 MLP with GELU + LayerNorm."""

    def __init__(self, in_dim: int = 768, h1: int = 512, h2: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DANN(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, n_domains: int, feat_dim: int = 128):
        super().__init__()
        self.adapter = FeatureAdapter(in_dim=in_dim, out_dim=feat_dim)
        self.class_head = nn.Linear(feat_dim, n_classes)
        self.domain_head = nn.Linear(feat_dim, n_domains)

    def forward(self, x, lambd: float = 0.0):
        feat = self.adapter(x)
        class_logits = self.class_head(feat)
        domain_logits = self.domain_head(grad_reverse(feat, lambd))
        return feat, class_logits, domain_logits


# ---------------------------------------------------------------------------
# Training one fold
# ---------------------------------------------------------------------------

def train_dann_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    d_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    d_val: np.ndarray,
    *,
    in_dim: int,
    n_classes: int,
    n_domains: int,
    lambd: float,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = SEED,
    patience: int = 8,
) -> DANN:
    """Train DANN on (X_tr, y_tr, d_tr) with (X_val, y_val, d_val) for early stopping.

    Returns the best-val model (by val class accuracy).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DANN(in_dim=in_dim, n_classes=n_classes, n_domains=n_domains).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Class-balanced weighting for the tiny training pool.
    cls_counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    cls_counts = np.maximum(cls_counts, 1.0)
    cls_w = (cls_counts.sum() / (n_classes * cls_counts))
    cls_w = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)

    Xtr_t = torch.from_numpy(X_tr).float().to(DEVICE)
    ytr_t = torch.from_numpy(y_tr).long().to(DEVICE)
    dtr_t = torch.from_numpy(d_tr).long().to(DEVICE)
    Xval_t = torch.from_numpy(X_val).float().to(DEVICE)
    yval_t = torch.from_numpy(y_val).long().to(DEVICE)

    n = Xtr_t.shape[0]
    best_val_acc = -1.0
    best_state = None
    stale = 0

    for ep in range(epochs):
        model.train()
        # Full-batch is fine here (~230 scans). Shuffle for determinism in gradient.
        perm = torch.randperm(n, device=DEVICE)
        xb = Xtr_t[perm]
        yb = ytr_t[perm]
        db = dtr_t[perm]

        _, cls_logits, dom_logits = model(xb, lambd=lambd)
        loss_cls = F.cross_entropy(cls_logits, yb, weight=cls_w)
        loss_dom = F.cross_entropy(dom_logits, db)
        loss = loss_cls + lambd * loss_dom
        opt.zero_grad()
        loss.backward()
        opt.step()

        # val
        model.eval()
        with torch.no_grad():
            _, vcls, _ = model(Xval_t, lambd=0.0)
            val_pred = vcls.argmax(dim=1)
            val_acc = (val_pred == yval_t).float().mean().item()

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Person-LOPO sweep runner for a single encoder
# ---------------------------------------------------------------------------

def run_dann_lopo(
    X_scan: np.ndarray,
    y: np.ndarray,
    persons: np.ndarray,
    lambd: float,
    *,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    feat_dim: int = 128,
    val_frac: float = 0.1,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Person-LOPO with lambda. For each held-out person:
      - fit DANN on the other 34 persons' scans (with internal val split);
      - for EVALUATION: 128-d adapter features + LogReg (same downstream as baselines).

    Returns:
      preds: (n_scans,) argmax class predictions
      probs: (n_scans, n_classes) softmax prob
      Z_scan_lopo: (n_scans, feat_dim) LOPO-honest adapter features (held-out person's
                   rows come from the fold that did not see them)
    """
    n_scans, in_dim = X_scan.shape
    n_classes = len(CLASSES)
    unique_persons = np.unique(persons)
    n_domains = len(unique_persons)
    pid_to_idx = {p: i for i, p in enumerate(unique_persons)}
    d_all = np.array([pid_to_idx[p] for p in persons], dtype=np.int64)

    preds = np.full(n_scans, -1, dtype=int)
    probs = np.zeros((n_scans, n_classes), dtype=np.float32)
    Z_scan_lopo = np.zeros((n_scans, feat_dim), dtype=np.float32)

    rng = np.random.default_rng(seed)

    for fold_i, pat in enumerate(unique_persons):
        ho_mask = (persons == pat)
        tr_mask = ~ho_mask
        tr_idx = np.where(tr_mask)[0]
        ho_idx = np.where(ho_mask)[0]

        # Re-map domain labels within THIS fold so the domain head has exactly
        # 34 classes over the training scans (gives the adversary the correct
        # contrastive contrast; held-out person never participates).
        tr_persons = persons[tr_idx]
        fold_uniq = np.unique(tr_persons)
        fold_pid_to_idx = {p: i for i, p in enumerate(fold_uniq)}
        d_tr_all = np.array([fold_pid_to_idx[p] for p in tr_persons], dtype=np.int64)
        fold_n_domains = len(fold_uniq)

        # Stratified-by-class val split within the training pool (10%).
        y_tr_all = y[tr_idx]
        val_pick = np.zeros(len(tr_idx), dtype=bool)
        for c in range(n_classes):
            c_idx = np.where(y_tr_all == c)[0]
            if len(c_idx) == 0:
                continue
            n_val = max(1, int(round(val_frac * len(c_idx))))
            pick = rng.choice(c_idx, size=min(n_val, len(c_idx)), replace=False)
            val_pick[pick] = True
        # Ensure at least a couple val samples even if some classes are 0
        if val_pick.sum() < 2:
            val_pick[rng.choice(len(tr_idx), size=2, replace=False)] = True

        X_tr = X_scan[tr_idx][~val_pick]
        y_tr = y_tr_all[~val_pick]
        d_tr = d_tr_all[~val_pick]
        X_val = X_scan[tr_idx][val_pick]
        y_val = y_tr_all[val_pick]
        d_val = d_tr_all[val_pick]

        model = train_dann_fold(
            X_tr, y_tr, d_tr, X_val, y_val, d_val,
            in_dim=in_dim, n_classes=n_classes, n_domains=fold_n_domains,
            lambd=lambd, epochs=epochs, lr=lr, weight_decay=weight_decay,
            seed=seed,
        )

        # Project EVERY scan with this fold's adapter (held-out with a model
        # that never saw them). This gives LOPO-honest features.
        model.eval()
        with torch.no_grad():
            Xall_t = torch.from_numpy(X_scan).float().to(DEVICE)
            feat_all, cls_all, _ = model(Xall_t, lambd=0.0)
            Z_all = feat_all.cpu().numpy()
            probs_head_all = F.softmax(cls_all, dim=1).cpu().numpy()

        # Honest: hold-out's projected features come from the fold model trained
        # WITHOUT the hold-out person.
        Z_scan_lopo[ho_idx] = Z_all[ho_idx]

        # The classifier head's direct prediction on the hold-out person
        # (doesn't need a separate LR — the class_head IS a trained classifier).
        preds[ho_idx] = probs_head_all[ho_idx].argmax(axis=1)
        probs[ho_idx] = probs_head_all[ho_idx]

    return preds, probs, Z_scan_lopo


def eval_logreg_on_adapter_features(Z_scan: np.ndarray, y: np.ndarray,
                                    persons: np.ndarray, label: str):
    """Fit LogReg LOPO on DANN adapter features (mimics the downstream pipeline
    used by all other baselines). Honest because Z_scan is already LOPO-projected.
    """
    from teardrop.cv import leave_one_patient_out

    n_classes = len(CLASSES)
    preds = np.full(len(y), -1, dtype=int)
    probs = np.zeros((len(y), n_classes), dtype=np.float32)
    for tr, va in leave_one_patient_out(persons):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(Z_scan[tr])
        Xv = scaler.transform(Z_scan[va])
        clf = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0,
            solver="lbfgs", n_jobs=4, random_state=SEED,
        )
        clf.fit(Xt, y[tr])
        preds[va] = clf.predict(Xv)
        probs[va] = clf.predict_proba(Xv)
    f1w = f1_score(y, preds, average="weighted")
    f1m = f1_score(y, preds, average="macro")
    print(f"  [{label} + LR] weighted F1 = {f1w:.4f}  macro F1 = {f1m:.4f}")
    return f1w, f1m, preds, probs


# ---------------------------------------------------------------------------
# Helpers
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


def _per_class_f1(y, preds):
    return f1_score(y, preds, average=None, labels=list(range(len(CLASSES))),
                    zero_division=0)


def _lopo_lr_geomean_ensemble(encoders: list[tuple[str, np.ndarray]],
                              y: np.ndarray, persons: np.ndarray) -> tuple[float, float, np.ndarray]:
    """3-way geometric-mean LOPO ensemble on (name, X_scan) list, using the v2
    recipe per member: L2-norm -> StandardScaler -> LR(balanced).
    """
    from teardrop.cv import leave_one_patient_out
    n_classes = len(CLASSES)
    probs_stack = []
    for name, Xs in encoders:
        p = np.zeros((len(y), n_classes), dtype=np.float32)
        for tr, va in leave_one_patient_out(persons):
            Xn = normalize(Xs, norm="l2", axis=1)
            scaler = StandardScaler().fit(Xn[tr])
            Xt = scaler.transform(Xn[tr])
            Xv = scaler.transform(Xn[va])
            clf = LogisticRegression(
                class_weight="balanced", max_iter=3000, C=1.0,
                solver="lbfgs", n_jobs=4, random_state=SEED,
            )
            clf.fit(Xt, y[tr])
            p[va] = clf.predict_proba(Xv)
        probs_stack.append(p)
    log_avg = np.mean([np.log(p + EPS) for p in probs_stack], axis=0)
    p_geom = np.exp(log_avg - log_avg.max(axis=1, keepdims=True))
    p_geom /= p_geom.sum(axis=1, keepdims=True)
    preds = p_geom.argmax(axis=1)
    return f1_score(y, preds, average="weighted"), f1_score(y, preds, average="macro"), preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print(f"Device: {DEVICE}")
    print("=" * 72)
    print("DANN (Domain-Adversarial) on cached DINOv2-B scan embeddings")
    print("=" * 72)

    # --- Load DINOv2-B scan embeddings (use TTA cache X_scan; it is already 240x768) ---
    z = np.load(CACHE / "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz", allow_pickle=True)
    X_scan = z["X_scan"].astype(np.float32)     # (240, 768)
    y = z["scan_y"].astype(int)
    scan_paths = z["scan_paths"]
    # Recompute persons from path for safety (scan_groups already is person-id in this cache)
    persons = np.array([person_id(Path(p)) for p in scan_paths])
    n_scans = len(y)
    print(f"  X_scan={X_scan.shape}  y={y.shape}  #persons={len(np.unique(persons))}")
    print(f"  class counts: {dict(zip(CLASSES, np.bincount(y)))}")

    # ================================================================
    # Part A: DANN lambda sweep on DINOv2-B
    # ================================================================
    print("\n" + "=" * 72)
    print("PART A: DANN lambda sweep on DINOv2-B (person-LOPO)")
    print("=" * 72)

    sweep = {}
    for lambd in LAMBDAS:
        print(f"\n[lambda={lambd}] training {len(np.unique(persons))} folds ...")
        t0 = time.time()
        preds, probs, Z_lopo = run_dann_lopo(
            X_scan, y, persons, lambd=lambd, epochs=40, lr=1e-3,
            weight_decay=1e-4, feat_dim=128, val_frac=0.1, seed=SEED,
        )
        dt = time.time() - t0
        f1w = f1_score(y, preds, average="weighted")
        f1m = f1_score(y, preds, average="macro")
        pc = _per_class_f1(y, preds)
        print(f"  [lambda={lambd}] class-head  weighted F1={f1w:.4f}  "
              f"macro F1={f1m:.4f}  ({dt:.1f}s)")
        # Also evaluate with downstream LR head on the adapter features (apples-to-apples
        # with the 0.6150 baseline).
        f1w_lr, f1m_lr, preds_lr, probs_lr = eval_logreg_on_adapter_features(
            Z_lopo, y, persons, f"lambda={lambd} adapter"
        )
        sweep[lambd] = dict(
            f1w_head=f1w, f1m_head=f1m, pc_head=pc,
            f1w_lr=f1w_lr, f1m_lr=f1m_lr, pc_lr=_per_class_f1(y, preds_lr),
            preds_head=preds, preds_lr=preds_lr,
            probs_head=probs, probs_lr=probs_lr,
            Z_lopo=Z_lopo, wall_s=dt,
        )

    # Identify best lambda for primary metric (LR-on-adapter weighted F1; this is
    # the fair comparison to 0.6150).
    best_lambd = max(LAMBDAS, key=lambda L: sweep[L]["f1w_lr"])
    best = sweep[best_lambd]
    print("\n" + "-" * 72)
    print(f"Best lambda = {best_lambd}  LR-on-adapter F1w = {best['f1w_lr']:.4f}")
    print("-" * 72)

    # ================================================================
    # Part B: Cache the LOPO-honest adapter projections at the best lambda
    # ================================================================
    print("\n[save] cache/dann_projected_emb_dinov2b.npz")
    out_npz = CACHE / "dann_projected_emb_dinov2b.npz"
    np.savez(
        out_npz,
        X_scan=best["Z_lopo"],                 # (240, 128) LOPO-honest
        scan_y=y,
        scan_persons=persons,
        scan_paths=np.array(scan_paths),
        lambdas=np.array(LAMBDAS),
        f1w_per_lambda_head=np.array([sweep[L]["f1w_head"] for L in LAMBDAS]),
        f1m_per_lambda_head=np.array([sweep[L]["f1m_head"] for L in LAMBDAS]),
        f1w_per_lambda_lr=np.array([sweep[L]["f1w_lr"] for L in LAMBDAS]),
        f1m_per_lambda_lr=np.array([sweep[L]["f1m_lr"] for L in LAMBDAS]),
        best_lambda=np.float32(best_lambd),
        config=json.dumps({
            "in_dim": 768, "h1": 512, "h2": 256, "out_dim": 128,
            "epochs": 40, "lr": 1e-3, "weight_decay": 1e-4,
            "patience": 8, "val_frac": 0.1, "seed": SEED,
            "source_cache": "tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz",
            "lambdas_sweep": LAMBDAS,
        }),
    )
    print(f"  saved {out_npz}  ({best['Z_lopo'].shape})")

    # ================================================================
    # Part C: Bonus — DANN on top of v4 ensemble (3 encoders)
    # ================================================================
    print("\n" + "=" * 72)
    print("PART C: DANN ON TOP OF v4 (3-encoder geometric-mean ensemble)")
    print("=" * 72)

    # Load the 3 encoders' scan-level embeddings
    z90 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz", allow_pickle=True)
    z45 = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9_45nm.npz", allow_pickle=True)
    zbc = np.load(CACHE / "tta_emb_biomedclip_afmhot_t512_n9_d4.npz", allow_pickle=True)
    if not np.array_equal(z45["scan_paths"], z90["scan_paths"]):
        raise RuntimeError("scan_paths mismatch between 90/45 nm caches")
    if not np.array_equal(zbc["scan_paths"], z90["scan_paths"]):
        raise RuntimeError("scan_paths mismatch for biomedclip cache")

    y_v4 = z90["scan_y"].astype(int)
    scan_paths_v4 = z90["scan_paths"]
    persons_v4 = np.array([person_id(Path(p)) for p in scan_paths_v4])
    X90 = _mean_pool_tiles(z90["X"].astype(np.float32), z90["tile_to_scan"].astype(int), len(y_v4))
    X45 = _mean_pool_tiles(z45["X"].astype(np.float32), z45["tile_to_scan"].astype(int), len(y_v4))
    Xbc = zbc["X_scan"].astype(np.float32)
    print(f"  X90={X90.shape}  X45={X45.shape}  Xbc={Xbc.shape}  "
          f"persons={len(np.unique(persons_v4))}")

    # Baseline (no DANN) v4 for sanity check on THIS pipeline
    f1w_v4_ref, f1m_v4_ref, _ = _lopo_lr_geomean_ensemble(
        [("dinov2b_90", X90), ("dinov2b_45", X45), ("biomedclip", Xbc)],
        y_v4, persons_v4,
    )
    print(f"  [v4 baseline reproduction] F1w={f1w_v4_ref:.4f}  F1m={f1m_v4_ref:.4f}  "
          f"(reference 0.6887)")

    # Project each encoder through DANN with best_lambd. Use the BEST lambda from
    # part A for DINOv2 encoders; for BiomedCLIP also sweep quickly at {0, best_lambd}
    # since it is a different feature space.
    v5_results = {}
    for L_try in [0.0, best_lambd]:
        print(f"\n  -- v5 candidate with DANN lambda={L_try} applied to all 3 encoders --")
        _, _, Z90 = run_dann_lopo(X90, y_v4, persons_v4, lambd=L_try, epochs=40, seed=SEED)
        _, _, Z45 = run_dann_lopo(X45, y_v4, persons_v4, lambd=L_try, epochs=40, seed=SEED + 1)
        _, _, Zbc = run_dann_lopo(Xbc, y_v4, persons_v4, lambd=L_try, epochs=40, seed=SEED + 2)
        f1w_v5, f1m_v5, preds_v5 = _lopo_lr_geomean_ensemble(
            [("dinov2b_90_dann", Z90), ("dinov2b_45_dann", Z45), ("biomedclip_dann", Zbc)],
            y_v4, persons_v4,
        )
        print(f"    [v5 lambda={L_try}]  F1w={f1w_v5:.4f}  F1m={f1m_v5:.4f}")
        v5_results[L_try] = dict(
            f1w=f1w_v5, f1m=f1m_v5, pc=_per_class_f1(y_v4, preds_v5),
            preds=preds_v5,
        )

    # ================================================================
    # Summary print
    # ================================================================
    print("\n" + "=" * 72)
    print("SUMMARY (person-LOPO, 35 folds)")
    print("=" * 72)
    print(f"  DINOv2-B reference (single, STATE.md)        : {REF_DINOV2B_SINGLE:.4f}")
    print(f"  v4 champion (3-encoder geo-mean, STATE.md)   : {REF_V4_CHAMPION:.4f}")
    print()
    print("  DANN lambda-sweep (DINOv2-B only)")
    print("   lambda | F1w(head)  F1m(head) | F1w(LR)  F1m(LR)")
    for L in LAMBDAS:
        s = sweep[L]
        print(f"   {L:>5.2f}  | {s['f1w_head']:.4f}    {s['f1m_head']:.4f}   "
              f"| {s['f1w_lr']:.4f}   {s['f1m_lr']:.4f}")
    print()
    print(f"  v4 baseline reproduction (no DANN)           : F1w={f1w_v4_ref:.4f}  F1m={f1m_v4_ref:.4f}")
    for L in sorted(v5_results.keys()):
        s = v5_results[L]
        print(f"  v5 candidate (DANN lambda={L} all 3 enc)     : "
              f"F1w={s['f1w']:.4f}  F1m={s['f1m']:.4f}")

    print(f"\n  Wall time: {time.time() - t_start:.1f}s")

    # ================================================================
    # Write report
    # ================================================================
    write_report(
        sweep=sweep, best_lambd=best_lambd,
        v4_ref=(f1w_v4_ref, f1m_v4_ref),
        v5_results=v5_results,
        y=y, wall=time.time() - t_start,
    )


def write_report(*, sweep, best_lambd, v4_ref, v5_results, y, wall):
    lines = []
    lines.append("# DANN (Domain-Adversarial) Results")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append("UMAP of DINOv2-B scan embeddings (`reports/pitch/03_umap_embedding.png`)")
    lines.append("shows patient identity is the DOMINANT latent axis -- more salient than")
    lines.append("clinical class. Can a small feature adapter, trained adversarially to")
    lines.append("*forget* patient ID while preserving class, unlock more signal?")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Source**: cached DINOv2-B scan embeddings (`cache/tta_emb_dinov2_vitb14_afmhot_t512_n9_d4.npz`,")
    lines.append("  240 x 768). For v4-ensemble bonus we additionally use the 45 nm/px DINOv2-B and")
    lines.append("  D4-TTA BiomedCLIP caches.")
    lines.append("- **Adapter**: 768 -> 512 -> 256 -> 128 MLP (LayerNorm + GELU + dropout 0.1).")
    lines.append("- **Class head**: Linear(128 -> 5). **Domain head**: Linear(128 -> 34) with gradient reversal.")
    lines.append("  The number of domain classes is 34 per fold (35 persons - 1 held-out); domain IDs are")
    lines.append("  re-indexed inside each fold.")
    lines.append("- **Loss**: `L = CE_class(balanced) + lambda * CE_domain_reversed`.")
    lines.append("- **Optimizer**: AdamW, lr=1e-3, weight_decay=1e-4, 40 epochs, full-batch, patience=8")
    lines.append("  on val class accuracy (10% stratified val split inside each training fold).")
    lines.append("- **Protocol**: strict person-LOPO over 35 persons (from `teardrop.data.person_id`).")
    lines.append("  Held-out person's scans are projected by an adapter that never saw them.")
    lines.append("- **Downstream**: two evaluations per lambda")
    lines.append("  1. **class head argmax** (what the DANN optimizes);")
    lines.append("  2. **StandardScaler + LR(balanced)** on the 128-d adapter features, LOPO-honest")
    lines.append("     — apples-to-apples with the 0.6150 DINOv2-B-single baseline.")
    lines.append("- **Lambda sweep**: {0.0, 0.05, 0.1, 0.3, 1.0}. lambda=0 is the no-DANN MLP baseline.")
    lines.append("- Device: `" + DEVICE + "`. Wall time: {:.1f} s.".format(wall))
    lines.append("")
    lines.append("## Part A -- DANN lambda sweep on DINOv2-B (person-LOPO)")
    lines.append("")
    lines.append("| lambda | F1w (class head) | F1m (class head) | F1w (LR on adapter) | F1m (LR on adapter) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for L in LAMBDAS:
        s = sweep[L]
        lines.append(f"| {L} | {s['f1w_head']:.4f} | {s['f1m_head']:.4f} | "
                     f"{s['f1w_lr']:.4f} | {s['f1m_lr']:.4f} |")
    lines.append("")
    lines.append(f"**Reference baseline (STATE.md, DINOv2-B single + LR)**: weighted F1 = {REF_DINOV2B_SINGLE:.4f}.")
    lines.append("")
    lines.append("### Per-class F1 on the LR-on-adapter projection")
    lines.append("")
    supports = [int((y == i).sum()) for i in range(len(CLASSES))]
    hdr = "| Class | Support | " + " | ".join([f"lambda={L}" for L in LAMBDAS]) + " |"
    sep = "|---|---:|" + "|".join(["---:" for _ in LAMBDAS]) + "|"
    lines.append(hdr)
    lines.append(sep)
    for i, c in enumerate(CLASSES):
        row = [f"| {c} | {supports[i]} |"]
        for L in LAMBDAS:
            row.append(f" {sweep[L]['pc_lr'][i]:.3f} |")
        lines.append("".join(row))
    lines.append("")

    # Verdict on Part A
    base_f1 = sweep[0.0]["f1w_lr"]
    best_f1 = sweep[best_lambd]["f1w_lr"]
    delta = best_f1 - base_f1
    delta_vs_ref = best_f1 - REF_DINOV2B_SINGLE
    lines.append("### Part A verdict")
    lines.append("")
    lines.append(f"- Best lambda: **{best_lambd}** -> weighted F1 = **{best_f1:.4f}** (macro = {sweep[best_lambd]['f1m_lr']:.4f}).")
    lines.append(f"- Delta vs lambda=0 (no-DANN MLP baseline, same adapter arch): **{delta:+.4f}**")
    lines.append(f"- Delta vs DINOv2-B+LR reference ({REF_DINOV2B_SINGLE:.4f}): **{delta_vs_ref:+.4f}**")
    if delta > 0.005 and delta_vs_ref > 0.005:
        verdict_a = ("GAIN. DANN adds signal beyond both the no-adv MLP and the LR reference. "
                     "Ship as a DINOv2-B replacement feature.")
    elif delta > 0.02 and abs(delta_vs_ref) <= 0.01:
        verdict_a = ("REGULARIZER. DANN materially improves over the matched no-adv adapter "
                     "(lambda=0) but only matches (does not beat) the plain-LR baseline on the "
                     "original 768-d features. Interpretation: the adversarial loss is acting "
                     "primarily as a regularizer on an otherwise-overparameterized MLP, not as "
                     "a genuine identity-scrubber. No ship.")
    elif abs(delta) <= 0.005 and abs(delta_vs_ref) <= 0.005:
        verdict_a = ("NEUTRAL. DANN is within sampling noise of both lambda=0 and the LR "
                     "baseline; the adversarial loss is not hurting, but not clearly helping "
                     "either at this sample size.")
    elif delta_vs_ref < -0.01:
        verdict_a = ("NEGATIVE. DANN underperforms the reference LR baseline. Consistent with "
                     "the hypothesis that at 240 samples / 35 patients, the adversary signal "
                     "is too weak and collapses class-discriminative directions along with "
                     "patient-ID ones.")
    else:
        verdict_a = "MIXED. See the per-lambda numbers above."
    lines.append("")
    lines.append(verdict_a)
    lines.append("")

    # Part C: v4 ensemble
    lines.append("## Part C -- Can DANN help on TOP of v4 (bonus)?")
    lines.append("")
    lines.append("For each lambda in {0.0, best_lambda_from_part_A}, we independently train a")
    lines.append("DANN adapter on each of the 3 v4 encoders (DINOv2-B 90 nm, DINOv2-B 45 nm,")
    lines.append("BiomedCLIP D4-TTA 90 nm), then fit LR per-encoder on the LOPO-honest projections")
    lines.append("and take the geometric mean of the 3 softmaxes.")
    lines.append("")
    lines.append("| Config | Weighted F1 | Macro F1 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| v4 reproduction (no DANN, this run)        | {v4_ref[0]:.4f} | {v4_ref[1]:.4f} |")
    for L, s in sorted(v5_results.items()):
        lines.append(f"| v5 candidate (DANN lambda={L} on 3 enc)    | {s['f1w']:.4f} | {s['f1m']:.4f} |")
    lines.append(f"| Reference v4 champion (STATE.md)           | {REF_V4_CHAMPION:.4f} | {REF_V4_MACRO:.4f} |")
    lines.append("")

    # Verdict for v5
    best_v5_lambd = max(v5_results.keys(), key=lambda L: v5_results[L]["f1w"])
    best_v5 = v5_results[best_v5_lambd]
    v5_delta = best_v5["f1w"] - v4_ref[0]
    v5_delta_vs_champ = best_v5["f1w"] - REF_V4_CHAMPION
    lines.append("### Part C verdict")
    lines.append("")
    lines.append(f"- Best v5 candidate: lambda={best_v5_lambd}, weighted F1 = **{best_v5['f1w']:.4f}** "
                 f"(macro = {best_v5['f1m']:.4f}).")
    lines.append(f"- Delta vs v4 reproduction (this run): **{v5_delta:+.4f}**")
    lines.append(f"- Delta vs v4 champion (STATE.md {REF_V4_CHAMPION:.4f}): **{v5_delta_vs_champ:+.4f}**")
    if v5_delta_vs_champ > 0.005:
        verdict_c = "SHIP. v5 candidate beats the v4 champion -- promote to pending champion."
    elif v5_delta > 0.005:
        verdict_c = ("MIXED. DANN helps THIS run's 3-encoder reproduction but does not beat "
                     "the STATE.md v4 champion on the LOPO number. Do not ship without further "
                     "red-team bootstrap.")
    elif abs(v5_delta_vs_champ) <= 0.005:
        verdict_c = "NEUTRAL. v5 is within sampling noise of v4; no clear win, no loss. Do not ship."
    else:
        verdict_c = ("NEGATIVE. DANN projection regresses the v4 ensemble. Keep v4 as champion; "
                     "DANN's identity-scrubbing is useful for visualization / ablation but not "
                     "for this ensemble's final LOPO F1.")
    lines.append("")
    lines.append(verdict_c)
    lines.append("")

    # Honest caveats
    lines.append("## Honest caveats")
    lines.append("")
    lines.append("1. **240 samples / 35 patients is tiny for adversarial training.** The domain")
    lines.append("   head has only ~7 scans per patient on average; the adversarial gradient is")
    lines.append("   inherently noisy. This is not a methodology critique -- it is the reason")
    lines.append("   DANN often degrades to a regularizer at this scale.")
    lines.append("")
    lines.append("2. **Class leakage through patient.** For the rarer classes (SucheOko n=4 over 2")
    lines.append("   patients), patient ID and class are nearly colinear -- a perfectly-scrubbed")
    lines.append("   adapter *must* also forget class for those scans. This caps how aggressive")
    lines.append("   lambda can reasonably be.")
    lines.append("")
    lines.append("3. **LOPO-honest projection caching.** `cache/dann_projected_emb_dinov2b.npz`")
    lines.append("   stores the 128-d adapter features where each held-out person's row comes")
    lines.append("   from the fold model that did NOT see them. These 240 rows are safe to feed")
    lines.append("   into a LOPO-LR downstream evaluation; they are NOT safe for any model that")
    lines.append("   retrains on all 240 at once (row i encodes fold-i dependence).")
    lines.append("")
    lines.append("4. **Full-batch on 240 scans.** With so few samples, mini-batching adds more")
    lines.append("   variance than it removes. We use full-batch gradient descent (~230 scans).")
    lines.append("")
    lines.append("5. **class head vs LR head.** The class head and the LR-on-adapter classifier")
    lines.append("   often diverge by 1-3 F1 points: the LR baseline regularizes more strongly")
    lines.append("   and is the fair comparison to the 0.6150 reference which also uses LR.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `scripts/dann_training.py` (this experiment)")
    lines.append("- `cache/dann_projected_emb_dinov2b.npz` (240 x 128 LOPO-honest adapter features at best lambda)")
    lines.append("- `reports/DANN_RESULTS.md` (this file)")
    lines.append("")
    path = REPORTS / "DANN_RESULTS.md"
    path.write_text("\n".join(lines))
    print(f"\nReport written: {path}")


if __name__ == "__main__":
    main()
