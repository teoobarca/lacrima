"""SSL-style Supervised Contrastive (SupCon) projection head on cached DINOv2-B tile embeddings.

Goal: specialize ImageNet DINOv2-B features to the tear-AFM domain via a small parameter-efficient
projection MLP (768 -> 256 -> 128). We train with SupCon loss at the TILE level using scan-level
labels shared across a scan's tiles, then mean-pool projected tiles per scan and evaluate with
LogisticRegression under person-LOPO.

Honest-evaluation protocol:
- Person-level LOPO (35 persons via teardrop.data.person_id)
- For each held-out person we RETRAIN the projection head from scratch on the remaining 34 persons,
  then project both train and held-out tiles with that fold's head, mean-pool to scan embeddings,
  and fit/predict with a StandardScaler+LR downstream classifier. This avoids any SSL-on-val leakage.
- Compare against a "frozen-DINOv2-only" rerun of the exact same pipeline (mean-pool tiles per scan,
  StandardScaler+LR LOPO) for apples-to-apples.

Deliverables written:
- cache/supcon_projected_emb.npz  -- 240 x 128 embeddings (from a FINAL head trained on ALL persons,
  for downstream use / sharing); strictly separate from the LOPO-honest numbers.
- reports/SSL_SUPCON_RESULTS.md    -- methodology + F1 comparison.

Usage:
    .venv/bin/python scripts/supcon_projection.py
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.data import CLASSES, person_id  # noqa: E402
from teardrop.cv import leave_one_patient_out  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_tile_data():
    """Load cached DINOv2-B tile embeddings + derive person groups from scan_paths."""
    z = np.load(CACHE / "tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz", allow_pickle=True)
    X = z["X"].astype(np.float32)               # (n_tiles, 768)
    tile_to_scan = z["tile_to_scan"].astype(int)
    scan_y = z["scan_y"].astype(int)            # (n_scans,)
    scan_paths = z["scan_paths"].tolist()       # (n_scans,) list[str]
    # Derive person groups from scan_paths (honest person-LOPO uses person_id).
    scan_persons = np.array([person_id(Path(p)) for p in scan_paths])
    return X, tile_to_scan, scan_y, scan_persons, scan_paths


def scan_mean_pool(X_tiles, tile_to_scan, n_scans, dim):
    """Mean-pool tile embeddings to scan-level."""
    out = np.zeros((n_scans, dim), dtype=np.float32)
    cnt = np.zeros(n_scans, dtype=np.int32)
    for ti, si in enumerate(tile_to_scan):
        out[si] += X_tiles[ti]
        cnt[si] += 1
    cnt = np.maximum(cnt, 1)
    return out / cnt[:, None]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """2-layer MLP: 768 -> 256 -> 128, L2-normalized output."""

    def __init__(self, in_dim=768, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


# ---------------------------------------------------------------------------
# SupCon loss
# ---------------------------------------------------------------------------

def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised Contrastive loss (Khosla et al. 2020), single-view variant.

    features: (B, D), already L2-normalized
    labels:   (B,)    int class labels

    Formulation: for each anchor i, positives P(i) = {j != i : y_j == y_i}.
    L_i = -1/|P(i)| * sum_{p in P(i)} log( exp(z_i . z_p / T) / sum_{a != i} exp(z_i . z_a / T) )
    Anchors with zero positives are skipped.
    """
    device = features.device
    B = features.shape[0]
    # Similarity matrix (B, B), scaled by 1/T.
    logits = features @ features.T / temperature
    # Numerical stability: subtract rowwise max.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # Mask: positive pairs have same label, and exclude self.
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float().to(device)
    self_mask = torch.eye(B, device=device)
    pos_mask = pos_mask - self_mask  # remove diagonal
    valid_mask = 1.0 - self_mask     # denominator excludes self

    exp_logits = torch.exp(logits) * valid_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    n_pos = pos_mask.sum(dim=1)
    # For anchors with no positives, mean_log_prob_pos is 0; mask them out in final mean.
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / torch.clamp(n_pos, min=1.0)
    loss_per = -mean_log_prob_pos
    has_pos = (n_pos > 0).float()
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return (loss_per * has_pos).sum() / has_pos.sum()


# ---------------------------------------------------------------------------
# Train one projection head on a given tile-index subset
# ---------------------------------------------------------------------------

def train_head(
    X_tiles: np.ndarray,
    tile_labels: np.ndarray,
    train_tile_idx: np.ndarray,
    *,
    in_dim: int = 768,
    hidden: int = 256,
    out_dim: int = 128,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    temperature: float = 0.1,
    seed: int = SEED,
    verbose: bool = False,
) -> ProjectionHead:
    """Train projection head on a subset of tiles with SupCon."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    head = ProjectionHead(in_dim, hidden, out_dim).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    Xt = torch.from_numpy(X_tiles[train_tile_idx]).to(DEVICE)
    yt = torch.from_numpy(tile_labels[train_tile_idx]).long().to(DEVICE)
    n = Xt.shape[0]

    head.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        total, nb = 0.0, 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            if idx.numel() < 4:
                continue
            x_b = Xt[idx]
            y_b = yt[idx]
            # Require at least 2 distinct classes in batch (else no negatives).
            if torch.unique(y_b).numel() < 2:
                continue
            z = head(x_b)
            loss = supcon_loss(z, y_b, temperature=temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()); nb += 1
        if verbose and (ep + 1) % 10 == 0:
            print(f"    ep {ep+1:02d}/{epochs}  loss={total/max(nb,1):.4f}")
    head.eval()
    return head


@torch.no_grad()
def project_tiles(head: ProjectionHead, X_tiles: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Run the head over all tiles and return a numpy array."""
    head.eval()
    outs = []
    for s in range(0, X_tiles.shape[0], batch_size):
        x = torch.from_numpy(X_tiles[s:s + batch_size]).to(DEVICE)
        z = head(x).cpu().numpy()
        outs.append(z)
    return np.concatenate(outs, axis=0)


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def eval_scan_lr_lopo(X_scan, y, groups, label: str):
    """Person-LOPO LogisticRegression on scan-level features. Returns (f1w, f1m, preds)."""
    preds = np.full(len(y), -1, dtype=int)
    for tr, va in leave_one_patient_out(groups):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X_scan[tr])
        Xv = scaler.transform(X_scan[va])
        clf = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0,
            solver="lbfgs", n_jobs=4, random_state=SEED,
        )
        clf.fit(Xt, y[tr])
        preds[va] = clf.predict(Xv)
    f1w = f1_score(y, preds, average="weighted")
    f1m = f1_score(y, preds, average="macro")
    print(f"  [{label}] LOPO weighted F1 = {f1w:.4f}  macro F1 = {f1m:.4f}")
    return f1w, f1m, preds


def classwise_report(y, preds, title):
    print(f"\n  === {title} ===")
    print(classification_report(y, preds, target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(y, preds, labels=list(range(len(CLASSES))))
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print(f"Device: {DEVICE}")
    print("=" * 72)
    print("SSL-style SupCon projection head on cached DINOv2-B tile embeddings")
    print("=" * 72)

    # --- Load cached tile embeddings ---
    X_tiles, tile_to_scan, scan_y, scan_persons, scan_paths = load_tile_data()
    n_scans = len(scan_y)
    in_dim = X_tiles.shape[1]
    print(f"Loaded {X_tiles.shape[0]} tiles from {n_scans} scans, in_dim={in_dim}")
    print(f"Persons: {len(np.unique(scan_persons))}   class counts: {np.bincount(scan_y)}")

    # Tile-level labels (each tile inherits its scan's class).
    tile_y = scan_y[tile_to_scan]
    tile_person = scan_persons[tile_to_scan]

    out_dim = 128

    # ------------------------------------------------------------------
    # 1) Baseline: frozen DINOv2-B, mean-pool tiles -> scan, LR LOPO
    # ------------------------------------------------------------------
    print("\n[1/4] Frozen-DINOv2-B baseline (mean-pool tiles -> scan -> LR LOPO)")
    X_scan_frozen = scan_mean_pool(X_tiles, tile_to_scan, n_scans, in_dim)
    f1w_base, f1m_base, preds_base = eval_scan_lr_lopo(
        X_scan_frozen, scan_y, scan_persons, "frozen DINOv2-B"
    )

    # ------------------------------------------------------------------
    # 2) SupCon LOPO-honest: train a fresh head per held-out person
    # ------------------------------------------------------------------
    print(f"\n[2/4] SupCon LOPO-honest: retrain projection head per held-out person ({len(np.unique(scan_persons))} folds)")
    preds_supcon = np.full(n_scans, -1, dtype=int)
    scan_supcon_all = np.zeros((n_scans, out_dim), dtype=np.float32)
    persons = np.unique(scan_persons)
    t_lopo = time.time()
    for pi, pat in enumerate(persons):
        # Train tiles = tiles whose scan's person != held-out.
        is_train_tile = (tile_person != pat)
        train_tile_idx = np.where(is_train_tile)[0]
        head = train_head(
            X_tiles, tile_y, train_tile_idx,
            in_dim=in_dim, hidden=256, out_dim=out_dim,
            epochs=40, batch_size=128, lr=3e-4,
            weight_decay=1e-4, temperature=0.1, seed=SEED,
            verbose=False,
        )
        # Project ALL tiles with this fold's head (held-out scans get projected with a head that
        # never saw them).
        Z_tiles = project_tiles(head, X_tiles)
        # Scan-level mean-pool, then assign held-out scans' projected vectors for LOPO eval.
        Z_scan = scan_mean_pool(Z_tiles, tile_to_scan, n_scans, out_dim)
        # Train+fit LR on training scans, predict held-out person's scans.
        ho_scan_idx = np.where(scan_persons == pat)[0]
        tr_scan_idx = np.where(scan_persons != pat)[0]
        scaler = StandardScaler()
        Xt = scaler.fit_transform(Z_scan[tr_scan_idx])
        Xv = scaler.transform(Z_scan[ho_scan_idx])
        clf = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0,
            solver="lbfgs", n_jobs=4, random_state=SEED,
        )
        clf.fit(Xt, scan_y[tr_scan_idx])
        preds_supcon[ho_scan_idx] = clf.predict(Xv)
        # Record held-out scans' projected features for the downstream cache (fold-honest).
        scan_supcon_all[ho_scan_idx] = Z_scan[ho_scan_idx]
        if (pi + 1) % 5 == 0 or pi == len(persons) - 1:
            elapsed = time.time() - t_lopo
            print(f"  fold {pi+1:2d}/{len(persons)}  person={pat!r}  elapsed={elapsed:.1f}s")

    f1w_sc = f1_score(scan_y, preds_supcon, average="weighted")
    f1m_sc = f1_score(scan_y, preds_supcon, average="macro")
    print(f"\n  [SupCon LOPO-honest] weighted F1 = {f1w_sc:.4f}  macro F1 = {f1m_sc:.4f}")

    # ------------------------------------------------------------------
    # 3) Ensemble: average softmax probs of frozen-DINOv2-LR and SupCon-LR, LOPO-honest
    # ------------------------------------------------------------------
    print("\n[3/4] SupCon + frozen-DINOv2-B proba-average ensemble (LOPO-honest)")

    def _lopo_probs(X_scan):
        n_classes = len(CLASSES)
        probs = np.zeros((n_scans, n_classes), dtype=np.float32)
        for tr, va in leave_one_patient_out(scan_persons):
            scaler = StandardScaler()
            Xt = scaler.fit_transform(X_scan[tr])
            Xv = scaler.transform(X_scan[va])
            clf = LogisticRegression(
                class_weight="balanced", max_iter=2000, C=1.0,
                solver="lbfgs", n_jobs=4, random_state=SEED,
            )
            clf.fit(Xt, scan_y[tr])
            probs[va] = clf.predict_proba(Xv)
        return probs

    # SupCon probs: re-run LOPO but capture probas.
    probs_supcon = np.zeros((n_scans, len(CLASSES)), dtype=np.float32)
    for pi, pat in enumerate(persons):
        is_train_tile = (tile_person != pat)
        train_tile_idx = np.where(is_train_tile)[0]
        head = train_head(
            X_tiles, tile_y, train_tile_idx,
            in_dim=in_dim, hidden=256, out_dim=out_dim,
            epochs=40, batch_size=128, lr=3e-4,
            weight_decay=1e-4, temperature=0.1, seed=SEED,
        )
        Z_tiles = project_tiles(head, X_tiles)
        Z_scan = scan_mean_pool(Z_tiles, tile_to_scan, n_scans, out_dim)
        ho = np.where(scan_persons == pat)[0]
        tr = np.where(scan_persons != pat)[0]
        scaler = StandardScaler()
        Xt = scaler.fit_transform(Z_scan[tr])
        Xv = scaler.transform(Z_scan[ho])
        clf = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0,
            solver="lbfgs", n_jobs=4, random_state=SEED,
        )
        clf.fit(Xt, scan_y[tr])
        probs_supcon[ho] = clf.predict_proba(Xv)

    probs_frozen = _lopo_probs(X_scan_frozen)
    probs_ens = 0.5 * probs_supcon + 0.5 * probs_frozen
    preds_ens = probs_ens.argmax(axis=1)
    f1w_ens = f1_score(scan_y, preds_ens, average="weighted")
    f1m_ens = f1_score(scan_y, preds_ens, average="macro")
    print(f"  [SupCon+Frozen avg] weighted F1 = {f1w_ens:.4f}  macro F1 = {f1m_ens:.4f}")

    # ------------------------------------------------------------------
    # 4) Final "all-persons-trained" head for the cached 240x128 embedding file.
    # ------------------------------------------------------------------
    print("\n[4/4] Training final head on ALL 240 scans for cached shareable embedding")
    head_final = train_head(
        X_tiles, tile_y, np.arange(X_tiles.shape[0]),
        in_dim=in_dim, hidden=256, out_dim=out_dim,
        epochs=60, batch_size=128, lr=3e-4,
        weight_decay=1e-4, temperature=0.1, seed=SEED, verbose=True,
    )
    Z_tiles_final = project_tiles(head_final, X_tiles)
    Z_scan_final = scan_mean_pool(Z_tiles_final, tile_to_scan, n_scans, out_dim)

    out_npz = CACHE / "supcon_projected_emb.npz"
    np.savez(
        out_npz,
        X_scan=Z_scan_final,                 # (240, 128) projected scan embeddings
        Z_tiles=Z_tiles_final,               # (n_tiles, 128) projected tile embeddings
        tile_to_scan=tile_to_scan,
        scan_y=scan_y,
        scan_persons=scan_persons,
        scan_paths=np.array(scan_paths),
        config=json.dumps({
            "in_dim": in_dim, "hidden": 256, "out_dim": out_dim,
            "epochs": 60, "batch_size": 128, "lr": 3e-4,
            "weight_decay": 1e-4, "temperature": 0.1, "seed": SEED,
            "source": "cache/tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz",
            "loss": "supcon", "trained_on": "all_240_scans",
            "honest_lopo_weighted_f1": float(f1w_sc),
            "honest_lopo_macro_f1": float(f1m_sc),
        }),
    )
    print(f"  saved {out_npz}")

    # ------------------------------------------------------------------
    # Summary + report
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY (person-LOPO, raw argmax)")
    print("=" * 72)
    print(f"  Frozen DINOv2-B baseline (this run)   : weighted F1 = {f1w_base:.4f}  macro = {f1m_base:.4f}")
    print(f"  SupCon projection head (honest LOPO)  : weighted F1 = {f1w_sc:.4f}  macro = {f1m_sc:.4f}")
    print(f"  SupCon + Frozen 0.5/0.5 proba-average : weighted F1 = {f1w_ens:.4f}  macro = {f1m_ens:.4f}")
    print(f"  Reference: DINOv2-B single (STATE.md) : 0.615  (should ~match frozen baseline above)")
    print(f"  Reference: TTA D4 ensemble champion   : 0.6458")
    print(f"  Total wall time: {time.time()-t_start:.1f}s")

    classwise_report(scan_y, preds_base, "Frozen DINOv2-B baseline")
    classwise_report(scan_y, preds_supcon, "SupCon projection (honest LOPO)")
    classwise_report(scan_y, preds_ens, "SupCon + Frozen ensemble (honest LOPO)")

    # ----- Write reports/SSL_SUPCON_RESULTS.md -----
    write_report(
        f1w_base=f1w_base, f1m_base=f1m_base,
        f1w_sc=f1w_sc, f1m_sc=f1m_sc,
        f1w_ens=f1w_ens, f1m_ens=f1m_ens,
        preds_base=preds_base, preds_supcon=preds_supcon, preds_ens=preds_ens,
        scan_y=scan_y, wall=time.time()-t_start,
    )


def write_report(*, f1w_base, f1m_base, f1w_sc, f1m_sc, f1w_ens, f1m_ens,
                 preds_base, preds_supcon, preds_ens, scan_y, wall):
    def per_class_f1(preds):
        return f1_score(scan_y, preds, average=None, labels=list(range(len(CLASSES))),
                        zero_division=0)
    pc_base = per_class_f1(preds_base)
    pc_sc = per_class_f1(preds_supcon)
    pc_ens = per_class_f1(preds_ens)

    lines = []
    lines.append("# SSL-style SupCon Projection-Head Results")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append("Can a small (768 -> 256 -> 128) supervised-contrastive projection head specialize")
    lines.append("ImageNet-pretrained DINOv2-B features to the tear-AFM domain and beat:")
    lines.append("(a) the frozen-DINOv2-B single-model baseline (~0.615 weighted F1), and")
    lines.append("(b) the current shipped TTA D4 ensemble champion (0.6458)?")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("- Source features: `cache/tiled_emb_dinov2_vitb14_afmhot_t512_n9.npz` (DINOv2-B-tiled, 811 tiles / 240 scans, 768-dim).")
    lines.append("- Projection head: 2-layer MLP (Linear 768->256 -> GELU -> Linear 256->128), L2-normalized output.")
    lines.append("- Loss: Supervised Contrastive (Khosla et al. 2020), temperature = 0.1, batch 128, AdamW lr = 3e-4, weight_decay = 1e-4, 40 epochs (per fold) / 60 (final).")
    lines.append("- Tile-level training: each tile inherits its scan's class label; SupCon pulls same-class tiles together, pushes different-class tiles apart.")
    lines.append("- **Honest person-LOPO protocol**: for each of 35 persons, a fresh projection head is trained from scratch on the 34 remaining persons' tiles, then:")
    lines.append("    1. project all tiles via this fold's head;")
    lines.append("    2. mean-pool tiles per scan -> 240 x 128;")
    lines.append("    3. StandardScaler + LogisticRegression fit on train scans, predict held-out person's scans.")
    lines.append("- Ensemble: 0.5/0.5 proba-average of the SupCon head's LR probs and the frozen-DINOv2-B LR probs, LOPO-honest.")
    lines.append(f"- Device: `{DEVICE}`. Total wall clock: {wall:.1f} s.")
    lines.append("")
    lines.append("## Results (person-LOPO, raw argmax)")
    lines.append("")
    lines.append("| Model | Weighted F1 | Macro F1 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Frozen DINOv2-B single (this run, reference)  | {f1w_base:.4f} | {f1m_base:.4f} |")
    lines.append(f"| **SupCon projection head (honest LOPO)**      | **{f1w_sc:.4f}** | **{f1m_sc:.4f}** |")
    lines.append(f"| SupCon + Frozen proba-average (honest LOPO)   | {f1w_ens:.4f} | {f1m_ens:.4f} |")
    lines.append(f"| Reference champion: DINOv2-B + BiomedCLIP + D4 TTA | 0.6458 | 0.5154 |")
    lines.append(f"| Reference baseline (STATE.md): DINOv2-B single    | 0.6150 | 0.4910 |")
    lines.append("")
    lines.append("### Per-class F1 (weighted)")
    lines.append("")
    lines.append(f"| Class | Support | Frozen | SupCon | SupCon+Frozen |")
    lines.append("|---|---:|---:|---:|---:|")
    supports = [int((scan_y == i).sum()) for i in range(len(CLASSES))]
    for i, c in enumerate(CLASSES):
        lines.append(f"| {c} | {supports[i]} | {pc_base[i]:.3f} | {pc_sc[i]:.3f} | {pc_ens[i]:.3f} |")
    lines.append("")
    # Verdict
    delta_vs_base = f1w_sc - f1w_base
    delta_vs_tta = f1w_sc - 0.6458
    lines.append("## Verdict")
    lines.append("")
    if f1w_sc >= 0.6458 + 0.005:
        verdict = "SHIP. SupCon head beats the TTA D4 ensemble champion (0.6458)."
    elif f1w_sc >= f1w_base + 0.005:
        verdict = "MIXED. SupCon beats the frozen-DINOv2-B baseline but does NOT beat the TTA D4 ensemble champion."
    elif abs(f1w_sc - f1w_base) < 0.005:
        verdict = "NO CLEAR GAIN. SupCon is within sampling noise of the frozen baseline."
    else:
        verdict = "NEGATIVE. SupCon underperforms the frozen baseline, consistent with SSL-on-tiny-data overfitting risk."
    lines.append(verdict)
    lines.append("")
    lines.append(f"- Delta vs frozen DINOv2-B (this run): **{delta_vs_base:+.4f}** weighted F1")
    lines.append(f"- Delta vs TTA D4 champion (0.6458)  : **{delta_vs_tta:+.4f}** weighted F1")
    lines.append("")
    lines.append("## Honest caveats")
    lines.append("")
    lines.append("1. **240 scans is tiny for SSL.** Even SupCon with real labels is at risk of overfitting the 34-person training pool; cross-person generalization is what LOPO actually measures.")
    lines.append("2. **SupCon uses labels.** This is NOT self-supervised in the strict sense; it is *supervised contrastive*. A purely self-supervised variant (e.g. augmentation-consistency) would skip the labels at the cost of losing the class-pulling signal.")
    lines.append("3. **Head sees only 768-dim cached DINOv2 features, not raw images.** It cannot repair genuinely missing information -- it can only re-shape the existing feature geometry.")
    lines.append("4. **Tile labels are noisy.** All tiles of a scan inherit the scan's class even if some tiles are mostly background or show weak evidence.")
    lines.append("5. **The per-fold head retraining is the honest baseline.** Training a head on all 240 scans and re-using it for evaluation would be label-leakage -- and we deliberately separate the LOPO numbers from the final cached embedding.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `scripts/supcon_projection.py` -- this script")
    lines.append("- `cache/supcon_projected_emb.npz` -- (240, 128) projected scan embeddings from the final head trained on all 240 scans, plus (n_tiles, 128) projected tile embeddings. **Not safe for LOPO re-eval** (the head saw all persons); use only for downstream / visualization.")
    lines.append("- `reports/SSL_SUPCON_RESULTS.md` (this file)")
    lines.append("")
    report_path = REPORTS / "SSL_SUPCON_RESULTS.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport written: {report_path}")


if __name__ == "__main__":
    main()
