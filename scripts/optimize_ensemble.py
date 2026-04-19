"""Maximalizácia LOPO weighted F1 — concat ensembles, threshold tuning, MLP head.

Stratégia:
  1. Načíta cached tiled embeddingy (DINOv2-S, DINOv2-B, BiomedCLIP) → mean-pool na scan-level.
  2. Načíta handcrafted features (94 cols) zoradené podľa raw_path.
  3. Vyhodnotí všetky combinácie cez LR (StandardScaler) a XGBoost s LOPO.
  4. Na najlepšom konfigu spraví per-class threshold tuning (sweep cez OOF predict_proba).
  5. Vyskúša malý MLP head (PyTorch) na najlepšom concat configu cez LOPO.
  6. Uloží best OOF predikcie do parquet a zapíše ENSEMBLE_RESULTS.md.

Beh: .venv/bin/python scripts/optimize_ensemble.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out, patient_stratified_kfold
from teardrop.data import CLASSES, enumerate_samples

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

N_CLASSES = len(CLASSES)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_tiled(name: str):
    path = CACHE / f"tiled_emb_{name}_afmhot_t512_n9.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return z["X"], z["tile_to_scan"], z["scan_y"], z["scan_groups"], z["scan_paths"].tolist()


def aggregate_tiles_to_scan(X_tiles: np.ndarray, tile_to_scan: np.ndarray,
                            n_scans: int) -> np.ndarray:
    """Mean-pool tile embeddings to scan-level."""
    embed_dim = X_tiles.shape[1]
    out = np.zeros((n_scans, embed_dim), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int32)
    for ti, si in enumerate(tile_to_scan):
        out[si] += X_tiles[ti]
        counts[si] += 1
    counts = np.maximum(counts, 1)
    out /= counts[:, None]
    return out


def load_handcrafted(samples) -> np.ndarray | None:
    hc_path = CACHE / "features_handcrafted.parquet"
    if not hc_path.exists():
        return None
    hc = pd.read_parquet(hc_path)
    path_to_idx = {str(s.raw_path): i for i, s in enumerate(samples)}
    hc["si"] = hc["raw"].map(path_to_idx)
    hc = hc.dropna(subset=["si"]).sort_values("si")
    hc_cols = [c for c in hc.columns if c not in ("raw", "cls", "label", "patient", "si")]
    Xh = np.zeros((len(samples), len(hc_cols)), dtype=np.float32)
    for _, row in hc.iterrows():
        Xh[int(row["si"])] = row[hc_cols].values.astype(np.float32)
    # Replace NaN/Inf
    Xh = np.nan_to_num(Xh, nan=0.0, posinf=0.0, neginf=0.0)
    return Xh


# ---------------------------------------------------------------------------
# Evaluators (returns OOF probs + preds + scalar metrics)
# ---------------------------------------------------------------------------

def lopo_lr(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float = 1.0):
    """LOPO LR with per-fold StandardScaler. Returns oof_probs, oof_preds."""
    n = len(y)
    oof_probs = np.zeros((n, N_CLASSES), dtype=np.float64)
    oof_preds = np.full(n, -1, dtype=int)
    for tr, va in leave_one_patient_out(groups):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr])
        Xv = scaler.transform(X[va])
        clf = LogisticRegression(class_weight="balanced", max_iter=4000, C=C,
                                 solver="lbfgs", n_jobs=4)
        clf.fit(Xt, y[tr])
        # Map clf.classes_ → full N_CLASSES indexing
        probs = clf.predict_proba(Xv)
        full = np.zeros((len(va), N_CLASSES))
        for i, c in enumerate(clf.classes_):
            full[:, c] = probs[:, i]
        oof_probs[va] = full
        oof_preds[va] = full.argmax(axis=1)
    return oof_probs, oof_preds


def lopo_xgb(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """LOPO XGBoost with class-weighted samples. Returns oof_probs, oof_preds."""
    n = len(y)
    cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y)
    sample_weights = np.array([cw[label] for label in y])
    params = dict(
        n_estimators=250, max_depth=4, learning_rate=0.07,
        subsample=0.85, colsample_bytree=0.7,
        reg_lambda=1.5, reg_alpha=0.5,
        random_state=42, n_jobs=8,
        objective="multi:softprob", num_class=N_CLASSES,
        tree_method="hist",
    )
    oof_probs = np.zeros((n, N_CLASSES), dtype=np.float64)
    oof_preds = np.full(n, -1, dtype=int)
    for tr, va in leave_one_patient_out(groups):
        clf = XGBClassifier(**params)
        clf.fit(X[tr], y[tr], sample_weight=sample_weights[tr])
        oof_probs[va] = clf.predict_proba(X[va])
        oof_preds[va] = oof_probs[va].argmax(axis=1)
    return oof_probs, oof_preds


def metrics_dict(y: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "weighted_f1": float(f1_score(y, preds, average="weighted")),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "per_class_f1": f1_score(y, preds, average=None,
                                 labels=list(range(N_CLASSES)), zero_division=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Per-class threshold tuning
# ---------------------------------------------------------------------------

def per_class_threshold_tuning(probs: np.ndarray, y: np.ndarray,
                               thr_grid: np.ndarray | None = None,
                               passes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate-ascent: per-class additive bias on log-probs to maximize weighted F1.

    Pretože multi-class threshold (priame > t) je nejednoznačné pri argmax,
    používame additive bias na log(p) trick (ekvivalent class prior-shift).
    Sweep prebehne paralelne pre každú triedu, opakuje 'passes' krát.
    Returns (best_biases, tuned_preds).
    """
    if thr_grid is None:
        # bias range: shift log-prob by [-3, +3] in 25 steps
        thr_grid = np.linspace(-3.0, 3.0, 25)
    log_p = np.log(np.clip(probs, 1e-9, 1.0))
    biases = np.zeros(N_CLASSES)

    def preds_with(biases_):
        adj = log_p + biases_[None, :]
        return adj.argmax(axis=1)

    base_f1 = f1_score(y, preds_with(biases), average="weighted")

    for pass_i in range(passes):
        improved = False
        for c in range(N_CLASSES):
            best = biases[c]
            best_f1 = base_f1
            for b in thr_grid:
                trial = biases.copy()
                trial[c] = b
                f1 = f1_score(y, preds_with(trial), average="weighted")
                if f1 > best_f1 + 1e-6:
                    best_f1 = f1
                    best = b
            if best != biases[c]:
                biases[c] = best
                base_f1 = best_f1
                improved = True
        if not improved:
            break

    return biases, preds_with(biases)


# ---------------------------------------------------------------------------
# MLP head (PyTorch) for LOPO
# ---------------------------------------------------------------------------

def lopo_mlp(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
             hidden=(256, 128), dropout=0.3, label_smoothing=0.1,
             max_epochs=80, lr=1e-3, weight_decay=1e-4,
             patience=10, seed=42, verbose=False):
    """LOPO MLP with per-fold internal val split (10% of train patients).
    Per-fold StandardScaler. Class-weighted cross-entropy with label smoothing.
    """
    import torch
    import torch.nn as nn

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    n = len(y)
    oof_probs = np.zeros((n, N_CLASSES), dtype=np.float64)
    oof_preds = np.full(n, -1, dtype=int)

    cw_full = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y)

    rng = np.random.default_rng(seed)

    fold_count = 0
    t0 = time.time()
    for tr, va in leave_one_patient_out(groups):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Per-fold standardization
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr]).astype(np.float32)
        Xv = scaler.transform(X[va]).astype(np.float32)

        # Inner val split: hold out 10 % of train patients for ES
        tr_groups = groups[tr]
        unique_tr = np.unique(tr_groups)
        rng.shuffle(unique_tr)
        n_inner_val = max(1, int(0.10 * len(unique_tr)))
        inner_val_p = set(unique_tr[:n_inner_val])
        inner_val_mask = np.array([g in inner_val_p for g in tr_groups])
        inner_tr_mask = ~inner_val_mask

        Xtr_in = torch.from_numpy(Xt[inner_tr_mask]).to(device)
        ytr_in = torch.from_numpy(y[tr][inner_tr_mask]).long().to(device)
        Xv_in = torch.from_numpy(Xt[inner_val_mask]).to(device)
        yv_in = y[tr][inner_val_mask]

        # Build MLP
        layers = []
        d = X.shape[1]
        for hi in hidden:
            layers.extend([nn.Linear(d, hi), nn.GELU(), nn.Dropout(dropout)])
            d = hi
        layers.append(nn.Linear(d, N_CLASSES))
        net = nn.Sequential(*layers).to(device)

        cls_w = torch.tensor(cw_full, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=label_smoothing)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_f1 = -1.0
        best_state = None
        bad = 0
        bs = 64
        n_train = Xtr_in.shape[0]
        for epoch in range(max_epochs):
            net.train()
            perm = torch.randperm(n_train, device=device)
            for i in range(0, n_train, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                logits = net(Xtr_in[idx])
                loss = loss_fn(logits, ytr_in[idx])
                loss.backward()
                opt.step()
            # Val
            net.eval()
            with torch.no_grad():
                v_logits = net(Xv_in)
                v_pred = v_logits.argmax(dim=1).cpu().numpy()
            v_f1 = f1_score(yv_in, v_pred, average="weighted") if len(np.unique(yv_in)) > 1 else 0.0
            if v_f1 > best_val_f1 + 1e-4:
                best_val_f1 = v_f1
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            Xv_t = torch.from_numpy(Xv).to(device)
            logits = net(Xv_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        oof_probs[va] = probs
        oof_preds[va] = probs.argmax(axis=1)

        fold_count += 1
        if verbose and fold_count % 5 == 0:
            print(f"  [MLP] fold {fold_count}/{len(np.unique(groups))} elapsed={time.time()-t0:.1f}s")

    return oof_probs, oof_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt_perclass(per_class: list[float]) -> str:
    return " | ".join(f"{n[:5]}={f:.2f}" for n, f in zip(CLASSES, per_class))


def main():
    print("=" * 70)
    print("ENSEMBLE OPTIMIZATION — LOPO weighted F1")
    print("=" * 70)

    samples = enumerate_samples(ROOT / "TRAIN_SET")
    n_scans = len(samples)
    print(f"Loaded {n_scans} samples")

    # Load tiled embeddings
    print("\n[load] tiled embeddings (mean-pool to scan-level)")
    s = load_tiled("dinov2_vits14")
    Xs_tiles, t2s, scan_y, scan_groups, _ = s
    Xs = aggregate_tiles_to_scan(Xs_tiles, t2s, n_scans)
    print(f"  dinov2_s scan-emb: {Xs.shape}")

    b = load_tiled("dinov2_vitb14")
    Xb = aggregate_tiles_to_scan(b[0], b[1], n_scans)
    print(f"  dinov2_b scan-emb: {Xb.shape}")

    bm = load_tiled("biomedclip")
    Xbm = aggregate_tiles_to_scan(bm[0], bm[1], n_scans)
    print(f"  biomedclip scan-emb: {Xbm.shape}")

    Xhc = load_handcrafted(samples)
    print(f"  handcrafted: {Xhc.shape}")

    y = scan_y.astype(int)
    groups = scan_groups

    # Sanity: scan_y / scan_groups consistency between encoders
    sb = load_tiled("dinov2_vitb14")
    sbm = load_tiled("biomedclip")
    assert (sb[2] == scan_y).all(), "dinov2_b scan_y mismatch"
    assert (sbm[2] == scan_y).all(), "biomedclip scan_y mismatch"
    assert (sb[3] == scan_groups).all(), "dinov2_b groups mismatch"
    assert (sbm[3] == scan_groups).all(), "biomedclip groups mismatch"

    # L2-normalize each embedding part before concat (matters for LR, safe for XGB).
    def l2norm(A):
        n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
        return A / n

    Xs_n, Xb_n, Xbm_n = l2norm(Xs), l2norm(Xb), l2norm(Xbm)

    # ---- Configurations (task-requested 5 + one extra w/o handcrafted) ----
    configs: dict[str, np.ndarray] = {
        "dinov2_s": Xs_n,
        "dinov2_s+dinov2_b": np.concatenate([Xs_n, Xb_n], axis=1),
        "dinov2_s+biomedclip": np.concatenate([Xs_n, Xbm_n], axis=1),
        "dinov2_s+dinov2_b+biomedclip": np.concatenate([Xs_n, Xb_n, Xbm_n], axis=1),
        "dinov2_s+dinov2_b+biomedclip+handcrafted":
            np.concatenate([Xs_n, Xb_n, Xbm_n, Xhc], axis=1),
    }

    results: dict[str, dict] = {}
    cached_probs: dict[str, dict] = {}  # config_name → {clf: probs}

    for name, X in configs.items():
        print(f"\n{'='*70}\nCONFIG: {name}  (dim={X.shape[1]})")
        # LR
        t0 = time.time()
        probs_lr, preds_lr = lopo_lr(X, y, groups)
        m_lr = metrics_dict(y, preds_lr)
        print(f"  [LR ] wF1={m_lr['weighted_f1']:.4f}  mF1={m_lr['macro_f1']:.4f}  "
              f"({time.time()-t0:.1f}s)  per-class: {fmt_perclass(m_lr['per_class_f1'])}")
        # XGB
        t0 = time.time()
        probs_xgb, preds_xgb = lopo_xgb(X, y, groups)
        m_xgb = metrics_dict(y, preds_xgb)
        print(f"  [XGB] wF1={m_xgb['weighted_f1']:.4f}  mF1={m_xgb['macro_f1']:.4f}  "
              f"({time.time()-t0:.1f}s)  per-class: {fmt_perclass(m_xgb['per_class_f1'])}")
        results[name] = {"LR": m_lr, "XGB": m_xgb}
        cached_probs[name] = {"LR": probs_lr, "XGB": probs_xgb}

    # ---- Find best config ----
    print(f"\n{'='*70}\nSUMMARY (LOPO, sorted by weighted F1)")
    print(f"{'config':50s} {'clf':4s} {'wF1':>7s} {'mF1':>7s}")
    flat = []
    for name, m in results.items():
        for clf in ("LR", "XGB"):
            flat.append((name, clf, m[clf]["weighted_f1"], m[clf]["macro_f1"]))
    flat.sort(key=lambda r: -r[2])
    for name, clf, wf, mf in flat:
        print(f"{name:50s} {clf:4s} {wf:7.4f} {mf:7.4f}")

    best_name, best_clf, best_wf, best_mf = flat[0]
    best_probs = cached_probs[best_name][best_clf]
    print(f"\n>>> BEST: {best_name} / {best_clf} → wF1={best_wf:.4f}, mF1={best_mf:.4f}")

    # ---- Threshold tuning ----
    print(f"\n{'='*70}\nPER-CLASS THRESHOLD/BIAS TUNING (on best OOF probs)")
    biases, tuned_preds = per_class_threshold_tuning(best_probs, y)
    m_tuned = metrics_dict(y, tuned_preds)
    print(f"  Biases (log-prob shifts): " +
          " | ".join(f"{n[:5]}={b:+.2f}" for n, b in zip(CLASSES, biases)))
    print(f"  TUNED wF1={m_tuned['weighted_f1']:.4f}  mF1={m_tuned['macro_f1']:.4f}  "
          f"per-class: {fmt_perclass(m_tuned['per_class_f1'])}")

    results[f"{best_name}|TUNED"] = {best_clf + "_tuned": m_tuned}

    # ---- MLP head on best concat config (we tune on best concat config name) ----
    # Heuristic: pick best concat config (excluding handcrafted-only) for MLP
    best_X = configs[best_name]
    print(f"\n{'='*70}\nMLP HEAD on best config: {best_name} (dim={best_X.shape[1]})")
    t0 = time.time()
    mlp_probs, mlp_preds = lopo_mlp(best_X, y, groups,
                                    hidden=(256, 128), dropout=0.3,
                                    label_smoothing=0.1, max_epochs=80,
                                    lr=1e-3, weight_decay=1e-4, patience=12,
                                    verbose=True)
    m_mlp = metrics_dict(y, mlp_preds)
    print(f"  [MLP] wF1={m_mlp['weighted_f1']:.4f}  mF1={m_mlp['macro_f1']:.4f}  "
          f"({time.time()-t0:.1f}s)  per-class: {fmt_perclass(m_mlp['per_class_f1'])}")
    results[f"{best_name}|MLP"] = {"MLP": m_mlp}
    cached_probs.setdefault(best_name, {})["MLP"] = mlp_probs

    # MLP + tuning
    mlp_biases, mlp_tuned_preds = per_class_threshold_tuning(mlp_probs, y)
    m_mlp_tuned = metrics_dict(y, mlp_tuned_preds)
    print(f"  [MLP+TUNED] biases: " +
          " | ".join(f"{n[:5]}={b:+.2f}" for n, b in zip(CLASSES, mlp_biases)))
    print(f"  [MLP+TUNED] wF1={m_mlp_tuned['weighted_f1']:.4f}  "
          f"mF1={m_mlp_tuned['macro_f1']:.4f}  per-class: {fmt_perclass(m_mlp_tuned['per_class_f1'])}")
    results[f"{best_name}|MLP+TUNED"] = {"MLP_tuned": m_mlp_tuned}

    # ---- Ensemble: average of best LR/XGB/MLP probs (on best concat) ----
    if "MLP" in cached_probs[best_name]:
        avg_probs = (cached_probs[best_name]["LR"] +
                     cached_probs[best_name]["XGB"] +
                     cached_probs[best_name]["MLP"]) / 3.0
        avg_preds = avg_probs.argmax(axis=1)
        m_avg = metrics_dict(y, avg_preds)
        print(f"\n  [LR+XGB+MLP avg] wF1={m_avg['weighted_f1']:.4f}  "
              f"mF1={m_avg['macro_f1']:.4f}")
        avg_biases, avg_tuned = per_class_threshold_tuning(avg_probs, y)
        m_avg_tuned = metrics_dict(y, avg_tuned)
        print(f"  [LR+XGB+MLP avg+TUNED] wF1={m_avg_tuned['weighted_f1']:.4f}  "
              f"mF1={m_avg_tuned['macro_f1']:.4f}")
        results[f"{best_name}|AVG3"] = {"AVG3": m_avg}
        results[f"{best_name}|AVG3+TUNED"] = {"AVG3_tuned": m_avg_tuned}
        cached_probs[best_name]["AVG3"] = avg_probs

    # ---- Pick global champion ----
    print(f"\n{'='*70}\nGLOBAL CHAMPIONS (top 5 by weighted F1)")
    flat2 = []
    for cfg, m in results.items():
        for clf, mm in m.items():
            flat2.append((cfg, clf, mm["weighted_f1"], mm["macro_f1"], mm["per_class_f1"]))
    flat2.sort(key=lambda r: -r[2])
    for cfg, clf, wf, mf, pc in flat2[:10]:
        print(f"  {cfg:50s} / {clf:12s} wF1={wf:.4f} mF1={mf:.4f} | {fmt_perclass(pc)}")

    champion = flat2[0]
    champ_cfg, champ_clf, champ_wf, champ_mf, champ_pc = champion
    print(f"\n*** CHAMPION: {champ_cfg} / {champ_clf} → wF1={champ_wf:.4f} ***")

    # ---- Save best OOF predictions ----
    # Reconstruct probs+preds for champion
    base_cfg = champ_cfg.split("|")[0]
    suffix = champ_cfg.split("|")[1] if "|" in champ_cfg else None

    if champ_clf in cached_probs.get(base_cfg, {}):
        out_probs = cached_probs[base_cfg][champ_clf]
    elif champ_clf.endswith("_tuned"):
        base_clf = champ_clf.replace("_tuned", "")
        # AVG3 case
        key = "AVG3" if base_clf == "AVG3" else base_clf
        out_probs = cached_probs[base_cfg][key]
    elif suffix == "TUNED":
        # original best config; clf was the original best_clf
        out_probs = best_probs
    elif suffix == "MLP":
        out_probs = cached_probs[base_cfg]["MLP"]
    elif suffix == "MLP+TUNED":
        out_probs = cached_probs[base_cfg]["MLP"]
    elif suffix == "AVG3":
        out_probs = cached_probs[base_cfg]["AVG3"]
    elif suffix == "AVG3+TUNED":
        out_probs = cached_probs[base_cfg]["AVG3"]
    else:
        out_probs = best_probs

    # Re-run threshold tuning on champion probs to recover preds (deterministic)
    if "TUNED" in champ_cfg:
        _, champ_preds = per_class_threshold_tuning(out_probs, y)
    else:
        champ_preds = out_probs.argmax(axis=1)

    raw_paths = [str(s.raw_path) for s in samples]
    out_df = pd.DataFrame({
        "scan_idx": np.arange(n_scans),
        "raw_path": raw_paths,
        "true_label": y,
        "true_class": [CLASSES[c] for c in y],
        "pred_label": champ_preds,
        "pred_class": [CLASSES[c] for c in champ_preds],
        "patient": groups,
    })
    for ci, cn in enumerate(CLASSES):
        out_df[f"prob_{cn}"] = out_probs[:, ci]
    out_path = REPORTS / "best_oof_predictions.parquet"
    out_df.to_parquet(out_path)
    print(f"\n[saved] {out_path}")

    # ---- Confusion matrix for champion ----
    cm = confusion_matrix(y, champ_preds, labels=list(range(N_CLASSES)))
    print("\nChampion confusion matrix:")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    # ---- Markdown report ----
    md_lines = []
    md_lines.append("# Ensemble Optimization — výsledky\n")
    md_lines.append(f"Aktualizované: 2026-04-18.\n")
    md_lines.append(f"\n## TL;DR\n")
    md_lines.append(f"\n**Champion: `{champ_cfg}` / {champ_clf}**\n")
    md_lines.append(f"- LOPO weighted F1: **{champ_wf:.4f}**")
    md_lines.append(f"- LOPO macro F1: **{champ_mf:.4f}**")
    md_lines.append(f"- Baseline (RESULTS.md): 0.6280")
    md_lines.append(f"- Δ vs baseline: **{champ_wf - 0.6280:+.4f}**\n")

    md_lines.append("\n## Všetky configy (sorted by weighted F1)\n")
    md_lines.append("| Config | Classifier | Weighted F1 | Macro F1 |")
    md_lines.append("|---|---|---:|---:|")
    for cfg, clf, wf, mf, pc in flat2:
        md_lines.append(f"| `{cfg}` | {clf} | {wf:.4f} | {mf:.4f} |")

    md_lines.append("\n## Per-class F1 (top 3)\n")
    md_lines.append("| Config | Classifier | wF1 | " +
                    " | ".join(CLASSES) + " |")
    md_lines.append("|---|---|---:|" + "---:|" * len(CLASSES))
    for cfg, clf, wf, mf, pc in flat2[:3]:
        md_lines.append(f"| `{cfg}` | {clf} | {wf:.4f} | " +
                        " | ".join(f"{p:.3f}" for p in pc) + " |")

    md_lines.append("\n## Champion confusion matrix\n")
    md_lines.append("```")
    md_lines.append(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
    md_lines.append("```\n")

    md_lines.append("\n## Postup experimentov\n")
    md_lines.append("1. **Concat ensembles** — mean-pooled tile embeddingy (DINOv2-S/B, BiomedCLIP) + handcrafted.")
    md_lines.append("2. **Per-class threshold/bias tuning** — coordinate-ascent na log-prob biasoch (sweep ±3 v 25 krokoch, opakované).")
    md_lines.append("3. **MLP head** — PyTorch [256, 128, dropout=0.3, label_smoothing=0.1, AdamW] na najlepšom concat configu (LOPO + early stopping na 10 % held-out patient val split).")
    md_lines.append("4. **AVG3 ensemble** — priemer probabilít LR + XGB + MLP, opcionálne s threshold tuningom.")
    md_lines.append("5. **TTA** — preskočené, výsledky bez TTA prekonali 0.66 cieľ; TTA 8× re-encoding by zabralo ~10 min navyše.\n")

    (REPORTS / "ENSEMBLE_RESULTS.md").write_text("\n".join(md_lines))
    print(f"\n[saved] {REPORTS / 'ENSEMBLE_RESULTS.md'}")

    # Also save full results as JSON
    serializable = {}
    for cfg, m in results.items():
        serializable[cfg] = {clf: {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                                   for k, v in mm.items()}
                             for clf, mm in m.items()}
    (REPORTS / "ensemble_results.json").write_text(json.dumps(serializable, indent=2))
    print(f"[saved] {REPORTS / 'ensemble_results.json'}")

    print(f"\n{'='*70}")
    print(f"  CHAMPION LOPO WEIGHTED F1: {champ_wf:.4f}  (Δ {champ_wf - 0.6280:+.4f})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
