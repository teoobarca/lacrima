"""CGNN retry on CPU (MPS hangs on GINEConv backward).

Loads cached graphs from `cache/graphs/graphs_t90_c512_p60.pt` and evaluates with:
  a) 5-fold StratifiedGroupKFold, eye-level groups (original `patient`)
  b) 5-fold StratifiedGroupKFold, person-level groups (via teardrop.data.person_id)

Reduced config for CPU: epochs=40, hidden=48, n_layers=2, batch=4.
Includes a fold-0 diagnostic pass (5 epochs) before the full run.
Writes a markdown report to reports/CGNN_CPU_RESULTS.md.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from teardrop.cv import patient_stratified_kfold  # noqa: E402
from teardrop.data import CLASSES, person_id  # noqa: E402
from teardrop.gin_model import CGNN  # noqa: E402

CACHE_FILE = ROOT / "cache" / "graphs" / "graphs_t90_c512_p60.pt"
REPORT_FILE = ROOT / "reports" / "CGNN_CPU_RESULTS.md"

# Reduced CPU config
EPOCHS = 40
HIDDEN = 48
N_LAYERS = 2
DROPOUT = 0.3
LR = 1e-3
BATCH_SIZE = 4
WEIGHT_DECAY = 1e-4


def ensure_y(graphs, labels):
    for i, g in enumerate(graphs):
        if not hasattr(g, "y") or g.y is None:
            g.y = torch.tensor([int(labels[i])], dtype=torch.long)


def feature_stats(graphs):
    xs = torch.cat([g.x for g in graphs], dim=0)
    eas = torch.cat([g.edge_attr for g in graphs if g.edge_attr.numel() > 0], dim=0)
    empty_edges = sum(1 for g in graphs if g.edge_index.shape[1] == 0)
    return {
        "x_min": float(xs.min()),
        "x_max": float(xs.max()),
        "x_mean": float(xs.mean()),
        "ea_min": float(eas.min()) if eas.numel() > 0 else None,
        "ea_max": float(eas.max()) if eas.numel() > 0 else None,
        "ea_mean": float(eas.mean()) if eas.numel() > 0 else None,
        "n_empty_edges": empty_edges,
        "n_total": len(graphs),
    }


def _class_weights(labels_train):
    """Balanced class weights w/ missing-class fallback → ones."""
    present = np.unique(labels_train)
    cw = compute_class_weight("balanced", classes=present, y=labels_train)
    full = np.ones(len(CLASSES), dtype=np.float32)
    for c, w in zip(present, cw):
        full[int(c)] = float(w)
    return full


def train_eval_one_split(
    graphs, labels, train_idx, val_idx, *,
    epochs=EPOCHS, lr=LR, hidden=HIDDEN, n_layers=N_LAYERS,
    dropout=DROPOUT, batch_size=BATCH_SIZE, device="cpu",
    verbose=False, track_loss=False,
):
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    model = CGNN(node_in=5, edge_in=5, hidden=hidden, n_layers=n_layers,
                 n_classes=len(CLASSES), dropout=dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    cw_full = _class_weights(labels[train_idx])
    weight_t = torch.tensor(cw_full, dtype=torch.float32, device=device)

    best_val_f1 = -1.0
    best_preds = None
    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            out = model(batch)
            loss = F.cross_entropy(out, batch.y, weight=weight_t, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss /= max(1, len(train_graphs))

        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                all_preds.append(out.argmax(dim=-1).cpu().numpy())
        val_preds = np.concatenate(all_preds) if all_preds else np.array([])
        f1w = f1_score(labels[val_idx], val_preds, average="weighted",
                       zero_division=0) if len(val_preds) > 0 else 0.0
        if f1w > best_val_f1:
            best_val_f1 = f1w
            best_preds = val_preds
        if track_loss:
            history.append((epoch + 1, train_loss, float(f1w)))
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}: loss={train_loss:.4f}  val_F1w={f1w:.4f}  best={best_val_f1:.4f}")

    return best_preds, best_val_f1, history


def run_kfold(graphs, labels, groups, *, grouping_label: str, n_splits=5, device="cpu"):
    print(f"\n--- StratifiedGroupKFold (k={n_splits}, grouping={grouping_label}) ---")
    t_total = time.time()
    oof = np.full(len(labels), -1, dtype=int)
    fold_f1w, fold_f1m = [], []
    per_fold_rows = []
    for fi, (tr, va) in enumerate(patient_stratified_kfold(labels, groups, n_splits, seed=42)):
        t0 = time.time()
        preds, _, _ = train_eval_one_split(
            graphs, labels, tr, va, device=device, verbose=False,
        )
        oof[va] = preds
        f1w = f1_score(labels[va], preds, average="weighted", zero_division=0)
        f1m = f1_score(labels[va], preds, average="macro", zero_division=0)
        fold_f1w.append(f1w)
        fold_f1m.append(f1m)
        dt = time.time() - t0
        classes_present = sorted(set(int(x) for x in labels[va]))
        print(f"  Fold {fi}: F1w={f1w:.4f}  F1m={f1m:.4f}  "
              f"classes={classes_present}  n_val={len(va)}  {dt:.1f}s")
        per_fold_rows.append({
            "fold": fi, "n_val": int(len(va)),
            "F1w": float(f1w), "F1m": float(f1m),
            "classes": classes_present, "seconds": float(dt),
        })
    total_dt = time.time() - t_total
    print(f"  Mean F1w: {np.mean(fold_f1w):.4f} ± {np.std(fold_f1w):.4f}   "
          f"Mean F1m: {np.mean(fold_f1m):.4f} ± {np.std(fold_f1m):.4f}   "
          f"total {total_dt:.1f}s")

    mask = oof >= 0
    if mask.sum() == 0:
        print("  [warn] no OOF preds")
        return None
    oof_f1w = f1_score(labels[mask], oof[mask], average="weighted", zero_division=0)
    oof_f1m = f1_score(labels[mask], oof[mask], average="macro", zero_division=0)
    print(f"  OOF (n={int(mask.sum())}): F1w={oof_f1w:.4f}  F1m={oof_f1m:.4f}")

    # Per-class F1 & report
    report_str = classification_report(
        labels[mask], oof[mask], target_names=CLASSES,
        zero_division=0, digits=3,
    )
    print("\n" + report_str)
    cm = confusion_matrix(labels[mask], oof[mask], labels=list(range(len(CLASSES))))
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    print("Confusion:")
    print(cm_df.to_string())

    # per-class f1 dict
    per_cls_f1 = f1_score(labels[mask], oof[mask],
                          average=None, labels=list(range(len(CLASSES))),
                          zero_division=0)
    per_cls = {c: float(v) for c, v in zip(CLASSES, per_cls_f1)}

    return {
        "grouping": grouping_label,
        "fold_f1w": [float(x) for x in fold_f1w],
        "fold_f1m": [float(x) for x in fold_f1m],
        "mean_f1w": float(np.mean(fold_f1w)),
        "std_f1w": float(np.std(fold_f1w)),
        "mean_f1m": float(np.mean(fold_f1m)),
        "std_f1m": float(np.std(fold_f1m)),
        "oof_f1w": float(oof_f1w),
        "oof_f1m": float(oof_f1m),
        "per_class_f1": per_cls,
        "confusion": cm_df,
        "report": report_str,
        "seconds": float(total_dt),
        "per_fold": per_fold_rows,
    }


def diagnostic_fold0(graphs, labels, groups, *, device="cpu"):
    print("\n--- Diagnostic: fold-0, 5 epochs ---")
    splits = list(patient_stratified_kfold(labels, groups, n_splits=5, seed=42))
    tr, va = splits[0]
    t0 = time.time()
    preds, best_f1, history = train_eval_one_split(
        graphs, labels, tr, va, epochs=5, verbose=True,
        track_loss=True, device=device,
    )
    dt = time.time() - t0
    print(f"  diagnostic done in {dt:.1f}s, best_val_F1w={best_f1:.4f}")
    loss_first = history[0][1] if history else None
    loss_last = history[-1][1] if history else None
    print(f"  loss epoch1={loss_first:.4f}  epoch5={loss_last:.4f}  (want: decreasing)")
    return {
        "best_f1w": float(best_f1),
        "loss_first": float(loss_first) if loss_first is not None else None,
        "loss_last": float(loss_last) if loss_last is not None else None,
        "history": history,
        "seconds": float(dt),
    }


def write_report(wall_secs, stats, diag, res_eye, res_person, budget_note=""):
    lines = []
    lines.append("# CGNN (CPU) — Results\n")
    lines.append(f"Date: 2026-04-18  |  Wall time: **{wall_secs:.1f}s** "
                 f"({wall_secs/60:.1f} min)  |  Device: **CPU** (MPS hangs on GINEConv backward)\n")
    lines.append("## Config\n")
    lines.append(f"- Epochs: **{EPOCHS}**")
    lines.append(f"- Hidden: **{HIDDEN}**")
    lines.append(f"- Layers: **{N_LAYERS}**")
    lines.append(f"- Dropout: **{DROPOUT}**")
    lines.append(f"- LR: **{LR}**")
    lines.append(f"- Batch: **{BATCH_SIZE}**")
    lines.append(f"- Weight decay: **{WEIGHT_DECAY}**")
    lines.append(f"- Class-weighting: balanced (sklearn), label_smoothing=0.1, grad_clip=1.0")
    lines.append("- Model: GINEConv × N_LAYERS, 5-dim node feats, 5-dim edge feats, "
                 "mean+max+sum pool, MLP head, 5 classes\n")

    lines.append("## Graph feature sanity\n")
    lines.append(f"- n_graphs = {stats['n_total']}")
    lines.append(f"- node x: min={stats['x_min']:.3f}  max={stats['x_max']:.3f}  "
                 f"mean={stats['x_mean']:.3f}  (normalized)")
    if stats["ea_min"] is not None:
        lines.append(f"- edge_attr: min={stats['ea_min']:.3f}  max={stats['ea_max']:.3f}  "
                     f"mean={stats['ea_mean']:.3f}")
    lines.append(f"- graphs with 0 edges: {stats['n_empty_edges']}")
    lines.append("")

    lines.append("## Diagnostic run (fold-0, 5 epochs)\n")
    lines.append(f"- best val F1w: **{diag['best_f1w']:.4f}**")
    lines.append(f"- loss epoch 1: {diag['loss_first']:.4f}  "
                 f"→ epoch 5: {diag['loss_last']:.4f}")
    delta = (diag["loss_first"] - diag["loss_last"]) if (diag["loss_first"] is not None) else 0.0
    lines.append(f"- loss Δ: {delta:+.4f}  "
                 f"({'learning' if delta > 0.02 else 'stagnant — suspect'})")
    lines.append(f"- time: {diag['seconds']:.1f}s")
    lines.append("")

    for res, header in ((res_eye, "Eye-level grouping (`patient`, 44 groups)"),
                        (res_person, "Person-level grouping (`person_id`, ~35 groups)")):
        lines.append(f"## {header}\n")
        if res is None:
            lines.append("_Skipped._\n")
            continue
        lines.append(f"- Mean fold F1w: **{res['mean_f1w']:.4f} ± {res['std_f1w']:.4f}**")
        lines.append(f"- Mean fold F1m: **{res['mean_f1m']:.4f} ± {res['std_f1m']:.4f}**")
        lines.append(f"- OOF F1w: **{res['oof_f1w']:.4f}**  |  OOF F1m: **{res['oof_f1m']:.4f}**")
        lines.append(f"- Total time: {res['seconds']:.1f}s")
        lines.append("\n**Per-fold:**\n")
        lines.append("| Fold | n_val | F1w | F1m | classes | sec |")
        lines.append("|---:|---:|---:|---:|---|---:|")
        for row in res["per_fold"]:
            lines.append(f"| {row['fold']} | {row['n_val']} | "
                         f"{row['F1w']:.4f} | {row['F1m']:.4f} | "
                         f"{row['classes']} | {row['seconds']:.1f} |")
        lines.append("\n**Per-class F1 (OOF):**\n")
        lines.append("| Class | F1 |")
        lines.append("|---|---:|")
        for c, v in res["per_class_f1"].items():
            lines.append(f"| {c} | {v:.3f} |")
        lines.append("\n**Classification report:**\n")
        lines.append("```")
        lines.append(res["report"].rstrip())
        lines.append("```\n")
        lines.append("**Confusion matrix:**\n")
        lines.append("```")
        lines.append(res["confusion"].to_string())
        lines.append("```\n")

    # Comparison
    lines.append("## Comparison to DINOv2 baseline\n")
    lines.append("| Model | Eye-LOPO F1w | Person-LOPO F1w | Macro F1 |")
    lines.append("|---|---:|---:|---:|")
    lines.append("| DINOv2-B tiled scan-mean + LR | 0.628 | **0.615** | 0.491 |")
    if res_eye is not None and res_person is not None:
        lines.append(f"| CGNN (GINE, CPU, this run) | {res_eye['oof_f1w']:.3f} | "
                     f"{res_person['oof_f1w']:.3f} | "
                     f"{res_person['oof_f1m']:.3f} |")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation\n")
    if res_person is not None:
        cgnn_person = res_person["oof_f1w"]
        dinov2_person = 0.615
        gap = cgnn_person - dinov2_person
        converged = diag["loss_first"] is not None and (
            diag["loss_first"] - diag["loss_last"]
        ) > 0.02
        if not converged:
            lines.append("- **Diagnostic shows loss did not drop meaningfully in 5 epochs** — "
                         "the GINE model may not be learning strong signal from the "
                         "5-dim node + 5-dim edge features alone.")
        if gap >= 0.0:
            lines.append(f"- CGNN **matches or exceeds** the DINOv2 person-LOPO baseline "
                         f"(Δ = {gap:+.3f}). Graph-topology features carry independent signal.")
        elif gap >= -0.05:
            lines.append(f"- CGNN is **within ~5 pp** of the DINOv2 baseline "
                         f"(Δ = {gap:+.3f}). Usable as an ensemble / interpretability lens, "
                         "but not a standalone replacement.")
        else:
            lines.append(f"- CGNN is **clearly weaker** than DINOv2 "
                         f"(Δ = {gap:+.3f}). Value is in **interpretability** "
                         "(node = junction/endpoint, edge = skeleton segment length/tortuosity) "
                         "rather than raw F1. Graph-topology features alone do not capture "
                         "the full AFM appearance.")
        lines.append("- Worth keeping as an **ensemble component** only if its softmax is "
                     "diverse w.r.t. DINOv2 errors — a separate analysis.")
    if budget_note:
        lines.append(f"\n_Note: {budget_note}_\n")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[wrote] {REPORT_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--skip-diag", action="store_true",
                        help="Skip fold-0 diagnostic run")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"[load] {CACHE_FILE}")
    bundle = torch.load(CACHE_FILE, weights_only=False)
    graphs = bundle["graphs"]
    labels = np.asarray(bundle["labels"])
    groups_eye = np.asarray(bundle["groups"])
    paths = bundle["paths"]

    # Re-derive person-level groups from cached paths
    groups_person = np.array([person_id(Path(p)) for p in paths])
    print(f"  n_graphs = {len(graphs)}")
    print(f"  unique eye groups: {len(set(groups_eye))}")
    print(f"  unique person groups: {len(set(groups_person))}")
    print(f"  class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    ensure_y(graphs, labels)

    stats = feature_stats(graphs)
    print(f"\nGraph feature sanity:")
    print(f"  node x: min={stats['x_min']:.3f} max={stats['x_max']:.3f} mean={stats['x_mean']:.3f}")
    if stats["ea_min"] is not None:
        print(f"  edge_attr: min={stats['ea_min']:.3f} max={stats['ea_max']:.3f} mean={stats['ea_mean']:.3f}")
    print(f"  empty-edge graphs: {stats['n_empty_edges']} / {stats['n_total']}")

    device = "cpu"
    t_wall = time.time()

    # Diagnostic
    if args.skip_diag:
        diag = {"best_f1w": -1.0, "loss_first": None, "loss_last": None,
                "history": [], "seconds": 0.0}
    else:
        diag = diagnostic_fold0(graphs, labels, groups_eye, device=device)

    # Full runs
    n_splits = args.n_splits
    budget_note = ""
    res_eye = run_kfold(graphs, labels, groups_eye,
                       grouping_label="eye", n_splits=n_splits, device=device)

    # Budget check: if eye run already took >10 min, fall back to 3-fold for person
    eye_secs = res_eye["seconds"] if res_eye else 0.0
    diag_secs = diag["seconds"]
    if (eye_secs + diag_secs) > 10 * 60 and n_splits > 3:
        budget_note = f"Eye-level k={n_splits} took {eye_secs:.1f}s; falling back to k=3 for person-level."
        print(f"\n[budget] {budget_note}")
        n_splits_person = 3
    else:
        n_splits_person = n_splits

    res_person = run_kfold(graphs, labels, groups_person,
                           grouping_label="person", n_splits=n_splits_person, device=device)

    wall = time.time() - t_wall
    print(f"\n>>> Total wall time: {wall:.1f}s ({wall/60:.1f} min)")

    write_report(wall, stats, diag, res_eye, res_person, budget_note)


if __name__ == "__main__":
    main()
