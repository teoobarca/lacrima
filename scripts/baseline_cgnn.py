"""Train Crystal Graph NN on tear AFM with patient-level CV."""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from teardrop.cv import leave_one_patient_out, patient_stratified_kfold
from teardrop.data import CLASSES, enumerate_samples, preprocess_spm
from teardrop.gin_model import CGNN
from teardrop.graph import height_to_graph, graph_summary

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache" / "graphs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def build_graph_dataset(samples, target_nm_per_px=90.0, crop_size=512,
                        threshold_pct=60.0, min_object_size=50):
    """Pre-compute graph for each scan, cache to disk."""
    cache_file = CACHE_DIR / f"graphs_t{int(target_nm_per_px)}_c{crop_size}_p{int(threshold_pct)}.pt"
    if cache_file.exists():
        print(f"[cache] {cache_file}")
        return torch.load(cache_file, weights_only=False)

    print(f"Building graphs for {len(samples)} scans...")
    graphs, labels, groups, paths = [], [], [], []
    t0 = time.time()
    n_nodes_total = 0
    n_edges_total = 0
    for i, s in enumerate(samples):
        try:
            h = preprocess_spm(s.raw_path, target_nm_per_px=target_nm_per_px,
                               crop_size=crop_size)
            g = height_to_graph(h, threshold_pct=threshold_pct,
                                min_object_size=min_object_size)
            g.y = torch.tensor([s.label], dtype=torch.long)
            graphs.append(g)
            labels.append(s.label)
            groups.append(s.patient)
            paths.append(str(s.raw_path))
            n_nodes_total += int(g.x.shape[0])
            n_edges_total += int(g.edge_index.shape[1] // 2)
        except Exception as e:
            print(f"  [err] {s.raw_path.name}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(samples)}] {time.time()-t0:.1f}s "
                  f"avg_nodes={n_nodes_total / (i+1):.0f} avg_edges={n_edges_total / (i+1):.0f}")

    print(f"  total {time.time()-t0:.1f}s")
    bundle = dict(graphs=graphs, labels=np.array(labels), groups=np.array(groups), paths=paths)
    torch.save(bundle, cache_file)
    print(f"[saved] {cache_file}")
    return bundle


def train_eval_one_split(graphs, labels, train_idx, val_idx, *,
                         epochs=80, lr=5e-4, hidden=64, n_layers=3, dropout=0.3,
                         device="cpu", verbose=False):
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]

    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)

    model = CGNN(node_in=5, edge_in=5, hidden=hidden, n_layers=n_layers,
                 n_classes=len(CLASSES), dropout=dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    cw = compute_class_weight("balanced", classes=np.unique(labels[train_idx]),
                              y=labels[train_idx])
    cw_full = np.ones(len(CLASSES))
    for i, c in enumerate(np.unique(labels[train_idx])):
        cw_full[c] = cw[i]
    weight_t = torch.tensor(cw_full, dtype=torch.float32, device=device)

    best_val_f1 = -1.0
    best_preds = None
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            out = model(batch)
            loss = F.cross_entropy(out, batch.y, weight=weight_t,
                                   label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_graphs)

        # Eval
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                all_preds.append(out.argmax(dim=-1).cpu().numpy())
        val_preds = np.concatenate(all_preds)
        f1w = f1_score(labels[val_idx], val_preds, average="weighted")
        if f1w > best_val_f1:
            best_val_f1 = f1w
            best_preds = val_preds
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}: loss={train_loss:.4f}  val F1={f1w:.4f}  best={best_val_f1:.4f}")

    return best_preds


def evaluate(graphs, labels, groups, n_splits=5, n_epochs=80,
             hidden=64, n_layers=3, dropout=0.3):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # K-fold first (faster, partial signal)
    print(f"\n--- StratifiedGroupKFold (k={n_splits}) ---")
    fold_f1s = []
    oof_preds = np.full(len(labels), -1, dtype=int)
    for fi, (tr, va) in enumerate(patient_stratified_kfold(labels, groups, n_splits, seed=42)):
        t0 = time.time()
        preds = train_eval_one_split(graphs, labels, tr, va,
                                     epochs=n_epochs, hidden=hidden,
                                     n_layers=n_layers, dropout=dropout,
                                     device=device)
        oof_preds[va] = preds
        f1 = f1_score(labels[va], preds, average="weighted")
        f1m = f1_score(labels[va], preds, average="macro")
        present = sorted(set(labels[va]))
        print(f"  Fold {fi}: F1w={f1:.4f}  F1m={f1m:.4f}  classes={present}  "
              f"n={len(va)}  {time.time()-t0:.1f}s")
        fold_f1s.append(f1)
    print(f"  Mean weighted F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    mask = oof_preds >= 0
    print(f"\n  OOF aggregate (n={mask.sum()}):")
    print(f"    Weighted F1: {f1_score(labels[mask], oof_preds[mask], average='weighted'):.4f}")
    print(f"    Macro F1:    {f1_score(labels[mask], oof_preds[mask], average='macro'):.4f}")
    print("\n" + classification_report(labels[mask], oof_preds[mask],
                                       target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(labels[mask], oof_preds[mask], labels=list(range(len(CLASSES))))
    print("Confusion:")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    return {"kfold_f1": float(np.mean(fold_f1s))}


def main():
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    print(f"Loaded {len(samples)} samples")

    bundle = build_graph_dataset(samples, target_nm_per_px=90.0, crop_size=512,
                                 threshold_pct=60.0, min_object_size=50)
    graphs = bundle["graphs"]
    labels = bundle["labels"]
    groups = bundle["groups"]

    # Sanity stats
    sizes = [int(g.x.shape[0]) for g in graphs]
    edges = [int(g.edge_index.shape[1] // 2) for g in graphs]
    print(f"\nGraph stats: nodes mean={np.mean(sizes):.0f} median={np.median(sizes):.0f} "
          f"min={np.min(sizes)} max={np.max(sizes)}")
    print(f"             edges mean={np.mean(edges):.0f} median={np.median(edges):.0f}")

    metrics = evaluate(graphs, labels, groups, n_splits=5, n_epochs=80,
                       hidden=64, n_layers=3, dropout=0.3)
    print(f"\n>>> CGNN final: {metrics}")


if __name__ == "__main__":
    main()
