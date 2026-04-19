"""Prototypical Networks for tear AFM classification.

Reference: Snell, Swersky, Zemel (2017) — "Prototypical Networks for Few-Shot
Learning" (NeurIPS). https://arxiv.org/abs/1703.05175

Our setting differs from canonical few-shot meta-learning: we have a FIXED set
of 5 classes (not episodic N-way), but the class counts are dramatically few-shot
at the person level: 2, 4, 5, 9, 15 persons per class. ProtoNets should shine
here because:

  1. They do not fit per-class decision boundaries (which LR does poorly with
     ~2 persons per class).
  2. They express each class purely as the centroid of its support embeddings —
     the minority class is on equal footing with the majority.
  3. An optional metric-learning adapter can reshape the embedding space so that
     class centroids are further apart than within-class spread, without requiring
     any class-count-dependent capacity.

Two prototype classifiers live here:

  `ProtoClassifier`  — training-free: prototype = mean(L2-normalized train
                        embeddings per class), classify by softmax over
                        negative (cosine / sq-Euclidean) distances.
  `MetricAdapter`    — small MLP (D -> 256 -> 128) trained with the
                        ProtoNet episodic loss on the training fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Distance = Literal["cosine", "sqeuclidean"]


def _l2norm(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


@dataclass
class ProtoClassifier:
    """Training-free prototypical classifier.

    prototype_c = mean(L2-normalized support embeddings of class c)

    Distance options:
        - "cosine":      d(q, p) = 1 - (q . p)          [q, p are L2-normed]
        - "sqeuclidean": d(q, p) = ||q - p||^2          [on L2-normed vectors]
    """

    distance: Distance = "sqeuclidean"
    temperature: float = 1.0        # softmax temperature (logit scale 1/T)
    n_classes: int = 5
    weighted: bool = False          # inverse-distance-within-class weighting

    prototypes_: np.ndarray | None = None    # shape (C, D)
    classes_: np.ndarray | None = None       # shape (C,)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProtoClassifier":
        Xn = _l2norm(X.astype(np.float32))
        classes = np.sort(np.unique(y))
        D = Xn.shape[1]
        P = np.zeros((self.n_classes, D), dtype=np.float32)
        for c in range(self.n_classes):
            mask = y == c
            if not mask.any():
                # absent class in this fold — leave zero prototype; distances
                # to it will still be finite but it will never win unless all
                # other classes are even further.
                continue
            sub = Xn[mask]
            if self.weighted and sub.shape[0] > 1:
                # Weight support points by inverse distance to the class centroid:
                # this down-weights outliers within a class (Mahalanobis-ish).
                centroid = sub.mean(axis=0)
                centroid /= max(np.linalg.norm(centroid), 1e-12)
                d = np.linalg.norm(sub - centroid, axis=1)    # shape (n_c,)
                w = 1.0 / (d + 1e-6)
                w = w / w.sum()
                proto = (w[:, None] * sub).sum(axis=0)
            else:
                proto = sub.mean(axis=0)
            # Re-normalize prototype (Snell et al. report minor help)
            proto /= max(np.linalg.norm(proto), 1e-12)
            P[c] = proto
        self.prototypes_ = P
        self.classes_ = classes
        return self

    def _distances(self, X: np.ndarray) -> np.ndarray:
        assert self.prototypes_ is not None
        Xn = _l2norm(X.astype(np.float32))
        P = self.prototypes_
        if self.distance == "cosine":
            # both L2-normalized ⇒ cosine-sim = dot product; distance = 1 - sim.
            sim = Xn @ P.T
            return (1.0 - sim).astype(np.float32)
        elif self.distance == "sqeuclidean":
            # For unit vectors: ||q - p||^2 = 2 - 2 q.p, but keep generic.
            xx = (Xn * Xn).sum(axis=1, keepdims=True)
            pp = (P * P).sum(axis=1)[None, :]
            qp = Xn @ P.T
            d2 = np.clip(xx + pp - 2.0 * qp, 0.0, None)
            return d2.astype(np.float32)
        else:
            raise ValueError(self.distance)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        d = self._distances(X)
        # logits = -d / temperature
        logits = -d / max(self.temperature, 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        return (ex / ex.sum(axis=1, keepdims=True)).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# -----------------------------------------------------------------------------
# Metric-learning adapter (episodic ProtoNet training on one LOPO fold)
# -----------------------------------------------------------------------------


class MetricAdapter(nn.Module):
    """Small MLP: D -> hidden -> out, L2-normalized output."""

    def __init__(self, in_dim: int, hidden: int = 256, out: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


def _sample_episode(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    k_support: int,
    k_query: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Sample a group-aware episode from a single training fold.

    Returns (X_s, y_s, X_q, y_q, classes) or None if sampling fails.

    Group-awareness: support and query of the same class may share persons
    (the LOPO fold already removed the validation person). To preserve a
    mild generalization signal, we still try to split persons between support
    and query when possible — if a class has only one person, support+query
    both come from it (can't be helped).
    """
    classes = np.arange(n_classes)
    rng.shuffle(classes)
    chosen = []
    for c in classes:
        mask = y == c
        if mask.sum() < (k_support + 1):
            continue
        chosen.append(int(c))
        if len(chosen) == n_classes:
            break
    if not chosen:
        return None

    Xs_list, ys_list, Xq_list, yq_list = [], [], [], []
    for c in chosen:
        mask = y == c
        idx_c = np.where(mask)[0]
        g_c = groups[idx_c]
        uniq_persons = np.unique(g_c)
        if len(uniq_persons) >= 2 and k_query > 0:
            # hold out one person for query
            rng.shuffle(uniq_persons)
            q_persons = set(uniq_persons[:1])
            s_idx = np.array([i for i in idx_c if groups[i] not in q_persons])
            q_idx = np.array([i for i in idx_c if groups[i] in q_persons])
        else:
            # single-person class: can't split; random split
            rng.shuffle(idx_c)
            s_idx = idx_c[:max(1, len(idx_c) - k_query)]
            q_idx = idx_c[max(1, len(idx_c) - k_query):]

        if len(s_idx) == 0 or len(q_idx) == 0:
            continue

        # subsample support to k_support, query to k_query
        k_s = min(k_support, len(s_idx))
        k_q = min(k_query, len(q_idx))
        rng.shuffle(s_idx)
        rng.shuffle(q_idx)
        Xs_list.append(X[s_idx[:k_s]])
        ys_list.append(np.full(k_s, c, dtype=np.int64))
        Xq_list.append(X[q_idx[:k_q]])
        yq_list.append(np.full(k_q, c, dtype=np.int64))

    if len(Xs_list) < 2:
        # need at least 2 classes in the episode for the loss to be meaningful
        return None

    X_s = np.concatenate(Xs_list, axis=0)
    y_s = np.concatenate(ys_list, axis=0)
    X_q = np.concatenate(Xq_list, axis=0)
    y_q = np.concatenate(yq_list, axis=0)
    class_arr = np.array(sorted({int(c) for c in y_s}), dtype=np.int64)
    return X_s, y_s, X_q, y_q, class_arr


def train_adapter(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_classes: int = 5,
    hidden: int = 256,
    out_dim: int = 128,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_episodes: int = 400,
    k_support: int = 4,
    k_query: int = 2,
    distance: Distance = "sqeuclidean",
    learnable_temperature: bool = False,
    temperature_init: float = 1.0,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = False,
) -> tuple[MetricAdapter, float]:
    """Train a MetricAdapter with episodic ProtoNet loss.

    Returns (trained adapter, final temperature).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = MetricAdapter(in_dim=X.shape[1], hidden=hidden, out=out_dim,
                          dropout=dropout).to(device)
    # temperature as log-scale parameter for stability
    log_temp = torch.tensor(
        np.log(max(temperature_init, 1e-3)), dtype=torch.float32, device=device,
        requires_grad=learnable_temperature,
    )
    params = list(model.parameters())
    if learnable_temperature:
        params = params + [log_temp]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    X_t = torch.from_numpy(X.astype(np.float32)).to(device)

    model.train()
    last_losses: list[float] = []
    for ep in range(n_episodes):
        sample = _sample_episode(
            X, y, groups, n_classes=n_classes,
            k_support=k_support, k_query=k_query, rng=rng,
        )
        if sample is None:
            continue
        Xs_np, ys_np, Xq_np, yq_np, ep_classes = sample

        # indices (into X) of support / query in this episode
        # (we sampled from raw X arrays, so just re-embed those rows)
        Xs = torch.from_numpy(Xs_np.astype(np.float32)).to(device)
        Xq = torch.from_numpy(Xq_np.astype(np.float32)).to(device)
        ys = torch.from_numpy(ys_np).to(device)
        yq = torch.from_numpy(yq_np).to(device)

        zs = model(Xs)       # (Ns, out)
        zq = model(Xq)       # (Nq, out)

        # Compute prototype per class in this episode
        protos = []
        for c in ep_classes:
            c_mask = ys == int(c)
            p = zs[c_mask].mean(dim=0)
            p = F.normalize(p, p=2, dim=-1)
            protos.append(p)
        P = torch.stack(protos, dim=0)   # (C_ep, out)

        # Compute distances (C_ep,) logits per query
        if distance == "sqeuclidean":
            # ||z - p||^2 = ||z||^2 + ||p||^2 - 2 z.p
            # (write it out — torch.cdist backward is not implemented on MPS)
            zz = (zq * zq).sum(dim=1, keepdim=True)            # (Nq, 1)
            pp = (P * P).sum(dim=1).unsqueeze(0)               # (1, C_ep)
            d = zz + pp - 2.0 * (zq @ P.T)
            d = torch.clamp(d, min=0.0)
        elif distance == "cosine":
            d = 1.0 - zq @ P.T
        else:
            raise ValueError(distance)

        temp = log_temp.exp()
        logits = -d / torch.clamp(temp, min=1e-3)
        # target class index within this episode's class list
        class_to_ep_idx = {int(c): i for i, c in enumerate(ep_classes.tolist())}
        tgt = torch.tensor([class_to_ep_idx[int(c)] for c in yq.cpu().numpy()],
                           dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, tgt)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        opt.step()

        last_losses.append(float(loss.item()))
        if verbose and (ep + 1) % 100 == 0:
            avg = float(np.mean(last_losses[-100:]))
            print(f"    ep {ep + 1:4d} loss-ma100={avg:.4f} T={float(temp):.3f}")

    return model, float(log_temp.exp().item())


@torch.no_grad()
def embed_with_adapter(
    model: MetricAdapter, X: np.ndarray, device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size].astype(np.float32)).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)
