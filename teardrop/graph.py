"""Crystal Graph extraction from AFM tear-desiccate height maps.

Pipeline:
1. Threshold height map (above-median = "crystal", below = "background")
2. Skeletonize (Lee-style 1-pixel-wide skeleton)
3. Convert skeleton → graph via sknw:
   - nodes = junctions (deg ≥ 3) and endpoints (deg = 1)
   - edges = skeleton segments between nodes
4. Compute features:
   - per-node: degree, normalized x/y position, local height
   - per-edge: length, mean height along path, tortuosity, principal angle
5. Return PyTorch Geometric `Data` object.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sknw
import torch
from skimage.morphology import skeletonize
from torch_geometric.data import Data


def height_to_skeleton(
    h: np.ndarray,
    threshold_pct: float = 60.0,
    min_object_size: int = 50,
) -> np.ndarray:
    """Threshold + skeletonize. Returns boolean skeleton (1-pixel wide)."""
    thr = np.percentile(h, threshold_pct)
    binary = h > thr
    if min_object_size > 0:
        from skimage.morphology import remove_small_objects
        binary = remove_small_objects(binary, min_size=min_object_size)
    return skeletonize(binary)


def skeleton_to_graph(skel: np.ndarray, h: np.ndarray) -> Data:
    """Convert binary skeleton → PyG Data with node/edge features.

    Node features (5-dim):
        x_norm, y_norm    — normalized coordinates [0, 1]
        height_local      — height value at node pixel (already in [0,1])
        degree            — graph degree (raw count)
        is_junction       — 1.0 if degree ≥ 3 else 0.0

    Edge features (5-dim):
        length_norm       — length in normalized units (px / image_size)
        mean_height       — mean height value along edge path
        tortuosity        — actual_length / euclidean_distance
        angle_sin         — sin of principal direction angle
        angle_cos         — cos of principal direction angle
    """
    H, W = h.shape
    g = sknw.build_sknw(skel.astype(np.uint8), multi=False)

    # If no nodes (empty skeleton), return single dummy node
    if g.number_of_nodes() == 0:
        x = torch.zeros((1, 5), dtype=torch.float32)
        ei = torch.zeros((2, 0), dtype=torch.long)
        ea = torch.zeros((0, 5), dtype=torch.float32)
        return Data(x=x, edge_index=ei, edge_attr=ea, n_nodes=torch.tensor(1))

    # Build node features
    nodes = list(g.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    node_feats = np.zeros((n_nodes, 5), dtype=np.float32)
    for n in nodes:
        idx = node_to_idx[n]
        # sknw stores node coords as 'o' (origin)
        coord = g.nodes[n]["o"]
        y, x = float(coord[0]), float(coord[1])
        deg = int(g.degree(n))
        node_feats[idx, 0] = x / W
        node_feats[idx, 1] = y / H
        # Sample height at the (rounded) coord location
        yi = max(0, min(H - 1, int(round(y))))
        xi = max(0, min(W - 1, int(round(x))))
        node_feats[idx, 2] = float(h[yi, xi])
        node_feats[idx, 3] = float(deg) / 10.0  # crude scaling
        node_feats[idx, 4] = 1.0 if deg >= 3 else 0.0

    # Build edge features — bidirectional for undirected graph
    edge_index_list = []
    edge_attr_list = []
    image_diag = float(np.sqrt(H * H + W * W))
    for u, v, data in g.edges(data=True):
        if u == v:
            continue
        pts = data.get("pts", np.empty((0, 2)))
        # Length = number of pixels in segment path
        path_len = float(len(pts))
        # Mean height along path
        mean_h = 0.0
        if len(pts) > 0:
            ys = np.clip(pts[:, 0].astype(int), 0, H - 1)
            xs = np.clip(pts[:, 1].astype(int), 0, W - 1)
            mean_h = float(h[ys, xs].mean())
        # Endpoints (cast to float64 to avoid uint overflow in subtraction)
        u_coord = np.asarray(g.nodes[u]["o"], dtype=np.float64)
        v_coord = np.asarray(g.nodes[v]["o"], dtype=np.float64)
        dy = float(v_coord[0] - u_coord[0])
        dx = float(v_coord[1] - u_coord[1])
        eucl = float(np.sqrt(dx * dx + dy * dy))
        tortuosity = path_len / max(1.0, eucl)
        if eucl > 1e-3:
            angle = np.arctan2(dy, dx)
        else:
            angle = 0.0

        feat = np.array([
            path_len / image_diag,
            mean_h,
            tortuosity,
            float(np.sin(2 * angle)),  # 2x for undirected angle
            float(np.cos(2 * angle)),
        ], dtype=np.float32)

        ui, vi = node_to_idx[u], node_to_idx[v]
        edge_index_list.append([ui, vi])
        edge_index_list.append([vi, ui])  # undirected → both directions
        edge_attr_list.append(feat)
        edge_attr_list.append(feat)

    if not edge_index_list:
        ei = torch.zeros((2, 0), dtype=torch.long)
        ea = torch.zeros((0, 5), dtype=torch.float32)
    else:
        ei = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        ea = torch.tensor(np.stack(edge_attr_list), dtype=torch.float32)

    x = torch.tensor(node_feats, dtype=torch.float32)
    return Data(x=x, edge_index=ei, edge_attr=ea, n_nodes=torch.tensor(n_nodes))


def height_to_graph(
    h: np.ndarray,
    threshold_pct: float = 60.0,
    min_object_size: int = 50,
) -> Data:
    """End-to-end height-map → PyG Data graph."""
    skel = height_to_skeleton(h, threshold_pct=threshold_pct,
                              min_object_size=min_object_size)
    return skeleton_to_graph(skel, h)


def graph_summary(data: Data) -> dict[str, float]:
    """Quick stats for sanity / debugging."""
    return {
        "n_nodes": int(data.x.shape[0]),
        "n_edges": int(data.edge_index.shape[1] // 2),
        "mean_degree": float(data.x[:, 3].mean() * 10.0),
        "n_junctions": int((data.x[:, 4] > 0.5).sum()),
        "n_endpoints": int(((data.x[:, 3] * 10.0) < 1.5).sum()),
    }
