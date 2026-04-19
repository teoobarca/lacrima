"""Multichannel AFM probe + encoding + LOPO evaluation.

Tasks:
 1. Survey channels present in every SPM file → reports/channel_survey.csv
 2. Build per-channel tiled DINOv2 embeddings for
        Height, Amplitude Error, Phase, and stacked RGB(H|A|P)
 3. Evaluate person-level LOPO F1 for each channel + the RGB variant + concat.

Pipeline mirrors scripts/baseline_tiled_ensemble.py (tile-level LR + mean-prob
scan aggregation) so F1 is directly comparable to the tiled DINOv2-B baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pySPM
from PIL import Image

ROOT = Path("/Users/rafael/Programming/teardrop-challenge")
sys.path.insert(0, str(ROOT))

from teardrop.data import (  # noqa: E402
    CLASSES, enumerate_samples,
    plane_level, resample_to_pixel_size, robust_normalize,
    tile,
)
from teardrop.cv import leave_one_patient_out  # noqa: E402
from teardrop.encoders import load_dinov2, height_to_pil  # noqa: E402


# ---------------------------------------------------------------------------
# Channel discovery
# ---------------------------------------------------------------------------

def list_channels(path: Path) -> list[str]:
    b = pySPM.Bruker(str(path))
    names: list[str] = []
    for layer in b.layers:
        raw = layer[b"@2:Image Data"][0].decode("latin1")
        m = re.match(r"([^ ]+) \[([^\]]*)\] \"([^\"]*)\"", raw)
        if m:
            names.append(m.group(3))
    return names


def channel_survey(samples, out_csv: Path):
    rows = []
    per_sample_channels: list[tuple[str, list[str]]] = []
    for s in samples:
        try:
            chans = list_channels(s.raw_path)
        except Exception as e:
            chans = [f"ERROR:{type(e).__name__}:{e}"]
        rows.append({
            "file": str(s.raw_path.relative_to(ROOT)),
            "class": s.cls,
            "channel_list": "|".join(chans),
            "n_channels": len(chans),
        })
        per_sample_channels.append((s.cls, chans))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total = len(per_sample_channels)
    counts: Counter[str] = Counter()
    per_class_counts: dict[str, Counter[str]] = {c: Counter() for c in CLASSES}
    per_class_total: dict[str, int] = defaultdict(int)
    for cls, chans in per_sample_channels:
        per_class_total[cls] += 1
        for ch in set(chans):
            counts[ch] += 1
            per_class_counts[cls][ch] += 1
    return {
        "total_files": total,
        "channel_occurrence": dict(counts),
        "per_class_channel_occurrence": {
            cls: dict(per_class_counts[cls]) for cls in CLASSES
        },
        "per_class_total": dict(per_class_total),
    }


# ---------------------------------------------------------------------------
# Per-channel load (pySPM direct; AFMReader only supports Height channel name)
# ---------------------------------------------------------------------------

def load_channel_array(path: Path, channel: str) -> tuple[np.ndarray, float]:
    b = pySPM.Bruker(str(path))
    for backward in (False, True):
        try:
            img = b.get_channel(channel=channel, backward=backward, lazy=False)
            arr = img.pixels.astype(np.float32)
            size = img.size
            x_nm = None
            unit = "nm"
            if isinstance(size, dict):
                real = size.get("real", {})
                x_nm = real.get("x")
                unit = real.get("unit", "nm")
            if x_nm is None:
                pixel_nm = 1.0
            else:
                if unit == "um":
                    x_nm = x_nm * 1000.0
                elif unit == "m":
                    x_nm = x_nm * 1e9
                pixel_nm = float(x_nm) / arr.shape[1]
            return arr, pixel_nm
        except Exception:
            continue
    raise RuntimeError(f"No channel {channel!r} found in {path.name}")


def preprocess_channel_tiles(
    path: Path, channel: str,
    target_nm_per_px: float = 90.0,
    tile_size: int = 512,
    max_tiles: int = 9,
) -> list[np.ndarray]:
    arr, px_nm = load_channel_array(path, channel)
    arr = plane_level(arr)
    arr = resample_to_pixel_size(arr, px_nm, target_nm_per_px)
    arr = robust_normalize(arr)
    if arr.shape[0] < tile_size or arr.shape[1] < tile_size:
        pad_h = max(0, tile_size - arr.shape[0])
        pad_w = max(0, tile_size - arr.shape[1])
        arr = np.pad(arr, ((pad_h // 2, pad_h - pad_h // 2),
                           (pad_w // 2, pad_w - pad_w // 2)), mode="reflect")
    tiles = tile(arr, tile_size)
    if not tiles:
        return [arr[:tile_size, :tile_size]]
    if len(tiles) > max_tiles:
        idx = np.linspace(0, len(tiles) - 1, max_tiles).astype(int)
        tiles = [tiles[i] for i in idx]
    return tiles


# ---------------------------------------------------------------------------
# Tiled embedding builder (per channel, single mode)
# ---------------------------------------------------------------------------

def build_tile_embeddings_singlechan(
    samples, channel: str, encoder,
    tile_size: int = 512, max_tiles: int = 9,
    render_mode: str = "afmhot",
    skip_missing: bool = True,
):
    """Returns X, tile_to_scan, scan_y, scan_groups, scan_paths."""
    all_pil = []
    tile_to_scan = []
    scan_y, scan_groups, scan_paths = [], [], []

    t0 = time.time()
    for si, s in enumerate(samples):
        try:
            tiles = preprocess_channel_tiles(
                s.raw_path, channel,
                tile_size=tile_size, max_tiles=max_tiles,
            )
        except Exception as e:
            if skip_missing:
                continue
            raise
        scan_idx = len(scan_y)
        for t in tiles:
            all_pil.append(height_to_pil(t, mode=render_mode))
            tile_to_scan.append(scan_idx)
        scan_y.append(s.label)
        scan_groups.append(s.person)
        scan_paths.append(str(s.raw_path))
        if (si + 1) % 40 == 0:
            print(f"  [{channel}] preproc {si + 1}/{len(samples)} "
                  f"tiles={len(all_pil)}  t={time.time()-t0:.1f}s", flush=True)

    print(f"  [{channel}] {len(all_pil)} tiles from {len(scan_y)} scans "
          f"(skipped {len(samples)-len(scan_y)})  preproc={time.time()-t0:.1f}s",
          flush=True)
    t1 = time.time()
    X = encoder.encode(all_pil, batch_size=16)
    print(f"  [{channel}] encoded {X.shape} in {time.time()-t1:.1f}s", flush=True)
    return (
        X,
        np.array(tile_to_scan),
        np.array(scan_y),
        np.array(scan_groups),
        scan_paths,
    )


def build_tile_embeddings_rgb(
    samples, encoder,
    tile_size: int = 512, max_tiles: int = 9,
):
    """RGB = (Height, Amplitude Error, Phase) stacked."""
    all_pil = []
    tile_to_scan = []
    scan_y, scan_groups, scan_paths = [], [], []

    t0 = time.time()
    for si, s in enumerate(samples):
        try:
            h_tiles = preprocess_channel_tiles(
                s.raw_path, "Height", tile_size=tile_size, max_tiles=max_tiles,
            )
            a_tiles = preprocess_channel_tiles(
                s.raw_path, "Amplitude Error", tile_size=tile_size, max_tiles=max_tiles,
            )
            try:
                p_tiles = preprocess_channel_tiles(
                    s.raw_path, "Phase", tile_size=tile_size, max_tiles=max_tiles,
                )
            except Exception:
                # Phase missing → fall back to replicating Height for that channel
                # (honest fallback for 13 diabetes files missing Phase)
                p_tiles = h_tiles
        except Exception as e:
            continue
        n = min(len(h_tiles), len(a_tiles), len(p_tiles))
        if n == 0:
            continue
        scan_idx = len(scan_y)
        for k in range(n):
            r = (np.clip(h_tiles[k], 0, 1) * 255).astype(np.uint8)
            g = (np.clip(a_tiles[k], 0, 1) * 255).astype(np.uint8)
            b = (np.clip(p_tiles[k], 0, 1) * 255).astype(np.uint8)
            pil = Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")
            all_pil.append(pil)
            tile_to_scan.append(scan_idx)
        scan_y.append(s.label)
        scan_groups.append(s.person)
        scan_paths.append(str(s.raw_path))
        if (si + 1) % 40 == 0:
            print(f"  [RGB] preproc {si + 1}/{len(samples)} "
                  f"tiles={len(all_pil)}  t={time.time()-t0:.1f}s", flush=True)

    print(f"  [RGB] {len(all_pil)} tiles from {len(scan_y)} scans  "
          f"preproc={time.time()-t0:.1f}s", flush=True)
    t1 = time.time()
    X = encoder.encode(all_pil, batch_size=16)
    print(f"  [RGB] encoded {X.shape} in {time.time()-t1:.1f}s", flush=True)
    return (
        X,
        np.array(tile_to_scan),
        np.array(scan_y),
        np.array(scan_groups),
        scan_paths,
    )


# ---------------------------------------------------------------------------
# LOPO: tile-level LR + mean-of-probs scan aggregation (mirrors baseline)
# ---------------------------------------------------------------------------

def lopo_tile_lr(X, tile_to_scan, scan_y, scan_groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    scan_y = np.asarray(scan_y)
    scan_groups = np.asarray(scan_groups)
    tile_to_scan = np.asarray(tile_to_scan)

    preds = np.full(len(scan_y), -1, dtype=int)
    for tr_scans, va_scans in leave_one_patient_out(scan_groups):
        tr_tiles = np.where(np.isin(tile_to_scan, tr_scans))[0]
        va_tiles = np.where(np.isin(tile_to_scan, va_scans))[0]
        if len(va_tiles) == 0:
            continue
        sc = StandardScaler()
        Xt = sc.fit_transform(X[tr_tiles])
        Xv = sc.transform(X[va_tiles])
        tr_y = scan_y[tile_to_scan[tr_tiles]]
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                                 solver="lbfgs", n_jobs=4)
        clf.fit(Xt, tr_y)
        tile_probs = clf.predict_proba(Xv)
        for si in va_scans:
            m = tile_to_scan[va_tiles] == si
            if m.any():
                preds[si] = tile_probs[m].mean(axis=0).argmax()

    valid = preds >= 0
    f1_macro = f1_score(scan_y[valid], preds[valid], average="macro")
    f1_weighted = f1_score(scan_y[valid], preds[valid], average="weighted")
    per_class = f1_score(scan_y[valid], preds[valid], average=None,
                         labels=list(range(len(CLASSES))))
    return {
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class_f1": per_class.tolist(),
        "n_scans": int(valid.sum()),
        "preds": preds.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["survey", "encode", "eval", "all"], default="all")
    p.add_argument("--cache-dir", default=str(ROOT / "cache"))
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--max-tiles", type=int, default=9)
    p.add_argument("--target-nm", type=float, default=90.0)
    p.add_argument("--dinov2-variant", default="vitb14")
    args = p.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    samples = enumerate_samples(ROOT / "TRAIN_SET")
    print(f"[probe] enumerated {len(samples)} samples")

    # Survey
    if args.mode in ("survey", "all"):
        print("[probe] channel survey ...")
        summary = channel_survey(samples, ROOT / "reports" / "channel_survey.csv")
        print(json.dumps({
            "channel_occurrence": summary["channel_occurrence"],
            "per_class_channel_occurrence": summary["per_class_channel_occurrence"],
            "per_class_total": summary["per_class_total"],
        }, indent=2))
        (ROOT / "reports" / "channel_survey_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        if args.mode == "survey":
            return

    cache_path = cache / (
        f"multichan_tiled_emb_dinov2{args.dinov2_variant}_"
        f"t{args.tile_size}_n{args.max_tiles}.npz"
    )

    # Encode
    if args.mode in ("encode", "all") and not cache_path.exists():
        print(f"[probe] loading DINOv2 {args.dinov2_variant} ...")
        enc = load_dinov2(args.dinov2_variant)
        print(f"[probe]   device={enc.device}  D={enc.embed_dim}")

        all_data = {}
        for key, chan in [("height", "Height"),
                          ("amplitude", "Amplitude Error"),
                          ("phase", "Phase")]:
            print(f"[probe] -- channel={chan} --")
            X, t2s, sy, sg, sp = build_tile_embeddings_singlechan(
                samples, chan, enc,
                tile_size=args.tile_size, max_tiles=args.max_tiles,
            )
            all_data[f"X_{key}"] = X
            all_data[f"t2s_{key}"] = t2s
            all_data[f"sy_{key}"] = sy
            all_data[f"sg_{key}"] = sg
            all_data[f"paths_{key}"] = np.array(sp)

        print("[probe] -- RGB stack (H+A+P) --")
        X, t2s, sy, sg, sp = build_tile_embeddings_rgb(
            samples, enc,
            tile_size=args.tile_size, max_tiles=args.max_tiles,
        )
        all_data["X_rgb"] = X
        all_data["t2s_rgb"] = t2s
        all_data["sy_rgb"] = sy
        all_data["sg_rgb"] = sg
        all_data["paths_rgb"] = np.array(sp)

        np.savez_compressed(cache_path, **all_data)
        print(f"[probe] saved → {cache_path}")
    elif args.mode in ("encode", "all"):
        print(f"[probe] cache hit: {cache_path}")

    # Eval
    if args.mode in ("eval", "all"):
        if not cache_path.exists():
            print(f"[probe] no cache at {cache_path}")
            return
        d = np.load(cache_path, allow_pickle=True)
        results = {}
        for key in ["height", "amplitude", "phase", "rgb"]:
            if f"X_{key}" not in d:
                continue
            X = d[f"X_{key}"]
            t2s = d[f"t2s_{key}"]
            sy = d[f"sy_{key}"]
            sg = d[f"sg_{key}"]
            if X.shape[0] == 0:
                print(f"[eval] {key}: empty")
                continue
            r = lopo_tile_lr(X, t2s, sy, sg)
            print(f"[eval] {key:10s}  macroF1={r['f1_macro']:.4f}  "
                  f"weightedF1={r['f1_weighted']:.4f}  N={r['n_scans']}")
            for ci, cname in enumerate(CLASSES):
                print(f"           {cname:25s} f1={r['per_class_f1'][ci]:.4f}")
            results[key] = r

        # Concat H|A|P at scan level: take mean over tiles per scan per channel,
        # then concat (skipping scans where any channel is missing).
        print("[eval] -- CONCAT(H|A|P) per-scan mean features --")
        def mean_per_scan(X, t2s, n_scans):
            out = np.zeros((n_scans, X.shape[1]), dtype=np.float32)
            for si in range(n_scans):
                m = t2s == si
                if m.any():
                    out[si] = X[m].mean(axis=0)
            return out

        path_to_xH = {}
        path_to_xA = {}
        path_to_xP = {}
        path_to_y = {}
        path_to_g = {}
        for key, path_to_x in [("height", path_to_xH),
                               ("amplitude", path_to_xA),
                               ("phase", path_to_xP)]:
            if f"X_{key}" not in d:
                continue
            X = d[f"X_{key}"]
            t2s = d[f"t2s_{key}"]
            sy = d[f"sy_{key}"]
            sg = d[f"sg_{key}"]
            paths = d[f"paths_{key}"]
            means = mean_per_scan(X, t2s, len(paths))
            for i, p_ in enumerate(paths):
                p_ = str(p_)
                path_to_x[p_] = means[i]
                path_to_y[p_] = int(sy[i])
                path_to_g[p_] = str(sg[i])

        common = [p for p in path_to_xH
                  if p in path_to_xA and p in path_to_xP]
        if common:
            X_c = np.stack([np.concatenate([path_to_xH[p],
                                            path_to_xA[p],
                                            path_to_xP[p]]) for p in common])
            y_c = np.array([path_to_y[p] for p in common])
            g_c = np.array([path_to_g[p] for p in common])

            # Scan-level LR LOPO (one vector per scan)
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import f1_score
            preds = np.full(len(y_c), -1)
            for tr, va in leave_one_patient_out(g_c):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X_c[tr])
                Xva = sc.transform(X_c[va])
                clf = LogisticRegression(max_iter=3000, C=1.0,
                                         class_weight="balanced", solver="lbfgs")
                clf.fit(Xtr, y_c[tr])
                preds[va] = clf.predict(Xva)
            valid = preds >= 0
            f1m = f1_score(y_c[valid], preds[valid], average="macro")
            f1w = f1_score(y_c[valid], preds[valid], average="weighted")
            per_c = f1_score(y_c[valid], preds[valid], average=None,
                             labels=list(range(len(CLASSES))))
            print(f"[eval] concat_HAP  macroF1={f1m:.4f}  weightedF1={f1w:.4f}  "
                  f"N={valid.sum()}")
            for ci, cname in enumerate(CLASSES):
                print(f"           {cname:25s} f1={per_c[ci]:.4f}")
            results["concat_HAP"] = {
                "f1_macro": float(f1m),
                "f1_weighted": float(f1w),
                "per_class_f1": per_c.tolist(),
                "n_scans": int(valid.sum()),
            }

        out_json = ROOT / "reports" / "multichannel_results.json"
        # Drop huge preds arrays from per-channel dicts before saving summary
        compact = {}
        for k, v in results.items():
            vv = dict(v)
            vv.pop("preds", None)
            compact[k] = vv
        out_json.write_text(json.dumps(compact, indent=2))
        print(f"[eval] saved → {out_json}")


if __name__ == "__main__":
    main()
