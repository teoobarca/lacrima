"""Data audit pre Hack Košice 2026 — TRAIN_SET.

Spustiť po extrakcii TRAIN_SET.zip. Napíše report na stdout + JSON na disk.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "TRAIN_SET"
REPORT = ROOT / "reports" / "data_audit.json"
REPORT.parent.mkdir(exist_ok=True)


def walk_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            yield p


def classify_path(p: Path, data_root: Path) -> str:
    rel = p.relative_to(data_root)
    parts = rel.parts
    return parts[0] if len(parts) > 1 else "(root)"


def file_extension(p: Path) -> str:
    return p.suffix.lower() if p.suffix else "(none)"


def inspect_image(p: Path) -> dict | None:
    try:
        with Image.open(p) as img:
            arr = np.array(img)
        return {
            "size": img.size,
            "mode": img.mode,
            "dtype": str(arr.dtype),
            "shape": arr.shape,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def sniff_binary_header(p: Path, n: int = 256) -> str:
    with open(p, "rb") as f:
        header = f.read(n)
    return header[:64].hex() + " | " + "".join(
        chr(b) if 32 <= b < 127 else "." for b in header[:64]
    )


def main() -> None:
    if not DATA.exists():
        print(f"ERROR: {DATA} not found. Extract TRAIN_SET.zip first.")
        sys.exit(1)

    files = list(walk_files(DATA))
    total_bytes = sum(p.stat().st_size for p in files)

    ext_counter: Counter[str] = Counter()
    class_counter: Counter[str] = Counter()
    class_ext: dict[str, Counter[str]] = defaultdict(Counter)
    class_sizes_bytes: dict[str, int] = defaultdict(int)

    for p in files:
        ext = file_extension(p)
        cls = classify_path(p, DATA)
        ext_counter[ext] += 1
        class_counter[cls] += 1
        class_ext[cls][ext] += 1
        class_sizes_bytes[cls] += p.stat().st_size

    print(f"\n=== TRAIN_SET overview ===")
    print(f"Root: {DATA}")
    print(f"Total files: {len(files)}")
    print(f"Total size:  {total_bytes / 1024 / 1024:.1f} MiB "
          f"({total_bytes / 1024 / 1024 / 1024:.2f} GiB)")

    print(f"\n=== Extensions (global) ===")
    for ext, n in ext_counter.most_common():
        print(f"  {ext:12s} {n:6d}")

    print(f"\n=== Per-class counts ===")
    for cls, n in sorted(class_counter.items(), key=lambda x: -x[1]):
        size_mb = class_sizes_bytes[cls] / 1024 / 1024
        exts_str = ", ".join(f"{e}:{c}" for e, c in class_ext[cls].most_common(5))
        print(f"  {cls:40s} {n:5d} files  {size_mb:8.1f} MiB  [{exts_str}]")

    samples_per_class = {}
    for cls in class_counter:
        cls_files = [p for p in files if classify_path(p, DATA) == cls]
        samples_per_class[cls] = [str(p.relative_to(DATA)) for p in cls_files[:5]]

    print(f"\n=== Sample files per class (first 3) ===")
    for cls, samples in samples_per_class.items():
        print(f"  [{cls}]")
        for s in samples[:3]:
            print(f"    {s}")

    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_samples = [p for p in files if file_extension(p) in image_extensions]
    binary_samples = [p for p in files if file_extension(p) not in image_extensions]

    print(f"\n=== Image file probe (first {min(5, len(image_samples))}) ===")
    for p in image_samples[:5]:
        info = inspect_image(p)
        print(f"  {p.relative_to(DATA)}: {info}")

    print(f"\n=== Binary file header probe (first {min(5, len(binary_samples))}) ===")
    for p in binary_samples[:5]:
        print(f"  {p.relative_to(DATA)} ({p.stat().st_size} B):")
        print(f"    {sniff_binary_header(p)}")

    report = {
        "root": str(DATA),
        "total_files": len(files),
        "total_bytes": total_bytes,
        "extensions": dict(ext_counter),
        "class_counts": dict(class_counter),
        "class_extensions": {c: dict(e) for c, e in class_ext.items()},
        "class_sizes_bytes": dict(class_sizes_bytes),
        "samples_per_class": samples_per_class,
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report: {REPORT}")


if __name__ == "__main__":
    main()
