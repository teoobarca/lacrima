"""Vizualizuj vzorky obrázkov pre každú triedu v TRAIN_SET/.

Uloží jednu mriežku (PNG) pre každú triedu do reports/samples/.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "TRAIN_SET"
OUT = ROOT / "reports" / "samples"
OUT.mkdir(parents=True, exist_ok=True)

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
N_PER_CLASS = 9
SEED = 42


def main() -> None:
    random.seed(SEED)
    classes = sorted([d for d in DATA.iterdir() if d.is_dir()])
    print(f"Found {len(classes)} class directories")

    for cls_dir in classes:
        imgs = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT]
        if not imgs:
            print(f"  [skip] {cls_dir.name}: no images found")
            continue

        pick = random.sample(imgs, min(N_PER_CLASS, len(imgs)))
        n = len(pick)
        cols = min(3, n)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = np.atleast_2d(axes).flatten()

        for ax, path in zip(axes, pick):
            try:
                with Image.open(path) as img:
                    arr = np.array(img)
                if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                    ax.imshow(arr)
                else:
                    ax.imshow(arr, cmap="afmhot")
                ax.set_title(f"{path.name}\n{arr.shape} {arr.dtype}", fontsize=7)
            except Exception as e:
                ax.set_title(f"ERR: {str(e)[:50]}", fontsize=7)
            ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle(f"{cls_dir.name}  (n_total={len(imgs)})", fontsize=12)
        fig.tight_layout()
        out_path = OUT / f"{cls_dir.name}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  [ok]  {cls_dir.name}: {len(imgs)} imgs → {out_path.name}")


if __name__ == "__main__":
    main()
