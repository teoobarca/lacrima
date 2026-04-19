"""Vizualizuj raw AFM height matrices per class vs. ich BMP náprotivky.

Uloží 4-column grid (raw height colormap + BMP) pre 3 vzorky z každej triedy.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logging.getLogger().setLevel(logging.WARNING)

from AFMReader.spm import load_spm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "TRAIN_SET"
OUT = ROOT / "reports" / "raw_samples"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(0)
N_PER_CLASS = 3


def is_raw(p: Path) -> bool:
    return p.is_file() and p.suffix and p.suffix[1:].isdigit()


def level_plane(h: np.ndarray) -> np.ndarray:
    """Subtract a 1st-order polynomial fit — removes scanner tilt."""
    y, x = np.mgrid[0:h.shape[0], 0:h.shape[1]]
    A = np.column_stack([np.ones(x.size), x.ravel(), y.ravel()])
    coef, *_ = np.linalg.lstsq(A, h.ravel(), rcond=None)
    plane = (coef[0] + coef[1] * x + coef[2] * y).reshape(h.shape)
    return h - plane


def main() -> None:
    classes = sorted(d for d in DATA.iterdir() if d.is_dir())
    for cls_dir in classes:
        raw_files = [p for p in cls_dir.iterdir() if is_raw(p)]
        pick = random.sample(raw_files, min(N_PER_CLASS, len(raw_files)))

        fig, axes = plt.subplots(len(pick), 3, figsize=(13, 4 * len(pick)))
        if len(pick) == 1:
            axes = axes[None, :]

        for r, rawp in enumerate(pick):
            try:
                h, px_nm = load_spm(file_path=rawp, channel="Height")
            except Exception as e:
                axes[r, 0].set_title(f"ERR {rawp.name}: {e}", fontsize=8)
                continue

            h_level = level_plane(h)
            scan_um = h.shape[0] * px_nm / 1000

            p2, p98 = np.percentile(h_level, [2, 98])
            ax = axes[r, 0]
            ax.imshow(np.clip(h_level, p2, p98), cmap="afmhot")
            ax.set_title(
                f"{rawp.name}\nraw {h.shape}  scan {scan_um:.1f}μm\n"
                f"Δz [{p2:.0f}..{p98:.0f}] nm  px={px_nm:.1f}nm",
                fontsize=8,
            )
            ax.axis("off")

            bmp_candidates = list(rawp.parent.glob(f"{rawp.stem}{rawp.suffix}_1.bmp"))
            if bmp_candidates:
                bmp = np.array(Image.open(bmp_candidates[0]))
                axes[r, 1].imshow(bmp)
                axes[r, 1].set_title(f"BMP preview {bmp.shape}", fontsize=8)
            else:
                axes[r, 1].set_title("NO BMP", fontsize=8)
            axes[r, 1].axis("off")

            axes[r, 2].hist(h_level.flatten(), bins=80, color="C0")
            axes[r, 2].set_title("Height histogram", fontsize=8)
            axes[r, 2].axvline(p2, color="r", lw=0.5)
            axes[r, 2].axvline(p98, color="r", lw=0.5)

        fig.suptitle(cls_dir.name, fontsize=14)
        fig.tight_layout()
        out = OUT / f"{cls_dir.name}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  {cls_dir.name}: {len(pick)} samples → {out.name}")


if __name__ == "__main__":
    main()
