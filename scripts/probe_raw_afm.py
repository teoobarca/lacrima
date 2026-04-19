"""Prejdi všetky raw Bruker SPM súbory a zisti ich geometriu (tvar, pixel->nm, rozsah výšky).

Výstup: reports/raw_afm_probe.csv + sumár na stdout.
"""
from __future__ import annotations

import csv
import io
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

# Silence AFMReader INFO spam
logging.getLogger().setLevel(logging.WARNING)

from AFMReader.spm import load_spm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "TRAIN_SET"
OUT = ROOT / "reports" / "raw_afm_probe.csv"
OUT.parent.mkdir(exist_ok=True)


def is_raw_afm(p: Path) -> bool:
    if not p.is_file():
        return False
    suf = p.suffix
    return bool(suf and suf[1:].isdigit())


def probe(path: Path) -> dict:
    row = {
        "class": path.parent.name,
        "file": path.name,
        "bytes": path.stat().st_size,
        "status": "",
        "H": "", "W": "", "dtype": "",
        "z_min_nm": "", "z_max_nm": "", "z_mean_nm": "", "z_std_nm": "",
        "pixel_to_nm": "", "scan_um": "",
        "channel": "Height",
    }
    try:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            arr, px_nm = load_spm(file_path=path, channel="Height")
        row["status"] = "ok"
        row["H"] = arr.shape[0]
        row["W"] = arr.shape[1]
        row["dtype"] = str(arr.dtype)
        row["z_min_nm"] = f"{float(arr.min()):.3f}"
        row["z_max_nm"] = f"{float(arr.max()):.3f}"
        row["z_mean_nm"] = f"{float(arr.mean()):.3f}"
        row["z_std_nm"] = f"{float(arr.std()):.3f}"
        row["pixel_to_nm"] = f"{px_nm:.3f}"
        row["scan_um"] = f"{arr.shape[0] * px_nm / 1000:.2f}"
    except Exception as e:
        row["status"] = f"err:{type(e).__name__}:{str(e)[:80]}"
    return row


def main() -> None:
    raw_files = [p for p in DATA.rglob("*") if is_raw_afm(p)]
    print(f"Found {len(raw_files)} raw AFM candidate files")

    rows = []
    for i, f in enumerate(raw_files, 1):
        row = probe(f)
        rows.append(row)
        if i % 20 == 0 or i == len(raw_files):
            print(f"  [{i}/{len(raw_files)}] {f.name}: {row['status']} "
                  f"{row.get('H', '')}x{row.get('W', '')} "
                  f"{row.get('scan_um', '')}um")

    keys = list(rows[0].keys())
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {OUT}")

    # Summary
    from collections import Counter
    ok_rows = [r for r in rows if r["status"] == "ok"]
    err_rows = [r for r in rows if r["status"] != "ok"]
    print(f"\nOK: {len(ok_rows)}   Errors: {len(err_rows)}")
    if err_rows:
        err_types = Counter(r["status"].split(":")[1] if ":" in r["status"] else r["status"]
                            for r in err_rows)
        print("Error types:", dict(err_types))

    shapes = Counter((int(r["H"]), int(r["W"])) for r in ok_rows)
    print(f"\nShape distribution:")
    for sh, n in shapes.most_common():
        print(f"  {sh[0]}x{sh[1]}: {n}")

    scans = Counter(f"{float(r['scan_um']):.1f}" for r in ok_rows)
    print(f"\nScan size distribution (μm):")
    for s, n in scans.most_common():
        print(f"  {s} μm: {n}")

    print(f"\nPer-class OK / total:")
    cls_total = Counter(r["class"] for r in rows)
    cls_ok = Counter(r["class"] for r in ok_rows)
    for c in cls_total:
        print(f"  {c:30s} {cls_ok[c]:3d} / {cls_total[c]:3d}")


if __name__ == "__main__":
    main()
