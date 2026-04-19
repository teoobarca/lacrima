"""Generate 5 example clinical reports (one per class) as pitch artifacts.

Shares a single TTAPredictorV4 instance + retrieval index across all 5
scans to amortize the ~30 s cold-start model load.

Run:
    .venv/bin/python scripts/gen_sample_clinical_reports.py

Outputs:
    reports/sample_clinical_reports/{ZdraviLudia,Diabetes,PGOV_Glaukom,
                                     SklerozaMultiplex,SucheOko}.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teardrop.clinical_report import (  # noqa: E402
    DEFAULT_MODEL_DIR, DEFAULT_RETRIEVAL_CACHE, RetrievalIndex,
    generate_clinical_report,
)

DEMO_SAMPLES = {
    "ZdraviLudia": "TRAIN_SET/ZdraviLudia/2L.001",
    "Diabetes": "TRAIN_SET/Diabetes/DM_01.03.2024_LO.001",
    "PGOV_Glaukom": "TRAIN_SET/PGOV_Glaukom/21_LV_PGOV+SII.001",
    "SklerozaMultiplex": "TRAIN_SET/SklerozaMultiplex/1-SM-LM-18.001",
    "SucheOko": "TRAIN_SET/SucheOko/29_PM_suche_oko.001",
}

OUT_DIR = PROJECT_ROOT / "reports" / "sample_clinical_reports"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[gen] Loading TTAPredictorV4 (champion ensemble)...", flush=True)
    from models.ensemble_v4_multiscale.predict import TTAPredictorV4
    t0 = time.time()
    predictor = TTAPredictorV4.load(DEFAULT_MODEL_DIR)
    # Force-load both encoders so the per-scan loop is fast.
    _ = predictor.encoder_dinov2b
    _ = predictor.encoder_biomedclip
    print(f"[gen] Predictor + encoders ready in {time.time()-t0:.1f} s", flush=True)

    print("[gen] Loading retrieval index...", flush=True)
    t0 = time.time()
    retrieval_index = RetrievalIndex.load(DEFAULT_RETRIEVAL_CACHE)
    print(
        f"[gen] Retrieval index: "
        f"{'loaded' if retrieval_index is not None else 'UNAVAILABLE'} "
        f"in {time.time()-t0:.1f} s",
        flush=True,
    )

    for cls, rel in DEMO_SAMPLES.items():
        path = PROJECT_ROOT / rel
        if not path.exists():
            print(f"[gen] SKIP {cls}: missing file {path}", flush=True)
            continue
        print(f"[gen] Generating report for {cls} <- {rel}", flush=True)
        t0 = time.time()
        md = generate_clinical_report(
            scan_path=path,
            _predictor=predictor,
            _retrieval_index=retrieval_index,
        )
        dt = time.time() - t0
        out_path = OUT_DIR / f"{cls}.md"
        out_path.write_text(md)
        print(f"[gen]   -> {out_path.relative_to(PROJECT_ROOT)} "
              f"({len(md):,} chars, {dt:.1f} s)", flush=True)

    print("[gen] Done.", flush=True)


if __name__ == "__main__":
    main()
