"""Smoke test: predict one training scan per class with v4 bundle."""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from teardrop.data import CLASSES, is_raw_spm  # noqa: E402

bundle = ROOT / "models" / "ensemble_v4_multiscale"
spec = importlib.util.spec_from_file_location("v4_predict", bundle / "predict.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

p = mod.TTAPredictorV4.load(bundle)
print(f"[loaded] classes={p.classes}")

targets = []
for cls in CLASSES:
    root = ROOT / "TRAIN_SET" / cls
    raws = sorted(pth for pth in root.rglob("*") if is_raw_spm(pth))
    if not raws:
        print(f"  [WARN] no raw scans found under {root}")
        continue
    targets.append((cls, raws[0]))

n_correct = 0
for true_cls, path in targets:
    t0 = time.time()
    pred_cls, probs = p.predict_scan(path)
    dt = time.time() - t0
    ok = (pred_cls == true_cls)
    n_correct += int(ok)
    top = sorted(zip(p.classes, probs), key=lambda kv: -kv[1])[:3]
    top_str = ", ".join(f"{c}={pr:.3f}" for c, pr in top)
    print(f"  [{'OK ' if ok else 'miss'}] true={true_cls:20s} pred={pred_cls:20s} "
          f"({dt:.1f}s)  top3: {top_str}")

print(f"\nSmoke test: {n_correct}/{len(targets)} matched true class (training data).")
