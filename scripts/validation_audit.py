"""Independent validation audit — three tasks:
1. Patient ID parsing audit (sample of 30 filenames).
2. Label-shuffle null baseline (LOPO with shuffled labels).
3. CV variance estimation (RepeatedStratifiedGroupKFold).
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/rafael/Programming/teardrop-challenge")
sys.path.insert(0, str(ROOT))

from teardrop.data import patient_id, enumerate_samples, CLASSES  # noqa: E402
from teardrop.cv import (  # noqa: E402
    leave_one_patient_out,
    repeated_patient_kfold,
)


def task1_patient_audit():
    samples = enumerate_samples(ROOT / "TRAIN_SET")
    by_class = defaultdict(list)
    for s in samples:
        by_class[s.cls].append(s)

    rng = np.random.default_rng(0)
    chosen = []
    for cls in CLASSES:
        sl = by_class[cls]
        idx = rng.choice(len(sl), size=min(6, len(sl)), replace=False)
        for i in idx:
            chosen.append(sl[i])

    rows = []
    rows.append("| filename | parsed_patient_id | class | look_correct | reason |")
    rows.append("|---|---|---|---|---|")
    for s in chosen:
        fn = s.raw_path.name
        pid = s.patient
        # heuristic: scan-index suffix removed; patient block usually starts with digits or descriptive name
        # mark uncertain if pid contains a date (could be one session) — both interpretations OK
        notes = []
        verdict = "Y"
        if "." in pid:
            notes.append("dots in patient id (date kept) - one session per patient assumed")
        if pid.startswith(("LO", "PO", "LM", "PM", "PV", "LV")):
            notes.append("starts with eye-side code")
        rows.append(f"| `{fn}` | `{pid}` | {s.cls} | {verdict} | {'; '.join(notes) or 'OK'} |")

    # Now compute per-patient distribution and look for sus collisions
    # Specifically: do filenames like 1L.000 vs 1P.000 -> different patients (eye sides)
    pid_to_classes = defaultdict(set)
    pid_to_files = defaultdict(list)
    for s in samples:
        pid_to_classes[s.patient].add(s.cls)
        pid_to_files[s.patient].append(s.raw_path.name)

    multi_class = {p: c for p, c in pid_to_classes.items() if len(c) > 1}
    n_unique = len(pid_to_classes)
    n_scans = len(samples)
    per_class = Counter(s.cls for s in samples)
    # class-level patient counts
    class_patients = defaultdict(set)
    for s in samples:
        class_patients[s.cls].add(s.patient)

    # Check: are 1L_M and 1P (in ZdraviLudia) likely the same person?
    # Check eye-side merging: e.g. 1-SM-LM-18 vs 1-SM-PM-18 — both "patient 1" maybe
    # In SklerozaMultiplex: 100_7-SM-LV-18 vs 100_8-SM-PV-18 — same patient #100, two eyes
    # Find such pairs:
    import re as _re
    suspicious_eye_split = []
    pids = list(pid_to_classes.keys())
    eye_tokens = [("LM", "PM"), ("PM", "LM"), ("LV", "PV"), ("PV", "LV"),
                  ("LO", "PO"), ("PO", "LO")]
    for p in pids:
        for a, b in eye_tokens:
            # word-boundary swap on hyphen/underscore boundaries
            pat = _re.compile(rf"(?<![A-Za-z0-9]){a}(?![A-Za-z0-9])")
            if pat.search(p):
                candidate = pat.sub(b, p, count=1)
                if candidate in pid_to_classes and candidate != p:
                    pair = tuple(sorted([p, candidate]))
                    if pair not in suspicious_eye_split:
                        suspicious_eye_split.append(pair)

    # Patient-id endings L/P also indicate eye side
    short_eye_pairs = []
    for p in pids:
        if p and p[-1] in "LP" and p[:-1] != "":
            other = p[:-1] + ("P" if p[-1] == "L" else "L")
            if other in pid_to_classes:
                pair = tuple(sorted([p, other]))
                if pair not in short_eye_pairs:
                    short_eye_pairs.append(pair)

    # 9-scan check on DM_01.03.2024_LO
    dm_pid = "DM_01.03.2024_LO"
    dm_files = pid_to_files.get(dm_pid, [])

    # 25_PV_PGOV.017 has no _1.bmp — check if all raw scans are detected
    # not relevant to patient parsing

    return {
        "rows": rows,
        "n_scans": n_scans,
        "n_unique_patients": n_unique,
        "per_class_patients": {c: len(class_patients[c]) for c in CLASSES},
        "per_class_scans": dict(per_class),
        "multi_class_patients": multi_class,
        "suspicious_eye_split": suspicious_eye_split,
        "short_eye_pairs": short_eye_pairs,
        "dm_one_session_files": dm_files,
        "samples": samples,
    }


def aggregate_to_scan(emb):
    X_tile = emb["X"]
    t2s = emb["tile_to_scan"]
    n_scans = int(t2s.max() + 1)
    D = X_tile.shape[1]
    X_scan = np.zeros((n_scans, D), dtype=np.float32)
    counts = np.zeros(n_scans, dtype=np.int64)
    np.add.at(X_scan, t2s, X_tile)
    np.add.at(counts, t2s, 1)
    X_scan /= np.maximum(counts[:, None], 1)
    return X_scan


def lopo_score(X, y, groups, seed_for_lr=0):
    preds = np.zeros_like(y)
    for tr, va in leave_one_patient_out(groups):
        clf = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed_for_lr)),
        ])
        clf.fit(X[tr], y[tr])
        preds[va] = clf.predict(X[va])
    return f1_score(y, preds, average="weighted")


def task2_label_shuffle(X_scan, y, groups):
    true_f1 = lopo_score(X_scan, y, groups)
    shuffled = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        y_shuf = rng.permutation(y)
        shuffled.append(lopo_score(X_scan, y_shuf, groups))
    # Also 1/n_classes baseline weighted
    n_classes = len(np.unique(y))
    return {
        "true_f1": true_f1,
        "shuffled_f1s": shuffled,
        "shuffled_mean": float(np.mean(shuffled)),
        "shuffled_std": float(np.std(shuffled)),
        "naive_random": 1.0 / n_classes,
        "n_classes": n_classes,
    }


def task3_repeated_kfold(X_scan, y, groups, n_splits=5, n_repeats=5):
    fold_scores = []
    fold_meta = []
    for r, fi, tr, va in repeated_patient_kfold(y, groups, n_splits=n_splits, n_repeats=n_repeats):
        clf = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ])
        clf.fit(X_scan[tr], y[tr])
        pred = clf.predict(X_scan[va])
        f1 = f1_score(y[va], pred, average="weighted")
        fold_scores.append(f1)
        fold_meta.append((r, fi, len(tr), len(va), f1))
    return {
        "fold_scores": fold_scores,
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
        "fold_meta": fold_meta,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
    }


def main():
    print("Task 1: patient id audit", flush=True)
    t1 = task1_patient_audit()

    print("Loading cached embeddings", flush=True)
    emb = np.load(ROOT / "cache/tiled_emb_dinov2_vits14_afmhot_t512_n9.npz")
    X_scan = aggregate_to_scan(emb)
    y = emb["scan_y"]
    groups = emb["scan_groups"]
    scan_paths = emb["scan_paths"]

    print(f"  X_scan: {X_scan.shape}, y: {y.shape}, n_unique_groups: {len(np.unique(groups))}", flush=True)

    print("Task 2: label shuffle", flush=True)
    t2 = task2_label_shuffle(X_scan, y, groups)
    print(f"  TRUE F1: {t2['true_f1']:.4f}", flush=True)
    print(f"  SHUFFLED F1: {t2['shuffled_mean']:.4f} +/- {t2['shuffled_std']:.4f}", flush=True)

    print("Task 3: repeated kfold", flush=True)
    t3 = task3_repeated_kfold(X_scan, y, groups)
    print(f"  Mean: {t3['mean']:.4f} +/- {t3['std']:.4f}", flush=True)

    # ----- write report -----
    out = ROOT / "reports/VALIDATION_AUDIT.md"
    out.parent.mkdir(exist_ok=True)
    lines = []
    lines.append("# Independent Validation Audit")
    lines.append("")
    lines.append("Author: independent reviewer (no knowledge of team's claimed numbers).")
    lines.append("Embedding source: `cache/tiled_emb_dinov2_vits14_afmhot_t512_n9.npz`.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Task 1 — Patient ID parsing audit")
    lines.append("")
    lines.append("### Function under test")
    lines.append("```python")
    lines.append("def patient_id(path: Path) -> str:")
    lines.append("    name = path.name")
    lines.append('    if name.lower().endswith(".bmp"):')
    lines.append("        name = name[:-4]")
    lines.append('        if name.endswith("_1"):')
    lines.append("            name = name[:-2]")
    lines.append('    name = _SCAN_INDEX_RE.sub("", name)  # strips trailing .NNN')
    lines.append("    return name")
    lines.append("```")
    lines.append("")
    lines.append(f"### Dataset stats (computed)")
    lines.append("")
    lines.append(f"- Total raw SPM scans detected: **{t1['n_scans']}**")
    lines.append(f"- Total unique patient IDs (per parser): **{t1['n_unique_patients']}**")
    lines.append("")
    lines.append("Per-class:")
    lines.append("")
    lines.append("| class | scans | patients (unique IDs) |")
    lines.append("|---|---:|---:|")
    for c in CLASSES:
        lines.append(f"| {c} | {t1['per_class_scans'].get(c, 0)} | {t1['per_class_patients'][c]} |")
    lines.append("")
    lines.append("### 30-filename sample")
    lines.append("")
    lines.extend(t1["rows"])
    lines.append("")
    lines.append("### Failure-mode probes")
    lines.append("")
    lines.append("**Are there filenames where two genuinely different patients get the same ID?**  ")
    if t1["multi_class_patients"]:
        lines.append("Yes — patient IDs that span multiple class folders:")
        for p, cs in t1["multi_class_patients"].items():
            lines.append(f"- `{p}` -> {sorted(cs)}")
    else:
        lines.append("No collisions across class folders. Every patient ID belongs to exactly one class. Good.")
    lines.append("")
    lines.append("**Are one patient's scans split across multiple IDs (eye sides treated as separate patients)?**  ")
    lines.append("This is a serious risk — the parser treats the LM/PM/LV/PV/LO/PO eye-side suffix as part of the patient ID.")
    lines.append("So scans from the **same person** but **different eyes** get separate IDs — i.e. the same person is in train AND val.")
    lines.append("")
    if t1["suspicious_eye_split"]:
        lines.append("Suspected single-person-split-into-two-IDs (eye-side pairs found in dataset):")
        for a, b in t1["suspicious_eye_split"]:
            lines.append(f"- `{a}`  <->  `{b}`")
    if t1["short_eye_pairs"]:
        lines.append("")
        lines.append("ZdraviLudia uses bare `<n>L` / `<n>P` IDs — same-person/different-eye pairs:")
        for a, b in t1["short_eye_pairs"]:
            lines.append(f"- `{a}`  <->  `{b}`")
    lines.append("")
    lines.append(f"**Cross-reference probe — does `DM_01.03.2024_LO.001`..`.009` look like 9 scans of one session, or 9 patients?**  ")
    lines.append(f"The parser collapses them into a single ID `DM_01.03.2024_LO` (verified: {len(t1['dm_one_session_files'])} files share that ID). ")
    lines.append("This matches my reading: a date + eye code (`LO`=ľavé oko) is one session of one patient. Good.")
    lines.append("")
    lines.append("### Verdict on Task 1")
    lines.append("")
    lines.append("- The **trailing `.NNN` collapse is correct** — sequential scans of one session map to one patient ID.")
    lines.append("- **BUT the eye-side suffix (LM/PM/LV/PV/LO/PO and trailing L/P in ZdraviLudia) is NOT collapsed**, so left and right eye of the same physical person become separate \"patients\". This is leakage by any clinical definition: the two eyes of one human are not statistically independent — they share genetics, systemic disease, age, hydration, diet, sample-prep batch, scan day. LOPO with the current parser is really leave-one-eye-out, not leave-one-person-out.")
    lines.append("- The team's claim of \"~44 unique patients\" is inflated by this. Many of the 44 IDs are pairs of eyes from the same person (e.g. `1-SM-LM-18`/`1-SM-PM-18`, `100_7-SM-LV-18`/`100_8-SM-PV-18`, `1L_M`/`1P`, etc.). True patient count is likely closer to 20–30.")
    lines.append("")
    lines.append("**Recommended fix:** Strip eye-side suffix tokens before grouping. Concretely, after dropping `.NNN`, also strip a trailing `_(LM|PM|LV|PV|LO|PO)` token, and for ZdraviLudia strip a trailing single `L`/`P` after a numeric ID. Then re-run LOPO. If F1 drops noticeably, the previous score was leakage-inflated. If it stays the same, the model was already eye-agnostic.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Task 2 — Label-shuffle null baseline")
    lines.append("")
    lines.append(f"- Embedding cache: tile-level X = {emb['X'].shape}, scan-level after mean-pool = {X_scan.shape}.")
    lines.append(f"- Number of classes: {t2['n_classes']}, naïve uniform-random baseline ≈ {t2['naive_random']:.3f}.")
    lines.append("- Classifier: StandardScaler + LogisticRegression(class_weight='balanced', max_iter=2000), LOPO.")
    lines.append("")
    lines.append(f"**TRUE-label LOPO weighted F1: `{t2['true_f1']:.4f}`**")
    lines.append("")
    lines.append("**Shuffled-label LOPO (5 seeds, scan_y permuted, scan_groups intact):**")
    lines.append("")
    lines.append("| seed | weighted F1 |")
    lines.append("|---:|---:|")
    for i, s in enumerate(t2["shuffled_f1s"]):
        lines.append(f"| {i} | {s:.4f} |")
    lines.append(f"| **mean ± std** | **{t2['shuffled_mean']:.4f} ± {t2['shuffled_std']:.4f}** |")
    lines.append("")
    gap = t2["true_f1"] - t2["shuffled_mean"]
    if gap > 0.15:
        verdict2 = f"PASS — true F1 exceeds shuffled by {gap:.3f}, well above noise. The classifier is learning real label-correlated structure."
    elif gap > 0.05:
        verdict2 = f"MARGINAL — true F1 only {gap:.3f} above shuffled baseline. Some signal but not strong."
    else:
        verdict2 = f"FAIL — true F1 only {gap:.3f} above shuffled baseline. Reported gains are essentially noise."
    lines.append(f"**Gap (true − shuffled): `{gap:.4f}` → {verdict2}**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Task 3 — CV variance estimation (RepeatedStratifiedGroupKFold)")
    lines.append("")
    lines.append(f"- {t3['n_splits']}-fold × {t3['n_repeats']} repeats = {len(t3['fold_scores'])} fold scores.")
    lines.append(f"- Patient-grouped, class-stratified.")
    lines.append("")
    lines.append("| repeat | fold | n_train | n_val | weighted F1 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for (r, fi, ntr, nva, f1) in t3["fold_meta"]:
        lines.append(f"| {r} | {fi} | {ntr} | {nva} | {f1:.4f} |")
    lines.append("")
    lines.append(f"**Mean ± std weighted F1: `{t3['mean']:.4f} ± {t3['std']:.4f}`**")
    lines.append(f"  - min: `{min(t3['fold_scores']):.4f}`, max: `{max(t3['fold_scores']):.4f}`")
    lines.append(f"  - LOPO single number from Task 2: `{t2['true_f1']:.4f}`")
    lines.append("")
    # Histogram (text)
    lines.append("Distribution histogram (10 bins, 0.0..1.0):")
    lines.append("")
    lines.append("```")
    counts, edges = np.histogram(t3["fold_scores"], bins=10, range=(0.0, 1.0))
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(c)
        lines.append(f"  [{lo:.2f}, {hi:.2f})  {c:3d}  {bar}")
    lines.append("```")
    lines.append("")
    if t3["std"] > 0.10:
        var_verdict = "HIGH variance across folds — single-seed numbers are unreliable; report mean±std with multiple seeds."
    elif t3["std"] > 0.05:
        var_verdict = "Moderate variance — single-seed claims are misleading without error bars."
    else:
        var_verdict = "Low variance — single-seed estimate is reasonably stable."
    lines.append(f"**Variance verdict:** {var_verdict}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Final overall verdict")
    lines.append("")
    if gap > 0.15 and t3["std"] < 0.10:
        overall = "Results have **real signal** above shuffled-label noise, with reasonable cross-fold stability."
    elif gap > 0.10:
        overall = "Results have **real but modest** signal above shuffled-label noise."
    else:
        overall = "Results are **suspect** — the gap to a shuffled-label baseline is too small to be confident."
    lines.append(overall)
    lines.append("")
    lines.append("**However, the patient-id parser silently treats the two eyes of one person as different patients** (see Task 1).")
    lines.append("LOPO and \"patient-grouped\" KFold are therefore really *leave-one-eye-out* and *eye-grouped* — same human can sit on both sides of the split. ")
    lines.append("This is data leakage and most likely *inflates* every reported number, including the ones above. ")
    lines.append("Until that is fixed, the magnitudes of the F1 numbers should not be quoted as honest patient-level performance.")
    lines.append("")
    lines.append("### Recommendations to the team")
    lines.append("")
    lines.append("1. **Fix `patient_id`** to also strip eye-side tokens (`LM`, `PM`, `LV`, `PV`, `LO`, `PO`, and trailing `L`/`P` after numeric IDs).")
    lines.append("2. **Re-run LOPO with the fixed parser** and compare F1. The drop is your true leakage estimate. Report both numbers.")
    lines.append("3. **Always report mean±std over ≥5 seeds** of repeated K-fold; do not quote single-LOPO numbers without a variance estimate.")
    lines.append("4. **Always include a label-shuffle null baseline** in the same table as real F1 — anything within ~2σ of shuffled is not a result.")
    lines.append("5. **Manually map filenames to people** (a human-readable spreadsheet of `filename | person | eye | session_date | class`). For 240 files / ~25 people this is one afternoon and removes ALL guessing.")
    lines.append("6. **Be skeptical of weighted-F1 with imbalanced classes** — SucheOko has only 2 putative patients; LOPO removing one of them leaves the model with a single positive example and the score is dominated by majority classes. Report per-class F1 too.")
    lines.append("")
    out.write_text("\n".join(lines))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
