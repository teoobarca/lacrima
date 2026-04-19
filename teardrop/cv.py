"""Patient-level cross-validation splitters.

Crucial: TRAIN_SET has 240 scans but only 44 unique patients (SucheOko: only 2!).
Naive image-level KFold would leak the same patient into train+val and inflate F1.

Strategy:
- StratifiedGroupKFold with `groups=patient` ensures each fold's val set has
  patient-disjoint scans.
- For SucheOko (2 patients), with k=5 folds you get 0 SucheOko-val scans in 3 folds.
  Solution: cap k to min(2, smallest_class_n_patients) OR use repeated splits.
- Default: 5 folds, but warn / fallback if SucheOko ends up empty in some folds.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def patient_stratified_kfold(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) with patient-disjoint folds, stratified by class.

    Args:
        labels: shape (n_samples,) integer class labels
        groups: shape (n_samples,) string patient IDs
        n_splits: number of folds
        seed: rng seed
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(np.zeros(len(labels)), labels, groups):
        yield train_idx, val_idx


def leave_one_patient_out(
    groups: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """LOPO: for each unique patient, val = that patient's scans, train = rest.

    Best honest eval for tiny datasets: every scan participates in val exactly once,
    yet patient-level disjoint guaranteed. Cost = #patients model trainings.
    """
    unique_patients = np.unique(groups)
    all_idx = np.arange(len(groups))
    for pat in unique_patients:
        val = all_idx[groups == pat]
        train = all_idx[groups != pat]
        yield train, val


def repeated_patient_kfold(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 5,
    base_seed: int = 42,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Repeat StratifiedGroupKFold n_repeats times with different seeds.

    Yields (repeat_idx, fold_idx, train_idx, val_idx).
    """
    for r in range(n_repeats):
        for fi, (tr, va) in enumerate(
            patient_stratified_kfold(labels, groups, n_splits, seed=base_seed + r)
        ):
            yield r, fi, tr, va


def fold_summary(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    class_names: list[str] | None = None,
) -> str:
    """Print per-fold class & patient distribution for sanity check."""
    lines = [f"=== StratifiedGroupKFold(k={n_splits}, seed={seed}) ==="]
    folds = list(patient_stratified_kfold(labels, groups, n_splits, seed))
    for fi, (tr, va) in enumerate(folds):
        tr_cls = Counter(labels[tr])
        va_cls = Counter(labels[va])
        tr_pat = len(set(groups[tr]))
        va_pat = len(set(groups[va]))
        lines.append(
            f"\nFold {fi}: train={len(tr)} ({tr_pat} patients)  val={len(va)} ({va_pat} patients)"
        )
        all_cls = sorted(set(labels))
        for c in all_cls:
            tr_n = tr_cls.get(c, 0)
            va_n = va_cls.get(c, 0)
            name = class_names[c] if class_names else f"class{c}"
            mark = " <-- EMPTY VAL!" if va_n == 0 else ""
            lines.append(f"  {name:25s} train={tr_n:4d}  val={va_n:4d}{mark}")
    return "\n".join(lines)
