"""Open-set abstention wrapper.

If the hidden test set has more classes than we trained on (e.g., 9 classes
from the PDF brief vs. 5 in TRAIN_SET), a closed-set classifier will always
map novel inputs to ONE OF its 5 known classes, confidently and wrongly.

Defensive strategy: max-softmax thresholding.
    if max(softmax) < T  ->  predict "UNKNOWN"
    else                  ->  predict argmax class

The threshold T is chosen from OOF predictions on TRAIN_SET.  A common choice
is to set T at the percentile of correct-prediction confidences that we are
willing to tolerate flagging as unknown.  We default to the 10th percentile of
correct-prediction confidences (so ~90% of genuinely-confident correct
predictions still pass through).  See `scripts/eval_open_set.py` for the
honest simulation where we hold out an entire class ("SucheOko") during
training and measure how well the threshold catches it.

This wrapper composes around the shipped v4 multi-scale predictor and does
NOT require retraining any component.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


UNKNOWN_LABEL = "UNKNOWN"


class _BaseProbaPredictor(Protocol):
    classes: list[str]

    def predict_scan(self, path) -> tuple[str, np.ndarray]: ...


@dataclass
class OpenSetPredictor:
    """Wraps a base predictor with max-softmax abstention.

    Parameters
    ----------
    base : any object with `classes: list[str]` and
           `predict_scan(path) -> (label, probs)`
    threshold : float in [0, 1]; if max(softmax) < threshold, emit UNKNOWN.
    unknown_label : str, label to emit for OOD / low-confidence samples.

    Examples
    --------
    >>> from models.ensemble_v4_multiscale.predict import TTAPredictorV4
    >>> base = TTAPredictorV4.load()
    >>> open_clf = OpenSetPredictor(base, threshold=0.55)
    >>> open_clf.predict_scan(path)
    ('UNKNOWN', array([0.28, 0.22, 0.20, 0.19, 0.11]))
    """
    base: object
    threshold: float = 0.50
    unknown_label: str = UNKNOWN_LABEL
    classes: list[str] = field(init=False)

    def __post_init__(self):
        if not hasattr(self.base, "classes"):
            raise ValueError("base predictor must expose `classes: list[str]`")
        if not hasattr(self.base, "predict_scan"):
            raise ValueError("base predictor must expose `predict_scan(path)`")
        self.classes = list(self.base.classes) + [self.unknown_label]

    def predict_scan(self, path: Path | str) -> tuple[str, np.ndarray]:
        _base_label, probs = self.base.predict_scan(path)
        max_p = float(probs.max())
        if max_p < self.threshold:
            return self.unknown_label, probs
        return self.base.classes[int(probs.argmax())], probs

    def predict_directory(self, root: Path | str,
                          **kwargs) -> pd.DataFrame:
        root = Path(root)
        # Most base predictors expose predict_directory; we do NOT rely on it
        # because we need per-scan probs to apply the threshold.  So we walk
        # the directory ourselves using the base's file filter.
        from teardrop.data import is_raw_spm  # lazy to avoid torch import
        rows = []
        all_files = sorted([p for p in root.rglob("*") if is_raw_spm(p)])
        for i, p in enumerate(all_files):
            try:
                cls_pred, probs = self.predict_scan(p)
                row = {"file": str(p.relative_to(root)),
                       "predicted_class": cls_pred,
                       "max_prob": float(probs.max())}
                for c, pr in zip(self.base.classes, probs):
                    row[f"prob_{c}"] = float(pr)
                rows.append(row)
            except Exception as e:
                rows.append({"file": str(p.relative_to(root)),
                             "error": str(e)[:200]})
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(all_files)}] processed")
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Threshold selection utilities (for use from eval scripts).
# ---------------------------------------------------------------------------


def pick_threshold_from_oof(proba: np.ndarray, y_true: np.ndarray,
                            correct_floor_pct: float = 10.0) -> float:
    """Pick max-softmax threshold T from OOF probabilities.

    Set T = percentile(correct-prediction max-probs, `correct_floor_pct`).
    Interpretation: at this T we retain (100 - correct_floor_pct)% of truly
    correct predictions.

    Default 10th percentile means ~90% of correct-confident predictions pass,
    which is a reasonable abstention floor for a 5-class problem that may
    balloon to 9 classes.
    """
    pred = proba.argmax(axis=1)
    correct = pred == y_true
    if correct.sum() == 0:
        return 0.5
    max_p_correct = proba.max(axis=1)[correct]
    return float(np.percentile(max_p_correct, correct_floor_pct))


def max_softmax_auroc(proba_known: np.ndarray,
                      proba_unknown: np.ndarray) -> float:
    """AUROC of `max(softmax)` as a known-vs-unknown discriminator.

    Higher max-softmax should indicate known class.  Returns value in [0, 1];
    0.5 = chance, 1.0 = perfect separation.
    """
    from sklearn.metrics import roc_auc_score
    s_known = proba_known.max(axis=1)
    s_unknown = proba_unknown.max(axis=1)
    # Label: 1 = known (high softmax should be HIGH)
    # We treat max-softmax as a score that separates known (label 1) from
    # unknown (label 0); higher score -> more "known".
    scores = np.concatenate([s_known, s_unknown])
    labels = np.concatenate([
        np.ones(len(s_known), dtype=int),
        np.zeros(len(s_unknown), dtype=int),
    ])
    return float(roc_auc_score(labels, scores))
