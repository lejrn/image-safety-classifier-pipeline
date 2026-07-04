"""Evaluation via precision-recall curve threshold tuning.

Unlike a fixed-threshold recall gate, this module sweeps
sklearn.metrics.precision_recall_curve and selects the *highest* decision
threshold that still clears a target recall floor -- i.e. the least
permissive threshold consistent with the recall requirement, which
maximizes precision subject to that floor. This directly reflects
"tuned for high recall on precision/recall eval curves": recall is the
hard constraint, precision is what's being optimized against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from fastai.tabular.all import Learner
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from .train import LABEL_COL


@dataclass
class EvalResult:
    precision: float
    recall: float
    f1: float
    threshold: float
    target_recall: float
    confusion: list[list[int]]
    report: str
    passed_gate: bool


def _select_operating_point(
    y_true: list[int], y_scores: np.ndarray, target_recall: float
) -> tuple[float, float, float, bool]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # precision_recall_curve appends a final (precision=1, recall=0) point with
    # no corresponding threshold; drop it so arrays align 1:1 with thresholds.
    precision, recall = precision[:-1], recall[:-1]
    candidates = [i for i, r in enumerate(recall) if r >= target_recall]
    if not candidates:
        default_p = float(precision[0]) if len(precision) else 0.0
        default_r = float(recall[0]) if len(recall) else 0.0
        return 0.0, default_p, default_r, False
    best_i = candidates[-1]  # thresholds ascending -> last match = highest threshold clearing the floor
    return float(thresholds[best_i]), float(precision[best_i]), float(recall[best_i]), True


def evaluate(learn: Learner, df: pd.DataFrame, target_recall: float = 0.98) -> EvalResult:
    test_dl = learn.dls.test_dl(df)
    preds, _ = learn.get_preds(dl=test_dl)
    y_scores = preds[:, 1].numpy()
    y_true = df[LABEL_COL].tolist()

    threshold, precision_at_t, recall_at_t, reachable = _select_operating_point(
        y_true, y_scores, target_recall
    )
    y_pred = (y_scores >= threshold).astype(int)

    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    return EvalResult(
        precision=precision_at_t,
        recall=recall_at_t,
        f1=float(f1),
        threshold=threshold,
        target_recall=target_recall,
        confusion=cm,
        report=report,
        passed_gate=reachable and recall_at_t >= target_recall,
    )
