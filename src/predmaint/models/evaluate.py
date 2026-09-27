"""Metrics.

RMSE alone is the wrong scorecard for maintenance. Predicting failure 20 cycles
too late grounds an aircraft; predicting it 20 cycles too early costs one early
inspection. The NASA C-MAPSS scoring function encodes exactly that asymmetry and
is the number this project optimises for.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Official C-MAPSS asymmetric score. Lower is better.

    d = predicted - actual
    d < 0 (too late, engine fails before we predicted):  exp(-d / 13) - 1
    d >= 0 (too early, we service a healthy engine):     exp( d / 10) - 1

    The steeper 1/13 branch makes a late prediction cost roughly twice an early
    one of the same magnitude.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalty = np.where(d < 0, np.expm1(-d / 13.0), np.expm1(d / 10.0))
    return float(np.sum(penalty))


def regression_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RUL regression scorecard."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "nasa_score": nasa_score(y_true, y_pred),
        "nasa_score_per_engine": nasa_score(y_true, y_pred) / max(len(y_true), 1),
        "late_prediction_rate": float(np.mean(errors > 0)),
        "mean_late_error": float(errors[errors > 0].mean()) if (errors > 0).any() else 0.0,
        "n": int(len(y_true)),
    }


def classification_report_dict(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Failure-risk scorecard.

    Recall is the headline: a missed at-risk engine is the expensive error.
    PR-AUC is reported because the positive class is a minority.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_hat = (y_prob >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2
    return {
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "n": int(len(y_true)),
    }
