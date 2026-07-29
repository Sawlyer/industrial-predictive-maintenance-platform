"""The scoring function encodes the business rule, so it gets its own tests."""

from __future__ import annotations

import numpy as np

from predmaint.models.evaluate import classification_report_dict, nasa_score, regression_report_dict


def test_perfect_prediction_scores_zero():
    y = np.array([10.0, 50.0, 120.0])
    assert nasa_score(y, y) == 0.0


def test_late_predictions_cost_more_than_early_ones():
    """Same absolute error, but announcing too late must hurt more."""
    truth = np.array([50.0])
    early = nasa_score(truth, truth - 20)  # predicted 30, engine had 50 left
    late = nasa_score(truth, truth + 20)  # predicted 70, engine had 50 left
    assert late > early


def test_score_grows_with_error_magnitude():
    truth = np.array([60.0])
    assert nasa_score(truth, truth + 30) > nasa_score(truth, truth + 10)


def test_regression_report_flags_late_predictions():
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    preds = np.array([15.0, 18.0, 35.0, 38.0])
    report = regression_report_dict(truth, preds)
    assert report["late_prediction_rate"] == 0.5
    assert report["n"] == 4
    assert report["rmse"] > 0


def test_classification_report_handles_single_class():
    truth = np.zeros(5, dtype=int)
    probs = np.linspace(0.1, 0.4, 5)
    report = classification_report_dict(truth, probs)
    assert np.isnan(report["roc_auc"])
    assert report["positive_rate"] == 0.0
