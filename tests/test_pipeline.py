"""End-to-end smoke test on a small synthetic fleet.

Trains both heads, saves artifacts, reloads them and scores a fleet. Slow enough
to be worth one test, fast enough to run in CI.
"""

from __future__ import annotations

import pandas as pd
import pytest

from predmaint.config import ModelConfig
from predmaint.data.download import generate_demo_data
from predmaint.data.loader import add_rul, add_test_rul, load_test, load_test_rul, load_train
from predmaint.models.predict import load_predictor
from predmaint.models.train import train_models


@pytest.fixture(scope="module")
def tiny_raw(tmp_path_factory):
    raw_dir = tmp_path_factory.mktemp("raw")
    generate_demo_data("FD001", n_train_units=8, n_test_units=4, raw_dir=raw_dir, seed=7)
    return raw_dir


def test_add_rul_counts_down_to_zero(tiny_raw):
    train = add_rul(load_train("FD001", tiny_raw))
    last = train.groupby("unit")["rul"].min()
    assert (last == 0).all()


def test_test_rul_never_reaches_zero(tiny_raw):
    """Test engines are truncated before failure, so RUL stays strictly positive."""
    test = add_test_rul(load_test("FD001", tiny_raw), load_test_rul("FD001", tiny_raw))
    assert (test.groupby("unit")["rul"].min() > 0).all()


def test_train_and_score_roundtrip(tiny_raw, tmp_path, monkeypatch):
    monkeypatch.setattr("predmaint.models.train.load_subset", lambda subset: (
        add_rul(load_train(subset, tiny_raw)),
        add_test_rul(load_test(subset, tiny_raw), load_test_rul(subset, tiny_raw)),
    ))
    monkeypatch.setattr("predmaint.models.train.METRICS_PATH", tmp_path / "metrics.json")

    config = ModelConfig(n_splits=2, rolling_windows=(5,))
    report = train_models(subset="FD001", config=config, models_dir=tmp_path)

    assert report["test"]["model"]["rmse"] > 0
    assert 0.0 <= report["test"]["risk"]["recall"] <= 1.0
    assert (tmp_path / "rul_regressor.joblib").exists()

    predictor = load_predictor(models_dir=tmp_path)
    raw = add_test_rul(load_test("FD001", tiny_raw), load_test_rul("FD001", tiny_raw))
    fleet = predictor.score_fleet(raw)

    assert len(fleet) == raw["unit"].nunique()
    assert fleet["predicted_rul"].between(0, predictor.rul_cap).all()
    assert fleet["failure_probability"].between(0, 1).all()
    assert fleet["predicted_rul"].is_monotonic_increasing  # sorted worst first
    assert isinstance(fleet, pd.DataFrame)
