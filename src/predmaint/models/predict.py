"""Inference surface shared by the CLI, the REST API and the dashboard.

One class, one contract: give it raw telemetry rows, get back RUL, failure
probability and an operational risk band. The feature pipeline is rebuilt from
the artifacts saved at training time, so serving can never drift from training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from predmaint.config import (
    CLASSIFIER_PATH,
    CYCLE_COL,
    MODELS_DIR,
    REGRESSOR_PATH,
    UNIT_COL,
    risk_band,
)
from predmaint.features.build import build_features, last_cycle_snapshot


@dataclass
class FleetPredictor:
    """RUL regressor + risk classifier behind a single scoring call."""

    regressor: Any
    classifier: Any
    baseline: Any
    sensors: list[str]
    features: list[str]
    metadata: dict[str, Any]

    @property
    def risk_horizon(self) -> int:
        return int(self.metadata.get("config", {}).get("risk_horizon", 30))

    @property
    def rul_cap(self) -> int:
        return int(self.metadata.get("config", {}).get("rul_cap", 125))

    def score_history(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Score every cycle of every engine. Used for trajectory plots."""
        frame = build_features(raw, self.sensors)
        X = frame[self.features]
        out = frame[[UNIT_COL, CYCLE_COL]].copy()
        out["predicted_rul"] = self.regressor.predict(X).clip(0, self.rul_cap)
        out["failure_probability"] = self.classifier.predict_proba(X)[:, 1]
        out["risk_band"] = out["predicted_rul"].map(risk_band)
        return out

    def score_fleet(self, raw: pd.DataFrame) -> pd.DataFrame:
        """One row per engine, at its latest observed cycle. The fleet view."""
        scored = self.score_history(raw)
        fleet = last_cycle_snapshot(scored)
        return fleet.sort_values("predicted_rul").reset_index(drop=True)

    def score_unit(self, raw: pd.DataFrame, unit: int) -> pd.DataFrame:
        """Trajectory for a single engine."""
        return self.score_history(raw[raw[UNIT_COL] == unit])


def load_predictor(models_dir: Path = MODELS_DIR) -> FleetPredictor:
    """Load the trained bundles from disk.

    Raises
    ------
    FileNotFoundError
        When the models have not been trained yet, with the command to run.
    """
    reg_path = models_dir / REGRESSOR_PATH.name
    clf_path = models_dir / CLASSIFIER_PATH.name
    if not reg_path.exists() or not clf_path.exists():
        raise FileNotFoundError(
            f"No trained model in {models_dir}. Run `make train` (or `predmaint train`) first."
        )

    reg_bundle = joblib.load(reg_path)
    clf_bundle = joblib.load(clf_path)
    artifacts = reg_bundle["artifacts"]
    return FleetPredictor(
        regressor=reg_bundle["estimator"],
        classifier=clf_bundle["estimator"],
        baseline=reg_bundle.get("baseline"),
        sensors=artifacts["sensors"],
        features=artifacts["features"],
        metadata=artifacts,
    )
