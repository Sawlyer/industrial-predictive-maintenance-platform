"""Training pipeline: baseline first, then the model that ships.

Two heads are trained on the same feature matrix:

* regressor  - clipped Remaining Useful Life, the planning number
* classifier - "will this engine fail within `risk_horizon` cycles", the alert

Validation uses GroupKFold on engine id. A random split would put cycle 41 of an
engine in train and cycle 42 in test; adjacent cycles are near-identical, so the
score would be inflated and meaningless. Splitting by engine is the only honest
option here, and it mirrors deployment: the model always meets new engines.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from predmaint.config import (
    CLASSIFIER_PATH,
    DEFAULT_SUBSET,
    METRICS_PATH,
    MODEL_CONFIG,
    OP_SETTING_COLS,
    REGRESSOR_PATH,
    RUL_COL,
    UNIT_COL,
    ModelConfig,
    ensure_dirs,
)
from predmaint.data.loader import load_subset
from predmaint.features.build import (
    build_features,
    clip_rul,
    drop_dead_sensors,
    feature_columns,
    last_cycle_snapshot,
)
from predmaint.models.evaluate import classification_report_dict, regression_report_dict


@dataclass
class TrainingArtifacts:
    """Everything needed to reproduce inference, saved next to the estimator."""

    subset: str
    sensors: list[str]
    features: list[str]
    config: dict[str, Any]
    trained_at: str
    n_train_rows: int
    n_train_units: int


def _make_regressor(config: ModelConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        random_state=config.random_state, **config.regressor_params
    )


def _make_classifier(config: ModelConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        random_state=config.random_state, **config.classifier_params
    )


def _baseline_regressor() -> Pipeline:
    """Ridge on the current-cycle snapshot. No trend, no memory.

    Kept in the repo on purpose: it is the number the gradient boosting model has
    to beat, and it quantifies what the feature engineering actually bought.
    """
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])


def cross_validate(
    X: pd.DataFrame,
    y_rul: np.ndarray,
    y_risk: np.ndarray,
    groups: np.ndarray,
    config: ModelConfig = MODEL_CONFIG,
) -> dict[str, Any]:
    """GroupKFold CV on engine id for both heads plus the baseline."""
    splitter = GroupKFold(n_splits=config.n_splits)
    folds: list[dict[str, Any]] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y_rul, groups)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]

        reg = _make_regressor(config).fit(X_tr, y_rul[train_idx])
        base = _baseline_regressor().fit(X_tr, y_rul[train_idx])
        clf = _make_classifier(config).fit(X_tr, y_risk[train_idx])

        folds.append(
            {
                "fold": fold,
                "model": regression_report_dict(y_rul[valid_idx], reg.predict(X_va)),
                "baseline": regression_report_dict(y_rul[valid_idx], base.predict(X_va)),
                "risk": classification_report_dict(
                    y_risk[valid_idx], clf.predict_proba(X_va)[:, 1]
                ),
            }
        )

    def mean_of(section: str) -> dict[str, float]:
        keys = folds[0][section].keys()
        return {
            key: float(np.mean([f[section][key] for f in folds]))
            for key in keys
            if isinstance(folds[0][section][key], (int, float))
        }

    return {
        "n_splits": config.n_splits,
        "model": mean_of("model"),
        "baseline": mean_of("baseline"),
        "risk": mean_of("risk"),
        "per_fold": folds,
    }


def train_models(
    subset: str = DEFAULT_SUBSET,
    config: ModelConfig = MODEL_CONFIG,
    run_cv: bool = True,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Full training run: features, CV, final fit, held-out test, artifacts."""
    ensure_dirs()
    started = time.perf_counter()

    train_raw, test_raw = load_subset(subset)
    sensors = drop_dead_sensors(train_raw)
    features = feature_columns(sensors)

    train = build_features(train_raw, sensors)
    test = build_features(test_raw, sensors)

    X = train[features]
    y_rul = clip_rul(train[RUL_COL], config.rul_cap)
    y_risk = (train[RUL_COL] <= config.risk_horizon).astype(int).to_numpy()
    groups = train[UNIT_COL].to_numpy()

    cv = cross_validate(X, y_rul, y_risk, groups, config) if run_cv else {}

    regressor = _make_regressor(config).fit(X, y_rul)
    classifier = _make_classifier(config).fit(X, y_risk)
    baseline = _baseline_regressor().fit(X, y_rul)

    # Held-out evaluation: last observed cycle of each test engine, which is the
    # exact question asked in production ("how long does this engine have left?").
    #
    # The target is clipped for *training* but not for *scoring* here. That is the
    # published C-MAPSS protocol: models learn the piecewise-linear target, and are
    # then graded against the true remaining cycles, so these numbers stay
    # comparable to the literature. Cross-validation above stays on the clipped
    # target, where the flat healthy region would otherwise dominate the error.
    snapshot = last_cycle_snapshot(test)
    X_test = snapshot[features]
    y_test_rul = snapshot[RUL_COL].to_numpy(dtype=float)
    y_test_risk = (snapshot[RUL_COL] <= config.risk_horizon).astype(int).to_numpy()

    test_metrics = {
        "model": regression_report_dict(y_test_rul, regressor.predict(X_test)),
        "baseline": regression_report_dict(y_test_rul, baseline.predict(X_test)),
        "risk": classification_report_dict(y_test_risk, classifier.predict_proba(X_test)[:, 1]),
    }

    artifacts = TrainingArtifacts(
        subset=subset,
        sensors=sensors,
        features=features,
        config=asdict(config),
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        n_train_rows=int(len(train)),
        n_train_units=int(train[UNIT_COL].nunique()),
    )

    models_dir = models_dir or REGRESSOR_PATH.parent
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"estimator": regressor, "baseline": baseline, "artifacts": asdict(artifacts)},
        models_dir / REGRESSOR_PATH.name,
    )
    joblib.dump(
        {"estimator": classifier, "artifacts": asdict(artifacts)},
        models_dir / CLASSIFIER_PATH.name,
    )

    report = {
        "subset": subset,
        "trained_at": artifacts.trained_at,
        "train_seconds": round(time.perf_counter() - started, 2),
        "n_features": len(features),
        "n_sensors_kept": len(sensors),
        "dropped_sensors": sorted(set(train_raw.filter(like="sensor_").columns) - set(sensors)),
        "op_settings": OP_SETTING_COLS,
        "cv": cv,
        "test": test_metrics,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
