"""Feature engineering.

Design rule: every feature is *causal*. A row at cycle t may only use data from
cycles <= t of the same engine. That is what makes offline metrics mean anything
once the model is served on a live engine.

Three families:

* raw sensor value at the current cycle       - where the engine is now
* rolling mean / std over W cycles            - denoised level, and volatility
* delta over W cycles and since first cycle   - the degradation trend

Trend is the part that matters. A single snapshot cannot separate an engine that
has always run hot from one that is heating up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from predmaint.config import (
    CYCLE_COL,
    MODEL_CONFIG,
    OP_SETTING_COLS,
    SENSOR_COLS,
    UNIT_COL,
)


def drop_dead_sensors(
    frame: pd.DataFrame, threshold: float = MODEL_CONFIG.variance_threshold
) -> list[str]:
    """Return the sensor columns that actually vary.

    Several C-MAPSS channels are constant across the whole fleet. Selecting on
    the training frame only keeps the choice honest.
    """
    variances = frame[SENSOR_COLS].var(numeric_only=True)
    return [col for col in SENSOR_COLS if variances.get(col, 0.0) > threshold]


def _rolling_block(
    frame: pd.DataFrame, sensors: list[str], window: int
) -> dict[str, pd.Series]:
    """Rolling mean, std and window delta for one window size."""
    grouped = frame.groupby(UNIT_COL)[sensors]
    rolled = grouped.rolling(window=window, min_periods=1)

    mean = rolled.mean().reset_index(level=0, drop=True)
    std = rolled.std().reset_index(level=0, drop=True).fillna(0.0)
    lagged = grouped.shift(window - 1) if window > 1 else frame[sensors]
    delta = (frame[sensors] - lagged.fillna(frame[sensors])) / float(window)

    block: dict[str, pd.Series] = {}
    for col in sensors:
        block[f"{col}_mean_{window}"] = mean[col]
        block[f"{col}_std_{window}"] = std[col]
        block[f"{col}_slope_{window}"] = delta[col]
    return block


def build_features(
    frame: pd.DataFrame,
    sensors: list[str] | None = None,
    windows: tuple[int, ...] = MODEL_CONFIG.rolling_windows,
) -> pd.DataFrame:
    """Return `frame` with engineered feature columns appended.

    Parameters
    ----------
    frame
        Raw C-MAPSS rows, sorted or not; sorting is enforced here.
    sensors
        Sensor columns to use. Pass the list fitted on the training set so that
        train, test and live inference share the exact same feature space.
    """
    sensors = sensors or drop_dead_sensors(frame)
    out = frame.sort_values([UNIT_COL, CYCLE_COL]).reset_index(drop=True)

    blocks: dict[str, pd.Series] = {}
    for window in windows:
        blocks.update(_rolling_block(out, sensors, window))

    # Drift since the engine's first observed cycle: cumulative degradation.
    first = out.groupby(UNIT_COL)[sensors].transform("first")
    for col in sensors:
        blocks[f"{col}_drift"] = out[col] - first[col]

    return pd.concat([out, pd.DataFrame(blocks, index=out.index)], axis=1)


def feature_columns(
    sensors: list[str], windows: tuple[int, ...] = MODEL_CONFIG.rolling_windows
) -> list[str]:
    """The exact ordered feature list a fitted model expects."""
    cols = [CYCLE_COL, *OP_SETTING_COLS, *sensors]
    for window in windows:
        for col in sensors:
            cols += [f"{col}_mean_{window}", f"{col}_std_{window}", f"{col}_slope_{window}"]
    cols += [f"{col}_drift" for col in sensors]
    return cols


def last_cycle_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per engine: its most recent cycle.

    This is the inference view. In production you score the latest telemetry of
    every engine in the fleet, not its whole history.
    """
    idx = frame.groupby(UNIT_COL)[CYCLE_COL].idxmax()
    return frame.loc[idx].sort_values(UNIT_COL).reset_index(drop=True)


def clip_rul(rul: pd.Series | np.ndarray, cap: int = MODEL_CONFIG.rul_cap) -> np.ndarray:
    """Piecewise-linear RUL target: flat while healthy, linear once degrading."""
    return np.minimum(np.asarray(rul, dtype=float), float(cap))
