"""Feature engineering must be causal and reproducible.

The leakage test is the important one: if a feature at cycle t moves when a
future cycle changes, every offline metric in this repo is a lie.
"""

from __future__ import annotations

import pandas as pd

from predmaint.config import UNIT_COL
from predmaint.features.build import (
    build_features,
    clip_rul,
    drop_dead_sensors,
    feature_columns,
    last_cycle_snapshot,
)


def test_drop_dead_sensors_removes_constant_channels(toy_fleet):
    kept = drop_dead_sensors(toy_fleet)
    assert "sensor_1" not in kept
    assert "sensor_2" in kept


def test_feature_columns_match_built_frame(toy_fleet):
    sensors = drop_dead_sensors(toy_fleet)
    built = build_features(toy_fleet, sensors)
    expected = feature_columns(sensors)
    assert set(expected).issubset(built.columns)
    assert built[expected].notna().all().all()


def test_features_do_not_use_the_future(toy_fleet):
    """Truncating an engine's history must not change its earlier features."""
    sensors = drop_dead_sensors(toy_fleet)
    cols = feature_columns(sensors)

    full = build_features(toy_fleet, sensors)
    truncated_input = toy_fleet[~((toy_fleet[UNIT_COL] == 1) & (toy_fleet["cycle"] > 15))]
    truncated = build_features(truncated_input, sensors)

    left = full[(full[UNIT_COL] == 1) & (full["cycle"] <= 15)][cols].reset_index(drop=True)
    right = truncated[truncated[UNIT_COL] == 1][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-9)


def test_last_cycle_snapshot_keeps_one_row_per_engine(toy_fleet):
    snapshot = last_cycle_snapshot(toy_fleet)
    assert len(snapshot) == toy_fleet[UNIT_COL].nunique()
    assert snapshot.loc[snapshot[UNIT_COL] == 1, "cycle"].item() == 30


def test_clip_rul_caps_the_target():
    assert clip_rul([0, 100, 300], cap=125).tolist() == [0.0, 100.0, 125.0]
