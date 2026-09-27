from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predmaint.config import OP_SETTING_COLS, SENSOR_COLS


@pytest.fixture
def toy_fleet() -> pd.DataFrame:
    """Two engines, deterministic, with one dead sensor channel."""
    rng = np.random.default_rng(0)
    blocks = []
    for unit, life in ((1, 30), (2, 20)):
        frame = pd.DataFrame(
            rng.normal(100.0, 1.0, size=(life, len(SENSOR_COLS))), columns=SENSOR_COLS
        )
        frame["sensor_1"] = 42.0  # dead channel
        frame["sensor_2"] = np.linspace(0.0, 10.0, life)  # clean upward drift
        frame.insert(0, "unit", unit)
        frame.insert(1, "cycle", np.arange(1, life + 1))
        for i, col in enumerate(OP_SETTING_COLS):
            frame.insert(2 + i, col, 0.0)
        blocks.append(frame)
    return pd.concat(blocks, ignore_index=True)
