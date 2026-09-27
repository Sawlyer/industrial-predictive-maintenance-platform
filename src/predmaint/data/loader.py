"""Loading and RUL labelling for the NASA C-MAPSS dataset.

Raw layout, per subset (FD001..FD004):

    train_FD001.txt  one row per engine per cycle, run until failure
    test_FD001.txt   same schema, truncated some time before failure
    RUL_FD001.txt    one integer per test engine: RUL at its last observed cycle

Files are whitespace separated with a trailing separator, which pandas reads as
two extra all-NaN columns. We name the 26 real columns and drop the rest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from predmaint.config import (
    CYCLE_COL,
    DEFAULT_SUBSET,
    RAW_COLUMNS,
    RAW_DIR,
    RUL_COL,
    UNIT_COL,
)


def _read_cmapss_file(path: Path) -> pd.DataFrame:
    """Read one whitespace-separated C-MAPSS file into a typed DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make data` (real dataset) or "
            f"`make demo-data` (synthetic stand-in) first."
        )
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    frame = frame.iloc[:, : len(RAW_COLUMNS)]
    frame.columns = RAW_COLUMNS
    frame[UNIT_COL] = frame[UNIT_COL].astype(int)
    frame[CYCLE_COL] = frame[CYCLE_COL].astype(int)
    return frame


def load_train(subset: str = DEFAULT_SUBSET, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Training runs, each ending at engine failure."""
    return _read_cmapss_file(raw_dir / f"train_{subset}.txt")


def load_test(subset: str = DEFAULT_SUBSET, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Test runs, truncated before failure."""
    return _read_cmapss_file(raw_dir / f"test_{subset}.txt")


def load_test_rul(subset: str = DEFAULT_SUBSET, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Ground-truth RUL for each test engine at its last observed cycle."""
    path = raw_dir / f"RUL_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make data` or `make demo-data` first.")
    values = pd.read_csv(path, sep=r"\s+", header=None, engine="python").iloc[:, 0]
    return pd.DataFrame({UNIT_COL: np.arange(1, len(values) + 1), RUL_COL: values.astype(int)})


def add_rul(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the run-to-failure RUL column to a training frame.

    A training engine is observed until it breaks, so its RUL at any cycle is
    simply (last cycle of that engine) - (current cycle).
    """
    out = frame.copy()
    last_cycle = out.groupby(UNIT_COL)[CYCLE_COL].transform("max")
    out[RUL_COL] = last_cycle - out[CYCLE_COL]
    return out


def add_test_rul(frame: pd.DataFrame, test_rul: pd.DataFrame) -> pd.DataFrame:
    """Add RUL to a test frame using the provided end-of-run ground truth.

    The engine still has `rul_at_end` cycles left after its last observed cycle,
    so at cycle t its RUL is rul_at_end + (last observed cycle - t).
    """
    out = frame.copy()
    last_cycle = out.groupby(UNIT_COL)[CYCLE_COL].transform("max")
    end_rul = out[UNIT_COL].map(test_rul.set_index(UNIT_COL)[RUL_COL])
    out[RUL_COL] = end_rul + (last_cycle - out[CYCLE_COL])
    return out


def load_subset(
    subset: str = DEFAULT_SUBSET, raw_dir: Path = RAW_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one subset, fully labelled.

    Returns
    -------
    (train, test)
        Both frames carry a `rul` column, so evaluation code stays symmetric.
    """
    train = add_rul(load_train(subset, raw_dir))
    test = add_test_rul(load_test(subset, raw_dir), load_test_rul(subset, raw_dir))
    return train, test
