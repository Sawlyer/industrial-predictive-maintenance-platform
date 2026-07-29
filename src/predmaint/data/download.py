"""Get raw data into `data/raw/`.

Two paths:

1. `fetch()` downloads the real NASA C-MAPSS archive from the Prognostics Data
   Repository bucket and unpacks the four subsets. No account, no API key.
2. `generate_demo_data()` writes a synthetic fleet with the same schema and the
   same degradation shape, for offline demos and for CI, where downloading a
   12 MB archive on every run would be wasteful.

The synthetic mode is clearly labelled everywhere it is used. It exists to make
the demo reproducible, not to inflate reported metrics.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from predmaint.config import (
    DEFAULT_SUBSET,
    OP_SETTING_COLS,
    RAW_DIR,
    SENSOR_COLS,
    SUBSETS,
    ensure_dirs,
)

# NASA Prognostics Data Repository, dataset 6. The archive nests a second zip.
CMAPSS_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
INNER_ARCHIVE = "CMAPSSData.zip"
KAGGLE_COMMAND = "kaggle datasets download -d behrad3d/nasa-cmaps -p data/raw --unzip"
NASA_PORTAL = "https://www.nasa.gov/intelligent-systems-division/  (Prognostics Data Repository)"
SYNTHETIC_MARKER = "SYNTHETIC.txt"

# Zero-variance sensor channels, mirroring sensors 1, 5, 10, 16, 18 and 19 of FD001.
DEAD_CHANNELS = {0: 518.67, 4: 14.62, 9: 1.30, 15: 0.03, 17: 2388.0, 18: 100.0}


def required_files(subset: str = DEFAULT_SUBSET) -> list[str]:
    return [f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"]


def is_synthetic(raw_dir: Path = RAW_DIR) -> bool:
    """True when `data/raw` holds the generated stand-in, not the NASA files."""
    return (raw_dir / SYNTHETIC_MARKER).exists()


def is_available(subset: str = DEFAULT_SUBSET, raw_dir: Path = RAW_DIR) -> bool:
    """True when the real subset files are on disk."""
    if is_synthetic(raw_dir):
        return False
    return all((raw_dir / name).exists() for name in required_files(subset))


def _extract(archive: bytes, raw_dir: Path) -> list[str]:
    """Unpack the nested archive, keeping the data files and the NASA readme."""
    outer = zipfile.ZipFile(io.BytesIO(archive))
    inner_name = next(n for n in outer.namelist() if n.endswith(INNER_ARCHIVE))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))

    written = []
    for member in inner.namelist():
        name = Path(member).name
        if not name.endswith(".txt"):
            continue
        (raw_dir / name).write_bytes(inner.read(member))
        written.append(name)
    return sorted(written)


def fetch(subset: str = DEFAULT_SUBSET, raw_dir: Path = RAW_DIR, force: bool = False) -> bool:
    """Download and unpack the real C-MAPSS dataset.

    Returns True once every file of `subset` is present. On a network failure it
    prints the manual alternatives rather than raising, so `make data` stays a
    safe command to run.
    """
    ensure_dirs()
    if is_available(subset, raw_dir) and not force:
        print(f"C-MAPSS {subset} already present in {raw_dir}.")
        return True

    print(f"Downloading C-MAPSS from {CMAPSS_URL}")
    try:
        with urllib.request.urlopen(CMAPSS_URL, timeout=120) as response:
            archive = response.read()
    except OSError as exc:
        print(f"Download failed: {exc}")
        print(f"\nManual alternatives:\n    {KAGGLE_COMMAND}\n    {NASA_PORTAL}")
        print("\nOr run everything offline on the synthetic fleet:\n    make demo-data")
        return False

    print(f"Downloaded {len(archive) / 1e6:.1f} MB, unpacking into {raw_dir}")
    written = _extract(archive, raw_dir)
    (raw_dir / SYNTHETIC_MARKER).unlink(missing_ok=True)

    found = [s for s in SUBSETS if all((raw_dir / n).exists() for n in required_files(s))]
    print(f"Wrote {len(written)} files. Subsets available: {', '.join(found)}")
    return is_available(subset, raw_dir)


# --------------------------------------------------------------------------- #
# Synthetic fallback
# --------------------------------------------------------------------------- #


def _simulate_engine(
    rng: np.random.Generator, n_cycles: int, n_sensors: int = len(SENSOR_COLS)
) -> np.ndarray:
    """One run-to-failure trajectory.

    Each sensor is a healthy baseline plus a monotone degradation term that
    accelerates near end of life, plus noise. A handful of sensors are left
    constant, mirroring the real dataset where several channels carry no signal.
    """
    t = np.linspace(0.0, 1.0, n_cycles)
    baselines = rng.normal(loc=500.0, scale=120.0, size=n_sensors)
    amplitudes = rng.normal(loc=0.0, scale=18.0, size=n_sensors)
    curvature = rng.uniform(1.6, 3.4, size=n_sensors)

    signal = baselines + amplitudes * (t[:, None] ** curvature)
    noise = rng.normal(0.0, 1.2, size=signal.shape)
    out = signal + noise

    # Six dead channels, as in FD001: constant across the whole fleet, so the
    # variance filter drops them exactly the way it does on the real dataset.
    for idx, value in DEAD_CHANNELS.items():
        out[:, idx] = value
    return out


def generate_demo_data(
    subset: str = DEFAULT_SUBSET,
    n_train_units: int = 100,
    n_test_units: int = 50,
    raw_dir: Path = RAW_DIR,
    seed: int = 42,
) -> Path:
    """Write synthetic train/test/RUL files with the C-MAPSS schema."""
    ensure_dirs()
    rng = np.random.default_rng(seed)

    def build(n_units: int, truncate: bool) -> tuple[pd.DataFrame, list[int]]:
        rows, remaining = [], []
        for unit in range(1, n_units + 1):
            life = int(rng.integers(140, 340))
            sensors = _simulate_engine(rng, life)
            observed = life
            if truncate:
                observed = int(rng.integers(int(life * 0.45), life - 5))
                remaining.append(life - observed)
            block = pd.DataFrame(sensors[:observed], columns=SENSOR_COLS)
            block.insert(0, "unit", unit)
            block.insert(1, "cycle", np.arange(1, observed + 1))
            for i, col in enumerate(OP_SETTING_COLS):
                block.insert(2 + i, col, rng.normal(0.0, 0.002, size=observed).round(4))
            rows.append(block)
        return pd.concat(rows, ignore_index=True), remaining

    train, _ = build(n_train_units, truncate=False)
    test, rul = build(n_test_units, truncate=True)

    raw_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(raw_dir / f"train_{subset}.txt", sep=" ", header=False, index=False)
    test.to_csv(raw_dir / f"test_{subset}.txt", sep=" ", header=False, index=False)
    pd.Series(rul).to_csv(raw_dir / f"RUL_{subset}.txt", header=False, index=False)

    marker = raw_dir / "SYNTHETIC.txt"
    marker.write_text(
        "These files are synthetic, generated by predmaint.data.download.\n"
        "Metrics computed on them are NOT comparable to published C-MAPSS results.\n"
        f"Replace them with the real dataset: {KAGGLE_COMMAND}\n",
        encoding="utf-8",
    )
    print(f"Synthetic {subset} written to {raw_dir} ({n_train_units} train / {n_test_units} test).")
    return raw_dir
