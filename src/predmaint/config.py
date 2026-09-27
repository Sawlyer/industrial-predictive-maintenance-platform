"""Central configuration: paths, dataset schema and modelling constants.

Every tunable lives here so that notebooks, CLI, API and Streamlit app share one
source of truth instead of drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REGRESSOR_PATH = MODELS_DIR / "rul_regressor.joblib"
CLASSIFIER_PATH = MODELS_DIR / "risk_classifier.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Dataset schema (NASA C-MAPSS)
# --------------------------------------------------------------------------- #

DEFAULT_SUBSET = "FD001"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")

UNIT_COL = "unit"
CYCLE_COL = "cycle"
OP_SETTING_COLS = ["op_1", "op_2", "op_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
RAW_COLUMNS = [UNIT_COL, CYCLE_COL, *OP_SETTING_COLS, *SENSOR_COLS]

RUL_COL = "rul"
LABEL_COL = "will_fail_soon"

# --------------------------------------------------------------------------- #
# Modelling constants
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    """Modelling choices, each defensible in an interview.

    rul_cap
        C-MAPSS engines run healthy for a long time before degrading. Sensors
        carry no signal about "480 cycles left" vs "300 cycles left", so the
        target is clipped to a piecewise-linear RUL. 125 is the value used by
        the reference literature, which keeps our numbers comparable.
    risk_horizon
        Maintenance is planned, not instant. A binary "fails within N cycles"
        head turns the regression into an actionable work-order trigger.
    rolling_windows
        Degradation is a trend, not a snapshot. Rolling mean / std / slope over
        several horizons let a tabular model see the trajectory.
    """

    rul_cap: int = 125
    risk_horizon: int = 30
    rolling_windows: tuple[int, ...] = (5, 10, 20)
    variance_threshold: float = 1e-6
    n_splits: int = 5
    random_state: int = 42
    regressor_params: dict = field(
        default_factory=lambda: {
            "max_iter": 400,
            "learning_rate": 0.06,
            "max_depth": 6,
            "min_samples_leaf": 40,
            "l2_regularization": 1.0,
            "early_stopping": False,
        }
    )
    classifier_params: dict = field(
        default_factory=lambda: {
            "max_iter": 400,
            "learning_rate": 0.06,
            "max_depth": 6,
            "min_samples_leaf": 40,
            "l2_regularization": 1.0,
            "early_stopping": False,
        }
    )


MODEL_CONFIG = ModelConfig()

# Risk bands used by the dashboard and the API response.
RISK_BANDS = (
    ("critical", 0, 30),
    ("warning", 30, 60),
    ("watch", 60, 100),
    ("healthy", 100, 10_000),
)


# Band identifiers stay English: they are data values, stored in the `risk_band`
# column and used as dictionary keys across the code. Only their display labels
# are translated, in the mappings below.
RISK_BAND_LABELS = {
    "critical": "Critique",
    "warning": "Alerte",
    "watch": "Surveillance",
    "healthy": "Sain",
}

# What each band means to the person reading the dashboard.
RISK_BAND_ACTIONS = {
    "critical": "Immobiliser le moteur, intervenir maintenant",
    "warning": "Réserver un créneau de maintenance",
    "watch": "Surveiller, pas d'action pour l'instant",
    "healthy": "Rien à faire",
}


def risk_band(rul_value: float) -> str:
    """Map a predicted RUL to an operational risk band."""
    for name, low, high in RISK_BANDS:
        if low <= rul_value < high:
            return name
    return "healthy"


def band_range_label(name: str) -> str:
    """Human-readable RUL range for a band, derived from RISK_BANDS."""
    for band, low, high in RISK_BANDS:
        if band != name:
            continue
        if low == 0:
            return f"moins de {high} cycles restants"
        if high >= 10_000:
            return f"{low} cycles restants ou plus"
        return f"{low} à {high} cycles restants"
    return ""


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for path in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
