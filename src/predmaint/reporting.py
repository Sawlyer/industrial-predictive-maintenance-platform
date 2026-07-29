"""Export the figures used by the README.

PNG when `kaleido` is installed, HTML otherwise, so `make figures` never fails
on a fresh machine.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from predmaint.config import (
    DEFAULT_SUBSET,
    FIGURES_DIR,
    RUL_COL,
    ensure_dirs,
)
from predmaint.data.loader import load_subset
from predmaint.features.build import drop_dead_sensors, last_cycle_snapshot
from predmaint.models.predict import load_predictor
from predmaint.viz.plots import (
    error_distribution,
    fleet_risk_bar,
    prediction_vs_actual,
    rul_trajectory,
    sensor_degradation,
)


def _save(fig: go.Figure, name: str, out_dir: Path, width: int = 1000, height: int = 560) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    try:
        fig.write_image(str(png_path), width=width, height=height, scale=2)
        return png_path
    except Exception:  # kaleido missing or no rendering backend
        html_path = out_dir / f"{name}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        return html_path


def export_figures(subset: str = DEFAULT_SUBSET, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Render every README figure from the trained model and the raw data."""
    ensure_dirs()
    train_raw, test_raw = load_subset(subset)
    predictor = load_predictor()

    sensors = drop_dead_sensors(train_raw)
    focus_sensor = sensors[len(sensors) // 2]

    fleet = predictor.score_fleet(test_raw)
    snapshot = last_cycle_snapshot(test_raw).sort_values("unit")
    # Graded against true remaining cycles, matching the published protocol.
    y_true = snapshot[RUL_COL].to_numpy(dtype=float)
    y_pred = fleet.sort_values("unit")["predicted_rul"].to_numpy()

    worst_unit = int(fleet.iloc[0]["unit"])
    unit_history = test_raw[test_raw["unit"] == worst_unit]
    unit_scored = predictor.score_unit(test_raw, worst_unit)
    unit_actual = (
        unit_history["rul"].reset_index(drop=True) if RUL_COL in unit_history else None
    )

    figures = {
        "01_sensor_degradation": sensor_degradation(train_raw, focus_sensor),
        "02_fleet_risk": fleet_risk_bar(fleet),
        "03_prediction_vs_actual": prediction_vs_actual(y_true, y_pred),
        "04_error_distribution": error_distribution(y_true, y_pred),
        "05_rul_trajectory": rul_trajectory(unit_scored, unit_actual),
    }
    return [_save(fig, name, out_dir) for name, fig in figures.items()]
