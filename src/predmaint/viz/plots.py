"""Plotly figures shared by the notebooks, the README exports and the dashboard.

One theme, one palette, one set of rules, so every chart in the project reads as
the same product:

* series colors come from a fixed categorical order, never cycled
* risk uses a reserved status palette that is never reused for a data series
* status is always carried by color plus a written label, never color alone
* grid and axes stay recessive; the data is the only thing with contrast

The palette is colorblind-safe: worst all-pairs deutan separation is dE 9.2
(light) and 9.4 (dark) in OKLab x100, above the 8.0 target.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from predmaint.config import CYCLE_COL, MODEL_CONFIG, RISK_BAND_LABELS, UNIT_COL

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #

THEME: dict[str, Any] = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    # Fixed categorical order. Slots are assigned, never cycled.
    "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    # Reserved status palette. Never used for a data series.
    "status": {
        "healthy": "#0ca30c",
        "watch": "#fab219",
        "warning": "#ec835a",
        "critical": "#d03b3b",
    },
    "font": 'system-ui, -apple-system, "Segoe UI", sans-serif',
}

BAND_ORDER = ["critical", "warning", "watch", "healthy"]


def _base_layout(title: str, x_title: str, y_title: str) -> dict[str, Any]:
    return {
        "title": {
            "text": title,
            "font": {"size": 17, "color": THEME["ink"]},
            "x": 0.0,
            "xanchor": "left",
            "y": 0.97,
            "yanchor": "top",
        },
        "paper_bgcolor": THEME["surface"],
        "plot_bgcolor": THEME["surface"],
        "font": {"family": THEME["font"], "size": 12, "color": THEME["ink_secondary"]},
        "margin": {"l": 64, "r": 24, "t": 92, "b": 52},
        "hovermode": "closest",
        "xaxis": {
            "title": x_title,
            "gridcolor": THEME["grid"],
            "linecolor": THEME["axis"],
            "zeroline": False,
            "tickfont": {"color": THEME["muted"]},
        },
        "yaxis": {
            "title": y_title,
            "gridcolor": THEME["grid"],
            "linecolor": THEME["axis"],
            "zeroline": False,
            "tickfont": {"color": THEME["muted"]},
        },
        "legend": {"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0.0},
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def sensor_degradation(
    history: pd.DataFrame, sensor: str, units: list[int] | None = None, max_units: int = 3
) -> go.Figure:
    """Raw sensor traces for a handful of engines, aligned on cycle number.

    This is the chart that makes the problem obvious in one look: healthy for
    hundreds of cycles, then a visible drift toward failure.
    """
    units = units or sorted(history[UNIT_COL].unique())[:max_units]
    fig = go.Figure()
    for slot, unit in enumerate(units[:max_units]):
        block = history[history[UNIT_COL] == unit]
        fig.add_trace(
            go.Scatter(
                x=block[CYCLE_COL],
                y=block[sensor],
                mode="lines",
                name=f"Moteur {unit}",
                line={"width": 2, "color": THEME["series"][slot % len(THEME["series"])]},
                hovertemplate=(
                    f"Moteur {unit}<br>cycle %{{x}}<br>{sensor} %{{y:.2f}}<extra></extra>"
                ),
            )
        )
    fig.update_layout(**_base_layout(f"{sensor} sur la vie du moteur", "Cycle", sensor))
    return fig


def rul_trajectory(scored: pd.DataFrame, actual: pd.Series | None = None) -> go.Figure:
    """Predicted RUL cycle by cycle for one engine, against ground truth."""
    fig = go.Figure()
    if actual is not None:
        fig.add_trace(
            go.Scatter(
                x=scored[CYCLE_COL],
                y=np.minimum(actual, MODEL_CONFIG.rul_cap),
                mode="lines",
                name=f"Durée de vie réelle (plafonnée à {MODEL_CONFIG.rul_cap})",
                line={"width": 2, "color": THEME["muted"], "dash": "dot"},
                hovertemplate="cycle %{x}<br>réel %{y:.0f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=scored[CYCLE_COL],
            y=scored["predicted_rul"],
            mode="lines",
            name="Durée de vie prédite",
            line={"width": 2, "color": THEME["series"][0]},
            hovertemplate="cycle %{x}<br>prédit %{y:.0f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=MODEL_CONFIG.risk_horizon,
        line={"width": 1, "color": THEME["status"]["critical"], "dash": "dash"},
        annotation_text=f"Horizon de maintenance ({MODEL_CONFIG.risk_horizon} cycles)",
        annotation_position="top left",
        annotation_font={"color": THEME["status"]["critical"], "size": 11},
    )
    layout = _base_layout(
        "Trajectoire de la durée de vie restante", "Cycle", "Cycles restants"
    )
    fig.update_layout(**layout)
    return fig


def prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Held-out predictions against ground truth, with the perfect line.

    Points below the diagonal are late predictions: the engine failed sooner
    than announced. Those are the expensive ones, so they are labelled.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    late = y_pred > y_true
    limit = float(max(y_true.max(), y_pred.max())) * 1.05

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, limit],
            y=[0, limit],
            mode="lines",
            name="Prédiction parfaite",
            line={"width": 2, "color": THEME["muted"], "dash": "dash"},
            hoverinfo="skip",
        )
    )
    for mask, label, color in (
        (~late, "En avance ou exact (sans risque)", THEME["series"][0]),
        (late, "En retard (le moteur casse avant)", THEME["status"]["critical"]),
    ):
        fig.add_trace(
            go.Scatter(
                x=y_true[mask],
                y=y_pred[mask],
                mode="markers",
                name=label,
                marker={
                    "size": 9,
                    "color": color,
                    "line": {"width": 2, "color": THEME["surface"]},
                },
                hovertemplate="réel %{x:.0f}<br>prédit %{y:.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        **_base_layout(
            "Prédiction contre réalité (moteurs jamais vus)",
            "Cycles restants réels",
            "Cycles restants prédits",
        )
    )
    return fig


def error_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Signed error histogram. The left tail is what costs money."""
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    fig = go.Figure(
        go.Histogram(
            x=errors,
            nbinsx=40,
            marker={"color": THEME["series"][0], "line": {"width": 2, "color": THEME["surface"]}},
            name="Erreur de prédiction",
            hovertemplate="erreur %{x:.0f} cycles<br>%{y} moteurs<extra></extra>",
        )
    )
    fig.add_vline(x=0, line={"width": 1, "color": THEME["axis"]})
    fig.update_layout(
        **_base_layout(
            "Erreur de prédiction (prédit moins réel)", "Erreur en cycles", "Moteurs"
        )
    )
    return fig


def fleet_risk_bar(fleet: pd.DataFrame, top_n: int = 15, title: str | None = None) -> go.Figure:
    """The engines a maintenance planner should look at today, worst first."""
    if fleet.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title or "Aucun moteur", "", ""))
        fig.add_annotation(
            text="Aucun moteur dans cette sélection",
            showarrow=False,
            font={"color": THEME["muted"], "size": 14},
        )
        return fig

    block = fleet.nsmallest(top_n, "predicted_rul").sort_values("predicted_rul", ascending=False)
    colors = [THEME["status"][band] for band in block["risk_band"]]
    fig = go.Figure(
        go.Bar(
            x=block["predicted_rul"],
            y=[f"Moteur {u}" for u in block[UNIT_COL]],
            orientation="h",
            marker={"color": colors, "line": {"width": 2, "color": THEME["surface"]}},
            text=[
                f"{rul:.0f} cycles - {RISK_BAND_LABELS[band].lower()}"
                for rul, band in zip(block["predicted_rul"], block["risk_band"], strict=False)
            ],
            textposition="outside",
            textfont={"color": THEME["ink_secondary"], "size": 11},
            hovertemplate="%{y}<br>%{x:.0f} cycles restants<extra></extra>",
            showlegend=False,
        )
    )
    layout = _base_layout(
        title or f"Moteurs les plus à risque (top {min(top_n, len(block))})",
        "Cycles restants estimés",
        "",
    )
    layout["margin"]["l"] = 110
    layout["margin"]["r"] = 40
    # Leave room for the outside labels so they never clip at the plot edge.
    layout["xaxis"]["range"] = [0, float(block["predicted_rul"].max()) * 1.55 + 5]
    layout["bargap"] = 0.28
    # Grow with the number of bars so rows never squeeze into hairlines.
    layout["height"] = 150 + 34 * len(block)
    # No drag-zoom: a click on this chart is a navigation, not a zoom gesture.
    layout["dragmode"] = False
    fig.update_layout(**layout)
    return fig
