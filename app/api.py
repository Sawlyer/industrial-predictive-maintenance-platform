"""REST API.

    GET  /health        liveness plus model status
    GET  /model         what is loaded: subset, sensors, features, horizon
    POST /predict       score one engine from its telemetry history
    GET  /fleet         score the held-out fleet, worst engines first

Scoring needs history, not a single row: the model reads trends over the last
cycles. The endpoint accepts the full cycle list for one engine and answers for
its most recent cycle.

Run with:  uvicorn app.api:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predmaint.config import (
    CYCLE_COL,
    DEFAULT_SUBSET,
    OP_SETTING_COLS,
    SENSOR_COLS,
    UNIT_COL,
    risk_band,
)
from predmaint.models.predict import FleetPredictor, load_predictor

STATE: dict[str, Any] = {"predictor": None, "error": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model once at startup; keep the API up if it is missing."""
    try:
        STATE["predictor"] = load_predictor()
    except FileNotFoundError as exc:
        STATE["error"] = str(exc)
    yield
    STATE.clear()


app = FastAPI(
    title="Plateforme de maintenance prédictive industrielle",
    description=(
        "Estimation de la durée de vie restante et du risque de panne pour une flotte "
        "de moteurs d'avion, à partir de la télémétrie brute des capteurs."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _predictor() -> FleetPredictor:
    if STATE["predictor"] is None:
        raise HTTPException(status_code=503, detail=STATE["error"] or "Model not loaded.")
    return STATE["predictor"]


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class CycleReading(BaseModel):
    """One telemetry row: three operating settings and 21 sensor channels."""

    cycle: int = Field(..., ge=1, description="Numéro de cycle, à partir de 1, sans trou.")
    op_settings: list[float] = Field(
        ..., min_length=3, max_length=3, description="Les 3 réglages de fonctionnement."
    )
    sensors: list[float] = Field(
        ..., min_length=21, max_length=21, description="Les 21 canaux de capteurs."
    )


class PredictRequest(BaseModel):
    unit: int = Field(1, description="Identifiant du moteur, pour la traçabilité uniquement.")
    history: list[CycleReading] = Field(
        ...,
        min_length=1,
        description="Historique des cycles du moteur, du plus ancien au plus récent.",
    )


class PredictResponse(BaseModel):
    unit: int
    cycle: int
    predicted_rul: float
    failure_probability: float
    risk_band: str
    risk_horizon: int
    recommendation: str


class FleetEngine(BaseModel):
    unit: int
    cycle: int
    predicted_rul: float
    failure_probability: float
    risk_band: str


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": STATE["predictor"] is not None,
        "detail": STATE["error"],
    }


@app.get("/model")
def model_info() -> dict[str, Any]:
    """Décrit le modèle chargé : jeu de données, variables, capteurs, horizon."""
    predictor = _predictor()
    meta = predictor.metadata
    return {
        "subset": meta.get("subset"),
        "trained_at": meta.get("trained_at"),
        "n_features": len(predictor.features),
        "sensors_used": predictor.sensors,
        "risk_horizon": predictor.risk_horizon,
        "rul_cap": predictor.rul_cap,
    }


def _to_frame(request: PredictRequest) -> pd.DataFrame:
    rows = []
    for reading in request.history:
        row = {UNIT_COL: request.unit, CYCLE_COL: reading.cycle}
        row.update(dict(zip(OP_SETTING_COLS, reading.op_settings, strict=True)))
        row.update(dict(zip(SENSOR_COLS, reading.sensors, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def _recommendation(rul: float, horizon: int) -> str:
    if rul < horizon:
        return "Immobiliser le moteur et planifier la maintenance immédiatement."
    if rul < horizon * 2:
        return "Réserver un créneau de maintenance dans la prochaine fenêtre de planification."
    return "Aucune action. Poursuivre la surveillance."


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Évalue le dernier cycle d'un moteur à partir de son historique de télémétrie."""
    predictor = _predictor()
    scored = predictor.score_history(_to_frame(request))
    latest = scored.iloc[-1]
    rul = float(latest["predicted_rul"])
    return PredictResponse(
        unit=request.unit,
        cycle=int(latest[CYCLE_COL]),
        predicted_rul=round(rul, 1),
        failure_probability=round(float(latest["failure_probability"]), 4),
        risk_band=risk_band(rul),
        risk_horizon=predictor.risk_horizon,
        recommendation=_recommendation(rul, predictor.risk_horizon),
    )


@app.get("/fleet", response_model=list[FleetEngine])
def fleet(limit: int = 20, subset: str = DEFAULT_SUBSET) -> list[FleetEngine]:
    """Évalue la flotte de test, du moteur le plus critique au moins critique."""
    from predmaint.data.loader import add_test_rul, load_test, load_test_rul

    predictor = _predictor()
    try:
        raw = add_test_rul(load_test(subset), load_test_rul(subset))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    scored = predictor.score_fleet(raw).head(limit)
    return [
        FleetEngine(
            unit=int(row[UNIT_COL]),
            cycle=int(row[CYCLE_COL]),
            predicted_rul=round(float(row["predicted_rul"]), 1),
            failure_probability=round(float(row["failure_probability"]), 4),
            risk_band=str(row["risk_band"]),
        )
        for _, row in scored.iterrows()
    ]
