"""Maintenance control room.

Three views, matching how the work actually happens:

    Fleet     which engines need attention today
    Engine    why this one is flagged, and how confident we are
    Model     what the model is worth, measured on engines it never saw

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from predmaint.config import (  # noqa: E402
    CYCLE_COL,
    DEFAULT_SUBSET,
    METRICS_PATH,
    MODEL_CONFIG,
    RAW_DIR,
    RISK_BAND_ACTIONS,
    RISK_BAND_LABELS,
    RUL_COL,
    UNIT_COL,
    band_range_label,
)
from predmaint.data.download import fetch, generate_demo_data, is_synthetic  # noqa: E402
from predmaint.data.loader import add_test_rul, load_test, load_test_rul  # noqa: E402
from predmaint.features.build import drop_dead_sensors  # noqa: E402
from predmaint.models.predict import load_predictor  # noqa: E402
from predmaint.models.train import train_models  # noqa: E402
from predmaint.viz.plots import (  # noqa: E402
    BAND_ORDER,
    THEME,
    error_distribution,
    fleet_risk_bar,
    prediction_vs_actual,
    rul_trajectory,
    sensor_degradation,
)

st.set_page_config(
    page_title="Plateforme de maintenance prédictive",
    page_icon="🛠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .stApp {{ background: {THEME["surface"]}; }}
      .block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
      h1, h2, h3 {{ color: {THEME["ink"]}; letter-spacing: -0.01em; }}
      .kpi {{
        border: 1px solid {THEME["grid"]}; border-radius: 10px;
        padding: 1rem 1.1rem; background: #ffffff; height: 100%;
      }}
      .kpi .label {{
        font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.07em;
        color: {THEME["muted"]};
      }}
      .kpi .value {{ font-size: 1.85rem; font-weight: 650; color: {THEME["ink"]}; }}
      .kpi .foot {{ font-size: 0.78rem; color: {THEME["ink_secondary"]}; }}
      .pill {{
        display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; color: #ffffff;
      }}
      .explainer {{
        border-left: 3px solid {THEME["series"][0]}; background: #ffffff;
        border-radius: 0 8px 8px 0; padding: 0.85rem 1.1rem; margin-bottom: 1.4rem;
        font-size: 0.9rem; line-height: 1.55; color: {THEME["ink_secondary"]};
      }}
      .explainer b {{ color: {THEME["ink"]}; }}
      .legend {{
        border: 1px solid {THEME["grid"]}; border-radius: 10px;
        padding: 0.7rem 0.85rem; background: #ffffff; height: 100%;
      }}
      .legend-top {{
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 0.35rem;
      }}
      .legend-count {{ font-size: 1.3rem; font-weight: 650; color: {THEME["ink"]}; }}
      .legend-range {{ font-size: 0.78rem; color: {THEME["ink_secondary"]}; }}
      .legend-action {{ font-size: 0.78rem; color: {THEME["muted"]}; margin-top: 0.2rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_predictor():
    return load_predictor()


@st.cache_data(show_spinner=False)
def get_fleet_raw(subset: str) -> pd.DataFrame:
    return add_test_rul(load_test(subset), load_test_rul(subset))


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def kpi(label: str, value: str, foot: str = "") -> str:
    return (
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="foot">{foot}</div></div>'
    )


def band_pill(band: str) -> str:
    label = RISK_BAND_LABELS[band]
    return f'<span class="pill" style="background:{THEME["status"][band]}">{label}</span>'


# theme=None keeps the project palette and typography; Streamlit's own chart
# theme would otherwise repaint every figure.
CHART_OPTS = {"width": "stretch", "theme": None, "config": {"displayModeBar": False}}


# --------------------------------------------------------------------------- #
# Boot
# --------------------------------------------------------------------------- #

st.title("Plateforme de maintenance prédictive industrielle")
st.caption(
    "Durée de vie restante et risque de panne pour une flotte de moteurs "
    "d'avion, calculés à partir de la télémétrie brute des capteurs."
)

subset = st.sidebar.selectbox("Jeu de données", [DEFAULT_SUBSET], index=0)

try:
    predictor = get_predictor()
    raw = get_fleet_raw(subset)
except FileNotFoundError:
    # First boot on a fresh machine (a new clone, a fresh container): fetch the
    # dataset and train once, then carry on. Roughly 40 seconds.
    st.info("Premier lancement sur cette machine : téléchargement et entraînement, une seule fois.")
    with st.spinner("Téléchargement du jeu de données NASA C-MAPSS..."):
        if not fetch(subset):
            st.warning("Téléchargement indisponible, bascule sur la flotte synthétique.")
            generate_demo_data(subset)
    with st.spinner("Entraînement du modèle de durée de vie et du classifieur de risque..."):
        train_models(subset=subset, run_cv=False)
    get_predictor.clear()
    get_fleet_raw.clear()
    get_metrics.clear()
    predictor = get_predictor()
    raw = get_fleet_raw(subset)

if is_synthetic(RAW_DIR):
    st.sidebar.warning(
        "Données synthétiques. Les métriques ne sont pas comparables aux résultats "
        "publiés sur C-MAPSS. Lancer `make data` pour le vrai jeu de données."
    )

fleet = predictor.score_fleet(raw)
horizon = predictor.risk_horizon

st.sidebar.markdown("### Modèle")
for label, value in (
    ("Entraîné le", predictor.metadata.get("trained_at", "n/a")),
    ("Variables", len(predictor.features)),
    ("Capteurs utilisés", len(predictor.sensors)),
    ("Horizon de risque", f"{horizon} cycles"),
):
    st.sidebar.markdown(
        f'<div style="display:flex;justify-content:space-between;'
        f'padding:0.25rem 0;border-bottom:1px solid {THEME["grid"]}">'
        f'<span style="color:{THEME["muted"]};font-size:0.82rem">{label}</span>'
        f'<span style="color:{THEME["ink"]};font-size:0.82rem">{value}</span></div>',
        unsafe_allow_html=True,
    )

# Views are driven by session state rather than st.tabs, because the fleet chart
# needs to send the user to a specific engine, and a tab cannot be switched
# programmatically.
VIEWS = ["Flotte", "Moteur", "Modèle"]
st.session_state.setdefault("view", "Flotte")
st.session_state.setdefault("nonce", 0)

# The control is deliberately NOT bound to the "view" key: Streamlit forbids
# writing to a widget's key after that widget has been instantiated, and the
# fleet chart needs to change the view further down the script. Rotating the
# widget key on navigation makes the control pick up the new default.
choice = st.segmented_control(
    "Vue",
    VIEWS,
    default=st.session_state.view,
    key=f"view_control_{st.session_state.nonce}",
    label_visibility="collapsed",
    width="content",
)
if choice:
    st.session_state.view = choice
view = st.session_state.view


def go_to(target: str, unit: int | None = None) -> None:
    """Switch view, optionally focusing one engine, and rerun.

    The nonce rotation also resets the chart and table selections, so a stale
    click does not re-fire the navigation on the way back.
    """
    if unit is not None:
        st.session_state.engine_select = unit
    st.session_state.view = target
    st.session_state.nonce += 1
    st.rerun()


# --------------------------------------------------------------------------- #
# Fleet
# --------------------------------------------------------------------------- #

if view == "Flotte":
    st.markdown(
        f"""
        <div class="explainer">
          <b>Ce que vous regardez.</b> Chaque ligne est un moteur d'avion en service. Les
          capteurs sont enregistrés à chaque cycle de vol. À partir des
          {max(MODEL_CONFIG.rolling_windows)} derniers cycles de cette télémétrie, le modèle
          estime combien de cycles il reste à chaque moteur avant la panne, et quelle est sa
          probabilité de tomber en panne dans les {horizon} prochains cycles. Les moteurs
          sont triés du plus critique au moins critique : le haut de cette page est la liste
          de travail du jour.
        </div>
        """,
        unsafe_allow_html=True,
    )

    counts = fleet["risk_band"].value_counts()
    critical = int(counts.get("critical", 0))
    at_risk = int((fleet["predicted_rul"] < horizon * 2).sum())

    cols = st.columns(4)
    cols[0].markdown(
        kpi("Moteurs suivis", f"{len(fleet)}", "évalués sur leur dernier cycle"), True
    )
    cols[1].markdown(
        kpi("À immobiliser", f"{critical}", f"moins de {horizon} cycles restants"), True
    )
    cols[2].markdown(
        kpi("Créneau à réserver", f"{at_risk}", f"moins de {horizon * 2} cycles restants"), True
    )
    cols[3].markdown(
        kpi(
            "Moteur médian",
            f"{fleet['predicted_rul'].median():.0f}",
            "cycles restants sur la flotte",
        ),
        True,
    )

    # ----------------------------------------------------------------------- #
    # Filter: pick the bands you want to act on
    # ----------------------------------------------------------------------- #

    st.markdown("### Filtrer par niveau de risque")
    st.caption(
        "Un niveau est simplement une plage de cycles restants. Sélectionnez-en un ou "
        "plusieurs pour filtrer le graphique et la file de maintenance ci-dessous. "
        "Aucune sélection affiche toute la flotte."
    )

    band_labels = {
        band: f"{RISK_BAND_LABELS[band]} ({int(counts.get(band, 0))})" for band in BAND_ORDER
    }
    selected = st.pills(
        "Niveaux de risque",
        options=BAND_ORDER,
        selection_mode="multi",
        default=[],
        format_func=lambda b: band_labels[b],
        label_visibility="collapsed",
        key="band_filter",
    )
    active_bands = selected or BAND_ORDER

    legend_cols = st.columns(len(BAND_ORDER))
    for col, band in zip(legend_cols, BAND_ORDER, strict=False):
        dimmed = "" if band in active_bands else "opacity:0.35;"
        col.markdown(
            f'<div class="legend" style="{dimmed}">'
            f'<div class="legend-top">{band_pill(band)}'
            f'<span class="legend-count">{int(counts.get(band, 0))}</span></div>'
            f'<div class="legend-range">{band_range_label(band)}</div>'
            f'<div class="legend-action">{RISK_BAND_ACTIONS[band]}</div></div>',
            unsafe_allow_html=True,
        )

    selection = fleet[fleet["risk_band"].isin(active_bands)]

    st.markdown("")
    if selection.empty:
        st.info("Aucun moteur dans les niveaux sélectionnés. Choisissez-en un autre ci-dessus.")
    else:
        shown = (
            ", ".join(RISK_BAND_LABELS[b] for b in selected) if selected else "toute la flotte"
        )
        rows = min(12, len(selection))
        left, right = st.columns([7, 5], gap="large")
        with left:
            st.caption("Cliquez sur une barre pour ouvrir la fiche du moteur.")
            event = st.plotly_chart(
                fleet_risk_bar(
                    selection,
                    top_n=rows,
                    title=f"Les plus proches de la panne - {shown} ({len(selection)} moteurs)",
                ),
                key=f"fleet_chart_{st.session_state.nonce}",
                on_select="rerun",
                selection_mode="points",
                **CHART_OPTS,
                height=560,
            )
            points = (event or {}).get("selection", {}).get("points", [])
            if points:
                # Bar labels read "Moteur 34"; the trailing token is the unit id.
                go_to("Moteur", int(str(points[0]["y"]).split()[-1]))

        with right:
            st.markdown("#### File de maintenance")
            st.caption(
                f"Les {rows} plus urgents parmi les {len(selection)} moteurs sélectionnés. "
                "La colonne Restants donne les cycles restants estimés."
            )
            columns = [UNIT_COL, CYCLE_COL, "predicted_rul", "failure_probability", "risk_band"]
            table = selection.head(rows)[columns].rename(
                columns={
                    UNIT_COL: "Moteur",
                    CYCLE_COL: "Cycles",
                    "predicted_rul": "Restants",
                    "failure_probability": "P(panne)",
                    "risk_band": "Niveau",
                }
            )
            table["Niveau"] = table["Niveau"].map(RISK_BAND_LABELS)
            picked_rows = st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
                height=460,
                key=f"fleet_table_{st.session_state.nonce}",
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "Moteur": st.column_config.NumberColumn(
                        width="small", help="Identifiant du moteur."
                    ),
                    "Cycles": st.column_config.NumberColumn(
                        width="small", help="Cycles de vol effectués depuis la mise en service."
                    ),
                    "Restants": st.column_config.NumberColumn(
                        format="%.0f",
                        width="small",
                        help="Cycles restants estimés avant la panne.",
                    ),
                    "P(panne)": st.column_config.ProgressColumn(
                        format="%.2f",
                        min_value=0.0,
                        max_value=1.0,
                        width="small",
                        help=f"Probabilité de panne dans les {horizon} prochains cycles.",
                    ),
                    "Niveau": st.column_config.TextColumn(
                        width="small", help="Niveau déduit des cycles restants estimés."
                    ),
                },
            )
            chosen = (picked_rows or {}).get("selection", {}).get("rows", [])
            if chosen:
                go_to("Moteur", int(table.iloc[chosen[0]]["Moteur"]))

        st.caption(
            "Une barre ou une ligne du tableau ouvre la fiche du moteur et explique "
            "pourquoi il est signalé."
        )


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

elif view == "Moteur":
    st.markdown(
        """
        <div class="explainer">
          <b>Pourquoi ce moteur est-il signalé ?</b> Sélectionnez-en un ci-dessous. Le
          premier graphique rejoue l'estimation du modèle à chaque cycle de la vie du
          moteur, face à la réalité : on voit à quel moment l'alerte se serait déclenchée.
          Le second montre un capteur brut sur la même période. Il est plat tant que le
          moteur est sain, puis dérive à mesure qu'il s'use. C'est cette dérive que le
          modèle lit.
        </div>
        """,
        unsafe_allow_html=True,
    )

    units = fleet[UNIT_COL].tolist()
    pick_col, back_col = st.columns([4, 1], vertical_alignment="bottom")
    unit = pick_col.selectbox(
        "Moteur (triés du plus critique au moins critique)",
        units,
        index=0,
        format_func=lambda u: f"Moteur {u}",
        key="engine_select",
    )
    if back_col.button("Retour à la flotte", use_container_width=True):
        go_to("Flotte")
    row = fleet[fleet[UNIT_COL] == unit].iloc[0]
    history = raw[raw[UNIT_COL] == unit]
    scored = predictor.score_unit(raw, unit)

    band = str(row["risk_band"])
    cols = st.columns(4)
    cols[0].markdown(
        kpi("Cycles restants", f"{row['predicted_rul']:.0f}", "estimation du modèle"), True
    )
    cols[1].markdown(
        kpi(
            "Risque de panne",
            f"{row['failure_probability']:.0%}",
            f"dans les {horizon} prochains cycles",
        ),
        True,
    )
    cols[2].markdown(
        kpi("Cycles effectués", f"{int(row[CYCLE_COL])}", "depuis la mise en service"), True
    )
    cols[3].markdown(kpi("Verdict", band_pill(band), RISK_BAND_ACTIONS[band]), True)

    st.markdown("")
    actual = history[RUL_COL].reset_index(drop=True) if RUL_COL in history else None
    st.plotly_chart(rul_trajectory(scored, actual), **CHART_OPTS, height=460)

    sensors = drop_dead_sensors(raw)
    default_sensor = sensors[len(sensors) // 2] if sensors else None
    picked = st.selectbox(
        "Capteur (les 6 canaux constants sont écartés avant l'entraînement)",
        sensors,
        index=sensors.index(default_sensor),
    )
    st.plotly_chart(sensor_degradation(history, picked, units=[unit]), **CHART_OPTS, height=420)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

elif view == "Modèle":
    metrics = get_metrics()
    if not metrics:
        st.info("Aucun fichier de métriques. Lancez `make train`.")
    else:
        test = metrics.get("test", {})
        model, baseline = test.get("model", {}), test.get("baseline", {})
        risk = test.get("risk", {})

        n_test = int(model.get("n", 0))
        st.markdown(
            f"""
            <div class="explainer">
              <b>Est-ce que ça marche ?</b> Tous les chiffres de cette page viennent de
              {n_test} moteurs que le modèle n'a jamais vus pendant l'entraînement, évalués
              à leur dernier cycle enregistré. L'erreur typique est d'environ
              {model.get("mae", 0):.0f} cycles. Le score NASA est la métrique officielle de
              ce jeu de données : elle pénalise une panne annoncée trop tard environ deux
              fois plus qu'une annonce trop tôt, parce qu'une alerte tardive cloue un avion
              au sol alors qu'une alerte prématurée coûte une inspection.
            </div>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns(4)
        cols[0].markdown(
            kpi("Erreur typique", f"{model.get('mae', 0):.1f}", "cycles d'écart en moyenne"), True
        )
        cols[1].markdown(
            kpi("RMSE", f"{model.get('rmse', 0):.1f}", "cycles, pénalise les gros écarts"), True
        )
        cols[2].markdown(
            kpi(
                "Score NASA",
                f"{model.get('nasa_score', 0):.0f}",
                f"sur {n_test} moteurs, plus bas est mieux",
            ),
            True,
        )
        cols[3].markdown(
            kpi(
                "Moteurs à risque détectés",
                f"{risk.get('recall', 0):.0%}",
                f"signalés avant {horizon} cycles restants",
            ),
            True,
        )

        st.markdown("#### Modèle de référence contre modèle retenu")
        st.caption(
            "Le modèle de référence est une régression Ridge sur le seul relevé du dernier "
            "cycle, sans notion de tendance. Il mesure ce que rapporte le travail sur les "
            "variables. Plus bas est mieux sur les quatre lignes."
        )
        comparison = pd.DataFrame(
            {
                "Métrique": [
                    "RMSE (cycles)",
                    "Erreur moyenne (cycles)",
                    "Score NASA par moteur",
                    "Taux de prédictions tardives",
                ],
                "Ridge sur le dernier relevé": [
                    baseline.get("rmse"), baseline.get("mae"),
                    baseline.get("nasa_score_per_engine"), baseline.get("late_prediction_rate"),
                ],
                "Gradient boosting avec tendances": [
                    model.get("rmse"), model.get("mae"),
                    model.get("nasa_score_per_engine"), model.get("late_prediction_rate"),
                ],
            }
        )
        st.dataframe(
            comparison.round(2),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ridge sur le dernier relevé": st.column_config.NumberColumn(format="%.2f"),
                "Gradient boosting avec tendances": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        snapshot = fleet.sort_values(UNIT_COL)
        truth = (
            raw.groupby(UNIT_COL)[RUL_COL].min().reindex(snapshot[UNIT_COL]).to_numpy()
            if RUL_COL in raw
            else None
        )
        if truth is not None:
            # Graded against true remaining cycles, not the clipped training target.
            y_true = pd.Series(truth).to_numpy(dtype=float)
            y_pred = snapshot["predicted_rul"].to_numpy()
            left, right = st.columns(2, gap="large")
            left.plotly_chart(prediction_vs_actual(y_true, y_pred), **CHART_OPTS, height=460)
            right.plotly_chart(error_distribution(y_true, y_pred), **CHART_OPTS, height=460)
