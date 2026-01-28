from __future__ import annotations

import pandas as pd
import streamlit as st

from cdc_platform.modeling.seir.forecasting import forecast_cases_seir
from cdc_platform.serving.jobs.nightly_pipeline import run_nightly_pipeline


def render():
    st.header("Forecasts (SEIR demo)")
    with st.sidebar:
        start = st.date_input("Start date", value=pd.Timestamp.today().date() - pd.Timedelta(days=180))
        end = st.date_input("End date", value=pd.Timestamp.today().date())
        horizon = st.slider("Horizon (days)", 7, 56, 28, 7)
        pop = st.number_input("Population", min_value=100_000, value=1_000_000, step=50_000)

    master = run_nightly_pipeline(str(start), str(end))
    regions = sorted(master["region"].unique().tolist())
    region = st.selectbox("Region", regions)

    g = master[master["region"] == region].copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date")

    out = forecast_cases_seir(g["cases"], pop=int(pop), horizon_days=int(horizon))
    st.metric("Estimated R0 (rough)", round(float(out["rt0_est"]), 2))

    hist_idx = g["date"]
    fut_idx = pd.date_range(hist_idx.max() + pd.Timedelta(days=1), periods=len(out["forecast"]), freq="D")
    fc = pd.DataFrame(
        {"date": list(hist_idx) + list(fut_idx), "pred_cases": list(out["history_pred"]) + list(out["forecast"])}
    ).set_index("date")

    st.line_chart(fc["pred_cases"])


render()
