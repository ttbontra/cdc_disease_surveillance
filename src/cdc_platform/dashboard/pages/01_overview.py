from __future__ import annotations

import pandas as pd
import streamlit as st

from cdc_platform.dashboard.app import run_nightly_pipeline  # reuse pipeline from app module


def render():
    st.header("Overview")
    with st.sidebar:
        start = st.date_input("Start date", value=pd.Timestamp.today().date() - pd.Timedelta(days=120))
        end = st.date_input("End date", value=pd.Timestamp.today().date())

    master = run_nightly_pipeline(str(start), str(end))
    master["date"] = pd.to_datetime(master["date"])

    st.metric("Regions", master["region"].nunique())
    st.metric("Latest date", master["date"].max().date().isoformat())

    latest = master.sort_values("date").groupby("region").tail(1)
    st.subheader("Latest by region")
    st.dataframe(latest[["region", "date", "cases", "hosp", "ww_viral_load", "mobility_index"]].sort_values("cases", ascending=False))


render()
