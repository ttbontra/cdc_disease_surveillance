from __future__ import annotations

import pandas as pd
import streamlit as st

from cdc_platform.serving.jobs.nightly_pipeline import run_nightly_pipeline


def render():
    st.header("Data Quality")
    with st.sidebar:
        start = st.date_input("Start date", value=pd.Timestamp.today().date() - pd.Timedelta(days=120))
        end = st.date_input("End date", value=pd.Timestamp.today().date())

    master = run_nightly_pipeline(str(start), str(end))
    st.subheader("Missingness")
    miss = master.isna().mean().sort_values(ascending=False).reset_index()
    miss.columns = ["column", "missing_rate"]
    st.dataframe(miss)

    st.subheader("Summary")
    st.dataframe(master.describe(include="all"))


render()
