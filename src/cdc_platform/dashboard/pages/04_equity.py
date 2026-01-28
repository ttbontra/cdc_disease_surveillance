from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from cdc_platform.serving.jobs.nightly_pipeline import run_nightly_pipeline


def render():
    st.header("Equity (portfolio placeholder)")
    st.caption("This page demonstrates how you'd structure equity monitoring. Replace synthetic logic with real stratified data.")

    with st.sidebar:
        start = st.date_input("Start date", value=pd.Timestamp.today().date() - pd.Timedelta(days=180))
        end = st.date_input("End date", value=pd.Timestamp.today().date())

    master = run_nightly_pipeline(str(start), str(end))
    master["date"] = pd.to_datetime(master["date"])

    # Placeholder: synthetic “vulnerability index” per region
    regions = sorted(master["region"].unique().tolist())
    rs = np.random.RandomState(10)
    vuln = pd.DataFrame({"region": regions, "vulnerability_index": rs.uniform(0, 1, size=len(regions))})
    latest = master.sort_values("date").groupby("region").tail(1)[["region", "cases", "hosp"]]
    joined = latest.merge(vuln, on="region", how="left").sort_values("vulnerability_index", ascending=False)

    st.subheader("Latest outcomes vs vulnerability (demo)")
    st.dataframe(joined)


render()
