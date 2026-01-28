from __future__ import annotations

import pandas as pd
import streamlit as st

from cdc_platform.modeling.early_warning.alert_rules import generate_alerts
from cdc_platform.serving.jobs.nightly_pipeline import run_nightly_pipeline


def render():
    st.header("Alerts")
    with st.sidebar:
        start = st.date_input("Start date", value=pd.Timestamp.today().date() - pd.Timedelta(days=120))
        end = st.date_input("End date", value=pd.Timestamp.today().date())

    master = run_nightly_pipeline(str(start), str(end))
    alerts = generate_alerts(master)
    df = pd.DataFrame([a.__dict__ for a in alerts]) if alerts else pd.DataFrame(columns=["date", "region", "level", "reason"])

    st.subheader("Active flags")
    if df.empty:
        st.success("No alerts flagged.")
    else:
        df["date"] = pd.to_datetime(df["date"])
        st.dataframe(df.sort_values(["date", "level"], ascending=[False, True]))


render()
