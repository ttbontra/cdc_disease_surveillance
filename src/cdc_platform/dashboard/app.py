from __future__ import annotations
import pandas as pd
import streamlit as st

from cdc_platform.config.settings import settings
from cdc_platform.serving.jobs.nightly_pipeline import run_nightly_pipeline
from cdc_platform.modeling.early_warning.alert_rules import generate_alerts
from cdc_platform.modeling.seir.forecasting import forecast_cases_seir


st.set_page_config(page_title=settings.dashboard_title, layout="wide")
st.title("🧬 " + settings.dashboard_title)

with st.sidebar:
    st.header("Controls")
    start = st.date_input("Start date", value=pd.Timestamp.today().date() - pd.Timedelta(days=120))
    end = st.date_input("End date", value=pd.Timestamp.today().date())
    horizon = st.slider("Forecast horizon (days)", 7, 56, 28, 7)
    population = st.number_input("Population (for SEIR)", min_value=100_000, value=1_000_000, step=50_000)

master = run_nightly_pipeline(str(start), str(end))

st.subheader("Latest Surveillance Snapshot")
col1, col2, col3 = st.columns(3)
latest_date = pd.to_datetime(master["date"]).max().date().isoformat()
col1.metric("Latest date", latest_date)
col2.metric("Regions", master["region"].nunique())
col3.metric("Total cases (latest)", int(master[pd.to_datetime(master["date"]).dt.date.astype(str) == latest_date]["cases"].sum()))

st.divider()

regions = sorted(master["region"].unique().tolist())
region = st.selectbox("Region", regions, index=0)

g = master[master["region"] == region].copy()
g["date"] = pd.to_datetime(g["date"])
g = g.sort_values("date")

c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Cases trend")
    st.line_chart(g.set_index("date")["cases"])

with c2:
    st.subheader("Early warning")
    alerts = [a for a in generate_alerts(master) if a.region == region]
    if not alerts:
        st.success("No active alerts.")
    else:
        st.warning(f"{len(alerts)} alerts flagged")
        st.dataframe(pd.DataFrame([a.__dict__ for a in alerts]).sort_values(["date", "level"], ascending=[False, True]))

st.divider()

st.subheader("SEIR-based Forecast (portfolio demo)")
out = forecast_cases_seir(g["cases"], pop=int(population), horizon_days=int(horizon))
hist_pred = out["history_pred"]
fut_pred = out["forecast"]

hist_idx = g["date"]
fut_idx = pd.date_range(hist_idx.max() + pd.Timedelta(days=1), periods=len(fut_pred), freq="D")

fc_df = pd.DataFrame({
    "date": list(hist_idx) + list(fut_idx),
    "pred_cases": list(hist_pred) + list(fut_pred)
}).set_index("date")

st.metric("Estimated R0 (rough)", round(float(out["rt0_est"]), 2))
st.line_chart(fc_df["pred_cases"])
