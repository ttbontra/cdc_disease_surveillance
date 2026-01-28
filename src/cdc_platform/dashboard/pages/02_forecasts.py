from __future__ import annotations

import pandas as pd
import streamlit as st

from cdc_platform.modeling.seir.forecasting import forecast_cases_seir_with_uncertainty
from cdc_platform.modeling.early_warning.alert_rules import generate_alerts
from cdc_platform.serving.jobs.nightly_pipeline import run_nightly_pipeline


def _recommend_actions(surge_prob: float | None, hosp_pred: float | None, alert_level: str | None) -> list[str]:
    """
    Portfolio-safe mitigation guidance: rules map risk → actions.
    Tune thresholds to your use case (disease, scale, jurisdiction).
    """
    actions: list[str] = []

    if alert_level == "warning":
        actions += [
            "Escalate to incident review: confirm signals (cases, wastewater, hospitalizations).",
            "Coordinate with local partners on surge staffing and bed capacity planning.",
            "Increase testing and case investigation capacity in high-risk areas.",
            "Targeted public messaging: symptoms, isolation guidance, vaccination reminders.",
        ]
    elif alert_level == "watch":
        actions += [
            "Increase monitoring frequency and validate data quality (reporting delays/backfills).",
            "Pre-position resources: test kits, outreach materials, clinic scheduling.",
            "Conduct situational briefing with stakeholders (schools, hospitals, local health).",
        ]
    else:
        actions += ["Continue routine monitoring; no escalation triggered."]

    # ML-driven adjustments
    if surge_prob is not None:
        if surge_prob >= 0.70:
            actions.insert(0, "High surge probability: prioritize vaccination/outreach and rapid response planning.")
        elif surge_prob >= 0.55:
            actions.insert(0, "Moderate surge probability: prepare targeted outreach and confirm leading indicators.")

    if hosp_pred is not None and hosp_pred >= 25:
        actions.insert(0, "Hospitalization pressure risk: pre-alert healthcare system, review surge capacity plans.")

    # Deduplicate while preserving order
    seen = set()
    out = []
    for a in actions:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out


def render():
    st.header("Forecasts + Mitigation Decision Support")

    with st.sidebar:
        st.subheader("Data window")
        start = st.date_input("Start date", value=pd.Timestamp.today().date() - pd.Timedelta(days=180))
        end = st.date_input("End date", value=pd.Timestamp.today().date())

        st.subheader("Forecasting")
        horizon = st.slider("Horizon (days)", 7, 56, 28, 7)
        pop = st.number_input("Population (SEIR)", min_value=100_000, value=1_000_000, step=50_000)

        st.subheader("Uncertainty")
        n_samples = st.slider("SEIR samples (bands)", 50, 600, 250, 50)
        beta_sd_frac = st.slider("Beta uncertainty (fraction)", 0.05, 0.40, 0.15, 0.01)

        st.caption("Tip: run `cdc train-ml` so this page can display ML surge & hospitalization risk.")

    master = run_nightly_pipeline(str(start), str(end))
    regions = sorted(master["region"].unique().tolist())
    region = st.selectbox("Region", regions)

    g = master[master["region"] == region].copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date")

    if g.empty:
        st.warning("No data for selected region.")
        return

    # --- Pull latest ML scores if present in the master table (nightly pipeline merges them on latest date) ---
    latest = g.tail(1).iloc[0]
    surge_prob = None
    hosp_pred = None

    if "surge_prob_gb" in g.columns and pd.notna(latest.get("surge_prob_gb")):
        surge_prob = float(latest["surge_prob_gb"])
    if "hosp_next7_pred" in g.columns and pd.notna(latest.get("hosp_next7_pred")):
        hosp_pred = float(latest["hosp_next7_pred"])

    # --- Alerts (already includes ML escalation if columns exist) ---
    all_alerts = generate_alerts(master)
    region_alerts = [a for a in all_alerts if a.region == region]
    region_alerts_df = pd.DataFrame([a.__dict__ for a in region_alerts]) if region_alerts else pd.DataFrame()

    alert_level = None
    if not region_alerts_df.empty:
        # pick most severe on latest date if present; otherwise overall most severe
        region_alerts_df["date"] = pd.to_datetime(region_alerts_df["date"])
        latest_date = g["date"].max().date()
        same_day = region_alerts_df[region_alerts_df["date"].dt.date == latest_date]
        pick = same_day if not same_day.empty else region_alerts_df

        # severity order
        sev = {"warning": 2, "watch": 1, "info": 0}
        pick["sev"] = pick["level"].map(sev).fillna(0)
        alert_level = pick.sort_values(["sev", "date"], ascending=[False, False]).iloc[0]["level"]

    # --- Forecast with uncertainty bands ---
    out = forecast_cases_seir_with_uncertainty(
        g["cases"],
        pop=int(pop),
        horizon_days=int(horizon),
        n_samples=int(n_samples),
        beta_sd_frac=float(beta_sd_frac),
        quantiles=(0.1, 0.5, 0.9),
    )

    # --- Layout: Metrics + risk ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimated R0 (rough)", round(float(out["rt0_est"]), 2))

    if surge_prob is not None:
        c2.metric("Surge prob (GB)", f"{surge_prob*100:.1f}%")
    else:
        c2.metric("Surge prob (GB)", "N/A")

    if hosp_pred is not None:
        c3.metric("Hosp in 7d (pred)", f"{hosp_pred:.1f}")
    else:
        c3.metric("Hosp in 7d (pred)", "N/A")

    c4.metric("Alert level", alert_level or "none")

    st.divider()

    # --- Charts: Observed + bands ---
    st.subheader("Cases (Observed) + SEIR Forecast Bands")

    hist_idx = g["date"]
    fut_idx = pd.date_range(hist_idx.max() + pd.Timedelta(days=1), periods=int(horizon), freq="D")

    hist_df = pd.DataFrame(
        {
            "date": hist_idx,
            "observed": g["cases"].astype(float).values,
            "q10": out["history"][0.1],
            "q50": out["history"][0.5],
            "q90": out["history"][0.9],
        }
    ).set_index("date")

    fut_df = pd.DataFrame(
        {
            "date": fut_idx,
            "observed": [None] * len(fut_idx),
            "q10": out["forecast"][0.1],
            "q50": out["forecast"][0.5],
            "q90": out["forecast"][0.9],
        }
    ).set_index("date")

    chart_df = pd.concat([hist_df, fut_df], axis=0)

    # Streamlit line_chart doesn't do shaded bands; we show q10/q50/q90 + observed
    st.line_chart(chart_df[["observed", "q10", "q50", "q90"]])

    st.caption("Bands represent uncertainty from sampling transmission rate (beta) around the calibrated SEIR model.")

    st.divider()

    # --- Alerts table ---
    st.subheader("Signals & Alerts")
    if region_alerts_df.empty:
        st.success("No alerts flagged for this region in the selected window.")
    else:
        st.dataframe(region_alerts_df.sort_values(["date", "level"], ascending=[False, True]))

    st.divider()

    # --- Mitigation recommendations panel ---
    st.subheader("Mitigation Recommendations (Decision Support)")
    actions = _recommend_actions(surge_prob=surge_prob, hosp_pred=hosp_pred, alert_level=alert_level)
    for a in actions:
        st.write(f"- {a}")

    st.caption(
        "These recommendations are rule-based decision support intended for portfolio demonstration. "
        "In real public health operations, actions are confirmed through epidemiologic investigation and stakeholder coordination."
    )


render()
