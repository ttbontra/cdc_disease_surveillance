from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from .changepoints import simple_changepoint_flags, rolling_growth_rate
from .anomalies import anomaly_flags


@dataclass(frozen=True)
class Alert:
    date: str
    region: str
    level: str     # "info"|"watch"|"warning"
    reason: str


def generate_alerts(master: pd.DataFrame) -> list[Alert]:
    """
    Rule-based alerting using cases, growth, anomalies, changepoints,
    PLUS optional ML features if present:
      - surge_prob_gb (0..1)
      - hosp_next7_pred (float)
    """
    alerts: list[Alert] = []
    df = master.copy()
    df["date"] = pd.to_datetime(df["date"])

    # thresholds (tune to taste)
    SURGE_PROB_WATCH = 0.55
    SURGE_PROB_WARNING = 0.70
    HOSP_PRED_WARNING = 25.0  # absolute placeholder; tune per disease/scale

    has_ml = ("surge_prob_gb" in df.columns) and ("hosp_next7_pred" in df.columns)

    for region, g in df.groupby("region"):
        g = g.sort_values("date")
        cases = g["cases"].astype(float)

        growth = rolling_growth_rate(cases, window=7)
        cp = simple_changepoint_flags(cases)
        anom = anomaly_flags(cases, threshold=3.0)

        for i in range(len(g)):
            d = g.iloc[i]["date"].date().isoformat()
            level = None
            reasons = []

            # Base rules
            if anom.iloc[i]:
                level = "watch"
                reasons.append("cases_anomaly")

            if cp.iloc[i]:
                level = "warning"
                reasons.append("changepoint_growth_shift")

            if pd.notna(growth.iloc[i]) and growth.iloc[i] > 0.05:
                level = "warning" if level in (None, "watch") else level
                reasons.append("sustained_growth")

            # ML escalation (if available)
            if has_ml:
                surge_prob = float(g.iloc[i]["surge_prob_gb"]) if pd.notna(g.iloc[i]["surge_prob_gb"]) else None
                hosp_pred = float(g.iloc[i]["hosp_next7_pred"]) if pd.notna(g.iloc[i]["hosp_next7_pred"]) else None

                if surge_prob is not None:
                    if surge_prob >= SURGE_PROB_WARNING:
                        level = "warning"
                        reasons.append(f"ml_surge_prob>= {SURGE_PROB_WARNING:.2f}")
                    elif surge_prob >= SURGE_PROB_WATCH and level is None:
                        level = "watch"
                        reasons.append(f"ml_surge_prob>= {SURGE_PROB_WATCH:.2f}")

                if hosp_pred is not None and hosp_pred >= HOSP_PRED_WARNING:
                    level = "warning"
                    reasons.append(f"ml_hosp_next7_pred>= {HOSP_PRED_WARNING:.0f}")

            if level:
                alerts.append(Alert(date=d, region=region, level=level, reason=";".join(reasons)))

    dedup = {(a.date, a.region, a.level, a.reason): a for a in alerts}
    return list(dedup.values())
