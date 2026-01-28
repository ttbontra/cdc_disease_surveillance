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
    Rule-based alerting using cases, growth, anomalies, and changepoints.
    """
    alerts: list[Alert] = []
    df = master.copy()
    df["date"] = pd.to_datetime(df["date"])

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

            if anom.iloc[i]:
                level = "watch"
                reasons.append("cases_anomaly")
            if cp.iloc[i]:
                level = "warning"
                reasons.append("changepoint_growth_shift")
            if growth.iloc[i] > 0.05:  # ~5% daily growth proxy on smoothed values
                level = "warning" if level else "watch"
                reasons.append("sustained_growth")

            if level:
                alerts.append(Alert(date=d, region=region, level=level, reason=";".join(reasons)))

    # de-dupe by (date, region, level, reason)
    dedup = {(a.date, a.region, a.level, a.reason): a for a in alerts}
    return list(dedup.values())
