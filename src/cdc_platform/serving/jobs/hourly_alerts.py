# src/cdc_platform/serving/jobs/hourly_alerts.py
from __future__ import annotations

import pandas as pd

from ...modeling.early_warning.alert_rules import generate_alerts


def run_hourly_alerts(master: pd.DataFrame) -> pd.DataFrame:
    """
    In a real system: push to Slack/Teams/email, create tickets, etc.
    Here: return a DataFrame of current alerts.
    """
    alerts = generate_alerts(master)
    return pd.DataFrame([a.__dict__ for a in alerts])
