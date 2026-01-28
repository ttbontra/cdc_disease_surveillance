# src/cdc_platform/dashboard/components/charts.py
from __future__ import annotations

import pandas as pd


def tidy_timeseries(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out.sort_values(date_col)
