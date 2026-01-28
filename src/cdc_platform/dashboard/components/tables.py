# src/cdc_platform/dashboard/components/tables.py
from __future__ import annotations

import pandas as pd


def top_regions_by(df: pd.DataFrame, metric: str, n: int = 10) -> pd.DataFrame:
    cols = [c for c in ["region", metric] if c in df.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return df[cols].sort_values(metric, ascending=False).head(n).reset_index(drop=True)
