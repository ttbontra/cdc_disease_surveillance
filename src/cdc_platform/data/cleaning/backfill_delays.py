from __future__ import annotations
import pandas as pd

def simple_delay_adjustment(df: pd.DataFrame, value_col: str, delay_days: int = 7) -> pd.DataFrame:
    """
    Placeholder nowcast adjustment:
    shifts recent values upward to approximate under-reporting.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    max_date = out["date"].max()
    cutoff = max_date - pd.Timedelta(days=delay_days)
    # inflate last 'delay_days' by a ramp factor
    mask = out["date"] > cutoff
    if mask.any():
        n = mask.sum()
        ramp = pd.Series(range(n), index=out.index[mask]).astype(float)
        ramp = 1.0 + (delay_days - ramp.values) / (delay_days * 5.0)  # mild inflation
        out.loc[mask, value_col] = (out.loc[mask, value_col].values * ramp).round()
    out["date"] = out["date"].dt.date.astype(str)
    return out
