from __future__ import annotations
import pandas as pd

def add_lags(df: pd.DataFrame, group_col: str, date_col: str, value_col: str, lags=(1, 7, 14)) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values([group_col, date_col])
    for l in lags:
        out[f"{value_col}_lag{l}"] = out.groupby(group_col)[value_col].shift(l)
    out[date_col] = out[date_col].dt.date.astype(str)
    return out
