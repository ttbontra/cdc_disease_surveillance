from __future__ import annotations
import pandas as pd

def add_mobility_rollups(df: pd.DataFrame, group_col="region", date_col="date", mob_col="mobility_index") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values([group_col, date_col])
    out[f"{mob_col}_ma7"] = out.groupby(group_col)[mob_col].transform(lambda s: s.rolling(7, min_periods=1).mean())
    out[date_col] = out[date_col].dt.date.astype(str)
    return out
