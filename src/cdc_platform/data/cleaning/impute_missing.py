from __future__ import annotations
import pandas as pd

def impute_missing_daily(df: pd.DataFrame, date_col="date", group_col="region", value_cols=None) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    value_cols = value_cols or [c for c in out.columns if c not in [date_col, group_col]]

    frames = []
    for region, g in out.groupby(group_col):
        g = g.sort_values(date_col)
        full = pd.date_range(g[date_col].min(), g[date_col].max(), freq="D")
        g2 = g.set_index(date_col).reindex(full).rename_axis(date_col).reset_index()
        g2[group_col] = region
        for c in value_cols:
            g2[c] = g2[c].interpolate(limit_direction="both")
        frames.append(g2)
    out2 = pd.concat(frames, ignore_index=True)
    out2[date_col] = out2[date_col].dt.date.astype(str)
    return out2
