from __future__ import annotations
import pandas as pd

def suppress_small_cells(df: pd.DataFrame, count_col: str, min_cell_count: int) -> pd.DataFrame:
    out = df.copy()
    mask = out[count_col] < min_cell_count
    # Replace small cells with NA (typical suppression)
    for c in out.columns:
        if c != count_col:
            out.loc[mask, c] = pd.NA
    out.loc[mask, count_col] = pd.NA
    return out
