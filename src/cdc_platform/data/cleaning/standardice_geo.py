from __future__ import annotations
import pandas as pd

def standardize_region(df: pd.DataFrame, col: str = "region") -> pd.DataFrame:
    out = df.copy()
    out[col] = out[col].astype(str).str.strip()
    return out
