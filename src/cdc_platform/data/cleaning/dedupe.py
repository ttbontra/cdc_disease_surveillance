from __future__ import annotations
import pandas as pd

def dedupe(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
