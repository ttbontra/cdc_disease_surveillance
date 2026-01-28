from __future__ import annotations
import pandas as pd
from ..cleaning.impute_missing import impute_missing_daily
from .lag_features import add_lags
from .mobility_features import add_mobility_rollups

def build_master_table(cases: pd.DataFrame, hosp: pd.DataFrame, ww: pd.DataFrame, mob: pd.DataFrame) -> pd.DataFrame:
    # Standardize types
    for df in (cases, hosp, ww, mob):
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    master = cases.merge(hosp, on=["date", "region"], how="left") \
                  .merge(ww, on=["date", "region"], how="left") \
                  .merge(mob, on=["date", "region"], how="left")

    master = impute_missing_daily(master, value_cols=["cases", "hosp", "ww_viral_load", "mobility_index"])
    master = add_lags(master, group_col="region", date_col="date", value_col="cases", lags=(1,7,14))
    master = add_mobility_rollups(master, group_col="region", date_col="date", mob_col="mobility_index")
    return master
