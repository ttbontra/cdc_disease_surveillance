# src/cdc_platform/modeling/nowcasting/nowcast.py
from __future__ import annotations

import numpy as np
import pandas as pd

from .delay_model import fit_reporting_delay_distribution, apply_delay_nowcast


def nowcast_latest_cases(df_region: pd.DataFrame, value_col: str = "cases", max_delay_days: int = 14) -> dict:
    """
    Returns a small dict with nowcasted latest value and the delay distribution.
    """
    g = df_region.copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date")
    y = g[value_col].astype(float).values

    w = fit_reporting_delay_distribution(pd.Series(y), max_delay_days=max_delay_days)
    nc = apply_delay_nowcast(y, w)

    return {
        "latest_date": g["date"].max().date().isoformat(),
        "observed_latest": float(y[-1]) if len(y) else 0.0,
        "nowcast_latest": float(nc),
        "delay_weights": w.tolist(),
    }
