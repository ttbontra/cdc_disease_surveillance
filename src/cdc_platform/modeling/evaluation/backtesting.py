# src/cdc_platform/modeling/evaluation/backtesting.py
from __future__ import annotations

import pandas as pd

from ..seir.forecasting import forecast_cases_seir
from ...common.metrics import mae


def rolling_backtest_seir(
    df_region: pd.DataFrame,
    pop: int,
    horizon_days: int = 14,
    min_history_days: int = 60,
    step_days: int = 7,
) -> pd.DataFrame:
    """
    Rolling-origin backtest for SEIR forecast (portfolio demo).
    Returns a DataFrame of cutoffs and MAE.
    """
    g = df_region.copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date")

    results = []
    for cutoff_idx in range(min_history_days, len(g) - horizon_days, step_days):
        hist = g.iloc[:cutoff_idx]
        fut = g.iloc[cutoff_idx : cutoff_idx + horizon_days]
        out = forecast_cases_seir(hist["cases"], pop=pop, horizon_days=horizon_days)
        pred = out["forecast"][:horizon_days]
        err = mae(fut["cases"].values, pred)
        results.append(
            {
                "cutoff_date": hist["date"].max().date().isoformat(),
                "horizon_days": horizon_days,
                "mae": err,
            }
        )

    return pd.DataFrame(results)
