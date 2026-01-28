from __future__ import annotations
import numpy as np
import pandas as pd

def rolling_growth_rate(series: pd.Series, window: int = 7, eps: float = 1e-6) -> pd.Series:
    s = series.astype(float).clip(lower=0.0)
    ma = s.rolling(window, min_periods=max(2, window//2)).mean()
    g = np.log((ma + eps) / (ma.shift(1) + eps))
    return g

def simple_changepoint_flags(series: pd.Series, z_thresh: float = 2.5) -> pd.Series:
    """
    Detect sudden shifts in growth rate via z-score of rolling growth.
    """
    g = rolling_growth_rate(series)
    mu = g.rolling(28, min_periods=10).mean()
    sd = g.rolling(28, min_periods=10).std().replace(0, np.nan)
    z = (g - mu) / sd
    return (z.abs() >= z_thresh).fillna(False)
