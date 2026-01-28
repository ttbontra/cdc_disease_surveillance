from __future__ import annotations
import numpy as np
import pandas as pd

def anomaly_score_iqr(series: pd.Series, window: int = 28) -> pd.Series:
    s = series.astype(float)
    med = s.rolling(window, min_periods=10).median()
    q1 = s.rolling(window, min_periods=10).quantile(0.25)
    q3 = s.rolling(window, min_periods=10).quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    score = (s - med).abs() / iqr
    return score.fillna(0.0)

def anomaly_flags(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    return anomaly_score_iqr(series) >= threshold
