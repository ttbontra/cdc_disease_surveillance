from __future__ import annotations
import numpy as np
import pandas as pd

def estimate_rt_from_cases(cases: pd.Series, serial_interval_days: float = 5.0, eps: float = 1e-6) -> pd.Series:
    """
    Simple growth-rate proxy for Rt:
    Rt_t ≈ exp( g_t * SI ), where g_t = log(c_t / c_{t-1})
    """
    c = cases.astype(float).clip(lower=0.0)
    g = np.log((c + eps) / (c.shift(1) + eps))
    rt = np.exp(g * serial_interval_days)
    return rt.clip(lower=0.0, upper=10.0)
