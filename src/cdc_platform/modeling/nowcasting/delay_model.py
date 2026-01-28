# src/cdc_platform/modeling/nowcasting/delay_model.py
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_reporting_delay_distribution(cases: pd.Series, max_delay_days: int = 14) -> np.ndarray:
    """
    Portfolio-friendly placeholder:
    produces a simple geometric-ish delay distribution summing to 1.

    Replace with a Bayesian delay model (e.g., negative binomial / discrete hazard)
    if you want realism.
    """
    max_delay_days = int(max_delay_days)
    d = np.arange(0, max_delay_days + 1, dtype=float)
    # heavier weight on recent days
    lam = 0.35
    w = np.exp(-lam * d)
    w = w / w.sum()
    return w


def apply_delay_nowcast(observed_recent: np.ndarray, delay_weights: np.ndarray) -> float:
    """
    Estimate 'true' most-recent value from partially reported counts.
    Very simplified: divide by cumulative reporting fraction.
    """
    observed_recent = float(observed_recent[-1])
    cum_reported = float(delay_weights[0])  # day 0 fraction
    cum_reported = max(cum_reported, 1e-6)
    return observed_recent / cum_reported
