from __future__ import annotations
import numpy as np
import pandas as pd
from .calibration import calibrate_seir_to_cases
from .model import simulate_seir, rt_from_params

def forecast_cases_seir(daily_cases: pd.Series, pop: int, horizon_days: int = 28) -> dict:
    daily_cases = daily_cases.fillna(0).astype(float)
    calib = calibrate_seir_to_cases(daily_cases, pop=pop)

    sim = simulate_seir(
        pop=calib.pop,
        S0=calib.s0, E0=calib.e0, I0=calib.i0, R0=calib.r0,
        params=calib.params,
        days=len(daily_cases) - 1 + horizon_days
    )
    I = sim["I"]
    y = daily_cases.values
    # Fit k on history
    I_hist = I[: len(y)]
    k = (y @ I_hist) / (I_hist @ I_hist + 1e-9)
    pred_all = k * I

    hist_pred = pred_all[: len(y)]
    fut_pred = pred_all[len(y):]

    return {
        "rt0_est": rt_from_params(calib.params),
        "history_pred": hist_pred,
        "forecast": fut_pred,
    }
