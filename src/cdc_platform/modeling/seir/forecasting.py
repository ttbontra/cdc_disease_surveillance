from __future__ import annotations
import numpy as np
import pandas as pd
from .calibration import calibrate_seir_to_cases
from .model import simulate_seir, rt_from_params

def forecast_cases_seir_with_uncertainty(
    daily_cases: pd.Series,
    pop: int,
    horizon_days: int = 28,
    n_samples: int = 200,
    beta_sd_frac: float = 0.15,
    quantiles=(0.1, 0.5, 0.9),
) -> dict:
    """
    Uncertainty bands via sampling beta around calibrated beta.
    This is a lightweight, portfolio-safe approach that gives useful bands.

    Returns:
      - rt0_est
      - history_qXX arrays
      - forecast_qXX arrays
    """
    daily_cases = daily_cases.fillna(0).astype(float)
    calib = calibrate_seir_to_cases(daily_cases, pop=pop)

    # Base sim length
    total_days = (len(daily_cases) - 1) + horizon_days
    y = daily_cases.values

    # initial fit constant mapping I->cases using base beta
    base_sim = simulate_seir(
        pop=calib.pop,
        S0=calib.s0, E0=calib.e0, I0=calib.i0, R0=calib.r0,
        params=calib.params,
        days=total_days
    )
    I_base = base_sim["I"]
    I_hist = I_base[: len(y)]
    k = (y @ I_hist) / (I_hist @ I_hist + 1e-9)

    # Sample betas
    beta0 = float(calib.params.beta)
    sd = max(1e-6, abs(beta0) * beta_sd_frac)

    preds = []
    rng = np.random.RandomState(123)
    for _ in range(int(n_samples)):
        beta_s = max(1e-6, float(rng.normal(loc=beta0, scale=sd)))
        params_s = type(calib.params)(beta=beta_s, sigma=calib.params.sigma, gamma=calib.params.gamma)

        sim = simulate_seir(
            pop=calib.pop,
            S0=calib.s0, E0=calib.e0, I0=calib.i0, R0=calib.r0,
            params=params_s,
            days=total_days
        )
        I = sim["I"]
        pred = k * I
        preds.append(pred)

    preds = np.stack(preds, axis=0)  # (S, T)

    qs = {}
    for q in quantiles:
        qs[q] = np.quantile(preds, q=q, axis=0)

    out = {
        "rt0_est": rt_from_params(calib.params),
        "quantiles": quantiles,
        "history": {q: qs[q][: len(y)] for q in quantiles},
        "forecast": {q: qs[q][len(y):] for q in quantiles},
    }
    return out