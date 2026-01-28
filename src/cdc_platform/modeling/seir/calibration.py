from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .model import SEIRParams, simulate_seir

@dataclass(frozen=True)
class CalibResult:
    params: SEIRParams
    s0: float
    e0: float
    i0: float
    r0: float
    pop: int

def calibrate_seir_to_cases(daily_cases: pd.Series, pop: int,
                            incubation_days: float = 4.0, infectious_days: float = 6.0) -> CalibResult:
    """
    Lightweight calibration: grid-search beta to match observed case curve shape.
    This is intentionally simple for a portfolio repo (swap with PyMC/Stan later).
    """
    y = daily_cases.fillna(0).astype(float).values
    days = len(y)

    sigma = 1.0 / incubation_days
    gamma = 1.0 / infectious_days

    # initial conditions
    i0 = max(1.0, float(y[:7].mean()))
    e0 = i0 * 2.0
    r0 = 0.0
    s0 = float(pop) - e0 - i0 - r0

    # fit beta via coarse grid
    betas = np.linspace(0.05, 1.0, 60)
    best_beta = 0.2
    best_loss = float("inf")

    # We map model infectious -> "cases" via a proportionality constant k fitted by least squares
    for beta in betas:
        params = SEIRParams(beta=float(beta), sigma=float(sigma), gamma=float(gamma))
        sim = simulate_seir(pop=pop, S0=s0, E0=e0, I0=i0, R0=r0, params=params, days=days-1)
        I = sim["I"]
        # Fit k to map I to cases
        k = (y @ I) / (I @ I + 1e-9)
        pred = k * I
        loss = float(np.mean((y - pred) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_beta = float(beta)

    return CalibResult(
        params=SEIRParams(beta=best_beta, sigma=sigma, gamma=gamma),
        s0=s0, e0=e0, i0=i0, r0=r0, pop=pop
    )
