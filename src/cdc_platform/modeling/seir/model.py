from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SEIRParams:
    beta: float   # transmission rate
    sigma: float  # incubation rate (1/incubation_period)
    gamma: float  # recovery rate (1/infectious_period)

def simulate_seir(pop: int, S0: float, E0: float, I0: float, R0: float,
                  params: SEIRParams, days: int, dt: float = 1.0) -> dict[str, np.ndarray]:
    """
    Simple Euler integration; stable enough for daily steps in demo.
    Returns arrays for S, E, I, R.
    """
    n_steps = int(days / dt) + 1
    S = np.zeros(n_steps); E = np.zeros(n_steps); I = np.zeros(n_steps); R = np.zeros(n_steps)
    S[0], E[0], I[0], R[0] = S0, E0, I0, R0

    for t in range(1, n_steps):
        s, e, i, r = S[t-1], E[t-1], I[t-1], R[t-1]
        beta, sigma, gamma = params.beta, params.sigma, params.gamma

        new_exposed = beta * s * i / pop
        new_infectious = sigma * e
        new_recovered = gamma * i

        S[t] = max(0.0, s - new_exposed * dt)
        E[t] = max(0.0, e + (new_exposed - new_infectious) * dt)
        I[t] = max(0.0, i + (new_infectious - new_recovered) * dt)
        R[t] = max(0.0, r + new_recovered * dt)

    return {"S": S, "E": E, "I": I, "R": R}

def rt_from_params(params: SEIRParams) -> float:
    # In classic SIR/SEIR with constant params, R0 approx beta/gamma
    return float(params.beta / params.gamma)
