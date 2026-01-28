# src/cdc_platform/modeling/evaluation/forecast_metrics.py
from __future__ import annotations

import numpy as np


def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def wis(y_true: np.ndarray, quantiles: dict[float, np.ndarray]) -> float:
    """
    Weighted Interval Score (simplified) using provided quantiles.
    quantiles: {q: pred_at_q}, e.g. {0.1: ..., 0.5: ..., 0.9: ...}
    """
    y = np.asarray(y_true, dtype=float)
    qs = sorted(quantiles.keys())
    # require median
    if 0.5 not in quantiles:
        raise ValueError("quantiles must include 0.5 (median).")

    med = np.asarray(quantiles[0.5], dtype=float)
    score = np.mean(np.abs(y - med))

    # add interval penalties
    for q in qs:
        if q == 0.5:
            continue
        qv = np.asarray(quantiles[q], dtype=float)
        # pair symmetric quantiles if possible
    # symmetric pairing
    pairs = []
    for q in qs:
        if q < 0.5 and (1 - q) in quantiles:
            pairs.append((q, 1 - q))
    for qlo, qhi in pairs:
        lo = np.asarray(quantiles[qlo], dtype=float)
        hi = np.asarray(quantiles[qhi], dtype=float)
        alpha = 2 * qlo
        width = hi - lo
        below = (lo - y) * (y < lo)
        above = (y - hi) * (y > hi)
        interval = np.mean(width + (2 / alpha) * (below + above))
        score += interval

    return float(score)
