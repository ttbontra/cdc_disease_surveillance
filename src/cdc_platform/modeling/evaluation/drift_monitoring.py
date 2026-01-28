# src/cdc_platform/modeling/evaluation/drift_monitoring.py
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """
    Population Stability Index (PSI) for drift monitoring.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # quantile bins from expected
    qs = np.linspace(0, 1, bins + 1)
    cut = np.quantile(expected, qs)
    # ensure strict monotonic
    cut = np.unique(cut)
    if len(cut) < 3:
        return 0.0

    def bucket(arr):
        return np.clip(np.digitize(arr, cut[1:-1], right=True), 0, len(cut) - 2)

    e_b = bucket(expected)
    a_b = bucket(actual)

    e_pct = np.bincount(e_b, minlength=len(cut) - 1) / max(len(expected), 1)
    a_pct = np.bincount(a_b, minlength=len(cut) - 1) / max(len(actual), 1)

    e_pct = np.clip(e_pct, eps, 1)
    a_pct = np.clip(a_pct, eps, 1)

    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in reference.columns or c not in current.columns:
            continue
        val = psi(reference[c].dropna().values, current[c].dropna().values)
        rows.append({"feature": c, "psi": val})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
