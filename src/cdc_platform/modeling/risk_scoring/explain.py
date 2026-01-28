from __future__ import annotations
import numpy as np
import pandas as pd
from .train import RiskModel

def linear_contributions(df: pd.DataFrame, model: RiskModel, mu: np.ndarray, sd: np.ndarray) -> pd.DataFrame:
    X = df[model.feature_cols].astype(float).values
    sd2 = sd.copy()
    sd2[sd2 == 0] = 1.0
    Xn = (X - mu) / sd2
    contrib = Xn * model.weights.reshape(1, -1)
    out = pd.DataFrame(contrib, columns=[f"contrib_{c}" for c in model.feature_cols])
    return out
