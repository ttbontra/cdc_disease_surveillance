from __future__ import annotations
import numpy as np
import pandas as pd
from .train import RiskModel

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def score_risk(df: pd.DataFrame, model: RiskModel, mu: np.ndarray, sd: np.ndarray) -> pd.DataFrame:
    X = df[model.feature_cols].astype(float).values
    sd2 = sd.copy()
    sd2[sd2 == 0] = 1.0
    Xn = (X - mu) / sd2
    p = sigmoid(Xn @ model.weights + model.bias)
    out = df[["date", "region"]].copy()
    out["risk_score"] = p
    return out
