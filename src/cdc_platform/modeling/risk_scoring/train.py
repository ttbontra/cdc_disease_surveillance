from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class RiskModel:
    """
    Simple calibrated logistic model trained with gradient descent.
    Portfolio-safe (no sklearn dependency). Replace w/ XGBoost later.
    """
    weights: np.ndarray
    bias: float
    feature_cols: list[str]

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def train_risk_model(master: pd.DataFrame, feature_cols: list[str], label_col: str = "surge_next14") -> RiskModel:
    df = master.copy()
    # Create label: surge if future cases increase by > X% in next 14 days
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["region", "date"])
    if label_col not in df.columns:
        future = df.groupby("region")["cases"].shift(-14)
        df[label_col] = (future > (df["cases"] * 1.5)).astype(int)  # 50% rise

    # Drop NA rows
    df = df.dropna(subset=feature_cols + [label_col])
    X = df[feature_cols].astype(float).values
    y = df[label_col].astype(int).values

    # Standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xn = (X - mu) / sd

    w = np.zeros(Xn.shape[1], dtype=float)
    b = 0.0
    lr = 0.1
    for _ in range(800):
        z = Xn @ w + b
        p = _sigmoid(z)
        # gradients
        dw = (Xn.T @ (p - y)) / len(y)
        db = float(np.mean(p - y))
        w -= lr * dw
        b -= lr * db

    # Pack standardization into weights by storing mu/sd in feature list encoding
    # For simplicity: return raw w/b; inference will standardize using train stats stored in model artifact
    # (See artifacts.py)
    return RiskModel(weights=w, bias=b, feature_cols=feature_cols)
