from __future__ import annotations

import numpy as np
import pandas as pd

from ...modeling.risk_scoring.train import train_risk_model
from ..registry.model_registry import risk_model_path
from ..registry.artifacts import save_risk_model


def retrain_risk_model(master: pd.DataFrame) -> str:
    """
    Weekly retrain (demo): trains the logistic risk model and writes artifact.
    Returns artifact path string.
    """
    feature_cols = [
        "cases_lag1", "cases_lag7", "cases_lag14",
        "mobility_index", "mobility_index_ma7",
        "ww_viral_load",
    ]

    train_df = master.dropna(subset=feature_cols).copy()
    X = train_df[feature_cols].astype(float).values
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0

    model = train_risk_model(master, feature_cols=feature_cols)
    path = risk_model_path()
    save_risk_model(path, model, mu=mu, sd=sd)
    return str(path)
