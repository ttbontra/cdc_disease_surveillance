from __future__ import annotations
import typer
import numpy as np
import pandas as pd
from ...common.io import read_csv
from ...config.settings import settings
from ...modeling.risk_scoring.train import train_risk_model
from ...serving.registry.model_registry import risk_model_path
from ...serving.registry.artifacts import save_risk_model

def train_risk_cmd():
    path = settings.processed_dir / "master.csv"
    if not path.exists():
        raise typer.BadParameter("Missing master.csv. Run `cdc ingest --start ... --end ...` first.")
    df = read_csv(path)

    feature_cols = [
        "cases_lag1", "cases_lag7", "cases_lag14",
        "mobility_index", "mobility_index_ma7",
        "ww_viral_load"
    ]
    # Prepare training matrix + store standardization stats
    train_df = df.dropna(subset=feature_cols).copy()
    X = train_df[feature_cols].astype(float).values
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0

    model = train_risk_model(df, feature_cols=feature_cols)
    save_risk_model(risk_model_path(), model, mu=mu, sd=sd)
    typer.echo(f"Saved risk model to: {risk_model_path()}")
