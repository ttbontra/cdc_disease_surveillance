from __future__ import annotations

import typer

from ...common.io import read_csv, write_csv
from ...config.settings import settings
from ...modeling.risk_scoring.sklearn_models import train_ml_models, score_latest
from ...serving.registry.model_registry import ml_artifacts_path
from ...serving.registry.artifacts_ml import save_ml_artifacts


def train_ml_cmd():
    master_path = settings.processed_dir / "master.csv"
    if not master_path.exists():
        raise typer.BadParameter("Missing master.csv. Run `cdc ingest --start ... --end ...` first.")

    df = read_csv(master_path)

    feature_cols = [
        "cases_lag1", "cases_lag7", "cases_lag14",
        "mobility_index", "mobility_index_ma7",
        "ww_viral_load",
    ]

    artifacts = train_ml_models(df, feature_cols=feature_cols)
    save_ml_artifacts(ml_artifacts_path(), artifacts)
    typer.echo(f"Saved ML artifacts to: {ml_artifacts_path()}")

    # quick scoring snapshot (optional)
    scores = score_latest(df, artifacts)
    out_path = settings.processed_dir / "ml_latest_scores.csv"
    write_csv(scores, out_path)
    typer.echo(f"Wrote latest ML scores to: {out_path}")
