from __future__ import annotations

import typer

from ...common.io import read_csv
from ...config.settings import settings
from ...modeling.bayes.hierarchical_growth import fit_hierarchical_growth_model
from ...serving.registry.model_registry import bayes_artifacts_path
from ...serving.registry.artifacts_bayes import save_bayes_result


def train_bayes_cmd(draws: int = typer.Option(800), tune: int = typer.Option(800), min_days: int = typer.Option(60)):
    master_path = settings.processed_dir / "master.csv"
    if not master_path.exists():
        raise typer.BadParameter("Missing master.csv. Run `cdc ingest --start ... --end ...` first.")

    df = read_csv(master_path)
    result = fit_hierarchical_growth_model(df, min_days=int(min_days), draws=int(draws), tune=int(tune))
    save_bayes_result(bayes_artifacts_path(), result)
    typer.echo(f"Saved Bayesian result to: {bayes_artifacts_path()}")
