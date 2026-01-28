
from __future__ import annotations

import typer
import pandas as pd

from ...common.io import read_csv, write_csv
from ...config.settings import settings
from ...modeling.evaluation.backtesting import rolling_backtest_seir


def run_backtests_cmd(population: int = typer.Option(1_000_000), horizon_days: int = typer.Option(14)):
    master_path = settings.processed_dir / "master.csv"
    if not master_path.exists():
        raise typer.BadParameter("Missing master.csv. Run `cdc ingest --start ... --end ...` first.")
    df = read_csv(master_path)
    df["date"] = pd.to_datetime(df["date"])

    frames = []
    for region, g in df.groupby("region"):
        bt = rolling_backtest_seir(g, pop=int(population), horizon_days=int(horizon_days))
        bt["region"] = region
        frames.append(bt)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = settings.repo_root / "reports" / "backtests.csv"
    write_csv(out, out_path)
    typer.echo(f"Wrote: {out_path}")
