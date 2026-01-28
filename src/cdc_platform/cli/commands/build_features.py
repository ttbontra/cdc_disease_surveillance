from __future__ import annotations
import typer
import pandas as pd
from ...common.io import read_csv, write_csv
from ...config.settings import settings

def features_cmd():
    path = settings.processed_dir / "master.csv"
    if not path.exists():
        raise typer.BadParameter("Missing master.csv. Run `cdc ingest --start ... --end ...` first.")
    df = read_csv(path)
    # already feature-built in nightly pipeline; in a real build you'd expand here
    out_path = settings.processed_dir / "features.csv"
    write_csv(df, out_path)
    typer.echo(f"Wrote: {out_path}")
