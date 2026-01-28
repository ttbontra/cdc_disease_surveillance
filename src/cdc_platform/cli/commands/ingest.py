from __future__ import annotations
import typer
from ...serving.jobs.nightly_pipeline import run_nightly_pipeline
from ...common.io import write_csv
from ...config.settings import settings

def ingest_cmd(start: str = typer.Option(...), end: str = typer.Option(...)):
    master = run_nightly_pipeline(start, end)
    out_path = settings.processed_dir / "master.csv"
    write_csv(master, out_path)
    typer.echo(f"Wrote: {out_path}")
