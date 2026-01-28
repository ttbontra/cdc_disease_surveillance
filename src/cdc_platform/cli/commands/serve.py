from __future__ import annotations
import typer
import uvicorn
import pandas as pd

from ...config.settings import settings
from ...common.io import read_csv
from ...serving.api.main import app, wire_master_cache

def serve_cmd():
    # Load cache if available
    master_path = settings.processed_dir / "master.csv"
    if master_path.exists():
        df = read_csv(master_path)
        wire_master_cache(df)

    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level="info")
