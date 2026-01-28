from __future__ import annotations

from pathlib import Path
import pandas as pd

from cdc_platform.serving.jobs.nightly_pipeline import run_nightly_pipeline
from cdc_platform.common.io import write_csv
from cdc_platform.config.settings import settings


def main():
    master = run_nightly_pipeline("2025-09-01", "2026-01-01")
    out = settings.processed_dir / "master.csv"
    write_csv(master, out)
    print(f"Wrote demo master to: {out}")

if __name__ == "__main__":
    main()
