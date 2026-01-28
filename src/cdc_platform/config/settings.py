from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

def _env(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key)
    return v if v is not None and v != "" else default

@dataclass(frozen=True)
class Settings:
    # Paths
    repo_root: Path = Path(__file__).resolve().parents[3]
    data_dir: Path = Path(_env("DATA_DIR", "")) if _env("DATA_DIR", "") else Path(__file__).resolve().parents[3] / "data"
    processed_dir: Path = Path(_env("PROCESSED_DIR", "")) if _env("PROCESSED_DIR", "") else Path(__file__).resolve().parents[3] / "data" / "processed"

    # API
    api_host: str = _env("API_HOST", "0.0.0.0") or "0.0.0.0"
    api_port: int = int(_env("API_PORT", "8000") or 8000)

    # Dashboard
    dashboard_title: str = _env("DASHBOARD_TITLE", "Infectious Disease Surveillance Platform") or "Infectious Disease Surveillance Platform"

    # Modeling defaults
    forecast_horizon_days: int = int(_env("FORECAST_HORIZON_DAYS", "28") or 28)
    nowcast_reporting_delay_days: int = int(_env("NOWCAST_DELAY_DAYS", "7") or 7)

    # Privacy / aggregation
    min_cell_count: int = int(_env("MIN_CELL_COUNT", "11") or 11)

settings = Settings()
