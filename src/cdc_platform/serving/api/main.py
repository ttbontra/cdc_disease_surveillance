from __future__ import annotations
import pandas as pd
from fastapi import FastAPI

from .routes.health import router as health_router
from .routes.forecasts import router as forecasts_router, set_master as set_master_forecasts
from .routes.alerts import router as alerts_router, set_master as set_master_alerts
from .routes.risk_scores import router as risk_router, set_master as set_master_risk

app = FastAPI(title="CDC Surveillance Forecasting API", version="0.1.0")

app.include_router(health_router)
app.include_router(forecasts_router)
app.include_router(alerts_router)
app.include_router(risk_router)

def wire_master_cache(master: pd.DataFrame) -> None:
    set_master_forecasts(master)
    set_master_alerts(master)
    set_master_risk(master)
