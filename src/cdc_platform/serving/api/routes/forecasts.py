from __future__ import annotations
import pandas as pd
from fastapi import APIRouter, HTTPException
from ..schemas import ForecastRequest, ForecastResponse
from ....modeling.seir.forecasting import forecast_cases_seir

router = APIRouter()

# In a real system this comes from processed storage.
# For demo, we keep it in-memory via dependency injection or cached global.
MASTER_CACHE: pd.DataFrame | None = None

def set_master(df: pd.DataFrame) -> None:
    global MASTER_CACHE
    MASTER_CACHE = df.copy()

@router.post("/forecasts", response_model=ForecastResponse)
def forecasts(req: ForecastRequest):
    if MASTER_CACHE is None:
        raise HTTPException(500, "MASTER_CACHE not initialized. Run pipeline first.")
    df = MASTER_CACHE[MASTER_CACHE["region"] == req.region].sort_values("date")
    if df.empty:
        raise HTTPException(404, f"Unknown region: {req.region}")

    out = forecast_cases_seir(df["cases"], pop=req.population, horizon_days=req.horizon_days)
    return ForecastResponse(
        region=req.region,
        rt0_est=float(out["rt0_est"]),
        history_pred=[float(x) for x in out["history_pred"]],
        forecast=[float(x) for x in out["forecast"]],
    )
