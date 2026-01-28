from __future__ import annotations
import pandas as pd
from fastapi import APIRouter, HTTPException
from ..schemas import AlertsResponse
from ....modeling.early_warning.alert_rules import generate_alerts

router = APIRouter()

MASTER_CACHE: pd.DataFrame | None = None

def set_master(df: pd.DataFrame) -> None:
    global MASTER_CACHE
    MASTER_CACHE = df.copy()

@router.get("/alerts", response_model=AlertsResponse)
def alerts():
    if MASTER_CACHE is None:
        raise HTTPException(500, "MASTER_CACHE not initialized. Run pipeline first.")
    alerts = generate_alerts(MASTER_CACHE)
    return AlertsResponse(alerts=[a.__dict__ for a in alerts])
