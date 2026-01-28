from __future__ import annotations
from pydantic import BaseModel
from typing import List, Optional

class ForecastRequest(BaseModel):
    region: str
    horizon_days: int = 28
    population: int = 1_000_000

class ForecastResponse(BaseModel):
    region: str
    rt0_est: float
    history_pred: List[float]
    forecast: List[float]

class AlertsResponse(BaseModel):
    alerts: List[dict]

class RiskScoresResponse(BaseModel):
    scores: List[dict]
