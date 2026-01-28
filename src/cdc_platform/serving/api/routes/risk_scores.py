from __future__ import annotations
import pandas as pd
from fastapi import APIRouter, HTTPException

from ..schemas import RiskScoresResponse
from ...registry.model_registry import ml_artifacts_path
from ...registry.artifacts_ml import load_ml_artifacts
from ....modeling.risk_scoring.sklearn_models import score_latest

router = APIRouter()
MASTER_CACHE: pd.DataFrame | None = None

def set_master(df: pd.DataFrame) -> None:
    global MASTER_CACHE
    MASTER_CACHE = df.copy()

@router.get("/risk-scores", response_model=RiskScoresResponse)
def risk_scores():
    if MASTER_CACHE is None:
        raise HTTPException(500, "MASTER_CACHE not initialized. Run pipeline first.")

    path = ml_artifacts_path()
    if not path.exists():
        raise HTTPException(500, "ML artifacts missing. Train them via CLI: `cdc train-ml`.")

    artifacts = load_ml_artifacts(path)
    scores = score_latest(MASTER_CACHE, artifacts)
    return RiskScoresResponse(scores=scores.to_dict(orient="records"))
