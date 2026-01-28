from __future__ import annotations
from pathlib import Path
from ...config.settings import settings

def registry_dir() -> Path:
    p = settings.repo_root / "artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p

def risk_model_path() -> Path:
    return registry_dir() / "risk_model.json"

def ml_artifacts_path() -> Path:
    return registry_dir() / "ml_artifacts.joblib"

def bayes_artifacts_path() -> Path:
    return registry_dir() / "bayes_hierarchical.joblib"