from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import joblib

from ...common.io import ensure_dir
from ...modeling.risk_scoring.sklearn_models import MLArtifacts


def save_ml_artifacts(path: Path, artifacts: MLArtifacts) -> None:
    ensure_dir(path.parent)
    # joblib can persist sklearn pipelines
    joblib.dump(artifacts, path)


def load_ml_artifacts(path: Path) -> MLArtifacts:
    return joblib.load(path)
