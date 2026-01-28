from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np

from ...common.io import ensure_dir
from ...modeling.risk_scoring.train import RiskModel

def save_risk_model(path: Path, model: RiskModel, mu: np.ndarray, sd: np.ndarray) -> None:
    ensure_dir(path.parent)
    payload = {
        "model": {"weights": model.weights.tolist(), "bias": model.bias, "feature_cols": model.feature_cols},
        "mu": mu.tolist(),
        "sd": sd.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2))

def load_risk_model(path: Path) -> tuple[RiskModel, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    m = payload["model"]
    model = RiskModel(weights=np.array(m["weights"], dtype=float), bias=float(m["bias"]), feature_cols=list(m["feature_cols"]))
    mu = np.array(payload["mu"], dtype=float)
    sd = np.array(payload["sd"], dtype=float)
    return model, mu, sd
