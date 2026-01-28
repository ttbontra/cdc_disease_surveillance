from __future__ import annotations

from pathlib import Path
import joblib

from ...common.io import ensure_dir
from cdc_platform.modeling.bayes.hierarchical_growth import BayesHierarchicalResult


def save_bayes_result(path: Path, result: BayesHierarchicalResult) -> None:
    ensure_dir(path.parent)
    joblib.dump(result, path)


def load_bayes_result(path: Path) -> BayesHierarchicalResult:
    return joblib.load(path)
