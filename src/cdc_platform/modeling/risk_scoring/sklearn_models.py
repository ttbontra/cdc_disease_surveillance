from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor


@dataclass(frozen=True)
class MLArtifacts:
    surge_model_name: str
    hosp_model_name: str
    surge_pipeline: Pipeline
    hosp_pipeline: Pipeline
    feature_cols: List[str]
    surge_label_col: str
    hosp_label_col: str
    meta: Dict[str, Any]


def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["region", "date"])

    # Ensure numeric types
    for c in ["cases", "hosp", "ww_viral_load", "mobility_index"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def add_default_labels(df: pd.DataFrame,
                       surge_horizon_days: int = 14,
                       surge_multiplier: float = 1.5,
                       hosp_horizon_days: int = 7) -> pd.DataFrame:
    """
    Create:
      - surge_next14: 1 if cases in 14 days > 1.5x current
      - hosp_next7: hospitalization count in 7 days (regression target)
    """
    out = _make_features(df)

    future_cases = out.groupby("region")["cases"].shift(-surge_horizon_days)
    out["surge_next14"] = (future_cases > (out["cases"] * surge_multiplier)).astype(int)

    future_hosp = out.groupby("region")["hosp"].shift(-hosp_horizon_days)
    out["hosp_next7"] = future_hosp.astype(float)

    return out


def train_ml_models(master: pd.DataFrame,
                    feature_cols: List[str],
                    surge_label_col: str = "surge_next14",
                    hosp_label_col: str = "hosp_next7",
                    random_state: int = 42) -> MLArtifacts:
    df = add_default_labels(master)

    # Drop rows missing labels
    df = df.dropna(subset=feature_cols + [surge_label_col, hosp_label_col]).copy()

    X = df[feature_cols]
    y_surge = df[surge_label_col].astype(int)
    y_hosp = df[hosp_label_col].astype(float)

    # Preprocess: impute + scale numeric
    numeric_features = feature_cols
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
        ],
        remainder="drop"
    )

    # --- Surge models ---
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    gb = GradientBoostingClassifier(
        random_state=random_state
    )

    # Choose GB as primary for calibrated-ish probabilities, keep RF available for comparison
    # (We keep both by fitting two pipelines; artifact saving will store both.)
    surge_pipe_gb = Pipeline(steps=[("pre", pre), ("model", gb)])
    surge_pipe_rf = Pipeline(steps=[("pre", pre), ("model", rf)])

    surge_pipe_gb.fit(X, y_surge)
    surge_pipe_rf.fit(X, y_surge)

    # --- Hospitalization regression model ---
    # A regressor for expected hosp in 7 days (demo proxy for "hospitalization risk")
    hosp_reg = GradientBoostingRegressor(random_state=random_state)
    hosp_pipe = Pipeline(steps=[("pre", pre), ("model", hosp_reg)])
    hosp_pipe.fit(X, y_hosp)

    # Store both surge models inside a dict-like meta; we return GB pipeline as primary, RF as secondary
    meta = {
        "surge_models": ["GradientBoostingClassifier", "RandomForestClassifier"],
        "hosp_model": "GradientBoostingRegressor",
        "label_defs": {
            "surge_next14": "cases(t+14) > 1.5 * cases(t)",
            "hosp_next7": "hosp(t+7)"
        }
    }

    # We'll pack RF pipeline into the meta for saving/loading
    # and keep GB as primary in the dataclass fields.
    return MLArtifacts(
        surge_model_name="GradientBoostingClassifier",
        hosp_model_name="GradientBoostingRegressor",
        surge_pipeline=surge_pipe_gb,
        hosp_pipeline=hosp_pipe,
        feature_cols=feature_cols,
        surge_label_col=surge_label_col,
        hosp_label_col=hosp_label_col,
        meta={"secondary_surge_pipeline": surge_pipe_rf, **meta},
    )


def score_latest(master: pd.DataFrame, artifacts: MLArtifacts) -> pd.DataFrame:
    df = _make_features(master)
    # Latest row per region
    latest = df.sort_values("date").groupby("region").tail(1).copy()

    X = latest[artifacts.feature_cols]
    # Surge probability
    if hasattr(artifacts.surge_pipeline.named_steps["model"], "predict_proba"):
        surge_prob = artifacts.surge_pipeline.predict_proba(X)[:, 1]
    else:
        # fallback
        surge_prob = artifacts.surge_pipeline.predict(X).astype(float)

    # RF as secondary (optional)
    rf_pipe = artifacts.meta.get("secondary_surge_pipeline")
    if rf_pipe is not None and hasattr(rf_pipe.named_steps["model"], "predict_proba"):
        surge_prob_rf = rf_pipe.predict_proba(X)[:, 1]
    else:
        surge_prob_rf = np.full(len(latest), np.nan, dtype=float)

    # Expected hosp in 7 days
    hosp_pred = artifacts.hosp_pipeline.predict(X)

    out = latest[["date", "region"]].copy()
    out["surge_prob_gb"] = surge_prob
    out["surge_prob_rf"] = surge_prob_rf
    out["hosp_next7_pred"] = hosp_pred
    return out
