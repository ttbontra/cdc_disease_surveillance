from __future__ import annotations
import pandas as pd
from ...data.ingest.pull_cases import pull_cases
from ...data.ingest.pull_hosp import pull_hosp
from ...data.ingest.pull_wastewater import pull_wastewater
from ...data.ingest.pull_mobility import pull_mobility
from ...data.features.build_features import build_master_table
from ...data.cleaning.backfill_delays import simple_delay_adjustment
from ...config.settings import settings

def run_nightly_pipeline(start_date: str, end_date: str) -> pd.DataFrame:
    cases = pull_cases(start_date, end_date)
    hosp = pull_hosp(start_date, end_date)
    ww = pull_wastewater(start_date, end_date)
    mob = pull_mobility(start_date, end_date)

    # basic delay adjustment on cases
    cases = simple_delay_adjustment(cases, value_col="cases", delay_days=settings.nowcast_reporting_delay_days)

    master = build_master_table(cases, hosp, ww, mob)
    try:
        from ..registry.model_registry import ml_artifacts_path
        from ..registry.artifacts_ml import load_ml_artifacts
        from ...modeling.risk_scoring.sklearn_models import score_latest

        path = ml_artifacts_path()
        if path.exists():
            artifacts = load_ml_artifacts(path)
            latest_scores = score_latest(master, artifacts)
            # Join latest scores back onto master for the latest day only
            # We'll merge on region and date, leaving historical rows unchanged.
            master = master.merge(latest_scores, on=["date", "region"], how="left")
    except Exception:
        # Keep pipeline resilient; ML is optional.
        pass

    return master

