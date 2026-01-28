from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

def _task():
    from cdc_platform.common.io import read_csv
    from cdc_platform.config.settings import settings
    from cdc_platform.modeling.seir.forecasting import forecast_cases_seir
    import pandas as pd

    master_path = settings.processed_dir / "master.csv"
    if not master_path.exists():
        return
    df = read_csv(master_path)
    df["date"] = pd.to_datetime(df["date"])

    rows = []
    for region, g in df.groupby("region"):
        g = g.sort_values("date")
        out = forecast_cases_seir(g["cases"], pop=1_000_000, horizon_days=28)
        rows.append({"region": region, "rt0_est": out["rt0_est"], "forecast_sum_28d": float(sum(out["forecast"]))})

    pd.DataFrame(rows).to_csv(settings.processed_dir / "forecasts_summary.csv", index=False)

with DAG(
    dag_id="forecast_daily",
    start_date=datetime(2025, 1, 1),
    schedule="30 2 * * *",
    catchup=False,
    default_args={"retries": 0},
) as dag:
    forecast = PythonOperator(task_id="run_forecast_daily", python_callable=_task)
