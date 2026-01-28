from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

def _task():
    from cdc_platform.common.io import read_csv
    from cdc_platform.config.settings import settings
    from cdc_platform.serving.jobs.hourly_alerts import run_hourly_alerts

    master_path = settings.processed_dir / "master.csv"
    if not master_path.exists():
        return
    df = read_csv(master_path)
    alerts_df = run_hourly_alerts(df)
    alerts_df.to_csv(settings.processed_dir / "alerts.csv", index=False)

with DAG(
    dag_id="alerts_hourly",
    start_date=datetime(2025, 1, 1),
    schedule="15 * * * *",
    catchup=False,
    default_args={"retries": 0},
) as dag:
    alerts = PythonOperator(task_id="run_alerts_hourly", python_callable=_task)
