from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

def _task():
    # In real use, call your CLI or python entrypoint.
    # Example: run nightly pipeline + write master.csv
    from cdc_platform.serving.jobs.nightly_pipeline import run_nightly_pipeline
    from cdc_platform.common.io import write_csv
    from cdc_platform.config.settings import settings

    end = datetime.utcnow().date().isoformat()
    start = (datetime.utcnow().date() - timedelta(days=180)).isoformat()
    master = run_nightly_pipeline(start, end)
    write_csv(master, settings.processed_dir / "master.csv")

with DAG(
    dag_id="ingest_daily",
    start_date=datetime(2025, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
) as dag:
    ingest = PythonOperator(task_id="run_ingest_daily", python_callable=_task)
