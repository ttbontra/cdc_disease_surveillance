from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

def _task():
    from cdc_platform.common.io import read_csv
    from cdc_platform.config.settings import settings
    from cdc_platform.modeling.nowcasting.nowcast import nowcast_latest_cases
    import pandas as pd

    master_path = settings.processed_dir / "master.csv"
    if not master_path.exists():
        return
    df = read_csv(master_path)
    out = []
    for region, g in df.groupby("region"):
        out.append({"region": region, **nowcast_latest_cases(g)})
    # write to processed for dashboard/API usage
    pd.DataFrame(out).to_csv(settings.processed_dir / "nowcast.csv", index=False)

with DAG(
    dag_id="nowcast_hourly",
    start_date=datetime(2025, 1, 1),
    schedule="0 * * * *",
    catchup=False,
    default_args={"retries": 0},
) as dag:
    nowcast = PythonOperator(task_id="run_nowcast_hourly", python_callable=_task)
