from __future__ import annotations

from pathlib import Path
import pandas as pd

from cdc_platform.common.io import read_csv
from cdc_platform.config.settings import settings
from cdc_platform.modeling.early_warning.alert_rules import generate_alerts


def main():
    master_path = settings.processed_dir / "master.csv"
    if not master_path.exists():
        raise SystemExit("Missing master.csv. Run `cdc ingest ...` or scripts/seed_demo_data.py first.")

    df = read_csv(master_path)
    alerts = generate_alerts(df)
    out_dir = settings.repo_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([a.__dict__ for a in alerts]).to_csv(out_dir / "alerts_report.csv", index=False)
    print(f"Wrote: {out_dir / 'alerts_report.csv'}")

if __name__ == "__main__":
    main()
