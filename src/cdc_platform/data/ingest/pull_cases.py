from __future__ import annotations
import numpy as np
import pandas as pd

def pull_cases(start_date: str, end_date: str, regions: list[str] | None = None) -> pd.DataFrame:
    """
    Demo ingest: generates synthetic daily cases by region.
    Replace this with real ingestion (API/db/files) later.
    """
    rng = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="D")
    regions = regions or ["RegionA", "RegionB", "RegionC"]

    rows = []
    rs = np.random.RandomState(42)
    for region in regions:
        base = rs.randint(30, 120)
        trend = rs.uniform(0.98, 1.03)
        noise = rs.normal(0, 8, size=len(rng))
        vals = np.maximum(0, base * (trend ** np.arange(len(rng))) + noise).round().astype(int)
        for d, v in zip(rng, vals):
            rows.append({"date": d.date().isoformat(), "region": region, "cases": int(v)})
    return pd.DataFrame(rows)
