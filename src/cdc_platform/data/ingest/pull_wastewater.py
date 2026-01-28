from __future__ import annotations
import numpy as np
import pandas as pd

def pull_wastewater(start_date: str, end_date: str, regions: list[str] | None = None) -> pd.DataFrame:
    rng = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="D")
    regions = regions or ["RegionA", "RegionB", "RegionC"]
    rows = []
    rs = np.random.RandomState(99)
    for region in regions:
        base = rs.uniform(1.0, 3.0)
        trend = rs.uniform(0.99, 1.04)
        noise = rs.normal(0, 0.2, size=len(rng))
        vals = np.maximum(0, base * (trend ** np.arange(len(rng))) + noise)
        for d, v in zip(rng, vals):
            rows.append({"date": d.date().isoformat(), "region": region, "ww_viral_load": float(v)})
    return pd.DataFrame(rows)
