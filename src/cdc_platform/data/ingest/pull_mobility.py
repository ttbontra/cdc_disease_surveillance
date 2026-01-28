from __future__ import annotations
import numpy as np
import pandas as pd

def pull_mobility(start_date: str, end_date: str, regions: list[str] | None = None) -> pd.DataFrame:
    rng = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="D")
    regions = regions or ["RegionA", "RegionB", "RegionC"]
    rows = []
    rs = np.random.RandomState(123)
    for region in regions:
        # mobility index centered at 1.0
        vals = np.clip(rs.normal(1.0, 0.08, size=len(rng)), 0.7, 1.3)
        for d, v in zip(rng, vals):
            rows.append({"date": d.date().isoformat(), "region": region, "mobility_index": float(v)})
    return pd.DataFrame(rows)
