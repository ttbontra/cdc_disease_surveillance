from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def to_date(x) -> pd.Timestamp:
    return pd.to_datetime(x).normalize()

def date_range(end_date: str, days: int) -> pd.DatetimeIndex:
    end = pd.to_datetime(end_date).normalize()
    start = end - pd.Timedelta(days=days - 1)
    return pd.date_range(start=start, end=end, freq="D")
