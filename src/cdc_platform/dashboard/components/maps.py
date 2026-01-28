# src/cdc_platform/dashboard/components/maps.py
from __future__ import annotations

import pandas as pd


def prep_map_table(df: pd.DataFrame, region_col: str = "region") -> pd.DataFrame:
    """
    Placeholder: In a real project you'd join to a shapefile/geojson and render choropleths.
    """
    return df[[region_col]].drop_duplicates().reset_index(drop=True)
