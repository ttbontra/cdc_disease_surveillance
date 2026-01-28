import pandas as pd
from cdc_platform.data.features.lag_features import add_lags

def test_add_lags():
    df = pd.DataFrame({
        "date": ["2026-01-01","2026-01-02","2026-01-03"],
        "region": ["A","A","A"],
        "cases": [10, 20, 30]
    })
    out = add_lags(df, group_col="region", date_col="date", value_col="cases", lags=(1,))
    assert "cases_lag1" in out.columns
    assert pd.isna(out.loc[0,"cases_lag1"])
    assert out.loc[1,"cases_lag1"] == 10
