import pandas as pd
from cdc_platform.data.cleaning.dedupe import dedupe
from cdc_platform.data.cleaning.impute_missing import impute_missing_daily

def test_dedupe():
    df = pd.DataFrame({"date": ["2026-01-01","2026-01-01"], "region": ["A","A"], "cases": [1,2]})
    out = dedupe(df, subset=["date","region"])
    assert len(out) == 1
    assert out.iloc[0]["cases"] == 2

def test_impute_missing_daily():
    df = pd.DataFrame({"date": ["2026-01-01","2026-01-03"], "region": ["A","A"], "cases": [10,30]})
    out = impute_missing_daily(df, value_cols=["cases"])
    assert len(out) == 3
    assert out.iloc[1]["cases"] == 20