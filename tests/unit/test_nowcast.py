import pandas as pd
from cdc_platform.modeling.nowcasting.nowcast import nowcast_latest_cases

def test_nowcast_output():
    df = pd.DataFrame({"date": ["2026-01-01","2026-01-02"], "region": ["A","A"], "cases": [10,12]})
    out = nowcast_latest_cases(df)
    assert "nowcast_latest" in out.columns
