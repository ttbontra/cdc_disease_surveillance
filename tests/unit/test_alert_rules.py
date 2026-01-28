import pandas as pd
from cdc_platform.modeling.early_warning.alert_rules import generate_alerts

def test_generate_alerts_runs():
    df = pd.DataFrame({
        "date": ["2026-01-01","2026-01-02","2026-01-03","2026-01-04"],
        "region": ["A","A","A","A"],
        "cases": [10, 12, 18, 30],
        "hosp": [1,1,2,3],
        "ww_viral_load": [1.0,1.0,1.2,1.6],
        "mobility_index": [1.0,1.0,1.0,1.0],
    })
    alerts = generate_alerts(df)
    assert isinstance(alerts, list)
