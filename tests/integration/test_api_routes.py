from fastapi.testclient import TestClient
import pandas as pd

from cdc_platform.serving.api.main import app, wire_master_cache

def test_api_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200

def test_api_alerts():
    df = pd.DataFrame({
        "date": ["2026-01-01","2026-01-02","2026-01-03"],
        "region": ["A","A","A"],
        "cases": [10,15,25],
        "hosp": [1,1,2],
        "ww_viral_load": [1.0,1.1,1.5],
        "mobility_index": [1.0,1.0,1.0],
    })
    wire_master_cache(df)
    client = TestClient(app)
    r = client.get("/alerts")
    assert r.status_code == 200
