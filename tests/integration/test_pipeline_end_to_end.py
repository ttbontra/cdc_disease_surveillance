from cdc_platform.serving.jobs.nightly_pipeline import run_nightly_pipeline

def test_pipeline_end_to_end():
    master = run_nightly_pipeline("2026-01-01", "2026-01-20")
    assert not master.empty
    assert {"date","region","cases"}.issubset(master.columns)
