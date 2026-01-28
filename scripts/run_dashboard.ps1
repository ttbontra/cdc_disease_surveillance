$env:PYTHONPATH = "$PSScriptRoot\..\src"
cdc ingest --start 2025-09-01 --end 2026-01-01

# 2) train ML models (RF + GB + hosp regressor)
cdc train-ml

# 3) train Bayesian hierarchical comparator
cdc train-bayes --draws 800 --tune 800 --min-days 60
# 4) run dashboard
streamlit run "$PSScriptRoot\..\src\cdc_platform\dashboard\app.py"
