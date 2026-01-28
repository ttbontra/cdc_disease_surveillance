#!/usr/bin/env bash
set -e
# 1) create master.csv
python -m cdc_platform.cli.main ingest --start 2025-09-01 --end 2026-01-01
# 2) train risk model artifact
python -m cdc_platform.cli.main train-risk
# 3) start API
python -m cdc_platform.cli.main serve
