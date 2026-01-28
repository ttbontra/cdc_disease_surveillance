# src/cdc_platform/data/registry/lineage.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...common.dates import utc_now_iso
from ...common.io import ensure_dir


@dataclass(frozen=True)
class LineageRecord:
    run_id: str
    created_at_utc: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    params: dict[str, Any]


def append_lineage(log_path: Path, record: LineageRecord) -> None:
    ensure_dir(log_path.parent)
    # JSONL for easy appends / parsing
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.__dict__) + "\n")


def new_record(run_id: str, inputs: dict[str, Any], outputs: dict[str, Any], params: dict[str, Any]) -> LineageRecord:
    return LineageRecord(
        run_id=run_id,
        created_at_utc=utc_now_iso(),
        inputs=inputs,
        outputs=outputs,
        params=params,
    )
