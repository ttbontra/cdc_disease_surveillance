# src/cdc_platform/data/registry/dataset_manifest.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...common.dates import utc_now_iso
from ...common.io import ensure_dir


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    path: str
    sha256: str
    rows: int
    created_at_utc: str


def write_manifest(manifest_path: Path, entries: list[DatasetEntry]) -> None:
    ensure_dir(manifest_path.parent)
    payload = {
        "created_at_utc": utc_now_iso(),
        "entries": [e.__dict__ for e in entries],
    }
    manifest_path.write_text(json.dumps(payload, indent=2))


def build_manifest_for_files(manifest_path: Path, files: list[Path]) -> None:
    entries: list[DatasetEntry] = []
    for p in files:
        if not p.exists() or not p.is_file():
            continue
        # lightweight row count (safe fallback if not csv)
        rows = 0
        if p.suffix.lower() == ".csv":
            rows = sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore")) - 1
            rows = max(rows, 0)
        entries.append(
            DatasetEntry(
                name=p.name,
                path=str(p),
                sha256=_sha256_file(p),
                rows=rows,
                created_at_utc=utc_now_iso(),
            )
        )
    write_manifest(manifest_path, entries)
