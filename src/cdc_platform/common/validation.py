from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import pandas as pd

@dataclass
class ValidationIssue:
    level: str  # "error"|"warn"
    message: str

def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    missing = [c for c in cols if c not in df.columns]
    if missing:
        issues.append(ValidationIssue("error", f"Missing columns: {missing}"))
    return issues

def assert_no_errors(issues: list[ValidationIssue]) -> None:
    errs = [i for i in issues if i.level == "error"]
    if errs:
        msg = "\n".join([f"- {e.message}" for e in errs])
        raise ValueError(f"Validation failed:\n{msg}")
