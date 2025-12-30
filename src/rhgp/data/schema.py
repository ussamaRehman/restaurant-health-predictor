from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Columns:
    camis: str = "camis"
    inspection_date: str = "inspection_date"
    inspection_type: str = "inspection_type"
    grade: str = "grade"
    score: str = "score"
    violation_code: str = "violation_code"
    violation_description: str = "violation_description"
    critical_flag: str = "critical_flag"


COLS = Columns()


def selected_columns() -> list[str]:
    return [
        COLS.camis,
        COLS.inspection_date,
        COLS.inspection_type,
        COLS.grade,
        COLS.score,
        COLS.violation_code,
        COLS.violation_description,
        COLS.critical_flag,
    ]


def normalize_grade(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().upper()
    if v in {"A", "B", "C"}:
        return v
    return None

