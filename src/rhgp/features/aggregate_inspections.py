from __future__ import annotations

import pandas as pd

from rhgp.data.schema import COLS, normalize_grade


def aggregate_to_inspections(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df[COLS.grade] = df[COLS.grade].map(normalize_grade)
    df[COLS.score] = pd.to_numeric(df.get(COLS.score), errors="coerce")

    # Normalize and ensure date typed.
    df[COLS.inspection_date] = pd.to_datetime(df[COLS.inspection_date], errors="coerce").dt.date

    key_cols = [COLS.camis, COLS.inspection_date]
    # Violation aggregates at inspection t (allowed features).
    df["_has_violation"] = df[COLS.violation_code].notna()
    df["_is_critical"] = df.get(COLS.critical_flag).astype(str).str.upper().eq("CRITICAL")

    agg = (
        df.groupby(key_cols, dropna=False)
        .agg(
            inspection_type=(COLS.inspection_type, "first"),
            grade_t=(COLS.grade, "first"),
            score_t=(COLS.score, "first"),
            n_violations_t=("_has_violation", "sum"),
            n_critical_violations_t=("_is_critical", "sum"),
        )
        .reset_index()
        .rename(columns={COLS.camis: "camis", COLS.inspection_date: "inspection_date_t"})
    )

    return agg

