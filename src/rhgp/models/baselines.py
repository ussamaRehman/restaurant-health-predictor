from __future__ import annotations

from typing import cast

import pandas as pd


def always_a_proba(df: pd.DataFrame) -> pd.Series:
    # Predict P(fail)=0 for everyone.
    return pd.Series(0.0, index=df.index, name="p_fail")


def persistence_proba(df: pd.DataFrame) -> pd.Series:
    # "Last-grade persistence": predict t+1 is fail if the most recently observed
    # grade up to t is fail.
    # Uses grade at t when present; otherwise falls back to prev_grade (history up to t).
    grade_t = cast(pd.Series, df["grade_t"])
    prev_grade = cast(pd.Series, df["prev_grade"]) if "prev_grade" in df.columns else grade_t
    last_grade = grade_t.fillna(prev_grade)
    is_fail_now = last_grade.isin(["B", "C"])
    out = is_fail_now.astype(float)
    out.name = "p_fail"
    return out


def grade_conditional_fail_rate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    """
    A more meaningful "persistence" baseline: estimate P(fail at t+1 | grade at t)
    from the training split, then apply to the test split.
    """
    train_grade = cast(pd.Series, train_df["grade_t"])
    train_y = cast(pd.Series, train_df["y_t1"]).astype(int)

    rates = cast(pd.Series, train_y.groupby(train_grade).mean())
    overall = float(train_y.mean()) if len(train_y) else 0.0

    test_grade = cast(pd.Series, test_df["grade_t"])
    out = test_grade.map(rates).fillna(overall).astype(float)
    out.name = "p_fail"
    return out
