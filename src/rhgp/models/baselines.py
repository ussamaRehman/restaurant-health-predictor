from __future__ import annotations

from typing import cast

import pandas as pd


def always_a_proba(df: pd.DataFrame) -> pd.Series:
    # Predict P(fail)=0 for everyone.
    return pd.Series(0.0, index=df.index, name="p_fail")


def persistence_proba(df: pd.DataFrame) -> pd.Series:
    # Predict next inspection is fail if current grade is not A.
    grade_t = cast(pd.Series, df["grade_t"])
    is_fail_now = grade_t.isin(["B", "C"])
    out = is_fail_now.astype(float)
    out.name = "p_fail"
    return out
