from __future__ import annotations

import pandas as pd


def always_a_proba(df: pd.DataFrame) -> pd.Series:
    # Predict P(fail)=0 for everyone.
    return pd.Series(0.0, index=df.index, name="p_fail")


def persistence_proba(df: pd.DataFrame) -> pd.Series:
    # Predict next inspection is fail if current grade is not A.
    is_fail_now = df["grade_t"].isin({"B", "C"})
    return is_fail_now.astype(float).rename("p_fail")

