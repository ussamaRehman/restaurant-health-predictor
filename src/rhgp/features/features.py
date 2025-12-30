from __future__ import annotations

import pandas as pd


def add_history_features(inspections_t: pd.DataFrame) -> pd.DataFrame:
    df = inspections_t.sort_values(["camis", "inspection_date_t"]).copy()

    # Previous inspection summaries (strictly < t).
    df["prev_grade"] = df.groupby("camis")["grade_t"].shift(1)
    df["prev_score"] = df.groupby("camis")["score_t"].shift(1)
    df["prev_n_violations"] = df.groupby("camis")["n_violations_t"].shift(1)
    df["prev_n_critical_violations"] = df.groupby("camis")["n_critical_violations_t"].shift(1)

    # Rolling means over prior inspections (exclude current via shift(1)).
    for col in ["score_t", "n_violations_t", "n_critical_violations_t"]:
        df[f"{col}_mean_prev3"] = (
            df.groupby("camis")[col]
            .apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
    return df

