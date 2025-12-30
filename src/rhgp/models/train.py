from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS_NUM = [
    "score_t",
    "n_violations_t",
    "n_critical_violations_t",
    "prev_score",
    "prev_n_violations",
    "prev_n_critical_violations",
    "score_t_mean_prev3",
    "n_violations_t_mean_prev3",
    "n_critical_violations_t_mean_prev3",
]
FEATURE_COLUMNS_CAT = ["inspection_type", "grade_t", "prev_grade"]


def feature_columns() -> list[str]:
    # Explicit allowlist to avoid accidental inclusion of identifiers (e.g., CAMIS)
    # or any t+1 fields.
    return FEATURE_COLUMNS_NUM + FEATURE_COLUMNS_CAT


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, FEATURE_COLUMNS_NUM),
            ("cat", categorical, FEATURE_COLUMNS_CAT),
        ]
    )


def time_split(
    df: pd.DataFrame, test_start: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["inspection_date_t1"] = pd.to_datetime(d["inspection_date_t1"], errors="coerce")
    cutoff = pd.to_datetime(test_start) if test_start else d["inspection_date_t1"].quantile(0.8)
    train = cast(pd.DataFrame, d.loc[d["inspection_date_t1"] < cutoff].copy())
    test = cast(pd.DataFrame, d.loc[d["inspection_date_t1"] >= cutoff].copy())
    return (train, test)


def build_pipeline() -> Pipeline:
    pre = build_preprocessor()
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train logistic regression baseline.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--test-start", type=str, default=None, help="YYYY-MM-DD cutoff for t+1 date")
    args = p.parse_args(argv)

    df = pd.read_parquet(args.data)
    train_df, _ = time_split(df, test_start=args.test_start)

    X_train = train_df[feature_columns()]
    y_train = train_df["y_t1"].astype(int)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
