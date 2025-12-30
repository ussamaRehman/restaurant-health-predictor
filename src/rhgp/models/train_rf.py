from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from rhgp.models.train import build_preprocessor, feature_columns, time_split


def build_rf_pipeline(
    *,
    n_estimators: int = 400,
    max_depth: int | None = None,
    random_state: int = 42,
) -> Pipeline:
    pre = build_preprocessor()
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train RandomForest baseline.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--test-start", type=str, default=None, help="YYYY-MM-DD cutoff for t+1 date")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=None)
    args = p.parse_args(argv)

    df = pd.read_parquet(args.data)
    train_df, _ = time_split(df, test_start=args.test_start)

    X_train = train_df[feature_columns()]
    y_train = train_df["y_t1"].astype(int)

    pipe = build_rf_pipeline(n_estimators=args.n_estimators, max_depth=args.max_depth)
    pipe.fit(X_train, y_train)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

