from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pandas as pd

from rhgp.config import FAIL_GRADES
from rhgp.features.aggregate_inspections import aggregate_to_inspections
from rhgp.features.features import add_history_features


def build_supervised_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    t = cast(pd.DataFrame, aggregate_to_inspections(raw))
    t = add_history_features(t)

    t = t.sort_values(["camis", "inspection_date_t"]).copy()
    t["inspection_date_t1"] = t.groupby("camis")["inspection_date_t"].shift(-1)
    t["grade_t1"] = t.groupby("camis")["grade_t"].shift(-1)

    # Label is based on next grade only; rows without a next inspection are dropped.
    t = t[t["grade_t1"].notna()].copy()
    grade_t1 = cast(pd.Series, t["grade_t1"])
    t["y_t1"] = grade_t1.isin(list(FAIL_GRADES)).astype(int)

    # Hard leakage rule is enforced downstream by an explicit feature column allowlist.
    # This dataset retains identifiers and label/split columns for analysis.
    return cast(pd.DataFrame, t.reset_index(drop=True))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build supervised dataset: features at t, label from t+1."
    )
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_path", type=Path, required=True)
    args = p.parse_args(argv)

    raw = pd.read_parquet(args.in_path)
    ds = build_supervised_dataset(raw)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(args.out_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
