from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from rhgp.data.schema import selected_columns


DATASET_ID = "43nn-pn8j"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"


@dataclass(frozen=True)
class FetchConfig:
    since_date: date
    limit: int = 50_000


def _since_years_to_date(since_years: int) -> date:
    today = date.today()
    return date(today.year - since_years, today.month, today.day)


def build_params(cfg: FetchConfig, offset: int) -> dict[str, Any]:
    cols = selected_columns()
    # Socrata supports SoQL with $select/$where/$order, and pagination via $limit/$offset.
    return {
        "$select": ", ".join(cols),
        "$where": f"inspection_date >= '{cfg.since_date.isoformat()}'",
        "$order": "inspection_date ASC",
        "$limit": cfg.limit,
        "$offset": offset,
    }


def fetch_all(cfg: FetchConfig, session: requests.Session | None = None) -> pd.DataFrame:
    sess = session or requests.Session()
    token = os.getenv("SOCRATA_APP_TOKEN")
    if token:
        sess.headers.update({"X-App-Token": token})

    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = build_params(cfg, offset=offset)
        resp = sess.get(BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        batch = resp.json()
        if not isinstance(batch, list):
            raise TypeError(f"Expected list JSON; got {type(batch)}")
        if not batch:
            break
        rows.extend(batch)
        offset += cfg.limit
        if len(batch) < cfg.limit:
            break

    df = pd.DataFrame.from_records(rows)
    if "inspection_date" in df.columns:
        df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce").dt.date
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch NYC inspections data from Socrata.")
    p.add_argument("--since-years", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    since_date = _since_years_to_date(args.since_years)
    cfg = FetchConfig(since_date=since_date)

    df = fetch_all(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    meta = {
        "dataset_id": DATASET_ID,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "since_date": since_date.isoformat(),
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    meta_path = args.out.with_suffix(".meta.json")
    pd.Series(meta).to_json(meta_path, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

