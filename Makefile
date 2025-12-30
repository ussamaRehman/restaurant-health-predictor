.PHONY: setup data preprocess train eval app lint type test

PY := python

setup:
\t$(PY) -m pip install -U pip
\t$(PY) -m pip install -e ".[dev]"

data:
\t$(PY) -m rhgp.data.fetch --since-years 3 --out data/raw/inspections_43nn-pn8j_last3y.parquet

preprocess:
\t$(PY) -m rhgp.features.build_examples --in data/raw/inspections_43nn-pn8j_last3y.parquet --out data/processed/dataset.parquet

train:
\t$(PY) -m rhgp.models.train --data data/processed/dataset.parquet --out models/logreg.joblib

eval:
\t$(PY) -m rhgp.models.eval --data data/processed/dataset.parquet --model models/logreg.joblib --out-dir reports

app:
\t$(PY) -m pip install -e ".[app]"
\tstreamlit run app/streamlit_app.py

lint:
\t$(PY) -m ruff check .

type:
\t$(PY) -m pyright

test:
\t$(PY) -m pytest -q

