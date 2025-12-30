.PHONY: setup data preprocess train eval app lint type test

PY := python
TMPDIR := $(CURDIR)/.tmp
export TMPDIR

setup:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

data:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m rhgp.data.fetch --since-years 3 --out data/raw/inspections_43nn-pn8j_last3y.parquet

preprocess:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m rhgp.features.build_examples --in data/raw/inspections_43nn-pn8j_last3y.parquet --out data/processed/dataset.parquet

train:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m rhgp.models.train --data data/processed/dataset.parquet --out models/logreg.joblib

eval:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m rhgp.models.eval --data data/processed/dataset.parquet --model models/logreg.joblib --out-dir reports

app:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m pip install -e ".[app]"
	streamlit run app/streamlit_app.py

lint:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m ruff check .

type:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m pyright

test:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m pytest -q
