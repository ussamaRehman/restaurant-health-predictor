.PHONY: setup data preprocess train train-rf eval eval-rf compare app lint type test check ml

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

train-rf:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m rhgp.models.train_rf --data data/processed/dataset.parquet --out models/rf.joblib

eval:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m rhgp.models.eval --data data/processed/dataset.parquet --model models/logreg.joblib --out-dir reports

eval-rf:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m rhgp.models.eval --data data/processed/dataset.parquet --model models/rf.joblib --out-dir reports/rf

compare:
	@mkdir -p "$(TMPDIR)"
	$(PY) -m rhgp.models.eval --data data/processed/dataset.parquet --model models/logreg.joblib --out-dir reports --thresholds "0.5,0.7" >/dev/null
	$(PY) -m rhgp.models.eval --data data/processed/dataset.parquet --model models/rf.joblib --out-dir reports/rf --thresholds "0.5,0.7" >/dev/null
	@$(PY) -c "import json; m1=json.load(open('reports/metrics.json')); m2=json.load(open('reports/rf/metrics.json')); print('model\\tthr\\tprec\\trec\\tf1');\nfor k,m in [('logreg',m1),('rf',m2)]:\n  rows=m.get(f'{k}_threshold_tuning',{}).get('rows',[]);\n  for t in [0.5,0.7]:\n    r=next((x for x in rows if abs(x['threshold']-t)<1e-9),None);\n    if r: print(f\"{k}\\t{t}\\t{r['precision_fail']:.3f}\\t{r['recall_fail']:.3f}\\t{r['f1_fail']:.3f}\")"
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

check: lint type test

ml: setup data preprocess train eval check
