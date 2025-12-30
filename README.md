# Restaurant Health Grade Predictor (NYC)

Predict whether a restaurant will receive **Grade A** vs **Grade B/C (or worse)** on its **next** NYC health inspection.

## Dataset
- NYC OpenData (Socrata) inspections dataset id: `43nn-pn8j`
- We fetch the last 3 years via the API and cache a raw snapshot under `data/raw/` for reproducibility.

## Setup
This repo assumes Python via `pyenv` (see `.python-version`).

```bash
make setup
```

## Pipeline

One command:
```bash
make ml
```

Step-by-step:
```bash
make data
make preprocess
make train
make eval
```

## Results
Latest run (default split: last 20% by `inspection_date_t1`, threshold=0.5):
- Logistic regression: precision_fail=0.249, recall_fail=0.951, f1_fail=0.394 (n_test=5671)
- Persistence (last observed grade up to `t`): precision_fail=0.241, recall_fail=0.291, f1_fail=0.263
- RandomForest: precision_fail=0.337, recall_fail=0.393, f1_fail=0.363
- Run `make eval` to generate `reports/confusion_matrix.csv` locally

Threshold tradeoff (logistic regression, FAIL class):
- At `threshold=0.5`: precision_fail≈0.249, recall_fail≈0.947, f1_fail≈0.395
- At `threshold=0.7`: precision_fail≈0.387, recall_fail≈0.520, f1_fail≈0.444

Raising the threshold improves precision (fewer false alarms) but reduces recall (more missed B/C+).

Threshold tradeoff (RandomForest, FAIL class):
- At `threshold=0.5`: precision_fail≈0.337, recall_fail≈0.393, f1_fail≈0.363
- At `threshold=0.7`: precision_fail≈0.356, recall_fail≈0.113, f1_fail≈0.171

For this run, RF’s best F1 among `{0.5,0.7}` is at `0.5`.

Run `make eval` to generate metrics under `reports/` locally (focus is precision/recall/F1 on the “fail” class = `B/C+`).

## Leakage constraints (summary)
- Rows are inspection events `t`.
- Features may use information available up to and including `t` (including grade/score/violations at `t`).
- Target is grade at `t+1`.
- Never include any `t+1` fields in features; split is time-based on `t+1` date.

See `docs/leakage.md`.

## Limitations
- Observational data; policies and inspection practices can change over time.
- Restaurants with sparse history are harder to model.

## Ethics note
Predictions should not be used to penalize businesses without due process; outputs are probabilistic and may encode historical biases in enforcement and reporting.
