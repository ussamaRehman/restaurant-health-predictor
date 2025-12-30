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
- Confusion matrix in `reports/confusion_matrix.csv`

Metrics are written to `reports/` (focus is precision/recall/F1 on the “fail” class = `B/C+`).

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
