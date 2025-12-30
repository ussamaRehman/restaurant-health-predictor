# Pipeline

## Stages
1. **Fetch** (`make data`)
   - Pull last 3 years from Socrata API.
   - Cache raw snapshot in `data/raw/`.
2. **Preprocess** (`make preprocess`)
   - Aggregate raw rows to one row per inspection event `t`.
   - Construct supervised examples by pairing each `t` with its next inspection `t+1` label.
   - Write dataset to `data/processed/dataset.parquet`.
3. **Train** (`make train`)
   - Train baselines and logistic regression (scikit-learn).
   - Save model artifact to `models/`.
4. **Eval** (`make eval`)
   - Produce precision/recall/F1 for fail class (`B/C+`) and confusion matrix.
   - Save metrics to `reports/`.

## Make targets
- `make setup` installs dependencies.
- `make lint` runs ruff.
- `make type` runs pyright.
- `make test` runs pytest.

