## Repository workflow

### Goal
Predict whether a restaurant will receive **Grade A** vs **Grade B/C (or worse)** on its **next** NYC health inspection using the NYC OpenData inspections dataset (`43nn-pn8j`).

### Prediction problem (strict definition)
- **Row**: an inspection event `t` (one row per `CAMIS` + `inspection_date_t`) after aggregating violation rows into inspection-level aggregates.
- **Features**: derived from information available **up to and including inspection `t`** (allowed: grade/score/violation aggregates at `t`, plus historical summaries up to `t`).
- **Target**: the grade on the **next** inspection `t+1` for the same `CAMIS`, binarized as `A` vs `B/C+`.
- **Split**: time-based split based on `inspection_date_t1` (train on older `t+1`, test on newer `t+1`).
- **Hard leakage rule**: never use any fields from inspection `t+1` in features (no joins/shift mistakes).

### Commands
- Setup: `make setup`
- Fetch data: `make data`
- Build dataset: `make preprocess`
- Train: `make train`
- Evaluate: `make eval`
- Quality: `make test`, `make lint`, `make type`
- App (optional): `make app`

### Definition of done
- Raw snapshot cached in `data/raw/` and reproducible via `make data`.
- Processed supervised dataset in `data/processed/` built with leakage guards and tests.
- Baselines + logistic regression trained and evaluated; metrics + confusion matrix saved under `reports/`.
- `ruff`, `pyright`, and `pytest` are green.
- `README.md` documents problem, pipeline commands, results, limitations, and an ethics note.

