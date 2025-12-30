# Backlog (Nice-to-Haves)

## Reproducibility & Packaging
- Add a dependency lockfile (uv.lock or pip-tools) for deterministic installs.
- Improve pyproject metadata (license, authors, classifiers, URLs).
- Add CLI entrypoints (console_scripts) for fetch/preprocess/train/eval.

## Data & Validation
- Add raw data schema validation (required columns + dtypes).

## Modeling & Evaluation
- Save model metadata alongside artifacts (features list, cutoff date, dataset fingerprint).
- Add extra evaluation metrics (ROC-AUC/PR-AUC for FAIL + calibration plots).
- Add a compare summary command/report for baseline vs logreg vs RF.

## Developer Experience
- Add pre-commit hooks (ruff/pyright/pytest).
- Optional Streamlit demo app.
