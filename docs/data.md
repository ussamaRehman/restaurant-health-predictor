# Data notes

## Source
- NYC OpenData (Socrata) dataset id: `43nn-pn8j`

## API usage principles
- Fetch only needed columns.
- Filter by date (default: last 3 years).
- Paginate using `$limit`/`$offset`.
- Cache a raw snapshot in `data/raw/` for reproducibility.

## Canonical keys (expected)
- `camis` (restaurant identifier)
- `inspection_date`
- `inspection_type`
- `grade` and `score`
- `violation_code`, `violation_description`, `critical_flag` (when present)

Column names and availability will be validated against dataset metadata at runtime.

