# Leakage rules

## Intended supervised learning setup
- **Example row**: inspection event `t` for a given `CAMIS`.
- **Features**: anything known up to and including inspection `t`.
  - Allowed: grade at `t`, score at `t`, aggregated violations at `t`, and summaries of past inspections `< t`.
- **Target**: grade for the next inspection `t+1` (`A` vs `B/C+`).
- **Split**: time-based split using `inspection_date_t1` (train older, test newer).

## Hard leakage rule
Never use any fields from `t+1` in features. This includes:
- Direct columns from `t+1` (grade, score, violations, etc.).
- Any “future” joins keyed by `CAMIS` that accidentally attach `t+1` values.
- Any rolling features computed across time that include the next inspection.

## Guardrails in code
- Dataset builder constructs `t` inspection rows first, then attaches `t+1` label using a strict shift per `CAMIS`.
- Tests assert no `*_t1` columns exist among features and that feature timestamps do not exceed `inspection_date_t`.

