import pandas as pd

from rhgp.models.baselines import persistence_proba


def test_persistence_predicts_some_fails_when_grade_t_has_fails() -> None:
    df = pd.DataFrame(
        {
            "grade_t": ["A", "B", "A", "C", None],
            "prev_grade": ["A", "A", "A", "A", "B"],
        }
    )
    p = persistence_proba(df)
    assert float(p.sum()) >= 2.0
