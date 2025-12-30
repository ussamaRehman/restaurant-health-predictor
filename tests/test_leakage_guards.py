import pandas as pd

from rhgp.features.build_examples import build_supervised_dataset
from rhgp.models.train import feature_columns


def test_no_t1_columns_in_feature_block() -> None:
    raw = pd.DataFrame(
        {
            "camis": [1, 1],
            "inspection_date": ["2024-01-01", "2024-02-01"],
            "inspection_type": ["Cycle", "Cycle"],
            "grade": ["A", "B"],
            "score": [5, 20],
            "violation_code": ["10F", "06C"],
            "violation_description": ["desc1", "desc2"],
            "critical_flag": ["Not Critical", "Critical"],
        }
    )
    ds = build_supervised_dataset(raw)
    assert "grade_t1" in ds.columns
    assert "inspection_date_t1" in ds.columns
    assert bool(ds["y_t1"].isin([0, 1]).all())


def test_model_features_exclude_identifiers_and_future() -> None:
    cols = feature_columns()
    assert "camis" not in cols
    assert "inspection_date_t" not in cols
    assert all(not c.endswith("_t1") for c in cols)
