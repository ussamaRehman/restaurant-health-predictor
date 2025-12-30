import pytest

from rhgp.models.eval import parse_thresholds


def test_parse_thresholds_parses_comma_list() -> None:
    assert parse_thresholds("0.2, 0.5,0.7") == [0.2, 0.5, 0.7]


def test_parse_thresholds_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        parse_thresholds("1.2")

