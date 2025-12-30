from datetime import date

from rhgp.data.fetch import FetchConfig, build_params


def test_build_params_includes_where_limit_offset() -> None:
    cfg = FetchConfig(since_date=date(2023, 1, 1), limit=123)
    p0 = build_params(cfg, offset=0)
    assert p0["$limit"] == 123
    assert p0["$offset"] == 0
    assert "inspection_date" in p0["$where"]

