"""Exact readback must reject swapped values, units and duplicate metrics."""

import pytest
from tools.i6e_publish_olap import verify_readback


def test_exact_readback_and_corruption():
    reports = [dict(report_sha256="abc", metrics=[dict(metric="MAE", value=.1,
        unit="log_return", horizon=72, split="validation")])]
    row = dict(report_sha256="abc", metric="MAE", value=.1, unit="log_return", horizon=72, split="validation")
    verify_readback(reports, [row])
    for change in ({"value":99}, {"unit":"USD"}, {"report_sha256":"wrong"}, {"horizon":1}):
        with pytest.raises(ValueError):
            verify_readback(reports, [{**row, **change}])
    with pytest.raises(ValueError):
        verify_readback(reports, [row, row])
