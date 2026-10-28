"""Exact-clock support must precede any resolution campaign fit."""

import numpy as np
import pytest

from tools.i6e_support import support_report


def test_dense_clock_has_identical_context_and_future_support():
    report = support_report(np.arange(400) * 3600)
    assert report["common_origins"] == 184
    assert all(a["span_hours"] == 144 for a in report["arms"])
    assert [a["samples"] for a in report["arms"]] == [3, 5, 9]
    assert report["status"] == "SUPPORTED"


def test_gap_is_not_interpolated_or_counted_as_a_row_horizon():
    ts = np.delete(np.arange(400) * 3600, 150)
    report = support_report(ts)
    assert report["common_origins"] < 184
    assert report["input_missing_origins"] > 0


def test_future_presence_is_required_and_timestamps_are_unique():
    assert support_report(np.arange(73) * 3600)["status"] == "EMPTY_SUPPORT"
    with pytest.raises(ValueError, match="strictly increasing"):
        support_report([0, 0, 3600])
