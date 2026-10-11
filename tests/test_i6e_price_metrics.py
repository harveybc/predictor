"""Price-unit metrics require exact endpoints and the retained clock convention."""

import numpy as np
import pandas as pd
import pytest

from tools.i6e_price_metrics import hourly_close, price_errors


def test_new_york_bar_start_to_utc_hour_end_and_no_empty_bars():
    raw = pd.DataFrame(dict(timestamp=pd.to_datetime([
        "2024-01-02 10:00Z", "2024-01-02 10:55Z", "2024-01-02 12:00Z"]), close=[1., 2., 3.]))
    close = hourly_close(raw)
    assert list(close.index) == list(pd.to_datetime(["2024-01-02 16:00Z", "2024-01-02 18:00Z"]))
    np.testing.assert_array_equal(close.values, [2., 3.])


def test_price_errors_use_current_price_as_naive_and_reconcile_target():
    result = price_errors(np.array([1., 2.]), np.array([1.1, 2.2]),
                          np.log([1.1, 1.1]), np.zeros(2))
    assert result["mae"] == pytest.approx(.15)
    assert result["naive_mae"] == pytest.approx(.15)
    assert result["skill_mae"] == pytest.approx(0.)
    with pytest.raises(ValueError, match="target"):
        price_errors(np.array([1.]), np.array([2.]), np.array([0.]), np.array([0.]))


def test_nonpositive_prices_rejected():
    with pytest.raises(ValueError):
        price_errors(np.array([0.]), np.array([1.]), np.array([0.]), np.array([0.]))
