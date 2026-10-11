"""Calendar holes are explicit, never shifted to a neighbouring quote."""

import numpy as np
import pytest

from tools.i6e_clock_windows import clock_windows


def test_exact_endpoints_and_market_mask():
    ts = np.array([0, 18, 54, 72]) * 3600
    values = np.array([[1.], [2.], [4.], [5.]])
    x, observed = clock_windows(ts, values, [3], 18, 5)
    np.testing.assert_array_equal(x[0, :, 0], [1, 2, 0, 4, 5])
    np.testing.assert_array_equal(observed, [[True, True, False, True, True]])


def test_future_perturbation_and_boundary_support():
    ts = np.arange(10) * 3600
    values = np.arange(10, dtype=float)[:, None]
    before = clock_windows(ts, values, [6], 2, 4)
    values[7:] = 999
    after = clock_windows(ts, values, [6], 2, 4)
    np.testing.assert_array_equal(before[0], after[0])
    with pytest.raises(ValueError, match="history boundary"):
        clock_windows(ts, values, [5], 2, 4)


def test_invalid_origin_and_nonfinite_values_rejected():
    with pytest.raises(ValueError):
        clock_windows([0, 3600], [[1], [np.nan]], [1], 1, 2)
    with pytest.raises(ValueError):
        clock_windows([0, 3600], [[1], [2]], [-1], 1, 2)
