"""TRAIN-only pilot partitions must respect physical support and inner purging."""

import numpy as np
import pytest

from tools.i6e_compact_run import partition_origins, scaled_inputs


def test_partition_uses_calendar_hours_not_rows_and_purges():
    ts = np.arange(600) * 3600
    target = np.ones(600)
    fit, inner = partition_origins(ts, target, 0, 600 * 3600, inner_hours=168)
    assert ts[fit].min() == 144 * 3600
    assert np.all(ts[fit] + 72 * 3600 <= 432 * 3600)
    assert np.all(ts[inner] >= 432 * 3600)
    assert np.all(ts[inner] + 72 * 3600 < 600 * 3600)
    assert not np.intersect1d(fit, inner).size


def test_scaler_ignores_inner_future_values_and_keeps_source_masks():
    values = np.array([[1., np.nan], [3., 4.], [100., 100.]])
    before, scale = scaled_inputs(values, [0, 1])
    values[-1] *= 100
    after, _ = scaled_inputs(values, [0, 1])
    np.testing.assert_array_equal(before[:2], after[:2])
    np.testing.assert_array_equal(before[0, 1::2], [1, 0])
    assert np.isfinite(before).all()


def test_invalid_targets_do_not_enter_either_partition():
    ts = np.arange(600) * 3600
    y = np.ones(600)
    y[200] = np.nan
    fit, inner = partition_origins(ts, y, 0, 600 * 3600, inner_hours=168)
    assert 200 not in fit and 200 not in inner
