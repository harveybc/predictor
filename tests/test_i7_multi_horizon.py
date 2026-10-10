"""Temporal, target and reduction contracts for the business horizon sweep."""

import numpy as np
import pytest

from tools import i7_multi_horizon as sweep


def test_required_horizons_and_diagnostic_only_one_hour():
    assert sweep.HORIZONS == (1, 2, 3, 4, 5, 6, 24, 48, 72, 96, 120)
    assert sweep.target_names() == tuple(
        f"Y_s_{h}h" if h <= 6 else f"Y_l_{h}h" for h in sweep.HORIZONS
    )
    assert sweep.OPERATING_HORIZONS == sweep.HORIZONS[1:]


def test_targets_require_exact_future_time_and_zero_staleness():
    ts = np.arange(150, dtype=np.int64) * 3600
    targets = {name: np.ones(150) for name in sweep.target_names()}
    for h in sweep.HORIZONS[6:]:
        targets[f"Y_l_{h}h_staleness_h"] = np.zeros(150)
    targets["Y_l_24h_staleness_h"][0] = 1
    y = sweep.exact_targets(ts, targets)
    assert np.isnan(y[0, 6])
    assert np.isfinite(y[0, 0])
    assert np.isnan(y[-1]).all()
    with pytest.raises(ValueError, match="missing"):
        sweep.exact_targets(ts, {})


def test_scale_is_per_horizon_and_fitted_on_training_only():
    y = np.array([[0., 0.], [1., 100.], [2., 200.], [1000., 1e9]])
    center, scale = sweep.target_scaler(y[:3])
    np.testing.assert_allclose(center, [1., 100.])
    assert scale[1] == pytest.approx(100 * scale[0])
    with pytest.raises(ValueError, match="degenerate"):
        sweep.target_scaler(np.zeros((3, 2)))


def test_same_row_naive_and_pooled_not_mean_of_weekly_means():
    first = sweep.error_record(np.array([1.]), np.array([0.]), ["a"])
    second = sweep.error_record(np.array([2., 2., 2.]), np.zeros(3), ["b", "c", "d"])
    total = sweep.pool_records([first, second])
    assert total["mae"] == total["naive_mae"] == 1.75
    assert total["skill_mae"] == 0
    assert total["n"] == 4
    with pytest.raises(ValueError, match="finite"):
        sweep.error_record(np.array([np.nan]), np.array([0.]), ["a"])


def test_forecast_shape_does_not_allow_flattened_horizon_mixup():
    good = np.zeros((7, 11, 1))
    assert sweep.check_prediction(good, 7).shape == (7, 11)
    with pytest.raises(ValueError, match="shape"):
        sweep.check_prediction(np.zeros((7, 11)), 7)


def test_horizon_support_is_physical_hours_not_rows():
    ts = np.delete(np.arange(150, dtype=np.int64) * 3600, 1)
    targets = {name: np.ones(len(ts)) for name in sweep.target_names()}
    for h in sweep.HORIZONS[6:]:
        targets[f"Y_l_{h}h_staleness_h"] = np.zeros(len(ts))
    y = sweep.exact_targets(ts, targets)
    assert np.isnan(y[0, 0])
    assert np.isfinite(y[0, 1])


def test_masked_labels_do_not_require_all_horizons_at_once():
    y = np.array([[1., np.nan], [np.nan, 100.], [2., 200.]])
    center, scale = sweep.target_scaler(y)
    targets, weights = sweep.masked_training_targets(y, center, scale)
    assert np.isfinite(targets).all()
    np.testing.assert_array_equal(weights == 0, ~np.isfinite(y))
    np.testing.assert_allclose(weights.sum(axis=0), [3., 3.])
    with pytest.raises(ValueError, match="empty"):
        sweep.masked_training_targets(np.array([[np.nan, 1.]]), np.zeros(2), np.ones(2))
