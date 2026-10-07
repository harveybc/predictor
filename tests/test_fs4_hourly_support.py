import numpy as np
import pytest

from tools.fs4_hourly_support import hourly_windows
from tools import fs4_temporal_predictor as predictor


HOUR = 3600


def test_four_hour_bars_use_24_elapsed_hours_and_a_three_hour_cutoff():
    ts = np.arange(9, dtype=np.int64) * 4 * HOUR
    x = np.arange(9, dtype=np.float64).reshape(-1, 1)
    raw, kept = hourly_windows(x, ts, [7], mean=[0], sd=[1], window_hours=24)
    lagged, lag_kept = hourly_windows(x, ts, [7], mean=[0], sd=[1], window_hours=24, lag_hours=3)
    assert kept.tolist() == lag_kept.tolist() == [7]
    assert raw.shape == lagged.shape == (1, 24, 2)
    assert raw[0, :, 1].sum() == 6
    assert raw[0, -1, 0] == 7 and raw[0, -1, 1] == 1
    assert lagged[0, :, 1].sum() == 6
    assert lagged[0, -1, 1] == 0
    assert lagged[0, -2, 0] == 6 and lagged[0, -2, 1] == 1


def test_no_future_or_pre_fit_history_enters_a_window():
    ts = np.arange(40, dtype=np.int64) * HOUR
    x = np.arange(40, dtype=np.float64).reshape(-1, 1)
    first, kept = hourly_windows(x, ts, [25], mean=[0], sd=[1], min_timestamp=2 * HOUR)
    changed = x.copy()
    changed[26:] = 1e9
    second, _ = hourly_windows(changed, ts, [25], mean=[0], sd=[1], min_timestamp=2 * HOUR)
    assert kept.tolist() == [25]
    assert np.array_equal(first, second)
    _, dropped = hourly_windows(x, ts, [24], mean=[0], sd=[1], min_timestamp=2 * HOUR)
    assert dropped.size == 0


def test_nonfinite_values_are_masked_and_time_contract_is_strict():
    ts = np.arange(25, dtype=np.int64) * HOUR
    x = np.arange(25, dtype=np.float64).reshape(-1, 1)
    x[15] = np.nan
    windows, kept = hourly_windows(x, ts, [24], mean=[0], sd=[1])
    assert kept.tolist() == [24]
    assert windows[0, 14].tolist() == [0, 0]
    with pytest.raises(ValueError, match="TIMESTAMPS_NOT_STRICTLY_INCREASING"):
        hourly_windows(x, ts[::-1], [24], mean=[0], sd=[1])
    with pytest.raises(ValueError, match="HOURLY_GRID_MISALIGNED"):
        hourly_windows(x, ts + np.arange(25), [24], mean=[0], sd=[1])


def test_predictor_raw_input_uses_elapsed_hours_not_dataset_rows():
    ts = np.arange(9, dtype=np.int64) * 4 * HOUR
    x = np.arange(9, dtype=np.float64).reshape(-1, 1)
    scaler = predictor.Standardiser.fit(x[:6])
    windows, kept = predictor._inputs_for(predictor.PredictorSpec(), None, x, scaler, [7],
                                          timestamps=ts, min_timestamp=0)
    assert kept.tolist() == [7]
    assert windows.shape == (1, 24, 2)
    assert windows[0, :, 1].sum() == 6
    assert windows[0, -1, 1] == 1
    with pytest.raises(ValueError, match="HOURLY_GRID_MISALIGNED"):
        predictor._inputs_for(predictor.PredictorSpec(), None, x, scaler, [7],
                              timestamps=ts + np.arange(9), min_timestamp=0)
