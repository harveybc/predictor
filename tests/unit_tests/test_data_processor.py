import numpy as np
import pandas as pd

from app.data_processor import process_data, create_sliding_windows_single


def _write_close_csv(path, n, start=1.0):
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    df = pd.DataFrame({
        'DATE_TIME': dates,
        'CLOSE': np.linspace(start, start + n - 1, n),
    })
    df.to_csv(path, index=False)


def test_create_sliding_windows_single_shapes_and_targets():
    data = np.arange(10, dtype=np.float32)
    windows, targets, date_windows = create_sliding_windows_single(
        data, window_size=4, time_horizon=2
    )
    # n - window_size - time_horizon + 1 = 10 - 4 - 2 + 1 = 5 samples
    assert windows.shape == (5, 4)
    assert targets.shape == (5,)
    np.testing.assert_array_equal(windows[0], np.array([0, 1, 2, 3], dtype=np.float32))
    # target = data[i + window_size + time_horizon - 1] = data[5]
    assert targets[0] == 5.0
    assert date_windows == []


def test_process_data_single_step(tmp_path):
    n_rows = 40
    for name in ('train', 'val', 'test'):
        _write_close_csv(tmp_path / f'{name}.csv', n_rows)

    window_size = 8
    time_horizon = 2
    config = {
        'x_train_file': str(tmp_path / 'train.csv'),
        'x_validation_file': str(tmp_path / 'val.csv'),
        'x_test_file': str(tmp_path / 'test.csv'),
        'headers': True,
        'window_size': window_size,
        'time_horizon': time_horizon,
    }

    datasets = process_data(config)

    n_samples = n_rows - window_size - time_horizon + 1
    assert datasets['x_train'].shape == (n_samples, window_size, 1)
    assert datasets['x_val'].shape == (n_samples, window_size, 1)
    assert datasets['x_test'].shape == (n_samples, window_size, 1)
    assert datasets['y_train_array'].shape == (n_samples, 1)
    assert datasets['y_test_array'].shape == (n_samples, 1)
    assert datasets['test_close_prices'].shape == (n_samples,)
    # Linear series: the target is always window end + horizon.
    first_window_end = datasets['x_train'][0, -1, 0]
    assert datasets['y_train_array'][0, 0] == first_window_end + time_horizon


def test_process_data_use_returns(tmp_path):
    n_rows = 30
    for name in ('train', 'val', 'test'):
        _write_close_csv(tmp_path / f'{name}.csv', n_rows)

    config = {
        'x_train_file': str(tmp_path / 'train.csv'),
        'x_validation_file': str(tmp_path / 'val.csv'),
        'x_test_file': str(tmp_path / 'test.csv'),
        'headers': True,
        'window_size': 6,
        'time_horizon': 3,
        'use_returns': True,
    }

    datasets = process_data(config)

    # With a strictly linear close series, every return target equals the horizon.
    np.testing.assert_allclose(datasets['y_train_array'], 3.0)
    assert 'baseline_train' in datasets
    assert 'baseline_test' in datasets
