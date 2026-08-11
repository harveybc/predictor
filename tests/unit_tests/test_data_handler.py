import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from io import StringIO
from app.data_handler import load_csv, write_csv

# Mock data for tests
mock_data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1]
})


# Test loading CSV file with headers
def test_load_csv_with_headers():
    with patch("builtins.open", mock_open(read_data=mock_data.to_csv(index=False))):
        data = load_csv('test.csv', headers=True)
    pd.testing.assert_frame_equal(data, mock_data)


# Test loading CSV file without headers: the contract names the columns
# col_0..col_N and coerces values to numeric. This is the default public API
# (headers=False); pandas yields integer column labels for header=None, and
# the finding-216 fix normalizes them via str(c) before detection so this
# path no longer raises AttributeError.
def test_load_csv_without_headers():
    csv_text = mock_data.to_csv(index=False, header=False)
    with patch("builtins.open", mock_open(read_data=csv_text)):
        data = load_csv('test.csv', headers=False)
    expected = mock_data.copy()
    expected.columns = ['col_0', 'col_1']
    pd.testing.assert_frame_equal(data, expected)


# Test the DATE_TIME path: a (case-insensitively, whitespace-tolerantly)
# detected DATE_TIME column becomes a datetime index and the remaining
# columns are converted to numeric.
def test_load_csv_with_date_time_index():
    csv_text = (
        " Date_Time ,A,B\n"
        "2026-01-01 00:00:00,1,5\n"
        "2026-01-01 04:00:00,2,4\n"
        "2026-01-01 08:00:00,3,3\n"
    )
    with patch("builtins.open", mock_open(read_data=csv_text)):
        data = load_csv('test.csv', headers=True)
    assert isinstance(data.index, pd.DatetimeIndex)
    assert list(data.index) == [
        pd.Timestamp('2026-01-01 00:00:00'),
        pd.Timestamp('2026-01-01 04:00:00'),
        pd.Timestamp('2026-01-01 08:00:00'),
    ]
    assert list(data.columns) == ['A', 'B']
    assert data['A'].tolist() == [1, 2, 3]
    assert data['B'].tolist() == [5, 4, 3]


# Test max_rows limiting on both the headers and the no-headers paths.
def test_load_csv_max_rows_with_headers():
    with patch("builtins.open", mock_open(read_data=mock_data.to_csv(index=False))):
        data = load_csv('test.csv', headers=True, max_rows=2)
    pd.testing.assert_frame_equal(data, mock_data.iloc[:2])


def test_load_csv_max_rows_without_headers():
    csv_text = mock_data.to_csv(index=False, header=False)
    with patch("builtins.open", mock_open(read_data=csv_text)):
        data = load_csv('test.csv', headers=False, max_rows=3)
    expected = mock_data.iloc[:3].copy()
    expected.columns = ['col_0', 'col_1']
    pd.testing.assert_frame_equal(data, expected)


# Test writing CSV file (current contract takes a DataFrame)
def test_write_csv():
    with patch("builtins.open", mock_open()) as mocked_file:
        write_csv('test_write.csv', mock_data, include_date=False, headers=True)
        handle = mocked_file()
        handle.write.assert_called()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        written_df = pd.read_csv(StringIO(written_content))
        pd.testing.assert_frame_equal(written_df, mock_data)


if __name__ == "__main__":
    pytest.main()
