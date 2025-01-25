import pandas as pd
from typing import Optional
from app.reconstruction import unwindow_data


def load_csv(file_path: str, headers: bool = False, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Loads a CSV file with optional row limiting and processes it into a cleaned DataFrame.

    This function reads a CSV file from the specified path, optionally limiting the number of rows
    based on the `max_rows` parameter. If `headers` is `True`, the function attempts to parse a
    column named 'DATE_TIME' as datetime and set it as the DataFrame index. If 'DATE_TIME' is
    absent or cannot be parsed, it falls back to checking if the first column is of datetime type.
    All remaining columns are converted to numeric types, with NaN values filled with zeros.

    Args:
        file_path (str): The path to the CSV file to be loaded.
        headers (bool, optional): Indicates whether the CSV file includes headers.
            Defaults to `False`.
        max_rows (int, optional): The maximum number of rows to read from the CSV file.
            If `None`, all rows are read. Defaults to `None`.

    Returns:
        pd.DataFrame: A processed DataFrame with appropriate indexing and numeric conversions.

    Raises:
        Exception: Propagates any exception that occurs during the CSV loading or processing.

    Example:
        >>> df = load_csv("data/train.csv", headers=True, max_rows=1000)
    """
    try:
        # Load raw CSV data with optional row limit
        if headers:
            data = pd.read_csv(file_path, sep=',', dtype=str, nrows=max_rows)
        else:
            data = pd.read_csv(file_path, header=None, sep=',', dtype=str, nrows=max_rows)

        # If the CSV has a 'DATE_TIME' column, parse it as datetime and set it as index
        # Otherwise, fallback to the original logic of checking if the first column is datetime
        if headers and 'DATE_TIME' in data.columns:
            # Parse DATE_TIME column
            data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'], errors='coerce')
            # Set DATE_TIME as index
            data.set_index('DATE_TIME', inplace=True)
            # Drop any additional 'DATE_TIME' columns (case-insensitive)
            date_time_columns = [c for c in data.columns if c.lower() == 'date_time']
            if date_time_columns:
                data.drop(columns=date_time_columns, inplace=True, errors='ignore')
        else:
            # Original fallback logic for date detection
            first_col = data.iloc[:, 0]
            if headers and pd.api.types.is_datetime64_any_dtype(first_col):
                data.columns = ['date'] + [f'col_{i}' for i in range(1, len(data.columns))]
                data.set_index('date', inplace=True)
            else:
                data.columns = [f'col_{i}' for i in range(len(data.columns))]

        # Convert all columns to numeric types, filling NaNs with zeros
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

        # Check for remaining NaNs and issue a warning if any are found
        if data.isnull().values.any():
            print("Warning: NaN values found in the data after processing. "
                  "Please review the loaded dataset.")

    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        raise

    return data


def write_csv(file_path: str, data: pd.DataFrame, include_date: bool = True,
              headers: bool = True, window_size: Optional[int] = None) -> None:
    """
    Writes a DataFrame to a CSV file with optional date inclusion and headers.

    This function exports the provided DataFrame to a CSV file at the specified path.
    It allows for conditional inclusion of the date column and headers. An optional
    `window_size` parameter is present for future extensions but is not utilized in
    the current implementation.

    Args:
        file_path (str): The destination path for the CSV file.
        data (pd.DataFrame): The DataFrame to be written to the CSV.
        include_date (bool, optional): Determines whether to include the date column
            in the CSV. If `True` and the DataFrame contains a 'date' column, it is included
            as the index. Defaults to `True`.
        headers (bool, optional): Indicates whether to write the column headers to the CSV.
            Defaults to `True`.
        window_size (int, optional): Placeholder for windowing functionality.
            Not used in the current implementation. Defaults to `None`.

    Raises:
        Exception: Propagates any exception that occurs during the CSV writing process.

    Example:
        >>> write_csv("data/output.csv", df, include_date=True, headers=True)
    """
    try:
        if include_date and 'date' in data.columns:
            data.to_csv(file_path, index=True, header=headers)
        else:
            data.to_csv(file_path, index=False, header=headers)
    except Exception as e:
        print(f"An error occurred while writing the CSV: {e}")
        raise
