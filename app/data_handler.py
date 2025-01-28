import pandas as pd
from typing import Optional
from app.reconstruction import unwindow_data


import pandas as pd
from typing import Optional
import sys

def load_csv(file_path: str, headers: bool = False, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Loads a CSV file with optional row limiting and processes it into a cleaned DataFrame.

    This function reads a CSV file from the specified path, optionally limiting the number of rows
    based on `max_rows`. If `headers` is `True`, it attempts to parse a column named 'DATE_TIME'
    (in any case) as datetime and set it as the DataFrame index. If not found or can't be parsed,
    it falls back to checking whether the first column is datetime. All remaining columns are
    converted to numeric types, with NaN values filled with zeros.
    """
    try:
        # 1) Load raw CSV data
        if headers:
            data = pd.read_csv(file_path, sep=',', dtype=str, nrows=max_rows)
        else:
            data = pd.read_csv(file_path, header=None, sep=',', dtype=str, nrows=max_rows)

        # 2) Detect any column named 'DATE_TIME' in a case-insensitive way
        date_time_cols = [c for c in data.columns if c.strip().lower() == 'date_time']

        if headers and date_time_cols:
            # 2a) Use the first match as the main date column
            main_dt_col = date_time_cols[0]
            data[main_dt_col] = pd.to_datetime(data[main_dt_col], errors='coerce')
            data.set_index(main_dt_col, inplace=True)

            # If there are duplicates of that column, drop them
            extra_dt_cols = date_time_cols[1:]
            for c in extra_dt_cols:
                if c in data.columns:
                    data.drop(columns=[c], inplace=True, errors='ignore')

        else:
            # 2b) Fallback logic: check the first column
            first_col = data.iloc[:, 0]
            # If user declared `headers=True` and that first column is a recognized datetime
            if headers and pd.api.types.is_datetime64_any_dtype(pd.to_datetime(first_col, errors='coerce')):
                # rename columns for clarity
                data.columns = ['date'] + [f'col_{i}' for i in range(1, len(data.columns))]
                data['date'] = pd.to_datetime(data['date'], errors='coerce')
                data.set_index('date', inplace=True)
            else:
                # Otherwise just rename columns generically if headers=False
                # or if no date column found
                data.columns = [f'col_{i}' for i in range(len(data.columns))]

        # 3) Convert columns to numeric, fillNa=0
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

        # 4) Debug info about final shape and index type
        print(f"[DEBUG] Loaded CSV '{file_path}' -> shape={data.shape}, index={data.index.dtype}, headers={headers}")

        # 5) Check for leftover NaNs
        if data.isnull().values.any():
            print(f"Warning: NaN values found after converting CSV: {file_path}")

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
