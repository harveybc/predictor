import pandas as pd
from app.reconstruction import unwindow_data

def load_csv(file_path, headers=False):
    """
    Loads a CSV file and, if headers is True, attempts to parse a column named 'DATE_TIME' as datetime and use it as the index.
    Falls back to the original logic if 'DATE_TIME' is not present or cannot be parsed. 
    Converts all remaining columns to numeric, filling NaNs with zeros.
    """
    try:
        # Load raw CSV data
        if headers:
            data = pd.read_csv(file_path, sep=',', dtype=str)
        else:
            data = pd.read_csv(file_path, header=None, sep=',', dtype=str)

        # If the CSV has a 'DATE_TIME' column, parse it as datetime and set it as index
        # Otherwise, fallback to the original logic of checking if the first column is datetime
        if headers and 'DATE_TIME' in data.columns:
            # Parse DATE_TIME column
            data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'], errors='coerce')
            # Set DATE_TIME as index
            data.set_index('DATE_TIME', inplace=True)
            # If there's still a column literally named 'DATE_TIME', drop it
            # (some CSVs might have uppercase/lowercase variants)
            data.drop(columns=[c for c in data.columns if c.lower() == 'date_time'], inplace=True, errors='ignore')
        else:
            # Original fallback logic for date detection
            if headers and pd.api.types.is_datetime64_any_dtype(data.iloc[:, 0]):
                data.columns = ['date'] + [f'col_{i-1}' for i in range(1, len(data.columns))]
                data.set_index('date', inplace=True)
            else:
                data.columns = [f'col_{i}' for i in range(len(data.columns))]

        # Convert all columns to numeric, fill NaNs with zeros
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

        # Check for remaining NaNs
        if data.isnull().values.any():
            print("Warning: NaN values found in the data after processing. Please review the loaded dataset.")
            
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        raise
    return data

def write_csv(file_path, data, include_date=True, headers=True, window_size=None):
    try:
        if include_date and 'date' in data.columns:
            data.to_csv(file_path, index=True, header=headers)
        else:
            data.to_csv(file_path, index=False, header=headers)
    except Exception as e:
        print(f"An error occurred while writing the CSV: {e}")
        raise