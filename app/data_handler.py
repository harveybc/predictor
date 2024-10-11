import pandas as pd
from app.reconstruction import unwindow_data

def load_csv(file_path, headers=False):
    try:
        if headers:
            data = pd.read_csv(file_path, sep=',', dtype=str)
        else:
            data = pd.read_csv(file_path, header=None, sep=',', dtype=str)

        # Handle headers or create default column names
        if headers and pd.api.types.is_datetime64_any_dtype(data.iloc[:, 0]):
            data.columns = ['date'] + [f'col_{i-1}' for i in range(1, len(data.columns))]
            data.set_index('date', inplace=True)
        else:
            data.columns = [f'col_{i}' for i in range(len(data.columns))]

        # Convert all columns except 'date' to numeric, fill NaNs with zeros
        for col in data.columns:
            if col != 'date':
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

        # Check if there are any remaining NaN values and print diagnostics if any exist
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