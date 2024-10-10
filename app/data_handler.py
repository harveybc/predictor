import pandas as pd
from app.reconstruction import unwindow_data

def load_csv(file_path, headers=False):
    try:
        if headers:
            data = pd.read_csv(file_path, sep=',', parse_dates=[0], dayfirst=True)
        else:
            data = pd.read_csv(file_path, header=None, sep=',', dtype=str)
            
            # Convert the first column to datetime if possible
            try:
                data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0], dayfirst=True, errors='coerce')
            except Exception as e:
                print(f"Error while parsing date column: {e}")
            
            if pd.api.types.is_datetime64_any_dtype(data.iloc[:, 0]):
                data.columns = ['date'] + [f'col_{i-1}' for i in range(1, len(data.columns))]
                data.set_index('date', inplace=True)
            else:
                data.columns = [f'col_{i}' for i in range(len(data.columns))]

            # Ensure numeric columns are properly converted, fill NaNs with zeros
            for col in data.columns:
                if col != 'date':
                    data[col] = pd.to_numeric(data[col].str.replace(',', '').str.replace('E', 'e'), errors='coerce').fillna(0)
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