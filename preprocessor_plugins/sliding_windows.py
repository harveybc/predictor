import numpy as np
import pandas as pd


def create_sliding_windows(data, config, date_times=None):
    """
    Creates sliding windows for feature processing (handles both DataFrame and dict of DataFrames).
    
    CRITICAL: Each window contains data[t-window_size+1:t+1] where t is the current tick.
    The baseline is data[t] (last element of window).
    
    Args:
        data: Dictionary of DataFrames (e.g., {'x_train_df': df1, 'y_train_df': df2, ...}) 
              or single DataFrame for single dataset processing
        config: Configuration dictionary containing window_size and other parameters
        date_times: Optional datetime index for the data
        
    Returns:
        Dictionary with sliding windows for each dataset or tuple for single dataset
    """
    window_size = config.get("window_size", 48)
    print(f"Creating sliding windows (Size={window_size})...", end="")
    
    # Handle single DataFrame input (backward compatibility)
    if isinstance(data, pd.DataFrame):
        return _create_sliding_windows_single_df(data, window_size, date_times)
    
    # Handle dictionary of DataFrames (main use case)
    if not isinstance(data, dict):
        raise ValueError("Data must be either a DataFrame or a dictionary of DataFrames")
    
    results = {}
    
    # Process each dataset
    for data_key, df in data.items():
        if df is None or len(df) == 0:
            print(f" WARN: Empty data for {data_key}, skipping...")
            continue
            
        dataset_type = _get_dataset_type(data_key)  # 'train', 'val', or 'test'
        
        n = len(df)
        if n < window_size:
            print(f" WARN: Insufficient data ({n}) for {data_key} with window size {window_size}. Need at least {window_size}.")
            continue
        
        # Create sliding windows for this dataset
        windows, window_dates = _create_sliding_windows_single_df(df, window_size, df.index if hasattr(df, 'index') else None)
        
        # Store results with proper naming convention
        results[f'X_{dataset_type}'] = windows
        results[f'x_dates_{dataset_type}'] = window_dates
        results['feature_names'] = list(df.columns)
    
    print(f" Done.")
    return results


def _get_dataset_type(data_key):
    """Extract dataset type from data key."""
    if 'train' in data_key:
        return 'train'
    elif 'val' in data_key:
        return 'val'
    elif 'test' in data_key:
        return 'test'
    else:
        return 'unknown'


def _create_sliding_windows_single_df(df, window_size, date_times=None):
    """Create sliding windows for a single DataFrame."""
    windows = []
    date_windows = []
    n = len(df)
    
    # Calculate usable range: can create windows from index window_size-1 to n-1
    min_baseline_idx = window_size - 1
    max_baseline_idx = n - 1
    
    if max_baseline_idx < min_baseline_idx:
        return np.array(windows, dtype=np.float32), np.array(date_windows, dtype=object)
    
    # Create windows: each window ends at baseline time t
    for baseline_idx in range(min_baseline_idx, max_baseline_idx + 1):
        window_start = baseline_idx - window_size + 1
        window_end = baseline_idx + 1
        
        # Extract window as DataFrame slice then convert to numpy
        window_df = df.iloc[window_start:window_end]
        window_array = window_df.values  # Convert to numpy array
        windows.append(window_array)
        
        # Date corresponds to baseline time (last element of window)
        if date_times is not None and baseline_idx < len(date_times):
            if hasattr(date_times, 'iloc'):
                date_windows.append(date_times.iloc[baseline_idx])
            else:
                date_windows.append(date_times[baseline_idx])
        else:
            date_windows.append(None)
    
    # Convert to numpy array with shape (n_windows, window_size, n_features)
    windows_array = np.array(windows, dtype=np.float32)
    date_windows_arr = np.array(date_windows, dtype=object)
    
    return windows_array, date_windows_arr


def extract_baselines_from_sliding_windows(sliding_windows_dict, config):
    """
    Extract baselines (last element of each window for target column) from sliding windows.
    
    Args:
        sliding_windows_dict: Dictionary containing sliding windows data from create_sliding_windows
        config: Configuration dictionary containing target_column
        
    Returns:
        Dictionary with baselines for each dataset split
    """
    target_column = config.get("target_column", "CLOSE")
    baselines = {}
    
    print(f"Extracting baselines for target column '{target_column}'...", end="")
    
    # Get feature names to find target column index
    feature_names = sliding_windows_dict.get('feature_names', [])
    if target_column not in feature_names:
        raise ValueError(f"Target column '{target_column}' not found in feature names: {feature_names}")
    
    target_col_idx = feature_names.index(target_column)
    
    # Extract baselines from each dataset's sliding windows
    for dataset_type in ['train', 'val', 'test']:
        windows_key = f'X_{dataset_type}'
        if windows_key not in sliding_windows_dict:
            baselines[f'baseline_{dataset_type}'] = np.array([])
            continue
            
        windows = sliding_windows_dict[windows_key]
        if len(windows) == 0:
            baselines[f'baseline_{dataset_type}'] = np.array([])
            continue
        
        # Extract last element of each window for target column
        # windows shape: (n_windows, window_size, n_features)
        # baseline = windows[:, -1, target_col_idx] (last timestep, target column)
        baseline_values = windows[:, -1, target_col_idx]
        baselines[f'baseline_{dataset_type}'] = baseline_values.astype(np.float32)
    
    print(" Done.")
    return baselines