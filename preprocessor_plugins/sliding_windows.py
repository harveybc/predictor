import numpy as np
import pandas as pd




def create_sliding_windows(data, config):
    """
    Creates sliding windows for feature processing (not limited by horizon).
    
    CRITICAL: Each window contains data[t-window_size+1:t+1] where t is the current tick.
    The baseline is data[t] (last element of window).
    
    Args:
        data: 1D array of values
        window_size: Size of each window  
        date_times: Optional datetime index for the data
        
    Returns:
        Tuple of (windows_array, date_windows_array)
    """

    window_size = config.get("window_size", 48)  # Default to 48 if not specified
    print(f"Creating sliding windows (Size={window_size})...", end="")
    windows = []
    date_windows = []
    n = len(data)
    
    # Calculate usable range: can create windows from index window_size-1 to n-1
    min_baseline_idx = window_size - 1
    max_baseline_idx = n - 1
    
    if max_baseline_idx < min_baseline_idx:
        print(f" WARN: Insufficient data ({n}) for window size {window_size}. Need at least {window_size}.")
        return np.array(windows, dtype=np.float32), np.array(date_windows, dtype=object)
    
    # Create windows: each window ends at baseline time t
    for baseline_idx in range(min_baseline_idx, max_baseline_idx + 1):
        window_start = baseline_idx - window_size + 1
        window_end = baseline_idx + 1
        window = data[window_start:window_end]
        windows.append(window)
        
        # Date corresponds to baseline time (last element of window)
        if date_times is not None and baseline_idx < len(date_times):
            date_windows.append(date_times[baseline_idx])
        else:
            date_windows.append(None)
    
    # Keep dates as pandas Timestamps for matplotlib compatibility
    date_windows_arr = np.array(date_windows, dtype=object)
    
    print(f" Done ({len(windows)} windows).")
    return np.array(windows, dtype=np.float32), date_windows_arr


def extract_baselines_from_sliding_windows(sliding_windows, config):
    """
    Extract baselines (last elements of each window for target column).
    """
    target_column = config.get("target_column", "CLOSE")
    baselines = []

    for window in sliding_windows:
        if target_column in window:
            baselines.append(window[target_column].iloc[-1])
        else:
            # Raise error and stop processing
            raise ValueError(f"Target column '{target_column}' not found in sliding window data")

    return np.array(baselines, dtype=np.float32)