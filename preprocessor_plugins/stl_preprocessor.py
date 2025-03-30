# Ensure these imports are present at the top of the file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
import json
from app.data_handler import load_csv # Assuming load_csv is here
import os
from sklearn.preprocessing import StandardScaler
# from app.data_handler import write_csv # Not used directly in process_data
from scipy.signal import hilbert
from scipy.stats import shapiro
# Include the user-provided verify_date_consistency function
# Try importing optional dependencies
try:
    import pywt # For Wavelets
except ImportError:
    print("WARN: pywt library not found. Wavelet features ('use_wavelets=True') will be unavailable.")
    pywt = None
try:
    from scipy.signal.windows import dpss # For MTM tapers
except ImportError:
    print("WARN: scipy.signal.windows not found. MTM features ('use_multi_tapper=True') may be unavailable.")
    dpss = None


# Include the user-provided verify_date_consistency function (as provided in the prompt)
def verify_date_consistency(date_lists, dataset_name):
    """
    Verifies that all date arrays in date_lists have the same first and last elements.
    Prints a warning if any array does not match.
    """
    if not date_lists:
        # print(f"No date lists provided for {dataset_name} consistency check.") # Less verbose
        return

    valid_date_lists = [d for d in date_lists if d is not None and len(d) > 0]
    if not valid_date_lists:
        # print(f"No valid date arrays with content found for {dataset_name}.") # Less verbose
        return

    first_len = len(valid_date_lists[0])
    if not all(len(d) == first_len for d in valid_date_lists):
         print(f"WARN: Length mismatch in {dataset_name} dates: {[len(d) for d in valid_date_lists]}")
         return

    # print(f"{dataset_name} date lengths are consistent: {first_len}") # Less verbose

    try:
        first = valid_date_lists[0][0]
        last = valid_date_lists[0][-1]
    except IndexError:
        print(f"WARN: Could not access first/last element in {dataset_name} dates.")
        return

    consistent = True
    for i, d in enumerate(valid_date_lists):
        try:
            if len(d) > 0:
                if d[0] != first or d[-1] != last:
                    print(f"Warning: Date array {i} in {dataset_name} range ({d[0]} to {d[-1]}) does not match first array ({first} to {last}).")
                    consistent = False
            else:
                print(f"Warning: Date array {i} in {dataset_name} is empty.")
                consistent = False
        except Exception as e:
            print(f"Warning: Could not compare dates in {dataset_name} for array {i}. Error: {e}.")
            consistent = False

    # if consistent: print(f"{dataset_name} date ranges appear consistent.") # Less verbose


class PreprocessorPlugin:
    # Default plugin parameters - Adjusted for hourly EURUSD context
    plugin_params = {
        # --- File Paths ---
        "x_train_file": "data/eurusd_h1_train_x.csv", # Example path
        "y_train_file": "data/eurusd_h1_train_y.csv", # Example path
        "x_validation_file": "data/eurusd_h1_val_x.csv", # Example path
        "y_validation_file": "data/eurusd_h1_val_y.csv", # Example path
        "x_test_file": "data/eurusd_h1_test_x.csv",   # Example path
        "y_test_file": "data/eurusd_h1_test_y.csv",   # Example path
        # --- Data Loading ---
        "headers": True,
        "max_steps_train": None, "max_steps_val": None, "max_steps_test": None,
        "target_column": "TARGET", # Column name in Y files
        # --- Windowing & Horizons ---
        "window_size": 72, # Example: 3 days of hourly data
        "predicted_horizons": [1, 6, 12, 24], # Predict 1H, 6H, 12H, 24H ahead
        # --- Feature Engineering Flags ---
        "use_stl": False,           # Default to NOT using STL
        "use_wavelets": True,       # Default to using Wavelets
        "use_multi_tapper": False,  # Default to NOT using MTM
        "use_returns_as_target": True, # Target = TargetPrice - BaselinePrice
        "normalize_features": True, # Apply StandardScaler to features
        # --- STL Parameters (if use_stl=True) ---
        "stl_period": 24,         # Daily period for hourly data
        "stl_window": None,       # Auto-calculate based on period if None
        "stl_trend": None,        # Auto-calculate based on period if None
        "stl_plot_file": "stl_decomposition.png",
        # --- Wavelet Parameters (if use_wavelets=True) ---
        "wavelet_name": 'db4',      # Common choice
        "wavelet_levels": 6,        # Enough levels to potentially capture weekly patterns (auto if None)
        "wavelet_mode": 'symmetric', # Padding mode for DWT/MODWT if needed by implementation
        "wavelet_plot_file": "wavelet_features.png",
        # --- Multitaper Parameters (if use_multi_tapper=True) ---
        "mtm_window_len": 168,     # Window length in samples (e.g., 1 week = 24*7)
        "mtm_step": 1,             # Step size for rolling window (1 hour)
        "mtm_time_bandwidth": 5.0, # NW product (adjust for resolution/variance trade-off)
        "mtm_num_tapers": None,    # Auto: 2*NW-1
        # Freq bands (cycles/hour) targeting weekly, daily, intraday
        "mtm_freq_bands": [(0, 0.01), (0.01, 0.06), (0.06, 0.2), (0.2, 0.5)],
        "tapper_plot_file": "mtm_features.png",
        "tapper_plot_points": 500, # Points for MTM plot
    }
    # Update debug vars list
    plugin_debug_vars = [
        "window_size", "predicted_horizons", "use_returns_as_target", "normalize_features",
        "use_stl", "stl_period", "stl_window", "stl_trend", "stl_plot_file",
        "use_wavelets", "wavelet_name", "wavelet_levels", "wavelet_mode", "wavelet_plot_file",
        "use_multi_tapper", "mtm_window_len", "mtm_step", "mtm_time_bandwidth", "mtm_num_tapers", "mtm_freq_bands", "tapper_plot_file", "tapper_plot_points"
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.scalers = {}

    def set_params(self, **kwargs):
        """Update plugin parameters with global configuration."""
        # Update params FIRST
        for key, value in kwargs.items():
            self.params[key] = value

        # THEN apply logic based on updated params
        config = self.params # Use local reference for clarity
        if config.get("stl_period") is not None and config.get("stl_period") > 1:
            if config.get("stl_window") is None:
                config["stl_window"] = 2 * config["stl_period"] + 1
            if config.get("stl_trend") is None:
                current_stl_window = config.get("stl_window")
                if current_stl_window is not None and current_stl_window > 3:
                     try:
                         trend_calc = int(1.5 * config["stl_period"] / (1 - 1.5 / current_stl_window)) + 1
                         config["stl_trend"] = max(3, trend_calc) # Ensure minimum length
                     except ZeroDivisionError:
                         config["stl_trend"] = config["stl_period"] + 1
                else:
                     config["stl_trend"] = config["stl_period"] + 1
            # Ensure stl_trend is odd
            if config.get("stl_trend") is not None and config["stl_trend"] % 2 == 0:
                config["stl_trend"] += 1


    def get_debug_info(self):
        """Return debug information for the plugin."""
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """Add plugin debug info to the provided dictionary."""
        debug_info.update(self.get_debug_info())

    # --- Helper Methods ---
    def _load_data(self, file_path, max_rows, headers):
        # === Uses YOUR app.data_handler.load_csv ===
        print(f"Loading data: {file_path} (Max rows: {max_rows})...", end="")
        try:
            df = load_csv(file_path, headers=headers, max_rows=max_rows)
            if df is None or df.empty:
                 raise ValueError(f"load_csv returned None or empty for {file_path}")
            print(f" Done. Shape: {df.shape}")

            # --- Index Handling ---
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"Attempting to convert index of {file_path} to DatetimeIndex...", end="")
                original_index_name = df.index.name # Store original name
                try:
                    # Try converting existing index first
                    df.index = pd.to_datetime(df.index)
                    print(" Success (from index).")
                except Exception:
                    try:
                        # If index conversion failed, try first column
                        df.index = pd.to_datetime(df.iloc[:, 0])
                        # Optionally remove the first column after using it as index
                        # df = df.iloc[:, 1:]
                        print(" Success (from first column).")
                    except Exception as e_col:
                        print(f" FAILED. Error: {e_col}. Dates unavailable.")
                        df.index = None # Mark dates as unavailable
                # Restore index name if it existed
                if original_index_name: df.index.name = original_index_name

            # --- Column Checks ---
            required_cols = ["CLOSE"]
            if 'y_' in os.path.basename(file_path).lower(): # Check target only in Y files
                required_cols.append(self.params.get("target_column", "TARGET"))

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in {file_path}: {missing_cols}. Available: {df.columns.tolist()}")

            return df

        except FileNotFoundError:
             print(f"\nCRITICAL ERROR: File not found at {file_path}.")
             raise
        except Exception as e:
             print(f"\nCRITICAL ERROR loading or processing {file_path}: {e}")
             import traceback
             traceback.print_exc()
             raise

    def _normalize_series(self, series, name, fit=False):
        """Normalizes a time series using StandardScaler."""
        if not self.params.get("normalize_features", True):
            return series.astype(np.float32)

        series = series.astype(np.float32)
        if np.any(np.isnan(series)) or np.any(np.isinf(series)):
            print(f"WARN: NaNs/Infs in '{name}' before normalization. Filling...", end="")
            series = pd.Series(series).fillna(method='ffill').fillna(method='bfill').values
            if np.any(np.isnan(series)) or np.any(np.isinf(series)):
                 print(f" Filling FAILED. Filling remaining with 0.", end="")
                 series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0) # Use with caution
            print(" Done.")

        data_reshaped = series.reshape(-1, 1)
        if fit:
            scaler = StandardScaler()
            if np.std(data_reshaped) < 1e-9:
                 print(f"WARN: Feature '{name}' is constant. Using dummy scaler.")
                 class DummyScaler:
                     def fit(self, X): pass
                     def transform(self, X): return X.astype(np.float32)
                     def inverse_transform(self, X): return X.astype(np.float32)
                 scaler = DummyScaler()
            else:
                 scaler.fit(data_reshaped)
            self.scalers[name] = scaler
        else:
            if name not in self.scalers:
                raise RuntimeError(f"Scaler for '{name}' not fitted.")
            scaler = self.scalers[name]

        normalized_data = scaler.transform(data_reshaped)
        return normalized_data.flatten()

    # --- Original _rolling_stl (kept exactly as provided by user) ---
    def _rolling_stl(self, series, stl_window, period):
        # Placeholder: Uses original rolling STL logic
        print("Performing rolling STL decomposition...")
        n = len(series)
        num_points = n - stl_window + 1
        if num_points <= 0:
            raise ValueError(f"stl_window ({stl_window}) is too large for the series length ({n}).")
        trend = np.zeros(num_points)
        seasonal = np.zeros(num_points)
        resid = np.zeros(num_points)
        # --- MODIFICATION: Get stl_trend from resolved self.params ---
        stl_trend_smoother = self.params.get("stl_trend") # Use resolved trend smoother length

        for i in tqdm(range(stl_window, n + 1), desc="STL Decomposition", unit="window", disable=None, leave=False): # Enable progress bar
            window = series[i - stl_window: i]
            # Ensure stl_trend_smoother is odd and suitable for STL window size
            current_stl_trend = stl_trend_smoother
            if current_stl_trend is not None:
                # --- Ensure int and positive ---
                if not isinstance(current_stl_trend, int) or current_stl_trend <=0:
                    current_stl_trend = None # Disable if invalid
                elif current_stl_trend >= len(window):
                     current_stl_trend = len(window) - 1 if len(window) > 1 else None # Adjust if too large
                if current_stl_trend is not None and current_stl_trend % 2 == 0:
                     current_stl_trend += 1 # Ensure odd

            try:
                 # --- Pass potentially adjusted current_stl_trend ---
                 stl = STL(window, period=period, trend=current_stl_trend, robust=True)
                 result = stl.fit()
                 trend[i - stl_window] = result.trend[-1]
                 seasonal[i - stl_window] = result.seasonal[-1]
                 resid[i - stl_window] = result.resid[-1]
            except Exception as e:
                 print(f"WARN: STL fit failed for window ending at index {i-1}. Error: {e}. Filling with NaN.")
                 trend[i - stl_window] = np.nan
                 seasonal[i - stl_window] = np.nan
                 resid[i - stl_window] = np.nan

        # Handle potential NaNs from failed fits if necessary (e.g., forward fill)
        trend = pd.Series(trend).fillna(method='ffill').fillna(method='bfill').values
        seasonal = pd.Series(seasonal).fillna(method='ffill').fillna(method='bfill').values
        resid = pd.Series(resid).fillna(method='ffill').fillna(method='bfill').values

        return trend, seasonal, resid
    # --- End of original _rolling_stl ---

    # --- Original _plot_decomposition (kept exactly as provided by user) ---
    def _plot_decomposition(self, series, trend, seasonal, resid, file_path):
        # Placeholder: Uses original plotting logic
        print(f"Plotting STL decomposition to {file_path}...")
        plot_points = 480 # Limit plot points for readability
        # --- Robust Slicing ---
        n_series = len(series)
        n_comp = min(len(trend), len(seasonal), len(resid))
        if n_comp == 0: print("WARN: Zero length components for STL plotting."); return

        points_to_plot = min(plot_points, n_series, n_comp)
        if points_to_plot <= 0: print("WARN: No points to plot for STL."); return

        series_plot = series[-points_to_plot:]
        trend_plot = trend[-points_to_plot:]
        seasonal_plot = seasonal[-points_to_plot:]
        resid_plot = resid[-points_to_plot:]
        # --- End Slicing ---
        try:
            plt.figure(figsize=(12, 9))
            plt.subplot(411); plt.plot(series_plot); plt.title("Log-Transformed Series (Recent)"); plt.grid(True, alpha=0.5)
            plt.subplot(412); plt.plot(trend_plot, color="orange"); plt.title("Trend"); plt.grid(True, alpha=0.5)
            plt.subplot(413); plt.plot(seasonal_plot, color="green"); plt.title("Seasonal"); plt.grid(True, alpha=0.5)
            plt.subplot(414); plt.plot(resid_plot, color="red"); plt.title("Residual"); plt.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.savefig(file_path, dpi=300)
            plt.close()
            print(f"STL decomposition plot saved to {file_path}")
        except Exception as e:
             print(f"WARN: Failed to plot STL decomposition: {e}")
             plt.close() # Ensure figure is closed even if error occurs
    # --- End of original _plot_decomposition ---

    def _compute_wavelet_features(self, series):
        """Computes Wavelet features using MODWT (pywt.swt)."""
        # (Implementation from previous correct step - kept the same)
        if pywt is None: print("ERROR: pywt library not installed."); return {}
        name = self.params['wavelet_name']; levels = self.params['wavelet_levels']; n_original = len(series)
        if levels is None:
            try:
                levels = pywt.swt_max_level(n_original); print(f"Auto wavelet levels: {levels}")
            except Exception as e: print(f"ERROR calculating max wavelet levels: {e}."); return {}
        if levels <= 0: print(f"ERROR: Wavelet levels ({levels}) not positive."); return {}
        print(f"Computing Wavelets (MODWT/SWT): {name}, Levels={levels}...", end="")
        try:
            coeffs = pywt.swt(series, wavelet=name, level=levels, trim_approx=True, norm=True)
            features = {'approx_L{}'.format(levels): coeffs[0][0]}
            for i in range(levels): features['detail_L{}'.format(levels - i)] = coeffs[i][1]
            # Sanity check lengths
            for k, v in features.items():
                 if len(v) != n_original: print(f"WARN: Wavelet '{k}' length ({len(v)}) != original ({n_original}).")
            print(f" Done ({len(features)} channels).")
            return features
        except Exception as e: print(f" FAILED. Error: {e}"); return {}

    def _plot_wavelets(self, original_series, wavelet_features, file_path):
        """Plots original series and computed Wavelet features."""
        # (Implementation from previous correct step - kept the same)
        print(f"Plotting Wavelet features to {file_path}...")
        plot_points = 480; num_features = len(wavelet_features)
        if num_features == 0: print("WARN: No wavelet features to plot."); return
        start_idx = max(0, len(original_series) - plot_points); original_plot = original_series[start_idx:]
        actual_plot_points = len(original_plot)
        if actual_plot_points <= 0: print("WARN: No original series points for Wavelet plot."); return
        num_plots = num_features + 1; plt.figure(figsize=(12, 2 * num_plots))
        plt.subplot(num_plots, 1, 1); plt.plot(original_plot); plt.title(f"Original Series (Recent {actual_plot_points} points)"); plt.grid(True, alpha=0.5)
        plot_index = 2
        for name, feature_series in wavelet_features.items():
             feat_start_idx = max(0, len(feature_series) - actual_plot_points); feature_plot = feature_series[feat_start_idx:]
             if len(feature_plot) != actual_plot_points: print(f"WARN: Skip plot Wavelet '{name}', length mismatch."); continue
             plt.subplot(num_plots, 1, plot_index); plt.plot(feature_plot, label=name); plt.title(f"Wavelet Feature: {name}"); plt.grid(True, alpha=0.5); plot_index += 1
        if plot_index > 2:
             try:
                 plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.suptitle("Wavelet Decomposition Features", fontsize=14)
                 plt.savefig(file_path, dpi=300); plt.close(); print(f"Wavelet plot saved.")
             except Exception as e: print(f"WARN: Failed to save Wavelet plot: {e}"); plt.close()
        else: print("WARN: No wavelet features plotted."); plt.close()

    def _compute_mtm_features(self, series):
        """Computes Rolling Multitaper Method spectral power in bands."""
        # (Implementation from previous correct step - kept the same)
        if dpss is None: print("ERROR: scipy.signal.windows unavailable for MTM."); return {}
        window_len = self.params['mtm_window_len']; step = self.params['mtm_step']; nw = self.params['mtm_time_bandwidth']
        num_tapers = self.params['mtm_num_tapers']; freq_bands = self.params['mtm_freq_bands']; n_original = len(series)
        if num_tapers is None: num_tapers = max(1, int(2 * nw - 1))
        if n_original < window_len: print(f"ERROR: Series length ({n_original}) < MTM window ({window_len})."); return {}
        print(f"Computing MTM: Win={window_len}, Step={step}, NW={nw}, K={num_tapers}...", end="")
        try: tapers = dpss(window_len, nw, num_tapers)
        except ValueError as e: print(f" FAILED generating DPSS tapers: {e}."); return {}
        num_windows = (n_original - window_len) // step + 1
        mtm_features = {f"band_{i}": np.full(num_windows, np.nan) for i in range(len(freq_bands))}
        fft_freqs = np.fft.rfftfreq(window_len) # Freqs for sample rate = 1 (cycles/sample or cycles/hour here)
        freq_masks = [(fft_freqs >= f_low) & (fft_freqs < f_high) for f_low, f_high in freq_bands]
        for i in tqdm(range(num_windows), desc="MTM Calc", unit="window", disable=True, leave=False): # Disable progress bar nested here
            start = i * step; end = start + window_len; window_data = series[start:end]
            if np.any(np.isnan(window_data)): continue
            spectra = np.zeros((num_tapers, len(fft_freqs)))
            for k in range(num_tapers): spectra[k, :] = np.abs(np.fft.rfft(window_data * tapers[k]))**2
            avg_spectrum = np.mean(spectra, axis=0)
            for band_idx, mask in enumerate(freq_masks):
                 if np.any(mask): mtm_features[f"band_{band_idx}"][i] = np.mean(avg_spectrum[mask])
                 else: mtm_features[f"band_{band_idx}"][i] = 0.0
        # Fill leading/trailing NaNs from calculation start/end
        for name in mtm_features: mtm_features[name] = pd.Series(mtm_features[name]).fillna(method='ffill').fillna(method='bfill').values
        print(f" Done ({len(mtm_features)} channels).")
        return mtm_features

    def _plot_mtm(self, mtm_features, file_path, points_to_plot=500):
        """Plots computed Multitaper features (power bands)."""
        # (Implementation from previous correct step - kept the same)
        print(f"Plotting MTM features to {file_path}...")
        num_features = len(mtm_features)
        if num_features == 0: print("WARN: No MTM features to plot."); return
        min_len = 0; valid_keys = [k for k, v in mtm_features.items() if v is not None and len(v) > 0]
        if valid_keys: min_len = min(len(mtm_features[k]) for k in valid_keys)
        plot_points = min(points_to_plot, min_len)
        if plot_points <= 0: print("WARN: No MTM data points to plot."); return
        plt.figure(figsize=(12, 2 * num_features))
        plot_index = 1
        for name in valid_keys:
             feature_series = mtm_features[name]; start_idx = max(0, len(feature_series) - plot_points); feature_plot = feature_series[start_idx:]
             plt.subplot(num_features, 1, plot_index); plt.plot(feature_plot, label=name); plt.title(f"MTM Feature: {name} (Recent {plot_points} points)"); plt.grid(True, alpha=0.5); plot_index += 1
        if plot_index > 1:
             try:
                 plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.suptitle("Multitaper Spectral Power Features", fontsize=14)
                 plt.savefig(file_path, dpi=300); plt.close(); print(f"MTM plot saved.")
             except Exception as e: print(f"WARN: Failed to save MTM plot: {e}"); plt.close()
        else: print("WARN: No MTM features plotted."); plt.close()

    # --- Original create_sliding_windows (kept exactly as provided by user) ---
    def create_sliding_windows(self, data, window_size, time_horizon, date_times=None):
        # Placeholder: Uses original window creation logic
        # Requires time_horizon argument (passed as max_horizon) to limit window creation
        print(f"Creating sliding windows (size={window_size}, max_horizon={time_horizon})...", end="") # Added end=""
        windows = []
        targets = [] # This target list will be ignored later
        date_windows = []
        n = len(data)
        # Stop loop early enough to allow for the maximum horizon
        num_possible_windows = n - window_size - time_horizon + 1 # Correct calculation
        if num_possible_windows <= 0:
             print(f" WARN: Data length ({n}) too short for window_size ({window_size}) + max_horizon ({time_horizon}). No windows created.")
             return np.array(windows, dtype=np.float32), np.array(date_windows, dtype=object), np.array(targets, dtype=np.float32)

        for i in range(num_possible_windows): # Iterate correct number of times
            window_start_idx = i # Start index of window
            window_end_idx = i + window_size # End index for slicing window
            window = data[window_start_idx : window_end_idx]
            # target_idx = window_end_idx + time_horizon - 1 # Original target index logic
            # target = data[target_idx] # This target is ignored
            windows.append(window)
            targets.append(0) # Append dummy target 0
            if date_times is not None:
                # Date corresponds to the last point IN the input window
                date_index = window_end_idx - 1
                if date_index < len(date_times):
                    date_windows.append(date_times[date_index])
                else:
                    # This indicates a mismatch between data length and dates length passed
                    print(f"WARN: Date index {date_index} out of bounds for date_times (len {len(date_times)}) during windowing. Appending None.")
                    date_windows.append(None) # Append None or handle as error

        # Convert dates to numpy array, handling potential None values
        if date_times is not None:
             date_windows_arr = np.array(date_windows, dtype=object)
             # Attempt conversion to datetime64 only if all elements are valid Timestamps
             if all(isinstance(d, pd.Timestamp) for d in date_windows if d is not None): # Check non-None
                  try: date_windows_arr = np.array(date_windows, dtype='datetime64[ns]')
                  except (ValueError, TypeError): pass # Keep as object array if conversion fails
        else: date_windows_arr = np.array(date_windows, dtype=object) # Empty if date_times was None

        print(f" Done ({len(windows)} windows).") # Added confirmation
        return np.array(windows, dtype=np.float32), date_windows_arr, np.array(targets, dtype=np.float32)
    # --- End of original create_sliding_windows ---

    def align_features(self, feature_dict, base_length):
         """Aligns feature time series to a common length by truncating the beginning."""
         # (Implementation from previous correct step - kept the same)
         aligned_features = {}
         min_len = base_length
         feature_lengths = {'base': base_length}
         valid_keys = [k for k, v in feature_dict.items() if v is not None]
         if not valid_keys: return {}, 0 # Return empty if no valid features

         for name in valid_keys:
             feature_lengths[name] = len(feature_dict[name])
             min_len = min(min_len, feature_lengths[name])

         needs_alignment = any(l != min_len for l in feature_lengths.values() if l > 0)
         if needs_alignment:
             # print(f"Aligning features to length: {min_len}. Orig lengths: {feature_lengths}") # Verbose
             for name in valid_keys:
                 series = feature_dict[name]
                 current_len = len(series)
                 if current_len > min_len: aligned_features[name] = series[current_len - min_len:]
                 elif current_len == min_len: aligned_features[name] = series
                 else: print(f"WARN: Feature '{name}' length ({current_len}) < target align length ({min_len})."); aligned_features[name] = None # Mark inconsistent
         else:
             # print("Feature lengths consistent, no alignment needed.") # Verbose
             aligned_features = {k: feature_dict[k] for k in valid_keys} # Ensure only valid keys are kept

         # Verify final lengths
         final_lengths = {name: len(s) for name, s in aligned_features.items() if s is not None}
         unique_lengths = set(final_lengths.values())
         if len(unique_lengths) > 1: raise RuntimeError(f"Alignment FAILED! Inconsistent final lengths: {final_lengths}")

         return aligned_features, min_len


    # --- process_data Method ---
    def process_data(self, config):
        """
        Processes data for multi-horizon forecasting with optional features.
        Uses YOUR existing logic where specified, integrates new features.
        """
        # 0. Setup & Parameter Resolution
        print("\n" + "="*15 + " Starting Preprocessing " + "="*15)
        self.set_params(**config) # Resolve defaults (like stl_window/trend) based on final config
        config = self.params # Use the fully resolved params
        self.scalers = {} # Reset scalers

        # --- 1. Load Data ---
        print("\n--- 1. Loading Data ---")
        x_train_df = self._load_data(config["x_train_file"], config.get("max_steps_train"), config.get("headers"))
        x_val_df = self._load_data(config["x_validation_file"], config.get("max_steps_val"), config.get("headers"))
        x_test_df = self._load_data(config["x_test_file"], config.get("max_steps_test"), config.get("headers"))
        y_train_df = self._load_data(config["y_train_file"], config.get("max_steps_train"), config.get("headers"))
        y_val_df = self._load_data(config["y_validation_file"], config.get("max_steps_val"), config.get("headers"))
        y_test_df = self._load_data(config["y_test_file"], config.get("max_steps_test"), config.get("headers"))

        # --- 2. Initial Prep: Log Transform, Dates ---
        print("\n--- 2. Initial Data Prep ---")
        try:
            close_train = x_train_df["CLOSE"].astype(np.float32).values
            close_val = x_val_df["CLOSE"].astype(np.float32).values
            close_test = x_test_df["CLOSE"].astype(np.float32).values
        except KeyError: raise ValueError("'CLOSE' column not found in X data.")
        except Exception as e: raise ValueError(f"Error converting 'CLOSE' to numeric: {e}")

        # Log Transform (using log1p for robustness to zero/small values)
        log_train = np.log1p(np.maximum(0, close_train))
        log_val = np.log1p(np.maximum(0, close_val))
        log_test = np.log1p(np.maximum(0, close_test))
        print(f"Log transform applied. Train shape: {log_train.shape}")

        dates_train = x_train_df.index if isinstance(x_train_df.index, pd.DatetimeIndex) else None
        dates_val = x_val_df.index if isinstance(x_val_df.index, pd.DatetimeIndex) else None
        dates_test = x_test_df.index if isinstance(x_test_df.index, pd.DatetimeIndex) else None

        # --- 3. Feature Generation ---
        print("\n--- 3. Feature Generation ---")
        features_train = {}
        features_val = {}
        features_test = {}

        # 3.a. Log Returns (Always compute for alignment reference)
        log_ret_train = np.diff(log_train, prepend=log_train[0])
        log_ret_val = np.diff(log_val, prepend=log_val[0])
        log_ret_test = np.diff(log_test, prepend=log_test[0])
        features_train['log_return'] = self._normalize_series(log_ret_train, 'log_return', fit=True)
        features_val['log_return'] = self._normalize_series(log_ret_val, 'log_return', fit=False)
        features_test['log_return'] = self._normalize_series(log_ret_test, 'log_return', fit=False)
        print("Generated: Log Returns (Normalized)")

        # 3.b. Conditional STL Features (using YOUR _rolling_stl)
        if config.get('use_stl'):
             print("Attempting STL features...")
             stl_period=config['stl_period']; stl_window=config['stl_window'] # Use resolved params
             try:
                 trend_train, seasonal_train, resid_train = self._rolling_stl(log_train, stl_window, stl_period)
                 trend_val, seasonal_val, resid_val = self._rolling_stl(log_val, stl_window, stl_period)
                 trend_test, seasonal_test, resid_test = self._rolling_stl(log_test, stl_window, stl_period)
                 if len(trend_train) > 0: # Check if STL produced output
                      features_train['stl_trend'] = self._normalize_series(trend_train, 'stl_trend', fit=True)
                      features_train['stl_seasonal'] = self._normalize_series(seasonal_train, 'stl_seasonal', fit=True)
                      features_train['stl_resid'] = self._normalize_series(resid_train, 'stl_resid', fit=True)
                      features_val['stl_trend'] = self._normalize_series(trend_val, 'stl_trend', fit=False)
                      features_val['stl_seasonal'] = self._normalize_series(seasonal_val, 'stl_seasonal', fit=False)
                      features_val['stl_resid'] = self._normalize_series(resid_val, 'stl_resid', fit=False)
                      features_test['stl_trend'] = self._normalize_series(trend_test, 'stl_trend', fit=False)
                      features_test['stl_seasonal'] = self._normalize_series(seasonal_test, 'stl_seasonal', fit=False)
                      features_test['stl_resid'] = self._normalize_series(resid_test, 'stl_resid', fit=False)
                      print("Generated: STL Trend, Seasonal, Residual (Normalized)")
                      # Plotting (using YOUR _plot_decomposition)
                      stl_plot_file = config.get("stl_plot_file")
                      if stl_plot_file:
                           plot_start_idx = len(log_train) - len(trend_train)
                           self._plot_decomposition(log_train[plot_start_idx:], trend_train, seasonal_train, resid_train, stl_plot_file)
                 else: print("WARN: STL computation resulted in zero-length output.")
             except Exception as e: print(f"ERROR processing STL: {e}. Skipping.")
        else: print("Skipped: STL features.")

        # 3.c. Conditional Wavelet Features
        if config.get('use_wavelets'):
             print("Attempting Wavelet features...")
             try:
                 wav_features_train = self._compute_wavelet_features(log_train)
                 if wav_features_train:
                      wav_features_val = self._compute_wavelet_features(log_val)
                      wav_features_test = self._compute_wavelet_features(log_test)
                      for name in wav_features_train.keys():
                           features_train[f'wav_{name}'] = self._normalize_series(wav_features_train[name], f'wav_{name}', fit=True)
                           if name in wav_features_val: features_val[f'wav_{name}'] = self._normalize_series(wav_features_val[name], f'wav_{name}', fit=False)
                           if name in wav_features_test: features_test[f'wav_{name}'] = self._normalize_series(wav_features_test[name], f'wav_{name}', fit=False)
                      print(f"Generated: {len(wav_features_train)} Wavelet features (Normalized).")
                      wav_plot_file = config.get("wavelet_plot_file")
                      if wav_plot_file: self._plot_wavelets(log_train, wav_features_train, wav_plot_file)
             except Exception as e: print(f"ERROR processing Wavelets: {e}. Skipping.")
        else: print("Skipped: Wavelet features.")

        # 3.d. Conditional MTM Features
        if config.get('use_multi_tapper'):
             print("Attempting MTM features...")
             try:
                 mtm_features_train = self._compute_mtm_features(log_train)
                 if mtm_features_train:
                      mtm_features_val = self._compute_mtm_features(log_val)
                      mtm_features_test = self._compute_mtm_features(log_test)
                      for name in mtm_features_train.keys():
                           features_train[f'mtm_{name}'] = self._normalize_series(mtm_features_train[name], f'mtm_{name}', fit=True)
                           if name in mtm_features_val: features_val[f'mtm_{name}'] = self._normalize_series(mtm_features_val[name], f'mtm_{name}', fit=False)
                           if name in mtm_features_test: features_test[f'mtm_{name}'] = self._normalize_series(mtm_features_test[name], f'mtm_{name}', fit=False)
                      print(f"Generated: {len(mtm_features_train)} MTM features (Normalized).")
                      tapper_plot_file = config.get("tapper_plot_file")
                      if tapper_plot_file: self._plot_mtm(mtm_features_train, tapper_plot_file, config.get("tapper_plot_points", 500))
             except Exception as e: print(f"ERROR processing MTM: {e}. Skipping.")
        else: print("Skipped: MTM features.")


        # --- 4. Align Feature Lengths ---
        print("\n--- 4. Aligning Feature Lengths ---")
        # Use log_return as the base reference length
        base_len_train = len(features_train['log_return'])
        base_len_val = len(features_val['log_return'])
        base_len_test = len(features_test['log_return'])

        features_train, aligned_len_train = self.align_features(features_train, base_len_train)
        features_val, aligned_len_val = self.align_features(features_val, base_len_val)
        features_test, aligned_len_test = self.align_features(features_test, base_len_test)

        # Adjust dates to match the final aligned feature length
        dates_train_aligned = dates_train[-aligned_len_train:] if dates_train is not None and aligned_len_train > 0 else None
        dates_val_aligned = dates_val[-aligned_len_val:] if dates_val is not None and aligned_len_val > 0 else None
        dates_test_aligned = dates_test[-aligned_len_test:] if dates_test is not None and aligned_len_test > 0 else None
        print(f"Final aligned feature length: Train={aligned_len_train}, Val={aligned_len_val}, Test={aligned_len_test}")


        # --- 5. Windowing & Channel Stacking (using YOUR create_sliding_windows) ---
        print("\n--- 5. Windowing and Channel Stacking ---")
        window_size = config["window_size"]
        predicted_horizons = config['predicted_horizons']
        max_horizon = max(predicted_horizons)

        X_train_channels, X_val_channels, X_test_channels = [], [], []
        feature_names = []
        x_dates_train, x_dates_val, x_dates_test = None, None, None # To store dates from first windowed feature
        first_feature_dates_captured = False

        # Define order - log_return first, then others based on flags
        feature_order = ['log_return']
        if config.get('use_stl'): feature_order.extend(['stl_trend', 'stl_seasonal', 'stl_resid'])
        if config.get('use_wavelets'): feature_order.extend(sorted([k for k in features_train if k.startswith('wav_')]))
        if config.get('use_multi_tapper'): feature_order.extend(sorted([k for k in features_train if k.startswith('mtm_')]))

        # Keep only Log Returns if all optional flags are false
        if not config.get('use_stl') and not config.get('use_wavelets') and not config.get('use_multi_tapper'):
            print("Only including Log Returns feature.")
            feature_order = ['log_return']

        print(f"Attempting to window features: {feature_order}")
        for name in feature_order:
            if name in features_train and features_train[name] is not None:
                if name in features_val and features_val[name] is not None and \
                   name in features_test and features_test[name] is not None:
                    print(f"Windowing feature: {name}...", end="")
                    # Use YOUR create_sliding_windows
                    win_train, dates_win_train, _ = self.create_sliding_windows(features_train[name], window_size, max_horizon, dates_train_aligned)
                    win_val, dates_win_val, _   = self.create_sliding_windows(features_val[name], window_size, max_horizon, dates_val_aligned)
                    win_test, dates_win_test, _ = self.create_sliding_windows(features_test[name], window_size, max_horizon, dates_test_aligned)

                    # Check if windowing was successful (produced samples)
                    if win_train.shape[0] > 0 and win_val.shape[0] > 0 and win_test.shape[0] > 0:
                        X_train_channels.append(win_train)
                        X_val_channels.append(win_val)
                        X_test_channels.append(win_test)
                        feature_names.append(name)
                        print(" Appended.")
                        if not first_feature_dates_captured:
                             x_dates_train, x_dates_val, x_dates_test = dates_win_train, dates_win_val, dates_win_test
                             first_feature_dates_captured = True
                             print(f"Captured dates from '{name}'.")
                    else:
                        print(f" Skipping channel '{name}' (windowing produced 0 samples).")
                else:
                     print(f"Skipping channel '{name}' (missing in val/test after alignment).")
            # else: print(f"Feature '{name}' not available/computed, skipping.") # Too verbose

        # --- 6. Stack channels ---
        if not X_train_channels: raise RuntimeError("No feature channels available after windowing!")
        print("\n--- 6. Stacking Feature Channels ---")
        # Check consistency of sample counts before stacking
        num_samples_train = X_train_channels[0].shape[0]
        num_samples_val = X_val_channels[0].shape[0]
        num_samples_test = X_test_channels[0].shape[0]
        if not all(c.shape[0] == num_samples_train for c in X_train_channels): raise RuntimeError("Inconsistent samples in train channels.")
        if not all(c.shape[0] == num_samples_val for c in X_val_channels): raise RuntimeError("Inconsistent samples in val channels.")
        if not all(c.shape[0] == num_samples_test for c in X_test_channels): raise RuntimeError("Inconsistent samples in test channels.")

        # Stack along the last axis -> (num_samples, window_size, num_channels)
        X_train_combined = np.stack(X_train_channels, axis=-1).astype(np.float32)
        X_val_combined = np.stack(X_val_channels, axis=-1).astype(np.float32)
        X_test_combined = np.stack(X_test_channels, axis=-1).astype(np.float32)
        print(f"Final X shapes: Train={X_train_combined.shape}, Val={X_val_combined.shape}, Test={X_test_combined.shape}")
        print(f"Included features: {feature_names}")

        # --- 7. Baseline & Target Processing (YOUR ORIGINAL LOGIC PRESERVED) ---
        # NOTE: This section uses the original calculation based on `window_data_start_index`
        # derived from the STL output length. If non-STL features significantly change
        # the aligned length before windowing, this might need adjustment or switching
        # fully to the date-based alignment method developed previously.
        # For now, keeping exactly as provided by user in the initial prompt.
        print("\n--- 7. Baseline and Target Processing (Original Logic) ---")
        # --- Re-calculate window_data_start_index based on original STL output ---
        # This assumes STL was the primary driver of length before, needs re-eval if only using others
        if config.get('use_stl') and 'stl_trend' in features_train: # If STL was used
             stl_train_len_before_align = len(self._rolling_stl(log_train, config['stl_window'], config['stl_period'])[0]) # Re-calc length quickly
             window_data_start_index = len(log_train) - stl_train_len_before_align
             print(f"Using STL-based start index offset: {window_data_start_index}")
        elif 'log_return' in features_train: # Fallback if STL not used, use alignment length
             window_data_start_index = len(log_train) - aligned_len_train
             print(f"Using alignment-based start index offset: {window_data_start_index}")
        else:
             raise RuntimeError("Cannot determine window_data_start_index for baseline/target alignment.")


        num_train_windows = X_train_combined.shape[0] # Use actual number of samples generated
        num_val_windows = X_val_combined.shape[0]
        num_test_windows = X_test_combined.shape[0]

        # Calculate baseline start index in original 'close' series
        baseline_start_index = window_data_start_index + window_size - 1
        print(f"Calculated baseline start index in original series: {baseline_start_index}")

        # Extract baseline values
        if baseline_start_index + num_train_windows > len(close_train): raise ValueError("Baseline train slice out of bounds.")
        baseline_train = close_train[baseline_start_index : baseline_start_index + num_train_windows]
        if baseline_start_index + num_val_windows > len(close_val): raise ValueError("Baseline val slice out of bounds.")
        baseline_val = close_val[baseline_start_index : baseline_start_index + num_val_windows]
        if baseline_start_index + num_test_windows > len(close_test): raise ValueError("Baseline test slice out of bounds.")
        baseline_test = close_test[baseline_start_index : baseline_start_index + num_test_windows]

        # Verify lengths
        if len(baseline_train) != num_train_windows: raise ValueError("Baseline train length mismatch.")
        if len(baseline_val) != num_val_windows: raise ValueError("Baseline val length mismatch.")
        if len(baseline_test) != num_test_windows: raise ValueError("Baseline test length mismatch.")
        print(f"Baseline shapes: Train: {baseline_train.shape}, Val: {baseline_val.shape}, Test: {baseline_test.shape}")


        # Process Multi-Horizon Targets (Original Logic)
        target_column = config["target_column"]
        if target_column not in y_train_df.columns: raise ValueError(f"Column '{target_column}' not found in training Y data.")
        target_train_raw = y_train_df[target_column].astype(np.float32).values
        target_val_raw = y_val_df[target_column].astype(np.float32).values
        target_test_raw = y_test_df[target_column].astype(np.float32).values

        y_train_list_final = []
        y_val_list_final = []
        y_test_list_final = []
        # --- Use correct config name ---
        use_returns_as_target = config.get("use_returns_as_target", True) # Corrected name

        print(f"Processing targets for horizons: {predicted_horizons} (Use Returns={use_returns_as_target})...")
        for h in predicted_horizons:
            target_start_idx = baseline_start_index + h
            target_end_idx_train = target_start_idx + num_train_windows # End index exclusive
            target_end_idx_val   = target_start_idx + num_val_windows
            target_end_idx_test  = target_start_idx + num_test_windows

            if target_end_idx_train > len(target_train_raw): raise ValueError(f"Not enough target_train_raw data for H={h}")
            if target_end_idx_val > len(target_val_raw): raise ValueError(f"Not enough target_val_raw data for H={h}")
            if target_end_idx_test > len(target_test_raw): raise ValueError(f"Not enough target_test_raw data for H={h}")

            target_train_h = target_train_raw[target_start_idx : target_end_idx_train]
            target_val_h = target_val_raw[target_start_idx : target_end_idx_val]
            target_test_h = target_test_raw[target_start_idx : target_end_idx_test]

            if len(target_train_h) != num_train_windows: raise ValueError(f"Target train length mismatch H={h}")
            if len(target_val_h) != num_val_windows: raise ValueError(f"Target val length mismatch H={h}")
            if len(target_test_h) != num_test_windows: raise ValueError(f"Target test length mismatch H={h}")

            if use_returns_as_target: # Use corrected flag name
                target_train_h = target_train_h - baseline_train
                target_val_h = target_val_h - baseline_val
                target_test_h = target_test_h - baseline_test

            y_train_list_final.append(target_train_h.astype(np.float32))
            y_val_list_final.append(target_val_h.astype(np.float32))
            y_test_list_final.append(target_test_h.astype(np.float32))

        # Assign Dates for Targets (using dates from windowing)
        y_dates_train = x_dates_train # Date corresponds to end of input window
        y_dates_val = x_dates_val
        y_dates_test = x_dates_test
        print("Target processing complete.")

        # Prepare Test Close Prices for output (Aligned with baseline/targets)
        test_close_prices_aligned = baseline_test # Baseline is the close price at end of window

        # --- 8. Final Date Consistency Check (Original Logic) ---
        print("\n--- 8. Final Date Consistency Checks ---")
        # Note: baseline_* arrays are prices, not dates. Checking X dates.
        verify_date_consistency([list(x_dates_train) if x_dates_train is not None else None], "Final Train X Dates")
        verify_date_consistency([list(x_dates_val) if x_dates_val is not None else None], "Final Val X Dates")
        verify_date_consistency([list(x_dates_test) if x_dates_test is not None else None], "Final Test X Dates")


        # --- 9. Prepare Return Dictionary ---
        # Uses the combined features for x_* and preserves original y/baseline structure
        print("\n--- 9. Preparing Final Output ---")
        ret = {
            # --- USE COMBINED FEATURES ---
            "x_train": X_train_combined,
            "x_val": X_val_combined,
            "x_test": X_test_combined,
            # --- REMOVE OLD SEPARATE STL KEYS ---
            # "x_train_trend": ..., (Now part of x_train if use_stl=True)
            # --- KEEP ORIGINAL Y / BASELINE STRUCTURE ---
            "y_train": y_train_list_final,
            "y_val": y_val_list_final,
            "y_test": y_test_list_final,
            "x_train_dates": x_dates_train,
            "y_train_dates": y_dates_train, # Same as x_dates_train
            "x_val_dates": x_dates_val,
            "y_val_dates": y_dates_val,     # Same as x_dates_val
            "x_test_dates": x_dates_test,
            "y_test_dates": y_dates_test,     # Same as x_dates_test
            "baseline_train": baseline_train,
            "baseline_val": baseline_val,
            "baseline_test": baseline_test,
            "baseline_train_dates": y_dates_train,
            "baseline_val_dates": y_dates_val,
            "baseline_test_dates": y_dates_test,
            "test_close_prices": test_close_prices_aligned,
            # --- ADD FEATURE NAMES ---
            "feature_names": feature_names
        }

        # Optional: Clean up memory
        del x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df
        del features_train, features_val, features_test

        print("\n" + "="*15 + " Preprocessing Finished " + "="*15)
        return ret

    # --- Original run_preprocessing (kept exactly as provided by user) ---
    def run_preprocessing(self, config):
        """Convenience method to execute data processing."""
        # Merge instance defaults with passed config
        run_config = self.params.copy()
        run_config.update(config)
        # --- MODIFICATION: Call set_params to resolve defaults AFTER merge ---
        self.set_params(**run_config)
        # --- Run with the fully resolved self.params ---
        return self.process_data(self.params)
    # --- End of original run_preprocessing ---
# Example placeholder for create_sliding_windows if needed outside class
# def create_sliding_windows_standalone(data, window_size, time_horizon, date_times=None): ...