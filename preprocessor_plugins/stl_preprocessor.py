# Ensure these imports are present at the top of the file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # Added for normalization
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
import json
import os # Added for checking file paths

# Assuming load_csv is correctly imported from app.data_handler
try:
    from app.data_handler import load_csv
    # from app.data_handler import write_csv # Keep commented as per original
except ImportError:
    print("CRITICAL ERROR: Could not import 'load_csv' from 'app.data_handler'.")
    raise

from scipy.signal import hilbert
from scipy.stats import shapiro

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


# Include the user-provided verify_date_consistency function
# (Kept exactly as provided in the prompt)
def verify_date_consistency(date_lists, dataset_name):
    """
    Verifies that all date arrays in date_lists have the same first and last elements.
    Prints a warning if any array does not match.
    """
    if not date_lists: return
    valid_date_lists = [d for d in date_lists if d is not None and len(d) > 0]
    if not valid_date_lists: return
    first_len = len(valid_date_lists[0])
    if not all(len(d) == first_len for d in valid_date_lists):
         print(f"WARN: Length mismatch in {dataset_name} dates: {[len(d) for d in valid_date_lists]}")
         return
    try: first = valid_date_lists[0][0]; last = valid_date_lists[0][-1]
    except IndexError: print(f"WARN: Could not access first/last element in {dataset_name} dates."); return
    consistent = True
    for i, d in enumerate(valid_date_lists):
        try:
            if len(d) > 0:
                if d[0] != first or d[-1] != last:
                    print(f"Warning: Date array {i} in {dataset_name} range ({d[0]} to {d[-1]}) does not match first array ({first} to {last}).")
                    consistent = False
            else: print(f"Warning: Date array {i} in {dataset_name} is empty."); consistent = False
        except Exception as e: print(f"Warning: Could not compare dates in {dataset_name} for array {i}. Error: {e}."); consistent = False
    # if consistent: print(f"{dataset_name} date ranges appear consistent.") # Less verbose


class PreprocessorPlugin:
    # Default plugin parameters - Merging original and new params
    plugin_params = {
        # --- File Paths ---
        "x_train_file": "data/x_train.csv",
        "y_train_file": "data/y_train.csv",
        "x_validation_file": "data/x_val.csv",
        "y_validation_file": "data/y_val.csv",
        "x_test_file": "data/x_test.csv",
        "y_test_file": "data/y_test.csv",
        # --- Data Loading ---
        "headers": True,
        "max_steps_train": None, "max_steps_val": None, "max_steps_test": None,
        "target_column": "TARGET", # Target col name in Y files (from original)
        # --- Windowing & Horizons ---
        "window_size": 48, # Original Default
        # "time_horizon": 6, # Original single horizon - Replaced by multi-horizon
        "predicted_horizons": [1, 6, 12, 24], # New multi-horizon (used for Y calc)
        # --- Feature Engineering Flags ---
        "use_stl": False,
        "use_wavelets": True,
        "use_multi_tapper": False,
        "use_returns": True, # Original name for use_returns_as_target
        "normalize_features": True,
        # --- STL Parameters ---
        "stl_period": 24,
        "stl_window": 48, # Default, will be resolved if None later
        "stl_trend": 49,  # Default, will be resolved if None later
        "stl_plot_file": "stl_plot.png",
        # --- Wavelet Parameters ---
        "wavelet_name": 'db4',
        "wavelet_levels": 2, # Set based on previous error message
        "wavelet_mode": 'symmetric',
        "wavelet_plot_file": "wavelet_features.png",
        # --- Multitaper Parameters ---
        # Note: "multitaper" is the standard spelling
        "mtm_window_len": 168,
        "mtm_step": 1,
        "mtm_time_bandwidth": 5.0,
        "mtm_num_tapers": None,
        "mtm_freq_bands": [(0, 0.01), (0.01, 0.06), (0.06, 0.2), (0.2, 0.5)],
        "tapper_plot_file": "mtm_features.png",
        "tapper_plot_points": 480, # Consistent plot points
        # "pos_encoding_dim": 16 # Keep commented as per original
    }
    # Update debug vars list
    plugin_debug_vars = [
        "window_size", #"time_horizon", # Removed single horizon
        "predicted_horizons", # Added multi-horizon
        "use_returns", # Keep original name
        "normalize_features",
        "use_stl", "stl_period", "stl_window", "stl_trend", "stl_plot_file",
        "use_wavelets", "wavelet_name", "wavelet_levels", "wavelet_mode", "wavelet_plot_file",
        "use_multi_tapper", "mtm_window_len", "mtm_step", "mtm_time_bandwidth", "mtm_num_tapers", "mtm_freq_bands", "tapper_plot_file", "tapper_plot_points"
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.scalers = {}

    def set_params(self, **kwargs):
        """Update plugin parameters with global configuration."""
        for key, value in kwargs.items():
            self.params[key] = value
        # Apply logic based on updated params
        config = self.params
        if config.get("stl_period") is not None and config.get("stl_period") > 1:
            if config.get("stl_window") is None: config["stl_window"] = 2 * config["stl_period"] + 1
            if config.get("stl_trend") is None:
                current_stl_window = config.get("stl_window")
                if current_stl_window is not None and current_stl_window > 3:
                     try:
                         trend_calc = int(1.5 * config["stl_period"] / (1 - 1.5 / current_stl_window)) + 1
                         config["stl_trend"] = max(3, trend_calc) # Ensure minimum length
                     except ZeroDivisionError: config["stl_trend"] = config["stl_period"] + 1
                else: config["stl_trend"] = config["stl_period"] + 1
            if config.get("stl_trend") is not None and config["stl_trend"] % 2 == 0: config["stl_trend"] += 1

    def get_debug_info(self):
        """Return debug information for the plugin."""
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """Add plugin debug info to the provided dictionary."""
        debug_info.update(self.get_debug_info())

    # --- Helper Methods ---
    def _load_data(self, file_path, max_rows, headers):
        # (Implementation from previous step - uses YOUR load_csv)
        print(f"Loading data: {file_path} (Max rows: {max_rows})...", end="")
        try:
            df = load_csv(file_path, headers=headers, max_rows=max_rows)
            if df is None or df.empty: raise ValueError(f"load_csv returned None or empty for {file_path}")
            print(f" Done. Shape: {df.shape}")
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"Attempting to convert index of {file_path} to DatetimeIndex...", end="")
                original_index_name = df.index.name
                try: df.index = pd.to_datetime(df.index); print(" Success (from index).")
                except Exception:
                    try: df.index = pd.to_datetime(df.iloc[:, 0]); print(" Success (from first column).")
                    except Exception as e_col: print(f" FAILED. Error: {e_col}. Dates unavailable."); df.index = None
                if original_index_name: df.index.name = original_index_name
            required_cols = ["CLOSE"]
            target_col_name = self.params.get("target_column", "TARGET")
            # Check target only in Y files (assuming 'y_' in filename indicates Y file)
            if 'y_' in os.path.basename(file_path).lower(): required_cols.append(target_col_name)
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols: raise ValueError(f"Missing required columns in {file_path}: {missing_cols}. Available: {df.columns.tolist()}")
            return df
        except FileNotFoundError: print(f"\nCRITICAL ERROR: File not found at {file_path}."); raise
        except Exception as e: print(f"\nCRITICAL ERROR loading or processing {file_path}: {e}"); import traceback; traceback.print_exc(); raise

    def _normalize_series(self, series, name, fit=False):
        """Normalizes a time series using StandardScaler."""
        # (Implementation from previous step)
        if not self.params.get("normalize_features", True): return series.astype(np.float32)
        series = series.astype(np.float32)
        if np.any(np.isnan(series)) or np.any(np.isinf(series)):
            print(f"WARN: NaNs/Infs in '{name}' before normalization. Filling...", end="")
            series = pd.Series(series).fillna(method='ffill').fillna(method='bfill').values
            if np.any(np.isnan(series)) or np.any(np.isinf(series)):
                 print(f" Filling FAILED. Filling remaining with 0.", end="")
                 series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
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
            else: scaler.fit(data_reshaped)
            self.scalers[name] = scaler
        else:
            if name not in self.scalers: raise RuntimeError(f"Scaler for '{name}' not fitted.")
            scaler = self.scalers[name]
        normalized_data = scaler.transform(data_reshaped)
        return normalized_data.flatten()

    # --- Original _rolling_stl (kept exactly as provided by user) ---
    def _rolling_stl(self, series, stl_window, period):
        # Uses original rolling STL logic from user provided snippet
        print("Performing rolling STL decomposition (Original Method)...")
        n = len(series)
        num_points = n - stl_window + 1
        if num_points <= 0: raise ValueError(f"stl_window ({stl_window}) is too large for the series length ({n}).")
        trend = np.zeros(num_points); seasonal = np.zeros(num_points); resid = np.zeros(num_points)
        # --- Get stl_trend from resolved self.params ---
        stl_trend = self.params.get("stl_trend") # Use resolved trend smoother length

        for i in tqdm(range(stl_window, n + 1), desc="STL Decomposition", unit="window", disable=None, leave=False):
            window = series[i - stl_window: i]
            # --- Validate and Adjust stl_trend locally ---
            current_stl_trend = stl_trend
            if current_stl_trend is not None:
                if not isinstance(current_stl_trend, int) or current_stl_trend <=0: current_stl_trend = None
                elif current_stl_trend >= len(window): current_stl_trend = len(window) - 1 if len(window) > 1 else None
                if current_stl_trend is not None and current_stl_trend % 2 == 0: current_stl_trend += 1
            # --- End validation ---
            try:
                 stl = STL(window, period=period, trend=current_stl_trend, robust=True) # Use adjusted trend
                 result = stl.fit()
                 trend[i - stl_window] = result.trend[-1]
                 seasonal[i - stl_window] = result.seasonal[-1]
                 resid[i - stl_window] = result.resid[-1]
            except Exception as e:
                 print(f"WARN: STL fit failed window ending {i-1}. Err: {e}. NaN used.")
                 trend[i - stl_window]=np.nan; seasonal[i - stl_window]=np.nan; resid[i - stl_window]=np.nan
        trend = pd.Series(trend).fillna(method='ffill').fillna(method='bfill').values
        seasonal = pd.Series(seasonal).fillna(method='ffill').fillna(method='bfill').values
        resid = pd.Series(resid).fillna(method='ffill').fillna(method='bfill').values
        return trend, seasonal, resid
    # --- End of original _rolling_stl ---

    # --- Original _plot_decomposition (kept exactly as provided by user, with robust slicing) ---
    def _plot_decomposition(self, series, trend, seasonal, resid, file_path):
        # Uses original plotting logic, limited points
        print(f"Plotting STL decomposition to {file_path}...")
        plot_points = 480 # Limit plot points for readability (from original)
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
            # Using original subplot structure and labels
            plt.figure(figsize=(12, 9))
            plt.subplot(411); plt.plot(series_plot, label="Log-Transformed Series"); plt.legend(loc="upper left"); plt.grid(True, alpha=0.5)
            plt.subplot(412); plt.plot(trend_plot, label="Trend", color="orange"); plt.legend(loc="upper left"); plt.grid(True, alpha=0.5)
            plt.subplot(413); plt.plot(seasonal_plot, label="Seasonal", color="green"); plt.legend(loc="upper left"); plt.grid(True, alpha=0.5)
            plt.subplot(414); plt.plot(resid_plot, label="Residual", color="red"); plt.legend(loc="upper left"); plt.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.savefig(file_path, dpi=300)
            plt.close()
            print(f"STL decomposition plot saved to {file_path}")
        except Exception as e:
             print(f"WARN: Failed to plot STL decomposition: {e}")
             plt.close()
    # --- End of original _plot_decomposition ---

    # --- Updated _compute_wavelet_features with TypeError Fix Attempt ---
    def _compute_wavelet_features(self, series):
        """Computes Wavelet features using MODWT (pywt.swt). Includes TypeError fix attempts."""
        if pywt is None: print("ERROR: pywt library not installed."); return {}
        name = self.params['wavelet_name']; levels = self.params['wavelet_levels'];
        n_original_check = len(series) if hasattr(series, '__len__') else 'N/A (Not iterable?)'

        # --- Input Check Block ---
        # print(f"\nDEBUG Wavelet Input: Type={type(series)}, Shape={getattr(series, 'shape', 'N/A')}, Len={n_original_check}") # Keep for debugging if needed
        if not isinstance(series, (np.ndarray, pd.Series, list)) or len(series) < 2:
            print(f"ERROR: Wavelet input not array/list or too short.")
            return {}
        try:
            # --- Attempting Fix 1: Ensure float64 and check NaNs ---
            series_clean = np.asarray(series, dtype=np.float64)
            if np.any(np.isnan(series_clean)) or np.any(np.isinf(series_clean)):
                print("WARN: NaNs/Infs in wavelet input. Filling...", end="")
                series_clean = pd.Series(series_clean).fillna(method='ffill').fillna(method='bfill').values
                if np.any(np.isnan(series_clean)) or np.any(np.isinf(series_clean)):
                    print(" Fill FAILED. Skipping."); return {}
                print(" Done.")
            # --- End Fix 1 ---
        except Exception as e: print(f"ERROR during wavelet input validation: {e}"); return {}
        # --- End Input Check Block ---

        if levels is None:
            try: levels = pywt.swt_max_level(len(series_clean)); print(f"Auto wavelet levels: {levels}")
            except Exception as e: print(f"ERROR calculating max wavelet levels: {e}."); return {}
        if levels <= 0: print(f"ERROR: Wavelet levels ({levels}) not positive."); return {}

        print(f"Computing Wavelets (MODWT/SWT): {name}, Levels={levels}...", end="")
        try:
            # Use the cleaned float64 series
            coeffs = pywt.swt(series_clean, wavelet=name, level=levels, trim_approx=True, norm=True)

            features = {'approx_L{}'.format(levels): coeffs[0][0]}
            for i in range(levels): features['detail_L{}'.format(levels - i)] = coeffs[i][1]
            n_original_len = len(series_clean)
            for k, v in features.items():
                 if len(v) != n_original_len: print(f"WARN: Wavelet '{k}' len({len(v)})!=orig({n_original_len}).")
            print(f" Done ({len(features)} channels).")
            return features
        except TypeError as e:
             print(f" FAILED. TypeError: {e}") # Keep reporting specific error
             # --- Attempting Fix 2: Add context if possible ---
             print(f"      Occurred during pywt.swt call. Input length={len(series_clean)}, Levels={levels}, Wavelet='{name}'.")
             # --- End Fix 2 ---
             return {}
        except Exception as e: print(f" FAILED. Error: {e}"); import traceback; traceback.print_exc(); return {}
    # --- End of _compute_wavelet_features ---


    def _plot_wavelets(self, original_series, wavelet_features, file_path):
        """Plots original series and computed Wavelet features."""
        # (Implementation from previous correct step - kept the same, uses 480 points)
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
        else: print("WARN: No wavelet features were successfully plotted."); plt.close() # Ensure plot closed


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
        fft_freqs = np.fft.rfftfreq(window_len)
        freq_masks = [(fft_freqs >= f_low) & (fft_freqs < f_high) for f_low, f_high in freq_bands]
        for i in tqdm(range(num_windows), desc="MTM Calc", unit="window", disable=True, leave=False):
            start = i * step; end = start + window_len; window_data = series[start:end]
            if np.any(np.isnan(window_data)): continue
            spectra = np.zeros((num_tapers, len(fft_freqs)))
            for k in range(num_tapers): spectra[k, :] = np.abs(np.fft.rfft(window_data * tapers[k]))**2
            avg_spectrum = np.mean(spectra, axis=0)
            for band_idx, mask in enumerate(freq_masks):
                 if np.any(mask): mtm_features[f"band_{band_idx}"][i] = np.mean(avg_spectrum[mask])
                 else: mtm_features[f"band_{band_idx}"][i] = 0.0
        for name in mtm_features: mtm_features[name] = pd.Series(mtm_features[name]).fillna(method='ffill').fillna(method='bfill').values
        print(f" Done ({len(mtm_features)} channels).")
        return mtm_features

    def _plot_mtm(self, mtm_features, file_path, points_to_plot=500):
        """Plots computed Multitaper features (power bands)."""
        # (Implementation from previous correct step - uses points_to_plot)
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
    # --- !!! IMPORTANT NOTE: This version calculates targets internally, which we override later !!! ---
    def create_sliding_windows(self, data, window_size, time_horizon, date_times=None):
        """
        Creates sliding windows for a univariate series.
        Original user version - calculates targets internally but they are ignored later.
        Requires a single `time_horizon`.
        """
        print(f"Creating sliding windows (Original Method - Size={window_size}, Horizon={time_horizon})...", end="")
        windows = []
        targets = [] # This target list will be ignored by the main process_data logic now
        date_windows = []
        n = len(data)
        num_possible_windows = n - window_size - time_horizon + 1 # Original calculation
        if num_possible_windows <= 0:
             print(f" WARN: Data length ({n}) too short for Win={window_size} + Horizon={time_horizon}. No windows.")
             return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32), np.array(date_windows, dtype=object)

        for i in range(num_possible_windows):
            window = data[i: i + window_size]
            target = data[i + window_size + time_horizon - 1] # Original target calculation (ignored)
            windows.append(window)
            targets.append(target) # Ignored target
            if date_times is not None:
                date_index = i + window_size - 1
                if date_index < len(date_times): date_windows.append(date_times[date_index])
                else: date_windows.append(None)

        # Convert dates
        if date_times is not None:
             date_windows_arr = np.array(date_windows, dtype=object)
             if all(isinstance(d, pd.Timestamp) for d in date_windows if d is not None):
                  try: date_windows_arr = np.array(date_windows, dtype='datetime64[ns]')
                  except (ValueError, TypeError): pass
        else: date_windows_arr = np.array(date_windows, dtype=object)

        print(f" Done ({len(windows)} windows).")
        return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32), date_windows_arr # Return dates array
    # --- End of original create_sliding_windows ---

    def align_features(self, feature_dict, base_length):
        """Aligns feature time series to a common length by truncating the beginning."""
        # (Implementation from previous correct step - kept the same)
        aligned_features = {}
        min_len = base_length; feature_lengths = {'base': base_length}
        valid_keys = [k for k, v in feature_dict.items() if v is not None]
        if not valid_keys: return {}, 0
        for name in valid_keys: feature_lengths[name] = len(feature_dict[name]); min_len = min(min_len, feature_lengths[name])
        needs_alignment = any(l != min_len for l in feature_lengths.values() if l > 0)
        if needs_alignment:
             for name in valid_keys:
                 series = feature_dict[name]; current_len = len(series)
                 if current_len > min_len: aligned_features[name] = series[current_len - min_len:]
                 elif current_len == min_len: aligned_features[name] = series
                 else: print(f"WARN: Feature '{name}' length ({current_len}) < target align length ({min_len})."); aligned_features[name] = None
        else: aligned_features = {k: feature_dict[k] for k in valid_keys}
        final_lengths = {name: len(s) for name, s in aligned_features.items() if s is not None}
        unique_lengths = set(final_lengths.values())
        if len(unique_lengths) > 1: raise RuntimeError(f"Alignment FAILED! Inconsistent final lengths: {final_lengths}")
        return aligned_features, min_len


    # --- process_data Method ---
    def process_data(self, config):
        """
        Processes data for forecasting with optional features, using original Y/Baseline logic.
        """
        # 0. Setup & Parameter Resolution
        print("\n" + "="*15 + " Starting Preprocessing " + "="*15)
        self.set_params(**config) # Resolve defaults based on final config
        config = self.params # Use the fully resolved params
        self.scalers = {}

        # Get key parameters used in multiple places
        window_size = config['window_size']
        predicted_horizons = config['predicted_horizons']
        if not isinstance(predicted_horizons, list) or not predicted_horizons:
            raise ValueError("'predicted_horizons' must be a non-empty list of integers.")
        # *** Use max_horizon for windowing consistency ***
        max_horizon = max(predicted_horizons)
        stl_window = config.get('stl_window') # Needed for original offset calc, ensure resolved

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
        log_train = np.log1p(np.maximum(0, close_train)) # log1p for robustness
        log_val = np.log1p(np.maximum(0, close_val))
        log_test = np.log1p(np.maximum(0, close_test))
        print(f"Log transform applied. Train shape: {log_train.shape}")
        dates_train = x_train_df.index if isinstance(x_train_df.index, pd.DatetimeIndex) else None
        dates_val = x_val_df.index if isinstance(x_val_df.index, pd.DatetimeIndex) else None
        dates_test = x_test_df.index if isinstance(x_test_df.index, pd.DatetimeIndex) else None

        # --- 3. Feature Generation (Conditional) ---
        print("\n--- 3. Feature Generation ---")
        features_train, features_val, features_test = {}, {}, {}
        # 3.a. Log Returns
        log_ret_train=np.diff(log_train,prepend=log_train[0]); features_train['log_return']=self._normalize_series(log_ret_train,'log_return',fit=True)
        log_ret_val=np.diff(log_val,prepend=log_val[0]); features_val['log_return']=self._normalize_series(log_ret_val,'log_return',fit=False)
        log_ret_test=np.diff(log_test,prepend=log_test[0]); features_test['log_return']=self._normalize_series(log_ret_test,'log_return',fit=False)
        print("Generated: Log Returns (Normalized)")
        # 3.b. STL
        if config.get('use_stl'):
             print("Attempting STL features...")
             stl_period=config['stl_period']; # stl_window already resolved
             try:
                 trend_train, seasonal_train, resid_train = self._rolling_stl(log_train, stl_window, stl_period)
                 trend_val, seasonal_val, resid_val = self._rolling_stl(log_val, stl_window, stl_period)
                 trend_test, seasonal_test, resid_test = self._rolling_stl(log_test, stl_window, stl_period)
                 if len(trend_train) > 0:
                      features_train['stl_trend']=self._normalize_series(trend_train,'stl_trend',fit=True)
                      features_train['stl_seasonal']=self._normalize_series(seasonal_train,'stl_seasonal',fit=True)
                      features_train['stl_resid']=self._normalize_series(resid_train,'stl_resid',fit=True)
                      features_val['stl_trend']=self._normalize_series(trend_val,'stl_trend',fit=False)
                      features_val['stl_seasonal']=self._normalize_series(seasonal_val,'stl_seasonal',fit=False)
                      features_val['stl_resid']=self._normalize_series(resid_val,'stl_resid',fit=False)
                      features_test['stl_trend']=self._normalize_series(trend_test,'stl_trend',fit=False)
                      features_test['stl_seasonal']=self._normalize_series(seasonal_test,'stl_seasonal',fit=False)
                      features_test['stl_resid']=self._normalize_series(resid_test,'stl_resid',fit=False)
                      print("Generated: STL Trend, Seasonal, Residual (Normalized)")
                      stl_plot_file = config.get("stl_plot_file")
                      if stl_plot_file: self._plot_decomposition(log_train[len(log_train)-len(trend_train):], trend_train, seasonal_train, resid_train, stl_plot_file)
                 else: print("WARN: STL output zero length.")
             except Exception as e: print(f"ERROR processing STL: {e}. Skipping.")
        else: print("Skipped: STL features.")
        # 3.c. Wavelets
        if config.get('use_wavelets'):
             print("Attempting Wavelet features...")
             try:
                 wav_features_train = self._compute_wavelet_features(log_train) # Using fixed version
                 if wav_features_train:
                      wav_features_val = self._compute_wavelet_features(log_val)
                      wav_features_test = self._compute_wavelet_features(log_test)
                      for name in wav_features_train.keys():
                           features_train[f'wav_{name}']=self._normalize_series(wav_features_train[name],f'wav_{name}',fit=True)
                           if name in wav_features_val: features_val[f'wav_{name}']=self._normalize_series(wav_features_val[name],f'wav_{name}',fit=False)
                           if name in wav_features_test: features_test[f'wav_{name}']=self._normalize_series(wav_features_test[name],f'wav_{name}',fit=False)
                      print(f"Generated: {len(wav_features_train)} Wavelet features (Normalized).")
                      wavelet_plot_file = config.get("wavelet_plot_file")
                      if wavelet_plot_file: self._plot_wavelets(log_train, wav_features_train, wavelet_plot_file)
                 else: print("WARN: Wavelet computation for train failed/returned no features.")
             except Exception as e: print(f"ERROR processing Wavelets: {e}. Skipping.")
        else: print("Skipped: Wavelet features.")
        # 3.d. Multitaper
        if config.get('use_multi_tapper'):
             print("Attempting MTM features...")
             try:
                 mtm_features_train = self._compute_mtm_features(log_train)
                 if mtm_features_train:
                      mtm_features_val = self._compute_mtm_features(log_val)
                      mtm_features_test = self._compute_mtm_features(log_test)
                      for name in mtm_features_train.keys():
                           features_train[f'mtm_{name}']=self._normalize_series(mtm_features_train[name],f'mtm_{name}',fit=True)
                           if name in mtm_features_val: features_val[f'mtm_{name}']=self._normalize_series(mtm_features_val[name],f'mtm_{name}',fit=False)
                           if name in mtm_features_test: features_test[f'mtm_{name}']=self._normalize_series(mtm_features_test[name],f'mtm_{name}',fit=False)
                      print(f"Generated: {len(mtm_features_train)} MTM features (Normalized).")
                      tapper_plot_file = config.get("tapper_plot_file")
                      if tapper_plot_file: self._plot_mtm(mtm_features_train, tapper_plot_file, config.get("tapper_plot_points"))
                 else: print("WARN: MTM computation for train failed/returned no features.")
             except Exception as e: print(f"ERROR processing MTM: {e}. Skipping.")
        else: print("Skipped: MTM features.")

        # --- 4. Align Feature Lengths ---
        print("\n--- 4. Aligning Feature Lengths ---")
        base_len_train=len(features_train['log_return']); base_len_val=len(features_val['log_return']); base_len_test=len(features_test['log_return'])
        features_train, aligned_len_train = self.align_features(features_train, base_len_train)
        features_val, aligned_len_val = self.align_features(features_val, base_len_val)
        features_test, aligned_len_test = self.align_features(features_test, base_len_test)
        dates_train_aligned = dates_train[-aligned_len_train:] if dates_train is not None and aligned_len_train > 0 else None
        dates_val_aligned = dates_val[-aligned_len_val:] if dates_val is not None and aligned_len_val > 0 else None
        dates_test_aligned = dates_test[-aligned_len_test:] if dates_test is not None and aligned_len_test > 0 else None
        print(f"Final aligned feature length: Train={aligned_len_train}, Val={aligned_len_val}, Test={aligned_len_test}")

        # --- 5. Windowing Features & Stacking ---
        print("\n--- 5. Windowing Features & Channel Stacking ---")
        X_train_channels, X_val_channels, X_test_channels = [], [], []
        feature_names = []; x_dates_train, x_dates_val, x_dates_test = None, None, None; first_feature_dates_captured = False
        feature_order = ['log_return'] # Determine order
        if config.get('use_stl'): feature_order.extend(['stl_trend', 'stl_seasonal', 'stl_resid'])
        if config.get('use_wavelets'): feature_order.extend(sorted([k for k in features_train if k.startswith('wav_')]))
        if config.get('use_multi_tapper'): feature_order.extend(sorted([k for k in features_train if k.startswith('mtm_')]))
        if not config.get('use_stl') and not config.get('use_wavelets') and not config.get('use_multi_tapper'): feature_order = ['log_return']

        print(f"Attempting to window features: {feature_order}")
        # --- Use max_horizon for windowing function ---
        time_horizon_for_windowing = max_horizon

        for name in feature_order:
            if name in features_train and features_train[name] is not None and \
               name in features_val and features_val[name] is not None and \
               name in features_test and features_test[name] is not None:
                print(f"Windowing feature: {name}...", end="")
                # --- Call YOUR create_sliding_windows, passing max_horizon ---
                win_train, _, dates_win_train = self.create_sliding_windows(features_train[name], window_size, time_horizon_for_windowing, dates_train_aligned)
                win_val, _, dates_win_val   = self.create_sliding_windows(features_val[name], window_size, time_horizon_for_windowing, dates_val_aligned)
                win_test, _, dates_win_test = self.create_sliding_windows(features_test[name], window_size, time_horizon_for_windowing, dates_test_aligned)
                if win_train.shape[0] > 0 and win_val.shape[0] > 0 and win_test.shape[0] > 0:
                    X_train_channels.append(win_train); X_val_channels.append(win_val); X_test_channels.append(win_test)
                    feature_names.append(name); print(" Appended.")
                    if not first_feature_dates_captured:
                         x_dates_train, x_dates_val, x_dates_test = dates_win_train, dates_win_val, dates_win_test
                         first_feature_dates_captured = True; print(f"Captured dates from '{name}'.")
                else: print(f" Skipping channel '{name}' (windowing produced 0 samples).")
            # else: print(f"Feature '{name}' not available, skipping.") # Verbose

        # --- 6. Stack channels ---
        if not X_train_channels: raise RuntimeError("No feature channels available after windowing!")
        print("\n--- 6. Stacking Feature Channels ---")
        num_samples_train = X_train_channels[0].shape[0]; num_samples_val = X_val_channels[0].shape[0]; num_samples_test = X_test_channels[0].shape[0]
        if not all(c.shape[0] == num_samples_train for c in X_train_channels): raise RuntimeError("Inconsistent samples in train channels.")
        if not all(c.shape[0] == num_samples_val for c in X_val_channels): raise RuntimeError("Inconsistent samples in val channels.")
        if not all(c.shape[0] == num_samples_test for c in X_test_channels): raise RuntimeError("Inconsistent samples in test channels.")
        X_train_combined = np.stack(X_train_channels, axis=-1).astype(np.float32)
        X_val_combined = np.stack(X_val_channels, axis=-1).astype(np.float32)
        X_test_combined = np.stack(X_test_channels, axis=-1).astype(np.float32)
        print(f"Final X shapes: Train={X_train_combined.shape}, Val={X_val_combined.shape}, Test={X_test_combined.shape}")
        print(f"Included features: {feature_names}")

        # --- 7. Baseline & Target Processing (Reverting to User's Original Logic) ---
        print("\n--- 7. Baseline and Target Processing (Using Original Logic Structure) ---")
        target_column = config["target_column"]
        use_returns = config.get("use_returns", False) # Original flag name

        # --- Original Offset Calculation ---
        # Requires stl_window even if STL not used - ensure it's resolved
        if stl_window is None:
             print("WARN: stl_window not resolved, defaulting to window_size for offset calculation.")
             effective_stl_window = window_size # Fallback if STL wasn't run/configured
        else:
             effective_stl_window = stl_window
        original_offset = effective_stl_window + window_size - 2
        print(f"Calculated original logic offset: {original_offset} (using effective_stl_window={effective_stl_window})")

        # --- Load Raw Target Data ---
        if target_column not in y_train_df.columns: raise ValueError(f"Column '{target_column}' not found in training Y data.")
        target_train_raw = y_train_df[target_column].astype(np.float32).values
        target_val_raw = y_val_df[target_column].astype(np.float32).values
        target_test_raw = y_test_df[target_column].astype(np.float32).values

        # --- Apply Original Slicing to Raw Targets & Dates ---
        # Slice raw targets based on original offset calculation
        target_train_sliced = target_train_raw[original_offset:]
        target_val_sliced = target_val_raw[original_offset:]
        target_test_sliced = target_test_raw[original_offset:]

        # Slice original dates based on original offset (and max_horizon for length)
        # The length needs to align with the *expected* number of samples *before* the final shift
        # Expected length before shift = num_samples + max_horizon - 1 ??? This is tricky.
        # Let's align the DATES using the x_dates_* derived from the actual X windows.
        y_dates_train = x_dates_train
        y_dates_val = x_dates_val
        y_dates_test = x_dates_test
        print("Using X window dates for Y/Baseline date alignment.")

        # --- Calculate Baseline using original logic's slice indices ---
        # Slice based on offset, ensure length matches final X samples
        baseline_slice_end_train = original_offset + num_samples_train
        baseline_slice_end_val = original_offset + num_samples_val
        baseline_slice_end_test = original_offset + num_samples_test

        if original_offset >= len(close_train) or baseline_slice_end_train > len(close_train): raise ValueError(f"Baseline train indices [{original_offset}:{baseline_slice_end_train}] out of bounds for close_train len {len(close_train)}.")
        baseline_train = close_train[original_offset : baseline_slice_end_train]

        if original_offset >= len(close_val) or baseline_slice_end_val > len(close_val): raise ValueError(f"Baseline val indices [{original_offset}:{baseline_slice_end_val}] out of bounds for close_val len {len(close_val)}.")
        baseline_val = close_val[original_offset : baseline_slice_end_val]

        if original_offset >= len(close_test) or baseline_slice_end_test > len(close_test): raise ValueError(f"Baseline test indices [{original_offset}:{baseline_slice_end_test}] out of bounds for close_test len {len(close_test)}.")
        baseline_test = close_test[original_offset : baseline_slice_end_test]

        # Final length check for baseline
        if len(baseline_train) != num_samples_train: raise ValueError(f"Final Baseline train length mismatch: Expected {num_samples_train}, Got {len(baseline_train)}")
        if len(baseline_val) != num_samples_val: raise ValueError(f"Final Baseline val length mismatch: Expected {num_samples_val}, Got {len(baseline_val)}")
        if len(baseline_test) != num_samples_test: raise ValueError(f"Final Baseline test length mismatch: Expected {num_samples_test}, Got {len(baseline_test)}")
        print(f"Baseline shapes (Original Logic, Adjusted Length): Train: {baseline_train.shape}, Val: {baseline_val.shape}, Test: {baseline_test.shape}")


        # --- Shift Targets for Each Horizon (Adapting Original Shift) ---
        y_train_final_list = []
        y_val_final_list = []
        y_test_final_list = []
        print(f"Processing targets for horizons: {predicted_horizons} (Use Returns={use_returns})...")

        for h in predicted_horizons:
            # Original shift logic: y_sw = target_sliced[h:]
            # We need to slice target_sliced to match num_samples AFTER the shift
            target_train_shifted = target_train_sliced[h:]
            target_val_shifted = target_val_sliced[h:]
            target_test_shifted = target_test_sliced[h:]

            # Now truncate/slice this shifted array to match the final num_samples
            if len(target_train_shifted) < num_samples_train: raise ValueError(f"Not enough shifted target data for H={h} (Train). Needed {num_samples_train}, got {len(target_train_shifted)}")
            target_train_h = target_train_shifted[:num_samples_train]

            if len(target_val_shifted) < num_samples_val: raise ValueError(f"Not enough shifted target data for H={h} (Val). Needed {num_samples_val}, got {len(target_val_shifted)}")
            target_val_h = target_val_shifted[:num_samples_val]

            if len(target_test_shifted) < num_samples_test: raise ValueError(f"Not enough shifted target data for H={h} (Test). Needed {num_samples_test}, got {len(target_test_shifted)}")
            target_test_h = target_test_shifted[:num_samples_test]

            # Apply returns adjustment
            if use_returns:
                target_train_h = target_train_h - baseline_train
                target_val_h = target_val_h - baseline_val
                target_test_h = target_test_h - baseline_test

            y_train_final_list.append(target_train_h.astype(np.float32))
            y_val_final_list.append(target_val_h.astype(np.float32))
            y_test_final_list.append(target_test_h.astype(np.float32))

        print("Target processing complete (Original Logic Structure).")

        # Prepare Test Close Prices (Original Logic)
        # Needs to align with baseline_test length
        test_close_prices = baseline_test # Baseline is already the aligned close prices

        # --- 8. Final Date Consistency Check ---
        print("\n--- 8. Final Date Consistency Checks ---")
        # Verify the dates associated with the final X samples
        verify_date_consistency([list(x_dates_train) if x_dates_train is not None else None, list(y_dates_train) if y_dates_train is not None else None], "Train X/Y Dates")
        verify_date_consistency([list(x_dates_val) if x_dates_val is not None else None, list(y_dates_val) if y_dates_val is not None else None], "Val X/Y Dates")
        verify_date_consistency([list(x_dates_test) if x_dates_test is not None else None, list(y_dates_test) if y_dates_test is not None else None], "Test X/Y Dates")


        # --- 9. Prepare Return Dictionary (Matching Original Structure) ---
        print("\n--- 9. Preparing Final Output (Original Structure) ---")
        ret = {}
        # Combined X features under "x_train", "x_val", "x_test"
        ret["x_train"] = X_train_combined
        ret["x_val"] = X_val_combined
        ret["x_test"] = X_test_combined

        # Extract STL channels if they exist
        stl_keys = ['stl_trend', 'stl_seasonal', 'stl_resid']
        channel_map = {name: i for i, name in enumerate(feature_names)}
        for key in stl_keys:
            if key in channel_map:
                idx = channel_map[key]
                ret[f"x_train_{key.split('_')[1]}"] = X_train_combined[:, :, idx:idx+1] # Keep channel dim
                ret[f"x_val_{key.split('_')[1]}"] = X_val_combined[:, :, idx:idx+1]
                ret[f"x_test_{key.split('_')[1]}"] = X_test_combined[:, :, idx:idx+1]
            # else: # Omit key if STL feature wasn't generated/included
            #     pass

        # Y data (List of arrays for multi-horizon)
        ret["y_train"] = y_train_final_list
        ret["y_val"] = y_val_final_list
        ret["y_test"] = y_test_final_list
        # Omit y_*_array keys as they don't fit multi-horizon

        # Dates (Aligned with final X/Y samples)
        ret["x_train_dates"] = x_dates_train
        ret["y_train_dates"] = y_dates_train
        ret["x_val_dates"] = x_dates_val
        ret["y_val_dates"] = y_dates_val
        ret["x_test_dates"] = x_dates_test
        ret["y_test_dates"] = y_dates_test

        # Baseline data and dates
        ret["baseline_train"] = baseline_train
        ret["baseline_val"] = baseline_val
        ret["baseline_test"] = baseline_test
        ret["baseline_train_dates"] = y_dates_train # Use Y dates (same as X dates)
        ret["baseline_val_dates"] = y_dates_val
        ret["baseline_test_dates"] = y_dates_test

        # Test Close Prices
        ret["test_close_prices"] = test_close_prices

        print(f"Final returned keys: {list(ret.keys())}")
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
        # Call set_params again AFTER merge to resolve defaults
        self.set_params(**run_config)
        # Run with the fully resolved self.params
        return self.process_data(self.params)
    # --- End of original run_preprocessing ---

# --- NO if __name__ == '__main__': block ---