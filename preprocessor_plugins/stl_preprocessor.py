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


class PreprocessorPlugin:
    # Default plugin parameters - Restored working version + Original Y defaults
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
        "target_column": "TARGET", # From original Y logic
        # --- Windowing & Horizons ---
        "window_size": 48, # Original Default
        # "time_horizon": 6, # Replaced by predicted_horizons
        "predicted_horizons": [1, 6, 12, 24], # Multi-horizon support
        # --- Feature Engineering Flags ---
        "use_stl": False,
        "use_wavelets": True,
        "use_multi_tapper": False,
        "use_returns": True, # Original flag name for Y calculation
        "normalize_features": True,
    # --- Decomposition inclusion flags ---
    "use_predicted_decompositions": True,
    "use_real_decompositions": True,
        # --- STL Parameters ---
        "stl_period": 24,
        "stl_window": None, # Default, resolved later if None
        "stl_trend": None,  # Default, resolved later if None
        "stl_plot_file": "stl_plot.png",
        # --- Wavelet Parameters ---
        "wavelet_name": 'db4',
        "wavelet_levels": 2, # Based on previous error
        "wavelet_mode": 'symmetric',
        "wavelet_plot_file": "wavelet_features.png",
        # --- Multitaper Parameters ---
        "mtm_window_len": 168,
        "mtm_step": 1,
        "mtm_time_bandwidth": 5.0,
        "mtm_num_tapers": None,
        "mtm_freq_bands": [(0, 0.01), (0.01, 0.06), (0.06, 0.2), (0.2, 0.5)],
        "tapper_plot_file": "mtm_features.png",
        "tapper_plot_points": 480, # Plot points
        # "pos_encoding_dim": 16 # Keep commented
    }
    plugin_debug_vars = [ # Updated list
        "window_size", "predicted_horizons", "use_returns", "normalize_features",
        "use_stl", "stl_period", "stl_window", "stl_trend", "stl_plot_file",
        "use_wavelets", "wavelet_name", "wavelet_levels", "wavelet_mode", "wavelet_plot_file",
        "use_multi_tapper", "mtm_window_len", "mtm_step", "mtm_time_bandwidth", "mtm_num_tapers", "mtm_freq_bands", "tapper_plot_file", "tapper_plot_points"
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.scalers = {}

    def set_params(self, **kwargs):
        """Update plugin parameters with global configuration."""
        # (Logic from previous working version)
        for key, value in kwargs.items(): self.params[key] = value
        config = self.params
        if config.get("stl_period") is not None and config.get("stl_period") > 1:
            if config.get("stl_window") is None: config["stl_window"] = 2 * config["stl_period"] + 1
            if config.get("stl_trend") is None:
                current_stl_window = config.get("stl_window")
                if current_stl_window is not None and current_stl_window > 3:
                     try: trend_calc = int(1.5 * config["stl_period"] / (1 - 1.5 / current_stl_window)) + 1; config["stl_trend"] = max(3, trend_calc)
                     except ZeroDivisionError: config["stl_trend"] = config["stl_period"] + 1
                else: config["stl_trend"] = config["stl_period"] + 1
            if config.get("stl_trend") is not None and config["stl_trend"] % 2 == 0: config["stl_trend"] += 1

    def get_debug_info(self):
        """Return debug information for the plugin."""
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """Add plugin debug info to the provided dictionary."""
        debug_info.update(self.get_debug_info())

    # --- Helper Methods (Restored working versions) ---
    def _ewma_causal(self, series: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Compute causal EWMA over series (float64 for stability), return float32."""
        if series is None or len(series) == 0:
            return np.array([], dtype=np.float32)
        x = np.asarray(series, dtype=np.float64)
        y = np.empty_like(x)
        y[0] = x[0]
        a = float(alpha)
        for i in range(1, len(x)):
            y[i] = a * x[i] + (1.0 - a) * y[i-1]
        return y.astype(np.float32)

    def _lag_with_pad(self, series: np.ndarray, lag: int) -> np.ndarray:
        """Return series shifted right by lag with left pad as first value (causal, no wrap)."""
        s = np.asarray(series)
        n = len(s)
        if n == 0 or lag <= 0:
            return s.copy()
        out = np.empty_like(s)
        out[:lag] = s[0]
        out[lag:] = s[:-lag]
        return out

    def _load_data(self, file_path, max_rows, headers):
        # (Implementation from previous working version)
        print(f"Loading data: {file_path} (Max rows: {max_rows})...", end="")
        try:
            df = load_csv(file_path, headers=headers, max_rows=max_rows)
            if df is None or df.empty: raise ValueError(f"load_csv None/empty for {file_path}")
            print(f" Done. Shape: {df.shape}")
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"Attempting DatetimeIndex conversion for {file_path}...", end="")
                original_index_name = df.index.name
                try: df.index = pd.to_datetime(df.index); print(" OK (from index).")
                except Exception:
                    try: df.index = pd.to_datetime(df.iloc[:, 0]); print(" OK (from col 0).")
                    except Exception as e_col: print(f" FAILED ({e_col}). Dates unavailable."); df.index = None
                if original_index_name: df.index.name = original_index_name
            required_cols = ["CLOSE"]; target_col_name=self.params.get("target_column","TARGET")
            if 'y_' in os.path.basename(file_path).lower(): required_cols.append(target_col_name)
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols: raise ValueError(f"Missing cols in {file_path}: {missing_cols}")
            return df
        except FileNotFoundError: print(f"\nERROR: File not found: {file_path}."); raise
        except Exception as e: print(f"\nERROR loading/processing {file_path}: {e}"); import traceback; traceback.print_exc(); raise

    def _normalize_series(self, series, name, fit=False):
        """Normalizes a time series using StandardScaler."""
        # (Implementation from previous working version)
        if not self.params.get("normalize_features", True): return series.astype(np.float32)
        series = series.astype(np.float32)
        if np.any(np.isnan(series)) or np.any(np.isinf(series)):
            print(f"WARN: NaNs/Infs in '{name}' pre-norm. Filling...", end="")
            series = pd.Series(series).fillna(method='ffill').fillna(method='bfill').values
            if np.any(np.isnan(series)) or np.any(np.isinf(series)):
                 print(f" FAILED. Filling with 0.", end="")
                 series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
            print(" OK.")
        data_reshaped = series.reshape(-1, 1)
        if fit:
            scaler = StandardScaler()
            if np.std(data_reshaped) < 1e-9:
                 print(f"WARN: '{name}' constant. Dummy scaler.")
                 class DummyScaler:
                     def fit(self,X):pass; transform=lambda self,X:X.astype(np.float32); inverse_transform=lambda self,X:X.astype(np.float32)
                 scaler = DummyScaler()
            else: scaler.fit(data_reshaped)
            self.scalers[name] = scaler
        else:
            if name not in self.scalers: raise RuntimeError(f"Scaler '{name}' not fitted.")
            scaler = self.scalers[name]
        normalized_data = scaler.transform(data_reshaped)
        return normalized_data.flatten()

    # --- _rolling_stl (Restored working version - NOT original user version) ---
    # Using the version compatible with conditional logic and explicit trend smoother
    def _rolling_stl(self, series, stl_window, period, trend_smoother):
        # Uses the robust version compatible with conditional logic
        print(f"Performing rolling STL: Win={stl_window}, Period={period}, Trend={trend_smoother}...", end="")
        n = len(series)
        num_points = n - stl_window + 1
        if num_points <= 0: raise ValueError(f"stl_window ({stl_window}) > series length ({n}).")
        trend=np.zeros(num_points); seasonal=np.zeros(num_points); resid=np.zeros(num_points)
        if trend_smoother is not None: # Validate trend smoother parameter
             if not isinstance(trend_smoother,int) or trend_smoother<=0: trend_smoother=None
             elif trend_smoother % 2 == 0: trend_smoother += 1
        for i in tqdm(range(stl_window, n + 1), desc="STL", unit="w", disable=None, leave=False):
            window = series[i - stl_window: i]
            current_trend = trend_smoother
            if current_trend is not None and current_trend >= len(window):
                 current_trend = len(window) - 1 if len(window) > 1 else None
                 if current_trend is not None and current_trend % 2 == 0: current_trend = max(1, current_trend -1 )
            try:
                 stl = STL(window, period=period, trend=current_trend, robust=True); result = stl.fit()
                 trend[i-stl_window]=result.trend[-1]; seasonal[i-stl_window]=result.seasonal[-1]; resid[i-stl_window]=result.resid[-1]
            except Exception as e: trend[i-stl_window]=np.nan; seasonal[i-stl_window]=np.nan; resid[i-stl_window]=np.nan # Keep NaN fill on error
        trend = pd.Series(trend).fillna(method='ffill').fillna(method='bfill').values
        seasonal = pd.Series(seasonal).fillna(method='ffill').fillna(method='bfill').values
        resid = pd.Series(resid).fillna(method='ffill').fillna(method='bfill').values
        print(" Done.")
        return trend, seasonal, resid
    # --- End of _rolling_stl ---

    # --- _plot_decomposition (Restored working version) ---
    # Uses the robust version compatible with conditional logic
    def _plot_decomposition(self, series, trend, seasonal, resid, file_path):
        print(f"Plotting STL decomposition to {file_path}...")
        plot_points = 480; n_series = len(series); n_comp = min(len(trend), len(seasonal), len(resid))
        if n_comp==0: print("WARN: Zero length STL components."); return
        points_to_plot = min(plot_points, n_series, n_comp)
        if points_to_plot <= 0: print("WARN: No points to plot for STL."); return
        series_plot=series[-points_to_plot:]; trend_plot=trend[-points_to_plot:]; seasonal_plot=seasonal[-points_to_plot:]; resid_plot=resid[-points_to_plot:]
        final_len=min(len(series_plot), len(trend_plot)); series_plot=series_plot[:final_len]; trend_plot=trend_plot[:final_len]; seasonal_plot=seasonal_plot[:final_len]; resid_plot=resid_plot[:final_len]
        if final_len <= 0: print("WARN: No data points left after STL plot slicing."); return
        try:
            plt.figure(figsize=(12, 9)); plt.subplot(411); plt.plot(series_plot); plt.title("Original Series (Recent)"); plt.grid(True, alpha=0.5)
            plt.subplot(412); plt.plot(trend_plot, color="orange"); plt.title("Trend"); plt.grid(True, alpha=0.5)
            plt.subplot(413); plt.plot(seasonal_plot, color="green"); plt.title("Seasonal"); plt.grid(True, alpha=0.5)
            plt.subplot(414); plt.plot(resid_plot, color="red"); plt.title("Residual"); plt.grid(True, alpha=0.5)
            plt.tight_layout(); plt.savefig(file_path, dpi=300); plt.close(); print(f"STL plot saved.")
        except Exception as e: print(f"WARN: Failed plot STL: {e}"); plt.close()
    # --- End of _plot_decomposition ---

    # --- Updated _compute_wavelet_features with TypeError Fix Attempt ---
    # --- Inside PreprocessorPlugin Class ---

    # --- Updated _compute_wavelet_features: Trying trim_approx=False ---
    def _compute_wavelet_features(self, series):
        """
        Computes Wavelet features using MODWT (pywt.swt).
        Includes causality correction (forward shift/first value padding). So no future data leaks.
        """
        # --- Ensure necessary import for Wavelet object ---
        # Add this check in case Wavelet wasn't imported successfully earlier
        global Wavelet
        if 'Wavelet' not in globals() or Wavelet is None:
            try:
                from pywt import Wavelet
            except ImportError:
                print("ERROR: Cannot import pywt.Wavelet, required for causality correction.")
                Wavelet = None  # Keep it None if import fails here too


        # --- Original input validation and setup ---
        if pywt is None: print("ERROR: pywt library not installed."); return {}
        name = self.params['wavelet_name']; levels = self.params['wavelet_levels'];
        n_original_check = len(series) if hasattr(series, '__len__') else 'N/A'
        if not isinstance(series, (np.ndarray, pd.Series, list)) or len(series) < 2: print(f"ERROR: Wavelet input not array/list or too short."); return {}
        try:
            series_clean = np.asarray(series, dtype=np.float64) # Use float64
            if np.any(np.isnan(series_clean)) or np.any(np.isinf(series_clean)):
                print("WARN: NaNs/Infs in wavelet input. Filling...", end="")
                series_clean = pd.Series(series_clean).fillna(method='ffill').fillna(method='bfill').values
                if np.any(np.isnan(series_clean)) or np.any(np.isinf(series_clean)): print(" Fill FAILED. Skipping."); return {}
                print(" OK.")
        except Exception as e: print(f"ERROR during wavelet input validation: {e}"); return {}
        if levels is None:
            try: levels = pywt.swt_max_level(len(series_clean)); print(f"Auto wavelet levels: {levels}")
            except Exception as e: print(f"ERROR calculating max wavelet levels: {e}."); return {}
        if levels <= 0: print(f"ERROR: Wavelet levels ({levels}) not positive."); return {}

        print(f"Computing Wavelets (MODWT/SWT): {name}, Levels={levels}...", end="")
        try:
            # --- Call pywt.swt ---
            coeffs = pywt.swt(series_clean,
                              wavelet=name,
                              level=levels,
                              trim_approx=False,
                              norm=True) # Boundary mode, no future data is used in padding

            # --- Extract coefficients ---
            if not isinstance(coeffs, list) or len(coeffs) != levels:
                 print(f" FAILED. Unexpected output structure from swt (expected list of length {levels}). Got type {type(coeffs)}")
                 return {}

            features = {}
            # swt returns list [(cA_n, cD_n), ..., (cA_1, cD_1)]
            for i in range(levels):
                level_index_from_end = levels - 1 - i
                if len(coeffs[level_index_from_end]) == 2:
                    details_coeffs = coeffs[level_index_from_end][1]
                    features[f'detail_L{i+1}'] = details_coeffs
                else: print(f"WARN: Unexpected structure in swt output at index {level_index_from_end}"); return {}

            if len(coeffs[0]) == 2:
                approx_coeffs = coeffs[0][0]
                features[f'approx_L{levels}'] = approx_coeffs
            else: print(f"WARN: Could not extract final approximation coeffs."); return {}

            # --- Validate extracted features ---
            n_original_len = len(series_clean)
            valid_features = {}
            for k, v in features.items():
                 if v is not None and hasattr(v, '__len__') and len(v) == n_original_len:
                      valid_features[k] = v
                 else:
                      print(f"WARN: Wavelet '{k}' has unexpected length/type (len={len(v) if hasattr(v,'__len__') else 'N/A'}, type={type(v)}). Discarding.")

            # --- Check if any valid features remain ---
            if not valid_features:
                print(f" FAILED. No valid features extracted after length check.")
                return {}

            # --- [START] CAUSALITY SHIFT CORRECTION (Forward Shift / First Value Padding Method) ---
            try:
                # Calculate the necessary shift based on the filter length
                if Wavelet is None: # Check if import failed
                    raise ImportError("pywt.Wavelet not available for causality check.")

                wavelet = Wavelet(name)
                filter_len = wavelet.dec_len
                # Shift amount based on centered filter lookahead. max(0, (L // 2) - 1)
                shift_amount = max(0, (filter_len // 2) - 1)

                if shift_amount > 0:
                    print(f" Applying causality shift (shifting data forward by {shift_amount}, padding start with first value)...", end="")
                    shifted_features = {}
                    # Get length from one of the valid features before shift
                    try:
                        # Use next(iter(...)) to get first value without knowing the key
                        original_length = len(next(iter(valid_features.values())))
                        # Check if series is long enough to perform the shift meaningfully
                        if original_length <= shift_amount:
                             print(f" FAILED. Shift amount ({shift_amount}) >= series length ({original_length}). Cannot create causal features.")
                             return {} # Return empty if shift is too large
                    except StopIteration:
                        print(" FAILED. Cannot determine length from empty valid_features.")
                        return {} # Return empty if valid_features was somehow empty here

                    for k_feat, v in valid_features.items():
                        if len(v) == original_length:
                            # Get the first value that will be part of the shifted data (v[0] = C[0])
                            # This is the first value calculated by SWT before shifting.
                            first_known_value = v[0]

                            # Create new array of the same size, initialized with the FIRST KNOWN VALUE
                            shifted_v = np.full(original_length, first_known_value, dtype=v.dtype) # PAD WITH FIRST VALUE

                            # Copy the relevant part of the original data, shifted forward
                            # Target: F[t] = C[t-k]  => Implement as: F[k:] = C[:N-k]
                            # This overwrites the padding from index shift_amount onwards
                            shifted_v[shift_amount:] = v[:-shift_amount]
                            shifted_features[k_feat] = shifted_v
                        else:
                             # Handle defensively if length mismatch somehow occurred
                             print(f" WARN: Inconsistent length for feature '{k_feat}' before shift. Skipping shift for this feature.")
                             shifted_features[k_feat] = v # Keep original if length mismatch

                    valid_features = shifted_features # Replace with shifted features
                    print(f" Done. Features shifted, start padded.", end="")
                else:
                    # Only print if shift wasn't needed (e.g., Haar)
                    print(f" No causality shift needed (filter length {filter_len}).", end="")

            except Exception as e_shift:
                # Catch errors during shift calculation/application specifically
                print(f" FAILED applying causality shift. Error: {e_shift}. Returning unshifted features (potential leakage).")
            # --- [END] CAUSALITY SHIFT CORRECTION ---

            # Final print message for the wavelet computation step before returning
            print(f" Done ({len(valid_features)} channels).")
            return valid_features # Return the shifted features (NO NANS ADDED)

        # --- Exception handling for the main pywt.swt call and feature extraction ---
        except TypeError as e:
             print(f" FAILED. TypeError: {e}")
             print(f"      Occurred during pywt.swt call (Input Len={len(series_clean)}, Levels={levels}, Wavelet='{name}', trim_approx=False, norm=True).")
             return {}
        except Exception as e:
             # Catch errors from the main wavelet computation part before shift correction
             print(f" FAILED. Error during Wavelet computation: {e}"); import traceback; traceback.print_exc(); return {}# --- End of _compute_wavelet_features ---

    # --- process_data Method ---
    # (Keep the rest of the process_data method and other helpers exactly as in the previous response)
   

    def _plot_wavelets(self, original_series, wavelet_features, file_path):
        """Plots original series and computed Wavelet features."""
        # (Implementation from previous working step - plots last N points)
        print(f"Plotting Wavelet features to {file_path}...")
        plot_points = 480; num_features = len(wavelet_features)
        if num_features == 0: print("WARN: No wavelet features to plot."); return
        start_idx=max(0,len(original_series)-plot_points); original_plot=original_series[start_idx:]
        actual_plot_points = len(original_plot)
        if actual_plot_points <= 0: print("WARN: No original points for Wavelet plot."); return
        num_plots = num_features + 1; plt.figure(figsize=(12, 2 * num_plots))
        plt.subplot(num_plots, 1, 1); plt.plot(original_plot); plt.title(f"Original Series (Recent {actual_plot_points} points)"); plt.grid(True, alpha=0.5)
        plot_index = 2
        for name, feature_series in wavelet_features.items():
             feat_start_idx=max(0,len(feature_series)-actual_plot_points); feature_plot=feature_series[feat_start_idx:]
             if len(feature_plot)!=actual_plot_points: print(f"WARN: Skip plot Wavelet '{name}', length mismatch."); continue
             plt.subplot(num_plots,1,plot_index); plt.plot(feature_plot,label=name); plt.title(f"Wavelet: {name}"); plt.grid(True, alpha=0.5); plot_index += 1
        if plot_index > 2:
             try: plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.suptitle("Wavelet Features", fontsize=14); plt.savefig(file_path, dpi=300); plt.close(); print(f"Wavelet plot saved.")
             except Exception as e: print(f"WARN: Failed save Wavelet plot: {e}"); plt.close()
        else: print("WARN: No wavelet features plotted."); plt.close()


    def _compute_mtm_features(self, series):
        """Computes Rolling Multitaper Method spectral power in bands. It uses a window of past data to calculate each value, so no future data leaks"""
        # (Implementation from previous working step)
        if dpss is None: print("ERROR: scipy.signal.windows unavailable for MTM."); return {}
        window_len=self.params['mtm_window_len']; step=self.params['mtm_step']; nw=self.params['mtm_time_bandwidth']
        num_tapers=self.params['mtm_num_tapers']; freq_bands=self.params['mtm_freq_bands']; n_original=len(series)
        if num_tapers is None: num_tapers = max(1, int(2 * nw - 1))
        if n_original < window_len: print(f"ERROR: Series length ({n_original}) < MTM window ({window_len})."); return {}
        print(f"Computing MTM: Win={window_len}, Step={step}, NW={nw}, K={num_tapers}...", end="")
        try: tapers = dpss(window_len, nw, num_tapers)
        except ValueError as e: print(f" FAILED generating DPSS tapers: {e}."); return {}
        num_windows=(n_original - window_len) // step + 1
        mtm_features={f"band_{i}": np.full(num_windows, np.nan) for i in range(len(freq_bands))}
        fft_freqs=np.fft.rfftfreq(window_len); freq_masks=[(fft_freqs >= f_low) & (fft_freqs < f_high) for f_low, f_high in freq_bands]
        for i in range(num_windows): # Removed tqdm here for cleaner logs
            start=i*step; end=start+window_len; window_data=series[start:end]
            if np.any(np.isnan(window_data)): continue
            spectra = np.zeros((num_tapers, len(fft_freqs)))
            for k in range(num_tapers): spectra[k, :] = np.abs(np.fft.rfft(window_data * tapers[k]))**2
            avg_spectrum = np.mean(spectra, axis=0)
            for band_idx, mask in enumerate(freq_masks):
                 if np.any(mask): mtm_features[f"band_{band_idx}"][i] = np.mean(avg_spectrum[mask])
                 else: mtm_features[f"band_{band_idx}"][i] = 0.0
        for name in mtm_features: mtm_features[name]=pd.Series(mtm_features[name]).fillna(method='ffill').fillna(method='bfill').values
        print(f" Done ({len(mtm_features)} channels).")
        return mtm_features

    # --- Predicted decomposition helpers (causal, per-horizon) ---
    def _predict_components_per_horizon(self, comp_dict: dict, horizons: list, name_prefix: str) -> dict:
        """Given a dict of component arrays comp_dict, build causal forecasts for each horizon.
        Uses a simple, robust causal baseline: EWMA smoother + lag by h.
        Returns a flat dict: {f"{name_prefix}_{comp}_h{h}": series}
        """
        if not comp_dict:
            return {}
        predicted = {}
        # Use a moderate smoothing to reduce noise while keeping causality
        alpha = 0.3
        for comp_name, comp_series in comp_dict.items():
            if comp_series is None or len(comp_series) == 0:
                continue
            # Smooth causally, then lag by horizon as a naive forecast F_t(h) = Smooth_{t}
            smooth = self._ewma_causal(np.asarray(comp_series, dtype=np.float32), alpha=alpha)
            for h in horizons:
                if h <= 0:
                    continue
                # Predicted value for timestamp t corresponds to actual comp at t+h; to avoid leakage
                # we provide the smoothed signal lagged by h, so input at t uses only <= t history
                lagged = self._lag_with_pad(smooth, h)
                predicted[f"{name_prefix}_{comp_name}_h{h}"] = lagged.astype(np.float32)
        return predicted

    def _plot_mtm(self, mtm_features, file_path, points_to_plot=500):
        """Plots computed Multitaper features (power bands)."""
        # (Implementation from previous working step - uses points_to_plot)
        print(f"Plotting MTM features to {file_path}...")
        num_features = len(mtm_features); points_to_plot = self.params.get("tapper_plot_points", 480) # Get from params
        if num_features == 0: print("WARN: No MTM features to plot."); return
        min_len = 0; valid_keys = [k for k, v in mtm_features.items() if v is not None and len(v) > 0]
        if valid_keys: min_len = min(len(mtm_features[k]) for k in valid_keys)
        plot_points = min(points_to_plot, min_len)
        if plot_points <= 0: print("WARN: No MTM data points to plot."); return
        plt.figure(figsize=(12, 2 * num_features))
        plot_index = 1
        for name in valid_keys:
             feature_series = mtm_features[name]; start_idx = max(0, len(feature_series) - plot_points); feature_plot = feature_series[start_idx:]
             plt.subplot(num_features, 1, plot_index); plt.plot(feature_plot, label=name); plt.title(f"MTM: {name} (Recent {plot_points} points)"); plt.grid(True, alpha=0.5); plot_index += 1
        if plot_index > 1:
             try: plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.suptitle("Multitaper Spectral Power Features", fontsize=14); plt.savefig(file_path, dpi=300); plt.close(); print(f"MTM plot saved.")
             except Exception as e: print(f"WARN: Failed to save MTM plot: {e}"); plt.close()
        else: print("WARN: No MTM features plotted."); plt.close()


    # --- Original create_sliding_windows (kept exactly as provided by user) ---
    # This is used by the FEATURE generation loop now, requires single time_horizon
    def create_sliding_windows(self, data, window_size, time_horizon, date_times=None):
        """
        Creates sliding windows for a univariate series.
        Original user version - calculates targets internally but they are ignored.
        Requires a single `time_horizon` argument (use max_horizon here).
        """
        print(f"Creating sliding windows (Orig Method - Size={window_size}, Horizon={time_horizon})...", end="")
        windows = []; targets = []; date_windows = [] # Initialize targets list
        n = len(data)
        num_possible_windows = n - window_size - time_horizon + 1 # Use passed horizon
        if num_possible_windows <= 0:
             print(f" WARN: Data short ({n}) for Win={window_size}+Horizon={time_horizon}. No windows.")
             return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32), np.array(date_windows, dtype=object)
        for i in range(num_possible_windows):
            window = data[i: i + window_size]
            target = data[i + window_size + time_horizon - 1] # Original target calc (ignored)
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
        # (Implementation from previous working step - kept the same)
        aligned_features = {}
        min_len = base_length; feature_lengths = {'base': base_length}
        valid_keys = [k for k, v in feature_dict.items() if v is not None]
        if not valid_keys: return {}, 0
        for name in valid_keys: feature_lengths[name] = len(feature_dict[name]); min_len = min(min_len, feature_lengths[name])
        needs_alignment = any(l != min_len for l in feature_lengths.values() if l > 0)
        if needs_alignment:
             # print(f"Aligning features to length: {min_len}. Orig: {feature_lengths}") # Verbose
             for name in valid_keys:
                 series = feature_dict[name]; current_len = len(series)
                 if current_len > min_len: aligned_features[name] = series[current_len - min_len:]
                 elif current_len == min_len: aligned_features[name] = series
                 else: print(f"WARN: Feature '{name}' len({current_len})<target({min_len})."); aligned_features[name] = None
        else: aligned_features = {k: feature_dict[k] for k in valid_keys}
        final_lengths = {name: len(s) for name, s in aligned_features.items() if s is not None}
        unique_lengths = set(final_lengths.values())
        if len(unique_lengths) > 1: raise RuntimeError(f"Alignment FAILED! Inconsistent lengths: {final_lengths}")
        # print(f"Aligned feature length: {min_len}") # Verbose
        return aligned_features, min_len


    # --- process_data Method ---
    def process_data(self, config):
        """
        Processes data using new feature generation and ORIGINAL Y/Baseline logic.
        """
        # 0. Setup & Parameter Resolution
        print("\n" + "="*15 + " Starting Preprocessing " + "="*15)
        self.set_params(**config) # Resolve defaults based on final config
        config = self.params # Use the fully resolved params
        self.scalers = {}

        # Get key parameters used in multiple places
        window_size = config['window_size']
        predicted_horizons = config['predicted_horizons']
        if not isinstance(predicted_horizons, list) or not predicted_horizons: raise ValueError("'predicted_horizons' must be a non-empty list.")
        max_horizon = max(predicted_horizons) # Use max for windowing alignment
        # Get stl_window for original Y calc - MUST be resolved here
        stl_window = config.get('stl_window')
        if stl_window is None: raise ValueError("stl_window parameter must be resolved before processing Y/Baseline.")


        # --- 1. Load Data ---
        print("\n--- 1. Loading Data ---")
        x_train_df=self._load_data(config["x_train_file"], config.get("max_steps_train"), config.get("headers"))
        x_val_df=self._load_data(config["x_validation_file"], config.get("max_steps_val"), config.get("headers"))
        x_test_df=self._load_data(config["x_test_file"], config.get("max_steps_test"), config.get("headers"))
        y_train_df=self._load_data(config["y_train_file"], config.get("max_steps_train"), config.get("headers"))
        y_val_df=self._load_data(config["y_validation_file"], config.get("max_steps_val"), config.get("headers"))
        y_test_df=self._load_data(config["y_test_file"], config.get("max_steps_test"), config.get("headers"))

        # --- 2. Initial Prep: Log Transform, Dates ---
        print("\n--- 2. Initial Data Prep ---")
        try: close_train=x_train_df["CLOSE"].astype(np.float32).values; close_val=x_val_df["CLOSE"].astype(np.float32).values; close_test=x_test_df["CLOSE"].astype(np.float32).values
        except Exception as e: raise ValueError(f"Error converting 'CLOSE': {e}")
        log_train=np.log1p(np.maximum(0, close_train)); log_val=np.log1p(np.maximum(0, close_val)); log_test=np.log1p(np.maximum(0, close_test))
        print(f"Log transform applied. Train shape: {log_train.shape}")
        dates_train=x_train_df.index if isinstance(x_train_df.index, pd.DatetimeIndex) else None
        dates_val=x_val_df.index if isinstance(x_val_df.index, pd.DatetimeIndex) else None
        dates_test=x_test_df.index if isinstance(x_test_df.index, pd.DatetimeIndex) else None

        # --- 3. Feature Generation (Conditional) ---
        print("\n--- 3. Feature Generation ---")
        features_train, features_val, features_test = {}, {}, {}
        # 3.a. Log Returns
        log_ret_train=np.diff(log_train,prepend=log_train[0]); features_train['log_return']=self._normalize_series(log_ret_train,'log_return',fit=True)
        log_ret_val=np.diff(log_val,prepend=log_val[0]); features_val['log_return']=self._normalize_series(log_ret_val,'log_return',fit=False)
        log_ret_test=np.diff(log_test,prepend=log_test[0]); features_test['log_return']=self._normalize_series(log_ret_test,'log_return',fit=False)
        print("Generated: Log Returns (Normalized)")
        # 3.b. STL (using restored working version of _rolling_stl)
        if config.get('use_stl'):
            print("Attempting STL features...")
            stl_period = config['stl_period']; stl_trend = config['stl_trend']
            try:
                trend_train, seasonal_train, resid_train = self._rolling_stl(log_train, stl_window, stl_period, stl_trend)
                trend_val, seasonal_val, resid_val = self._rolling_stl(log_val, stl_window, stl_period, stl_trend)
                trend_test, seasonal_test, resid_test = self._rolling_stl(log_test, stl_window, stl_period, stl_trend)
                if len(trend_train) > 0:
                    if config.get('use_real_decompositions', True):
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
                        stl_plot_file = config.get("stl_plot_file")
                        if stl_plot_file:
                            self._plot_decomposition(log_train[len(log_train)-len(trend_train):], trend_train, seasonal_train, resid_train, stl_plot_file)
                    # Predicted STL components per horizon (causal), if enabled
                    if config.get('use_predicted_decompositions', True):
                        stl_comp_train = { 'stl_trend': trend_train, 'stl_seasonal': seasonal_train, 'stl_resid': resid_train }
                        stl_comp_val   = { 'stl_trend': trend_val,   'stl_seasonal': seasonal_val,   'stl_resid': resid_val }
                        stl_comp_test  = { 'stl_trend': trend_test,  'stl_seasonal': seasonal_test,  'stl_resid': resid_test }
                        stl_pred_train = self._predict_components_per_horizon(stl_comp_train, predicted_horizons, 'pred')
                        stl_pred_val   = self._predict_components_per_horizon(stl_comp_val,   predicted_horizons, 'pred')
                        stl_pred_test  = self._predict_components_per_horizon(stl_comp_test,  predicted_horizons, 'pred')
                        for k,v in stl_pred_train.items(): features_train[k] = self._normalize_series(v, k, fit=True)
                        for k,v in stl_pred_val.items():   features_val[k]   = self._normalize_series(v, k, fit=False)
                        for k,v in stl_pred_test.items():  features_test[k]  = self._normalize_series(v, k, fit=False)
                else:
                    print("WARN: STL output zero length.")
            except Exception as e:
                print(f"ERROR processing STL: {e}. Skipping.")
        else:
            print("Skipped: STL features.")
        # 3.c. Wavelets (using fixed _compute_wavelet_features)
        if config.get('use_wavelets'):
            print("Attempting Wavelet features...")
            try:
                wav_features_train = self._compute_wavelet_features(log_train)
                if wav_features_train:
                    wav_features_val = self._compute_wavelet_features(log_val)
                    wav_features_test = self._compute_wavelet_features(log_test)
                    if config.get('use_real_decompositions', True):
                        for name in wav_features_train.keys():
                            features_train[f'wav_{name}'] = self._normalize_series(wav_features_train[name], f'wav_{name}', fit=True)
                            if name in wav_features_val: features_val[f'wav_{name}'] = self._normalize_series(wav_features_val[name], f'wav_{name}', fit=False)
                            if name in wav_features_test: features_test[f'wav_{name}'] = self._normalize_series(wav_features_test[name], f'wav_{name}', fit=False)
                        print(f"Generated: {len(wav_features_train)} Wavelet features (Normalized).")
                        wavelet_plot_file = config.get("wavelet_plot_file")
                        if wavelet_plot_file:
                            self._plot_wavelets(log_train, wav_features_train, wavelet_plot_file)
                    # Predicted Wavelet components (causal), if enabled
                    if config.get('use_predicted_decompositions', True):
                        wav_pred_train = self._predict_components_per_horizon(wav_features_train, predicted_horizons, 'pred_wav')
                        wav_pred_val   = self._predict_components_per_horizon(wav_features_val,   predicted_horizons, 'pred_wav') if wav_features_val else {}
                        wav_pred_test  = self._predict_components_per_horizon(wav_features_test,  predicted_horizons, 'pred_wav') if wav_features_test else {}
                        for k,v in wav_pred_train.items(): features_train[k] = self._normalize_series(v, k, fit=True)
                        for k,v in wav_pred_val.items():   features_val[k]   = self._normalize_series(v, k, fit=False)
                        for k,v in wav_pred_test.items():  features_test[k]  = self._normalize_series(v, k, fit=False)
                else:
                    print("WARN: Wavelet computation for train failed/returned no features.")
            except Exception as e:
                print(f"ERROR processing Wavelets: {e}. Skipping.")
        else:
            print("Skipped: Wavelet features.")
        # 3.d. Multitaper
        if config.get('use_multi_tapper'):
            print("Attempting MTM features...")
            try:
                mtm_features_train = self._compute_mtm_features(log_train)
                if mtm_features_train:
                    mtm_features_val = self._compute_mtm_features(log_val)
                    mtm_features_test = self._compute_mtm_features(log_test)
                    if config.get('use_real_decompositions', True):
                        for name in mtm_features_train.keys():
                            features_train[f'mtm_{name}'] = self._normalize_series(mtm_features_train[name], f'mtm_{name}', fit=True)
                            if name in mtm_features_val:  features_val[f'mtm_{name}']  = self._normalize_series(mtm_features_val[name],  f'mtm_{name}', fit=False)
                            if name in mtm_features_test: features_test[f'mtm_{name}'] = self._normalize_series(mtm_features_test[name], f'mtm_{name}', fit=False)
                        print(f"Generated: {len(mtm_features_train)} MTM features (Normalized).")
                        tapper_plot_file = config.get("tapper_plot_file")
                        if tapper_plot_file:
                            self._plot_mtm(mtm_features_train, tapper_plot_file, config.get("tapper_plot_points"))
                    # Predicted MTM components (causal), if enabled
                    if config.get('use_predicted_decompositions', True):
                        mtm_pred_train = self._predict_components_per_horizon(mtm_features_train, predicted_horizons, 'pred_mtm')
                        mtm_pred_val   = self._predict_components_per_horizon(mtm_features_val,   predicted_horizons, 'pred_mtm') if mtm_features_val else {}
                        mtm_pred_test  = self._predict_components_per_horizon(mtm_features_test,  predicted_horizons, 'pred_mtm') if mtm_features_test else {}
                        for k,v in mtm_pred_train.items(): features_train[k] = self._normalize_series(v, k, fit=True)
                        for k,v in mtm_pred_val.items():   features_val[k]   = self._normalize_series(v, k, fit=False)
                        for k,v in mtm_pred_test.items():  features_test[k]  = self._normalize_series(v, k, fit=False)
                else:
                    print("WARN: MTM computation for train failed/returned no features.")
            except Exception as e:
                print(f"ERROR processing MTM: {e}. Skipping.")
        else:
            print("Skipped: MTM features.")

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


        # --- NEW 4.b: Prepare and Align Original X Columns ---
        print("\n--- 4.b Preparing and Aligning Original X Columns ---")
        # Identify original columns (present in train_df, excluding 'CLOSE')
        if 'CLOSE' in x_train_df.columns:
            original_x_cols = [col for col in x_train_df.columns if col != 'CLOSE']
        else:
            # Handle case where CLOSE might not be present (though unlikely based on previous code)
            original_x_cols = list(x_train_df.columns)
            print("WARN: 'CLOSE' column not found in x_train_df. Including all columns as 'original'.")

        if not original_x_cols:
            print("WARN: No original columns found besides 'CLOSE' (or input is empty).")
            aligned_original_train_dict, aligned_original_val_dict, aligned_original_test_dict = {}, {}, {}
        else:
            print(f"Identified original columns to include: {original_x_cols}")
            # Ensure these columns exist in val and test as well (basic check)
            missing_val_cols = [c for c in original_x_cols if c not in x_val_df.columns]
            missing_test_cols = [c for c in original_x_cols if c not in x_test_df.columns]
            if missing_val_cols: print(f"WARN: Original columns missing in x_val_df: {missing_val_cols}")
            if missing_test_cols: print(f"WARN: Original columns missing in x_test_df: {missing_test_cols}")

            # Select existing original columns for each dataset
            original_x_cols_val = [c for c in original_x_cols if c in x_val_df.columns]
            original_x_cols_test = [c for c in original_x_cols if c in x_test_df.columns]

            # Align original columns to match the length of aligned generated features
            # Take the last 'aligned_len_xxx' rows based on index alignment
            aligned_original_train_df = x_train_df[original_x_cols].iloc[-aligned_len_train:]
            aligned_original_val_df = x_val_df[original_x_cols_val].iloc[-aligned_len_val:]
            aligned_original_test_df = x_test_df[original_x_cols_test].iloc[-aligned_len_test:]

            # Convert to dictionary of numpy arrays (float32 for consistency with generated features)
            # Use .get() with default None in case a column was missing after the warning
            aligned_original_train_dict = {col: aligned_original_train_df[col].values.astype(np.float32) for col in original_x_cols}
            aligned_original_val_dict = {col: df_col.values.astype(np.float32) if (df_col := aligned_original_val_df.get(col)) is not None else None for col in original_x_cols}
            aligned_original_test_dict = {col: df_col.values.astype(np.float32) if (df_col := aligned_original_test_df.get(col)) is not None else None for col in original_x_cols}

            # Verify lengths (optional but good practice)
            mismatched_train = [k for k, v in aligned_original_train_dict.items() if len(v) != aligned_len_train]
            mismatched_val = [k for k, v in aligned_original_val_dict.items() if v is not None and len(v) != aligned_len_val]
            mismatched_test = [k for k, v in aligned_original_test_dict.items() if v is not None and len(v) != aligned_len_test]
            if mismatched_train: print(f"WARN: Length mismatch after aligning original train columns: {mismatched_train}")
            if mismatched_val: print(f"WARN: Length mismatch after aligning original val columns: {mismatched_val}")
            if mismatched_test: print(f"WARN: Length mismatch after aligning original test columns: {mismatched_test}")
            print(f"Aligned original columns. Train={len(aligned_original_train_dict)}, Val={len(aligned_original_val_dict)}, Test={len(aligned_original_test_dict)}")

        # Combine generated features and aligned original columns for windowing
        # Generated features take precedence if names collide (unlikely here)
        all_features_train = {**aligned_original_train_dict, **features_train}
        all_features_val = {**aligned_original_val_dict, **features_val}
        all_features_test = {**aligned_original_test_dict, **features_test}



        # --- 5. Windowing Features & Stacking (MODIFIED to include original columns) ---
        print("\n--- 5. Windowing Features & Channel Stacking ---")
        X_train_channels, X_val_channels, X_test_channels = [], [], []
        feature_names = [] # Will store names of ALL features (generated + original) in final order
        x_dates_train, x_dates_val, x_dates_test = None, None, None
        first_feature_dates_captured = False

        # Define the order: generated features first (if they exist), then original columns alphabetically
        generated_feature_order = ['log_return'] # Always include log_return if generated
        # Real decompositions (only if enabled)
        if config.get('use_real_decompositions', True) and config.get('use_stl'):
            generated_feature_order.extend(['stl_trend', 'stl_seasonal', 'stl_resid'])
        if config.get('use_real_decompositions', True) and config.get('use_wavelets'):
            generated_feature_order.extend(sorted([k for k in features_train if k.startswith('wav_')]))
        if config.get('use_real_decompositions', True) and config.get('use_multi_tapper'):
            generated_feature_order.extend(sorted([k for k in features_train if k.startswith('mtm_')]))
        # Predicted decompositions (always include if present and flag is enabled)
        if config.get('use_predicted_decompositions', True):
            generated_feature_order.extend(sorted([k for k in features_train if k.startswith('pred_')]))

        # Filter generated_feature_order to only include features that were actually created and aligned
        generated_feature_order = [f for f in generated_feature_order if f in all_features_train and all_features_train[f] is not None]

        # Get original columns that were successfully aligned (using keys from the dict created in 4.b)
        original_feature_order = sorted([k for k, v in aligned_original_train_dict.items() if v is not None])

        # Combine the order lists
        windowing_order = generated_feature_order + original_feature_order
        print(f"Final feature order for windowing: {windowing_order}")

        # --- Use an adjusted horizon for windowing ---
        # We must account for the largest lookback among enabled features when creating X windows.
        # Otherwise X windows may reference later timestamps than Y/baseline, causing leakage.
        # Effective feature lookback is the max of STL window and MTM window (if enabled).
        effective_feature_window = max(
            stl_window or 0,
            int(config.get('mtm_window_len', 0)) if config.get('use_multi_tapper') else 0
        )
        # Reduce the number of X windows by (effective_feature_window - 1) so that after applying
        # the original offset and horizon shifts, targets exist for all samples/horizons.
        # num_windows = N - window_size - (max_horizon + effective_feature_window - 1) + 1
        time_horizon_for_windowing = max_horizon + max(0, effective_feature_window - 1)
        print(f"Windowing horizon adjusted for feature lookback. effective_feature_window={effective_feature_window}, time_horizon_for_windowing={time_horizon_for_windowing}")

        for name in windowing_order:
            # Get the correct aligned series for train, val, test from the combined dictionary
            series_train = all_features_train.get(name)
            series_val = all_features_val.get(name)
            series_test = all_features_test.get(name)

            # Ensure the feature exists and is valid for all splits before windowing
            if series_train is not None and series_val is not None and series_test is not None and \
               len(series_train) == aligned_len_train and \
               len(series_val) == aligned_len_val and \
               len(series_test) == aligned_len_test:

                print(f"Windowing feature: {name}...", end="")
                try:
                    # Pass adjusted horizon and the aligned dates
                    win_train, _, dates_win_train = self.create_sliding_windows(series_train, window_size, time_horizon_for_windowing, dates_train_aligned)
                    win_val, _, dates_win_val   = self.create_sliding_windows(series_val, window_size, time_horizon_for_windowing, dates_val_aligned)
                    win_test, _, dates_win_test = self.create_sliding_windows(series_test, window_size, time_horizon_for_windowing, dates_test_aligned)

                    # Check if windowing was successful (produced samples)
                    # Also check if the number of samples is consistent across splits (important!)
                    if win_train.shape[0] > 0 and win_val.shape[0] > 0 and win_test.shape[0] > 0:
                        # If this is the first feature, set the expected number of samples
                        if not first_feature_dates_captured:
                            expected_samples_train = win_train.shape[0]
                            expected_samples_val = win_val.shape[0]
                            expected_samples_test = win_test.shape[0]
                            print(f" Initializing sample counts: Train={expected_samples_train}, Val={expected_samples_val}, Test={expected_samples_test}", end="")

                        # Check consistency with previously windowed features
                        if win_train.shape[0] == expected_samples_train and \
                           win_val.shape[0] == expected_samples_val and \
                           win_test.shape[0] == expected_samples_test:

                            X_train_channels.append(win_train)
                            X_val_channels.append(win_val)
                            X_test_channels.append(win_test)
                            feature_names.append(name) # Add name to the list of included features
                            print(" Appended.")

                            if not first_feature_dates_captured:
                                x_dates_train, x_dates_val, x_dates_test = dates_win_train, dates_win_val, dates_win_test
                                first_feature_dates_captured = True
                                print(f"Captured dates from '{name}'. Lengths: T={len(x_dates_train)}, V={len(x_dates_val)}, Ts={len(x_dates_test)}")
                        else:
                            print(f" Skipping channel '{name}' due to inconsistent sample count after windowing.")
                            print(f"   Expected T={expected_samples_train}, V={expected_samples_val}, Ts={expected_samples_test}")
                            print(f"   Got      T={win_train.shape[0]}, V={win_val.shape[0]}, Ts={win_test.shape[0]}")

                    else:
                        print(f" Skipping channel '{name}' (windowing produced 0 samples in at least one split).")
                except Exception as e:
                    print(f" FAILED windowing '{name}'. Error: {e}. Skipping.")
            else:
                # This handles cases where original columns might have been missing or had length mismatches earlier
                print(f"WARN: Feature '{name}' skipped. Not valid or consistently aligned across train/val/test before windowing.")

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

        # --- 7. Baseline & Target Processing (REVERTED TO USER'S ORIGINAL LOGIC - ALIGNED) ---
        print("\n--- 7. Baseline and Target Processing (Original Logic - Aligned) ---")
        target_column = config["target_column"]
        use_returns = config.get("use_returns", False) # Original flag name

        # --- Original Offset Calculation (adjusted for maximum feature lookback) ---
        # The baseline/target indices must align with the end of each X window in original series time.
        # Offset = (feature lookback - 1) + (window_size - 1)
        effective_feature_window = max(
            (stl_window or 0),
            int(config.get('mtm_window_len', 0)) if config.get('use_multi_tapper') else 0
        )
        original_offset = effective_feature_window + window_size - 2
        print(f"Calculated original logic offset: {original_offset} (effective_feature_window={effective_feature_window}, window_size={window_size})")

        # --- Load Raw Target Data ---
        if target_column not in y_train_df.columns: raise ValueError(f"Column '{target_column}' not found in Y train.")
        target_train_raw = y_train_df[target_column].astype(np.float32).values
        target_val_raw = y_val_df[target_column].astype(np.float32).values
        target_test_raw = y_test_df[target_column].astype(np.float32).values

        # --- Calculate Baseline using original offset logic BUT ensure length matches num_*_windows ---
        baseline_slice_end_train = original_offset + num_samples_train
        baseline_slice_end_val = original_offset + num_samples_val
        baseline_slice_end_test = original_offset + num_samples_test

        if original_offset < 0 or baseline_slice_end_train > len(close_train): raise ValueError(f"Baseline train indices invalid.")
        baseline_train = close_train[original_offset : baseline_slice_end_train]
        if original_offset < 0 or baseline_slice_end_val > len(close_val): raise ValueError(f"Baseline val indices invalid.")
        baseline_val = close_val[original_offset : baseline_slice_end_val]
        if original_offset < 0 or baseline_slice_end_test > len(close_test): raise ValueError(f"Baseline test indices invalid.")
        baseline_test = close_test[original_offset : baseline_slice_end_test]

        if len(baseline_train) != num_samples_train: raise ValueError(f"Baseline train length mismatch: Expected {num_samples_train}, Got {len(baseline_train)}")
        if len(baseline_val) != num_samples_val: raise ValueError(f"Baseline val length mismatch: Expected {num_samples_val}, Got {len(baseline_val)}")
        if len(baseline_test) != num_samples_test: raise ValueError(f"Baseline test length mismatch: Expected {num_samples_test}, Got {len(baseline_test)}")
        print(f"Baseline shapes (Original Logic): Train={baseline_train.shape}, Val={baseline_val.shape}, Test={baseline_test.shape}")

        # --- Process targets using original slicing and shifting logic, adjusted for multi-horizon ---
        # 1. Apply original initial slice
        target_train = target_train_raw[original_offset:]
        target_val = target_val_raw[original_offset:]
        target_test = target_test_raw[original_offset:]

        # 2. Calculate shifted targets for each horizon and slice to final length
        y_train_final_list = []; y_val_final_list = []; y_test_final_list = []
        print(f"Processing targets for horizons: {predicted_horizons} (Use Returns={use_returns})...")
        for h in predicted_horizons:
            # 2a. Apply original shift logic: target_sliced[h:]
            target_train_shifted = target_train[h:]
            target_val_shifted = target_val[h:]
            target_test_shifted = target_test[h:]

            # 2b. Slice the shifted result to match num_samples
            if len(target_train_shifted) < num_samples_train: raise ValueError(f"Not enough shifted target data for H={h} (Train). Needed {num_samples_train}, got {len(target_train_shifted)}")
            target_train_h = target_train_shifted[:num_samples_train]
            if len(target_val_shifted) < num_samples_val: raise ValueError(f"Not enough shifted target data for H={h} (Val). Needed {num_samples_val}, got {len(target_val_shifted)}")
            target_val_h = target_val_shifted[:num_samples_val]
            if len(target_test_shifted) < num_samples_test: raise ValueError(f"Not enough shifted target data for H={h} (Test). Needed {num_samples_test}, got {len(target_test_shifted)}")
            target_test_h = target_test_shifted[:num_samples_test]

            # 2c. Apply returns adjustment using the ALIGNED baseline
            if use_returns:
                target_train_h = target_train_h - baseline_train
                target_val_h = target_val_h - baseline_val
                target_test_h = target_test_h - baseline_test

            y_train_final_list.append(target_train_h.astype(np.float32))
            y_val_final_list.append(target_val_h.astype(np.float32))
            y_test_final_list.append(target_test_h.astype(np.float32))

        # --- Assign Dates based on X windows ---
        # The dates should correspond to the *end* of the input window (X)
        y_dates_train = x_dates_train
        y_dates_val = x_dates_val
        y_dates_test = x_dates_test
        print("Target processing complete (Original Logic Structure).")

        # --- Test Close Prices (Original Logic - Aligned) ---
        # This should match the baseline_test calculated above
        test_close_prices = baseline_test

        # --- 8. Final Date Consistency Check ---
        print("\n--- 8. Final Date Consistency Checks ---")
        # Dates for X and Y (and Baseline dates derived from Y) should align
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

        # Extract STL channels if they exist and add with original keys
        stl_keys_orig = {'stl_trend': 'trend', 'stl_seasonal': 'seasonal', 'stl_resid': 'noise'}
        channel_map = {name: i for i, name in enumerate(feature_names)}
        for stl_key_new, stl_key_orig_suffix in stl_keys_orig.items():
            if stl_key_new in channel_map:
                idx = channel_map[stl_key_new]
                ret[f"x_train_{stl_key_orig_suffix}"] = X_train_combined[:, :, idx:idx+1]
                ret[f"x_val_{stl_key_orig_suffix}"] = X_val_combined[:, :, idx:idx+1]
                ret[f"x_test_{stl_key_orig_suffix}"] = X_test_combined[:, :, idx:idx+1]

        # Y data (List of arrays for multi-horizon) - Use keys from original code
        ret["y_train"] = y_train_final_list # Multi-horizon list
        ret["y_val"] = y_val_final_list
        ret["y_test"] = y_test_final_list
        # Omit y_*_array keys as they don't fit multi-horizon naturally

        # Dates (Aligned with final X/Y samples) - Use keys from original code
        ret["x_train_dates"] = x_dates_train
        ret["y_train_dates"] = y_dates_train
        ret["x_val_dates"] = x_dates_val
        ret["y_val_dates"] = y_dates_val
        ret["x_test_dates"] = x_dates_test
        ret["y_test_dates"] = y_dates_test

        # Baseline data and dates - Use keys from original code
        ret["baseline_train"] = baseline_train
        ret["baseline_val"] = baseline_val
        ret["baseline_test"] = baseline_test
        ret["baseline_train_dates"] = y_dates_train # Use Y dates (same as X dates)
        ret["baseline_val_dates"] = y_dates_val
        ret["baseline_test_dates"] = y_dates_test

        # Test Close Prices - Use key from original code
        ret["test_close_prices"] = test_close_prices

        # Add feature names for reference (New key)
        ret["feature_names"] = feature_names

        print(f"Final returned keys: {list(ret.keys())}")
        del x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df # Cleanup
        del features_train, features_val, features_test

        print("\n" + "="*15 + " Preprocessing Finished " + "="*15)
        return ret

    # --- Original run_preprocessing (kept exactly as provided by user) ---
    # --- MODIFIED to call set_params correctly ---
    def run_preprocessing(self, config):
        """Convenience method to execute data processing."""
        # Merge instance defaults with passed config
        run_config = self.params.copy()
        run_config.update(config)
        # Call set_params again AFTER merge to resolve defaults
        self.set_params(**run_config)
        # Run with the fully resolved self.params
        return self.process_data(self.params)
    # --- End of run_preprocessing ---

# --- NO if __name__ == '__main__': block ---