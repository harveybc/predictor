import numpy as np
import pandas as pd
from .helpers import load_normalization_json, denormalize_all_datasets, load_normalized_csv, exclude_columns_from_datasets
from .sliding_windows import create_sliding_windows, extract_baselines_from_sliding_windows
from .target_calculation import calculate_targets_from_baselines
from .anti_naive_lock import apply_log_returns_to_series


class STLPreprocessorZScore:
    """
    1. Load already normalized CSV data ✅
    2. Denormalize all input datasets using JSON parameters
    3. Create sliding windows from denormalized data
    4. Extract baselines (last elements of each window for target column)
    5. Calculate log return targets with those baselines (train, validation, test)
     6. Create SECOND sliding windows matrix from the ORIGINAL normalized datasets transformed with per-column log-returns
         (applies to all numeric features). Dates preserved; no change to target pipeline.
     7. Keep baselines and targets unchanged (they're already calculated correctly)
    """

    # Plugin-specific parameters they get overwritten if declared in the config
    plugin_params = {
        "window_size": 48,
        "predicted_horizons": [1, 6],
        "target_column": "CLOSE",
        "use_returns": True,
        "anti_naive_lock_enabled": True,
        "feature_preprocessing_strategy": "selective"
    }
    
    plugin_debug_vars = ["window_size", "predicted_horizons", "target_column"]

    # Start of plugin interface methods    
    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value
    
    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}
    
    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())
    # End of plugin interface methods

    def process_data(self, config):
        # Main process orchestration
        try:
            self.set_params(**config)
            config = self.params
            
            predicted_horizons = config['predicted_horizons']
            if not isinstance(predicted_horizons, list) or not predicted_horizons:
                raise ValueError("predicted_horizons must be a non-empty list")
            
            # 1. Load already normalized CSV data
            print("Step 1: Load normalized CSV data")
            normalized_data, dates = load_normalized_csv(config)
            if not normalized_data:
                raise ValueError("No data loaded - check file paths in config")
            
            # 2. Denormalize all input datasets using JSON parameters
            print("Step 2: Denormalize all input datasets")
            denormalized_data = denormalize_all_datasets(normalized_data, config)
            
            # 3. Create FIRST sliding windows from denormalized data used only and only for baseline extraction
            print("Step 3: Create first sliding windows from denormalized data")
            denorm_sliding_windows = create_sliding_windows(denormalized_data, config, dates)

            # 4. Extract baselines from the sliding windows (last elements of each window for target column)
            print("Step 4: Extract baselines from sliding windows")
            baselines = extract_baselines_from_sliding_windows(denorm_sliding_windows, config)

            # 5. Calculate targets directly from baselines
            print("Step 5: Calculate targets from baselines")
            #TODO: verify this method is correct
            targets = calculate_targets_from_baselines(baselines, config)

            # 6. Create SECOND sliding windows from ORIGINAL normalized datasets after applying per-feature log-returns
            #    Apply to all numeric columns; non-numeric (e.g., DATE_TIME) are preserved untouched.
            print("Step 6: Apply log-returns to normalized X datasets and create second sliding windows")
            logret_normalized_x = self._build_logreturn_normalized_x(normalized_data, config)
            final_sliding_windows = create_sliding_windows(logret_normalized_x, config, dates)

            # 7. Align final sliding windows with target data length
            print("Step 7: Align sliding windows with target data")
            final_sliding_windows = self._align_sliding_windows_with_targets(final_sliding_windows, targets, config)
            
            # === Debugging & Invariants: Verify targets vs baselines and X statistics ===
            try:
                tgt_factor = float(config.get('target_factor', 1000.0))
                target_col = config.get('target_column', 'CLOSE')
                # Per-split feature names
                fn_train = final_sliding_windows.get('feature_names_train', final_sliding_windows.get('feature_names', []))
                fn_val = final_sliding_windows.get('feature_names_val', final_sliding_windows.get('feature_names', []))
                fn_test = final_sliding_windows.get('feature_names_test', final_sliding_windows.get('feature_names', []))

                # 1) Re-derive H1 targets from baselines and compare
                for split in ['train', 'val', 'test']:
                    base = baselines.get(f'baseline_{split}')
                    if base is None or len(base) < 2:
                        continue
                    # Recompute H1 log-returns scaled by target_factor
                    base = np.asarray(base, dtype=np.float64)
                    valid = (base[:-1] > 0) & (base[1:] > 0)
                    recomputed = np.zeros(len(base)-1, dtype=np.float64)
                    recomputed[valid] = tgt_factor * np.log(base[1:][valid] / base[:-1][valid])
                    # Truncate to max_samples used in target calc
                    max_h = max(config['predicted_horizons'])
                    max_samples = len(base) - max_h
                    if max_samples > 0:
                        recomputed = recomputed[:max_samples]
                        y_split = targets.get(f'y_{split}', {})
                        y_h1 = y_split.get('output_horizon_1')
                        if y_h1 is not None and len(y_h1) > 0:
                            y_h1 = np.asarray(y_h1, dtype=np.float64)
                            m = min(len(y_h1), len(recomputed))
                            diff = np.abs(y_h1[:m] - recomputed[:m])
                            print(f"DEBUG[{split.upper()}]: H1 targets check — mean|diff|={diff.mean():.6f}, max|diff|={diff.max():.6f}, samples={m}")

                # 2) Feature distributions for X (train)
                Xtr = final_sliding_windows.get('X_train')
                if Xtr is not None and hasattr(Xtr, 'shape'):
                    # global stats
                    print(f"DEBUG[X_train]: shape={Xtr.shape}, global mean={np.mean(Xtr):.6f}, std={np.std(Xtr):.6f}")
                    # CLOSE last-timestep stats and correlation with H1 target
                    if isinstance(fn_train, list) and target_col in fn_train:
                        ci = fn_train.index(target_col)
                        x_last_close = Xtr[:, -1, ci].astype(np.float64)
                        y_h1_tr = targets.get('y_train', {}).get('output_horizon_1')
                        if y_h1_tr is not None and len(y_h1_tr) == len(x_last_close):
                            y_arr = np.asarray(y_h1_tr, dtype=np.float64)
                            # Corr can be nan if std is zero; guard
                            corr = np.nan
                            if np.std(x_last_close) > 0 and np.std(y_arr) > 0:
                                corr = np.corrcoef(x_last_close, y_arr)[0,1]
                            print(f"DEBUG[CLOSE vs H1]: mean(x)={x_last_close.mean():.6f}, std(x)={x_last_close.std():.6f}, mean(y)={y_arr.mean():.6f}, std(y)={y_arr.std():.6f}, corr={corr}")
                        else:
                            print("DEBUG: Skipping CLOSE/H1 correlation — length mismatch")
                    else:
                        print("DEBUG: CLOSE not found in feature_names_train for X stats")
            except Exception as dbg_e:
                print(f"WARN: Debug invariant checks failed: {dbg_e}")

            # Return final results
            #TODO: verify this method is correct and required
            output, preprocessor_params = self._prepare_final_output(final_sliding_windows, targets, baselines, config)
            
            # Store baselines for access in output preparation
            self.extracted_baselines = baselines
            
            self.params.update(preprocessor_params)
            return output

        except Exception as e:
            print(f"ERROR in process_data: {e}")
            raise

    def _build_logreturn_normalized_x(self, normalized_data, config):
        """Build a dict with x_*_df only, applying log-returns to every numeric feature.

        Notes:
        - Uses the ORIGINAL normalized datasets loaded from CSVs.
        - Applies log-returns column-wise: ln(x_t / x_{t-1}); first element set to 0.0 by apply_log_returns_to_series.
        - Non-numeric columns are left as-is (e.g., datetime-like). Index is preserved; shape stays the same.
        """
        import pandas as pd

        out = {}
        for split in ['train', 'val', 'test']:
            key = f'x_{split}_df'
            if key not in normalized_data:
                continue
            df = normalized_data[key]
            if df is None or len(df) == 0:
                out[key] = df
                continue

            # Transform numeric columns with log-returns, preserve others
            df_transformed = df.copy()
            for col in df.columns:
                series = df[col]
                if pd.api.types.is_numeric_dtype(series):
                    try:
                        df_transformed[col] = apply_log_returns_to_series(series)
                    except Exception as e:
                        print(f"        WARN: log-returns failed for column '{col}' in {key}: {e}; preserving original")
                        df_transformed[col] = series
                else:
                    # Preserve non-numeric columns (e.g., DATE_TIME)
                    df_transformed[col] = series

            out[key] = df_transformed

        return out

    def _align_sliding_windows_with_targets(self, sliding_windows, targets, config):
        """Align sliding windows with target data to ensure same number of samples."""
        print("  Aligning sliding windows with target data...")
        
        # Get the first target to determine the target length
        predicted_horizons = config['predicted_horizons']
        first_horizon = predicted_horizons[0]
        
        # Find target lengths for each split
        target_lengths = {}
        for split in ['train', 'val', 'test']:
            target_key = f'y_{split}'
            if target_key in targets and f'output_horizon_{first_horizon}' in targets[target_key]:
                target_length = len(targets[target_key][f'output_horizon_{first_horizon}'])
                target_lengths[split] = target_length
                print(f"    {split} target length: {target_length}")
            else:
                target_lengths[split] = 0
        
        # Trim sliding windows to match target lengths
        aligned_windows = {}

        for key, windows in sliding_windows.items():
            if key.startswith('X_'):
                # Extract split name (train, val, test)
                split = key.split('_')[1]
                if split in target_lengths and target_lengths[split] > 0:
                    target_length = target_lengths[split]
                    if hasattr(windows, 'shape') and len(windows) > target_length:
                        aligned_windows[key] = windows[:target_length]
                        print(f"    Trimmed {key} from {len(windows)} to {target_length} samples")
                    else:
                        aligned_windows[key] = windows
                        
                else:
                    aligned_windows[key] = windows
                    
            else:
                # Keep non-window data as is
                aligned_windows[key] = windows
                

        return aligned_windows

    def _prepare_final_output(self, sliding_windows, targets, baselines, config):
        """Prepare final output structure."""
        # Use the baselines passed as parameter (extracted from denormalized data)
        baseline_data = {}
        if isinstance(baselines, dict):
            # baselines is already in the correct format
            baseline_data = baselines
        else:
            # Handle legacy format
            for split in ['train', 'val', 'test']:
                baseline_key = f'baseline_{split}'
                baseline_data[baseline_key] = np.array([])
        
        # Validate that we have the required data structures
        required_sliding_window_keys = ['X_train', 'X_val', 'X_test']
        required_target_keys = ['y_train', 'y_val', 'y_test']
        
        for key in required_sliding_window_keys:
            if key not in sliding_windows:
                print(f"WARNING: Missing sliding window data: {key}")
                sliding_windows[key] = np.array([])
        
        for key in required_target_keys:
            if key not in targets:
                print(f"WARNING: Missing target data: {key}")
                targets[key] = {}
        
        output = {
            # Final sliding windows for model (SECOND sliding windows after anti-naive-lock)
            "x_train": sliding_windows['X_train'],
            "x_val": sliding_windows['X_val'],
            "x_test": sliding_windows['X_test'],
            
            # Targets by horizon (calculated from FIRST sliding windows)
            "y_train": targets['y_train'],
            "y_val": targets['y_val'],
            "y_test": targets['y_test'],
            
            # Dates
            "x_train_dates": sliding_windows.get('x_dates_train'),
            "y_train_dates": sliding_windows.get('x_dates_train'),
            "x_val_dates": sliding_windows.get('x_dates_val'),
            "y_val_dates": sliding_windows.get('x_dates_val'),
            "x_test_dates": sliding_windows.get('x_dates_test'),
            "y_test_dates": sliding_windows.get('x_dates_test'),
            
            # Baselines for prediction reconstruction
            "baseline_train": baseline_data.get('baseline_train', np.array([])),
            "baseline_val": baseline_data.get('baseline_val', np.array([])),
            "baseline_test": baseline_data.get('baseline_test', np.array([])),
            
            # Metadata
            "feature_names": sliding_windows.get('feature_names', []),
            "feature_names_train": sliding_windows.get('feature_names_train', []),
            "feature_names_val": sliding_windows.get('feature_names_val', []),
            "feature_names_test": sliding_windows.get('feature_names_test', []),
            "target_returns_means": targets.get('target_returns_means', []),
            "target_returns_stds": targets.get('target_returns_stds', []),
            "predicted_horizons": config['predicted_horizons'],
            "normalization_json": load_normalization_json(config),
        }
        
        # Print summary statistics
        print("\nPreprocessing Summary:")
        print(f"  X_train shape: {output['x_train'].shape if hasattr(output['x_train'], 'shape') else 'N/A'}")
        print(f"  X_val shape: {output['x_val'].shape if hasattr(output['x_val'], 'shape') else 'N/A'}")
        print(f"  X_test shape: {output['x_test'].shape if hasattr(output['x_test'], 'shape') else 'N/A'}")
        print(f"  Feature names: {len(output['feature_names'])}")
        print(f"  Predicted horizons: {output['predicted_horizons']}")
        print(f"  Target normalization parameters available: {len(output['target_returns_means'])}")
        print(f"  Baseline train length: {len(output['baseline_train'])}")
        print(f"  Baseline val length: {len(output['baseline_val'])}")
        print(f"  Baseline test length: {len(output['baseline_test'])}")

        output, preprocessor_params = exclude_columns_from_datasets(output, self.params, config)

        return output, preprocessor_params
    
    def run_preprocessing(self, config):
        """Run preprocessing with configuration."""
        run_config = self.params.copy()
        run_config.update(config)
        self.set_params(**run_config)
        processed_data = self.process_data(self.params)
        
        params_with_targets = self.params.copy()
        params_with_targets.update({
            "target_returns_means": processed_data.get("target_returns_means", []),
            "target_returns_stds": processed_data.get("target_returns_stds", []),
            "normalization_json": processed_data.get("normalization_json", {})
        })
        
        return processed_data, params_with_targets


# Plugin interface alias for the system
PreprocessorPlugin = STLPreprocessorZScore
