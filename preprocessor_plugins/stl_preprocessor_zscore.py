import numpy as np
import pandas as pd
import os

try:
    from app.data_handler import load_csv
except ImportError:
    print("CRITICAL ERROR: Could not import 'load_csv' from 'app.data_handler'.")
    raise

from .helpers import load_normalization_json, denormalize, verify_date_consistency
from .sliding_windows import SlidingWindowsProcessor
from .target_calculation import TargetCalculationProcessor


class PreprocessorPlugin:
    """
    Modular preprocessor that orchestrates data loading, windowing, and target calculation.
    """
    
    plugin_params = {
        "x_train_file": "data/x_train.csv",
        "y_train_file": "data/y_train.csv",
        "x_validation_file": "data/x_val.csv",
        "y_validation_file": "data/y_val.csv",
        "x_test_file": "data/x_test.csv",
        "y_test_file": "data/y_test.csv",
        "headers": True,
        "max_steps_train": None, 
        "max_steps_val": None, 
        "max_steps_test": None,
        "target_column": "TARGET",
        "window_size": 48,
        "predicted_horizons": [24, 48, 72, 96, 120, 144],
        "use_returns": True,
        "normalize_features": True,
    }
    
    plugin_debug_vars = [
        "window_size", "predicted_horizons", "use_returns", "normalize_features",
        "target_returns_means", "target_returns_stds"
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.scalers = {}
        self.sliding_windows_processor = SlidingWindowsProcessor(self.scalers)
        self.target_calculation_processor = TargetCalculationProcessor()

    def set_params(self, **kwargs):
        for key, value in kwargs.items(): 
            self.params[key] = value

    def get_debug_info(self):
        debug_info = {}
        for var in self.plugin_debug_vars:
            value = self.params.get(var)
            debug_info[var] = value
        return debug_info

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def _load_data(self, file_path, max_rows, headers):
        """Load and validate data from CSV file."""
        print(f"Loading data: {file_path} (Max rows: {max_rows})...", end="")
        try:
            df = load_csv(file_path, headers=headers, max_rows=max_rows)
            if df is None or df.empty: 
                raise ValueError(f"load_csv None/empty for {file_path}")
            print(f" Done. Shape: {df.shape}")
            
            # Handle datetime index conversion
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"Attempting DatetimeIndex conversion for {file_path}...", end="")
                original_index_name = df.index.name
                try: 
                    df.index = pd.to_datetime(df.index)
                    print(" OK (from index).")
                except Exception:
                    try: 
                        df.index = pd.to_datetime(df.iloc[:, 0])
                        print(" OK (from col 0).")
                    except Exception as e_col: 
                        print(f" FAILED ({e_col}). Dates unavailable.")
                        df.index = None
                if original_index_name: 
                    df.index.name = original_index_name
            
            # Validate required columns
            required_cols = ["CLOSE"]
            target_col_name = self.params.get("target_column", "TARGET")
            if 'y_' in os.path.basename(file_path).lower(): 
                required_cols.append(target_col_name)
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols: 
                raise ValueError(f"Missing cols in {file_path}: {missing_cols}")
            return df
            
        except FileNotFoundError: 
            print(f"\nERROR: File not found: {file_path}.")
            raise
        except Exception as e: 
            print(f"\nERROR loading/processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _align_indices(self, x_train_df, y_train_df, x_val_df, y_val_df, x_test_df, y_test_df):
        """Align indices between X and Y dataframes for all splits."""
        print("\n--- 2. Aligning Indices ---")
        
        aligned_data = {}
        for split, x_df, y_df in [("Train", x_train_df, y_train_df), 
                                 ("Validation", x_val_df, y_val_df), 
                                 ("Test", x_test_df, y_test_df)]:
            
            # Align datetime indices if both are DatetimeIndex
            if isinstance(x_df.index, pd.DatetimeIndex) and isinstance(y_df.index, pd.DatetimeIndex):
                if not x_df.index.equals(y_df.index):
                    print(f"WARN: {split} X/Y index misalignment. X: {x_df.index[0]} to {x_df.index[-1]}, Y: {y_df.index[0]} to {y_df.index[-1]}")
                    common_index = x_df.index.intersection(y_df.index)
                    if len(common_index) == 0:
                        raise ValueError(f"No common indices between X and Y for {split} split.")
                    print(f"Aligning {split} to common indices: {len(common_index)} rows.")
                    x_df = x_df.loc[common_index]
                    y_df = y_df.loc[common_index]
            
            # Align lengths
            if len(x_df) != len(y_df):
                min_len = min(len(x_df), len(y_df))
                print(f"WARN: {split} X/Y length mismatch. X: {len(x_df)}, Y: {len(y_df)}. Truncating to {min_len}.")
                x_df = x_df.iloc[:min_len]
                y_df = y_df.iloc[:min_len]
            
            # Store aligned data
            split_key = split.lower().replace('validation', 'val')
            aligned_data[f'x_{split_key}_df'] = x_df
            aligned_data[f'y_{split_key}_df'] = y_df
        
        return aligned_data

    def _prepare_baseline_data(self, aligned_data, config):
        """Prepare baseline data from aligned dataframes."""
        print("\n--- 3. Preparing Baseline Data ---")
        
        norm_json = load_normalization_json(config)
        baseline_data = {'norm_json': norm_json}
        
        target_column = config.get("target_column", "CLOSE")
        window_size = config.get("window_size", 48)
        
        splits = ['train', 'val', 'test']
        for split in splits:
            x_df = aligned_data[f'x_{split}_df']
            y_df = aligned_data[f'y_{split}_df']
            
            # Store original dataframes (baseline - untouched)
            baseline_data[f'x_{split}_df'] = x_df
            baseline_data[f'y_{split}_df'] = y_df
            
            # Extract and denormalize CLOSE prices for windowing
            close_normalized = x_df["CLOSE"].astype(np.float32).values
            close_denormalized = denormalize(close_normalized, norm_json, "CLOSE")
            baseline_data[f'close_{split}'] = close_denormalized
            
            # Extract, denormalize and trim target column for baseline usage
            target_normalized = y_df[target_column].astype(np.float32).values
            target_denormalized = denormalize(target_normalized, norm_json, target_column)
            
            # Trim first window_size-1 elements to align with sliding windows
            target_trimmed = target_denormalized[window_size-1:]
            baseline_data[f'target_baseline_{split}'] = target_trimmed
            
            # Extract and trim dates
            dates = x_df.index if isinstance(x_df.index, pd.DatetimeIndex) else None
            if dates is not None:
                dates_trimmed = dates[window_size-1:]
                baseline_data[f'dates_{split}'] = dates_trimmed
            else:
                baseline_data[f'dates_{split}'] = None
            
            print(f"{split.capitalize()} baseline prepared: {len(close_denormalized)} samples, target baseline: {len(target_trimmed)} samples")
        
        return baseline_data

    def _calculate_sliding_windows_baseline(self, aligned_data, config):
        """
        Calculate sliding windows baseline using the SlidingWindowsProcessor.
        Maintains separation of concerns by delegating to the appropriate module.
        """
        return self.sliding_windows_processor.calculate_sliding_window_baselines(aligned_data, config)

    def _truncate_to_match_targets(self, windowed_data, target_data):
        """Truncate windowed data to match target data lengths."""
        print("\n--- Final Length Alignment ---")
        
        splits = ['train', 'val', 'test']
        for split in splits:
            # Get target data length for this split (all horizons should have same length now)
            if f'y_{split}' in target_data and target_data[f'y_{split}']:
                # Get the length from the first horizon
                first_horizon_key = list(target_data[f'y_{split}'].keys())[0]
                target_length = len(target_data[f'y_{split}'][first_horizon_key])
                
                # Truncate windowed features to match target length
                X_key = f'X_{split}'
                if X_key in windowed_data and len(windowed_data[X_key]) > target_length:
                    print(f"Truncating {split} X data from {len(windowed_data[X_key])} to {target_length} samples")
                    windowed_data[X_key] = windowed_data[X_key][:target_length]
                    
                    # Also truncate dates
                    dates_key = f'x_dates_{split}'
                    if dates_key in windowed_data and windowed_data[dates_key] is not None:
                        if len(windowed_data[dates_key]) > target_length:
                            windowed_data[dates_key] = windowed_data[dates_key][:target_length]
                    
                    # Update sample count
                    windowed_data[f'num_samples_{split}'] = target_length
            else:
                print(f"No target data found for {split} split")

    def process_data(self, config):
        """Main processing pipeline."""
        print("\n" + "="*15 + " Starting Preprocessing " + "="*15)
        self.set_params(**config)
        config = self.params
        self.scalers = {}  # Reset scalers
        
        # Validate configuration
        predicted_horizons = config['predicted_horizons']
        if not isinstance(predicted_horizons, list) or not predicted_horizons: 
            raise ValueError("'predicted_horizons' must be a non-empty list.")
        
        # --- 1. Load Data ---
        print("\n--- 1. Loading Data ---")
        x_train_df = self._load_data(config["x_train_file"], config.get("max_steps_train"), config.get("headers"))
        x_val_df = self._load_data(config["x_validation_file"], config.get("max_steps_val"), config.get("headers"))
        x_test_df = self._load_data(config["x_test_file"], config.get("max_steps_test"), config.get("headers"))
        y_train_df = self._load_data(config["y_train_file"], config.get("max_steps_train"), config.get("headers"))
        y_val_df = self._load_data(config["y_validation_file"], config.get("max_steps_val"), config.get("headers"))
        y_test_df = self._load_data(config["y_test_file"], config.get("max_steps_test"), config.get("headers"))
        
        # --- 2. Align Indices ---
        aligned_data = self._align_indices(x_train_df, y_train_df, x_val_df, y_val_df, x_test_df, y_test_df)
        
        # --- 3. Calculate Sliding Windows Immediately ---
        sliding_windows_data = self._calculate_sliding_windows_baseline(aligned_data, config)
        
        # --- 4. Prepare Baseline Data ---
        baseline_data = self._prepare_baseline_data(aligned_data, config)
        
        # --- 5. Add sliding window baselines to baseline_data ---
        baseline_data.update(sliding_windows_data)
        
        # --- 5. Generate Windowed Features ---
        windowed_data = self.sliding_windows_processor.generate_windowed_features(baseline_data, config)
        
        # --- 5. Calculate Targets ---
        target_data = self.target_calculation_processor.calculate_targets(baseline_data, windowed_data, config)
        
        # --- 6. Final Alignment ---
        self._truncate_to_match_targets(windowed_data, target_data)
        
        # --- 7. Update params with individual normalization stats ---
        self.params['target_returns_means'] = target_data['target_returns_means']
        self.params['target_returns_stds'] = target_data['target_returns_stds']
        
        # --- 8. Final Date Consistency Check ---
        print("\n--- Final Date Consistency Checks ---")
        splits = ['train', 'val', 'test']
        for split in splits:
            x_dates = windowed_data.get(f'x_dates_{split}')
            baseline_dates = target_data.get(f'baseline_{split}_dates')
            verify_date_consistency([
                list(x_dates) if x_dates is not None else None,
                list(baseline_dates) if baseline_dates is not None else None
            ], f"{split.capitalize()} X/Y Dates")
        
        # --- 9. Prepare Final Output ---
        print("\n--- Preparing Final Output ---")
        ret = {
            # Windowed features
            "x_train": windowed_data['X_train'],
            "x_val": windowed_data['X_val'],
            "x_test": windowed_data['X_test'],
            
            # Targets
            "y_train": target_data['y_train'],
            "y_val": target_data['y_val'],
            "y_test": target_data['y_test'],
            
            # Dates for windowed data
            "x_train_dates": windowed_data['x_dates_train'],
            "y_train_dates": windowed_data['x_dates_train'],  # Same as x_dates
            "x_val_dates": windowed_data['x_dates_val'],
            "y_val_dates": windowed_data['x_dates_val'],  # Same as x_dates
            "x_test_dates": windowed_data['x_dates_test'],
            "y_test_dates": windowed_data['x_dates_test'],  # Same as x_dates
            
            # Baseline data (denormalized target column, trimmed and aligned)
            "baseline_train": target_data['baseline_train'],
            "baseline_val": target_data['baseline_val'],
            "baseline_test": target_data['baseline_test'],
            
            # Baseline dates
            "baseline_train_dates": target_data['baseline_train_dates'],
            "baseline_val_dates": target_data['baseline_val_dates'],
            "baseline_test_dates": target_data['baseline_test_dates'],
            
            # Additional data for evaluation
            "test_close_prices": target_data['test_close_prices'],
            "y_test_raw": target_data['y_test_raw'],
            
            # Metadata
            "feature_names": windowed_data['feature_names'],
            "target_returns_means": target_data['target_returns_means'],
            "target_returns_stds": target_data['target_returns_stds'],
            "predicted_horizons": target_data['predicted_horizons'],
        }
        
        # Print final summary
        print(f"Final shapes:")
        predicted_horizons = config['predicted_horizons']
        if ret['y_train']:
            first_horizon = predicted_horizons[0]
            print(f"  X: Train={ret['x_train'].shape}, Val={ret['x_val'].shape}, Test={ret['x_test'].shape}")
            print(f"  Y: {len(predicted_horizons)} horizons, "
                  f"Train={len(ret['y_train'][f'output_horizon_{first_horizon}'])}, "
                  f"Val={len(ret['y_val'][f'output_horizon_{first_horizon}'])}, "
                  f"Test={len(ret['y_test'][f'output_horizon_{first_horizon}'])}")
        print(f"  Baselines: Train={len(ret['baseline_train'])}, Val={len(ret['baseline_val'])}, Test={len(ret['baseline_test'])}")
        print(f"  Horizons: {predicted_horizons}")
        print(f"  Features ({len(ret['feature_names'])}): {ret['feature_names']}")
        print(f"  Target normalization per horizon:")
        for i, h in enumerate(predicted_horizons):
            mean_h = ret['target_returns_means'][i]
            std_h = ret['target_returns_stds'][i]
            print(f"    Horizon {h}: mean={mean_h:.6f}, std={std_h:.6f}")
        
        print("\n" + "="*15 + " Preprocessing Finished " + "="*15)
        return ret

    def run_preprocessing(self, config):
        """Run preprocessing with given configuration."""
        run_config = self.params.copy()
        run_config.update(config)
        self.set_params(**run_config)
        processed_data = self.process_data(self.params)
        return processed_data, self.params