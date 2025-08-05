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
from .anti_naive_lock import AntiNaiveLockProcessor


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
        "anti_naive_lock_enabled": True,
        "feature_preprocessing_strategy": "selective",
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
        self.anti_naive_lock_processor = AntiNaiveLockProcessor()

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

    def _load_data(self, file_path, max_rows, headers, config):
        """Load and validate data from CSV file, then IMMEDIATELY denormalize."""
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
            
            # 🔑 CRITICAL: DENORMALIZE DATA IMMEDIATELY AFTER LOADING
            print(f"🔍 IMMEDIATE DENORMALIZATION: Processing {file_path}...")
            norm_json = load_normalization_json(config)
            
            # Denormalize each column that has normalization parameters
            denormalized_count = 0
            for column in df.columns:
                if column in norm_json:
                    original_data = df[column].values
                    denormalized_data = denormalize(original_data, norm_json, column)
                    df[column] = denormalized_data
                    denormalized_count += 1
                    
                    # DEBUG: Check denormalization for CLOSE prices
                    if column == 'CLOSE':
                        print(f"  🔍 {column}: NORMALIZED -> DENORMALIZED")
                        print(f"    Before: min={np.min(original_data):.6f}, max={np.max(original_data):.6f}, mean={np.mean(original_data):.6f}")
                        print(f"    After:  min={np.min(denormalized_data):.6f}, max={np.max(denormalized_data):.6f}, mean={np.mean(denormalized_data):.6f}")
                        if np.all(denormalized_data > 0):
                            print(f"    ✅ {column} successfully denormalized - all values positive")
                        else:
                            print(f"    ❌ {column} denormalization failed - some values non-positive")
            
            print(f"  ✅ Denormalized {denormalized_count} columns in {file_path}")
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
            
            # Store DENORMALIZED dataframes (aligned_data was denormalized in step 2.5)
            baseline_data[f'x_{split}_df'] = x_df  # These are now denormalized
            baseline_data[f'y_{split}_df'] = y_df  # These are now denormalized
            
            # CLOSE prices are now DENORMALIZED (from step 2.5) - sliding windows will contain real prices
            close_denormalized = x_df["CLOSE"].astype(np.float32).values
            baseline_data[f'close_{split}'] = close_denormalized  # Store denormalized for consistency
            
            # Extract, denormalize and trim target column for baseline usage
            target_from_y_normalized = y_df[target_column].astype(np.float32).values
            target_from_y_denormalized = denormalize(target_from_y_normalized, norm_json, target_column)
            
            # Trim first window_size-1 elements to align with sliding windows
            target_from_y_trimmed = target_from_y_denormalized[window_size-1:]
            baseline_data[f'target_baseline_{split}'] = target_from_y_trimmed
            
            # Extract and trim dates
            dates = x_df.index if isinstance(x_df.index, pd.DatetimeIndex) else None
            if dates is not None:
                dates_trimmed = dates[window_size-1:]
                baseline_data[f'dates_{split}'] = dates_trimmed
            else:
                baseline_data[f'dates_{split}'] = None
            
            print(f"{split.capitalize()} baseline prepared: {len(close_denormalized)} samples (now denormalized), target baseline: {len(target_from_y_trimmed)} samples (denormalized)")
        
        return baseline_data

    def _calculate_log_return_targets(self, baselines_denorm, predicted_horizons, config):
        """Calculate log return targets from denormalized baselines."""
        print("Calculating log return targets from denormalized baselines...")
        
        target_data = {'y_train': {}, 'y_val': {}, 'y_test': {}}
        
        for split in ['train', 'val', 'test']:
            baselines = baselines_denorm[f'baseline_{split}']
            if len(baselines) == 0:
                for h in predicted_horizons:
                    target_data[f'y_{split}'][f'output_horizon_{h}'] = np.array([])
                continue
            
            for h in predicted_horizons:
                if len(baselines) > h:
                    baseline_values = baselines[:-h]
                    future_values = baselines[h:]
                    
                    # Calculate log returns: log(future/baseline)
                    ratios = future_values / baseline_values
                    
                    # Check for invalid ratios
                    invalid_mask = (ratios <= 0) | np.isnan(ratios) | np.isinf(ratios)
                    if np.any(invalid_mask):
                        invalid_count = np.sum(invalid_mask)
                        print(f"WARNING: {invalid_count} invalid ratios in {split} H{h}")
                        # Replace invalid values with small positive number
                        ratios[invalid_mask] = 1e-8
                    
                    log_returns = np.log(ratios)
                    target_data[f'y_{split}'][f'output_horizon_{h}'] = log_returns.astype(np.float32)
                    print(f"  {split} H{h}: {len(log_returns)} targets")
                else:
                    target_data[f'y_{split}'][f'output_horizon_{h}'] = np.array([])
        
        # Return unnormalized target stats (mean=0, std=1 for all horizons)
        target_data['target_returns_means'] = [0.0] * len(predicted_horizons)
        target_data['target_returns_stds'] = [1.0] * len(predicted_horizons)
        
        return target_data

    def _apply_selective_preprocessing(self, aligned_denorm_data, config):
        """Apply selective preprocessing to input data (NOT sliding windows)."""
        print("Applying selective preprocessing to input data...")
        
        # Apply anti-naive-lock preprocessing to the raw denormalized data
        preprocessed_data = {}
        
        for split in ['train', 'val', 'test']:
            x_df = aligned_denorm_data[f'x_{split}_df']
            y_df = aligned_denorm_data[f'y_{split}_df']
            
            # Apply selective preprocessing to each dataframe
            x_preprocessed = self.anti_naive_lock_processor.preprocess_dataframe(x_df, config, split)
            
            preprocessed_data[f'x_{split}_df'] = x_preprocessed
            preprocessed_data[f'y_{split}_df'] = y_df  # Y data doesn't need preprocessing
            
            print(f"  {split}: {x_df.shape} -> {x_preprocessed.shape}")
        
        return preprocessed_data

    def _normalize_with_training_params(self, preprocessed_data):
        """Z-score normalize using training set parameters."""
        print("Z-score normalizing using training set parameters...")
        
        # Calculate normalization parameters from training set
        x_train_df = preprocessed_data['x_train_df']
        training_norm_params = {}
        
        for column in x_train_df.columns:
            col_data = x_train_df[column].values
            mean_val = np.mean(col_data)
            std_val = np.std(col_data)
            training_norm_params[column] = {'mean': mean_val, 'std': std_val}
        
        print(f"Calculated normalization parameters for {len(training_norm_params)} features")
        
        # Apply normalization to all splits
        normalized_data = {}
        
        for split in ['train', 'val', 'test']:
            x_df = preprocessed_data[f'x_{split}_df'].copy()
            y_df = preprocessed_data[f'y_{split}_df'].copy()
            
            # Normalize X data using training parameters
            for column in x_df.columns:
                if column in training_norm_params:
                    params = training_norm_params[column]
                    if params['std'] > 0:
                        x_df[column] = (x_df[column] - params['mean']) / params['std']
                    else:
                        x_df[column] = 0  # Handle constant columns
            
            # Y data normalization (for target column only, if needed)
            target_column = 'CLOSE'  # Assuming CLOSE is in Y data
            if target_column in y_df.columns and target_column in training_norm_params:
                params = training_norm_params[target_column]
                if params['std'] > 0:
                    y_df[target_column] = (y_df[target_column] - params['mean']) / params['std']
            
            normalized_data[f'x_{split}_df'] = x_df
            normalized_data[f'y_{split}_df'] = y_df
            
            print(f"  {split}: Applied normalization")
        
        return normalized_data, training_norm_params

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
        """Main processing pipeline - FOOL-PROOF PLAN IMPLEMENTATION."""
        print("\n" + "="*15 + " Starting Preprocessing - FOOL-PROOF PLAN " + "="*15)
        self.set_params(**config)
        config = self.params
        self.scalers = {}  # Reset scalers
        
        # Validate configuration
        predicted_horizons = config['predicted_horizons']
        if not isinstance(predicted_horizons, list) or not predicted_horizons: 
            raise ValueError("'predicted_horizons' must be a non-empty list.")
        
        # --- STEP 1: Load Data AND DENORMALIZE IMMEDIATELY ---
        print("\n--- STEP 1: Loading Data AND DENORMALIZING IMMEDIATELY ---")
        x_train_df_denorm = self._load_data(config["x_train_file"], config.get("max_steps_train"), config.get("headers"), config)
        x_val_df_denorm = self._load_data(config["x_validation_file"], config.get("max_steps_val"), config.get("headers"), config)
        x_test_df_denorm = self._load_data(config["x_test_file"], config.get("max_steps_test"), config.get("headers"), config)
        y_train_df_denorm = self._load_data(config["y_train_file"], config.get("max_steps_train"), config.get("headers"), config)
        y_val_df_denorm = self._load_data(config["y_validation_file"], config.get("max_steps_val"), config.get("headers"), config)
        y_test_df_denorm = self._load_data(config["y_test_file"], config.get("max_steps_test"), config.get("headers"), config)
        
        # Align indices
        aligned_denorm_data = self._align_indices(x_train_df_denorm, y_train_df_denorm, x_val_df_denorm, y_val_df_denorm, x_test_df_denorm, y_test_df_denorm)
        
        # --- STEP 2: Calculate Sliding Windows with DENORMALIZED Data ---
        print("\n--- STEP 2: Calculate Sliding Windows with DENORMALIZED Data ---")
        baseline_data_denorm = self._prepare_baseline_data(aligned_denorm_data, config)
        windowed_data_denorm = self.sliding_windows_processor.generate_windowed_features(baseline_data_denorm, config)
        
        # --- STEP 3: Extract Baselines (last element of each window for target_column) ---
        print("\n--- STEP 3: Extract Baselines from Denormalized Sliding Windows ---")
        target_column = config.get("target_column", "CLOSE")
        feature_names = windowed_data_denorm.get('feature_names', [])
        
        if target_column not in feature_names:
            raise ValueError(f"Target column '{target_column}' not found in features: {feature_names}")
        
        target_feature_index = feature_names.index(target_column)
        baselines_denorm = {}
        
        for split in ['train', 'val', 'test']:
            X_matrix = windowed_data_denorm[f'X_{split}']
            if X_matrix.shape[0] > 0:
                target_windows = X_matrix[:, :, target_feature_index]
                baselines_denorm[f'baseline_{split}'] = target_windows[:, -1]  # Last element of each window
                print(f"  {split}: Extracted {len(baselines_denorm[f'baseline_{split}'])} baselines")
            else:
                baselines_denorm[f'baseline_{split}'] = np.array([])
        
        # --- STEP 4: Calculate Targets using Log Returns from Baselines ---
        print("\n--- STEP 4: Calculate Targets using Log Returns from Baselines ---")
        target_data = self._calculate_log_return_targets(baselines_denorm, predicted_horizons, config)
        
        # --- STEP 5: Apply Selective Preprocessing to INPUT DATA (NOT sliding windows) ---
        print("\n--- STEP 5: Apply Selective Preprocessing to INPUT DATA ---")
        preprocessed_data = self._apply_selective_preprocessing(aligned_denorm_data, config)
        
        # --- STEP 6: Z-score Normalize using TRAINING SET parameters ---
        print("\n--- STEP 6: Z-score Normalize using TRAINING SET parameters ---")
        normalized_data, training_norm_params = self._normalize_with_training_params(preprocessed_data)
        
        # --- STEP 7: Normalize Val/Test using Training Normalization Parameters ---
        print("\n--- STEP 7: Already done in step 6 ---")
        
        # --- STEP 8: Create Sliding Windows Matrix with Normalized Preprocessed Data ---
        print("\n--- STEP 8: Create Sliding Windows Matrix with Normalized Preprocessed Data ---")
        baseline_data_norm = self._prepare_baseline_data(normalized_data, config)
        windowed_data_final = self.sliding_windows_processor.generate_windowed_features(baseline_data_norm, config)
        
        # --- Prepare Final Output ---
        print("\n--- Preparing Final Output ---")
        ret = {
            # Final sliding windows for training (normalized preprocessed data)
            "x_train": windowed_data_final['X_train'],
            "x_val": windowed_data_final['X_val'],
            "x_test": windowed_data_final['X_test'],
            
            # Targets (calculated from denormalized baselines)
            "y_train": target_data['y_train'],
            "y_val": target_data['y_val'],
            "y_test": target_data['y_test'],
            
            # Dates
            "x_train_dates": windowed_data_final['x_dates_train'],
            "y_train_dates": windowed_data_final['x_dates_train'],
            "x_val_dates": windowed_data_final['x_dates_val'],
            "y_val_dates": windowed_data_final['x_dates_val'],
            "x_test_dates": windowed_data_final['x_dates_test'],
            "y_test_dates": windowed_data_final['x_dates_test'],
            
            # Baselines (denormalized for prediction reconstruction)
            "baseline_train": baselines_denorm['baseline_train'],
            "baseline_val": baselines_denorm['baseline_val'],
            "baseline_test": baselines_denorm['baseline_test'],
            
            # Metadata
            "feature_names": windowed_data_final['feature_names'],
            "target_returns_means": target_data['target_returns_means'],
            "target_returns_stds": target_data['target_returns_stds'],
            "predicted_horizons": predicted_horizons,
            "training_norm_params": training_norm_params,  # For de-transformation
        }
        
        print("\n" + "="*15 + " Preprocessing Finished - FOOL-PROOF PLAN " + "="*15)
        return ret

    def run_preprocessing(self, config):
        """Run preprocessing with given configuration."""
        run_config = self.params.copy()
        run_config.update(config)
        self.set_params(**run_config)
        processed_data = self.process_data(self.params)
        return processed_data, self.params