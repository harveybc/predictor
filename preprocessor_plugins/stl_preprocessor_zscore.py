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
    Modular preprocessor implementing the fool-proof 11-step plan:
    1. Load and immediately denormalize all CSV data
    2. Create sliding windows from denormalized data
    3. Extract baselines from sliding windows for target calculation  
    4. Calculate log return targets with normalization
    5-7. REMOVED - Streamlined processing
    8. Apply anti-naive-lock to existing sliding windows
    9. Truncate data for final alignment
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
        """Load CSV data and KEEP IT NORMALIZED for neural network training."""
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
            
            # 🔑 CRITICAL FIX: KEEP DATA NORMALIZED FOR NEURAL NETWORK TRAINING
            print(f"✅ KEEPING NORMALIZED DATA: Input data is already z-score normalized and ready for neural networks")
            norm_json = load_normalization_json(config)
            
            # Verify normalization by checking a few key features
            for column in ['CLOSE', 'RSI', 'MACD']:
                if column in df.columns:
                    col_data = df[column].values
                    col_mean = np.mean(col_data)
                    col_std = np.std(col_data)
                    print(f"  � {column}: mean={col_mean:.3f}, std={col_std:.3f} (normalized scale)")
                    
                    # Check if data appears properly normalized (should be roughly mean~0, std~1 or similar small scale)
                    if abs(col_mean) > 100 or col_std > 100:
                        print(f"    ⚠️ WARNING: {column} may not be properly normalized (large values detected)")
                    else:
                        print(f"    ✅ {column} appears properly normalized for neural network training")
            
            print(f"  ✅ Data kept in normalized state for optimal neural network training")
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
            
            # Store DENORMALIZED dataframes (aligned_data was denormalized in step 1)
            baseline_data[f'x_{split}_df'] = x_df  # These are now denormalized
            baseline_data[f'y_{split}_df'] = y_df  # These are now denormalized
            
            # CLOSE prices are now DENORMALIZED (from step 1) - sliding windows will contain real prices
            close_denormalized = x_df["CLOSE"].astype(np.float32).values
            baseline_data[f'close_{split}'] = close_denormalized  # Store denormalized for consistency
            
            # Extract target column - ALREADY DENORMALIZED in step 1, no need to denormalize again
            target_from_y_denormalized = y_df[target_column].astype(np.float32).values
            
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
                
                # Verify all horizons have the same length
                for horizon_key, horizon_data in target_data[f'y_{split}'].items():
                    if len(horizon_data) != target_length:
                        print(f"WARNING: Inconsistent target lengths in {split}: {horizon_key} has {len(horizon_data)}, expected {target_length}")
                
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
        """Main processing pipeline - CORRECT FLOW IMPLEMENTATION."""
        print("\n" + "="*15 + " Starting Preprocessing - CORRECT FLOW " + "="*15)
        self.set_params(**config)
        config = self.params
        self.scalers = {}  # Reset scalers
        
        # Validate configuration
        predicted_horizons = config['predicted_horizons']
        if not isinstance(predicted_horizons, list) or not predicted_horizons: 
            raise ValueError("'predicted_horizons' must be a non-empty list.")
        
        # --- STEP 1: Load NORMALIZED CSV data (keep as-is from CSV) ---
        print("\n--- STEP 1: Loading NORMALIZED CSV data (keep as-is) ---")
        x_train_df_norm = self._load_data(config["x_train_file"], config.get("max_steps_train"), config.get("headers"), config)
        x_val_df_norm = self._load_data(config["x_validation_file"], config.get("max_steps_val"), config.get("headers"), config)
        x_test_df_norm = self._load_data(config["x_test_file"], config.get("max_steps_test"), config.get("headers"), config)
        y_train_df_norm = self._load_data(config["y_train_file"], config.get("max_steps_train"), config.get("headers"), config)
        y_val_df_norm = self._load_data(config["y_validation_file"], config.get("max_steps_val"), config.get("headers"), config)
        y_test_df_norm = self._load_data(config["y_test_file"], config.get("max_steps_test"), config.get("headers"), config)
        
        # Align indices of normalized data
        aligned_norm_data = self._align_indices(x_train_df_norm, y_train_df_norm, x_val_df_norm, y_val_df_norm, x_test_df_norm, y_test_df_norm)
        
        # --- STEP 2: Denormalize aligned data for baseline extraction ---
        print("\n--- STEP 2: Denormalize aligned data for baseline extraction ---")
        norm_json = load_normalization_json(config)
        aligned_denorm_data = {}
        
        for split in ['train', 'val', 'test']:
            print(f"Denormalizing {split} data for baseline extraction...")
            
            # Denormalize X data
            x_df_norm = aligned_norm_data[f'x_{split}_df']
            x_df_denorm = x_df_norm.copy()
            
            for column in x_df_denorm.columns:
                if column in norm_json:
                    norm_data = x_df_denorm[column].values
                    denorm_data = denormalize(norm_data, norm_json, column)
                    x_df_denorm[column] = denorm_data
                    
                    if column == 'CLOSE':
                        print(f"  {column}: ${np.mean(denorm_data):.2f} ± ${np.std(denorm_data):.2f} (denormalized)")
            
            # Denormalize Y data
            y_df_norm = aligned_norm_data[f'y_{split}_df']
            y_df_denorm = y_df_norm.copy()
            
            target_column = config.get("target_column", "CLOSE")
            if target_column in y_df_denorm.columns and target_column in norm_json:
                norm_data = y_df_denorm[target_column].values
                denorm_data = denormalize(norm_data, norm_json, target_column)
                y_df_denorm[target_column] = denorm_data
            
            aligned_denorm_data[f'x_{split}_df'] = x_df_denorm
            aligned_denorm_data[f'y_{split}_df'] = y_df_denorm
        
        # --- STEP 3: Create sliding windows from DENORMALIZED data ---
        print("\n--- STEP 3: Create sliding windows from DENORMALIZED data ---")
        baseline_data_denorm = self._prepare_baseline_data(aligned_denorm_data, config)
        baseline_data_denorm['norm_json'] = norm_json
        windowed_data_denorm = self.sliding_windows_processor.generate_windowed_features(baseline_data_denorm, config)
        
        # --- STEP 4: Extract baselines from denormalized sliding windows ---
        print("\n--- STEP 4: Extract baselines from denormalized sliding windows ---")
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
                print(f"  {split}: Extracted {len(baselines_denorm[f'baseline_{split}'])} denormalized baselines")
                print(f"    Sample baseline prices: ${baselines_denorm[f'baseline_{split}'][:3]}")
            else:
                baselines_denorm[f'baseline_{split}'] = np.array([])
        
        # --- STEP 5: Calculate targets using denormalized baselines ---
        print("\n--- STEP 5: Calculate targets using denormalized baselines ---")
        
        # Store baselines in baseline_data for target calculation
        for split in ['train', 'val', 'test']:
            baseline_data_denorm[f'sliding_baseline_{split}'] = baselines_denorm[f'baseline_{split}']
            baseline_data_denorm[f'sliding_baseline_{split}_dates'] = windowed_data_denorm.get(f'x_dates_{split}')
        
        # Calculate targets using the target calculation processor
        target_data = self.target_calculation_processor.calculate_targets(baseline_data_denorm, windowed_data_denorm, config)
        
        # --- STEP 6: Apply anti-naive-lock transformations to input datasets ---
        print("\n--- STEP 6: Apply anti-naive-lock transformations to input datasets ---")
        
        processed_data = aligned_denorm_data.copy()  # Start with denormalized data
        
        if config.get("anti_naive_lock_enabled", True):
            print("Applying anti-naive-lock transformations to denormalized input datasets...")
            
            # Apply transformations to each split's input data
            for split in ['train', 'val', 'test']:
                x_df_denorm = aligned_denorm_data[f'x_{split}_df']
                
                print(f"  Processing {split} data with anti-naive-lock...")
                
                # Apply anti-naive-lock transformations using the processor
                # For this we need to create a dummy sliding window to use the existing interface
                feature_matrix = x_df_denorm.values.astype(np.float32)
                feature_names_list = list(x_df_denorm.columns)
                
                # Create a temporary single-window matrix for processing
                if len(feature_matrix) > 0:
                    temp_window = feature_matrix.reshape(1, -1, len(feature_names_list))  # (1, time_steps, features)
                    
                    # Apply anti-naive-lock (only using the first output since we're processing input data)
                    processed_temp, _, _, processing_stats = \
                        self.anti_naive_lock_processor.process_sliding_windows(
                            temp_window, temp_window, temp_window,  # Same data for all splits in this call
                            feature_names_list,
                            config
                        )
                    
                    # Extract the processed data and convert back to DataFrame
                    processed_matrix = processed_temp[0, :, :]  # Remove batch dimension (1, time_steps, features) -> (time_steps, features)
                    processed_df = pd.DataFrame(processed_matrix, columns=feature_names_list, index=x_df_denorm.index)
                    processed_data[f'x_{split}_df'] = processed_df
                    
                    print(f"    Applied transformations to {len(feature_names_list)} features")
                else:
                    print(f"    Warning: No data found for {split}")
                    processed_data[f'x_{split}_df'] = x_df_denorm
        else:
            print("Anti-naive-lock disabled - using original denormalized data")
            processed_data = aligned_denorm_data
        
        # --- STEP 7: Create final sliding windows from processed datasets ---
        print("\n--- STEP 7: Create final sliding windows from processed datasets ---")
        
        # Update baseline_data with processed datasets
        final_baseline_data = self._prepare_baseline_data(processed_data, config)
        final_baseline_data['norm_json'] = norm_json
        
        # Create final sliding windows from processed data
        windowed_data_final = self.sliding_windows_processor.generate_windowed_features(final_baseline_data, config)
        self._truncate_to_match_targets(windowed_data_final, target_data)
        
        # Extract baselines from target_data (they're included in the result)
        baselines_final = {
            'baseline_train': target_data.get('baseline_train', np.array([])),
            'baseline_val': target_data.get('baseline_val', np.array([])),
            'baseline_test': target_data.get('baseline_test', np.array([]))
        }
        
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
            "baseline_train": baselines_final['baseline_train'],
            "baseline_val": baselines_final['baseline_val'],
            "baseline_test": baselines_final['baseline_test'],
            
            # Metadata
            "feature_names": windowed_data_final['feature_names'],
            "target_returns_means": target_data['target_returns_means'],
            "target_returns_stds": target_data['target_returns_stds'],
            "predicted_horizons": predicted_horizons,
            "training_norm_params": {},  # No normalization applied
            "normalization_json": self.denorm_params,  # Original normalization for denormalization
        }
        
        print("\n" + "="*15 + " Preprocessing Finished - FOOL-PROOF PLAN " + "="*15)
        return ret

    def run_preprocessing(self, config):
        """Run preprocessing with given configuration."""
        run_config = self.params.copy()
        run_config.update(config)
        self.set_params(**run_config)
        processed_data = self.process_data(self.params)
        
        # Include target normalization parameters in returned params
        params_with_targets = self.params.copy()
        params_with_targets.update({
            "target_returns_means": processed_data.get("target_returns_means", []),
            "target_returns_stds": processed_data.get("target_returns_stds", []),
            "training_norm_params": {},  # No additional normalization applied
            "normalization_json": processed_data.get("normalization_json", {})  # Original normalization
        })
        
        return processed_data, params_with_targets