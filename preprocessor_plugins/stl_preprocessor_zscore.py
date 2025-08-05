import numpy as np
import pandas as pd
from .helpers import denormalize, load_normalization_json
from .sliding_windows import SlidingWindowsProcessor
from .target_calculation import TargetCalculationProcessor
from .anti_naive_lock import AntiNaiveLockProcessor


class STLPreprocessorZScore:
    """Preprocessor implementing exact requirements - no compromises."""
    
    plugin_params = {
        "window_size": 48,
        "predicted_horizons": [1, 6],
        "target_column": "CLOSE",
        "use_returns": True,
        "anti_naive_lock_enabled": True,
        "feature_preprocessing_strategy": "selective"
    }
    
    plugin_debug_vars = ["window_size", "predicted_horizons", "target_column"]
    
    def __init__(self):
        self.params = self.plugin_params.copy()
        self.sliding_windows_processor = SlidingWindowsProcessor()
        self.target_calculation_processor = TargetCalculationProcessor()
        self.anti_naive_lock_processor = AntiNaiveLockProcessor()
    
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value
    
    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}
    
    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())
    
    def process_data(self, config):
        """
        EXACT REQUIREMENTS IMPLEMENTATION:
        1. Load already normalized CSV data ✅
        2. Denormalize all input datasets using JSON parameters
        3. Create sliding windows from denormalized data
        4. Extract baselines (last elements of each window for target column)
        5. Calculate log return targets with those baselines (train, validation, test)
        6. Apply anti-naive-lock transformations to the denormalized input datasets
        7. Create final sliding windows matrix from processed datasets
        Keep baselines and targets unchanged (they're already calculated correctly)
        """
        self.set_params(**config)
        config = self.params
        
        predicted_horizons = config['predicted_horizons']
        if not isinstance(predicted_horizons, list) or not predicted_horizons:
            raise ValueError("predicted_horizons must be a non-empty list")
        
        # 1. Load already normalized CSV data
        print("Step 1: Load normalized CSV data")
        normalized_data = self._load_normalized_csv(config)
        
        # 2. Denormalize all input datasets using JSON parameters
        print("Step 2: Denormalize all input datasets")
        denormalized_data = self._denormalize_all_datasets(normalized_data, config)
        
        # 3. Create sliding windows from denormalized data
        print("Step 3: Create sliding windows from denormalized data")
        denorm_sliding_windows = self._create_sliding_windows_from_denormalized(denormalized_data, config)
        
        # 4. SKIP baseline extraction here - we'll do it after anti-naive-lock
        
        # 5. SKIP target calculation here - we'll do it after we have final baselines
        
        # 6. Apply anti-naive-lock transformations to denormalized input datasets
        print("Step 6: Apply anti-naive-lock to denormalized datasets")
        processed_data = self._apply_anti_naive_lock_to_datasets(denormalized_data, config)
        
        # 7. Create final sliding windows matrix from processed datasets
        print("Step 7: Create final sliding windows from processed datasets")
        final_sliding_windows = self._create_final_sliding_windows(processed_data, config)
        
        # 4. Extract baselines from FINAL sliding windows (CRITICAL FIX)
        print("Step 4: Extract baselines from FINAL sliding windows")
        baselines = self._extract_baselines_from_windows(final_sliding_windows, config)
        
        # 5. Calculate log return targets with FINAL baselines
        print("Step 5: Calculate log return targets with FINAL baselines")
        targets = self._calculate_targets_with_baselines(baselines, denormalized_data, config)
        
        # Align final sliding windows with targets (both are now from the same matrix)
        self._align_sliding_windows_with_targets(final_sliding_windows, targets)
        
        # Return final results
        return self._prepare_final_output(final_sliding_windows, targets, baselines, config)
    
    def _load_normalized_csv(self, config):
        """Step 1: Load normalized CSV data as-is."""
        data = {}
        
        file_mappings = [
            ('x_train_df', 'x_train_file', 'max_steps_train'),
            ('y_train_df', 'y_train_file', 'max_steps_train'),
            ('x_val_df', 'x_validation_file', 'max_steps_val'),
            ('y_val_df', 'y_validation_file', 'max_steps_val'),
            ('x_test_df', 'x_test_file', 'max_steps_test'),
            ('y_test_df', 'y_test_file', 'max_steps_test')
        ]
        
        for data_key, file_key, max_steps_key in file_mappings:
            if file_key in config:
                df = pd.read_csv(config[file_key], parse_dates=True, index_col=0)
                if config.get(max_steps_key):
                    df = df.head(config[max_steps_key])
                data[data_key] = df
        
        return self._align_dataframe_indices(data)
    
    def _align_dataframe_indices(self, data):
        """Align all dataframe indices to common timestamps."""
        aligned = {}
        
        for split in ['train', 'val', 'test']:
            x_key = f'x_{split}_df'
            y_key = f'y_{split}_df'
            
            if x_key in data and y_key in data:
                x_df = data[x_key]
                y_df = data[y_key]
                common_index = x_df.index.intersection(y_df.index)
                aligned[x_key] = x_df.loc[common_index]
                aligned[y_key] = y_df.loc[common_index]
        
        return aligned
    
    def _denormalize_all_datasets(self, normalized_data, config):
        """Step 2: Denormalize all datasets using JSON parameters."""
        norm_json = load_normalization_json(config)
        denormalized = {}
        
        for data_key, normalized_df in normalized_data.items():
            denorm_df = normalized_df.copy()
            for column in denorm_df.columns:
                if column in norm_json:
                    denorm_df[column] = denormalize(normalized_df[column].values, norm_json, column)
            denormalized[data_key] = denorm_df
            
        return denormalized
    
    def _create_sliding_windows_from_denormalized(self, denormalized_data, config):
        """Step 3: Create sliding windows from denormalized data."""
        prepared_data = {'norm_json': load_normalization_json(config)}
        prepared_data.update(denormalized_data)
        
        # Add dates for sliding windows
        for split in ['train', 'val', 'test']:
            x_key = f'x_{split}_df'
            if x_key in denormalized_data:
                prepared_data[f'dates_{split}'] = denormalized_data[x_key].index
        
        return self.sliding_windows_processor.generate_windowed_features(prepared_data, config)
    
    def _extract_baselines_from_windows(self, sliding_windows, config):
        """Step 4: Extract baselines (last elements of each window for target column) with correct temporal alignment."""
        target_column = config['target_column']
        feature_names = sliding_windows['feature_names']
        
        if target_column not in feature_names:
            raise ValueError(f"Target column {target_column} not in features: {feature_names}")
        
        target_index = feature_names.index(target_column)
        baselines = {}
        
        for split in ['train', 'val', 'test']:
            X_key = f'X_{split}'
            dates_key = f'x_dates_{split}'
            if X_key in sliding_windows:
                X_matrix = sliding_windows[X_key]
                if X_matrix.shape[0] > 0:
                    # Extract last element of each window for target column (this is the baseline at time t)
                    baseline_prices = X_matrix[:, -1, target_index]
                    baselines[f'baseline_{split}'] = baseline_prices
                    
                    # Baseline dates correspond to the time when the baseline price occurs
                    # This should be the date of the last element in each sliding window
                    if dates_key in sliding_windows:
                        baseline_dates = sliding_windows[dates_key]
                        baselines[f'baseline_{split}_dates'] = baseline_dates
                    else:
                        baselines[f'baseline_{split}_dates'] = None
                        
                    print(f"  Extracted {len(baseline_prices)} baselines for {split}")
                else:
                    baselines[f'baseline_{split}'] = np.array([])
                    baselines[f'baseline_{split}_dates'] = np.array([])
                    print(f"  No data for {split} baseline extraction")
        
        return baselines
    
    def _calculate_targets_with_baselines(self, baselines, denormalized_data, config):
        """Step 5: Calculate log return targets with baselines."""
        # Prepare data for target calculation
        baseline_data = {'norm_json': load_normalization_json(config)}
        baseline_data.update(denormalized_data)
        baseline_data.update(baselines)
        
        # Create dummy windowed data (not used, but required by interface)
        dummy_windowed_data = {'feature_names': list(denormalized_data['x_train_df'].columns)}
        
        return self.target_calculation_processor.calculate_targets(baseline_data, dummy_windowed_data, config)
    
    def _apply_anti_naive_lock_to_datasets(self, denormalized_data, config):
        """Step 6: Apply anti-naive-lock transformations to denormalized datasets."""
        if not config.get("anti_naive_lock_enabled", True):
            return denormalized_data
        
        processed = {}
        
        for split in ['train', 'val', 'test']:
            x_key = f'x_{split}_df'
            y_key = f'y_{split}_df'
            
            if x_key in denormalized_data:
                x_df = denormalized_data[x_key]
                feature_names = list(x_df.columns)
                
                # Convert to format expected by anti-naive-lock processor
                x_matrix = x_df.values.reshape(1, len(x_df), len(feature_names))
                
                # Apply transformations
                processed_matrix, _, _, _ = self.anti_naive_lock_processor.process_sliding_windows(
                    x_matrix, x_matrix, x_matrix, feature_names, config
                )
                
                # Convert back to DataFrame
                processed[x_key] = pd.DataFrame(
                    processed_matrix[0], 
                    index=x_df.index, 
                    columns=feature_names
                )
            
            # Y data unchanged
            if y_key in denormalized_data:
                processed[y_key] = denormalized_data[y_key]
        
        return processed
    
    def _create_final_sliding_windows(self, processed_data, config):
        """Step 7: Create final sliding windows from processed datasets."""
        prepared_data = {'norm_json': load_normalization_json(config)}
        prepared_data.update(processed_data)
        
        # Add dates for sliding windows
        for split in ['train', 'val', 'test']:
            x_key = f'x_{split}_df'
            if x_key in processed_data:
                prepared_data[f'dates_{split}'] = processed_data[x_key].index
        
        return self.sliding_windows_processor.generate_windowed_features(prepared_data, config)
    
    def _align_sliding_windows_with_targets(self, sliding_windows, targets):
        """Align final sliding windows with targets."""
        predicted_horizons = targets['predicted_horizons']
        
        for split in ['train', 'val', 'test']:
            X_key = f'X_{split}'
            if X_key in sliding_windows and f'y_{split}' in targets:
                X_len = len(sliding_windows[X_key])
                y_len = len(targets[f'y_{split}'][f'output_horizon_{predicted_horizons[0]}'])
                
                min_len = min(X_len, y_len)
                sliding_windows[X_key] = sliding_windows[X_key][:min_len]
                
                # Align dates
                date_key = f'x_dates_{split}'
                if date_key in sliding_windows:
                    sliding_windows[date_key] = sliding_windows[date_key][:min_len]
    
    def _prepare_final_output(self, sliding_windows, targets, baselines, config):
        """Prepare final output structure."""
        return {
            # Final sliding windows for model
            "x_train": sliding_windows['X_train'],
            "x_val": sliding_windows['X_val'],
            "x_test": sliding_windows['X_test'],
            
            # Targets by horizon
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
            "baseline_train": baselines.get('baseline_train', np.array([])),
            "baseline_val": baselines.get('baseline_val', np.array([])),
            "baseline_test": baselines.get('baseline_test', np.array([])),
            
            # Metadata
            "feature_names": sliding_windows['feature_names'],
            "target_returns_means": targets['target_returns_means'],
            "target_returns_stds": targets['target_returns_stds'],
            "predicted_horizons": config['predicted_horizons'],
            "normalization_json": load_normalization_json(config),
        }
    
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
