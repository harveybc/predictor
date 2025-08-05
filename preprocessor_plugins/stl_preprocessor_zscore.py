import numpy as np
import pandas as pd
from .helpers import denormalize, load_normalization_json
from .sliding_windows import SlidingWindowsProcessor
from .target_calculation import TargetCalculationProcessor
from .anti_naive_lock import AntiNaiveLockProcessor


class STLPreprocessorZScore:
    """Clean preprocessor implementing exact requirements."""
    
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
        Exact requirements implementation:
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
            raise ValueError("'predicted_horizons' must be a non-empty list.")
        
        # 1. Load already normalized CSV data
        print("=== Step 1: Load normalized CSV data ===")
        normalized_data = self._load_normalized_data(config)
        
        # 2. Denormalize all input datasets using JSON parameters
        print("=== Step 2: Denormalize input datasets ===")
        denormalized_data = self._denormalize_datasets(normalized_data, config)
        
        # 3. Create sliding windows from denormalized data
        print("=== Step 3: Create sliding windows from denormalized data ===")
        baseline_data = self._prepare_data_for_processors(denormalized_data, config)
        sliding_windows_for_baselines = self.sliding_windows_processor.generate_windowed_features(baseline_data, config)
        
        # 4 & 5. Extract baselines and calculate targets
        print("=== Step 4-5: Extract baselines and calculate targets ===")
        target_data = self.target_calculation_processor.calculate_targets(baseline_data, sliding_windows_for_baselines, config)
        
        # 6. Apply anti-naive-lock transformations to denormalized input datasets
        print("=== Step 6: Apply anti-naive-lock transformations ===")
        processed_datasets = self._apply_anti_naive_lock(denormalized_data, config)
        
        # 7. Create final sliding windows matrix from processed datasets
        print("=== Step 7: Create final sliding windows from processed datasets ===")
        final_data = self._prepare_data_for_processors(processed_datasets, config)
        final_sliding_windows = self.sliding_windows_processor.generate_windowed_features(final_data, config)
        
        # Align final sliding windows with targets
        self._align_with_targets(final_sliding_windows, target_data)
        
        # Return final results
        return self._prepare_output(final_sliding_windows, target_data, config)
    
    def _load_normalized_data(self, config):
        """Load normalized CSV data as-is."""
        data = {}
        file_keys = [
            ('x_train_df', 'x_train_file'),
            ('y_train_df', 'y_train_file'),
            ('x_val_df', 'x_validation_file'),
            ('y_val_df', 'y_validation_file'),
            ('x_test_df', 'x_test_file'),
            ('y_test_df', 'y_test_file')
        ]
        
        for key, file_key in file_keys:
            if file_key in config:
                df = pd.read_csv(config[file_key], parse_dates=True, index_col=0)
                if config.get(f"max_steps_{key.split('_')[1]}", None):
                    df = df.head(config[f"max_steps_{key.split('_')[1]}"])
                data[key] = df
        
        # Align indices
        return self._align_indices(data)
    
    def _align_indices(self, data):
        """Align all dataframe indices."""
        splits = ['train', 'val', 'test']
        aligned = {}
        
        for split in splits:
            x_key = f'x_{split}_df'
            y_key = f'y_{split}_df'
            
            if x_key in data and y_key in data:
                x_df = data[x_key]
                y_df = data[y_key]
                
                common_index = x_df.index.intersection(y_df.index)
                aligned[x_key] = x_df.loc[common_index]
                aligned[y_key] = y_df.loc[common_index]
        
        return aligned
    
    def _denormalize_datasets(self, normalized_data, config):
        """Denormalize all datasets using JSON parameters."""
        norm_json = load_normalization_json(config)
        denormalized = {}
        
        for key, df in normalized_data.items():
            denorm_df = df.copy()
            for column in denorm_df.columns:
                if column in norm_json:
                    denorm_df[column] = denormalize(df[column].values, norm_json, column)
            denormalized[key] = denorm_df
            
        return denormalized
    
    def _prepare_data_for_processors(self, datasets, config):
        """Prepare data structure for sliding windows and target processors."""
        data = {'norm_json': load_normalization_json(config)}
        data.update(datasets)
        
        # Add dates
        for split in ['train', 'val', 'test']:
            x_key = f'x_{split}_df'
            if x_key in datasets:
                x_df = datasets[x_key]
                data[f'dates_{split}'] = x_df.index if isinstance(x_df.index, pd.DatetimeIndex) else None
        
        return data
    
    def _apply_anti_naive_lock(self, denormalized_data, config):
        """Apply anti-naive-lock transformations to input datasets."""
        if not config.get("anti_naive_lock_enabled", True):
            return denormalized_data
        
        processed = {}
        
        for split in ['train', 'val', 'test']:
            x_key = f'x_{split}_df'
            y_key = f'y_{split}_df'
            
            if x_key in denormalized_data:
                x_df = denormalized_data[x_key]
                
                # Convert to sliding window format for anti-naive-lock processor
                feature_names = list(x_df.columns)
                x_matrix = x_df.values.reshape(1, len(x_df), len(feature_names))
                
                # Apply anti-naive-lock transformations per feature configuration
                processed_matrix, _, _, _ = self.anti_naive_lock_processor.process_sliding_windows(
                    x_matrix, x_matrix, x_matrix, feature_names, config
                )
                
                # Convert back to DataFrame
                processed_df = pd.DataFrame(
                    processed_matrix[0], 
                    index=x_df.index, 
                    columns=feature_names
                )
                processed[x_key] = processed_df
            
            # Y data unchanged
            if y_key in denormalized_data:
                processed[y_key] = denormalized_data[y_key]
        
        return processed
    
    def _align_with_targets(self, sliding_windows, target_data):
        """Align sliding windows with target data."""
        for split in ['train', 'val', 'test']:
            X_key = f'X_{split}'
            y_key = f'y_{split}'
            
            if X_key in sliding_windows and y_key in target_data:
                X_len = len(sliding_windows[X_key])
                y_len = len(target_data[y_key][f'output_horizon_{target_data["predicted_horizons"][0]}'])
                
                min_len = min(X_len, y_len)
                sliding_windows[X_key] = sliding_windows[X_key][:min_len]
                
                # Align dates
                date_key = f'x_dates_{split}'
                if date_key in sliding_windows:
                    sliding_windows[date_key] = sliding_windows[date_key][:min_len]
    
    def _prepare_output(self, sliding_windows, target_data, config):
        """Prepare final output structure."""
        predicted_horizons = config['predicted_horizons']
        
        return {
            # Final sliding windows for model
            "x_train": sliding_windows['X_train'],
            "x_val": sliding_windows['X_val'],
            "x_test": sliding_windows['X_test'],
            
            # Targets by horizon
            "y_train": target_data['y_train'],
            "y_val": target_data['y_val'],
            "y_test": target_data['y_test'],
            
            # Dates
            "x_train_dates": sliding_windows.get('x_dates_train'),
            "y_train_dates": sliding_windows.get('x_dates_train'),
            "x_val_dates": sliding_windows.get('x_dates_val'),
            "y_val_dates": sliding_windows.get('x_dates_val'),
            "x_test_dates": sliding_windows.get('x_dates_test'),
            "y_test_dates": sliding_windows.get('x_dates_test'),
            
            # Baselines for prediction reconstruction
            "baseline_train": target_data.get('baseline_train', np.array([])),
            "baseline_val": target_data.get('baseline_val', np.array([])),
            "baseline_test": target_data.get('baseline_test', np.array([])),
            
            # Metadata
            "feature_names": sliding_windows['feature_names'],
            "target_returns_means": target_data['target_returns_means'],
            "target_returns_stds": target_data['target_returns_stds'],
            "predicted_horizons": predicted_horizons,
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
