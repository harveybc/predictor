import numpy as np
import pandas as pd
from .helpers import load_normalization_json, denormalize_all_datasets, load_normalized_csv, exclude_columns_from_datasets
from .sliding_windows import create_sliding_windows, extract_baselines_from_sliding_windows
from .target_calculation import calculate_targets_from_baselines
from .anti_naive_lock import apply_anti_naive_lock_to_datasets


class STLPreprocessorZScore:
    """
    1. Load already normalized CSV data ✅
    2. Denormalize all input datasets using JSON parameters
    3. Create sliding windows from denormalized data
    4. Extract baselines (last elements of each window for target column)
    5. Calculate log return targets with those baselines (train, validation, test)
    6. Apply anti-naive-lock transformations to the denormalized input datasets of step 2
    7. Create final sliding windows matrix from anti-naive-lock processed datasets
    8.Keep baselines and targets unchanged (they're already calculated correctly)
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
            normalized_data = load_normalized_csv(config)
            if not normalized_data:
                raise ValueError("No data loaded - check file paths in config")
            
            # 2. Denormalize all input datasets using JSON parameters
            print("Step 2: Denormalize all input datasets")
            denormalized_data = denormalize_all_datasets(normalized_data, config)
            
            # 3. Create FIRST sliding windows from denormalized data used only and only for baseline extraction
            print("Step 3: Create first sliding windows from denormalized data")
            #TODO: verify this method is correct
            denorm_sliding_windows = create_sliding_windows(denormalized_data, config)

            # 4. Extract baselines from the denorm_sliding_windows (last elements of each window per tick for target column)
            print("Step 4: Extract baselines from sliding windows")
            #TODO: verify this method is correct
            baselines = extract_baselines_from_sliding_windows(denorm_sliding_windows, config)

            # 5. Calculate targets directly from baselines
            print("Step 5: Calculate targets from baselines")
            #TODO: verify this method is correct
            targets = calculate_targets_from_baselines(baselines, config)

            # 6. Apply anti-naive-lock transformations to denormalized input datasets (creates "processed data")
            print("Step 6: Apply anti-naive-lock to denormalized datasets")
            #TODO: verify this method is correct
            processed_data = apply_anti_naive_lock_to_datasets(denormalized_data, config)
            
            # 7. Create SECOND sliding windows matrix from processed datasets (for model input only)
            #TODO: why its using a different methiod for sliding windows creation
            print("Step 7: Create second sliding windows from processed datasets")
            final_sliding_windows = create_sliding_windows(processed_data, config)
            
            # Return final results
            #TODO: verify this method is correct and required
            output, preprocessor_params = self._prepare_final_output(final_sliding_windows, targets, baselines, config)
            self.params.update(preprocessor_params)
            return output

        except Exception as e:
            print(f"ERROR in process_data: {e}")
            raise

    def _prepare_final_output(self, sliding_windows, targets, baselines, config):
        """Prepare final output structure."""
        # We need to preserve the baselines that were calculated during target computation
        # They are available in the target calculation processor's state
        baseline_data = {}
        for split in ['train', 'val', 'test']:
            # Extract baseline data from targets structure if available
            baseline_key = f'baseline_{split}'
            if hasattr(self, 'extracted_baselines') and baseline_key in self.extracted_baselines:
                baseline_data[baseline_key] = self.extracted_baselines[baseline_key]
            else:
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
