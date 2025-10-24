import numpy as np
import pandas as pd


class TargetPlugin:
    """
    Target calculation plugin with the same class/interface structure as other plugins.

    Responsibilities:
    - Accept pre-extracted baselines per split (train/val/test)
    - Compute targets per horizon using EXACTLY the same logic used previously
    - Expose standard plugin methods: set_params, get_debug_info, add_debug_info
    """

    # Plugin-specific parameters (kept minimal and aligned with target calculation needs)
    plugin_params = {
        "predicted_horizons": [1],
        "target_column": "CLOSE",
    }

    # Debug variables to surface
    plugin_debug_vars = ["predicted_horizons", "target_column"]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    @staticmethod
    def apply_centered_moving_average(data, window_size=2):
        """
        Apply a centered moving average to 1D data and return a numpy array.
        Accepts numpy arrays, pandas Series, or single-column DataFrames.
        """
        if window_size is None or window_size <= 1:
            return np.asarray(data)

        # Convert input to a pandas Series for rolling; if DataFrame, use the first column
        if isinstance(data, pd.DataFrame):
            if data.shape[1] == 0:
                return np.array([])
            series = data.iloc[:, 0]
        elif isinstance(data, pd.Series):
            series = data
        else:
            series = pd.Series(np.asarray(data))

        smoothed = series.rolling(window=window_size, center=True, min_periods=1).mean()
        return smoothed.to_numpy()

    def calculate_targets_from_baselines(self, baseline_data, config):
        """
        Execute target calculation using only baselines (direct-price targets), minimal flow.
        """
        # 1) Merge/lock configuration
        self.set_params(**config)
        cfg = self.params

        # 2) Read horizons and prepare containers
        horizons = cfg["predicted_horizons"]
        targets = {"train": {}, "val": {}, "test": {}}

        # 3) For each split, compute targets
        for split in ("train", "val", "test"):
            base_key = f"baseline_{split}"
            if base_key not in baseline_data:
                # 3.1) If baseline missing, create empty arrays per horizon
                for h in horizons:
                    targets[split][f"output_horizon_{h}"] = np.array([])
                continue

            baselines = baseline_data[base_key]
            if len(baselines) == 0:
                # 3.2) If baseline empty, create empty arrays per horizon
                for h in horizons:
                    targets[split][f"output_horizon_{h}"] = np.array([])
                continue

            # 3.3) Determine common max_samples bounded by max horizon
            max_h = max(horizons)
            max_samples = len(baselines) - max_h
            if max_samples <= 0:
                for h in horizons:
                    targets[split][f"output_horizon_{h}"] = np.array([])
                continue

            # 3.4) For each horizon, build direct-price targets from future baselines
            for h in horizons:
                vals = []
                for t in range(max_samples):
                    future_val = baselines[t + h]
                    vals.append(float(future_val) if np.isfinite(future_val) else 0.0)
                targets[split][f"output_horizon_{h}"] = np.asarray(vals, dtype=np.float32)

        # 4) Return compact result structure
        return {
            "y_train": targets["train"],
            "y_val": targets["val"],
            "y_test": targets["test"],
            "predicted_horizons": horizons,
        }

