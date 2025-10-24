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
        "use_returns": True,
        "target_softening": 1,  # moving average window; 1 disables smoothing
        "target_factor": 1000.0,
    }

    # Debug variables to surface
    plugin_debug_vars = ["predicted_horizons", "target_column", "use_returns", "target_softening"]

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
        Execute the target calculation using ONLY baselines.

        Input:
          - baseline_data: dict with keys baseline_train/baseline_val/baseline_test
          - config: configuration dict

        Output:
          - dict containing y_train/y_val/y_test per horizon, baseline_* echoes, and metadata
        """
        # Standardize/merge params
        self.set_params(**config)
        config = self.params

        print("Calculating targets from provided baselines...")

        predicted_horizons = config["predicted_horizons"]
        use_returns = config.get("use_returns", True)
        target_softening = config.get("target_softening", 2)

        target_data = {"train": {}, "val": {}, "test": {}}
        baseline_info = {}

        # Reset normalization stats (kept for compatibility, currently unused)
        target_returns_means = []
        target_returns_stds = []

        if not use_returns:
            print("Using direct target values (not returns)")

        # Optional smoothing of baselines before target computation
        if target_softening > 1:
            for split in ["train", "val", "test"]:
                baseline_key = f"baseline_{split}"
                if baseline_key in baseline_data:
                    baselines = baseline_data[baseline_key]
                    if use_returns:
                        baseline_data[baseline_key] = self.apply_centered_moving_average(
                            baselines, window_size=target_softening
                        )
                    else:
                        ser = pd.Series(np.asarray(baselines))
                        baseline_data[baseline_key] = (
                            ser.rolling(window=target_softening, center=False, min_periods=1)
                            .mean()
                            .to_numpy()
                        )

        print("Calculating targets (returns or direct price depending on use_returns)...")

        # Calculate targets for each split using ONLY baselines
        for split in ["train", "val", "test"]:
            baseline_key = f"baseline_{split}"

            if baseline_key not in baseline_data:
                print(f"No baseline data for {split}")
                continue

            baselines = baseline_data[baseline_key]
            if len(baselines) == 0:
                print(
                    f"Empty baselines for {split} -> creating empty target arrays for horizons {predicted_horizons}"
                )
                for horizon in predicted_horizons:
                    target_data[split][f"output_horizon_{horizon}"] = np.array([])
                continue

            # Calculate max horizon to ensure all horizons have same length
            max_horizon = max(predicted_horizons)
            max_samples = len(baselines) - max_horizon

            if max_samples <= 0:
                print(
                    f"  {split}: Insufficient baselines ({len(baselines)}) for max horizon {max_horizon}"
                )
                for horizon in predicted_horizons:
                    target_data[split][f"output_horizon_{horizon}"] = np.array([])
                continue

            print(
                f"  {split}: Using {max_samples} samples for ALL horizons (limited by H{max_horizon})"
            )
            target_factor = config.get("target_factor", 1000.0)

            # Calculate targets for each horizon using ONLY baselines
            for _, horizon in enumerate(predicted_horizons):
                horizon_targets = []

                # Calculate targets: log(baseline[t+horizon] / baseline[t]) or direct future value
                for t in range(max_samples):
                    baseline_current = baselines[t]
                    baseline_future = baselines[t + horizon]

                    if use_returns:
                        if baseline_current > 0 and baseline_future > 0:
                            return_value = target_factor * np.log(
                                baseline_future / baseline_current
                            )
                        else:
                            return_value = 0.0
                    else:
                        if np.isfinite(baseline_future):
                            return_value = float(baseline_future)
                        else:
                            return_value = 0.0

                    horizon_targets.append(return_value)

                # Store targets for this horizon
                if horizon_targets:
                    horizon_targets = np.array(horizon_targets, dtype=np.float32)
                    valid_targets = horizon_targets[~np.isnan(horizon_targets)]

                    if len(valid_targets) > 0:
                        normalized_targets = valid_targets
                        target_data[split][f"output_horizon_{horizon}"] = normalized_targets
                        print(
                            f"  {split} H{horizon}: {len(normalized_targets)} targets"
                        )
                    else:
                        print(f"  {split} H{horizon}: No valid targets")
                        target_data[split][f"output_horizon_{horizon}"] = np.array([])
                else:
                    print(f"  {split} H{horizon}: No targets calculated")
                    target_data[split][f"output_horizon_{horizon}"] = np.array([])

        # Store baseline info for output
        for split in ["train", "val", "test"]:
            baseline_key = f"baseline_{split}"
            if baseline_key in baseline_data:
                baseline_info[baseline_key] = baseline_data[baseline_key]

        # Prepare final result
        result = {
            "y_train": target_data["train"],
            "y_val": target_data["val"],
            "y_test": target_data["test"],
            **baseline_info,
            "target_returns_means": target_returns_means,
            "target_returns_stds": target_returns_stds,
            "predicted_horizons": predicted_horizons,
        }

        print(f"Target calculation complete. Horizons: {predicted_horizons}")
        print(f"Normalization means: {target_returns_means}")
        print(f"Normalization stds: {target_returns_stds}")

        return result

    def run_target(self, baseline_data, config):
        """Convenience runner mirroring other plugins' run_* methods."""
        run_config = self.params.copy()
        run_config.update(config)
        self.set_params(**run_config)
        processed = self.process_data(baseline_data, self.params)

        params_with_targets = self.params.copy()
        params_with_targets.update(
            {
                "target_returns_means": processed.get("target_returns_means", []),
                "target_returns_stds": processed.get("target_returns_stds", []),
            }
        )

        return processed, params_with_targets
