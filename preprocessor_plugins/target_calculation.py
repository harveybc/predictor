import numpy as np
import pandas as pd

# Delegate methods to the new TargetPlugin to keep behavior identical while centralizing logic.
from target_plugins.default_target import TargetPlugin as _TargetPlugin


def apply_centered_moving_average(data, window_size=2):
    """Alias to the plugin's centered moving average implementation."""
    return _TargetPlugin.apply_centered_moving_average(data, window_size=window_size)


def calculate_targets_from_baselines(baseline_data, config):
    """Alias that instantiates the TargetPlugin and runs the exact same calculation."""
    plugin = _TargetPlugin()
    plugin.set_params(**config)
    return plugin.process_data(baseline_data, plugin.params)
