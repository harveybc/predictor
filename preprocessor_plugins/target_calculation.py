import numpy as np
import pandas as pd

def apply_centered_moving_average(data, target_plugin, window_size=2):
    """Alias to the plugin's centered moving average implementation."""
    return target_plugin.apply_centered_moving_average(data, window_size=window_size)


def calculate_targets_from_baselines(baseline_data, target_plugin, config):
    """Alias that instantiates the TargetPlugin and runs the exact same calculation."""
    return target_plugin.calculate_targets_from_baselines(baseline_data, config)
