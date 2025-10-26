"""
Normalization helpers for the STL pipeline (use_returns=False).

Provides:
- denormalize: map normalized prices back to price space.
- denormalize_returns: scale differences/uncertainties to price units.
"""
from __future__ import annotations
from typing import Dict
import json
import numpy as np


def denormalize(data: np.ndarray, config: Dict) -> np.ndarray:
    """Map normalized prices back to price space using config normalization.

    Supports either min/max or mean/std forms under CLOSE entry.
    """
    data = np.asarray(data)
    if config.get("use_normalization_json"):
        norm_json = config["use_normalization_json"]
        if isinstance(norm_json, str):
            try:
                with open(norm_json, "r") as f:
                    norm_json = json.load(f)
            except Exception as e:
                print(f"WARN: Failed load norm JSON {norm_json}: {e}")
                return data
        if isinstance(norm_json, dict) and "CLOSE" in norm_json:
            try:
                close_info = norm_json["CLOSE"]
                if "min" in close_info and "max" in close_info:
                    close_min = close_info["min"]
                    close_max = close_info["max"]
                    diff = close_max - close_min
                    if diff == 0:
                        return data + close_min
                    return data * diff + close_min
                elif "mean" in close_info and "std" in close_info:
                    mean = close_info["mean"]
                    std = close_info["std"]
                    return data * std + mean
                else:
                    return data
            except KeyError as e:
                print(f"WARN: Missing key in norm JSON: {e}")
                return data
            except Exception as e:
                print(f"WARN: Error during denormalize: {e}")
                return data
    return data


def denormalize_returns(data: np.ndarray, config: Dict) -> np.ndarray:
    """Scale differences/uncertainties from normalized units to price units.

    For differences (pred - target) and uncertainties, scale by the range (min/max)
    or by the std (mean/std) without adding bias.
    """
    data = np.asarray(data)
    if config.get("use_normalization_json"):
        norm_json = config["use_normalization_json"]
        if isinstance(norm_json, str):
            try:
                with open(norm_json, "r") as f:
                    norm_json = json.load(f)
            except Exception as e:
                print(f"WARN: Failed load norm JSON {norm_json}: {e}")
                return data
        if isinstance(norm_json, dict) and "CLOSE" in norm_json:
            try:
                close_info = norm_json["CLOSE"]
                if "min" in close_info and "max" in close_info:
                    close_min = close_info["min"]
                    close_max = close_info["max"]
                    diff = close_max - close_min
                    if diff == 0:
                        return data
                    return data * diff
                elif "mean" in close_info and "std" in close_info:
                    std = close_info["std"]
                    return data * std
                else:
                    return data
            except KeyError as e:
                print(f"WARN: Missing key in norm JSON: {e}")
                return data
            except Exception as e:
                print(f"WARN: Error during denormalize_returns: {e}")
                return data
    return data
