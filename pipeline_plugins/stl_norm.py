"""
Normalization helpers for the STL pipeline (use_returns=False).

Provides:
- denormalize: map normalized prices back to price space.
- denormalize_returns: scale differences/uncertainties to price units.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import json
import numpy as np



# --- RP58: the transformation state is a declaration, not a shape -------------------------------

_LAST_DECISION: Dict = {"source": "NONE", "space": None, "inspected_values": False}
SPACES = ("NORMALIZED", "REAL")


def last_decision() -> Dict:
    """What the last call decided and on what grounds. A guess is recorded as a guess."""
    return dict(_LAST_DECISION)


def _record(source: str, space, *, inspected: bool, why: str = "") -> Dict:
    _LAST_DECISION.clear()
    _LAST_DECISION.update(source=source, space=space, inspected_values=inspected, why=why)
    if source == "HEURISTIC":
        _LAST_DECISION["how_to_remove_this_guess"] = (
            "declare prediction_space: NORMALIZED or REAL in the configuration; with it nothing is "
            "inspected, and strict_normalization_contract: true refuses instead of guessing")
    return dict(_LAST_DECISION)


def declared_space(config: Dict):
    """The space the CONTRACT says these arrays are in, or None when it says nothing.

    `targets_are_denormalized: true` is the older way of saying REAL and keeps working.
    """
    space = config.get("prediction_space")
    if space is not None:
        if space not in SPACES:
            raise ValueError(f"prediction_space must be one of {SPACES}, not {space!r}")
        return space
    if bool(config.get("targets_are_denormalized", False)):
        return "REAL"
    return None


def _space_or_refuse(config: Dict):
    space = declared_space(config)
    if space is None and bool(config.get("strict_normalization_contract", False)):
        raise ValueError("this configuration does not declare prediction_space, and "
                         "strict_normalization_contract forbids deciding it from the values")
    return space


def _select_norm_key(norm_json: dict, config: Dict) -> Optional[str]:
    """Pick the most appropriate column key from the normalization JSON.

    Preference order:
      1) config['target_column'] if present in norm_json
      2) 'CLOSE' if present
      3) None
    """
    target_key = config.get("target_column")
    if isinstance(target_key, str) and target_key in norm_json:
        return target_key
    if "CLOSE" in norm_json:
        return "CLOSE"
    return None


def _load_norm_json(config: Dict) -> Optional[dict]:
    norm_json = config.get("use_normalization_json")
    if not norm_json:
        return None
    if isinstance(norm_json, dict):
        return norm_json
    if isinstance(norm_json, str):
        try:
            with open(norm_json, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"WARN: Failed load norm JSON {norm_json}: {e}")
            return None
    return None


def _looks_normalized_like_standard_score(arr: np.ndarray, mean: float, std: float) -> bool:
    """Heuristic to avoid double-denormalizing.

    Returns True if arr's distribution looks closer to ~N(0,1) than to ~N(mean,std).
    """
    finite = np.asarray(arr)
    finite = finite[np.isfinite(finite)]
    if finite.size < 32:
        # Too few points; default to NOT denormalizing to avoid damaging already-real prices.
        return False

    a_mean = float(np.mean(finite))
    a_std = float(np.std(finite))

    # Distance to normalized space vs real space.
    d_norm = abs(a_mean - 0.0) + abs(a_std - 1.0)
    d_real = abs(a_mean - float(mean)) + abs(a_std - float(std))
    return d_norm < d_real


def denormalize(data: np.ndarray, config: Dict) -> np.ndarray:
    """Map normalized prices back to price space using config normalization.

    Supports either min/max or mean/std forms under CLOSE entry.
    """
    data = np.asarray(data)
    space = _space_or_refuse(config)
    if space == "REAL":
        _record("CONTRACT", "REAL", inspected=False,
                why="the configuration declares these arrays are already in price space")
        return data

    norm_json = _load_norm_json(config)
    if not isinstance(norm_json, dict):
        _record("NO_NORMALIZATION_DECLARED", None, inspected=False,
                why="the configuration names no normalization to invert")
        return data

    key = _select_norm_key(norm_json, config)
    if not key:
        _record("NO_NORMALIZATION_DECLARED", None, inspected=False,
                why="the normalization declares no entry for the target column or CLOSE")
        return data

    try:
        info = norm_json[key]
        if space == "NORMALIZED":
            _record("CONTRACT", "NORMALIZED", inspected=False,
                    why="the configuration declares these arrays are normalized")
            if "min" in info and "max" in info:
                lo, hi = float(info["min"]), float(info["max"])
                return data + lo if hi == lo else data*(hi-lo)+lo
            if "mean" in info and "std" in info:
                mean, std = float(info["mean"]), float(info["std"])
                return data + mean if std == 0 else data*std+mean
            return data
        if "min" in info and "max" in info:
            # Min-max normalization: assume normalized values are typically within [0,1].
            # Avoid double-denormalizing if values already look like real scale.
            close_min = float(info["min"])
            close_max = float(info["max"])
            diff = close_max - close_min
            if diff == 0:
                return data + close_min
            # Heuristic: if most values already within [min,max], likely already denormalized.
            finite = data[np.isfinite(data)]
            if finite.size >= 32:
                frac_in_range = float(np.mean((finite >= close_min) & (finite <= close_max)))
                if frac_in_range > 0.95:
                    _record("HEURISTIC", "REAL", inspected=True,
                            why=f"{frac_in_range:.3f} of the values fall inside [min, max]")
                    return data
            _record("HEURISTIC", "NORMALIZED", inspected=True,
                    why="the values do not look like they are already in price space")
            return data * diff + close_min

        if "mean" in info and "std" in info:
            mean = float(info["mean"])
            std = float(info["std"])
            if std == 0:
                return data + mean
            # Avoid double-denormalizing.
            if not _looks_normalized_like_standard_score(data, mean, std):
                _record("HEURISTIC", "REAL", inspected=True,
                        why="the values look closer to N(mean, std) than to N(0, 1)")
                return data
            _record("HEURISTIC", "NORMALIZED", inspected=True,
                    why="the values look closer to N(0, 1) than to N(mean, std)")
            return data * std + mean

        return data
    except Exception as e:
        print(f"WARN: Error during denormalize: {e}")
        return data


def denormalize_returns(data: np.ndarray, config: Dict) -> np.ndarray:
    """Scale differences/uncertainties from normalized units to price units.

    For differences (pred - target) and uncertainties, scale by the range (min/max)
    or by the std (mean/std) without adding bias.
    """
    data = np.asarray(data)
    space = _space_or_refuse(config)
    if space == "REAL":
        _record("CONTRACT", "REAL", inspected=False,
                why="the configuration declares these differences are already in price units")
        return data

    norm_json = _load_norm_json(config)
    if not isinstance(norm_json, dict):
        _record("NO_NORMALIZATION_DECLARED", None, inspected=False,
                why="the configuration names no normalization to invert")
        return data

    key = _select_norm_key(norm_json, config)
    if not key:
        _record("NO_NORMALIZATION_DECLARED", None, inspected=False,
                why="the normalization declares no entry for the target column or CLOSE")
        return data

    try:
        info = norm_json[key]
        if space == "NORMALIZED":
            _record("CONTRACT", "NORMALIZED", inspected=False,
                    why="the configuration declares these differences are in normalized units")
            if "min" in info and "max" in info:
                lo, hi = float(info["min"]), float(info["max"])
                return data if hi == lo else data*(hi-lo)
            if "mean" in info and "std" in info:
                std = float(info["std"])
                return data if std == 0 else data*std
            return data
        if "min" in info and "max" in info:
            close_min = float(info["min"])
            close_max = float(info["max"])
            diff = close_max - close_min
            if diff == 0:
                return data
            # For deltas, min/max scaling is linear; assume normalized deltas are in [0,1] scale.
            # Avoid scaling if deltas already look like real-world magnitude relative to diff.
            finite = data[np.isfinite(data)]
            if finite.size >= 32 and float(np.std(finite)) < 0.05 * abs(diff):
                return data
            return data * diff

        if "mean" in info and "std" in info:
            std = float(info["std"])
            if std == 0:
                return data
            finite = data[np.isfinite(data)]
            if finite.size < 32:
                return data
            # If deltas are in normalized units, their std will be closer to 1 than to the real std.
            a_std = float(np.std(finite))
            if abs(a_std - 1.0) < abs(a_std - std):
                return data * std
            return data

        return data
    except Exception as e:
        print(f"WARN: Error during denormalize_returns: {e}")
        return data
