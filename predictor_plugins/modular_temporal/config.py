"""Configuration defaults, normalization, and fail-closed validation."""

import math
import re

from .common import _keys, _partition, _positive_int, _copy
from .registry import DEFAULTS

def _normalize(config):
    c = _copy(config)
    _keys(c, {"window", "sample_hours", "feature_names", "branches", "branch_steps",
              "core", "fusion", "head", "output_steps", "output_channels", "entry_point_groups",
              "horizons", "target_count"}, "config")
    for key, default in (("window", 24), ("output_steps", 6), ("output_channels", 8)):
        c[key] = _positive_int(c.get(key, default), key)
    c.setdefault("branch_steps", c["window"])
    c["branch_steps"] = _positive_int(c["branch_steps"], "branch_steps")
    if c["branch_steps"] != c["window"]:
        raise ValueError("Branch extractors preserve the input time dimension; branch_steps must equal window")
    c["target_count"] = _positive_int(c.get("target_count", 1), "target_count")
    c.setdefault("horizons", [1])
    if not isinstance(c["horizons"], list) or not c["horizons"]:
        raise ValueError("horizons must be a nonempty ordered list")
    for horizon in c["horizons"]:
        _positive_int(horizon, "horizon")
    if c["horizons"] != sorted(set(c["horizons"])):
        raise ValueError("horizons must be strictly increasing")
    period = c.get("sample_hours")
    if (isinstance(period, bool) or not isinstance(period, (int, float)) or not math.isfinite(period)
            or period <= 0 or period * c["window"] < 24):
        raise ValueError("Explicit sample_hours and at least 24h physical window are required")
    names = c.get("feature_names")
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("feature_names must be unique ordered strings")
    branches = c.get("branches")
    if not isinstance(branches, list) or not branches:
        raise ValueError("At least one branch is required")
    groups = c.setdefault("entry_point_groups", {})
    _keys(groups, {"branch", "core", "fusion", "head"}, "entry_point_groups")
    for role in ("branch", "core", "fusion", "head"):
        groups.setdefault(role, "modular." + role)
        if not isinstance(groups[role], str) or not groups[role]:
            raise ValueError("Entry point group must be a nonempty string")
    seen = set()
    for spec in branches:
        _keys(spec, {"name", "features", "plugin", "params", "regime", "donor"}, "branch")
        name, features = spec.get("name"), spec.get("features")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in seen:
            raise ValueError("Branch names must be unique valid identifiers")
        seen.add(name)
        if (not isinstance(features, list) or not features or any(f not in names for f in features)
                or len(set(features)) != len(features)):
            raise ValueError("Branch features must be unique members of feature_names")
    for role, specs in (("branch", branches), ("core", [c.setdefault("core", {})]),
                        ("fusion", [c.setdefault("fusion", {})]), ("head", [c.setdefault("head", {})])):
        for spec in specs:
            if role != "branch":
                _keys(spec, {"plugin", "params", "regime", "donor"} if role == "core" else {"plugin", "params"}, role)
            spec.setdefault("plugin", DEFAULTS["modular." + role])
            spec.setdefault("params", {})
            if not isinstance(spec["plugin"], str) or not isinstance(spec["params"], dict):
                raise ValueError("Plugin and params must be a string and dict")
            if role in ("branch", "core"):
                spec.setdefault("regime", "R0")
                spec.setdefault("donor", None)
                if spec["regime"] not in ("R0", "R1", "R2"):
                    raise ValueError("Regime must be R0, R1 or R2")
                if (spec["regime"] == "R0" and spec["donor"] is not None
                        or spec["regime"] != "R0" and not spec["donor"]):
                    raise ValueError("R0 forbids donors; R1/R2 require explicit donors")
    _partition(tuple(range(c["window"])), c["branch_steps"])
    _partition(tuple(range(c["branch_steps"])), c["output_steps"])
    return c


def default_config(feature_names):
    """Create the default hourly model configuration.

    Parameters
    ----------
    feature_names : sequence[str]
        Ordered names of input channels. Each channel becomes its own branch.

    Returns
    -------
    dict
        Validated configuration with a 24-hour window, 12 branch steps, a
        6-step by 8-channel core representation, and one-step forecast head.
        Branch and core regimes default to R0.

    Raises
    ------
    ValueError
        If the feature list is empty, duplicated, or contains invalid names.
    """
    return _normalize({"feature_names": list(feature_names), "sample_hours": 1,
                       "branches": [{"name": f"branch_{i}", "features": [feature]}
                                    for i, feature in enumerate(feature_names)]})
