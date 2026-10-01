"""Configuration defaults, normalization, and fail-closed validation."""

import math
import re

from .common import _keys, _partition, _positive_int, _copy
from .registry import DEFAULTS

CONFIG_SCHEMA = "predictor.modular.v1"


def _normalize(config):
    c = _copy(config)
    _keys(c, {"schema", "window", "sample_hours", "feature_names", "branches", "branch_steps",
              "core", "fusion", "head", "output_steps", "output_channels", "entry_point_groups",
              "horizons", "target_count", "regime", "alignment_probe", "budget_caps",
              "excluded_features"}, "config")
    if c.setdefault("schema", CONFIG_SCHEMA) != CONFIG_SCHEMA:
        raise ValueError(f"Unsupported modular config schema {c['schema']!r}; expected {CONFIG_SCHEMA}")
    if c.setdefault("alignment_probe", True) is not True and c["alignment_probe"] is not False:
        raise ValueError("alignment_probe must be boolean")
    common = c.setdefault("regime", None)
    if common not in (None, "R0", "R1", "R2"):
        raise ValueError("Common regime must be null, R0, R1 or R2")
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
                if common is not None and spec.get("regime", common) != common:
                    raise ValueError("A component regime contradicts the declared common regime; "
                                     "set regime=null to declare a mixed-regime run")
                spec.setdefault("regime", common or "R0")
                spec.setdefault("donor", None)
                if spec["regime"] not in ("R0", "R1", "R2"):
                    raise ValueError("Regime must be R0, R1 or R2")
                if (spec["regime"] == "R0" and spec["donor"] is not None
                        or spec["regime"] != "R0" and not spec["donor"]):
                    raise ValueError("R0 forbids donors; R1/R2 require explicit donors")
    _partition(tuple(range(c["window"])), c["branch_steps"])
    _partition(tuple(range(c["branch_steps"])), c["output_steps"])
    if c["output_steps"] == 1 and c["window"] > 1:
        raise ValueError("TEMPORAL_COLLAPSE: output_steps=1 collapses the whole window into one step; "
                         "the latent must keep a temporal axis (the head may flatten the completed latent)")
    _check_budget_caps(c)
    routed = {f for spec in branches for f in spec["features"]}
    excluded = c.get("excluded_features", {})
    if not isinstance(excluded, dict) or any(not isinstance(v, str) or not v.strip() for v in excluded.values()):
        raise ValueError("excluded_features maps each excluded input to a nonempty reason")
    if set(excluded) - set(names) or set(excluded) & routed:
        raise ValueError("excluded_features must name declared inputs that are not routed to a branch")
    unrouted = [f for f in names if f not in routed and f not in excluded]
    # Under a declared budget cap an unrouted input is truncation-to-fit and is refused by name. Without
    # caps a sub-bundle (e.g. one-branch pretraining isolation) may route a subset; the budget then reports
    # the unrouted inputs explicitly, so nothing disappears silently.
    if unrouted and c.get("budget_caps"):
        raise ValueError(f"INPUT_TRUNCATED: declared inputs {unrouted} reach no branch and are not in "
                         "excluded_features with a reason; inputs are never dropped silently")
    return c


BUDGET_CAPS = ("max_branches", "max_fused_width", "max_materialization_bytes_per_row", "max_parameters")


def _check_budget_caps(c):
    caps = c.get("budget_caps")
    if caps is None:
        return
    if not isinstance(caps, dict) or set(caps) - set(BUDGET_CAPS):
        raise ValueError(f"budget_caps keys must be within {list(BUDGET_CAPS)}")
    for key, value in caps.items():
        _positive_int(value, key)


def default_config(feature_names):
    """Create the default hourly model configuration.

    Parameters
    ----------
    feature_names : sequence[str]
        Ordered names of input channels. Each channel becomes its own branch.

    Returns
    -------
    dict
        Validated configuration with a 24-hour window, branches that keep all
        24 steps, a 6-step by 8-channel core representation, and a one-step
        forecast head.
        Branch and core regimes default to R0.

    Raises
    ------
    ValueError
        If the feature list is empty, duplicated, or contains invalid names.
    """
    return _normalize({"feature_names": list(feature_names), "sample_hours": 1,
                       "branches": [{"name": f"branch_{i}", "features": [feature]}
                                    for i, feature in enumerate(feature_names)]})


def regime_summary(config):
    """The declared common regime, or MIXED with every component's regime."""
    c = _normalize(config)
    regimes = {"branch:" + b["name"]: b["regime"] for b in c["branches"]}
    regimes["core"] = c["core"]["regime"]
    values = set(regimes.values())
    if c["regime"] is not None:
        return {"common": c["regime"], "components": regimes}
    return {"common": values.pop() if len(values) == 1 else "MIXED", "declared": False,
            "components": regimes}
