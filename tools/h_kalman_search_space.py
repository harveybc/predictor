"""Lane H: DOIN hierarchical search-space declaration of the causal Kalman operator (DESIGN + executable checks).

Parameters are CONDITIONAL on the operator being active, so DOIN never explores the useless Cartesian product:

    kalman.active        {0, 1}                                   always present
    kalman.group         {level, slope, both}                     only if active   (lane B declared groups; the
                                                                  operator never selects features inside DOIN)
    kalman.param_source  {moments_train, declared_ratio}          only if active
    kalman.ratio_level   {1e-4, 1e-3, 1e-2, 1e-1, 1.0}            only if active and param_source == declared_ratio
    kalman.ratio_slope   {1e-6, 1e-4, 1e-2}                       only if active, declared_ratio and group in {slope, both}
    kalman.mode          {append, replace}                        only if active  (arm B vs arm C)

``level`` applies the local level model to the lane B LOCAL_LEVEL group, ``slope`` the local linear trend to the
LEVEL_PLUS_SLOPE group, ``both`` applies each to its group. Q/R are fitted ONLY on TRAIN (closed form) or fixed by the
declared ratios; the backward smoother is never selectable (``smoother: NEVER``).

Phase 1 (best bounded variant first): the space is {inactive, PROMOTED_VARIANT}, measured inside the differentiated
model. Phase 2 opens the full conditional space only if the phase 1 effect and the measured cost justify it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kf = _load("df_kalman_family")

SCHEMA = "h_kalman.search_space.v1"
SPACE = {
    "params": {
        "kalman.active": {"type": "int", "values": [0, 1], "active_if": None},
        "kalman.group": {"type": "str", "values": ["level", "slope", "both"], "active_if": "active"},
        "kalman.param_source": {"type": "str", "values": ["moments_train", "declared_ratio"], "active_if": "active"},
        "kalman.ratio_level": {"type": "float", "values": [0.0001, 0.001, 0.01, 0.1, 1.0],
                               "active_if": "active and param_source == declared_ratio"},
        "kalman.ratio_slope": {"type": "float", "values": [1e-06, 0.0001, 0.01],
                               "active_if": "active and param_source == declared_ratio and group in (slope, both)"},
        "kalman.mode": {"type": "str", "values": ["append", "replace"], "active_if": "active"},
    },
    "order": ["kalman.active", "kalman.group", "kalman.param_source", "kalman.ratio_level", "kalman.ratio_slope",
              "kalman.mode"],
}
# PROMOTED_VARIANT: the bounded variant measured first. Chosen from the lane H ETH pilot (see the evidence record).
PROMOTED_VARIANT = {"kalman.active": 1, "kalman.group": "both", "kalman.param_source": "moments_train",
                    "kalman.mode": "append"}


class SearchSpaceError(ValueError):
    pass


def _is_active(name: str, flat: dict) -> bool:
    if name == "kalman.active":
        return True
    if flat.get("kalman.active") != 1:
        return False
    if name in ("kalman.group", "kalman.param_source", "kalman.mode"):
        return True
    declared = flat.get("kalman.param_source") == "declared_ratio"
    if name == "kalman.ratio_level":
        return declared
    if name == "kalman.ratio_slope":
        return declared and flat.get("kalman.group") in ("slope", "both")
    raise SearchSpaceError(f"UNKNOWN parameter {name}")


def validate_flat(flat: dict) -> dict:
    if not isinstance(flat, dict):
        raise SearchSpaceError("flat parameters must be a dict")
    for k in flat:
        if k not in SPACE["params"]:
            raise SearchSpaceError(f"UNKNOWN parameter {k}")
    for name in SPACE["order"]:
        spec = SPACE["params"][name]
        active = _is_active(name, flat) if name == "kalman.active" or "kalman.active" in flat else False
        present = name in flat
        if present and not active:
            raise SearchSpaceError(f"parameter {name} is INACTIVE for this configuration and must be absent")
        if active and not present:
            raise SearchSpaceError(f"parameter {name} is active but MISSING")
        if present:
            v = flat[name]
            typ = {"int": int, "float": float, "str": str}[spec["type"]]
            if type(v) is not typ:
                raise SearchSpaceError(f"parameter {name} must be exactly {spec['type']}")
            if v not in spec["values"]:
                raise SearchSpaceError(f"parameter {name}={v!r} is outside the declared grid {spec['values']}")
    return flat


def cartesian_size() -> int:
    n = 1
    for spec in SPACE["params"].values():
        n *= len(spec["values"])
    return n


def enumerate_space(phase: int = 2):
    if phase == 1:
        yield {"kalman.active": 0}
        yield dict(PROMOTED_VARIANT)
        return
    yield {"kalman.active": 0}
    P = SPACE["params"]
    for group, src, mode in itertools.product(P["kalman.group"]["values"], P["kalman.param_source"]["values"],
                                              P["kalman.mode"]["values"]):
        base = {"kalman.active": 1, "kalman.group": group, "kalman.param_source": src, "kalman.mode": mode}
        if src == "moments_train":
            yield base
            continue
        slopes = P["kalman.ratio_slope"]["values"] if group in ("slope", "both") else [None]
        for rl in P["kalman.ratio_level"]["values"]:
            for rs in slopes:
                f = dict(base)
                f["kalman.ratio_level"] = rl
                if rs is not None:
                    f["kalman.ratio_slope"] = rs
                yield f


def conditional_size() -> int:
    return len(list(enumerate_space(2)))


def to_plan(flat: dict) -> dict:
    validate_flat(flat)
    plan = {"schema": SCHEMA, "fit_role": "TRAIN", "causal": True, "smoother": "NEVER", "specs": {}, "groups": {},
            "mode": None}
    if flat["kalman.active"] == 0:
        return plan
    group, src = flat["kalman.group"], flat["kalman.param_source"]
    plan["mode"] = flat["kalman.mode"]
    kinds = {"level": ["local_level"], "slope": ["local_linear_trend"], "both": ["local_level", "local_linear_trend"]}[group]
    for g in kinds:
        kind = kf.LOCAL_LEVEL if g == "local_level" else kf.LOCAL_LINEAR_TREND
        over = {"param_source": src}
        if src == "declared_ratio":
            over["ratio_level"] = flat["kalman.ratio_level"]
            if g == "local_linear_trend":
                over["ratio_slope"] = flat["kalman.ratio_slope"]
        spec = kf.default_spec(kind, **over)
        kf.validate_spec(spec)
        plan["specs"][g] = spec
        plan["groups"][g] = f"lane_b_declared_{'level' if g == 'local_level' else 'slope'}_group"
    return plan


def from_plan(plan: dict) -> dict:
    if plan["mode"] is None:
        return {"kalman.active": 0}
    specs = plan["specs"]
    has_ll, has_llt = "local_level" in specs, "local_linear_trend" in specs
    group = "both" if has_ll and has_llt else ("level" if has_ll else "slope")
    any_spec = next(iter(specs.values()))
    src = any_spec["params"]["param_source"]
    flat = {"kalman.active": 1, "kalman.group": group, "kalman.param_source": src, "kalman.mode": plan["mode"]}
    if src == "declared_ratio":
        flat["kalman.ratio_level"] = any_spec["params"]["ratio_level"]
        if has_llt:
            flat["kalman.ratio_slope"] = specs["local_linear_trend"]["params"]["ratio_slope"]
    return flat


def design_digest() -> str:
    body = {"schema": SCHEMA, "space": SPACE, "promoted_variant": PROMOTED_VARIANT}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def declaration() -> dict:
    return {"schema": SCHEMA, "params": SPACE["params"], "order": SPACE["order"],
            "promoted_variant": PROMOTED_VARIANT, "phases": {"1": "inactive + promoted variant", "2": "full conditional space"},
            "sizes": {"cartesian": cartesian_size(), "conditional": conditional_size()},
            "invariants": {"fit_role": "TRAIN only", "causal": True, "smoother": "NEVER selectable",
                           "feature_selection_inside_doin": False, "design_label": "DEVELOPMENT"},
            "design_sha256": design_digest()}


if __name__ == "__main__":
    print(json.dumps(declaration(), indent=1, sort_keys=True))
