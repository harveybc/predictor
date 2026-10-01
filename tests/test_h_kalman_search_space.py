"""Lane H: the DOIN hierarchical search-space declaration of the Kalman operator (design + executable checks)."""
from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ss = _load("h_kalman_search_space")
kf = ss.kf


def full_flat(**over):
    flat = {"kalman.active": 1, "kalman.group": "both", "kalman.param_source": "declared_ratio",
            "kalman.ratio_level": 0.01, "kalman.ratio_slope": 1e-4, "kalman.mode": "append"}
    flat.update(over)
    return flat


def test_inactive_operator_carries_no_other_parameter():
    assert ss.validate_flat({"kalman.active": 0}) == {"kalman.active": 0}
    for extra in ("kalman.group", "kalman.ratio_level", "kalman.mode"):
        with pytest.raises(ss.SearchSpaceError, match="INACTIVE"):
            ss.validate_flat({"kalman.active": 0, extra: ss.SPACE["params"][extra]["values"][0]})


def test_conditional_parameters_are_present_exactly_when_active():
    ss.validate_flat(full_flat())
    with pytest.raises(ss.SearchSpaceError, match="MISSING"):
        ss.validate_flat({"kalman.active": 1})
    f = full_flat(**{"kalman.param_source": "moments_train"})
    with pytest.raises(ss.SearchSpaceError, match="INACTIVE"):
        ss.validate_flat(f)                                          # ratios are inactive under moments_train
    f.pop("kalman.ratio_level"); f.pop("kalman.ratio_slope")
    ss.validate_flat(f)
    lvl = full_flat(**{"kalman.group": "level"})
    with pytest.raises(ss.SearchSpaceError, match="INACTIVE"):
        ss.validate_flat(lvl)                                        # slope ratio inactive for the level group
    lvl.pop("kalman.ratio_slope")
    ss.validate_flat(lvl)


def test_values_outside_the_declared_grid_or_wrong_type_are_refused():
    for bad in ({"kalman.ratio_level": 0.05}, {"kalman.ratio_level": 1}, {"kalman.mode": "mix"},
                {"kalman.group": "all"}, {"kalman.active": True}, {"kalman.active": 2}):
        with pytest.raises(ss.SearchSpaceError):
            ss.validate_flat(full_flat(**bad))
    with pytest.raises(ss.SearchSpaceError, match="UNKNOWN"):
        ss.validate_flat(full_flat(**{"kalman.nope": 1}))


def test_every_grid_point_maps_to_valid_family_specs_and_is_deterministic():
    n = 0
    for flat in ss.enumerate_space():
        plan = ss.to_plan(flat)
        for spec in plan["specs"].values():
            kf.validate_spec(spec)
        assert ss.to_plan(flat) == plan
        assert ss.from_plan(plan) == flat                          # reversible
        n += 1
    assert n == ss.conditional_size()


def test_conditional_space_is_much_smaller_than_the_cartesian_product():
    cart = ss.cartesian_size()
    cond = ss.conditional_size()
    assert cond < cart and cart == 2 * 3 * 2 * 5 * 3 * 2
    # inactive 1; active: group x source x mode with ratios only under declared_ratio (slope ratio only for slope/both)
    assert cond == 1 + 2 * (6 + 16 + 16) == 77
    assert ss.conditional_size() == len(list(ss.enumerate_space()))


def test_phase_one_is_a_single_promoted_variant_and_phase_two_is_the_full_conditional_space():
    p1 = list(ss.enumerate_space(phase=1))
    assert p1 == [{"kalman.active": 0}, ss.PROMOTED_VARIANT]
    ss.validate_flat(ss.PROMOTED_VARIANT)
    assert len(list(ss.enumerate_space(phase=2))) == ss.conditional_size()


def test_plan_names_groups_but_never_selects_features_inside_doin():
    plan = ss.to_plan(full_flat())
    assert set(plan["groups"]) == {"local_level", "local_linear_trend"}
    assert plan["fit_role"] == "TRAIN" and plan["causal"] is True and plan["smoother"] == "NEVER"
    assert plan["mode"] == "append"
    inactive = ss.to_plan({"kalman.active": 0})
    assert inactive["specs"] == {} and inactive["groups"] == {}


def test_declaration_document_matches_the_executable_space(tmp_path):
    doc = ss.declaration()
    assert doc["schema"] == ss.SCHEMA and doc["params"] == ss.SPACE["params"]
    assert doc["sizes"] == {"cartesian": ss.cartesian_size(), "conditional": ss.conditional_size()}
    assert doc["design_sha256"] == ss.design_digest()
    json.dumps(doc, allow_nan=False)
