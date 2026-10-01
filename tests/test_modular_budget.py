"""Measured-shape budget and named refusals (source-transform addendum 256c61a6, section 'manageable').

The engine reports, for any config, raw channels, branches, branch widths, fused width (the SUM of branch
widths), fused time, latent shape, materialization bytes per row and parameters, measured from the built
Keras graph and checked against the analytic prediction, so M04's budget model reads it instead of
recomputing. A cap that a config exceeds DEFERS the candidate with a named reason; nothing is truncated or
collapsed to fit. Synthetic component checks.
"""
import copy
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import tensorflow as tf

from predictor_plugins import modular_config as mc
from predictor_plugins import modular_temporal as mt


@pytest.fixture(autouse=True)
def seeded():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(4)
    yield


def cfg(f=3):
    return mt.default_config([f"f{i}" for i in range(f)])


def test_default_budget_is_measured_and_matches_the_analytic_prediction():
    b = mt.measure_budget(cfg(3))
    assert b["raw_channels"] == 3 and b["branches"] == 3 and b["branch_widths"] == [16, 16, 16]
    assert b["fused_width"] == 48 and b["fused_time"] == 24 and b["latent_shape"] == [6, 8]
    assert b["materialization_bytes_per_row"] == 24 * 48 * 4
    bundle = mt.build_modular(cfg(3))
    assert b["parameters"]["total"] == bundle.forecast_model.count_params()
    assert b["parameters"]["branches"] + b["parameters"]["core"] + b["parameters"]["head"] == b["parameters"]["total"]
    assert b["analytic_matches_measured"] is True and b["config_sha256"] == mt.config_digest(cfg(3))


def test_fused_width_is_the_sum_of_heterogeneous_branch_widths():
    c = cfg(3)
    for spec, width in zip(c["branches"], (4, 3, 2)):
        spec["params"] = {"channels": width}
    b = mt.measure_budget(c)
    assert b["branch_widths"] == [4, 3, 2] and b["fused_width"] == 9
    grouped = mt.default_config(["a", "b", "c", "d"])
    grouped["branches"] = [{"name": "g0", "features": ["a", "b", "c"]}, {"name": "g1", "features": ["d"]}]
    g = mt.measure_budget(grouped)
    assert g["raw_channels"] == 4 and g["branches"] == 2 and g["fused_width"] == 32


def test_ecl_scale_budget_matches_the_lane_a_identity_measurement():
    b = mt.measure_budget(cfg(321), build=False)
    assert b["fused_width"] == 5136 and b["materialization_bytes_per_row"] == 493056
    assert b["measured"] is False                                  # analytic only when asked not to build


def test_cap_overflow_defers_with_a_named_reason_and_never_truncates():
    c = cfg(3)
    c["budget_caps"] = {"max_fused_width": 40}
    with pytest.raises(mt.BudgetExceeded) as trouble:
        mt.build_modular(c)
    assert str(trouble.value).startswith("BUDGET_EXCEEDED_DEFERRED")
    assert trouble.value.dimension == "fused_width" and trouble.value.measured == 48 and trouble.value.cap == 40
    c["budget_caps"] = {"max_fused_width": 48, "max_materialization_bytes_per_row": 4608,
                        "max_parameters": 10 ** 7, "max_branches": 3}
    bundle = mt.build_modular(c)
    assert len(bundle.branch_models) == 3                           # nothing dropped to fit
    with pytest.raises(ValueError, match="budget_caps"):
        mt.build_modular(dict(c, budget_caps={"max_wobble": 1}))


def test_unrouted_declared_inputs_are_truncation_unless_excluded_with_reason():
    c = cfg(3)
    c["branches"] = c["branches"][:2]                              # f2 declared but routed nowhere
    with pytest.raises(ValueError, match="^INPUT_TRUNCATED"):
        mt.build_modular(c)
    c["excluded_features"] = {"f2": "DEFERRED: budget cap on fused width (named, reversible)"}
    b = mt.measure_budget(c)
    assert b["raw_channels"] == 3 and b["routed_channels"] == 2 and b["excluded_features"] == {
        "f2": "DEFERRED: budget cap on fused width (named, reversible)"}
    with pytest.raises(ValueError, match="excluded"):
        mt.build_modular(dict(c, excluded_features={"f0": "x"}))   # cannot exclude a routed input


def test_temporal_collapse_is_refused_by_name():
    c = cfg(3)
    c["output_steps"] = 1
    with pytest.raises(ValueError, match="^TEMPORAL_COLLAPSE"):
        mt.build_modular(c)


def test_budget_reaches_identity_and_grammar_without_changing_existing_digests():
    plain = cfg(3)
    before = mt.config_digest(plain)
    bundle = mt.build_modular(plain)
    assert bundle.component_manifests()["budget"]["fused_width"] == 48
    assert mt.config_digest(plain) == before                        # optional keys are not defaulted in
    capped = dict(copy.deepcopy(plain), budget_caps={"max_fused_width": 64},
                  excluded_features={})
    assert mc.unflatten(mc.flatten(capped)) == mt._normalize(capped)
    assert mc.budget(capped)["fused_width"] == 48
