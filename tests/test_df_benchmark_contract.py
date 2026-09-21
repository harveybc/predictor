"""BENCHMARK-CONTRACTS (RP67): typed contracts, comparability from fields, binding to the runtime.

Musashi's probes against 4ef9f71 are the red side of these rules: a scale change and a physical-time
change that stayed REPRODUCTION, a boolean that promoted a mismatched protocol, a null contract that
was accepted, and a foreign target/horizon with a re-digested outer design that the real factorial
validator let through. Each is refused here, and the green side proves the legitimate paths still
work: an identical protocol, a declared contrast, a closed reference run, an affine conversion.
"""
import copy
import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"
SOURCE = Path("~/.local/state/crispdm-data-foundation/e1_household_successor_v3").expanduser()


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load("df_benchmark_contract")


def _twin():
    ours = B.household_ours()
    theirs = replace(ours, task_id="their-copy", source={"kind": "PRIMARY", "citation": "x"})
    return ours, theirs


# --- Musashi's probes, now refused -------------------------------------------------------------------

def test_PROBE_a_scale_change_is_not_a_reproduction():
    ours, _ = _twin()
    d = B.decide(ours, replace(ours, metric_scale="native"))
    assert d["mode"] == "NOT_COMPARABLE" and "metric_scale" in d["fields_that_differ"]


def test_PROBE_a_physical_time_change_is_refused_as_invalid_or_non_comparable():
    ours, _ = _twin()
    inconsistent = replace(ours, horizon_seconds=72*3600)               # 60 steps of 60 s cannot be 72 h
    assert "physical time is inconsistent" in " ".join(inconsistent.validate())
    assert B.decide(ours, inconsistent)["mode"] == "NOT_COMPARABLE"
    consistent_72h = replace(ours, horizon_steps=72*60, horizon_seconds=72*3600)
    d = B.decide(ours, consistent_72h)
    assert d["mode"] == "NOT_COMPARABLE" and {"horizon_steps", "horizon_seconds"} <= set(d["fields_that_differ"])


def test_PROBE_the_boolean_authority_is_gone():
    ours, _ = _twin()
    with pytest.raises(TypeError):
        B.decide(ours, B.gasparin_2019(), reexecuted_on_our_rows=True)


def test_PROBE_a_null_contract_is_refused():
    empty = {k: None for k in B.REQUIRED_NONNULL}
    empty.update(schema=B.SCHEMA, comparability={"mode": "REPRODUCTION", "comparator_state": "NONE"})
    with pytest.raises(B.ContractRefusal):
        B.require({"benchmark_contract": empty})


def test_PROBE_unknown_placeholders_never_match_as_proof():
    k = B.kim_cho_2019()
    d = B.decide(k, replace(k, task_id="another-unknown-paper"))
    assert d["mode"] == "NOT_COMPARABLE" and "unknown identity fields" in d["why"] and d["reference_evidence"]["state"] == "NONE"


# --- typed validity ----------------------------------------------------------------------------------

@pytest.mark.parametrize("field,value,fragment", [
    ("resolution_seconds", 0, "positive integer"),
    ("horizon_steps", -1, "positive integer"),
    ("horizon_seconds", 3599, "physical time is inconsistent"),
    ("target_transform", "cube_root", "not one of"),
    ("metric_scale", "USD", "not one of"),
    ("target", "", "null or empty"),
    ("naive_baseline", None, "null or empty"),
])
def test_invalid_fields_are_named(field, value, fragment):
    ours = B.household_ours()
    bad = replace(ours, **{field: value})
    assert any(fragment in p for p in bad.validate()), bad.validate()


def test_a_zscore_contract_cannot_report_in_a_foreign_scale():
    ours = B.household_ours()
    assert any("z_train or in native" in p for p in replace(ours, metric_scale="log1p").validate())


def test_the_registry_contracts_of_ours_are_valid_and_every_source_is_not_comparable():
    reg = B.registry()
    assert reg["validity"]["household_W60_h60"] == [] and reg["validity"]["fx_eurusd_1h"] == []
    for name, d in reg["decisions_against_household_W60_h60"].items():
        assert d["mode"] == "NOT_COMPARABLE" and d["comparator_state"] == "NONE", name


# --- comparability from fields --------------------------------------------------------------------------

def test_identical_protocol_is_a_reproduction_without_a_comparator_until_a_run_exists():
    ours, theirs = _twin()
    d = B.decide(ours, theirs)
    assert d["mode"] == "REPRODUCTION" and d["fields_that_differ"] == [] and d["comparator_state"] == "NONE"


@pytest.mark.parametrize("field,value", [
    ("target", "Global_reactive_power"), ("horizon_steps", 96), ("split_rule", "last year test"),
    ("scaler_fit_population", "whole series"), ("metric_formula", "RMSE over steps"),
    ("metric_aggregation", "mean per series then concatenated"), ("target_transform", "log1p"),
    ("resolution_seconds", 900), ("missing_policy", "imputed by time-slot mean"), ("permitted_inputs", "target only"),
])
def test_a_changed_identity_field_rejects_comparability(field, value):
    ours, theirs = _twin()
    kw = {field: value}
    if field == "horizon_steps":
        kw["horizon_seconds"] = value*ours.resolution_seconds
    if field == "resolution_seconds":
        kw["horizon_seconds"] = value*ours.horizon_steps
    if field == "target_transform":
        kw["metric_scale"] = "log1p"
    theirs = replace(theirs, **kw)
    d = B.decide(ours, theirs)
    assert d["mode"] == "NOT_COMPARABLE" and field in d["fields_that_differ"]


def test_a_declared_contrast_under_one_estimand_stays_comparable_within_it():
    ours, theirs = _twin()
    ours = replace(ours, varying_factors=("permitted_inputs",), estimand="effect of calendar inputs at h60")
    theirs = replace(theirs, permitted_inputs="the 7 declared channels + 4 calendar channels",
                     varying_factors=("permitted_inputs",), estimand="effect of calendar inputs at h60")
    d = B.decide(ours, theirs)
    assert d["mode"] == "REPRODUCTION" and d["declared_contrast"] == ["permitted_inputs"]
    other = replace(theirs, estimand="something else")
    assert B.decide(ours, other)["mode"] == "NOT_COMPARABLE"


def test_the_matched_lane_needs_a_closed_reference_run_under_our_digest(tmp_path):
    ours, theirs = _twin()
    theirs = replace(theirs, horizon_steps=96, horizon_seconds=96*60)
    assert B.decide(ours, theirs, reference_run=tmp_path)["comparator_state"] == "PLANNED_REFERENCE"
    root = tmp_path/"ref"
    root.mkdir()
    block = ours.to_design_block(comparability={**B.decide(ours, theirs), "comparator_state": "NONE"})
    (root/"DESIGN.json").write_text(json.dumps({"design_sha256": "d"*64, "benchmark_contract": block}))
    (root/"TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": {"gru_s1": {"status": "COMPLETED"}}}))
    d = B.decide(ours, theirs, reference_run=root)
    assert d["mode"] == "MATCHED_DOMAIN_COMPARISON" and d["comparator_state"] == "VERIFIED_COMPARATOR"
    (root/"TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": {"gru_s1": {"status": "FAILED"}}}))
    assert B.decide(ours, theirs, reference_run=root)["comparator_state"] == "PLANNED_REFERENCE"


# --- affine re-expression ------------------------------------------------------------------------------

def test_affine_reexpression_reproduces_and_refuses():
    ok = B.affine_reexpression(0.5, from_transform={"kind": "identity"}, to_transform={"kind": "zscore", "sd": 2.0},
                               same_target=True, same_population=True, same_horizon=True)
    assert ok["ok"] and ok["value"] == pytest.approx(0.25)
    for bad in ({"kind": "zscore", "sd": 0.0}, {"kind": "zscore", "sd": float("nan")}, {"kind": "log1p"}, {"kind": "zscore"}):
        assert not B.affine_reexpression(0.5, from_transform=bad, to_transform={"kind": "identity"},
                                         same_target=True, same_population=True, same_horizon=True)["ok"]
    assert not B.affine_reexpression(float("inf"), from_transform={"kind": "identity"}, to_transform={"kind": "identity"},
                                     same_target=True, same_population=True, same_horizon=True)["ok"]
    assert not B.affine_reexpression(0.5, from_transform={"kind": "identity"}, to_transform={"kind": "identity"},
                                     same_target=True, same_population=False, same_horizon=True)["ok"]


# --- require() and bind() ------------------------------------------------------------------------------

def _good_block():
    ours = B.household_ours()
    return ours.to_design_block(comparability={**B.decide(ours, B.gasparin_2019())})


def test_require_checks_digest_mode_and_decision_identity():
    block = _good_block()
    assert B.require({"benchmark_contract": block}).task_id == block["task_id"]
    stale = copy.deepcopy(block); stale["target"] = "Voltage"                       # edited, digest not recomputed
    with pytest.raises(B.ContractRefusal, match="does not recompute"):
        B.require({"benchmark_contract": stale})
    foreign = copy.deepcopy(block); foreign["target"] = "Voltage"
    foreign["contract_sha256"] = B.BenchmarkContract.from_block(foreign).sha256()   # consistently re-digested
    with pytest.raises(B.ContractRefusal, match="another contract"):               # but the decision was for ours
        B.require({"benchmark_contract": foreign})
    mode = copy.deepcopy(block); mode["comparability"]["mode"] = "BEST_IN_CLASS"
    with pytest.raises(B.ContractRefusal, match="decided from fields"):
        B.require({"benchmark_contract": mode})


def test_bind_refuses_a_contract_that_does_not_match_the_prepared_data():
    ours = B.household_ours()
    data = {"input_columns": ["a", "b", "Global_active_power"], "target_channel": 2, "horizon": 60, "window": 60,
            "panel_sha256": ours.source["panel_sha256"], "scaler": {"sd": [1.0, 1.0, ours.source["sd_train"]]},
            "enumerator": {"validation": {"admissible": 10020}}}
    assert B.bind(ours, data)["bound"]
    for change, key, value in (("target", "target_channel", 0), ("horizon", "horizon", 72),
                               ("window", "window", 1440), ("panel", "panel_sha256", "f"*64),
                               ("population", "enumerator", {"validation": {"admissible": 9000}})):
        bad = dict(data); bad[key] = value
        with pytest.raises(B.ContractRefusal, match="does not bind"):
            B.bind(ours, bad)
    degenerate = dict(data); degenerate["scaler"] = {"sd": [1.0, 1.0, 0.0]}
    with pytest.raises(B.ContractRefusal, match="finite positive sd"):
        B.bind(ours, degenerate)


# --- the real entry points, on the real design ------------------------------------------------------------

@pytest.mark.skipif(not (SOURCE/"DESIGN.json").is_file(), reason="the successor run root is not on this host")
def test_PROBE_the_real_factorial_validator_refuses_a_foreign_task_even_when_re_digested():
    H = _load("df_e1_huber")
    d = H.seal(SOURCE)
    H.validate(d)                                                            # the genuine design passes
    foreign = copy.deepcopy(d)
    foreign["benchmark_contract"]["target"] = "unrelated_target"
    foreign["benchmark_contract"]["horizon_steps"] = 72
    foreign["benchmark_contract"]["horizon_seconds"] = 72*60
    foreign["benchmark_contract"]["contract_sha256"] = B.BenchmarkContract.from_block(foreign["benchmark_contract"]).sha256()
    foreign["benchmark_contract"]["comparability"]["ours_sha256"] = foreign["benchmark_contract"]["contract_sha256"]
    foreign["design_sha256"] = H.P._module("df_mod_e0").sha_obj({k: v for k, v in foreign.items() if k != "design_sha256"})
    with pytest.raises(B.ContractRefusal, match="does not bind"):            # consistent everywhere, foreign to the data
        H.validate(foreign)
    stale = copy.deepcopy(d)
    stale["benchmark_contract"]["target"] = "unrelated_target"
    stale["design_sha256"] = H.P._module("df_mod_e0").sha_obj({k: v for k, v in stale.items() if k != "design_sha256"})
    with pytest.raises(B.ContractRefusal, match="does not recompute"):        # inner digest stale
        H.validate(stale)


@pytest.mark.skipif(not (SOURCE/"DESIGN.json").is_file(), reason="the successor run root is not on this host")
def test_the_phase_runner_binds_before_it_acquires(tmp_path):
    P = _load("df_e1_phase1")
    design = P.seal(SOURCE)
    assert design["benchmark_contract"]["schema"] == B.SCHEMA
    stripped = {k: v for k, v in design.items() if k != "benchmark_contract"}
    with pytest.raises(B.ContractRefusal, match="without a benchmark contract"):
        P.run(stripped, root=tmp_path/"r", run_id="x", source_run=SOURCE, gov_url="http://127.0.0.1:1",
              api_key_file=tmp_path/"none", lake="public_panels", resource="r", cost_pilot_only=True)
