"""BENCHMARK-CONTRACTS: comparability is decided from fields, and the real entry points refuse without one.

Rules the owner's addition of 2026-09-21 asks for, in PROGRAM_METRICS_CONTRACT_v1: a change of target,
horizon, split, scaler, metric formula or aggregation must reject comparability; a known affine
conversion must reproduce the metric; the real route must stop a new scientific training without a
contract; an unmatched published score is never a comparator.
"""
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"


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
    """Two contracts that are the SAME protocol: the reproduction lane."""
    ours = B.household_ours()
    theirs = B.BenchmarkContract(**{**asdict(ours), "task_id": "their-copy", "source": {"kind": "PRIMARY", "citation": "x"}})
    return ours, theirs


def test_identical_protocol_is_a_reproduction():
    ours, theirs = _twin()
    d = B.decide(ours, theirs)
    assert d["mode"] == "REPRODUCTION" and d["fields_that_differ"] == []


@pytest.mark.parametrize("field,value", [
    ("target", "Global_reactive_power"),
    ("horizon_steps", 96),
    ("split_rule", "last year test"),
    ("scaler_fit_population", "whole series"),
    ("metric_formula", "RMSE over steps"),
    ("metric_aggregation", "mean per series then concatenated"),
    ("target_transform", "log1p"),
    ("resolution_seconds", 900),
    ("missing_policy", "imputed by time-slot mean"),
])
def test_a_changed_identity_field_rejects_comparability(field, value):
    ours, theirs = _twin()
    theirs = B.BenchmarkContract(**{**asdict(theirs), field: value})
    d = B.decide(ours, theirs)
    assert d["mode"] == "NOT_COMPARABLE" and field in d["fields_that_differ"]
    assert "re-execution" in d["resolution"] or "MATCHED_DOMAIN_COMPARISON" in d["resolution"]


def test_a_reexecuted_reference_opens_the_matched_lane_but_a_published_score_does_not():
    ours, theirs = _twin()
    theirs = B.BenchmarkContract(**{**asdict(theirs), "horizon_steps": 96})
    assert B.decide(ours, theirs)["mode"] == "NOT_COMPARABLE"
    assert B.decide(ours, theirs, reexecuted_on_our_rows=True)["mode"] == "MATCHED_DOMAIN_COMPARISON"


def test_every_literature_source_in_the_registry_is_not_comparable_with_our_task_by_its_fields():
    reg = B.registry()
    for name, d in reg["decisions_against_household_W60_h60"].items():
        assert d["mode"] == "NOT_COMPARABLE", name
        assert d["fields_that_differ"], name
    # the registry names, for each, the resolution — never their number in our column
    assert all("re-execution" in d["resolution"] for d in reg["decisions_against_household_W60_h60"].values())


def test_the_registry_records_sources_read_at_origin_and_the_paywalled_one_as_recorded_not_used():
    reg = B.registry()
    g = reg["literature"]["gasparin_2019"]
    assert g["resolution_seconds"] == 900 and g["horizon_steps"] == 96 and g["naive_baseline"] is None
    assert "Table 5" in g["reference_method"]
    s = reg["literature"]["saad_saoud_2022"]
    assert "whole series" in s["scaler_fit_population"]
    k = reg["literature"]["kim_cho_2019"]
    assert k["source"]["kind"] == "PRIMARY_PAYWALLED" and k["reference_method"] is None


def test_an_affine_reexpression_reproduces_the_metric_and_refuses_what_it_cannot():
    z = B.affine_reexpression(0.5, from_transform={"kind": "identity"}, to_transform={"kind": "zscore", "sd": 2.0},
                              same_target=True, same_population=True, same_horizon=True)
    assert z["ok"] and z["value"] == pytest.approx(0.25)
    back = B.affine_reexpression(z["value"], from_transform={"kind": "zscore", "sd": 2.0}, to_transform={"kind": "identity"},
                                 same_target=True, same_population=True, same_horizon=True)
    assert back["value"] == pytest.approx(0.5)
    assert not B.affine_reexpression(0.5, from_transform={"kind": "log1p"}, to_transform={"kind": "identity"},
                                     same_target=True, same_population=True, same_horizon=True)["ok"]
    assert not B.affine_reexpression(0.5, from_transform={"kind": "zscore"}, to_transform={"kind": "identity"},
                                     same_target=True, same_population=True, same_horizon=True)["ok"]
    assert not B.affine_reexpression(0.5, from_transform={"kind": "identity"}, to_transform={"kind": "identity"},
                                     same_target=True, same_population=False, same_horizon=True)["ok"]


def test_require_refuses_a_design_without_a_contract_and_accepts_a_complete_one():
    with pytest.raises(B.ContractRefusal, match="without a benchmark contract"):
        B.require({"design_sha256": "x"})
    with pytest.raises(B.ContractRefusal, match="lacks"):
        B.require({"benchmark_contract": {"schema": B.SCHEMA, "task_id": "t"}})
    good = {"schema": B.SCHEMA, "task_id": "t", "dataset_id": "d", "target": "y", "horizon_steps": 60,
            "split_rule": "s", "target_transform": "zscore_train", "scaler_fit_population": "train",
            "metric_formula": "MAE", "naive_baseline": "persistence", "comparability": {"mode": "NOT_COMPARABLE"}}
    assert B.require({"benchmark_contract": good})["task_id"] == "t"
    with pytest.raises(B.ContractRefusal, match="decided from fields"):
        B.require({"benchmark_contract": {**good, "comparability": {"mode": "BEST_IN_CLASS"}}})


# --- the real entry points ---------------------------------------------------------------------------

def test_the_phase_runner_seals_a_contract_and_refuses_a_design_without_one(tmp_path):
    """df_e1_phase1: `seal` embeds the contract; `run` calls require() before anything is acquired."""
    P = _load("df_e1_phase1")
    source = Path("~/.local/state/crispdm-data-foundation/e1_household_successor_v3").expanduser()
    if not (source/"DESIGN.json").is_file():
        pytest.skip("the successor run root is not on this host")
    design = P.seal(source)
    assert design.get("benchmark_contract", {}).get("schema") == B.SCHEMA
    assert design["benchmark_contract"]["comparability"]["mode"] in B.MODES
    stripped = {k: v for k, v in design.items() if k != "benchmark_contract"}
    with pytest.raises(B.ContractRefusal, match="without a benchmark contract"):
        P.run(stripped, root=tmp_path/"r", run_id="x", source_run=source, gov_url="http://127.0.0.1:1",
              api_key_file=tmp_path/"none", lake="public_panels", resource="r", cost_pilot_only=True)


def test_the_factorial_runner_refuses_a_design_without_a_contract(tmp_path):
    """df_e1_huber.validate: the same refusal, in the runner Musashi executed."""
    H = _load("df_e1_huber")
    source = Path("~/.local/state/crispdm-data-foundation/e1_household_successor_v3").expanduser()
    if not (source/"DESIGN.json").is_file():
        pytest.skip("the successor run root is not on this host")
    design = H.seal(source)
    assert design.get("benchmark_contract", {}).get("schema") == B.SCHEMA
    stripped = {k: v for k, v in design.items() if k != "benchmark_contract"}
    stripped["design_sha256"] = H.P._module("df_mod_e0").sha_obj({k: v for k, v in stripped.items() if k != "design_sha256"})
    with pytest.raises(B.ContractRefusal, match="without a benchmark contract"):
        H.validate(stripped)
