"""Phase 2 successor (v2): the design is sealed, MEASURED and honest — before any runner change runs.

What these rules pin: the sealed design carries the task's typed benchmark contract; capacity AND
reach are measured by building the models on the full graph (the crop arm reaches 60, the
local-support arm 67, never assumed); the exact null has the W60 arm's parameters; the calendar
control has the calendar arm's capacity; cadence is in observed updates with equal checkpoint
opportunities; volume keeps the common scaler; the v1 draft is preserved; the seal is deterministic;
and the state says it did not run.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
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


D = _load("df_e1_phase2_design")
B = _load("df_benchmark_contract")
K = _load("df_e1_block")


@pytest.fixture(scope="module")
def design():
    if not (SOURCE/"DESIGN.json").is_file():
        pytest.skip("the successor run root is not on this host")
    return D.seal(SOURCE)


def test_the_design_is_sealed_with_a_contract_and_did_not_run(design):
    assert design["state"] == "SEALED_NOT_EXECUTED" and design["schema"] == D.SCHEMA
    assert design["benchmark_contract"]["schema"] == B.SCHEMA
    assert design["benchmark_contract"]["comparability"]["mode"] == "NOT_COMPARABLE"
    B.require(design, purpose="phase 2")
    assert all(b["state"] == "SEALED_NOT_EXECUTED" for b in design["blocks"].values())
    assert (HERE.parent/design["supersedes"]["v1"]).is_file() and design["supersedes"]["cells_v1"] == 27


def test_the_seal_is_deterministic(design):
    assert D.seal(SOURCE)["design_sha256"] == design["design_sha256"]


def test_capacity_and_reach_are_measured_on_the_full_graph(design):
    caps = design["capacity_and_reach_measured_at_seal"]
    assert caps["modular_w60"]["parameters"] == 8127 == caps["long_window_crop60"]["parameters"] == caps["long_window_local_support_67"]["parameters"]
    assert caps["long_window_own_depth"]["parameters"] > 8127 and len(caps["long_window_own_depth"]["dilations"]) == 10
    assert caps["short_window_deep_core"]["parameters"] == caps["long_window_own_depth"]["parameters"]
    assert caps["long_window_crop60"]["reach_measured"]["reach_by_gradient"] == 60
    assert caps["long_window_local_support_67"]["reach_measured"]["reach_by_gradient"] == 67
    assert caps["long_window_local_support_67"]["reach_measured"]["perturbation_max_abs_change_at_row"]["1373"] > 0      # row 1440-67
    assert caps["long_window_crop60"]["reach_measured"]["perturbation_max_abs_change_at_row"]["1379"] == 0.0            # row 1440-61
    assert caps["modular_w60"]["reach_measured"]["reach_by_gradient"] == 60
    assert caps["gru_adapted_w60"]["parameters"] == 8901 and caps["gru_adapted_w60"]["family"] == "gru"
    assert "NOT a null" in caps["long_window_local_support_67"]["role"] and "EXACT_INFORMATION_NULL" in caps["long_window_crop60"]["role"]


def test_the_calendar_control_matches_the_calendar_arms_capacity_and_is_declared_prefix_stable(design):
    caps = design["capacity_and_reach_measured_at_seal"]
    assert caps["calendar"]["parameters"] == caps["randomised_calendar_control"]["parameters"] > caps["modular_w60"]["parameters"]
    assert "prefix-stable" in design["calendar"]["control"] and "not declared zero" in design["calendar"]["control"]


def test_the_daily_lag_sample_is_available_at_the_decision_for_h60_and_not_for_a_horizon_beyond_a_day():
    ok = D.daily_lag_availability(60)
    assert ok["available_at_decision"] and ok["sample_index_relative_to_origin"] == -1380
    late = D.daily_lag_availability(1500)
    assert not late["available_at_decision"] and "AFTER" in late["reading"]
    assert "UNKNOWN is not zero delay" in ok["delayed_observation_rule"]


def test_cadence_is_in_observed_updates_with_equal_opportunities_and_the_ceiling_censors(design):
    ad, r = design["training_adequacy"], design["recipe"]
    assert r["validate_every_updates"] == 200 and r["patience_events"] == 3 and r["max_updates"] == 4000
    assert ad["checkpoint_opportunities"] == 20 == r["max_updates"]//r["validate_every_updates"]
    assert "wherever its best event fell" in ad["ceiling_policy"] and "no convergence claim" in ad["claims"]


def test_volume_keeps_the_common_scaler_and_counts_from_identities(design):
    v = design["volume"]
    assert v["scaler"].startswith("COMMON") and "NOT_RUN" in v["scaler"] and v["tiers_days"] == [28, 56, 112]
    assert "row identities" in v["counts"] and "predeclared" in v["extension"]


def test_every_cell_belongs_to_one_block_and_seeds_are_paired(design):
    cells = design["cells"]
    assert len(cells) == 3*(2+2+6+2)                       # Q2 carries its own modular_w60 baseline (common train intersection)
    for name, b in design["blocks"].items():
        by_seed = {}
        for c in b["cells"]:
            by_seed.setdefault(c["seed"], set()).add(c["arm"])
        assert by_seed[1] == by_seed[2] == by_seed[3] == set(b["arms"])
    assert design["execution_priority"] == ["DEV_MATCHED", "Q1_CALENDAR", "Q2_CONTEXT", "Q3_VOLUME"]
    assert design["blocks"]["Q2_CONTEXT"]["rows"]["pad_rows"] == 1380 and design["blocks"]["Q3_VOLUME"]["rows"]["widest_train_days"] == 112
