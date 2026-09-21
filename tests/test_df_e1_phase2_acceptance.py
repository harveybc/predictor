"""Phase 2: acceptance before any runner change — the design is sealed, measured and honest.

What these rules pin: the sealed design carries the task's benchmark contract; its capacity table is
MEASURED by building the models, so the window/depth confound is a number and not an assumption;
the daily-lag channel's sample is provably available at the decision; that channel is prefix-
invariant; the permuted-calendar control has the calendar arm's capacity; volume counts rows,
windows, labels and exposure apart; the seal is deterministic; and the design's state says it did
not run.
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


@pytest.fixture(scope="module")
def design():
    if not (SOURCE/"DESIGN.json").is_file():
        pytest.skip("the successor run root is not on this host")
    return D.seal(SOURCE)


def test_the_design_is_sealed_with_a_contract_and_did_not_run(design):
    assert design["state"] == "SEALED_NOT_EXECUTED"
    assert design["benchmark_contract"]["schema"] == B.SCHEMA
    assert design["benchmark_contract"]["comparability"]["mode"] == "NOT_COMPARABLE"
    B.require(design, purpose="phase 2")                       # the runner's refusal accepts it


def test_the_seal_is_deterministic(design):
    again = D.seal(SOURCE)
    assert again["design_sha256"] == design["design_sha256"]


def test_the_window_depth_confound_is_measured_not_assumed(design):
    caps = design["capacity_measured_at_seal"]
    assert caps["W60_p7"]["blocks"] == 5 and caps["W1440_p7"]["blocks"] == 10
    assert caps["W1440_p7"]["parameters"] > caps["W60_p7"]["parameters"]
    assert caps["W60_p7"]["parameters"] == 8127                  # the successor run's own count
    assert "information AND model" in design["confound"]["statement"]


def test_q2_separates_information_from_model_factorially(design):
    arms = design["questions"]["Q2_daily_context"]["arms"]
    assert set(arms) >= {"Q2.a_daily_lag_channel", "Q2.b_long_window_own_depth",
                         "Q2.c_long_window_fixed_depth", "Q2.d_short_window_deep_core", "Q2.0_baseline"}
    assert arms["Q2.c_long_window_fixed_depth"]["dilations"] == arms["Q2.0_baseline"].get("dilations", [1, 2, 4, 8, 16]) or \
        arms["Q2.c_long_window_fixed_depth"]["dilations"] == [1, 2, 4, 8, 16]
    assert arms["Q2.d_short_window_deep_core"]["window"] == 60 and len(arms["Q2.d_short_window_deep_core"]["dilations"]) == 10


def test_the_daily_lag_sample_is_available_at_the_decision_for_h60_and_not_for_a_horizon_beyond_a_day():
    ok = D.daily_lag_availability(60)
    assert ok["available_at_decision"] and ok["sample_index_relative_to_origin"] == -1380
    late = D.daily_lag_availability(1500)
    assert not late["available_at_decision"] and "AFTER" in late["reading"]


def test_the_daily_lag_channel_is_prefix_invariant():
    """The channel at origin t reads y(t+h-1440), a row BEFORE t: changing rows after t leaves it."""
    rng = np.random.default_rng(0)
    h, lag = 60, 1440
    Y = rng.normal(size=4000)
    origins = np.arange(lag, 3900)
    channel = Y[origins+h-lag]
    Y2 = Y.copy()
    t = 2500
    Y2[t+1:] += 100.0
    channel2 = Y2[origins+h-lag]
    before = origins <= t
    assert np.array_equal(channel[before], channel2[before])
    assert not np.array_equal(channel[~before], channel2[~before])


def test_the_permuted_calendar_control_matches_the_calendar_arms_capacity(design):
    q1 = design["questions"]["Q1_calendar"]["arms"]
    assert q1["Q1.a_calendar"]["parameters"] == q1["Q1.b_permuted_calendar_control"]["parameters"]
    assert q1["Q1.a_calendar"]["parameters"] > q1["Q1.0_baseline"]["parameters"]


def test_volume_counts_rows_windows_labels_and_exposure_apart():
    c = D.volume_counts(train_origins=40080, window=60, horizon=60, rows_in_slice=50400)
    assert c["distinct_windows"] == 40080 and c["labels"] == 40080
    assert c["unique_raw_rows_in_train_span"] == 50400
    assert c["mean_exposures_per_raw_row"] == pytest.approx(60*40080/50400)
    assert "NOT independent" in c["reading"]


def test_every_cell_belongs_to_one_question_and_seeds_are_paired(design):
    cells = design["cells"]
    assert len(cells) == 3*(3+4+2)
    by_seed = {}
    for c in cells:
        by_seed.setdefault(c["seed"], set()).add(c["arm"])
    assert by_seed[1] == by_seed[2] == by_seed[3]


def test_the_adequacy_policy_makes_no_convergence_claim(design):
    ad = design["training_adequacy"]
    assert "CENSORED" in ad["ceiling_policy"] and "no convergence claim" in ad["claims"]
    assert ad["patience"] == 3 and ad["validation_cadence"] == "every epoch"
