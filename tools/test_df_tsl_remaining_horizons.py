"""Tests for Traffic's remaining horizons: what arithmetic settles, and what only a run can.

The audit's words are the specification: the 12-15 GPU-hour figure is a projection, not a bound
established by two endpoints.  So these tests demand that the batch geometry and the parameter-driven
bytes be reproduced EXACTLY at the two horizons somebody measured -- which is the only honest way to
show the derivation is right before applying it to the two nobody measured -- and that the per-step
time at an unmeasured horizon come back UNKNOWN with no number attached.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import df_tsl_remaining_horizons as H                                         # noqa: E402

# The sealed characterization's own counts, as published in
# docs/audits/work_plan/SATOSHI_TRAFFIC_TRAIN_PILOT_2026_09_29.md section 4 at 91a4c410.
PUBLISHED_BATCHES = {96: (756, 104), 192: (750, 98), 336: (741, 89), 720: (717, 65)}
# The per-epoch logging pass, published as seconds, divided by the measured validation rate: the
# test-loader batch count the projection used.  Recovered here so the arithmetic is checked on all
# three splits and not only the two the pilot opened.
PUBLISHED_TEST_BATCHES = {96: 214, 720: 175}


# --- the loader's arithmetic, checked where it can be checked ---------------------------------------

@pytest.mark.parametrize("h", sorted(PUBLISHED_BATCHES))
def test_the_loader_arithmetic_reproduces_the_sealed_counts_at_every_horizon(h):
    g = H.split_geometry(h)
    assert (g["splits"]["train"]["batches"], g["splits"]["vali"]["batches"]) == PUBLISHED_BATCHES[h]


@pytest.mark.parametrize("h", sorted(PUBLISHED_TEST_BATCHES))
def test_the_test_loader_count_also_reproduces(h):
    assert H.split_geometry(h)["splits"]["test"]["batches"] == PUBLISHED_TEST_BATCHES[h]


def test_the_geometry_at_an_unmeasured_horizon_is_still_exact_and_says_so():
    g = H.split_geometry(192)
    assert g["class"] == H.EXACT_LOADER
    assert g["splits"]["train"]["windows"] == 11993
    assert "no device" in g["basis"]


def test_the_windows_shrink_with_the_horizon_because_the_split_borders_are_fixed():
    w96, w720 = H.split_windows(96), H.split_windows(720)
    assert w96["train"] - w720["train"] == 720 - 96


# --- the declaration's algebra, checked against both measured records -------------------------------

def test_the_head_declares_the_parameter_slope_and_the_records_agree_exactly():
    slope = H.head_slope()["slope_params_per_output_step"]
    assert slope == 513                                        # d_model 512 x 1 patch, plus the bias
    for h, p in H.MEASURED_PILOTS.items():
        assert H.static_terms(h)["values"]["params"] == p["params"]


def test_pred_len_enters_the_model_in_exactly_one_place():
    assert "nn.Linear" in H.DESIGN["head_declaration"]
    assert "never sees pred_len" in H.DESIGN["pred_len_occurrences_in_the_model"]


@pytest.mark.parametrize("field", ["params", "gradient_bytes", "optimizer_slot_bytes", "checkpoint_bytes"])
def test_each_parameter_driven_law_reproduces_both_measured_records_to_the_byte(field):
    for h, p in H.MEASURED_PILOTS.items():
        assert H.static_terms(h)["values"][field] == p[field], field


def test_the_static_terms_at_the_unmeasured_horizons_are_exact_not_interpolated():
    for h in (192, 336):
        st = H.static_terms(h)
        assert st["class"] == H.EXACT_DECL
        assert st["measured_here"] is False
        assert st["values"]["params"] == 7_257_500 + 513 * h
        assert st["values"]["optimizer_slot_bytes"] == 2 * st["values"]["gradient_bytes"] + 184


def test_a_parameter_law_that_contradicted_the_head_would_be_refused(monkeypatch):
    monkeypatch.setitem(H.DESIGN, "d_model", 256)
    with pytest.raises(H.HorizonRefusal):
        H.static_terms(192)


def test_the_head_output_tensor_is_a_component_and_never_a_peak():
    a = H.head_activation_bytes(192)
    assert a["bytes_per_tensor"] == 16 * 862 * 192 * 4
    assert "NOT_A_PEAK" in a["class"]
    assert "never reported as one" in a["reading"]


# --- the time nobody measured ----------------------------------------------------------------------

@pytest.mark.parametrize("h", [192, 336])
def test_the_per_step_time_at_an_unmeasured_horizon_is_unknown_with_no_number(h):
    r = H.per_step_seconds(h)
    assert r["status"] == H.UNKNOWN
    assert r["seconds"] is None
    assert r["class"] == H.UNPRICED
    assert "monotone" in r["why"]


@pytest.mark.parametrize("h", [96, 720])
def test_the_per_step_time_where_it_was_measured_carries_its_record(h):
    r = H.per_step_seconds(h)
    assert r["status"] == H.MEASURED and r["seconds"] > 0
    assert r["record_sha256_prefix"]


@pytest.mark.parametrize("h", [192, 336])
def test_an_unmeasured_horizon_has_no_hours_and_borrows_none(h):
    c = H.cell_hours(h)
    assert c["cell_hours"] is None and c["epoch_seconds"] is None
    assert c["status"] == H.UNKNOWN
    assert "not filled in from another horizon" in c["why"]


def test_the_measured_horizons_keep_their_hours():
    for h in (96, 720):
        c = H.cell_hours(h)
        assert c["status"] == H.MEASURED and c["cell_hours"] > 0


def test_the_measured_hours_match_the_published_cell_hours_to_three_decimals():
    assert round(H.cell_hours(96)["cell_hours"], 3) == 0.930
    assert round(H.cell_hours(720)["cell_hours"], 3) == 1.277


# --- the total, and the withdrawn bracket -----------------------------------------------------------

def test_the_twelve_cell_total_is_unknown_and_the_old_figure_is_withdrawn_as_a_bound():
    p = H.price()
    assert p["twelve_cell_total"] is None
    assert p["twelve_cell_total_status"] == H.UNKNOWN
    assert p["superseded_figure"]["status"] == "WITHDRAWN AS A BOUND"
    assert "12.08 - 15.02" in p["superseded_figure"]["value"]


def test_the_priced_subtotal_covers_exactly_the_six_measured_cells():
    p = H.price()
    assert p["priced_cells"] == 6 and p["unpriced_cells"] == 6
    assert p["unpriced_horizons"] == [192, 336]
    expected = 3 * (H.cell_hours(96)["cell_hours"] + H.cell_hours(720)["cell_hours"])
    assert abs(p["priced_gpu_hours"] - expected) < 1e-9


def test_the_bracket_is_refused_unless_the_caller_names_the_assumption():
    with pytest.raises(H.HorizonRefusal):
        H.conditional_bracket()


def test_the_bracket_when_asked_for_carries_its_label_and_reproduces_the_old_numbers():
    b = H.conditional_bracket(assume_monotone_in_pred_len=True)
    assert b["class"] == "CONDITIONAL_ON_AN_UNPROVEN_ASSUMPTION"
    assert b["not_a_bound"] is True
    assert round(b["hours_lower"], 2) == 12.08 and round(b["hours_upper"], 2) == 15.02


def test_the_bracket_is_not_in_the_default_record():
    assert "conditional_bracket" not in H.price()


# --- the request, not a campaign --------------------------------------------------------------------

def test_the_minimal_request_is_two_short_children_and_says_it_holds_no_authority():
    r = H.minimal_request()
    assert r["children"] == 2 and r["horizons"] == [192, 336]
    assert r["requested_wall_seconds_total"] <= 600
    assert r["declared_cap_bytes"] == 8 * (1 << 30)
    assert "NOT HELD" in r["authority"]
    assert "no campaign" in r["what_it_does_not_buy"]


def test_the_request_never_lowers_the_sealed_cap():
    assert H.minimal_request()["declared_cap_bytes"] >= max(
        p["cgroup_peak_bytes"] for p in H.MEASURED_PILOTS.values())
    assert "never re-asked smaller" in H.minimal_request()["cap_rule"]
