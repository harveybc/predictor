#!/usr/bin/env python3
"""Tests for the Traffic TRAIN-only cost pilot's generators.

They test the parts that decide what a record is ALLOWED to say: the batch geometry, the clock, the verdict boundaries, the
refusal that stops one dataset's training footprint being applied to another, and the projection's labelling of measured
against derived terms and of headroom against a proven maximum. The measurement itself needs a GPU, the registered bytes and
an admitted cell scope, and is not simulated here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import df_sota_repro as S
import df_tsl_repro as R
import df_tsl_train_pilot as P

TRAFFIC_ROWS = 17544                                       # the registered resource's own row count
SEQ = 96
BATCH = 16


def design_for(dataset: str, horizons=(96, 720), sha: str = "d" * 64) -> dict:
    return {"dataset": dataset, "seq_len": SEQ, "design_sha256": sha, "seeds": [2021],
            "horizons": list(horizons), "lock": {"protocol_sha256": "p" * 64},
            "cells": [{"cell_id": f"{dataset}_L{SEQ}_h{h}_s2021", "horizon_steps": h, "seed": 2021,
                       "effective_args": {"batch_size": BATCH, "train_epochs": 30}} for h in horizons]}


def characterization_for(design: dict, rows: int = TRAFFIC_ROWS, channels: int = 862) -> dict:
    sets = {}
    for h in design["horizons"]:
        b = R.borders(rows, design["seq_len"], h)
        sets[f"L{design['seq_len']}_h{h}"] = {
            flag: {"windows": b["windows"][flag], "channels": channels} for flag in ("train", "vali", "test")}
    return {"design_sha256": design["design_sha256"], "sets": sets}


def pilot_record(design: dict, horizon: int, *, peak=3 << 30, cap=8 << 30, step=0.3, val_per_batch=0.1,
                 schema: str = P.PILOT_SCHEMA) -> dict:
    return {"schema": schema, "design_sha256": design["design_sha256"], "dataset": design["dataset"],
            "record_sha256": "r" * 64, "boot_id": "b" * 8, "device_uuid_measured_inside_child": "GPU-x",
            "clock": P.horizon_clock(design, horizon),
            "steps": {"timed": 30, "steady": 25, "steady_seconds_median": step},
            "validation": {"seconds_per_batch": val_per_batch, "batches": 10},
            "checkpoint": {"write_seconds": 0.5},
            "measured": {"whole_cgroup_peak_bytes_in_child": peak},
            "declared_cap_bytes": cap, "verdict": "MEASURED_WITHIN_CAP"}


# --- batch geometry: the author's drop_last=False is what is priced ---------------------------------------------------------

def test_final_batch_rows_is_the_remainder_when_drop_last_is_false():
    assert P.final_batch_rows(12089, 16) == 9
    assert P.final_batch_rows(1661, 16) == 13


def test_a_final_batch_is_full_size_when_the_population_divides_exactly():
    assert P.final_batch_rows(1600, 16) == 16


def test_drop_last_true_never_yields_a_partial_batch():
    assert P.final_batch_rows(12089, 16, drop_last=True) == 16


def test_an_empty_split_has_no_final_batch_and_a_zero_batch_size_is_refused():
    assert P.final_batch_rows(0, 16) == 0
    with pytest.raises(P.TrainPilotRefusal):
        P.final_batch_rows(10, 0)


def test_batch_counts_round_up_without_drop_last_and_down_with_it():
    assert P.n_batches(12089, 16) == 756
    assert P.n_batches(12089, 16, drop_last=True) == 755


def test_split_geometry_marks_the_test_split_as_counts_only_and_never_opened():
    d = design_for("traffic")
    g = P.split_geometry(d, characterization_for(d), 96)
    assert g["splits"]["test"]["opened_by_this_pilot"] is False
    assert "COUNTS ONLY" in g["splits"]["test"]["source"]
    assert g["splits"]["train"]["opened_by_this_pilot"] is True
    assert g["splits"]["vali"]["opened_by_this_pilot"] is True


def test_the_largest_final_batch_is_taken_over_the_splits_the_pilot_actually_opens():
    d = design_for("traffic")
    g = P.split_geometry(d, characterization_for(d), 96)
    opened = g["final_batch_rows_by_opened_split"]
    assert set(opened) == {"train", "vali"}
    assert g["largest_final_batch_rows_over_opened_splits"] == max(opened.values())
    # and it is a shape the steady loop never sees
    assert g["largest_final_batch_rows_over_opened_splits"] < g["batch_size"]


def test_the_geometry_records_the_authors_own_drop_last_setting():
    d = design_for("traffic")
    g = P.split_geometry(d, characterization_for(d), 96)
    assert g["drop_last"] is False
    assert "drop_last=False" in g["drop_last_source"]


# --- the clock: hourly steps are not sixteen-minute steps -------------------------------------------------------------------

def test_traffic_96_steps_are_96_hours():
    c = P.horizon_clock(design_for("traffic"), 96)
    assert c["step_seconds"] == 3600
    assert c["horizon_seconds"] == 345600
    assert c["horizon_hours"] == pytest.approx(96.0)
    assert c["horizon_days"] == pytest.approx(4.0)


def test_traffic_720_steps_are_thirty_days():
    c = P.horizon_clock(design_for("traffic"), 720)
    assert c["horizon_seconds"] == 2592000
    assert c["horizon_days"] == pytest.approx(30.0)


def test_the_same_step_count_is_not_the_same_elapsed_horizon_on_weather():
    t = P.horizon_clock(design_for("traffic"), 96)
    w = P.horizon_clock(design_for("weather"), 96)
    assert t["horizon_steps"] == w["horizon_steps"]
    assert t["horizon_hours"] == pytest.approx(96.0)
    assert w["horizon_hours"] == pytest.approx(16.0)
    assert t["horizon_seconds"] != w["horizon_seconds"]


# --- the verdict boundaries -------------------------------------------------------------------------------------------------

def _v(**kw):
    base = {"measured_peak_bytes": 1 << 30, "cap_bytes": 8 << 30, "optimizer_slot_bytes": 1 << 20,
            "stages_present": P.REQUIRED_STAGES, "test_split_opened": False}
    base.update(kw)
    return P.pilot_verdict(**base)


def test_a_complete_pilot_inside_its_cap_is_measured_within_cap():
    assert _v() == "MEASURED_WITHIN_CAP"


def test_the_cap_boundary_is_inclusive():
    assert _v(measured_peak_bytes=8 << 30, cap_bytes=8 << 30) == "MEASURED_WITHIN_CAP"
    assert _v(measured_peak_bytes=(8 << 30) + 1, cap_bytes=8 << 30) == "CAPACITY_DEFICIT"


def test_a_missing_peak_is_undetermined_and_never_success():
    assert _v(measured_peak_bytes=None) == "UNDETERMINED"
    assert _v(cap_bytes=None) == "UNDETERMINED"
    assert "UNKNOWN, never zero" in P.PILOT_VERDICTS["UNDETERMINED"]


def test_a_footprint_without_optimizer_slots_is_refused():
    assert _v(optimizer_slot_bytes=0) == "REFUSED_NO_OPTIMIZER_STATE"
    assert _v(optimizer_slot_bytes=None) == "REFUSED_NO_OPTIMIZER_STATE"


def test_a_missing_stage_is_refused_rather_than_reported_as_a_footprint():
    partial = tuple(s for s in P.REQUIRED_STAGES if s != "after_checkpoint_reload")
    assert _v(stages_present=partial) == "REFUSED_INCOMPLETE_STAGE_COVERAGE"


def test_every_required_stage_is_named_and_covers_the_ordered_obligations():
    for s in ("after_optimizer_build", "after_first_optimizer_step", "after_warmup_steps", "after_steady_steps",
              "after_final_batch_shapes", "after_checkpoint_write", "after_checkpoint_reload", "after_validation_pass"):
        assert s in P.REQUIRED_STAGES


def test_test_access_refuses_the_whole_record_whatever_else_is_true():
    assert _v(test_split_opened=True) == "REFUSED_TEST_ACCESS"
    assert _v(test_split_opened=True, measured_peak_bytes=None, optimizer_slot_bytes=0) == "REFUSED_TEST_ACCESS"


# --- the transfer that is forbidden -----------------------------------------------------------------------------------------

def test_a_weather_pilot_cannot_price_the_traffic_design():
    traffic = design_for("traffic")
    weather = design_for("weather", sha="w" * 64)
    with pytest.raises(P.TrainPilotRefusal):
        P.refuse_foreign_pilot(traffic, pilot_record(weather, 96))


def test_a_pilot_from_another_design_of_the_same_dataset_is_also_refused():
    traffic = design_for("traffic")
    other = design_for("traffic", sha="e" * 64)
    with pytest.raises(P.TrainPilotRefusal):
        P.refuse_foreign_pilot(traffic, pilot_record(other, 96))


def test_the_superseded_v1_schema_is_not_accepted_as_a_training_footprint():
    d = design_for("traffic")
    with pytest.raises(P.TrainPilotRefusal):
        P.refuse_foreign_pilot(d, pilot_record(d, 96, schema="df_tsl_train_pilot.v1"))
    assert "df_tsl_train_pilot.v1" in P.SUPERSEDES
    assert "optimizer-slot" in P.SUPERSEDES["df_tsl_train_pilot.v1"]
    assert "is a floor" in P.SUPERSEDES["df_tsl_train_pilot.v1"]


def test_a_matching_pilot_passes():
    d = design_for("traffic")
    P.refuse_foreign_pilot(d, pilot_record(d, 96))


# --- the projection ---------------------------------------------------------------------------------------------------------

def test_nothing_is_projected_from_no_measurement():
    d = design_for("traffic")
    with pytest.raises(P.TrainPilotRefusal):
        P.projection(d, characterization_for(d), [])


def test_the_projection_says_which_horizons_were_measured_and_which_were_not():
    d = design_for("traffic", horizons=(96, 192, 720))
    c = characterization_for(d)
    out = P.projection(d, c, [pilot_record(d, 96), pilot_record(d, 720)])
    assert out["horizons_measured"] == [96, 720]
    assert out["horizons_priced_by_interpolation"] == [192]
    by_h = {x["horizon_steps"]: x for x in out["cells"]}
    assert by_h[96]["per_step_seconds_class"].startswith("MEASURED")
    assert by_h[192]["per_step_seconds_class"].startswith("DERIVED")


def test_the_memory_line_is_a_projection_with_explicit_headroom_and_not_a_maximum():
    d = design_for("traffic")
    out = P.projection(d, characterization_for(d), [pilot_record(d, 96, peak=3 << 30)], headroom_fraction=0.25)
    m = out["memory"]
    assert m["class"] == "PROJECTION WITH EXPLICIT HEADROOM"
    assert m["headroom_fraction"] == 0.25
    assert m["projected_cap_bytes"] == int((3 << 30) * 1.25)
    assert "NOT a proof" in m["reading"]
    assert m["not_measured"]


def test_the_worst_measured_peak_governs_the_projection():
    d = design_for("traffic")
    out = P.projection(d, characterization_for(d), [pilot_record(d, 96, peak=2 << 30),
                                                    pilot_record(d, 720, peak=5 << 30)])
    assert out["memory"]["worst_measured_peak_bytes"] == 5 << 30


def test_an_unread_peak_does_not_become_zero_in_the_projection():
    d = design_for("traffic")
    out = P.projection(d, characterization_for(d), [pilot_record(d, 96, peak=None), pilot_record(d, 720, peak=4 << 30)])
    assert out["memory"]["measured_peak_bytes_by_horizon"][96] is None
    assert out["memory"]["worst_measured_peak_bytes"] == 4 << 30


def test_the_logging_test_pass_is_priced_as_derived_and_named_unpriced_in_memory():
    d = design_for("traffic")
    out = P.projection(d, characterization_for(d), [pilot_record(d, 96), pilot_record(d, 720)])
    for c in out["cells"]:
        assert "logging_test_pass_seconds_DERIVED" in c
        assert c["test_batches_counts_only"] > 0
    assert any("logging test pass" in t for t in out["unpriced_terms"])
    assert any("logging test pass" in t for t in out["memory"]["not_measured"])


def test_every_projected_cell_carries_its_own_clock():
    d = design_for("traffic", horizons=(96, 720))
    out = P.projection(d, characterization_for(d), [pilot_record(d, 96), pilot_record(d, 720)])
    by_h = {x["horizon_steps"]: x for x in out["cells"]}
    assert by_h[96]["clock"]["horizon_hours"] == pytest.approx(96.0)
    assert by_h[720]["clock"]["horizon_hours"] == pytest.approx(720.0)


def test_the_cell_total_is_the_sum_of_the_priced_cells():
    d = design_for("traffic")
    out = P.projection(d, characterization_for(d), [pilot_record(d, 96), pilot_record(d, 720)])
    assert out["seconds_total"] == pytest.approx(sum(c["cell_seconds"] for c in out["cells"]))
    assert out["hours_total"] == pytest.approx(out["seconds_total"] / 3600.0)


# --- the epoch budget is not an upper bound at the pinned commit ------------------------------------------------------------

@pytest.mark.skipif(not (S.AUTHOR_REPO / "utils/tools.py").is_file(), reason="the pinned author clone is not present here")
def test_early_stopping_cannot_fire_at_the_pinned_commit_so_the_epoch_budget_is_the_schedule():
    es = P.early_stopping_state(design_for("traffic"))
    assert es["early_stopping_can_fire"] is False
    assert "ACTUAL epoch count" in es["consequence"]
    assert es["checkpoint_writes_per_cell"] == "one per epoch"


@pytest.mark.skipif(not (S.AUTHOR_REPO / "utils/tools.py").is_file(), reason="the pinned author clone is not present here")
def test_the_projection_does_not_claim_early_stopping_shortens_what_cannot_stop_early():
    d = design_for("traffic")
    out = P.projection(d, characterization_for(d), [pilot_record(d, 96)])
    assert out["early_stopping"]["early_stopping_can_fire"] is False
    assert "not an upper bound" in out["time_reading"]


# --- what this module is not allowed to do, asserted against its own source -------------------------------------------------

SRC = (HERE / "df_tsl_train_pilot.py").read_text()


def test_the_pilot_refuses_the_test_split_in_its_own_source():
    assert 'if str(flag) == "test":' in SRC
    assert "does not open the test split" in SRC


def test_the_pilot_never_writes_a_cap_and_only_reads_the_one_it_was_admitted_under():
    assert "X.declared_cap_bytes()" in SRC
    for forbidden in ("cap = cap //", "cap = cap *", "cap = min(", "cap_bytes =", "MemoryMax"):
        assert forbidden not in SRC, forbidden


def test_the_pilot_records_no_validation_loss_value():
    assert '"loss_value_recorded": False' in SRC
    assert "the VALUE is discarded" in SRC


def test_the_sealed_modules_are_imported_and_not_redefined_here():
    assert "import df_sota_repro as S" in SRC and "import df_tsl_repro as R" in SRC
    for own in ("def _select_optimizer", "def vali(", "def data_provider", "def metric("):
        assert own not in SRC, own


def test_the_peak_scope_distinguishes_the_kernel_high_water_from_the_launchers_sampler():
    assert "not a sample" in SRC
    assert "child_shorter_than_sampler_interval" in SRC


# --- parity on TRAINED outputs, and what it is not ---------------------------------------------------------------------------

def test_the_parity_verdicts_keep_degenerate_outputs_apart_from_agreement():
    assert set(P.PARITY_VERDICTS) == {"BIT_EQUAL_ON_TRAINED_OUTPUTS", "REFUSED_PARITY_BROKEN", "REFUSED_FOREIGN_WEIGHTS",
                                      "REFUSED_DEGENERATE_OUTPUTS", "UNDETERMINED"}
    assert "vacuous" in P.PARITY_VERDICTS["REFUSED_DEGENERATE_OUTPUTS"] or \
           "prove nothing" in P.PARITY_VERDICTS["REFUSED_DEGENERATE_OUTPUTS"]


def test_the_parity_child_also_refuses_the_test_split():
    body = SRC.split("def trained_reducer_parity", 1)[1].split("\ndef ", 1)[0]
    assert 'if str(flag) == "test":' in body
    assert 'flag="val"' in body
    assert 'flag="test"' not in body


def test_the_parity_child_runs_both_reductions_from_other_peoples_code():
    body = SRC.split("def trained_reducer_parity", 1)[1].split("\ndef ", 1)[0]
    assert "S.author_metric_exact(" in body
    assert "MET.metric(" in body
    for own in ("np.mean(", "def _reduce", "chunk_mean"):
        assert own not in body, own


def test_the_parity_child_verifies_the_weights_are_the_cited_pilots():
    body = SRC.split("def trained_reducer_parity", 1)[1].split("\ndef ", 1)[0]
    assert 'pilot["checkpoint"]["sha256"]' in body
    assert "refuse_foreign_pilot(design, pilot)" in body


def test_the_parity_record_says_what_it_does_not_close():
    body = SRC.split("def trained_reducer_parity", 1)[1].split("\ndef ", 1)[0]
    assert "what_this_does_not_close" in body
    assert "row placement" in body


# --- the projection is a bracket, not a point --------------------------------------------------------------------------------

def test_the_projection_brackets_the_unmeasured_horizons_between_measured_rates():
    d = design_for("traffic", horizons=(96, 192, 336, 720))
    c = characterization_for(d)
    out = P.projection(d, c, [pilot_record(d, 96, step=0.123, val_per_batch=0.058),
                              pilot_record(d, 720, step=0.153, val_per_batch=0.180)])
    b = out["bracket"]
    assert b["hours_lower"] < b["hours_upper"]
    assert b["lower_prices_unmeasured_horizons_at_horizon"] == 96
    assert b["upper_prices_unmeasured_horizons_at_horizon"] == 720
    assert "monotone" in b["assumption"]
    assert b["class"] == "DERIVED BRACKET over MEASURED endpoints"
    assert b["hours_lower"] <= out["hours_total"] <= b["hours_upper"]
    assert "not a best estimate" in out["hours_total_class"]


def test_a_fully_measured_design_has_a_degenerate_bracket():
    d = design_for("traffic", horizons=(96, 720))
    out = P.projection(d, characterization_for(d), [pilot_record(d, 96, step=0.123), pilot_record(d, 720, step=0.153)])
    assert out["bracket"]["hours_lower"] == pytest.approx(out["bracket"]["hours_upper"])
    assert out["horizons_priced_by_interpolation"] == []
