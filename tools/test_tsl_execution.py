#!/usr/bin/env python3
"""Tests of the RB02 EXECUTION generators: the evaluation-path derivation, the probe's verdict, the comparison row and the
closure table. The owner's standing rule is that a closure table is generated from artifacts and that the generator carries
its own tests, so every number the return publishes has a test that fails when the generator lies.

Nothing here needs a GPU, the author's clone or the benchmark bytes: the model-facing functions are exercised through
synthetic cell records that carry the same shape as the real ones. The tests that need the SEALED design read the retained
`DESIGN.weather.L96.json` / `CHARACTERIZATION.weather.json` and skip when they are not on this host.

    python -m pytest tools/test_tsl_execution.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import df_sota_repro as S                                              # noqa: E402
import df_tsl_execute as X                                             # noqa: E402
import df_tsl_repro as R                                               # noqa: E402

STATE = Path.home() / ".local/state/crispdm-data-foundation/tsl_weather_rb02_20260928"
DESIGN_PATH = STATE / "DESIGN.weather.L96.json"
CHARS_PATH = STATE / "CHARACTERIZATION.weather.json"

#: The whole-cgroup peak the TRAIN-only pilot measured (TRAIN_PILOT.weather.h96.json). Used only as the derivation baseline.
PILOT_BASELINE_BYTES = 1_465_098_240

#: The matched-reference delivery derived "~ 4.4 GiB including the 1.3 GiB baseline the pilot measured" for Weather h720.
DOCUMENTED_DERIVED_GIB = 4.4


def _design():
    if not DESIGN_PATH.is_file() or not CHARS_PATH.is_file():
        pytest.skip("the sealed Weather design is not retained on this host")
    return json.loads(DESIGN_PATH.read_text()), json.loads(CHARS_PATH.read_text())


def _cell_record(design, horizon: int, seed: int, mse: float, mae: float, naive_mse: float = 1.0,
                 naive_mae: float = 0.75) -> dict:
    """A cell record with the real record's shape and the real receipt built by the SEALED builder, so a row composed from it
    exercises the same code path a measured cell does."""
    _, chars = _design()
    key = f"L{design['seq_len']}_h{horizon}"
    sets = chars["sets"][key]
    windows, channels = int(sets["test"]["windows"]), int(sets["test"]["channels"])
    elements = windows * horizon * channels
    population = {"sha256": "a" * 64, "windows": windows, "target_channels": channels, "elements": elements}
    receipt = R.build_receipt(design=design, characterization=chars, horizon=horizon, seed=seed,
                              metrics={"mse": mse, "mae": mae}, naive={"mse": naive_mse, "mae": naive_mae},
                              population=population, model_commit=R.PINNED_COMMIT, scorer_sha256="b" * 64,
                              campaign_key="k", campaign_sha256="c" * 64, unit_id=f"u{horizon}_{seed}",
                              delivery_id="d" * 32, availability_contract_sha256="e" * 64,
                              costs={"wall_seconds": 1.0, "cpu_seconds": 1.0}, started_at="2026-09-29T00:00:00Z",
                              finished_at="2026-09-29T00:01:00Z", comparison_class="MATCHED_PUBLISHED_RECIPE_EXECUTED")
    rec = {"schema": X.CELL_SCHEMA, "design_sha256": design["design_sha256"], "protocol_sha256": R.protocol_sha256(design),
           "dataset": design["dataset"], "cell_id": f"weather_L96_h{horizon}_s{seed}", "horizon_steps": horizon,
           "horizon_seconds": R.horizon_seconds(design["dataset"], horizon), "seed": seed, "seq_len": design["seq_len"],
           "metric": {"author_float32": {"mse": mse, "mae": mae},
                      "independent_float64": {"mse": mse, "mae": mae},
                      "space": "z_train (the training-scaler normalized target space; --inverse False)",
                      "reduction": R.contract()["reduction"], "scorer": {"sha256": "b" * 64}},
           "naive": {"mse": naive_mse, "mae": naive_mae, "definition": "persistence",
                     "paired_on_the_same_rows_proved_by": {"target_population_sha256": "a" * 64,
                                                           "naive_target_sha256": "a" * 64, "equal": True}},
           "population": {**population, "predictions_sha256": "f" * 64, "all_finite": True,
                          "expected_from_characterization": {"windows": windows, "channels": channels,
                                                             "elements": elements}},
           "convergence": {"status": "EARLY_STOPPED_ON_VALIDATION", "epochs_run": 5, "early_stopped": True,
                           "best_epoch_by_validation": 2, "epochs_sealed": 10},
           "transport": {"class": X.TRANSPORT_CLASS, "delivery_id": "d" * 32, "campaign_key": "k",
                         "reading": X.TRANSPORT_READING},
           "resources": {"wall_seconds": 100.0, "cpu_seconds": 90.0, "peak_gpu_allocated_bytes": 1,
                         "whole_cgroup_peak_bytes_in_child": 2, "declared_cap_bytes": 8 << 30},
           "device_uuid_measured_inside_child": "GPU-test", "file_sha256": R.dataset_facts(design["dataset"])["sha256"],
           "receipt": receipt}
    rec["record_sha256"] = S.sha_obj(rec)
    return rec


# --- the DERIVED figure ---------------------------------------------------------------------------------------------------

def test_the_derivation_reproduces_the_documented_figure_and_is_labelled_derived():
    design, chars = _design()
    d = X.eval_path_memory_derivation(design, chars, 720, baseline_bytes=PILOT_BASELINE_BYTES)
    assert d["kind"] == "DERIVED_NOT_MEASURED"
    assert d["elements"] == 148_478_400, d["elements"]
    assert abs(d["derived_peak_gib"] - DOCUMENTED_DERIVED_GIB) <= 0.4, d["derived_peak_gib"]
    # every term is named and none of them is the answer on its own
    assert set(d["terms"]) == {"accumulated_lists", "concatenate_stage_peak", "retained_after_concatenate",
                               "metric_temporaries"}
    assert d["derived_peak_bytes"] > d["arrays_peak_bytes"] > d["terms"]["retained_after_concatenate"]["bytes"]


def test_the_derivation_grows_with_the_horizon_and_never_shrinks():
    design, chars = _design()
    peaks = [X.eval_path_memory_derivation(design, chars, h, baseline_bytes=0)["derived_peak_bytes"]
             for h in (96, 192, 336, 720)]
    assert peaks == sorted(peaks) and len(set(peaks)) == 4, peaks


def test_the_derivation_refuses_a_characterization_that_does_not_reconcile():
    design, chars = _design()
    bad = json.loads(json.dumps(chars))
    bad["sets"]["L96_h720"]["elements_test"] = 1
    with pytest.raises(SystemExit):
        X.eval_path_memory_derivation(design, bad, 720, baseline_bytes=0)


# --- the probe's verdict --------------------------------------------------------------------------------------------------

def test_the_probe_verdict_is_a_comparison_of_two_measured_numbers():
    assert X.probe_verdict(4 << 30, 8 << 30) == "ADMISSIBLE"
    assert X.probe_verdict(8 << 30, 8 << 30) == "ADMISSIBLE"
    assert X.probe_verdict((8 << 30) + 1, 8 << 30) == "CAPACITY_DEFICIT"
    assert X.probe_verdict(None, 8 << 30) == "UNDETERMINED"
    assert X.probe_verdict(4 << 30, None) == "UNDETERMINED"
    assert set(X.PROBE_VERDICTS) >= {"ADMISSIBLE", "CAPACITY_DEFICIT", "UNDETERMINED"}


def test_this_module_contains_no_way_to_shrink_the_authors_evaluation():
    """The order's hard constraint, asserted against the source: gate one may not be passed by altering what it measures."""
    src = (HERE / "df_tsl_execute.py").read_text()
    assert "exp.test(setting, test=0)" in src, "the probe must run the author's own test()"
    assert "bounded_test" not in src, "the bounded (chunked) adapter must not be reachable from the execution path"
    for forbidden in ("float16", "astype(np.float16", "::2]", "sub_sample", "subsample"):
        assert forbidden not in src, forbidden


def test_a_cell_refuses_to_run_behind_a_gate_that_did_not_pass(monkeypatch, tmp_path):
    design, chars = _design()
    probe = {"design_sha256": design["design_sha256"], "verdict": "CAPACITY_DEFICIT"}
    with pytest.raises(SystemExit) as e:
        X.run_scored_cell(design, chars, data_path=tmp_path / "nope.csv", work=tmp_path, horizon=720, seed=2021,
                          transport={}, probe=probe)
    assert "REFUSED" in str(e.value)


# --- the margin: this dataset's own, never the other dataset's ------------------------------------------------------------

def test_the_margin_is_weathers_own_and_not_electricitys():
    design, _ = _design()
    m = X.margin(design)["std_paper"]
    assert (m["mse"], m["mae"]) == (0.006, 0.004), m
    assert (S.AGREEMENT["std_paper"]["mse"], S.AGREEMENT["std_paper"]["mae"]) == (0.005, 0.006)
    assert m != S.AGREEMENT["std_paper"], "Electricity's margin is not Weather's"


def test_the_margin_is_read_from_the_design_so_editing_the_design_moves_the_class():
    design, _ = _design()
    wide = json.loads(json.dumps(design))
    wide["lock"]["agreement"]["std_paper"] = {"mse": 0.05, "mae": 0.05}
    values = [0.30, 0.30, 0.30]
    assert X.classify(design, values, 0.239, "mse")["class"] == "OUTSIDE_OPERATIONAL_MARGIN"
    assert X.classify(wide, values, 0.239, "mse")["class"] == "OPERATIONAL_AGREEMENT"


def test_the_class_boundaries_are_the_predeclared_ones():
    design, _ = _design()
    p, s, r = 0.239, 0.006, 0.0005
    inside = X.classify(design, [p + 2 * s + r - 1e-9] * 3, p, "mse")
    partial = X.classify(design, [p + 2 * s + r + 1e-6] * 3, p, "mse")
    outside = X.classify(design, [p + 3 * s + r + 1e-6] * 3, p, "mse")
    assert inside["class"] == "OPERATIONAL_AGREEMENT"
    assert partial["class"] == "OPERATIONAL_PARTIAL"
    assert outside["class"] == "OUTSIDE_OPERATIONAL_MARGIN"
    assert inside["margin_source"].startswith("Table 7")


def test_no_values_is_no_new_measurement_not_agreement():
    design, _ = _design()
    c = X.classify(design, [], 0.239, "mse")
    assert c["class"] == "NO_NEW_MEASUREMENT" and c["mean"] is None


def test_the_seed_dispersion_is_reported_beside_the_class_and_never_replaces_it():
    design, _ = _design()
    c = X.classify(design, [0.239, 0.239, 0.239], 0.239, "mse")
    assert c["class"] == "OPERATIONAL_AGREEMENT" and c["seed_sd_ddof1"] == 0.0 and c["n_seeds"] == 3
    spread = X.classify(design, [0.200, 0.239, 0.278], 0.239, "mse")
    assert spread["class"] == "OPERATIONAL_AGREEMENT" and spread["seed_sd_ddof1"] > 0.03
    assert "never replaces the criterion" in spread["reading"]


# --- the comparison row ---------------------------------------------------------------------------------------------------

def test_a_row_carries_the_full_comparison():
    design, _ = _design()
    rec = _cell_record(design, 96, 2021, mse=0.1600, mae=0.2050, naive_mse=0.8, naive_mae=0.5)
    row = X.comparison_row(design, rec)
    assert row["published"]["mse"] == 0.153 and row["published"]["mae"] == 0.199
    assert row["horizon_elapsed"] == "16 h" and row["horizon_seconds"] == 96 * 600
    assert abs(row["difference_vs_published"]["mse"] - 0.007) < 1e-9
    assert abs(row["skill_vs_naive"]["mse"] - (1 - 0.16 / 0.8)) < 1e-12
    assert row["paired_naive"]["same_rows"]["equal"] is True
    assert row["metric"]["space"].startswith("z_train")
    assert "arXiv:2501.13041" in row["published"]["source"]
    assert "not a claim" in row["published"]["not_current_sota"]
    assert row["comparability"]["class"] == "MATCHED_PUBLISHED_RECIPE_EXECUTED"
    assert "never cured by rescaling" in row["comparability"]["never"]
    for f in ("wall_seconds", "cpu_seconds", "peak_gpu_allocated_bytes", "whole_cgroup_peak_bytes"):
        assert row["resources"][f] is not None


def test_a_single_seed_row_does_not_carry_an_agreement_class():
    design, _ = _design()
    row = X.comparison_row(design, _cell_record(design, 96, 2021, 0.153, 0.199))
    assert "single_seed_not_a_class" in row["comparability"]
    assert "OPERATIONAL_AGREEMENT" not in json.dumps(row["comparability"])


def test_a_tampered_record_produces_no_row():
    design, _ = _design()
    rec = _cell_record(design, 96, 2021, 0.153, 0.199)
    rec["metric"]["author_float32"]["mse"] = 0.100                     # a better number, the digest left alone
    with pytest.raises(SystemExit):
        X.comparison_row(design, rec)


def test_a_record_from_another_design_produces_no_row():
    design, _ = _design()
    rec = _cell_record(design, 96, 2021, 0.153, 0.199)
    rec["design_sha256"] = "0" * 64
    rec["record_sha256"] = S.sha_obj({k: v for k, v in rec.items() if k != "record_sha256"})
    with pytest.raises(SystemExit):
        X.comparison_row(design, rec)


# --- the closure ----------------------------------------------------------------------------------------------------------

def _full_set(design, per_horizon_mse, per_horizon_mae, jitter=0.0):
    recs = []
    for h in (96, 192, 336, 720):
        for i, s in enumerate((2021, 2022, 2023)):
            recs.append(_cell_record(design, h, s, per_horizon_mse[h] + i * jitter, per_horizon_mae[h] + i * jitter))
    return recs


def test_the_closure_reproduces_the_published_row_when_the_cells_do():
    design, _ = _design()
    pub = design["lock"]["published"]["per_horizon"]
    mse = {int(h): v["mse"] for h, v in pub.items()}
    mae = {int(h): v["mae"] for h, v in pub.items()}
    clo = X.closure(design, _full_set(design, mse, mae))
    assert clo["cells_present"] == 12 and clo["missing"] == []
    for h in ("96", "192", "336", "720"):
        assert clo["per_horizon"][h]["mse"]["class"] == "OPERATIONAL_AGREEMENT"
        assert clo["per_horizon"][h]["mse"]["difference"] == pytest.approx(0.0, abs=1e-12)
    a = clo["four_horizon_average"]
    assert a["mse"]["mean"] == pytest.approx(0.239, abs=5e-4) and a["mae"]["mean"] == pytest.approx(0.269, abs=5e-4)
    assert a["mse"]["class"] == "OPERATIONAL_AGREEMENT"


def test_the_four_horizon_average_is_formed_within_each_seed_first():
    design, _ = _design()
    pub = design["lock"]["published"]["per_horizon"]
    mse = {int(h): v["mse"] for h, v in pub.items()}
    mae = {int(h): v["mae"] for h, v in pub.items()}
    clo = X.closure(design, _full_set(design, mse, mae, jitter=0.01))
    a = clo["four_horizon_average"]
    assert sorted(a["within_seed_averages"]) == ["2021", "2022", "2023"]
    # the dispersion is over THREE within-seed averages, not over the twelve cells
    assert a["mse"]["n_seeds"] == 3
    within = [v["mse"] for v in a["within_seed_averages"].values()]
    assert a["mse"]["seed_sd_ddof1"] == pytest.approx(float(np.std(within, ddof=1)))
    twelve = [r["metric"]["author_float32"]["mse"] for r in _full_set(design, mse, mae, jitter=0.01)]
    assert a["mse"]["seed_sd_ddof1"] != pytest.approx(float(np.std(twelve, ddof=1)))


def test_an_incomplete_closure_names_what_is_missing_and_still_refuses_to_average():
    design, _ = _design()
    pub = design["lock"]["published"]["per_horizon"]
    mse = {int(h): v["mse"] for h, v in pub.items()}
    mae = {int(h): v["mae"] for h, v in pub.items()}
    recs = [r for r in _full_set(design, mse, mae) if not (r["horizon_steps"] == 720 and r["seed"] == 2023)]
    clo = X.closure(design, recs)
    assert clo["cells_present"] == 11
    assert clo["missing"] == [{"horizon_steps": 720, "cells_present": 2, "cells_sealed": 3}]
    assert clo["four_horizon_average"]["mse"]["n_seeds"] == 2, "no seed's four-horizon average may be formed from three cells"


def test_the_closure_carries_gate_one_and_the_governance_class():
    design, _ = _design()
    pub = design["lock"]["published"]["per_horizon"]
    mse = {int(h): v["mse"] for h, v in pub.items()}
    mae = {int(h): v["mae"] for h, v in pub.items()}
    probe = {"verdict": "ADMISSIBLE", "record_sha256": "1" * 64, "declared_cap_bytes": 8 << 30,
             "measured": {"whole_cgroup_peak_bytes_in_child": 3 << 30},
             "derived_for_contrast": {"derived_peak_bytes": 4 << 30}}
    clo = X.closure(design, _full_set(design, mse, mae), probe=probe)
    assert clo["gate_one"]["verdict"] == "ADMISSIBLE"
    assert clo["gate_one"]["measured_whole_cgroup_peak_bytes"] == 3 << 30
    assert clo["evidence_classes"]["governance"] == X.TRANSPORT_CLASS
    assert "NOT_INDEPENDENTLY_VERIFIED" in clo["evidence_classes"]["model_error_and_paired_naive"]
    assert "not a claim of current best SOTA" in clo["evidence_classes"]["published_row"]
    assert X.closure(design, _full_set(design, mse, mae))["gate_one"]["verdict"] == "NOT_PRESENTED"


def test_the_markdown_table_is_generated_from_the_artifact_and_carries_the_margin():
    design, _ = _design()
    pub = design["lock"]["published"]["per_horizon"]
    mse = {int(h): v["mse"] for h, v in pub.items()}
    mae = {int(h): v["mae"] for h, v in pub.items()}
    md = X.markdown(X.closure(design, _full_set(design, mse, mae)))
    assert "0.153" in md and "0.199" in md and "0.239 ± 0.006" in md and "0.269 ± 0.004" in md
    assert "16 h" in md and "120 h" in md
    assert "naive MSE (same rows)" in md and "skill MSE" in md
    assert "Table 7" in md and "arXiv:2501.13041" in md
    assert X.TRANSPORT_CLASS in md


# --- governance, stated rather than invented ------------------------------------------------------------------------------

def test_the_transport_is_declared_and_refuses_to_be_invented():
    ok = {"campaign_key": "k", "campaign_sha256": "c" * 64, "delivery_id": "d" * 32,
          "availability_contract_sha256": "e" * 64, "verification_state": "VERIFIED_TRANSFER",
          "adoption_record_sha256": "f" * 64}
    t = X.transport_record(ok)
    assert t["class"] == "DECLARED_TRANSPORT_OF_GOVERNED_BYTES_NOT_A_NEW_GOVERNED_UNIT"
    assert "did not go looking for" in t["reading"]
    for field in ok:
        broken = {**ok, field: ""}
        with pytest.raises(SystemExit):
            X.transport_record(broken)
    with pytest.raises(SystemExit):
        X.transport_record({**ok, "verification_state": "UNVERIFIED"})


def test_a_receipt_built_by_this_module_passes_the_sealed_producer_gate():
    design, _ = _design()
    rec = _cell_record(design, 720, 2021, 0.3400, 0.3400)
    gate = R.validate_receipt(rec["receipt"])
    assert gate["dataset"] == "weather" and gate["metric_rows"] == 4
    assert gate["horizon_seconds"] == "432000", "Weather's 720 steps are 120 h, not 720 h"


def test_the_fixture_identity_the_warehouse_still_accepts_is_refused_here():
    """Carried forward, not fixed here: the deployed warehouse would accept these tags from any producer. The refusal lives
    on the producer side, and this test is the record that it does."""
    design, _ = _design()
    rec = _cell_record(design, 96, 2021, 0.153, 0.199)
    body = json.loads(json.dumps(rec["receipt"]))
    body["tags"]["protocol_sha256"] = "fixture"
    body.pop("terminal_sha256", None)
    with pytest.raises(SystemExit) as e:
        R.validate_receipt(body)
    assert "protocol_sha256" in str(e.value)
