#!/usr/bin/env python3
"""Tests of the RB02 AUDIT generators: the identity reconciliation, the independently invoked replay's comparison, the
four-status closure, the custody reconciliation and the gap diagnostics.

The owner's standing rule is that a closure table is generated from artifacts and that the generator carries its own
tests. These tests are written so that they FAIL if the generator lies in any of the specific ways this audit could most
easily lie: by merging a bitwise result with a tolerance result, by letting one green status carry a red one, by
widening a frozen tolerance, by promoting a retained receipt to a governed unit, by concluding absence it never looked
for, or by adopting a reduction variant because it flatters the number.

Nothing here needs a GPU, the author's clone or the benchmark bytes. The tests that need the SEALED design read the
retained design and skip when it is not on this host.

    python -m pytest tools/test_tsl_replay.py -q
"""
from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import df_sota_repro as S                                              # noqa: E402
import df_tsl_replay as P                                              # noqa: E402
import df_tsl_gap_diagnostics_20260929 as G                            # noqa: E402
import df_tsl_custody_20260929 as C                                    # noqa: E402

STATE = Path.home() / ".local/state/crispdm-data-foundation/tsl_weather_rb02_20260928"
DESIGN_PATH = STATE / "DESIGN.weather.L96.json"


def _design():
    if not DESIGN_PATH.is_file():
        pytest.skip("the sealed Weather design is not retained on this host")
    return json.loads(DESIGN_PATH.read_text())


def _record(mse=0.25, mae=0.30, pred_sha="a" * 64, true_sha="b" * 64, naive_mse=0.4, naive_mae=0.35):
    return {"cell_id": "weather_L96_h336_s2021", "horizon_steps": 336, "seed": 2021,
            "metric": {"author_float32": {"mse": mse, "mae": mae},
                       "independent_float64": {"mse": mse, "mae": mae},
                       "scorer": {"sha256": "c" * 64}},
            "naive": {"mse": naive_mse, "mae": naive_mae},
            "population": {"sha256": true_sha, "predictions_sha256": pred_sha, "windows": 10204,
                           "target_channels": 21, "elements": 10204 * 336 * 21}}


def _compare(**kw):
    rec = _record()
    base = dict(author={"mse": 0.25, "mae": 0.30}, independent={"mse": 0.25, "mae": 0.30},
                pred_sha="a" * 64, true_sha="b" * 64,
                naive={"naive": {"mse": 0.4, "mae": 0.35}}, shape=(10204, 336, 21), rule=P.replay_rule())
    base.update(kw)
    return P.compare_to_record(rec, **base)


# --- the frozen rule is read, never written here -----------------------------------------------------------------------

def test_the_replay_rule_is_the_one_frozen_elsewhere_and_is_not_redefined_here():
    rule = P.replay_rule()
    assert rule["atol"] == S.AGREEMENT["replay"]["atol"]
    assert rule["rtol"] == S.AGREEMENT["replay"]["rtol"]
    assert rule["device"] == S.AGREEMENT["replay"]["device"]


def test_this_module_contains_no_tolerance_of_its_own():
    """A literal atol/rtol here would be a tolerance this audit invented. The only numeric thresholds allowed are the
    rule's own stated 1e-5 / 1e-6 readbacks, and they must appear as comparisons, never as a redefinition of the rule."""
    src = inspect.getsource(P)
    assert "AGREEMENT[\"replay\"]" in src
    for forbidden in ("atol=", "rtol=", "atol =", "rtol ="):
        assert forbidden not in src, f"{forbidden!r} would be a tolerance defined by the audit itself"


def test_a_widened_tolerance_would_change_the_verdict_so_the_verdict_depends_on_the_rule():
    tight = _compare(author={"mse": 0.25 + 1e-4, "mae": 0.30})
    assert tight["frozen_tolerance"]["author_metric_within_1e-5"] is False
    assert tight["frozen_tolerance"]["met"] is False


# --- bitwise and tolerance are two registers and are never merged ---------------------------------------------------------

def test_bitwise_and_tolerance_are_separate_and_a_tolerance_pass_is_not_a_bitwise_pass():
    out = _compare(pred_sha="z" * 64, author={"mse": 0.25 + 1e-9, "mae": 0.30 + 1e-9})
    assert out["frozen_tolerance"]["met"] is True
    assert out["bitwise"]["all"] is False
    assert out["bitwise"]["predictions_bitwise_equal"] is False


def test_a_bitwise_pass_requires_every_digest_and_every_reduced_number():
    assert _compare()["bitwise"]["all"] is True
    assert _compare(true_sha="z" * 64)["bitwise"]["all"] is False
    assert _compare(naive={"naive": {"mse": 0.4 + 1e-9, "mae": 0.35}})["bitwise"]["all"] is False


def test_the_same_arrays_clause_is_reported_apart_from_the_replay_criterion():
    out = _compare(pred_sha="z" * 64, author={"mse": 0.25 + 1e-9, "mae": 0.30 + 1e-9})
    clause = out["frozen_tolerance"]["stricter_same_arrays_clause"]
    assert clause["author_float32_bitwise_equal"] is False
    assert out["frozen_tolerance"]["met"] is True, "a cross-device replay must not be failed by a same-arrays clause"


# --- the four statuses -----------------------------------------------------------------------------------------------------

def _rep(cell_id, horizon, seed, *, bitwise=True, tol=True, identity=True, device="cuda"):
    return {"cell_id": cell_id, "horizon_steps": horizon, "seed": seed, "device_requested": device, "device": device,
            "replayed_record_sha256": "d" * 64,
            "identity": {"recipe": {"design_agrees": identity, "protocol_agrees": identity, "author_files_agree": identity},
                         "transport": {"agrees": identity}, "scorer": {"agrees": identity},
                         "weights": {"agrees": identity, "selection_rule": {"agrees": identity}}},
            "comparison": {"bitwise": {"all": bitwise, "predictions_bitwise_equal": bitwise,
                                       "target_population_bitwise_equal": bitwise,
                                       "author_float32_mae_exactly_equal": bitwise,
                                       "author_float32_mse_exactly_equal": bitwise,
                                       "naive_mae_exactly_equal": bitwise, "naive_mse_exactly_equal": bitwise},
                           "frozen_tolerance": {"met": tol, "author_float32_delta": {"mse": 0.0, "mae": 0.0},
                                                "author_metric_within_1e-5": tol, "naive_delta": {"mse": 0.0, "mae": 0.0}}},
            "resources": {"wall_seconds": 1.0, "whole_cgroup_peak_bytes_in_child": 1, "declared_cap_bytes": 8 << 30}}


def _all_reps(design, **kw):
    return [_rep(c["cell_id"], c["horizon_steps"], c["seed"], **kw) for c in design["cells"]]


def test_the_closure_publishes_four_statuses_and_never_merges_them():
    design = _design()
    clo = P.closure(design, _all_reps(design), [])
    assert set(clo) >= {"numerical_agreement", "replay", "custody", "scientific"}
    assert clo["custody"]["status"] == "REPORTED_SEPARATELY"
    assert clo["scientific"]["status"] == "REPORTED_SEPARATELY"


def test_a_green_replay_does_not_carry_a_red_identity():
    design = _design()
    reps = _all_reps(design)
    reps[0]["identity"]["scorer"]["agrees"] = False
    clo = P.closure(design, reps, [])
    assert clo["replay"]["status"] == "BITWISE_REPRODUCED"
    assert clo["numerical_agreement"]["status"] == "NOT_ESTABLISHED"


def test_one_missing_cell_is_not_a_reproduced_campaign():
    design = _design()
    reps = _all_reps(design)[:-1]
    clo = P.closure(design, reps, [])
    assert clo["replay"]["status"] == "NOT_ESTABLISHED"
    assert clo["numerical_agreement"]["status"] == "NOT_ESTABLISHED"
    assert clo["numerical_agreement"]["missing"]


def test_within_tolerance_is_never_reported_as_bitwise():
    design = _design()
    clo = P.closure(design, _all_reps(design, bitwise=False, tol=True), [])
    assert clo["replay"]["status"] == "WITHIN_FROZEN_TOLERANCE_NOT_BITWISE"
    assert clo["replay"]["cells_bitwise"] == 0


def test_cross_device_rows_are_kept_out_of_the_primary_replay_status():
    design = _design()
    reps = _all_reps(design) + _all_reps(design, bitwise=False, device="cpu")
    clo = P.closure(design, reps, [], device="cuda")
    assert clo["replay"]["status"] == "BITWISE_REPRODUCED"
    assert clo["replay"]["cells_replayed"] == len(design["cells"])
    assert len(clo["cross_device_replays"]["rows"]) == len(design["cells"])


# --- the identities are RE-DERIVED, and tampering is detected ---------------------------------------------------------------

def test_the_recipe_digests_recompute_from_the_retained_design():
    design = _design()
    out = P.recipe_identity(design)
    assert out["design_agrees"] and out["protocol_agrees"]
    assert out["design_sha256_recomputed"] == design["design_sha256"]


def test_an_edited_design_no_longer_recomputes_its_own_digest():
    design = copy.deepcopy(_design())
    design["lock"]["agreement"]["std_paper"]["mse"] = 0.05        # a widened margin, smuggled into the design
    out = P.recipe_identity(design)
    assert out["design_agrees"] is False


def test_the_weights_identity_refuses_a_checkpoint_that_is_not_the_recorded_one(tmp_path):
    ckpt = tmp_path / "checkpoint.pth"
    ckpt.write_bytes(b"not the weights")
    out = P.weights_identity({"checkpoint_sha256": "f" * 64, "training": {}}, ckpt, None)
    assert out["agrees"] is False


def test_the_weights_identity_reparses_the_selection_rule_rather_than_trusting_the_record(tmp_path):
    ckpt = tmp_path / "checkpoint.pth"
    ckpt.write_bytes(b"weights")
    log = tmp_path / "author_stdout.log"
    log.write_text("Epoch: 1, Steps: 10 | Train Loss: 0.5 Vali Loss: 0.9 Test Loss: 0.4\n"
                   "Epoch: 2, Steps: 10 | Train Loss: 0.4 Vali Loss: 0.8 Test Loss: 0.3\n")
    record = {"checkpoint_sha256": S.sha_file(ckpt), "training": {"best_epoch_by_vali": 1, "epochs_run": 2,
                                                                  "early_stopped": False}}
    out = P.weights_identity(record, ckpt, log)
    assert out["selection_rule"]["best_epoch_by_vali_reparsed"] == 2
    assert out["selection_rule"]["agrees"] is False, "a record that disagrees with its own log must not pass"


def test_the_staging_refuses_bytes_that_do_not_match_the_record(tmp_path):
    src = tmp_path / "checkpoint.pth"
    src.write_bytes(b"weights")
    with pytest.raises(SystemExit):
        P.stage_checkpoint(src, tmp_path / "w", "setting", "0" * 64)


# --- nothing here trains, chunks, downcasts or opens a governed unit --------------------------------------------------------

def test_the_replay_never_trains():
    src = inspect.getsource(P.replay_cell)
    assert "train=False" in src and "train=True" not in src


def test_the_replay_contains_no_chunking_downcast_or_subsampling_of_the_scored_population():
    src = inspect.getsource(P)
    for forbidden in ("bounded=True", "memmap", "astype(np.float16", "::2]", "sample("):
        assert forbidden not in src


def test_no_audit_module_can_open_a_governed_unit():
    for module in (P, G, C):
        src = inspect.getsource(module)
        for forbidden in ("submit_campaign", "post_json", "governed_download", "TerminalOutbox", "build_receipt"):
            assert forbidden not in src, f"{module.__name__} must not be able to {forbidden}"


def test_the_replay_declares_itself_not_a_promotion():
    src = inspect.getsource(P.replay_cell)
    assert "NOT_A_GOVERNED_UNIT_AND_NOT_A_PROMOTION" in src


# --- custody -----------------------------------------------------------------------------------------------------------------

def test_the_credential_reference_never_carries_its_content(tmp_path):
    key = tmp_path / "predictor.key"
    key.write_text("a-secret-value-that-must-never-appear")
    key.chmod(0o600)
    out = C.credential_reference(key)
    blob = json.dumps(out)
    assert "a-secret-value" not in blob
    assert out["present"] is True and out["content_read_here"] is False
    assert out["mode"] == "0o600" and out["created_or_last_written_utc"]


def test_an_absent_credential_is_reported_as_absent_and_not_guessed(tmp_path):
    out = C.credential_reference(tmp_path / "nothing")
    assert out["present"] is False


def test_the_warehouse_reports_unreadable_rather_than_concluding_absence():
    out = C.warehouse_evidence("http://127.0.0.1:1", None, "a" * 64, ["u1"])
    assert out["state"] == "UNREADABLE"
    assert "absence is NOT concluded" in out["why"]


def test_the_chronology_places_the_delivery_before_the_fits_and_refuses_to_call_it_authorization_of_the_fits():
    records = [{"cell_id": "c1", "started_at": "2026-09-29T03:49:09Z", "finished_at": "2026-09-29T04:20:48Z"}]
    adoption = {"routes": {"weather": {"terminals": {"units": {"route-1": {"payload": {
        "started_at": "2026-09-29T02:01:21Z", "finished_at": "2026-09-29T02:01:21Z"}}}}}}}
    cred = {"created_or_last_written_utc": "2026-09-14T17:21:29Z"}
    out = C.chronology(records, adoption, cred)
    assert out["input_authorized_before_fitting"] is True
    assert out["any_scored_cell_governed_before_fitting"] is False
    assert "retrospective ingestion" in out["reading"]
    assert [e["class"] for e in out["events"]][:2] == ["PRECONDITION", "GOVERNED_DELIVERY_OF_THE_INPUT"]


def test_a_later_delivery_than_the_first_fit_is_not_reported_as_authorization_before_fitting():
    records = [{"cell_id": "c1", "started_at": "2026-09-29T01:00:00Z", "finished_at": "2026-09-29T01:30:00Z"}]
    adoption = {"routes": {"weather": {"terminals": {"units": {"route-1": {"payload": {
        "started_at": "2026-09-29T02:01:21Z", "finished_at": "2026-09-29T02:01:21Z"}}}}}}}
    out = C.chronology(records, adoption, {"created_or_last_written_utc": None})
    assert out["input_authorized_before_fitting"] is False


# --- the gap diagnostics ------------------------------------------------------------------------------------------------------

def test_the_reduction_variants_are_predeclared_and_the_sealed_one_is_among_them():
    assert G.VARIANTS[0] == "author_full_mean"
    assert set(G.VARIANTS) == {"author_full_mean", "batch_means_unweighted", "drop_last_batch",
                               "per_channel_then_mean", "per_step_then_mean"}


def test_the_per_axis_controls_equal_the_full_mean_on_a_rectangular_population():
    rng = np.random.default_rng(0)
    p = rng.standard_normal((64, 8, 3)).astype(np.float32)
    t = rng.standard_normal((64, 8, 3)).astype(np.float32)
    red = G.reductions(p, t, 32)
    for v in ("per_channel_then_mean", "per_step_then_mean"):
        assert red[v]["mae"] == pytest.approx(red["author_full_mean"]["mae"], abs=1e-6)
        assert red[v]["mse"] == pytest.approx(red["author_full_mean"]["mse"], abs=1e-6)


def test_dropping_the_last_partial_batch_is_reported_with_the_windows_it_drops():
    rng = np.random.default_rng(1)
    p = rng.standard_normal((70, 4, 2)).astype(np.float32)
    t = rng.standard_normal((70, 4, 2)).astype(np.float32)
    red = G.reductions(p, t, 32)
    assert red["drop_last_batch"]["windows_dropped"] == 6
    assert red["batch_means_unweighted"]["batches"] == 3
    assert red["batch_means_unweighted"]["last_batch_windows"] == 6


def test_the_diagnostic_is_labelled_a_diagnostic_and_never_replaces_the_sealed_reduction():
    src = inspect.getsource(G)
    assert "DIAGNOSTIC_NOT_A_MEASUREMENT" in src
    assert "sealed_reduction" in src
    for forbidden in ("std_paper", "k_agree", "k_partial"):
        assert forbidden not in src, "the diagnostics must not be able to touch the agreement margin"


def test_the_class_gap_and_the_per_seed_gap_are_two_different_denominators(tmp_path):
    """The agreement class lives on the three-seed mean. A reduction variant that is large against ONE seed's
    difference can be a rounding error against the class quantity, so the two must never share a field."""
    design = {"dataset": "weather", "protocol": "L96", "seeds": [2021, 2022, 2023],
              "lock": {"published": {"per_horizon": {"96": {"mse": 0.153, "mae": 0.199}}}}}
    cells = tmp_path / "CELLS"
    cells.mkdir()
    for s, mse, mae in ((2021, 0.1530, 0.1992), (2022, 0.1556, 0.2025), (2023, 0.1587, 0.2046)):
        (cells / f"weather_L96_h96_s{s}.json").write_text(json.dumps(
            {"metric": {"author_float32": {"mse": mse, "mae": mae}}}))
    record = {"horizon_steps": 96, "metric": {"author_float32": {"mse": 0.1530, "mae": 0.1992}}}
    out = G.gaps(design, tmp_path, record)
    assert out["per_seed"]["mse"] == pytest.approx(0.0, abs=1e-9)
    assert out["three_seed_mean"]["mse"] == pytest.approx(0.0027667, abs=1e-6)
    assert out["class_quantity"] == "three_seed_mean"
    assert out["three_seed_mean"]["mse"] > out["per_seed"]["mse"]


def test_the_diagnostic_verdict_is_taken_against_the_class_gap_not_the_per_seed_one():
    src = inspect.getsource(G)
    assert "could_explain_the_class_gap" in src
    assert "fraction_of_the_per_seed_gap" in src, "the per-seed fraction is still reported, just not the verdict"
