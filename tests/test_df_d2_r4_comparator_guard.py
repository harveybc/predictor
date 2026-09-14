"""D2-R4 comparator: declared rules for what a stability conclusion requires.

Owner's finding (2026-09-14): "el comparador puede informar cero cambios con
entradas vacías; debe corregirse antes de aceptar su conclusión de estabilidad".
A comparison that covers nothing must not be reported as a comparison that found
nothing to change. Each test below is one rule; the adjudicator is stubbed so the
rules under test are the comparator's coverage accounting, not the D2 rules.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

TOOLS = Path(os.environ.get("D2_TOOLS_DIR") or (Path(__file__).resolve().parents[1] / "tools"))

REGIME = {"family": "motif", "perturbation": "white", "declared_snr_db": "5",
          "length": 2048, "missingness": "none"}
UNIT = "motif__white__snr5__none__n2048__v1__seed1"
EST = "mad_first_difference"


def _stub_adjudicator():
    stub = types.ModuleType("df_d2_adjudicate")

    def decide_snr(rows, design):
        subjects = sorted({r["estimator"] for r in rows})
        return [{"subject": s, "decision": "SNR_CALIBRATED_FOR_REGIME",
                 "evidence": {"ci95_abs_error_db": [0.0, 0.2]}} for s in subjects]

    stub.decide_snr = decide_snr
    sys.modules["df_d2_adjudicate"] = stub
    return stub


@pytest.fixture()
def replay_module():
    _stub_adjudicator()
    spec = importlib.util.spec_from_file_location("df_d2_r4_replay_under_test", TOOLS / "df_d2_r4_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _stub_adjudicator()  # the module reuses sys.modules; keep the stub after import
    return module


def _row(**over):
    row = {"unit_id": UNIT, "variable_index": 0, "estimator": EST, "partition": "confirmation",
           "regime": dict(REGIME), "snr_db_hat": 5.25, "true_snr_db": 5.0, "error_db": 0.25,
           "abs_error_db": 0.25, "identifiability": "ESTIMATED", "status": "OBSERVED",
           "ci_low_db": 4.9, "ci_high_db": 5.6, "ci_covers_true": 1.0}
    row.update(over)
    return row


def _fact(**over):
    fact = {"variable_index": 0, "estimator": EST, "partition": "confirmation", "status": "OBSERVED",
            "result": {"snr_db": 5.25, "status": "ESTIMATED", "reason": None,
                       "bootstrap": {"ci_low_db": 4.9, "ci_high_db": 5.6}}}
    fact.update(over)
    return fact


def _write(tmp, name, obj):
    path = tmp / name
    path.write_text(json.dumps(obj) + "\n", encoding="utf-8")
    return path


def _write_lines(tmp, name, rows):
    path = tmp / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _inputs(tmp, *, facts, snr_rows, subset_units=(UNIT,)):
    replay = _write(tmp, "R4_REPLAY_ROLE.json",
                    {"schema": "d2_r4_replay.v1", "role": "ROLE", "environment": {"cpu_model": "test"},
                     "design_sha256": "d", "subset_rule": "test", "df_snr_code_sha256": {},
                     "units": facts, "timings": {}, "cpu_seconds_total": 0.0, "peak_rss_bytes": 0})
    snr = _write_lines(tmp, "snr.jsonl", snr_rows)
    decisions = _write_lines(tmp, "decisions.jsonl", [
        {"subject_kind": "SNR_ESTIMATOR", "subject": EST, "regime": dict(REGIME),
         "decision": "SNR_CALIBRATED_FOR_REGIME", "evidence": {"ci95_abs_error_db": [0.0, 0.2]}}])
    subset = {"rule": "test", "units": {"regime": list(subset_units)}}
    return replay, snr, decisions, subset


def test_no_replayed_facts_is_inconclusive_not_zero_changes(replay_module, tmp_path):
    replay, snr, decisions, subset = _inputs(tmp_path, facts={}, snr_rows=[_row()])
    out = tmp_path / "out"
    out.mkdir()
    result = replay_module.compare({}, [replay], snr, decisions, out, 1e-9, subset=subset)
    assert result["verdict"] == "INCONCLUSIVE"
    assert "NO_REPLAYED_FACTS" in result["inconclusive_reasons"]
    assert "NO_COMPARED_CELLS" in result["inconclusive_reasons"]
    assert "MISSING_SELECTED_FACTS" in result["inconclusive_reasons"]
    assert result["decision_stability"]["changed"] is None


def test_facts_without_a_historical_row_are_counted_and_refuse(replay_module, tmp_path):
    replay, snr, decisions, subset = _inputs(tmp_path, facts={UNIT: [_fact()]},
                                             snr_rows=[_row(unit_id="another__unit")])
    out = tmp_path / "out"
    out.mkdir()
    result = replay_module.compare({}, [replay], snr, decisions, out, 1e-9, subset=subset)
    assert result["coverage"]["facts_without_historical_row"] == 1
    assert result["coverage"]["compared_cells"] == 0
    assert result["verdict"] == "INCONCLUSIVE"


def test_a_unit_missing_from_the_replay_is_named(replay_module, tmp_path):
    replay, snr, decisions, subset = _inputs(tmp_path, facts={UNIT: [_fact()]}, snr_rows=[_row()],
                                             subset_units=(UNIT, "a_second_unit"))
    out = tmp_path / "out"
    out.mkdir()
    result = replay_module.compare({}, [replay], snr, decisions, out, 1e-9, subset=subset)
    assert result["coverage"]["units_missing"] == ["a_second_unit"]
    assert "ROLE_MISSING_SELECTED_UNITS" in result["inconclusive_reasons"]


def test_a_replay_that_estimates_nothing_does_not_leave_the_old_estimate_standing(replay_module, tmp_path):
    """The substitution is the replayed state, not the published value when the replay failed."""
    unusable = _fact(result={"snr_db": None, "status": "NOT_IDENTIFIABLE", "reason": "flat", "bootstrap": {}})
    replay, snr, decisions, subset = _inputs(tmp_path, facts={UNIT: [unusable]}, snr_rows=[_row()])
    out = tmp_path / "out"
    out.mkdir()
    result = replay_module.compare({}, [replay], snr, decisions, out, 1e-9, subset=subset)
    assert result["coverage"]["substituted_non_estimates"] == 1
    assert result["substituted_states"]["NOT_IDENTIFIABLE"] == 1


def test_a_covered_comparison_is_measured_and_reports_per_estimator(replay_module, tmp_path):
    replay, snr, decisions, subset = _inputs(tmp_path, facts={UNIT: [_fact()]}, snr_rows=[_row()])
    out = tmp_path / "out"
    out.mkdir()
    result = replay_module.compare({}, [replay], snr, decisions, out, 1e-9, subset=subset)
    assert result["verdict"] == "MEASURED"
    assert result["inconclusive_reasons"] == []
    assert result["coverage"]["substituted_facts"] == 1
    assert result["decision_stability"] == {
        "verdict": "MEASURED", "compared": 1, "changed": 0, "changed_rows": [],
        "note": "stability is asserted only over the compared scope"}
    assert result["per_estimator"][EST]["bytes_equal"] == 1
    assert result["per_estimator"][EST]["outside_tolerance"] == 0


def test_a_deviating_estimator_is_separated_from_the_exact_ones(replay_module, tmp_path):
    other = "local_level_kalman"
    facts = {UNIT: [_fact(), _fact(estimator=other, variable_index=1,
                          result={"snr_db": 5.250001, "status": "ESTIMATED", "reason": None,
                                  "bootstrap": {"ci_low_db": 4.9, "ci_high_db": 5.6}})]}
    snr_rows = [_row(), _row(estimator=other, variable_index=1)]
    replay, snr, decisions, subset = _inputs(tmp_path, facts=facts, snr_rows=snr_rows)
    out = tmp_path / "out"
    out.mkdir()
    result = replay_module.compare({}, [replay], snr, decisions, out, 1e-9, subset=subset)
    assert result["per_estimator"][EST]["bytes_equal"] == 1
    assert result["per_estimator"][other]["bytes_equal"] == 0
    assert result["per_estimator"][other]["outside_tolerance"] == 1


def test_the_cli_refuses_an_empty_comparison(tmp_path):
    replay, snr, decisions, subset = _inputs(tmp_path, facts={}, snr_rows=[_row()])
    subset_path = _write(tmp_path, "subset.json", subset)
    design = _write(tmp_path, "design.json", {"design_sha256": "d", "snr": {}})
    proc = subprocess.run([sys.executable, "-B", str(TOOLS / "df_d2_r4_replay.py"), "--compare",
                           "--design", str(design), "--subset", str(subset_path),
                           "--out", str(tmp_path / "cli_out"), "--replay-file", str(replay),
                           "--snr-table", str(snr), "--decisions", str(decisions)],
                          capture_output=True, text=True)
    assert proc.returncode == 4, proc.stderr
    assert "REFUSED" in proc.stderr


# --------------------------------------------------------------------------------------
# B2 (Musashi, 2026-09-14): the comparison must enforce its frozen population.
# "Empty input, omitted unit/fact/role, duplicated fact and an unrelated reference row must
# be rejected or explicitly incomplete, never a successful zero-change summary."
# --------------------------------------------------------------------------------------

UNIT_B = "motif__white__snr5__none__n2048__v1__seed2"
DESIGN = {"design_sha256": "d" * 64}


def _multi_inputs(tmp, *, roles, facts_by_role, snr_rows, subset_units):
    """A replay file per role, the historical table, the published decisions, the subset."""
    replays = []
    for role in roles:
        path = tmp / f"R4_REPLAY_{role}.json"
        path.write_text(json.dumps(
            {"schema": "d2_r4_replay.v1", "role": role, "environment": {"cpu_model": role},
             "design_sha256": DESIGN["design_sha256"], "subset_rule": "test",
             "df_snr_code_sha256": {}, "units": facts_by_role[role], "timings": {},
             "cpu_seconds_total": 0.0, "peak_rss_bytes": 0}) + "\n", encoding="utf-8")
        replays.append(path)
    snr = _write_lines(tmp, "snr.jsonl", snr_rows)
    decisions = _write_lines(tmp, "decisions.jsonl", [
        {"subject_kind": "SNR_ESTIMATOR", "subject": EST, "regime": dict(REGIME),
         "decision": "SNR_CALIBRATED_FOR_REGIME", "evidence": {"ci95_abs_error_db": [0.0, 0.2]}}])
    subset = {"rule": "test", "units": {"regime": list(subset_units)}}
    return replays, snr, decisions, subset


def _run(module, tmp, **kwargs):
    out = tmp / ("out_" + str(len(list(tmp.glob("out_*")))))
    out.mkdir()
    replays, snr, decisions, subset = _multi_inputs(tmp, **kwargs)
    return module.compare(DESIGN, replays, snr, decisions, out, 1e-9, subset=subset)


def test_a_role_that_omits_a_selected_unit_is_incomplete_not_stable(replay_module, tmp_path):
    result = _run(replay_module, tmp_path, roles=["A", "B"],
                  facts_by_role={"A": {UNIT: [_fact()], UNIT_B: [_fact()]}, "B": {UNIT: [_fact()]}},
                  snr_rows=[_row(), _row(unit_id=UNIT_B)], subset_units=(UNIT, UNIT_B))
    assert result["verdict"] == "INCONCLUSIVE"
    assert "ROLE_MISSING_SELECTED_UNITS" in result["inconclusive_reasons"]
    assert result["coverage"]["units_missing_by_role"]["B"] == [UNIT_B]


def test_a_duplicated_fact_is_refused(replay_module, tmp_path):
    result = _run(replay_module, tmp_path, roles=["A"],
                  facts_by_role={"A": {UNIT: [_fact(), _fact()]}},
                  snr_rows=[_row()], subset_units=(UNIT,))
    assert result["verdict"] == "INCONCLUSIVE"
    assert "DUPLICATED_FACTS" in result["inconclusive_reasons"]
    assert result["coverage"]["duplicated_facts"]


def test_two_replay_files_of_the_same_role_are_refused(replay_module, tmp_path):
    out = tmp_path / "out_roles"
    out.mkdir()
    replays, snr, decisions, subset = _multi_inputs(
        tmp_path, roles=["A"], facts_by_role={"A": {UNIT: [_fact()]}},
        snr_rows=[_row()], subset_units=(UNIT,))
    result = replay_module.compare(DESIGN, [replays[0], replays[0]], snr, decisions, out, 1e-9,
                                   subset=subset)
    assert result["verdict"] == "INCONCLUSIVE"
    assert "DUPLICATED_ROLES" in result["inconclusive_reasons"]


def test_a_replay_of_another_design_is_refused(replay_module, tmp_path):
    out = tmp_path / "out_design"
    out.mkdir()
    replays, snr, decisions, subset = _multi_inputs(
        tmp_path, roles=["A"], facts_by_role={"A": {UNIT: [_fact()]}},
        snr_rows=[_row()], subset_units=(UNIT,))
    doc = json.loads(replays[0].read_text())
    doc["design_sha256"] = "e" * 64
    replays[0].write_text(json.dumps(doc))
    result = replay_module.compare(DESIGN, replays, snr, decisions, out, 1e-9, subset=subset)
    assert result["verdict"] == "INCONCLUSIVE"
    assert "REPLAY_OF_ANOTHER_DESIGN" in result["inconclusive_reasons"]


def test_a_selected_fact_that_was_not_replayed_is_missing_not_carried(replay_module, tmp_path):
    """A selected key with no replayed fact must be MISSING, never silently kept."""
    result = _run(replay_module, tmp_path, roles=["A"],
                  facts_by_role={"A": {UNIT: [_fact()]}},
                  snr_rows=[_row(), _row(variable_index=1)], subset_units=(UNIT,))
    assert result["verdict"] == "INCONCLUSIVE"
    assert "MISSING_SELECTED_FACTS" in result["inconclusive_reasons"]
    assert result["coverage"]["missing_selected_facts"] == 1


def test_an_unselected_historical_row_is_carried_and_counted_separately(replay_module, tmp_path):
    """Rows of units outside the subset stay untouched, and are reported as carried."""
    result = _run(replay_module, tmp_path, roles=["A"],
                  facts_by_role={"A": {UNIT: [_fact()]}},
                  snr_rows=[_row(), _row(unit_id="a_unit_outside_the_subset")],
                  subset_units=(UNIT,))
    assert result["coverage"]["carried_unselected_rows"] == 1
    assert result["coverage"]["missing_selected_facts"] == 0
    assert result["verdict"] == "MEASURED"


def test_a_replayed_non_estimate_replaces_the_old_estimate(replay_module, tmp_path):
    """A replay that fails to estimate must not leave the published value standing."""
    unusable = _fact(result={"snr_db": None, "status": "NOT_IDENTIFIABLE",
                             "reason": "signal_variance_nonpositive", "bootstrap": {}})
    result = _run(replay_module, tmp_path, roles=["A"], facts_by_role={"A": {UNIT: [unusable]}},
                  snr_rows=[_row()], subset_units=(UNIT,))
    assert result["coverage"]["substituted_non_estimates"] == 1
    substituted = result["substituted_states"]
    assert substituted["NOT_IDENTIFIABLE"] == 1
    assert result["verdict"] == "MEASURED"


def test_a_nonfinite_replayed_value_is_not_an_estimate(replay_module, tmp_path):
    for value in (float("nan"), float("inf")):
        result = _run(replay_module, tmp_path, roles=["A"],
                      facts_by_role={"A": {UNIT: [_fact(result={
                          "snr_db": value, "status": "ESTIMATED", "reason": None,
                          "bootstrap": {"ci_low_db": 4.9, "ci_high_db": 5.6}})]}},
                      snr_rows=[_row()], subset_units=(UNIT,))
        assert result["coverage"]["substituted_non_estimates"] == 1, value
        assert result["substituted_states"].get("INVALID_VALUE") == 1, value


def test_a_zero_difference_is_still_a_comparison(replay_module, tmp_path):
    """0.0 must compare as a number, not as a missing value."""
    result = _run(replay_module, tmp_path, roles=["A"],
                  facts_by_role={"A": {UNIT: [_fact(result={
                      "snr_db": 0.0, "status": "ESTIMATED", "reason": None,
                      "bootstrap": {"ci_low_db": -1.0, "ci_high_db": 1.0}})]}},
                  snr_rows=[_row(snr_db_hat=0.0, error_db=0.0, abs_error_db=0.0)],
                  subset_units=(UNIT,))
    assert result["verdict"] == "MEASURED"
    assert result["per_estimator"][EST]["comparable"] == 1
    assert result["per_estimator"][EST]["max_abs_delta_db"] == 0.0


def test_a_complete_population_reports_its_denominators(replay_module, tmp_path):
    result = _run(replay_module, tmp_path, roles=["A", "B"],
                  facts_by_role={"A": {UNIT: [_fact()], UNIT_B: [_fact()]},
                                 "B": {UNIT: [_fact()], UNIT_B: [_fact()]}},
                  snr_rows=[_row(), _row(unit_id=UNIT_B)], subset_units=(UNIT, UNIT_B))
    assert result["verdict"] == "MEASURED"
    denominators = result["coverage"]["denominators"]
    assert denominators["selected_units"] == 2
    assert denominators["expected_facts_per_role"] == 2
    assert denominators["roles"] == 2
    assert denominators["compared_cells"] == 4
