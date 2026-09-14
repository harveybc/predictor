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
    assert "UNITS_MISSING_FROM_REPLAY" in result["inconclusive_reasons"]


def test_a_regime_without_substitution_never_counts_as_stable(replay_module, tmp_path):
    unusable = _fact(result={"snr_db": None, "status": "NOT_IDENTIFIABLE", "reason": "flat", "bootstrap": {}})
    replay, snr, decisions, subset = _inputs(tmp_path, facts={UNIT: [unusable]}, snr_rows=[_row()])
    out = tmp_path / "out"
    out.mkdir()
    result = replay_module.compare({}, [replay], snr, decisions, out, 1e-9, subset=subset)
    assert result["coverage"]["substituted_facts"] == 0
    assert "NO_SUBSTITUTED_FACTS" in result["inconclusive_reasons"]
    assert result["decision_stability"]["changed"] is None


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
