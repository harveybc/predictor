"""C137-C138: measured against known truth, per partition; decisions only
from frozen rules; an RMSE gain that erases events does not advance; the
oracle is always rejected; failure regions are published."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("df_lab_evaluation", ROOT / "tools/df_lab_evaluation.py")
LAB = importlib.util.module_from_spec(spec)
sys.modules["df_lab_evaluation"] = LAB
spec.loader.exec_module(LAB)


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    out = tmp_path_factory.mktemp("bank") / "bank"
    subprocess.run([sys.executable, "-B", str(ROOT / "tools/df_synthetic_bank.py"), "--out", str(out)],
                   check=True, capture_output=True)
    return out


def names(bank, family, perturbation, snr):
    return sorted(p.name for p in bank.iterdir()
                  if p.is_dir() and p.name.startswith(f"{family}__{perturbation}__snr{snr}__none__n2048__"))


SPECS = [{"kind": "identity", "params": {}}, {"kind": "trailing_mean", "params": {"window": 9}},
         {"kind": "ewma", "params": {"alpha": 0.3}}, {"kind": "centered_mean_oracle", "params": {"window": 5}}]


@pytest.fixture(scope="module")
def result(bank, tmp_path_factory):
    units = (names(bank, "sinusoid", "white", "0") + names(bank, "impulses", "white", "-5")
             + names(bank, "sinusoid", "null", "inf"))
    assert len(units) == 9
    out = tmp_path_factory.mktemp("lab") / "run"
    summary = LAB.run(bank, out, specs=SPECS, unit_names=units, workers=1)
    decisions = [json.loads(line) for line in (out / "df_fact_lab_decision.jsonl").read_text().splitlines()]
    return summary, decisions, out


def decision(decisions, kind, family, perturbation):
    return next(d for d in decisions if d["operator_kind"] == kind and d["regime"]["family"] == family
                and d["regime"]["perturbation"] == perturbation)


def test_rules_are_frozen_and_bound():
    assert LAB.RULE_SHA256 == LAB.sha_obj(LAB.DECISION_RULES)
    assert "public eligibility" in LAB.DECISION_RULES["never"]


def test_a_smoother_that_improves_and_preserves_is_calibrated(result):
    _, decisions, _ = result
    d = decision(decisions, "trailing_mean", "sinusoid", "white")
    assert d["decision"] == "LAB_CALIBRATED", d["evidence"]
    assert d["evidence"]["calibration_medians"]["snr_improvement_db"] > 1.0


def test_identity_and_the_oracle_do_not_advance(result):
    _, decisions, _ = result
    assert decision(decisions, "identity", "sinusoid", "white")["decision"] == "LAB_REJECTED"
    d = decision(decisions, "centered_mean_oracle", "sinusoid", "white")
    assert d["decision"] == "LAB_REJECTED" and "NON_CAUSAL_CONTROL" in d["evidence"]["reasons"][0]


def test_an_rmse_gain_that_erases_impulses_is_rejected(result):
    _, decisions, _ = result
    d = decision(decisions, "trailing_mean", "impulses", "white")
    assert d["decision"] == "LAB_REJECTED"
    assert any(r.startswith("EVENT_DESTRUCTION") for r in d["evidence"]["reasons"]), d["evidence"]
    assert d["evidence"]["calibration_medians"]["impulse_retention"] < 0.5


def test_noise_free_regime_judges_false_positives(result):
    _, decisions, _ = result
    assert decision(decisions, "identity", "sinusoid", "null")["decision"] == "LAB_CALIBRATED"
    d = decision(decisions, "ewma", "sinusoid", "null")
    assert d["decision"] == "LAB_REJECTED" and "FALSE_POSITIVE_DISTORTION" in d["evidence"]["reasons"][0]


def test_failure_regions_are_published_and_rows_validate(result):
    summary, decisions, out = result
    ops = summary["operators"]
    assert all("failure_regions" in e for e in ops.values())
    assert any(e["failure_regions"] for e in ops.values())
    for table in ("df_fact_operator_run", "df_fact_operator_signal_metric", "df_fact_operator_delay_cost", "df_fact_lab_decision"):
        rows = [json.loads(line) for line in (out / f"{table}.jsonl").read_text().splitlines()]
        assert rows and all(LAB.LOADER.validate_row(table, r) == [] for r in rows)
    assert all(d["externally_reviewed"] is False for d in decisions)
    assert {d["decision"] for d in decisions} <= {"LAB_CALIBRATED", "REGIME_LIMITED", "NOT_IDENTIFIABLE", "LAB_REJECTED"}


def test_fitting_never_sees_calibration_or_confirmation(bank, tmp_path):
    import shutil
    unit = bank / names(bank, "sinusoid", "white", "0")[0]
    u = LAB.load_unit(unit)
    s = LAB.train_fit_slice(u["observed"], u["rec"]["partitions"]["train"])
    assert s.stop <= u["rec"]["partitions"]["train"][1]
    f1 = LAB.OPS.fit(SPECS[2], u["observed"][s], "train")
    changed = u["observed"].copy()
    changed[s.stop:] += 100.0
    f2 = LAB.OPS.fit(SPECS[2], changed[s], "train")
    assert f1["artifact_sha256"] == f2["artifact_sha256"]


def test_too_few_units_is_not_identifiable():
    spec_ = {"kind": "ewma", "params": {"alpha": 0.3}}
    recs = [{"status": "COMPLETED", "partitions": {"calibration": {"status": "COMPLETED", "rmse_ratio": 0.5}}}]
    d, reasons, _ = LAB.decide(spec_, {}, recs)
    assert d == "NOT_IDENTIFIABLE" and "valid" in reasons[0]


def test_outputs_are_write_once(bank, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(SystemExit, match="write-once"):
        LAB.run(bank, out, specs=SPECS[:1], unit_names=names(bank, "sinusoid", "white", "0"), workers=1)
