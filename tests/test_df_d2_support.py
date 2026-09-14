"""D2-R1: the support contract of the adjudicator, declared before the repair.

Eight rules, each a regression that the unrepaired adjudicator violates (PRE) and the
repaired one satisfies (POST). Rows come from the repository's small fixture
(tests/test_df_d2_lab.py) and the real adjudicator; nothing here touches campaign data.

  1. removing a metric that proves damage never turns a rejection into a pass;
  2. an INCONCLUSIVE delay or residual does not satisfy its limit;
  3. an event that does not exist by contract is inapplicable, not measured as zero;
  4. an event that is present with its metric missing does not become inapplicable;
  5. a variable without support stays in the declared universe and denominator;
  6. a seed without the primary contrast is not a complete seed;
  7. fewer than two applicable non-inferiority pairs never give passed=True;
  8. an SNR seed missing a required variable does not improve by dropping it.
"""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    spec = importlib.util.spec_from_file_location("d2_lab_fixture", ROOT / "tests" / "test_df_d2_lab.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["d2_lab_fixture"] = module
    spec.loader.exec_module(module)
    return module


F = _fixture()
A = F.A
D = F.D


@pytest.fixture(scope="module")
def design():
    return D.build_design(F.inputs())


def _ewma(design, rows):
    return F._by(A.decide_denoising(rows, design), "ewma")


def _drop(rows, kind, metric):
    return [r for r in rows if not (r["operator_kind"] == kind and r["metric"] == metric)]


def _inconclusive(rows, kind, metric, units=None):
    out = copy.deepcopy(rows)
    for r in out:
        if r["operator_kind"] == kind and r["metric"] == metric and (units is None or r["unit_id"] in units):
            r.update(value=None, status="INCONCLUSIVE", reason="undefined for this unit")
    return out


def test_control_full_fixture_still_calibrates(design):
    d = _ewma(design, F._den(design))
    assert d["decision"] == "LAB_CALIBRATED"
    assert d["evidence"]["support"]["seeds_complete"] == d["n_seeds_design"] == d["n_seeds_valid"]


def test_rule_1_removing_a_damage_metric_never_turns_rejection_into_pass(design):
    damaged = copy.deepcopy(F._den(design))
    for r in damaged:
        if r["operator_kind"] == "ewma" and r["metric"] == "extreme_retention":
            r["value"] = 0.2
    assert _ewma(design, damaged)["decision"] == "LAB_REJECTED"
    omitted = _ewma(design, _drop(damaged, "ewma", "extreme_retention"))
    assert omitted["decision"] != "LAB_CALIBRATED"
    assert omitted["decision"] == "NOT_IDENTIFIABLE"
    assert "extreme_retention" in omitted["evidence"]["support"]["unsupported_by_metric"]


@pytest.mark.parametrize("metric", ["delay_samples", "residual_signal_share"])
def test_rule_2_inconclusive_limit_metrics_do_not_pass(design, metric):
    d = _ewma(design, _inconclusive(F._den(design), "ewma", metric))
    assert d["decision"] != "LAB_CALIBRATED"
    assert d["evidence"]["support"]["unsupported_by_metric"].get(metric)


def test_rule_3_event_absent_by_contract_is_inapplicable_not_zero(design):
    rows = F._den(design, n_seeds=30, regime=F._cell_regime(1))  # steps regime, no step event in the window
    d = _ewma(design, rows)
    assert d["decision"] == "LAB_CALIBRATED"
    floor = d["evidence"]["checks"]["floor:step_delay_samples"]
    assert floor["applicable_seeds"] == 0 and floor.get("applicable") is False and floor["passed"] is True
    assert "step_delay_samples" in d["evidence"]["support"]["inapplicable_by_metric"]


def test_rule_4_event_present_with_missing_metric_is_unsupported(design):
    rows = F._den(design, n_seeds=30, regime=F._cell_regime(1))
    op = next(o for o in design["operators"] if o["kind"] == "ewma")
    for unit in ("u_200", "u_201"):
        rows.append(F.drow(design, unit, op, "step_delay_samples__events", 2.0, regime=F._cell_regime(1)))
    d = _ewma(design, rows)
    assert d["decision"] == "NOT_IDENTIFIABLE"
    assert sorted(d["evidence"]["support"]["unsupported_by_metric"]["step_delay_samples"]) == ["u_200", "u_201"]


def test_rule_5_unsupported_variable_stays_in_the_denominator(design):
    d = _ewma(design, _inconclusive(F._den(design), "ewma", "snr_improvement_db", units={"u_203"}))
    support = d["evidence"]["support"]
    assert support["seeds_planned"] == 10 and support["seeds_observed"] == 10 and support["seeds_complete"] == 9
    assert support["unsupported_by_metric"]["snr_improvement_db"] == ["u_203"]
    assert d["n_seeds_valid"] == 9 and d["decision"] == "NOT_IDENTIFIABLE"
    assert any("9" in reason and "10" in reason for reason in d["reasons"])


def test_rule_6_a_seed_with_only_cost_rows_is_not_complete(design):
    rows = [r for r in F._den(design)
            if not (r["operator_kind"] == "ewma" and r["unit_id"] == "u_204" and r["branch"] != "COST")]
    d = _ewma(design, rows)
    assert d["decision"] == "NOT_IDENTIFIABLE" and d["evidence"]["support"]["seeds_complete"] == 9
    assert d["evidence"]["support"]["seeds_observed"] == 10


def test_rule_7_fewer_than_two_applicable_pairs_never_pass(design):
    # the extreme geometry is undefined (raw INCONCLUSIVE) on nine seeds: inapplicable there,
    # one applicable pair remains -> no pass, no silent skip
    rows = _inconclusive(F._den(design), "ewma", "extreme_retention_raw", units={f"u_{200 + i}" for i in range(1, 10)})
    d = _ewma(design, rows)
    check = d["evidence"]["checks"]["non_inferiority:extreme_retention"]
    assert check.get("passed") is not True and check["n"] == 1 and check["applicable"] is True
    assert d["decision"] == "NOT_IDENTIFIABLE"


def test_rule_8_snr_seed_missing_a_required_variable_is_not_identifiable(design):
    rows = F._snr(design, [0.2] * 10)
    unit = rows[0]["unit_id"]
    second = dict(F.srow(unit, 9.0, ident="NOT_IDENTIFIABLE", design_sha=design["design_sha256"]))
    second.update(variable_id="v1", variable_index=1)
    (d,) = A.decide_snr(rows + [second], design)
    assert unit not in d["evidence"]["per_seed_mean_abs_error_db"]
    assert d["evidence"]["n_seeds_identifiable"] == 9 and d["decision"] == "SNR_NOT_IDENTIFIABLE"
    assert d["evidence"]["support"]["seeds_incomplete"] == [unit]
