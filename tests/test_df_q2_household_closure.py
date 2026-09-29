"""The closure generator of the Q2 household governed round must fail the ways that matter."""

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "df_q2_household_closure", REPO / "tools" / "df_q2_household_closure.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["df_q2_household_closure"] = module
    spec.loader.exec_module(module)
    return module


def _unit(tmp_path, name, *, peak, accepted=True, warehouse_ok=True, digest="d0"):
    doc = {
        "unit": name, "design_sha256": "de51",
        "delivery": {"delivery_id": "dv1", "sha256": "b319", "bytes": 10890295,
                     "cached": False, "verification_state": "VERIFIED_TRANSFER",
                     "availability_contract_sha256": "93d9"},
        "terminal": {"status": "COMPLETED", "accepted": accepted,
                     "receipt_terminal_sha256": digest},
        "warehouse": {"ok": warehouse_ok, "metric_rows": 3,
                      "terminal": {"terminal_sha256": digest}},
        "child_record": {"host_kind": "WORKER_A",
                         "consumed": {"sha256_reverified_in_child": "b319",
                                      "matches_delivery": True},
                         "work": {"rows": 2075259},
                         "memory": {"scope_peak_bytes": peak,
                                    "scope_peak_provenance": "WHOLE_CGROUP_PEAK",
                                    "rss_self_peak_bytes": 1,
                                    "declared_cap_bytes_as_the_kernel_holds_it": 2}},
    }
    (tmp_path / f"UNIT.{name}.json").write_text(json.dumps(doc))
    return doc


def test_a_round_that_scored_nothing_is_NO_NEW_MEASUREMENT_with_null_errors(tmp_path):
    _unit(tmp_path, "probe", peak=384962560)
    table = _mod().build(tmp_path)
    assert table["no_new_measurement"] is True
    assert table["scored_rows"] == 0
    row = table["rows"][0]
    for key, why in (("model_error", "model_error_absent_because"),
                     ("naive_error", "naive_absent_because"),
                     ("skill", "skill_absent_because")):
        assert row[key] is None, f"{key} must be an absent measurement, not a number"
        assert "ABSENT_MEASUREMENT" in row[why], f"{why} must name the absence explicitly"


def test_a_null_error_is_never_rendered_as_a_zero(tmp_path):
    _unit(tmp_path, "probe", peak=1)
    text = _mod().markdown(_mod().build(tmp_path))
    assert "`null`" in text
    assert "| 0 |" not in text and "| 0.0 |" not in text


def test_a_null_peak_is_reported_as_not_measured_and_never_as_small(tmp_path):
    _unit(tmp_path, "pilot", peak=None)
    table = _mod().build(tmp_path)
    row = table["rows"][0]
    assert row["scope_peak_bytes"] is None
    assert row["scope_peak_absent_because"] == "NULL_PEAK_MEANS_NOT_MEASURED_NEVER_SMALL"
    assert "NOT MEASURED, never small" in _mod().markdown(table)


def test_the_estimand_is_never_converged_accuracy(tmp_path):
    _unit(tmp_path, "probe", peak=1)
    table = _mod().build(tmp_path)
    assert table["rows"][0]["estimand"] == "MATCHED_BUDGET_DIFFERENCE"
    assert "CONVERGED_ACCURACY" not in json.dumps(table["rows"])
    assert "never CONVERGED_ACCURACY" in table["rules"]["estimand"]


def test_a_reference_that_was_not_carried_says_so_and_is_not_comparable(tmp_path):
    _unit(tmp_path, "probe", peak=1)
    row = _mod().build(tmp_path)["rows"][0]
    assert row["reference"] == "NOT_CARRIED"
    assert row["comparability"] == "NOT_COMPARABLE"
    assert len(row["comparability_reason"]) > 20


def test_custody_is_UNCHECKED_when_the_terminal_was_not_accepted(tmp_path):
    _unit(tmp_path, "probe", peak=1, accepted=False)
    assert _mod().build(tmp_path)["rows"][0]["custody"] == "UNCHECKED"


def test_custody_is_UNCHECKED_when_the_warehouse_holds_no_row(tmp_path):
    _unit(tmp_path, "probe", peak=1, warehouse_ok=False)
    assert _mod().build(tmp_path)["rows"][0]["custody"] == "UNCHECKED"


def test_a_warehouse_digest_that_disagrees_with_the_receipt_is_reported(tmp_path):
    doc = _unit(tmp_path, "probe", peak=1)
    doc["warehouse"]["terminal"]["terminal_sha256"] = "OTHER"
    (tmp_path / "UNIT.probe.json").write_text(json.dumps(doc))
    assert _mod().build(tmp_path)["rows"][0]["warehouse_digest_agrees"] is False


def test_the_round_never_claims_a_cap_for_a_cell(tmp_path):
    _unit(tmp_path, "pilot", peak=1463877632)
    table = _mod().build(tmp_path)
    assert any("no successor cap" in item
               for item in table["what_this_round_does_not_establish"])
