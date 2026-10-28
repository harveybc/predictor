"""V2 hypothesis tests, synthetic prices only."""

from datetime import date
import hashlib
import json

import pandas as pd
import pytest

from tools import i6n_external_clock_audit_v2 as audit


def test_holdouts_are_fixed_after_european_october_transition():
    assert audit.holdout_dates() == [date(2019, 10, 28), date(2020, 10, 26),
        date(2021, 11, 1), date(2022, 10, 31), date(2023, 10, 30), date(2024, 10, 28)]


def test_eu_rule_three_seasons_and_october_gap():
    wall = pd.to_datetime(["2024-01-15 08:00", "2024-03-18 08:00",
                          "2024-07-15 08:00", "2024-10-28 08:00"])
    hourly = pd.Series([1., 2., 3., 4.], index=wall)
    result, evidence = audit.eu_clock_series(hourly)
    assert list(result.index.hour) == [14, 14, 13, 14]  # includes hour end
    assert evidence["minimum_shift_vs_legacy_seconds"] == 0
    assert evidence["maximum_shift_vs_legacy_seconds"] == 3600
    assert evidence["moved_later_rows"] == 2
    assert evidence["never_moves_earlier"] is True


def test_equal_helsinki_formulation():
    wall = pd.date_range("2019-01-01", "2024-12-31", freq="7D") + pd.Timedelta(hours=12)
    hourly = pd.Series(range(len(wall)), index=wall, dtype=float)
    eu, _ = audit.eu_clock_series(hourly)
    expected = (wall + pd.Timedelta(hours=7)).tz_localize("Europe/Helsinki").tz_convert("UTC") + pd.Timedelta(hours=1)
    assert eu.index.equals(expected)


def sample():
    index = pd.date_range("2024-07-15 01:00", periods=24, freq="h", tz="UTC")
    return pd.Series([1.1 + i / 1e5 for i in range(24)], index=index)


@pytest.mark.parametrize("available,expected", [(24, "PASS"), (23, "PASS"), (17, "PASS"),
                                                        (12, "PASS"), (11, "INCONCLUSIVE")])
def test_explicit_missing_endpoints_and_minimum(available, expected):
    reference = sample()
    case = audit.compare_case(date(2024, 7, 15), reference, reference.iloc[:available])
    assert case["status"] == expected
    assert case["available_endpoints"] == available
    assert len(case["absent_endpoints_utc"]) == 24 - available
    assert case["matched_exact"] == available


def test_any_price_change_fails_without_relaxation():
    reference = sample()
    changed = reference.copy()
    changed.iloc[0] += 1e-13
    assert audit.compare_case(date(2024, 7, 15), reference, changed)["status"] == "INCONCLUSIVE"


def test_nonfinite_is_invalid_not_counted_as_absent():
    reference = sample()
    changed = reference.copy()
    changed.iloc[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        audit.compare_case(date(2024, 7, 15), reference, changed)


def test_complete_plan_and_holdouts_required():
    planned = ["2024-01-15", *[d.isoformat() for d in audit.holdout_dates()]]
    cases = [dict(date=d, status="PASS") for d in planned]
    assert audit.terminal_status(cases, planned, annual_ok=True) == "VERIFIED"
    assert audit.terminal_status(cases[:-1], planned, annual_ok=True) == "INCONCLUSIVE"
    assert audit.terminal_status(cases, planned, annual_ok=False) == "INCONCLUSIVE"
    cases[-1]["status"] = "INCONCLUSIVE"
    assert audit.terminal_status(cases, planned, annual_ok=True) == "INCONCLUSIVE"


def test_replay_report_must_be_hash_pinned(tmp_path):
    (tmp_path / "REPORT.json").write_bytes(b"{}")
    (tmp_path / "REPORT.sha256").write_text("0" * 64)
    with pytest.raises(ValueError, match="hash"):
        audit.load_replay(tmp_path, "a" * 64)


def test_policy_not_v1_and_no_plan_mutation():
    assert audit.CLOCK_POLICY == "EU_DST_OFFSET_MINUS5_BASE_BAR_START"
    before = audit.v1.planned_dates()
    audit.holdout_dates()
    assert before == audit.v1.planned_dates()
    assert audit.v1.CLOCK_POLICY == "FIXED_EST_BAR_START"
