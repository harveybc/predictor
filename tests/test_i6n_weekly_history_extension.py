"""Lane C: pre-scope history extension decision and row reconciliation (synthetic)."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from tools import i6n_weekly_history_extension as ext

UTC = timezone.utc
START = datetime(2019, 3, 12, 21, tzinfo=UTC)
END = datetime(2019, 12, 1, tzinfo=UTC)


def case(day, status, reason=None):
    return {"date": day, "status": status, "reason": reason}


def test_extension_waits_for_every_planned_case_in_span():
    cases = [case("2019-01-21", "INCONCLUSIVE", "ReadTimeout"), case("2019-03-18", "PASS"),
             case("2019-07-15", "PASS"), case("2019-10-28", "INCONCLUSIVE", "ValueError: HTTP 429"),
             case("2020-01-20", "PASS")]
    d = ext.clock_decision(cases, START, END)
    assert d["state"] == "WAIT_DEPENDENCY"
    assert [c["date"] for c in d["in_span"]] == ["2019-03-18", "2019-07-15", "2019-10-28"]
    assert d["missing"] == [
        {"object": "INDEPENDENT_CLOCK_CASE", "date": "2019-10-28", "status": "INCONCLUSIVE",
         "reason": "ValueError: HTTP 429"},
        {"object": "CLOCK_SCOPE_CERTIFICATE", "start": "2019-03-12T21:00:00+00:00",
         "end": "2019-12-01T00:00:00+00:00",
         "detail": d["missing"][1]["detail"]}]
    ok = ext.clock_decision([c if c["date"] != "2019-10-28" else case("2019-10-28", "PASS")
                             for c in cases], START, END)
    assert ok["state"] == "CLOCK_CASES_PASS_CERTIFICATE_NOT_ISSUED"
    assert [m["object"] for m in ok["missing"]] == ["CLOCK_SCOPE_CERTIFICATE"]


def test_extension_without_any_case_in_span_is_not_certifiable():
    d = ext.clock_decision([case("2020-01-20", "PASS")], START, END)
    assert d["state"] == "WAIT_DEPENDENCY"
    assert d["missing"][0]["object"] == "INDEPENDENT_CLOCK_CASE"


def test_row_reconciliation_exact_and_complete():
    idx = pd.DatetimeIndex([pd.Timestamp(1552500000 + 3600 * k, unit="s", tz="UTC") for k in range(5)])
    legacy = pd.Series([1.1, 1.2, 1.3, 1.4, 1.5], index=idx)
    ts = np.array([int(t.timestamp()) for t in idx], dtype=np.int64)
    price = legacy.to_numpy().copy()
    r = ext.reconcile_rows(ts, price, legacy)
    assert r == {"retained_rows": 5, "source_rows": 5, "matched": 5, "max_abs_price_difference": 0.0}
    bad = price.copy()
    bad[2] += 1e-9
    with pytest.raises(ext.ExtensionError) as e:
        ext.reconcile_rows(ts, bad, legacy)
    assert e.value.code == "PRICE_MISMATCH"
    with pytest.raises(ext.ExtensionError) as e:
        ext.reconcile_rows(ts[:4], price[:4], legacy)
    assert e.value.code == "ROW_SET_MISMATCH"


def test_feature_availability_reports_fractions_not_values():
    x = np.array([[1.0, np.nan], [2.0, np.inf], [3.0, 4.0]])
    out = ext.feature_availability(x, ["a", "b"])
    assert out == {"a": 1.0, "b": pytest.approx(1 / 3)}


def test_successor_suffix_must_be_byte_identical():
    full_ts = np.array([10, 20, 30, 40], dtype=np.int64) * 3600
    full_x = np.arange(8, dtype=float).reshape(4, 2)
    full_p = np.array([1.0, 1.1, 1.2, 1.3])
    k = ext.successor_suffix_index(full_ts, full_x, full_p, full_ts[2:], full_x[2:], full_p[2:])
    assert k == 2
    bad = full_x[2:].copy()
    bad[0, 1] = 99.0
    with pytest.raises(ext.ExtensionError) as e:
        ext.successor_suffix_index(full_ts, full_x, full_p, full_ts[2:], bad, full_p[2:])
    assert e.value.code == "SUCCESSOR_SUFFIX_MISMATCH"
    with pytest.raises(ext.ExtensionError):
        ext.successor_suffix_index(full_ts, full_x, full_p, full_ts[1:3], full_x[1:3], full_p[1:3])
    nan_x = full_x.copy()
    nan_x[3, 0] = np.nan
    assert ext.successor_suffix_index(full_ts, nan_x, full_p, full_ts[2:], nan_x[2:], full_p[2:]) == 2


def test_declared_assumption_names_limitation_and_evidence():
    cases = [case("2019-03-18", "PASS"), case("2019-07-15", "PASS"),
             case("2019-10-28", "INCONCLUSIVE", "ValueError: HTTP 429; no retry")]
    replay = [{"date": "2019-03-18", "status": "PASS", "matches_retained_case": True},
              {"date": "2019-07-15", "status": "PASS", "matches_retained_case": True}]
    a = ext.declared_assumption(cases, replay, START, END, attempts=["r1", "r2"])
    assert a["status"] == "DECLARED_ASSUMPTION"
    assert a["unverified_case"]["date"] == "2019-10-28"
    assert [c["date"] for c in a["passing_cases"]] == ["2019-03-18", "2019-07-15"]
    assert a["limitation"].startswith("NOT independently quote-verified")
    assert len(a["assumption_sha256"]) == 64
    with pytest.raises(ext.ExtensionError):
        ext.declared_assumption(cases, replay[:1], START, END, attempts=["r1"])
    with pytest.raises(ext.ExtensionError):
        ext.declared_assumption([case("2019-03-18", "PASS"), case("2019-07-15", "PASS")],
                                replay, START, END, attempts=[])
