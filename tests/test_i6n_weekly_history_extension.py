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
