"""Red/green tests of the shared eligibility mask (plan audit v3, V3-01): reopen and forced-flat 12 h rules,
k-window not crossing a closure, DST weeks (closures move in UTC), a short holiday session, weekend windows,
label usability per target and the accounting of every origin."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
E = importlib.import_module("tools.timing_eligibility")


def _week_bars(open_utc, close_utc):
    return pd.date_range(open_utc, pd.Timestamp(close_utc) - pd.Timedelta(hours=1), freq="h", tz="UTC")


def _two_weeks(winter=True):
    # Sunday 22:00 UTC open / Friday 22:00 UTC close in winter; 21:00 in summer (the DST move of the closure)
    h = 22 if winter else 21
    w1o, w1c = f"2023-01-08 {h}:00", f"2023-01-13 {h}:00"
    w2o, w2c = f"2023-01-15 {h}:00", f"2023-01-20 {h}:00"
    if not winter:
        w1o, w1c, w2o, w2c = (s.replace("2023-01", "2023-07").replace("-08 ", "-09 ").replace("-13 ", "-14 ")
                              .replace("-15 ", "-16 ").replace("-20 ", "-21 ") for s in (w1o, w1c, w2o, w2c))
    idx = _week_bars(w1o, w1c).append(_week_bars(w2o, w2c))
    closures = [[w1c, w2o]]
    return idx, closures, (pd.Timestamp(w1o, tz="UTC"), pd.Timestamp(w1c, tz="UTC"), pd.Timestamp(w2o, tz="UTC"))


@pytest.mark.parametrize("winter", [True, False])
def test_reopen_and_force_flat_twelve_hours_in_both_dst_regimes(winter):
    idx, closures, (o1, c1, o2) = _two_weeks(winter)
    m = E.eligibility(idx, closures, k_bars=2)
    ts = pd.DatetimeIndex(m.timestamp)
    dec = ts + pd.Timedelta(hours=1)
    second = ts >= o2
    early = second & (dec < o2 + pd.Timedelta(hours=12))
    assert (m.business_reason[early] == "BEFORE_REOPEN_12H").all() and early.sum() == 11
    assert m.business_eligible[second & (dec >= o2 + pd.Timedelta(hours=12)) & (ts + pd.Timedelta(hours=3) <= o2 + pd.Timedelta(days=5) - pd.Timedelta(hours=12))].all()
    late = (~second) & (ts + pd.Timedelta(hours=3) > c1 - pd.Timedelta(hours=12)) & (dec >= o1 + pd.Timedelta(hours=12))
    assert set(m.business_reason[late]) <= {"AFTER_FORCE_FLAT_12H", "WINDOW_CROSSES_CLOSURE"} and late.any()


def test_weekend_window_crossing_the_closure_is_never_eligible_and_inputs_are_not_dropped():
    idx, closures, _ = _two_weeks(True)
    m = E.eligibility(idx, closures, k_bars=2, preclose_hours=0.0)
    last = int(np.flatnonzero(pd.DatetimeIndex(m.timestamp) < pd.Timestamp("2023-01-13 22:00", tz="UTC"))[-1])
    assert m.business_reason.iloc[last] == "WINDOW_CROSSES_CLOSURE" and m.business_reason.iloc[last - 1] == "WINDOW_CROSSES_CLOSURE"
    assert len(m) == len(idx)                                            # every bar kept (historical inputs untouched)


def test_short_holiday_session_has_no_eligible_origin():
    idx = pd.date_range("2023-12-26 00:00", "2023-12-26 17:00", freq="h", tz="UTC")     # 18 h session
    closures = [["2023-12-22 22:00", "2023-12-26 00:00"], ["2023-12-26 18:00", "2023-12-27 00:00"]]
    m = E.eligibility(idx, closures, k_bars=2)
    assert not m.business_eligible.any()
    assert set(m.business_reason) <= {"BEFORE_REOPEN_12H", "AFTER_FORCE_FLAT_12H", "WINDOW_CROSSES_CLOSURE", "NO_FUTURE_BARS"}


def test_label_usability_is_per_target_and_every_origin_is_accounted():
    idx, closures, _ = _two_weeks(True)
    st = np.full(len(idx), "LABELLED", object)
    st[50] = "PURGED_SUPPORT_CROSSES_CUT"
    m = E.eligibility(idx, closures, k_bars=2, label_status={"t1": st})
    assert m.business_eligible.iloc[50] and m.t1_reason.iloc[50] == "LABEL_NOT_USABLE" and not m.t1_eligible.iloc[50]
    c = E.counts(m, {"all": ["2023-01-01", "2023-02-01"]}, "t1_reason")
    assert c["all"]["origins"] == len(idx)


def test_bar_inside_a_declared_closure_is_flagged_and_bad_closures_refused():
    idx = pd.date_range("2023-01-09 00:00", periods=40, freq="h", tz="UTC")
    m = E.eligibility(idx, [["2023-01-09 10:00", "2023-01-09 12:00"]], k_bars=2)
    assert (m.business_reason.iloc[10:12] == "INSIDE_CLOSURE").all()
    with pytest.raises(E.EligibilityError, match="CLOSURE_OVERLAP"):
        E.parse_closures([["2023-01-01", "2023-01-03"], ["2023-01-02", "2023-01-04"]])
    with pytest.raises(E.EligibilityError, match="INDEX_NOT_UTC"):
        E.eligibility(pd.date_range("2023-01-01", periods=5, freq="h"), [], k_bars=2)


def test_digest_changes_with_the_mask():
    idx, closures, _ = _two_weeks(True)
    a = E.eligibility(idx, closures, k_bars=2)
    b = E.eligibility(idx, closures, k_bars=3)
    assert E.digest(a) != E.digest(b) and E.digest(a) == E.digest(E.eligibility(idx, closures, k_bars=2))
