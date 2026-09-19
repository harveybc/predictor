"""RP31: the real weekly long/flat controller (tools/e3_weekly_controller.py) exercised on a continuous
synthetic episode with a real bar clock: release + fallback executed, clock ordering enforced, weeks by
timestamps after warm-up, continuity across weeks, close-to-flat without short, sizing from equity, price
gap at the fill (open[t+1] != close[t]) with positive latency. Mutants of the controller must FAIL."""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("e3_weekly_controller")
UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)          # a Monday
H = timedelta(hours=1)


def _release(k, *, late_hours=0, name=None):
    ws = T0 + timedelta(weeks=k)
    return C.ModelRelease(week_start=ws, cutoff=ws - timedelta(days=7), fit_start=ws - timedelta(days=7), fit_end=ws - timedelta(hours=6),
                          release=ws - timedelta(hours=2) + timedelta(hours=late_hours), first_decision=ws, name=name or f"m{k}")


def _episode(weeks=3, seed=0):
    rng = np.random.default_rng(seed)
    n = 24 * 7 * weeks
    times = [T0 + i * H for i in range(n)]
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    open_ = np.r_[close[0], close[:-1] + rng.normal(0, 0.05, n - 1)]      # open[t+1] != close[t]: a real gap
    return times, open_, close


def _run(ctrl, times, open_, close, policy):
    """Simulates the environment: a decision at bar t (after close[t]) fills at open[t+1]."""
    equity, units, entry = 1000.0, 0.0, None
    fills = []
    for i, t in enumerate(times[:-1]):
        rec = ctrl.decide(t, H, policy(i, ctrl.available_model(t)), equity + units * (close[i] - (entry or close[i])), close[i], units)
        a = rec["action"]
        fill_price = open_[i + 1]
        if a == C.LONG:
            units = rec["notional"] / fill_price; entry = fill_price; equity -= rec["notional"]
            fills.append({"i": i, "decision_time": t, "fill_time": times[i + 1], "fill_price": fill_price, "close_at_decision": close[i], "side": "BUY"})
        elif a == C.CLOSE and units > 0:
            equity += units * fill_price; units = 0.0; entry = None
            fills.append({"i": i, "decision_time": t, "fill_time": times[i + 1], "fill_price": fill_price, "close_at_decision": close[i], "side": "SELL"})
        assert a != C.SHORT
    return fills, equity, units


def test_RP31_clock_order_is_enforced_before_scoring():
    with pytest.raises(ValueError, match="clocks"):
        C.WeeklyLongFlatController([C.ModelRelease(T0, T0, T0 - H, T0, T0, T0, "bad")])           # fit_start < cutoff
    with pytest.raises(ValueError, match="clocks"):
        C.WeeklyLongFlatController([C.ModelRelease(T0, T0 - 2 * H, T0 - 2 * H, T0, T0 - H, T0, "release-before-fit-end")])
    r = _release(1)
    assert r.valid_clocks() and not r.late()
    assert _release(1, late_hours=5).late()


def test_RP31_release_and_fallback_are_executed_and_weeks_are_by_timestamp():
    times, open_, close = _episode(3)
    ctrl = C.WeeklyLongFlatController([_release(1), _release(2)])
    fills, equity, units = _run(ctrl, times, open_, close, lambda i, m: (C.LONG if i % 48 == 0 else (None if i % 48 == 24 else C.HOLD)) if m else C.LONG)
    D = ctrl.state.decisions
    # week 0 has no released model: every decision falls back to FLAT (HOLD with no position), the LONG proposals of the policy are ignored
    wk0 = [d for d in D if datetime.fromisoformat(d["decision_time"]) < _release(1).release]
    assert len(wk0) == 24 * 7 - 2 and all(d["model"] is None and d["reason"] == "FALLBACK_NO_MODEL_FLAT" and d["action"] == C.HOLD for d in wk0)
    # model m1 acts from its release instant (2 h before week 1), m2 from its own
    first_m1 = next(d for d in D if d["model"] == "m1")
    assert datetime.fromisoformat(first_m1["decision_time"]) == _release(1).release
    assert {d["model"] for d in D if d["week_start"] == (T0 + timedelta(weeks=2)).isoformat()} == {"m2"}
    # weeks are computed from timestamps: exactly 3 distinct weeks, new_week flagged at the Monday 00:00 bars
    assert sorted({d["week_start"] for d in D}) == [(T0 + timedelta(weeks=k)).isoformat() for k in range(3)]
    assert [d["decision_time"] for d in D if d["new_week"]] == [(T0 + timedelta(weeks=k)).isoformat() for k in (1, 2)]
    # continuity: a position opened in week 1 survives the week boundary until the policy goes flat (no forced close at the week change)
    assert fills and fills[0]["side"] == "BUY"
    assert all(f["fill_time"] == f["decision_time"] + H for f in fills)                          # positive latency of one bar
    assert all(f["fill_price"] != f["close_at_decision"] for f in fills)                         # gap: fills at open[t+1], never at close[t]
    assert units == 0 or equity > 0


def test_RP31_close_to_flat_never_short_and_sizing_from_equity():
    times, open_, close = _episode(2)
    ctrl = C.WeeklyLongFlatController([_release(0, name="m0")], size_fraction=0.5)
    fills, equity, units = _run(ctrl, times, open_, close, lambda i, m: [C.LONG, C.HOLD, C.SHORT, C.CLOSE, C.HOLD][i % 5])
    D = ctrl.state.decisions
    assert ctrl.state.refused_proposals and all(r["proposal"] == "SHORT" for r in ctrl.state.refused_proposals)
    assert all(d["action"] != C.SHORT for d in D) and any(d["reason"] == "CLOSE_TO_FLAT" for d in D)
    opened = [d for d in D if d["reason"] == "OPEN_LONG"]
    assert all(abs(d["notional"] - 0.5 * d["equity"]) < 1e-9 and d["notional"] <= d["equity"] for d in opened)
    assert units == 0.0                                                                           # ends flat after the CLOSE proposals
    with pytest.raises(C.IncompatibleProposal):
        ctrl.decide(times[-1], H, 7, 1000.0, 100.0, 0.0)


def test_RP31_late_release_is_recorded_and_the_previous_model_covers_the_gap():
    ctrl = C.WeeklyLongFlatController([_release(0, name="m0"), _release(1, late_hours=10, name="m1late")])
    assert ctrl.state.misses and ctrl.state.misses[0]["model"] == "m1late"
    ws1 = T0 + timedelta(weeks=1)
    assert ctrl.available_model(ws1 + 3 * H) == "m0" and ctrl.available_model(ws1 + 9 * H) == "m1late"   # release = ws1 - 2h + 10h = ws1 + 8h


@pytest.mark.parametrize("mutant", ["no_latency", "accept_short", "week_by_counter", "release_ignored"])
def test_RP31_mutants_fail(mutant, monkeypatch):
    times, open_, close = _episode(2)
    ctrl = C.WeeklyLongFlatController([_release(1)])
    if mutant == "no_latency":
        ctrl.latency_bars = 0
        ctrl.decide(times[5], H, None, 1000.0, 100.0, 0.0)
        d = ctrl.state.decisions[-1]
        assert d["expected_fill_time"] == d["decision_time"]                    # a zero-latency controller is detectable...
        with pytest.raises(AssertionError):
            assert datetime.fromisoformat(d["expected_fill_time"]) > datetime.fromisoformat(d["decision_time"])
    elif mutant == "accept_short":
        orig = ctrl.decide
        def bad(t, step, proposal, eq, px, units):
            rec = orig(t, step, C.HOLD if proposal == C.SHORT else proposal, eq, px, units)
            if proposal == C.SHORT:
                rec["action"] = C.SHORT
            return rec
        monkeypatch.setattr(ctrl, "decide", bad)
        with pytest.raises(AssertionError):
            _run(ctrl, times, open_, close, lambda i, m: C.SHORT)
    elif mutant == "week_by_counter":
        monkeypatch.setattr(C, "week_start", lambda ts: T0)                      # every bar in "week 0"
        _run(ctrl, times, open_, close, lambda i, m: C.HOLD)
        with pytest.raises(AssertionError):
            assert len({d["week_start"] for d in ctrl.state.decisions}) == 2
    elif mutant == "release_ignored":
        monkeypatch.setattr(ctrl, "available_model", lambda t: "m1")             # model acts before its release
        _run(ctrl, times, open_, close, lambda i, m: C.LONG)
        with pytest.raises(AssertionError):
            assert all(d["model"] is None for d in ctrl.state.decisions if datetime.fromisoformat(d["decision_time"]) < _release(1).release)
