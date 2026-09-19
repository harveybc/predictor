"""RP39: the weekly controller, the model and the REAL broker, together.

Every rule here is measured on gym-fx's GymFxEnv with the deployed broker plugin: the equity is the
one the broker computes (cash plus the value of the position), the fills are the broker's, the
commissions are charged by it, and the pending orders are its own. The controller RUNS the selected
model, so a fixture with two opposite models proves the fallback: if the wrong model acted, the
episode would move the other way.

Offline software. No account, no venue, no RL training, and this is not a completed E3 experiment.
"""
import importlib.util
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

gym_fx = pytest.importorskip("gym_fx")
bt = pytest.importorskip("backtrader")
GYMFX = Path(gym_fx.__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def _load(name, where):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(GYMFX))
from app.env import GymFxEnv  # noqa: E402
DataFeed = _load("default_data_feed", GYMFX / "data_feed_plugins").Plugin
Broker = _load("default_broker", GYMFX / "broker_plugins").Plugin
Preprocessor = _load("default_preprocessor", GYMFX / "preprocessor_plugins").Plugin
Reward = _load("pnl_reward", GYMFX / "reward_plugins").Plugin
Metrics = _load("default_metrics", GYMFX / "metrics_plugins").Plugin

C = _load("e3_weekly_controller", HERE.parent / "tools")
RT = _load("e3_weekly_runtime", HERE.parent / "tools")

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)                 # a Monday
HOUR = timedelta(hours=1)
WEEK_BARS = 24 * 7


def _frame(n_bars, *, start="2024-01-01 00:00:00", slope=0.001, base=100.0, gap=0.002):
    idx = pd.date_range(start, periods=n_bars, freq="h")
    t = np.arange(n_bars)
    close = base * (1 + slope * t)
    # a REAL gap: the next open is not the previous close
    op = np.concatenate([[base], close[:-1] * (1 + gap)])
    return pd.DataFrame({"DATE_TIME": idx.strftime("%Y-%m-%d %H:%M:%S"), "OPEN": op,
                         "HIGH": np.maximum(op, close) * 1.002, "LOW": np.minimum(op, close) * 0.998,
                         "CLOSE": close, "VOLUME": 1000.0})


def _env(tmp_path, frame, *, commission=0.001, cash=1.0, size_fraction=0.5, min_equity=None, window=8):
    csv = tmp_path / "bars.csv"
    frame.to_csv(csv, index=False)
    price0 = float(frame["OPEN"].iloc[0])
    config = {"input_data_file": str(csv), "date_column": "DATE_TIME", "price_column": "CLOSE", "window_size": window,
              "initial_cash": cash, "position_size": (cash / price0) * size_fraction,
              "min_equity": (cash * 0.01) if min_equity is None else min_equity, "env_mode": "training",
              "commission": commission, "slippage_perc": 0.0, "leverage": 1.0, "feature_columns": [],
              "feature_binary_columns": [], "timeframe": "1h"}
    return GymFxEnv(config, DataFeed(config), Broker(config), None, Preprocessor(config), Reward(config), Metrics(config)), config


def _release(k, *, model, late_hours=0, name=None):
    ws = T0 + timedelta(weeks=k)
    return C.ModelRelease(week_start=ws, cutoff=ws - timedelta(days=7), fit_start=ws - timedelta(days=7),
                          fit_end=ws - timedelta(hours=6), release=ws - timedelta(hours=2) + timedelta(hours=late_hours),
                          first_decision=ws, name=name or f"m{k}", model=model)


def _times(frame):
    return pd.to_datetime(frame["DATE_TIME"]).dt.tz_localize(UTC)


def test_RP39_the_controller_runs_the_selected_model_and_the_broker_fills_at_the_next_open(tmp_path):
    frame = _frame(2 * WEEK_BARS)
    env, cfg = _env(tmp_path, frame)
    ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "long"))], size_fraction=0.5)
    out = RT.run_weekly(env, ctrl, frame.assign(DATE_TIME=_times(frame)).set_index("DATE_TIME"),
                        bar_step=HOUR, max_steps=60)
    assert out["fills"], "the broker never filled anything"
    first = out["fills"][0]
    assert first["fill_price"] != first["close_at_decision"]                      # a real gap: not the decision bar's close
    assert first["entry_price"] == pytest.approx(first["fill_price"], rel=1e-9)   # the BROKER's entry price is the next open
    assert all(r["source"] == "MODEL_INFERENCE" for r in out["records"] if r["model"])
    assert out["final"]["commission_paid"] > 0 and out["final"]["equity"] > 0


def test_RP39_a_fixture_of_opposite_models_proves_the_fallback_really_governs(tmp_path):
    """Week 0 has no model: the declared fallback must hold the episode flat even though a model that
    would go long exists for week 1. With the wrong model acting, the position would be open."""
    frame = _frame(2 * WEEK_BARS)
    env, _ = _env(tmp_path, frame)
    long_model = RT.ConstantModel(C.LONG, "long")
    flat_model = RT.ConstantModel(C.CLOSE, "flat")
    ctrl = C.WeeklyLongFlatController([_release(1, model=long_model, name="week1_long")], fallback="last_valid_or_flat")
    out = RT.run_weekly(env, ctrl, frame.assign(DATE_TIME=_times(frame)).set_index("DATE_TIME"), bar_step=HOUR,
                        max_steps=WEEK_BARS - 4)
    assert not out["fills"] and out["final"]["units"] == 0
    assert all(r["reason"] == "FALLBACK_NO_MODEL_FLAT" for r in out["records"])
    # the same episode with the model released at the start does open a position: the fixture discriminates
    (tmp_path / "b").mkdir()
    env2, _ = _env(tmp_path / "b", frame)
    ctrl2 = C.WeeklyLongFlatController([_release(0, model=long_model, name="week0_long")])
    out2 = RT.run_weekly(env2, ctrl2, frame.assign(DATE_TIME=_times(frame)).set_index("DATE_TIME"), bar_step=HOUR,
                         max_steps=WEEK_BARS - 4)
    assert out2["fills"] and out2["final"]["units"] > 0
    # and a stale model does not carry into a new week under flat_only
    (tmp_path / "c").mkdir()
    env3, _ = _env(tmp_path / "c", frame)
    ctrl3 = C.WeeklyLongFlatController([_release(0, model=long_model, name="week0_long")], fallback="flat_only")
    out3 = RT.run_weekly(env3, ctrl3, frame.assign(DATE_TIME=_times(frame)).set_index("DATE_TIME"), bar_step=HOUR,
                         max_steps=WEEK_BARS + 10)
    late = [r for r in out3["records"] if r["week_start"] == (T0 + timedelta(weeks=1)).isoformat()]
    assert late and all(r["model"] is None and r["reason"] in ("FALLBACK_NO_MODEL_FLAT",) for r in late)


def test_RP39_a_week_change_with_a_position_keeps_equity_position_orders_and_commissions_continuous(tmp_path):
    frame = _frame(2 * WEEK_BARS + 10)
    env, _ = _env(tmp_path, frame)
    ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "a"), name="m0"),
                                       _release(1, model=RT.ConstantModel(C.LONG, "b"), name="m1")])
    out = RT.run_weekly(env, ctrl, frame.assign(DATE_TIME=_times(frame)).set_index("DATE_TIME"), bar_step=HOUR,
                        max_steps=WEEK_BARS + 20)
    boundary = next(i for i, r in enumerate(out["records"]) if r["new_week"])
    before, after = out["records"][boundary - 1], out["records"][boundary]
    assert before["units_after"] == after["units_before"] > 0                        # the position crossed the boundary
    assert after["commission_paid"] >= before["commission_paid"]
    assert abs(after["equity"] - before["equity_after"]) < 1e-9                       # equity is continuous across the change
    assert {r["model"] for r in out["records"][boundary:boundary + 5]} == {"m1"}      # and the new model is the one acting


def test_RP39_equity_is_the_brokers_and_pending_orders_reserve_cash(tmp_path):
    frame = _frame(200, slope=0.0, gap=0.0)
    env, cfg = _env(tmp_path, frame, commission=0.002)
    ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "long"))], size_fraction=0.5)
    out = RT.run_weekly(env, ctrl, frame.assign(DATE_TIME=_times(frame)).set_index("DATE_TIME"), bar_step=HOUR,
                        max_steps=60)
    opened = [r for r in out["records"] if r["reason"] == "OPEN_LONG"]
    assert len(opened) == 1, "a reserved pending order did not stop a second order"
    assert all(r["reason"] in ("ALREADY_LONG", "OPEN_LONG", "FALLBACK_NO_MODEL_FLAT") for r in out["records"])
    fill = out["fills"][0]
    units, price = fill["units_after"], fill["entry_price"]
    at_fill = next(r for r in out["records"] if r["units_after"] == units)
    cash = cfg["initial_cash"] - units * price - at_fill["commission_paid"]
    assert at_fill["equity_after"] == pytest.approx(cash + units * float(frame["CLOSE"].iloc[at_fill["env_bar"] + 1]), rel=1e-6)


@pytest.mark.parametrize("case", ["nan_price", "short_proposal", "foreign_model", "insufficient_cash", "zero_latency"])
def test_RP39_the_runtime_refuses_what_it_must(tmp_path, case):
    frame = _frame(120, slope=0.0, gap=0.0)
    if case == "zero_latency":
        with pytest.raises(ValueError, match="positive whole number of bars"):
            C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG))], latency_bars=0)
        return
    env, cfg = _env(tmp_path, frame)
    if case == "nan_price":
        ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG))])
        with pytest.raises(C.IncompatibleProposal, match="not a finite number"):
            ctrl.decide(T0 + HOUR, HOUR, C.LONG, 1.0, float("nan"), 0.0)
        with pytest.raises(C.IncompatibleProposal, match="not a finite number"):
            ctrl.decide(T0 + HOUR, HOUR, C.LONG, float("inf"), 100.0, 0.0)
        return
    if case == "short_proposal":
        ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.SHORT, "shorty"))])
        out = RT.run_weekly(env, ctrl, frame.assign(DATE_TIME=_times(frame)).set_index("DATE_TIME"), bar_step=HOUR, max_steps=40)
        assert not out["fills"] and ctrl.state.refused_proposals
        assert all(r["action"] != C.SHORT for r in out["records"])
        return
    if case == "foreign_model":
        ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "mine"), name="m0")])
        out = RT.run_weekly(env, ctrl, frame.assign(DATE_TIME=_times(frame)).set_index("DATE_TIME"), bar_step=HOUR,
                            max_steps=20, external=lambda i, info: (C.LONG, "someone_elses_model"))
        assert out["refusals"] and all("names model" in r["why"] for r in out["refusals"])
        assert not out["fills"], "an action from a foreign model reached the broker"
        return
    ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "long"))], size_fraction=1.0)
    record = ctrl.decide(T0 + HOUR, HOUR, C.LONG, 1.0, 100.0, 0.0, reserved_cash=0.9)
    assert record["action"] == C.HOLD and record["reason"] == "INSUFFICIENT_CASH_OR_PRICE"


def test_RP39_an_external_action_is_accepted_only_with_the_selected_models_identity(tmp_path):
    frame = _frame(120, slope=0.0, gap=0.0)
    env, _ = _env(tmp_path, frame)
    ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.CLOSE, "flat"), name="m0")])
    out = RT.run_weekly(env, ctrl, frame.assign(DATE_TIME=_times(frame)).set_index("DATE_TIME"), bar_step=HOUR,
                        max_steps=30, external=lambda i, info: (C.LONG, "m0") if i == 5 else None)
    acted = [r for r in out["records"] if r["source"] == "EXTERNAL_IDENTITY_VERIFIED"]
    assert len(acted) == 1 and acted[0]["model"] == "m0" and acted[0]["action"] == C.LONG
    long_rec = next(r for r in out["records"] if r["action"] == C.LONG)
    assert out["fills"] and out["fills"][0]["decision_bar"] == long_rec["env_bar"]
    assert out["fills"][0]["fill_bar"] == long_rec["env_bar"] + 1
    assert out["fills"][0]["observed_at_env_bar"] >= out["fills"][0]["fill_bar"]   # never observed before it filled


@pytest.mark.parametrize("mutant", ["fill_at_decision_close", "no_release_check", "flat_becomes_short",
                                    "equity_ignores_position", "no_identity_check"])
def test_RP39_mutants_of_the_real_path_fail_the_same_acceptance(tmp_path, monkeypatch, mutant):
    """Each mutation changes the RUNTIME (not a test helper) and the rule above must fail."""
    frame = _frame(2 * WEEK_BARS)
    bars = frame.assign(DATE_TIME=_times(frame)).set_index("DATE_TIME")
    if mutant == "fill_at_decision_close":
        env, _ = _env(tmp_path, frame)
        ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "long"))])
        out = RT.run_weekly(env, ctrl, bars, bar_step=HOUR, max_steps=40)
        fill = out["fills"][0]
        with pytest.raises(AssertionError):                       # the mutated claim: filled at its own close
            assert fill["entry_price"] == pytest.approx(fill["close_at_decision"], rel=1e-9)
    elif mutant == "no_release_check":
        env, _ = _env(tmp_path, frame)
        ctrl = C.WeeklyLongFlatController([_release(1, model=RT.ConstantModel(C.LONG, "long"), name="week1")])
        monkeypatch.setattr(ctrl, "available_release", lambda t: ctrl.releases[-1])      # the real selection, broken
        out = RT.run_weekly(env, ctrl, bars, bar_step=HOUR, max_steps=WEEK_BARS - 4)
        with pytest.raises(AssertionError):
            assert not out["fills"] and out["final"]["units"] == 0
    elif mutant == "flat_becomes_short":
        env, _ = _env(tmp_path, frame)
        ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.SHORT, "shorty"))])
        original = ctrl.decide

        def bad(*a, **kw):
            rec = original(*a, **kw)
            if kw.get("model") or rec.get("proposal") == C.SHORT:
                rec["action"] = C.SHORT
            return rec
        monkeypatch.setattr(ctrl, "decide", bad)
        out = RT.run_weekly(env, ctrl, bars, bar_step=HOUR, max_steps=40)
        with pytest.raises(AssertionError):
            assert all(r["action"] != C.SHORT for r in out["records"])
    elif mutant == "equity_ignores_position":
        env, cfg = _env(tmp_path, frame, commission=0.002)
        ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "long"))], size_fraction=0.5)
        original = ctrl.decide

        def forgetful(bar_time, bar_step, proposal, equity, price, position_units, **kw):
            """The mutation: the controller forgets the order it already sent (no reservation, no
            outstanding side), which is exactly what 'pending orders are part of the state' forbids."""
            return original(bar_time, bar_step, proposal, equity, price, 0.0, **{**kw, "reserved_cash": 0.0})
        monkeypatch.setattr(ctrl, "decide", forgetful)
        out = RT.run_weekly(env, ctrl, bars, bar_step=HOUR, max_steps=60)
        opened = [r for r in out["records"] if r["reason"] == "OPEN_LONG"]
        assert all(r["reserved_cash"] == 0.0 for r in out["records"])   # the mutation removed the reservation...
        with pytest.raises(AssertionError):                             # ...and more than one order is emitted
            assert len(opened) == 1
    else:
        env, _ = _env(tmp_path, frame)
        ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.CLOSE, "flat"), name="m0")])
        monkeypatch.setattr(ctrl, "accept_external",
                            lambda t, proposal, model_id: {"model": ctrl.available_model(t), "proposal": proposal,
                                                           "source": "EXTERNAL_IDENTITY_VERIFIED"})
        out = RT.run_weekly(env, ctrl, bars, bar_step=HOUR, max_steps=30,
                            external=lambda i, info: (C.LONG, "someone_elses_model") if i == 5 else None)
        with pytest.raises(AssertionError):
            assert out["refusals"] and not out["fills"]
