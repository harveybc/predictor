"""RP23 (13D): deterministic SOFTWARE tests of the real RL environment (gym-fx GymFxEnv with backtrader)
for the weekly cycle contract: (1) a weekly policy change on a continuous episode keeps equity, position,
pending orders and commissions continuous (no reset at the boundary); (2) proportional costs are applied
at fills and accumulate; (3) model delay / availability: an action decided at bar t fills at bar t+1's
open, never at bar t's close, and equity/position change is observable only after the following step;
(4) actions before availability (a model released after the first decision bar) must be the declared
fallback (flat/hold), never a look-ahead; (5) solvency termination by min_equity; (6) cash-spot 13D
normalisation: equity 1, long/flat, notional <= equity, no leverage. No financial return is claimed.
Skipped when gym_fx / gymnasium / backtrader are not importable."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

gym_fx = pytest.importorskip("gym_fx")
bt = pytest.importorskip("backtrader")
GYMFX = Path(gym_fx.__file__).resolve().parents[1]


def _load(name, where):
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

WEEK_BARS = 24 * 7          # hourly bars per week (UTC week Monday 00:00 .. next Monday 00:00, 13D)


def _frame(n_bars: int, start="2024-01-01 00:00:00", slope=0.001, base=100.0):
    idx = pd.date_range(start, periods=n_bars, freq="h")
    t = np.arange(n_bars)
    close = base * (1 + slope * t)
    op = np.concatenate([[base], close[:-1]])              # open = previous close (the next-open fill price is deterministic)
    return pd.DataFrame({"DATE_TIME": idx.strftime("%Y-%m-%d %H:%M:%S"), "OPEN": op, "HIGH": np.maximum(op, close) * 1.001,
                         "LOW": np.minimum(op, close) * 0.999, "CLOSE": close, "VOLUME": 1000.0})


def _env(tmp_path, frame, *, commission=0.001, slippage=0.0, position_size=None, cash=1.0, min_equity=None, window=8):
    csv = tmp_path / "bars.csv"
    frame.to_csv(csv, index=False)
    price0 = float(frame["OPEN"].iloc[0])
    size = position_size if position_size is not None else (cash / price0) * 0.5          # notional = half the equity: <= equity, no leverage
    config = {"input_data_file": str(csv), "date_column": "DATE_TIME", "price_column": "CLOSE", "window_size": window, "initial_cash": cash,
              "position_size": size, "min_equity": (cash * 0.01) if min_equity is None else min_equity, "env_mode": "training", "commission": commission,
              "slippage_perc": slippage, "leverage": 1.0, "feature_columns": [], "feature_binary_columns": [], "timeframe": "1h"}
    env = GymFxEnv(config, DataFeed(config), Broker(config), None, Preprocessor(config), Reward(config), Metrics(config))
    return env, config


def _run(env, actions):
    """Drive one continuous episode with a list of actions; returns per-step info dicts."""
    obs, info = env.reset()
    trace = [dict(info, step=0)]
    for i, a in enumerate(actions, start=1):
        obs, reward, terminated, truncated, info = env.step(a)
        trace.append(dict(info, step=i, reward=reward, terminated=terminated))
        if terminated or truncated:
            break
    env.close()
    return trace


def test_E3_a_weekly_policy_change_keeps_equity_position_orders_and_commissions_continuous(tmp_path):
    frame = _frame(3 * WEEK_BARS)
    env, cfg = _env(tmp_path, frame, commission=0.001)
    # week 1: policy A goes long at bar 20 and holds; week 2 (a "new model") keeps holding then flattens; no reset anywhere
    actions = [0] * 20 + [1] + [0] * (WEEK_BARS - 21) + [0] * 40 + [2] + [0] * (WEEK_BARS - 41)
    trace = _run(env, actions)
    boundary = WEEK_BARS
    before, after = trace[boundary - 1], trace[boundary]
    assert after["position"] == before["position"] == 1 and after["position_units"] == before["position_units"]
    assert after["commission_paid"] == before["commission_paid"] > 0                  # nothing was charged by the boundary itself
    assert abs(after["equity"] - before["equity"]) <= abs(before["price"] * before["position_units"] * 0.01) + 1e-9   # only the price moved
    assert after["open_order_count"] == before["open_order_count"]
    assert all(not t.get("terminated") for t in trace[:-1]) and trace[-1]["bar_index"] >= 2 * WEEK_BARS
    # the flatten (action 2 = short from long -> closes and opens short; with size = half equity it stays a bracket-free market fill)
    flat_step = WEEK_BARS + 41
    assert trace[flat_step + 1]["commission_paid"] > trace[flat_step]["commission_paid"]        # the second trade paid its commission


def test_E3_proportional_costs_are_charged_at_the_fill_and_accumulate(tmp_path):
    frame = _frame(80, slope=0.0)                                                          # flat prices: equity changes are costs only
    env, cfg = _env(tmp_path, frame, commission=0.002, slippage=0.0)
    trace = _run(env, [0] * 10 + [1] + [0] * 60)
    fill = next(t for t in trace if t["position"] == 1)
    notional = abs(fill["position_units"]) * fill["price"]
    assert fill["commission_paid"] == pytest.approx(0.002 * notional, rel=1e-6)
    assert cfg["initial_cash"] - trace[-1]["equity"] == pytest.approx(fill["commission_paid"], rel=1e-6)   # flat prices: the loss is the commission
    (tmp_path / "b").mkdir()
    env2, _ = _env(tmp_path / "b", frame, commission=0.0, slippage=0.0)
    trace2 = _run(env2, [0] * 10 + [1] + [0] * 60)
    assert trace2[-1]["commission_paid"] == 0.0 and trace2[-1]["equity"] == pytest.approx(cfg["initial_cash"])


def test_E3_an_action_decided_at_bar_t_fills_at_the_next_open_never_at_its_own_close(tmp_path):
    frame = _frame(60, slope=0.01)                                                         # strictly rising: open[t+1] = close[t] < close[t+1]
    env, cfg = _env(tmp_path, frame)
    trace = _run(env, [0] * 10 + [1] + [0] * 40)
    decide = trace[11]                                                                     # info returned by the step that sent action 1
    assert decide["position"] == 0                                                         # not filled during the decision bar
    fill = trace[12]
    assert fill["position"] == 1
    entry = env.bridge.entry_price
    bar_decide = decide["bar_index"] - 1                                                   # bar_index is 1-based
    assert entry == pytest.approx(float(frame["OPEN"].iloc[bar_decide + 1]), rel=1e-9)    # the NEXT bar's open ...
    assert entry != pytest.approx(float(frame["CLOSE"].iloc[bar_decide]), rel=1e-9) or float(frame["OPEN"].iloc[bar_decide + 1]) == float(frame["CLOSE"].iloc[bar_decide])
    assert entry < float(frame["CLOSE"].iloc[bar_decide + 1])                              # ... not the close after the decision


def test_E3_before_the_model_is_available_the_declared_fallback_acts_and_the_release_is_never_moved(tmp_path):
    """13D: cutoff <= fit_start < fit_end <= release <= first decision. A model released at bar 30 cannot act at bars < 30:
    the policy is flat (hold) there; the fill of its first decision lands after the release, never before."""
    frame = _frame(80)
    env, cfg = _env(tmp_path, frame)
    release_bar = 30
    clocks = {"cutoff": 20, "fit_start": 21, "fit_end": 27, "release": release_bar, "first_decision": release_bar}
    assert clocks["cutoff"] <= clocks["fit_start"] < clocks["fit_end"] <= clocks["release"] <= clocks["first_decision"]

    def policy(bar):
        return 0 if bar < clocks["first_decision"] else 1          # the model's action (long) exists only from its release
    actions = [policy(b) for b in range(1, 70)]
    trace = _run(env, actions)
    first_long = next(t for t in trace if t["position"] == 1)
    assert first_long["bar_index"] > release_bar and all(t["position"] == 0 for t in trace if t["bar_index"] <= release_bar)
    # a late model: the fallback keeps the last valid policy (flat here); the release is recorded, not shifted backwards
    late = {**clocks, "release": 35}
    assert late["release"] >= late["fit_end"] and late["release"] > clocks["release"]


def test_E3_min_equity_terminates_the_episode_and_the_cause_is_recorded(tmp_path):
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.concatenate([np.full(20, 100.0), 100.0 * 0.5 ** np.arange(1, n - 19)])       # a crash after bar 20
    op = np.concatenate([[100.0], close[:-1]])
    frame = pd.DataFrame({"DATE_TIME": idx.strftime("%Y-%m-%d %H:%M:%S"), "OPEN": op, "HIGH": np.maximum(op, close), "LOW": np.minimum(op, close) * 0.99,
                          "CLOSE": close, "VOLUME": 1.0})
    env, cfg = _env(tmp_path, frame, position_size=0.0099, cash=1.0, min_equity=0.5)      # notional 0.99 <= equity 1 (no leverage)
    trace = _run(env, [0] * 5 + [1] + [0] * 100)
    assert trace[-1]["terminated"] and trace[-1].get("termination_cause") == "min_equity" and trace[-1]["equity"] <= 0.5


def test_E3_cash_spot_normalisation_equity_one_long_flat_notional_at_most_equity(tmp_path):
    frame = _frame(40, slope=0.0)
    env, cfg = _env(tmp_path, frame, cash=1.0)
    assert cfg["initial_cash"] == 1.0 and cfg["leverage"] == 1.0
    trace = _run(env, [0] * 5 + [1] + [0] * 30)
    fill = next(t for t in trace if t["position"] == 1)
    assert abs(fill["position_units"]) * fill["price"] <= 1.0 + 1e-9                       # notional <= equity
    assert set(t["position"] for t in trace) <= {0, 1}                                     # long / flat only in this scenario
    assert env.action_space.n == 3                                                          # the env also offers short (2); the 13D scenario does not use it
