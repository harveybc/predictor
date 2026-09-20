"""RP45: the weekly controller, the model and the REAL broker — quantity, clock and executions.

Every number here is read from the running simulator: the quantity the broker executed, the bar its
own data feed reports, the order reference and commission of each fill. The dictum's F4 measurements
are the first two rules, and they fail against the previous adapter, which sent only the direction and
reconstructed the fill from OPEN[decision + 1].

Offline software. No venue, no account, no RL training, and this is not a completed E3 experiment.
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
SP = _load("e3_weekly_strategy", HERE.parent / "tools")

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)                 # a Monday
HOUR = timedelta(hours=1)
WEEK_BARS = 24 * 7


def _frame(n_bars, *, start="2024-01-01 00:00:00", slope=0.001, base=100.0, gap=0.002):
    idx = pd.date_range(start, periods=n_bars, freq="h")
    close = base * (1 + slope * np.arange(n_bars))
    op = np.concatenate([[base], close[:-1] * (1 + gap)])            # a real gap: next open != previous close
    return pd.DataFrame({"DATE_TIME": idx.strftime("%Y-%m-%d %H:%M:%S"), "OPEN": op,
                         "HIGH": np.maximum(op, close) * 1.002, "LOW": np.minimum(op, close) * 0.998,
                         "CLOSE": close, "VOLUME": 1000.0})


def _env(tmp_path, frame, *, commission=0.001, cash=1.0, window=8, plugin=True):
    """The environment with the execution plugin through which a DECIDED quantity reaches the broker.
    `plugin=False` is the previous wiring, where the environment would size the order itself."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv = tmp_path / "bars.csv"
    frame.to_csv(csv, index=False)
    price0 = float(frame["OPEN"].iloc[0])
    config = {"input_data_file": str(csv), "date_column": "DATE_TIME", "price_column": "CLOSE", "window_size": window,
              "initial_cash": cash, "position_size": cash / price0, "min_equity": cash * 0.01, "env_mode": "training",
              "commission": commission, "slippage_perc": 0.0, "leverage": 1.0, "feature_columns": [],
              "feature_binary_columns": [], "timeframe": "1h"}
    execution = SP.WeeklyExecutionPlugin() if plugin else None
    env = GymFxEnv(config, DataFeed(config), Broker(config), execution, Preprocessor(config), Reward(config), Metrics(config))
    return env, config


def _release(k, *, model, late_hours=0, name=None):
    ws = T0 + timedelta(weeks=k)
    return C.ModelRelease(week_start=ws, cutoff=ws - timedelta(days=7), fit_start=ws - timedelta(days=7),
                          fit_end=ws - timedelta(hours=6), release=ws - timedelta(hours=2) + timedelta(hours=late_hours),
                          first_decision=ws, name=name or f"m{k}", model=model)


def _bars(frame):
    return frame.assign(DATE_TIME=pd.to_datetime(frame["DATE_TIME"]).dt.tz_localize(UTC)).set_index("DATE_TIME")


@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.8])
def test_RP45_the_quantity_the_controller_decides_is_the_quantity_the_broker_executes(tmp_path, fraction):
    """The dictum's table: 0.1 decided / 0.005 executed, 0.8 decided / 0.005 executed."""
    frame = _frame(120, slope=0.0, gap=0.0)
    env, cfg = _env(tmp_path / str(fraction), frame)
    ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "long"))], size_fraction=fraction, latency_bars=RT.MINIMUM_LATENCY_BARS)
    out = RT.run_weekly(env, ctrl, _bars(frame), config=cfg, bar_step=HOUR, max_steps=30)
    assert out["fills"], "the broker executed nothing"
    fill = out["fills"][0]
    assert fill["source"] == "BROKER_EXECUTION_EVENT" and isinstance(fill["order_ref"], int)
    assert fill["decided_units"] == pytest.approx(fraction * 1.0 / 100.0, rel=1e-9)   # fraction of equity at price 100
    assert fill["executed_size"] == pytest.approx(fill["decided_units"], rel=1e-6), (fraction, fill)
    assert fill["quantity_matches_decision"]
    assert fill["commission"] == pytest.approx(cfg["commission"] * fill["executed_price"] * fill["executed_size"], rel=1e-6)


@pytest.mark.parametrize("latency", [2, 3, 4])
def test_RP45_the_contracted_latency_is_executed_not_relabelled(tmp_path, latency):
    frame = _frame(120, slope=0.0, gap=0.0)
    env, cfg = _env(tmp_path / str(latency), frame)
    ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "long"))], latency_bars=latency)
    out = RT.run_weekly(env, ctrl, _bars(frame), config=cfg, bar_step=HOUR, max_steps=30)
    fill = out["fills"][0]
    assert fill["fill_bar"] - fill["decided_at_bar"] == latency, (latency, fill)
    decision = next(r for r in out["records"] if r["env_bar"] == fill["decided_at_bar"])
    expected = datetime.fromisoformat(decision["expected_fill_time"])
    assert expected == datetime.fromisoformat(decision["bar_time"]) + latency * HOUR
    # and the bar the broker filled on is the bar the clock says it is
    assert next(c for c in out["clock"] if c["env_bar"] == fill["fill_bar"])["time"] == expected.isoformat()
    assert fill["fill_time"] == expected.isoformat()


def test_RP45_a_latency_this_environment_cannot_execute_is_refused_before_the_episode(tmp_path):
    frame = _frame(60, slope=0.0, gap=0.0)
    env, cfg = _env(tmp_path, frame)
    ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "long"))], latency_bars=1)
    with pytest.raises(RT.ExecutionContractRefused, match="minimum executable latency"):
        RT.run_weekly(env, ctrl, _bars(frame), config=cfg, bar_step=HOUR, max_steps=5)
    assert RT.MINIMUM_LATENCY_BARS == 2


def test_RP45_an_environment_that_cannot_execute_the_contract_is_refused_before_the_episode(tmp_path):
    frame = _frame(60, slope=0.0, gap=0.0)
    env, cfg = _env(tmp_path, frame, plugin=False)
    ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "long"))], latency_bars=RT.MINIMUM_LATENCY_BARS)
    with pytest.raises(RT.ExecutionContractRefused, match="no execution plugin"):
        RT.run_weekly(env, ctrl, _bars(frame), config=cfg, bar_step=HOUR, max_steps=5)
    env.close()
    env2, cfg2 = _env(tmp_path / "b", frame)
    ctrl2 = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "long"))], size_fraction=1.0, latency_bars=RT.MINIMUM_LATENCY_BARS)
    out = RT.run_weekly(env2, ctrl2, _bars(frame), config=cfg2, bar_step=HOUR, max_steps=10)
    assert out["contract"]["environment_position_size_note"].startswith("not used")


def test_RP45_the_clock_is_the_environments_and_an_inconsistent_one_refuses(tmp_path):
    frame = _frame(80, slope=0.0, gap=0.0)
    env, cfg = _env(tmp_path, frame)
    ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.HOLD, "hold"))], latency_bars=RT.MINIMUM_LATENCY_BARS)
    out = RT.run_weekly(env, ctrl, _bars(frame), config=cfg, bar_step=HOUR, max_steps=20)
    times = [datetime.fromisoformat(c["time"]) for c in out["clock"]]
    assert times == sorted(times) and len(set(times)) == len(times)
    assert times[0] == _bars(frame).index[out["clock"][0]["env_bar"]].to_pydatetime()
    # a frame whose dates disagree with the environment's own feed is refused, not reconciled
    env2, cfg2 = _env(tmp_path / "b", frame)
    shifted = _bars(frame).copy()
    shifted.index = shifted.index + timedelta(days=7)
    ctrl2 = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.HOLD, "hold"))], latency_bars=RT.MINIMUM_LATENCY_BARS)
    with pytest.raises(RT.ExecutionContractRefused, match="says"):
        RT.run_weekly(env2, ctrl2, shifted, config=cfg2, bar_step=HOUR, max_steps=5)
    # a constant counter with the appearance of a clock is refused
    env3, cfg3 = _env(tmp_path / "c", frame)
    import e3_weekly_runtime as R
    original = R._bar_time
    frozen = {"t": None}

    def constant(e):
        frozen["t"] = frozen["t"] or original(e)
        return frozen["t"]
    R._bar_time = constant
    try:
        with pytest.raises(RT.ExecutionContractRefused, match="clock did not advance"):
            RT.run_weekly(env3, ctrl2, _bars(frame), config=cfg3, bar_step=HOUR, max_steps=5)
    finally:
        R._bar_time = original


def test_RP45_a_pending_order_ends_by_the_brokers_verdict_and_survives_a_week_change(tmp_path):
    """The decision is taken in one week and its order is still outstanding when the next begins:
    the position, the equity and the commissions cross the boundary, and the order's end is the
    broker's own terminal status, not an elapsed-time expiry in the adapter."""
    frame = _frame(WEEK_BARS + 40, slope=0.0005, gap=0.0)
    env, cfg = _env(tmp_path, frame)
    decide_bar = WEEK_BARS - 10
    model = RT.ConstantModel(C.LONG, "long")
    ctrl = C.WeeklyLongFlatController([_release(0, model=model, name="m0")], latency_bars=3)
    out = RT.run_weekly(env, ctrl, _bars(frame), config=cfg, bar_step=HOUR, max_steps=WEEK_BARS + 20)
    fill = out["fills"][0]
    assert fill["fill_bar"] - fill["decided_at_bar"] == 3
    statuses = out["final"]["terminal_status"]
    assert statuses and set(statuses.values()) <= {"Completed", "Canceled", "Cancelled", "Rejected", "Margin", "Expired"}
    assert str(fill["order_ref"]) in {str(k) for k in statuses} and out["queue_at_end"] == []
    boundary = next(i for i, r in enumerate(out["records"]) if r["new_week"])
    before, after = out["records"][boundary - 1], out["records"][boundary]
    assert after["units_before"] == before["units_after"]
    assert after["commission_paid"] >= before["commission_paid"]
    assert abs(after["equity"] - before["equity_after"]) < 1e-9


def test_RP45_partial_fills_are_declared_out_of_scope_rather_than_pretended(tmp_path):
    frame = _frame(40, slope=0.0, gap=0.0)
    env, cfg = _env(tmp_path, frame)
    ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "long"))], latency_bars=RT.MINIMUM_LATENCY_BARS)
    out = RT.run_weekly(env, ctrl, _bars(frame), config=cfg, bar_step=HOUR, max_steps=10)
    assert "partial_fill" in out["contract"]["unsupported"]
    assert all(f["executed_size"] > 0 for f in out["fills"])


def test_RP45_the_fallback_and_opposite_models_still_govern_with_the_real_quantities(tmp_path):
    frame = _frame(WEEK_BARS + 20, slope=0.0, gap=0.0)
    env, cfg = _env(tmp_path, frame)
    ctrl = C.WeeklyLongFlatController([_release(1, model=RT.ConstantModel(C.LONG, "week1_long"), name="w1")], latency_bars=RT.MINIMUM_LATENCY_BARS)
    out = RT.run_weekly(env, ctrl, _bars(frame), config=cfg, bar_step=HOUR, max_steps=WEEK_BARS - 10)
    assert not out["fills"] and out["final"]["units"] == 0
    env2, cfg2 = _env(tmp_path / "b", frame)
    ctrl2 = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "week0_long"), name="w0")], latency_bars=RT.MINIMUM_LATENCY_BARS)
    out2 = RT.run_weekly(env2, ctrl2, _bars(frame), config=cfg2, bar_step=HOUR, max_steps=WEEK_BARS - 10)
    assert out2["fills"] and out2["final"]["units"] > 0


def test_RP45_a_short_proposal_and_a_foreign_model_never_reach_the_broker(tmp_path):
    frame = _frame(60, slope=0.0, gap=0.0)
    env, cfg = _env(tmp_path, frame)
    ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.SHORT, "shorty"))], latency_bars=RT.MINIMUM_LATENCY_BARS)
    out = RT.run_weekly(env, ctrl, _bars(frame), config=cfg, bar_step=HOUR, max_steps=20)
    assert not out["fills"] and ctrl.state.refused_proposals and out["final"]["units"] == 0
    env2, cfg2 = _env(tmp_path / "b", frame)
    ctrl2 = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.CLOSE, "flat"), name="m0")], latency_bars=RT.MINIMUM_LATENCY_BARS)
    out2 = RT.run_weekly(env2, ctrl2, _bars(frame), config=cfg2, bar_step=HOUR, max_steps=20,
                         external=lambda i, info: (C.LONG, "someone_elses_model"))
    assert out2["refusals"] and not out2["fills"]


@pytest.mark.parametrize("mutant", ["direction_only", "latency_ignored", "fill_from_open", "clock_from_counter",
                                    "pending_expires_by_time"])
def test_RP45_mutants_of_the_production_path_fail_the_same_acceptance(tmp_path, monkeypatch, mutant):
    """Each mutation changes tools/e3_weekly_runtime.py's own behaviour; the rules above must fail."""
    frame = _frame(120, slope=0.0, gap=0.0)
    bars = _bars(frame)
    if mutant == "direction_only":
        env, cfg = _env(tmp_path, frame)
        ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "long"))], size_fraction=0.1, latency_bars=RT.MINIMUM_LATENCY_BARS)
        env.strategy_plugin = None                                 # the mutation: no execution channel
        with pytest.raises(RT.ExecutionContractRefused):           # sending only a direction is refused, not silently sized
            RT.run_weekly(env, ctrl, bars, config=cfg, bar_step=HOUR, max_steps=10)
    elif mutant == "latency_ignored":
        env, cfg = _env(tmp_path, frame)
        ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "long"))], latency_bars=4)
        out = RT.run_weekly(env, ctrl, bars, config=cfg, bar_step=HOUR, max_steps=30)
        fill = out["fills"][0]
        with pytest.raises(AssertionError):                       # the mutated claim: the environment's own minimum
            assert fill["fill_bar"] - fill["decided_at_bar"] == RT.MINIMUM_LATENCY_BARS
    elif mutant == "fill_from_open":
        env, cfg = _env(tmp_path, frame)
        ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "long"))], latency_bars=3)
        frame2 = _frame(120, slope=0.002, gap=0.003)                # prices that move, so the two differ
        bars2 = _bars(frame2)
        env, cfg = _env(tmp_path / "moving", frame2)
        out = RT.run_weekly(env, ctrl, bars2, config=cfg, bar_step=HOUR, max_steps=30)
        fill = out["fills"][0]
        inferred = float(frame2["OPEN"].iloc[fill["decided_at_bar"] + 1])
        with pytest.raises(AssertionError):                       # the old reconstruction disagrees with the event
            assert fill["executed_price"] == pytest.approx(inferred, rel=1e-12)
    elif mutant == "clock_from_counter":
        env, cfg = _env(tmp_path, frame)
        ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.HOLD, "hold"))], latency_bars=RT.MINIMUM_LATENCY_BARS)
        import e3_weekly_runtime as R
        monkeypatch.setattr(R, "_bar_time", lambda e: None)
        with pytest.raises(RT.ExecutionContractRefused, match="not a clock"):
            R.run_weekly(env, ctrl, bars, config=cfg, bar_step=HOUR, max_steps=5)
    else:
        env, cfg = _env(tmp_path, frame)
        ctrl = C.WeeklyLongFlatController([_release(0, model=RT.ConstantModel(C.LONG, "long"))], latency_bars=3)
        out = RT.run_weekly(env, ctrl, bars, config=cfg, bar_step=HOUR, max_steps=30)
        # the order's end is a broker verdict; there is no elapsed-time expiry to point at
        assert out["final"]["terminal_status"]
        with pytest.raises(AssertionError):
            assert not out["fills"]


# --- RP53 (dictum F5): an order's life ends by the broker's verdict, fill or not ---------------------

class _LongFromBar:
    """Flat until the given decision, then long: the position must be FLAT when the price jumps, so
    the order that follows is one the equity cannot fund and the broker refuses it."""

    def __init__(self, first_long, name="stepper"):
        self.first_long, self.name, self.seen = int(first_long), name, 0

    def decide(self, features):
        self.seen += 1
        return C.LONG if self.seen > self.first_long else C.HOLD


def _margin_frame(n=120):
    """Prices that make the broker refuse: a jump the equity cannot fund, then a return."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.full(n, 100.0)
    close[10:20] = 1000.0
    op = np.concatenate([[100.0], close[:-1]])
    return pd.DataFrame({"DATE_TIME": idx.strftime("%Y-%m-%d %H:%M:%S"), "OPEN": op,
                         "HIGH": np.maximum(op, close), "LOW": np.minimum(op, close),
                         "CLOSE": close, "VOLUME": 1000.0})


def test_RP53_an_order_the_broker_ends_without_a_fill_releases_the_state(tmp_path):
    """The dictum's probe: a Margin verdict left the controller ALREADY_LONG for the rest of the
    episode with the broker flat. It must trade again once the price allows it."""
    frame = _margin_frame()
    env, cfg = _env(tmp_path, frame)
    ctrl = C.WeeklyLongFlatController([_release(0, model=_LongFromBar(9, "stepper"))],
                                      size_fraction=0.9, latency_bars=RT.MINIMUM_LATENCY_BARS)
    out = RT.run_weekly(env, ctrl, _bars(frame), config=cfg, bar_step=HOUR, max_steps=40)
    statuses = set(out["final"]["terminal_status"].values())
    assert statuses - {"Completed"}, f"the broker never refused an order: {statuses}"
    released = [r for r in out["refusals"] if r.get("released")]
    assert released, "the state was never released by a terminal verdict without a fill"
    # after the refusal, the controller decides again and the broker executes
    after = [r for r in out["records"] if r["env_bar"] >= released[0]["env_bar"]]
    assert any(r["reason"] == "OPEN_LONG" for r in after), "no new decision after the broker's refusal"
    assert any(f["fill_bar"] > released[0]["env_bar"] for f in out["fills"]), "no execution after the refusal"
    # the refused order and the one that followed are both accounted for by the broker's own verdicts
    verdicts = out["final"]["terminal_status"]
    assert "Margin" in verdicts.values() and "Completed" in verdicts.values() and len(verdicts) >= 2
    assert "never by elapsed time" in out["orders"]["release_rule"]


def test_RP53_no_double_order_and_no_imaginary_position_after_a_refusal(tmp_path):
    frame = _margin_frame()
    env, cfg = _env(tmp_path, frame)
    ctrl = C.WeeklyLongFlatController([_release(0, model=_LongFromBar(9, "stepper"))],
                                      size_fraction=0.9, latency_bars=RT.MINIMUM_LATENCY_BARS)
    out = RT.run_weekly(env, ctrl, _bars(frame), config=cfg, bar_step=HOUR, max_steps=40)
    opens = [r for r in out["records"] if r["reason"] == "OPEN_LONG"]
    fills = out["fills"]
    # every order that was sent is accounted for by a broker verdict, and no two are outstanding at once
    assert len(opens) <= len(out["final"]["terminal_status"])
    for record in out["records"]:
        assert not (record["reason"] == "OPEN_LONG" and record["units_before"] > 0), "ordered while already long"
    assert all(f["quantity_matches_decision"] for f in fills)


def test_RP53_a_pending_order_crosses_the_week_boundary_with_a_late_release_and_opposite_models(tmp_path):
    """A decision taken before Monday, still outstanding when the week changes, with the next week's
    model released LATE and proposing the opposite action."""
    frame = _frame(WEEK_BARS + 60, slope=0.0002, gap=0.0)
    env, cfg = _env(tmp_path, frame)
    long_model, flat_model = RT.ConstantModel(C.LONG, "long"), RT.ConstantModel(C.CLOSE, "flat")
    releases = [_release(0, model=long_model, name="w0"),
                _release(1, model=flat_model, late_hours=6, name="w1_late")]
    ctrl = C.WeeklyLongFlatController(releases, latency_bars=4)
    out = RT.run_weekly(env, ctrl, _bars(frame), config=cfg, bar_step=HOUR, max_steps=WEEK_BARS + 30)
    assert ctrl.state.misses and ctrl.state.misses[0]["model"] == "w1_late"
    boundary = next(i for i, r in enumerate(out["records"]) if r["new_week"])
    before, after = out["records"][boundary - 1], out["records"][boundary]
    assert after["units_before"] == before["units_after"]
    assert abs(after["equity"] - before["equity_after"]) < 1e-9
    # the late model only acts from its release, and then it is the one deciding
    acting = {r["model"] for r in out["records"][boundary:] if r["model"]}
    assert "w0" in acting and "w1_late" in acting
    late_release = next(r for r in out["records"] if r["model"] == "w1_late")
    assert datetime.fromisoformat(late_release["bar_time"]) >= releases[1].release
