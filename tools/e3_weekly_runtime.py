#!/usr/bin/env python3
"""RP39: the weekly controller driving the REAL environment and broker (gym-fx GymFxEnv with the
deployed broker plugin), not a simulator written inside a test.

What this adapter is responsible for, and what it refuses:

  * it asks the CONTROLLER which model may act at this bar and runs THAT model's inference; an action
    from elsewhere is admitted only through `accept_external`, which checks the model's identity;
  * it reads equity, position, pending orders and commissions from the environment's own information,
    never from a variable of its own: equity is cash plus the value of the position, as the broker
    computes it, and the cash a pending order reserves is not available to a new one;
  * an action decided at bar t reaches the environment at bar t and the broker fills it at the next
    bar's open: the adapter records (decision, order, fill) times and prices and checks the fill was
    not at the decision bar's close;
  * a non-finite price or equity, an incompatible proposal (short in a long/flat scenario) and a
    foreign model are refusals, recorded with the bar they happened on.

Offline simulation software: it opens no account, sends nothing to a venue and trains nothing.

    from e3_weekly_runtime import run_weekly
    record = run_weekly(env, controller, bars, bar_step=timedelta(hours=1))
"""

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("e3_weekly_controller")


def _utc(ts) -> datetime:
    value = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def run_weekly(env, controller, bars, *, bar_step: timedelta, features_of=None, external=None,
               max_steps: int | None = None) -> dict:
    """Drive one continuous episode on the real environment.

    Every index comes from the ENVIRONMENT's own bar counter, never from this loop: the episode starts
    after the observation window, so the frame row a decision belongs to is `info["bar_index"] - 1` and
    the bar that fills it is the next one. Orders in flight are part of the state: between the decision
    and the fill the cash they commit is reserved, so the controller cannot open the same position
    twice while its first order is still on its way to the broker.
    """
    obs, info = env.reset()
    records, fills, refusals = [], [], []
    equity_path = [float(info.get("equity"))]
    in_flight = []                                     # [{"bar": env bar, "notional": float}]
    steps = 0
    while max_steps is None or steps < max_steps:
        env_bar = int(info.get("bar_index") or 0) - 1
        if env_bar + 1 >= len(bars):
            break
        bar_time = _utc(bars.index[env_bar]) if hasattr(bars, "index") else _utc(bars["DATE_TIME"].iloc[env_bar])
        price = float(info.get("price", bars["CLOSE"].iloc[env_bar]))
        equity = float(info.get("equity"))
        units = float(info.get("position_units") or 0.0)
        # an order stays outstanding until its fill is observed, or until the declared latency has
        # passed without one; while it is outstanding its cash is reserved and its side is part of the state
        in_flight = [o for o in in_flight if env_bar - o["bar"] <= max(1, controller.latency_bars)]
        reserved = float(sum(o["notional"] for o in in_flight))
        chosen = controller.available_release(bar_time)
        try:
            if external is not None and external(steps, info) is not None:
                action, model_id = external(steps, info)
                decided = controller.accept_external(bar_time, action, model_id)
            else:
                decided = controller.infer(bar_time, features_of(env_bar, info) if features_of else None)
        except C.IncompatibleProposal as exc:
            refusals.append({"bar": env_bar, "time": bar_time.isoformat(), "why": str(exc)})
            decided = {"model": None, "proposal": None, "source": "REFUSED"}
        effective_units = units if units else (1.0 if in_flight and any(o["action"] == C.LONG for o in in_flight) else 0.0)
        try:
            record = controller.decide(bar_time, bar_step, decided["proposal"], equity, price, effective_units,
                                       reserved_cash=reserved, model=decided["model"])
        except C.IncompatibleProposal as exc:
            refusals.append({"bar": env_bar, "time": bar_time.isoformat(), "why": str(exc)})
            record = {"action": C.HOLD, "reason": f"REFUSED: {exc}", "decision_time": bar_time.isoformat(),
                      "model": decided["model"], "proposal": decided["proposal"], "equity": equity, "price": price,
                      "reserved_cash": reserved, "week_start": C.week_start(bar_time).isoformat(), "new_week": False}
        record["source"] = decided["source"]
        record["model_selected"] = chosen.name if chosen else None
        record["env_bar"] = env_bar
        if record["action"] == C.LONG:
            in_flight.append({"bar": env_bar, "notional": record.get("notional", 0.0), "action": C.LONG})
        obs, reward, terminated, truncated, info = env.step(int(record["action"]))
        after_units = float(info.get("position_units") or 0.0)
        record.update(equity_after=float(info.get("equity")), units_before=units, units_after=after_units,
                      commission_paid=float(info.get("commission_paid") or 0.0),
                      open_orders_after=int(info.get("open_order_count") or 0), terminated=bool(terminated))
        if after_units != units:
            decision_bar = in_flight[0]["bar"] if in_flight else env_bar
            fill_bar = decision_bar + 1                       # the bar whose OPEN pays for a decision taken at `decision_bar`
            fills.append({"decision_bar": decision_bar, "fill_bar": fill_bar, "observed_at_env_bar": env_bar,
                          "decision_time": (_utc(bars.index[decision_bar]) if hasattr(bars, "index") else bar_time).isoformat(),
                          "fill_time": (bar_time + bar_step).isoformat(),
                          "fill_price": float(bars["OPEN"].iloc[fill_bar]), "close_at_decision": float(bars["CLOSE"].iloc[decision_bar]),
                          "units_before": units, "units_after": after_units,
                          "entry_price": float(getattr(getattr(env, "bridge", None), "entry_price", float("nan")))})
            in_flight = []
        equity_path.append(float(info.get("equity")))
        records.append(record)
        steps += 1
        if terminated or truncated:
            break
    env.close()
    return {"records": records, "fills": fills, "refusals": refusals, "equity_path": equity_path,
            "final": {"equity": equity_path[-1], "units": float(info.get("position_units") or 0.0),
                      "commission_paid": float(info.get("commission_paid") or 0.0),
                      "open_orders": int(info.get("open_order_count") or 0)},
            "controller": {"misses": controller.state.misses, "refused_proposals": controller.state.refused_proposals,
                           "fallback": controller.fallback, "latency_bars": controller.latency_bars}}


class ConstantModel:
    """A model that always proposes one action. Two of them with opposite actions are what makes a
    fallback test real: if the controller silently used the wrong one, the episode would differ."""

    def __init__(self, action: int, name: str = "constant"):
        self.action, self.name = int(action), name

    def decide(self, features):
        return self.action


class ThresholdModel:
    """A model that proposes LONG when its feature is above a threshold. Used to show the controller
    runs the model's OWN inference rather than a proposal handed to it."""

    def __init__(self, threshold: float, name: str = "threshold"):
        self.threshold, self.name = float(threshold), name

    def decide(self, features):
        value = float(features if not hasattr(features, "__len__") else features[-1])
        if not math.isfinite(value):
            raise C.IncompatibleProposal("the model received a non-finite feature")
        return C.LONG if value > self.threshold else C.CLOSE
