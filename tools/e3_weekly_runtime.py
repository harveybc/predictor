#!/usr/bin/env python3
"""RP45: the weekly controller driving the REAL environment and broker, with the quantity actually
transmitted, the clock read from the environment and every fill taken from execution events.

What the dictum measured on the previous version, and what this one does instead:

  quantity   the controller decided 0.001 units and the broker executed 0.005, because the adapter
             sent only `env.step(action)` and the environment sized from its own `position_size`.
             Here the run is refused before the episode unless the environment can execute the
             contract: it must be in continuous/fractional mode, and the decided units must be
             expressible as a fraction of the environment's unit. The fraction travels in the action
             itself, which is what this environment's interface accepts, and every fill is checked
             against what was decided.
  latency    a contracted latency of 3 bars reported a fill one bar later. This environment fills at
             the NEXT bar, so a latency of k is honoured by holding the order and submitting it at
             the (k-1)th bar after the decision; a latency the runtime cannot honour is refused
             before the episode rather than relabelled.
  fills      taken from the strategy's completed-order evidence (order reference, executed price,
             executed size, commission, bar), never reconstructed from OPEN[decision + 1].
  pending    reconciled against the broker's own open-order inventory and its terminal statuses
             (completed, cancelled, rejected, margin, expired). Nothing expires by elapsed time.
  clock      each bar's timestamp comes from the environment's own data feed and must be strictly
             increasing and equal to the row the bar index names. An absent or constant counter is
             a refusal, not a clock.

Partial fills are outside this simulator's definition (its broker raises on a residual size), so the
scope is DECLARED and a partial fill is refused rather than pretended.

Offline simulation software: no venue, no account, no policy training.
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

#: what this simulator's broker does not define, declared instead of pretended
UNSUPPORTED = {"partial_fill": "this broker emits only terminal aggregate fills and raises on a residual size",
               "order_cancellation_by_caller": "the runtime never cancels; a pending order ends by the broker's own verdict"}


class ExecutionContractRefused(SystemExit):
    """The environment cannot execute what the controller would decide; nothing runs."""


def _utc(ts) -> datetime:
    value = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


# --- the execution contract, checked BEFORE the episode ---------------------------------------------

def check_execution_contract(env, config: dict, controller, *, bar_step: timedelta) -> dict:
    """Refuse, before any bar, a contract this environment cannot execute."""
    problems = []
    mode = str(config.get("action_space_mode") or "")
    fractional = bool(config.get("fractional_position_sizing"))
    unit = float(config.get("position_size") or 0.0)
    if mode != "continuous" or not fractional:
        problems.append("the environment is not in continuous/fractional mode, so a decided quantity cannot be "
                        f"transmitted (action_space_mode={mode!r}, fractional_position_sizing={fractional})")
    if unit <= 0:
        problems.append(f"the environment's unit (position_size) is {unit!r}")
    if controller.latency_bars < 1:
        problems.append("a latency below one bar cannot be executed by an environment that fills at the next bar")
    contract = {"environment_unit": unit, "fill_rule": "the broker fills at the next bar's open",
                "latency_bars_contracted": controller.latency_bars,
                "latency_implementation": ("the order is held and submitted at the (k-1)th bar after the decision, "
                                           "so the fill lands k bars after it"),
                "quantity_channel": "the fraction of the environment's unit travels in the continuous action",
                "unsupported": dict(UNSUPPORTED), "problems": problems}
    if problems:
        raise ExecutionContractRefused("REFUSED: " + "; ".join(problems))
    return contract


def _bar_time(env):
    """The environment's own clock for the current bar, read from its data feed."""
    strategy = getattr(env, "_strategy_instance", None)
    data = getattr(strategy, "data", None)
    if data is None:
        return None
    try:
        return _utc(data.datetime.datetime(0))
    except Exception:
        return None


def _completed_fills(env) -> list:
    strategy = getattr(env, "_strategy_instance", None)
    return list(getattr(strategy, "_completed_fills", []) or [])


def _open_orders(info: dict) -> tuple:
    inventory = info.get("open_order_inventory")
    return tuple(inventory) if inventory else ()


def run_weekly(env, controller, bars, *, config: dict, bar_step: timedelta, features_of=None, external=None,
               max_steps: int | None = None) -> dict:
    """One continuous episode on the real environment, with the quantity transmitted and every
    execution fact observed."""
    contract = check_execution_contract(env, config, controller, bar_step=bar_step)
    unit = contract["environment_unit"]
    obs, info = env.reset()
    records, fills, refusals, clock = [], [], [], []
    equity_path = [float(info.get("equity"))]
    queue = []                      # orders decided but not yet submitted (the contracted latency)
    outstanding = {}                # what the runtime knows about the order it last submitted
    seen_fill_refs = {int(f["order_ref"]) for f in _completed_fills(env)}
    steps = 0
    previous_time = None
    while max_steps is None or steps < max_steps:
        env_bar = int(info.get("bar_index") or 0) - 1
        bar_time = _bar_time(env)
        if bar_time is None:
            raise ExecutionContractRefused("REFUSED: the environment publishes no bar timestamp; a bar counter is not a clock")
        if previous_time is not None and not bar_time > previous_time:
            raise ExecutionContractRefused(f"REFUSED: the environment's clock did not advance ({previous_time} -> {bar_time})")
        if env_bar < 0 or env_bar + 1 >= len(bars):
            break
        frame_time = (_utc(bars.index[env_bar]) if hasattr(bars, "index") else _utc(bars["DATE_TIME"].iloc[env_bar]))
        if frame_time != bar_time:
            raise ExecutionContractRefused(f"REFUSED: bar {env_bar} of the environment says {bar_time} and the data says {frame_time}")
        clock.append({"env_bar": env_bar, "time": bar_time.isoformat()})
        previous_time = bar_time
        price = float(info.get("price"))
        equity = float(info.get("equity"))
        units = float(info.get("position_units") or 0.0)
        broker_open = _open_orders(info)
        reserved = float(sum(o["notional"] for o in queue))
        chosen = controller.available_release(bar_time)
        try:
            if external is not None and external(steps, info) is not None:
                action, model_id = external(steps, info)
                decided = controller.accept_external(bar_time, action, model_id)
            else:
                decided = controller.infer(bar_time, features_of(env_bar, info) if features_of else None)
        except C.IncompatibleProposal as exc:
            refusals.append({"env_bar": env_bar, "time": bar_time.isoformat(), "why": str(exc)})
            decided = {"model": None, "proposal": None, "source": "REFUSED"}
        # the state the controller decides on includes what it already sent and what the broker holds
        pending_long = any(o["action"] == C.LONG for o in queue) or bool(broker_open) or bool(outstanding.get("awaiting_fill"))
        effective_units = units if units else (1.0 if pending_long else 0.0)
        try:
            record = controller.decide(bar_time, bar_step, decided["proposal"], equity, price, effective_units,
                                       reserved_cash=reserved, model=decided["model"])
        except C.IncompatibleProposal as exc:
            refusals.append({"env_bar": env_bar, "time": bar_time.isoformat(), "why": str(exc)})
            record = {"action": C.HOLD, "reason": f"REFUSED: {exc}", "decision_time": bar_time.isoformat(),
                      "model": decided["model"], "proposal": decided["proposal"], "equity": equity, "price": price,
                      "reserved_cash": reserved, "week_start": C.week_start(bar_time).isoformat(), "new_week": False}
        record.update(source=decided["source"], model_selected=chosen.name if chosen else None, env_bar=env_bar,
                      bar_time=bar_time.isoformat(), broker_open_orders=list(broker_open))
        # --- the contracted latency: hold the decision until its submission bar ---------------------
        if record["action"] in (C.LONG, C.CLOSE):
            queue.append({"decided_at_bar": env_bar, "decided_at": bar_time, "action": record["action"],
                          "units": float(record.get("size_units") or 0.0),
                          "notional": float(record.get("notional") or 0.0),
                          "submit_at_bar": env_bar + max(0, controller.latency_bars - 1)})
        submit = [o for o in queue if o["submit_at_bar"] <= env_bar]
        queue = [o for o in queue if o["submit_at_bar"] > env_bar]
        env_action = 0.0
        submitted = None
        if submit:
            submitted = submit[0]
            if submitted["action"] == C.LONG:
                fraction = submitted["units"] / unit if unit > 0 else 0.0
                if not math.isfinite(fraction) or fraction <= 0 or fraction > 1 + 1e-12:
                    raise ExecutionContractRefused(
                        f"REFUSED: the decided quantity {submitted['units']} is not expressible as a fraction of the "
                        f"environment's unit {unit} (fraction {fraction})")
                env_action = float(fraction)          # continuous + fractional: direction and magnitude in one value
            else:
                env_action = 0.0                      # flat target: the fractional path closes the position
            outstanding.update(awaiting_fill=True, **{k: v for k, v in submitted.items() if k != "action"})
            outstanding["action"] = submitted["action"]
        record.update(submitted_now=bool(submitted), env_action=env_action,
                      queued_orders=[{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in o.items()}
                                     for o in queue])
        obs, reward, terminated, truncated, info = env.step(env_action)
        after_units = float(info.get("position_units") or 0.0)
        # --- fills, from the broker's own completed-order evidence -----------------------------------
        for fill in _completed_fills(env):
            ref = int(fill["order_ref"])
            if ref in seen_fill_refs:
                continue
            seen_fill_refs.add(ref)
            decided_for = outstanding if outstanding.get("awaiting_fill") else {}
            decided_units = decided_for.get("units")
            executed = abs(float(fill["size"]))
            fills.append({"order_ref": ref, "executed_price": float(fill["price"]), "executed_size": executed,
                          "signed_size": float(fill["size"]), "commission": float(fill["commission"]),
                          "fill_bar": int(fill["bar_index"]) - 1,
                          "decided_at_bar": decided_for.get("decided_at_bar"), "decided_units": decided_units,
                          "decision_time": (decided_for.get("decided_at").isoformat()
                                            if hasattr(decided_for.get("decided_at"), "isoformat") else None),
                          "quantity_matches_decision": (decided_units is not None
                                                        and abs(executed - decided_units) <= 1e-9 + 1e-6 * max(1.0, decided_units)),
                          "source": "BROKER_EXECUTION_EVENT"})
            outstanding["awaiting_fill"] = False
        record.update(equity_after=float(info.get("equity")), units_before=units, units_after=after_units,
                      commission_paid=float(info.get("commission_paid") or 0.0),
                      open_orders_after=int(info.get("open_order_count") or 0),
                      broker_terminal_status=dict(getattr(getattr(env, "bridge", None), "order_terminal_status", {}) or {}),
                      terminated=bool(terminated))
        equity_path.append(float(info.get("equity")))
        records.append(record)
        steps += 1
        if terminated or truncated:
            break
    env.close()
    return {"contract": contract, "records": records, "fills": fills, "refusals": refusals, "clock": clock,
            "equity_path": equity_path, "queue_at_end": [{k: (v.isoformat() if hasattr(v, "isoformat") else v)
                                                          for k, v in o.items()} for o in queue],
            "final": {"equity": equity_path[-1], "units": float(info.get("position_units") or 0.0),
                      "commission_paid": float(info.get("commission_paid") or 0.0),
                      "open_orders": int(info.get("open_order_count") or 0),
                      "terminal_status": dict(getattr(getattr(env, "bridge", None), "order_terminal_status", {}) or {})},
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
    """A model that proposes LONG when its feature is above a threshold, to show the controller runs
    the model's OWN inference rather than a proposal handed to it."""

    def __init__(self, threshold: float, name: str = "threshold"):
        self.threshold, self.name = float(threshold), name

    def decide(self, features):
        value = float(features if not hasattr(features, "__len__") else features[-1])
        if not math.isfinite(value):
            raise C.IncompatibleProposal("the model received a non-finite feature")
        return C.LONG if value > self.threshold else C.CLOSE
