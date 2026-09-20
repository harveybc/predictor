#!/usr/bin/env python3
"""RP45: the execution plugin through which the weekly controller's DECIDED QUANTITY reaches the
broker, and through which every execution fact comes back.

The environment already has the extension point for this: a strategy plugin whose `apply_action`
takes over order placement and whose `notify_order` sees every broker notification. The previous
adapter did not use it, sent only a direction, and the broker sized the order from its own
`position_size` — the dictum measured 0.001 decided against 0.005 executed.

This plugin:
  * places exactly the size the controller decided (`strategy.buy(size=units)`), or closes the
    position explicitly; it never sizes by itself and refuses an order it was given no size for;
  * records, per bar, the DATA FEED's own timestamp — the environment's clock, not a counter;
  * records each submitted order's broker reference, and on every notification the status and, for a
    completed one, the executed price, size, commission and the bar it happened on;
  * refuses a residual (partial) fill instead of silently aggregating it, because this simulator's
    broker does not define one.

It holds no policy: what to do is decided by the controller, this only executes and observes.
"""

from __future__ import annotations

from datetime import timezone


class PartialFillUnsupported(RuntimeError):
    pass


class WeeklyExecutionPlugin:
    """One episode's execution channel. The runtime sets `next_order` before each step."""

    def __init__(self):
        #: {"action": 1|3, "units": float, "decided_at_bar": int} or None
        self.next_order = None
        self.submissions = []            # every order this plugin placed, with its broker reference
        self.events = []                 # every broker notification, by order reference
        self.fills = []                  # completed orders, with their executed facts
        self.clock = []                  # (bar index, the data feed's own timestamp)
        self.refusals = []

    # --- the environment's hooks --------------------------------------------------------------
    def on_reset(self, strategy, config):
        self.submissions, self.events, self.fills, self.clock, self.refusals = [], [], [], [], []
        self.next_order = None

    def observe_bar(self, strategy) -> dict:
        """The bar the STRATEGY is on, from backtrader's own processed-bar count and the feed's
        timestamp. The bridge's `bar_index` is published after the action is applied, so it lags by
        one here and is recorded beside the authoritative count rather than used as the clock."""
        bar = int(len(strategy)) - 1
        stamp = strategy.data.datetime.datetime(0)
        stamp = stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
        record = {"bar_index": bar, "bridge_bar_index": int(getattr(strategy.bridge, "bar_index", -1)),
                  "time": stamp.isoformat()}
        if not self.clock or self.clock[-1]["time"] != record["time"]:
            self.clock.append(record)
        return record

    def apply_action(self, strategy, action, config):
        """Place the order the runtime prepared. A direction with no size is a refusal, not a guess."""
        self.observe_bar(strategy)
        order = self.next_order
        self.next_order = None
        if int(action) == 0 or order is None:
            if int(action) in (1, 2, 3) and order is None:
                self.refusals.append({"bar_index": int(len(strategy)) - 1,
                                      "why": f"action {int(action)} arrived with no decided quantity"})
            return
        if int(action) == 2:
            self.refusals.append({"bar_index": int(len(strategy)) - 1,
                                  "why": "a short is not part of this long/flat scenario"})
            return
        if int(action) == 3 or order.get("action") == 3:
            if strategy.position.size != 0:
                placed = strategy.close()
                self._record(strategy, placed, "close", order)
            return
        units = float(order.get("units") or 0.0)
        if units <= 0:
            self.refusals.append({"bar_index": int(len(strategy)) - 1,
                                  "why": f"a long with size {units!r} is not executable"})
            return
        placed = strategy.buy(size=units)
        self._record(strategy, placed, "entry", order)

    def notify_order(self, strategy, order, config):
        ref = int(getattr(order, "ref", -1))
        status = order.Status[order.status]
        executed = getattr(order, "executed", None)
        seen = self.observe_bar(strategy)
        event = {"order_ref": ref, "status": status, "bar_index": seen["bar_index"], "time": seen["time"]}
        if status == "Completed" and executed is not None:
            remaining = float(getattr(executed, "remsize", 0.0) or 0.0)
            if abs(remaining) > 1e-12:
                raise PartialFillUnsupported(f"order {ref} completed with residual size {remaining}")
            # the EXECUTION instant is the broker's own (`executed.dt`), not the moment its
            # notification was observed: backtrader delivers the notification on the following bar
            executed_time = None
            try:
                import backtrader as bt
                raw = getattr(executed, "dt", None)
                if raw is not None:
                    stamp = bt.num2date(float(raw))
                    executed_time = (stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)).isoformat()
            except Exception:
                executed_time = None
            event.update(executed_price=float(getattr(executed, "price", 0.0) or 0.0),
                         executed_size=float(getattr(executed, "size", 0.0) or 0.0),
                         commission=float(getattr(executed, "comm", 0.0) or 0.0),
                         executed_time=executed_time, notified_at=seen["time"],
                         notified_at_bar=seen["bar_index"])
            submission = next((s for s in self.submissions if s["order_ref"] == ref), {})
            self.fills.append({**event, "decided_units": submission.get("units"),
                               "decided_at_bar": submission.get("decided_at_bar"),
                               "role": submission.get("role"),
                               "source": "BROKER_EXECUTION_EVENT"})
        self.events.append(event)

    # --- internals -------------------------------------------------------------------------------
    def _record(self, strategy, placed, role, order):
        refs = []
        for item in (placed if isinstance(placed, (list, tuple)) else [placed]):
            ref = getattr(item, "ref", None)
            if ref is None:
                continue
            refs.append(int(ref))
            seen = self.observe_bar(strategy)
            self.submissions.append({"order_ref": int(ref), "role": role,
                                     "units": float(order.get("units") or 0.0) if role == "entry" else None,
                                     "decided_at_bar": order.get("decided_at_bar"),
                                     "submitted_at_bar": seen["bar_index"], "submitted_at": seen["time"]})
            try:
                strategy.bridge.register_order_role(int(ref), "entry" if role == "entry" else "close")
            except Exception:
                pass
        return refs
