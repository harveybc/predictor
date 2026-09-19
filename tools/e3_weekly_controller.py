#!/usr/bin/env python3
"""RP31: the weekly long/flat controller of the 13D scenario — a production component (not a policy
written inside a test) that, on every bar of a continuous episode, decides which model is available,
whether the declared fallback applies, and which environment action is emitted.

Clocks per week (UTC, [Monday 00:00, next Monday 00:00)), fixed BEFORE scoring for each retraining:
  cutoff <= fit_start < fit_end <= release <= first_decision
A model proposal for week k is admissible at bar t only when release_k <= information_time(t) is false...
more precisely: the model released at instant `release` may act on decisions whose bar timestamp is >= release;
before that, the LAST VALID released model acts, and if none exists, the policy is FLAT (fallback declared
as `fallback="last_valid_or_flat"`). A late model (release after its planned first decision) is recorded as a
miss; the release is never moved backwards.

Scenario adaptation (13D cash-spot): the environment can go short (action 2) but this scenario is
long/flat: proposals are LONG or FLAT; SHORT is an incompatible proposal and is refused (recorded, converted
to the fallback of holding the current state). Closing a long position uses the explicit close-to-flat
action of the environment (3), never a reversal (2). Sizing: notional <= equity (no leverage), cash
sufficiency checked against the current equity before a LONG is emitted; equity varies, so the size is
re-derived per decision from the current equity.

Information: a decision at bar t may use only bars with timestamp <= t (the environment publishes bar t's
close before asking for the action) and the model's own availability; the fill happens at the next
eligible bar (the environment's rule: next open with a positive latency of one bar), which the controller
records per decision as (decision_time, order_time, expected_fill_time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from datetime import datetime, timedelta, timezone

HOLD, LONG, SHORT, CLOSE = 0, 1, 2, 3
#: the declared fallbacks; the parameter DECIDES behaviour, it is not decoration
FALLBACKS = {"last_valid_or_flat", "flat_only"}


class IncompatibleProposal(ValueError):
    pass


@dataclass
class ModelRelease:
    week_start: datetime           # Monday 00:00 UTC of the week the model is meant for
    cutoff: datetime
    fit_start: datetime
    fit_end: datetime
    release: datetime
    first_decision: datetime
    name: str
    #: RP39: the model itself. The controller RUNS the selected model's inference; an action that
    #: arrives from elsewhere is accepted only when it names the model that produced it AND that model
    #: is the one selected. A proposal with no identity is never labelled with the last released model.
    model: object | None = None

    def valid_clocks(self) -> bool:
        """The fitting clocks must be ordered; release <= first_decision is the PLAN, and a violation is a
        late release (recorded as a miss, see `late()`), not an invalid model."""
        return self.cutoff <= self.fit_start < self.fit_end <= self.release

    def late(self) -> bool:
        return self.release > self.first_decision


@dataclass
class ControllerState:
    active_model: str | None = None
    position: int = 0                      # 0 flat, 1 long (the scenario never holds -1)
    current_week: datetime | None = None
    decisions: list = field(default_factory=list)
    misses: list = field(default_factory=list)
    refused_proposals: list = field(default_factory=list)


def week_start(ts: datetime) -> datetime:
    ts = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    monday = (ts - timedelta(days=ts.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return monday


class WeeklyLongFlatController:
    def __init__(self, releases: list, *, size_fraction: float = 0.5, latency_bars: int = 1, fallback: str = "last_valid_or_flat"):
        if not isinstance(latency_bars, int) or isinstance(latency_bars, bool) or latency_bars < 1:
            raise ValueError("latency_bars must be a positive whole number of bars: a decision cannot be filled "
                             "on the bar that produced it")
        if fallback not in FALLBACKS:
            raise ValueError(f"unknown fallback {fallback!r}; the declared ones are {sorted(FALLBACKS)}")
        if not (0 < float(size_fraction) <= 1):
            raise ValueError("size_fraction is a fraction of equity in (0, 1]")
        for r in releases:
            if not r.valid_clocks():
                raise ValueError(f"release {r.name}: clocks violate cutoff <= fit_start < fit_end <= release")
        self.releases = sorted(releases, key=lambda r: r.release)
        self.size_fraction = float(size_fraction)
        self.latency_bars = int(latency_bars)
        self.fallback = fallback
        self.state = ControllerState()
        for r in self.releases:
            if r.late():
                self.state.misses.append({"model": r.name, "release": r.release.isoformat(), "planned_first_decision": r.first_decision.isoformat(),
                                          "disposition": "LATE: acts only from its release; the fallback covers the gap; the release is not moved"})

    def available_release(self, bar_time: datetime):
        """The release whose model may act at this bar: the last one released at or before it. Under the
        `flat_only` fallback only the release of THIS week may act, so a stale model never carries over."""
        avail = [r for r in self.releases if r.release <= bar_time]
        if not avail:
            return None
        chosen = avail[-1]
        if self.fallback == "flat_only" and week_start(chosen.week_start) != week_start(bar_time):
            return None
        return chosen

    def available_model(self, bar_time: datetime) -> str | None:
        """The name of the model that may act at this bar (None when the fallback governs)."""
        chosen = self.available_release(bar_time)
        return chosen.name if chosen else None

    def infer(self, bar_time: datetime, features) -> dict:
        """Run the SELECTED model's own inference. This is where the action comes from: nothing else
        may be labelled with this model's name."""
        chosen = self.available_release(bar_time)
        if chosen is None:
            return {"model": None, "proposal": None, "source": "NO_MODEL_AVAILABLE"}
        if chosen.model is None:
            raise IncompatibleProposal(f"release {chosen.name} carries no model to run")
        action = chosen.model.decide(features)
        if action not in (HOLD, LONG, SHORT, CLOSE):
            raise IncompatibleProposal(f"model {chosen.name} proposed {action!r}, which is not an action")
        return {"model": chosen.name, "proposal": int(action), "source": "MODEL_INFERENCE"}

    def accept_external(self, bar_time: datetime, proposal: int, model_id: str) -> dict:
        """An action produced elsewhere is admitted only when it names the model that produced it and
        that model is the selected one."""
        chosen = self.available_release(bar_time)
        if chosen is None:
            raise IncompatibleProposal("no model is available: an external action cannot be attributed to one")
        if model_id != chosen.name:
            raise IncompatibleProposal(f"the action names model {model_id!r} but the selected model is {chosen.name!r}")
        return {"model": chosen.name, "proposal": int(proposal), "source": "EXTERNAL_IDENTITY_VERIFIED"}

    def decide(self, bar_time: datetime, bar_step: timedelta, proposal: int | None, equity: float, price: float,
               position_units: float, *, reserved_cash: float = 0.0, model: str | None = None) -> dict:
        """One decision. `proposal` is the active model's proposed action (LONG/FLAT) or None when no model is
        available; returns the environment action and the record of the decision."""
        model = model if model is not None else self.available_model(bar_time)
        for name, value in (("price", price), ("equity", equity), ("position_units", position_units),
                            ("reserved_cash", reserved_cash)):
            if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise IncompatibleProposal(f"{name} is not a finite number ({value!r}): no action is emitted")
        wk = week_start(bar_time)
        new_week = self.state.current_week is not None and wk != self.state.current_week
        self.state.current_week = wk
        record = {"decision_time": bar_time.isoformat(), "order_time": bar_time.isoformat(),
                  "expected_fill_time": (bar_time + bar_step * self.latency_bars).isoformat(), "week_start": wk.isoformat(),
                  "new_week": bool(new_week), "model": model, "proposal": proposal, "equity": float(equity),
                  "price": float(price), "reserved_cash": float(reserved_cash), "fallback": self.fallback}
        if model is None:
            action = HOLD if position_units == 0 else CLOSE                    # no model ever released: FLAT is the declared fallback
            record.update(action=action, reason="FALLBACK_NO_MODEL_FLAT")
        else:
            self.state.active_model = model
            if proposal == SHORT:
                self.state.refused_proposals.append({"time": bar_time.isoformat(), "model": model, "proposal": "SHORT"})
                action = HOLD
                record.update(action=action, reason="INCOMPATIBLE_PROPOSAL_SHORT_REFUSED_HOLD")
            elif proposal == LONG:
                if position_units > 0:
                    action, reason = HOLD, "ALREADY_LONG"
                else:
                    # cash that pending orders already reserve is not available to a new one
                    free = equity - float(reserved_cash)
                    notional = self.size_fraction * equity
                    if notional <= 0 or notional > free + 1e-12 or price <= 0:
                        action, reason = HOLD, "INSUFFICIENT_CASH_OR_PRICE"
                    else:
                        action, reason = LONG, "OPEN_LONG"
                        record["size_units"] = notional / price
                        record["notional"] = notional
                record.update(action=action, reason=reason)
            elif proposal in (HOLD, None):
                if position_units > 0 and proposal is None:
                    action, reason = CLOSE, "FLAT_PROPOSAL_CLOSE"
                else:
                    action, reason = HOLD, "HOLD"
                record.update(action=action, reason=reason)
            elif proposal == CLOSE:
                action, reason = (CLOSE, "CLOSE_TO_FLAT") if position_units > 0 else (HOLD, "ALREADY_FLAT")
                record.update(action=action, reason=reason)
            else:
                raise IncompatibleProposal(f"unknown proposal {proposal!r}")
        if record["action"] == SHORT:
            raise IncompatibleProposal("the controller never emits SHORT in the long/flat scenario")
        self.state.decisions.append(record)
        return record

    def size_for(self, equity: float, price: float) -> float:
        return self.size_fraction * equity / price if price > 0 else 0.0
