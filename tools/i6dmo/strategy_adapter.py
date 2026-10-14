"""I6-DMO strategy adapter and business executor (strategy lane).

Prepared, not run: this module performs no strategy fit, reads no TEST row and
places no order. It connects the two direct multi-output families to the pinned
heuristic-strategy business pair policy and simulates execution with explicit
fills.

* Plugin: the executor never resolves ``heuristic_strategy.plugins`` through
  entry points or ``sys.path`` (that name resolves to different code depending
  on the interpreter; see :func:`map_strategy_resolution`). It loads ONE pinned
  blob, ``app/policies/business_pair_policy.py`` at :data:`STRATEGY_PIN`, from
  the heuristic-strategy git object store and verifies its sha256.
* Named outputs: a head is ``(family, horizon_hours)``; positional vectors and
  ``Prediction_h_i`` columns are refused. Per-head availability = C1 operational
  eligibility E at the origin (calendar only, never a future label) AND a finite
  prediction AND, when supplied, validation eligibility. An unavailable head has
  ``value=None`` and a reason; it is never filled with the current price.
* Decisions: entry from the long family only (minimum valid long heads);
  management consumes both families in horizon order and closes early only when
  the first sampled barrier touch of the CURRENT position is the stop loss.
* Execution (one asset, one position, initial equity 10000): a decision at
  origin ``t`` (the close of the bar labelled ``t``) becomes a market order that
  fills at the OPEN of the next available bar. The calendar force-flat order is
  submitted at the first origin ``t >= B - force_close_lead_hours`` so that its
  fill bar closes by ``B``. A missing fill is a retained execution failure; the
  end of the data never invents a fill (the legacy backtrader ``stop()`` calls
  ``close()`` without a fill).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import types
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from tools.i6n_contracts import eligibility as c1

SCHEMA = "i6dmo.strategy_adapter.v1"
UTC = timezone.utc
HOUR = timedelta(hours=1)
TEST_START = datetime(2025, 1, 1, tzinfo=UTC)
SHORT_HORIZONS_HOURS = (2, 3, 4, 5, 6, 12)
LONG_HORIZONS_HOURS = (18, 24, 36, 48, 60, 72)
FAMILIES = {"short": SHORT_HORIZONS_HOURS, "long": LONG_HORIZONS_HOURS}

STRATEGY_PIN = {
    "repository": "heuristic-strategy",
    "branch": "satoshi/i6dmo-strategy-20261014",
    "commit": "d40fd3b054e4f952054a86ea203e8f74e6b295e1",
    "path": "app/policies/business_pair_policy.py",
    "blob": "79b6df7a7bffea7dd7f67c2ab33eeb64941bd44c",
    "sha256": "6b6c5d96cafed149d3d832462b1eea27c2cd1cb8ca88a3ac495e7b8f1cbccf80",
}
LEGACY_PLUGIN_PATH = "app/plugins/plugin_long_short_predictions.py"
LEGACY_POLICY_PATH = "app/policies/prediction_entry_exit.py"
ENTRY_POINT = "heuristic_strategy.plugins:ls_pred_strategy"

# Entry parameters: the legacy plugin defaults (plugin_params) plus the
# market-week contract's proposed minimum of two distinct long heads.
DEFAULT_ENTRY_PARAMETERS = {
    "pip_cost": 0.00001, "rel_volume": 0.02, "min_order_volume": 10_000.0,
    "max_order_volume": 1_000_000.0, "leverage": 100.0, "profit_threshold": 5.0,
    "min_drawdown_pips": 10.0, "tp_multiplier": 0.9, "sl_multiplier": 2.0,
    "lower_rr_threshold": 0.5, "upper_rr_threshold": 2.0, "max_trades_per_5days": 3,
    "min_valid_long_heads": 2,
}

EXECUTION_RULES = {
    "fill": "MARKET_ORDER_AT_OPEN_OF_NEXT_AVAILABLE_BAR",
    "bar_label": "BAR_CLOSE_TIME; bar covers (close_time - 1h, close_time]; open instant = close_time - 1h",
    "entry_fill_deadline": "fill bar close_time <= min(entry_cutoff K, B - force_close_lead_hours)",
    "close_fill_deadline": "fill bar close_time <= B of the position's week",
    "force_close_trigger": "first origin t >= B - force_close_lead_hours",
    "missing_fill": "order keeps working; a later fill is flagged; end of data leaves the position open (no invented fill)",
    "intrabar_both_barriers": "STOP_LOSS_FIRST",
    "protective_gap": "fill at bar open when the open is already beyond the barrier",
    "costs_per_fill": "half spread + slippage adverse to the order side; commission half of round trip per fill",
    "swap": "swap_per_lot_per_day * lots * elapsed_hours / 24, debited to equity",
    "one_position_per_asset": "no entry decision while a position or a pending order exists",
    "trade_frequency": "max_trades_per_5days entries submitted within the previous 120 elapsed hours",
}


class StrategyAdapterError(ValueError):
    """Typed refusal with a stable ``code``."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def parse_time(text: str) -> datetime:
    return c1.parse_instant("time", text)


def _sanitize(text: str) -> str:
    for real, alias in ((str(Path.home()), "<home>"), (sys.prefix, "<python-prefix>")):
        if real and real != "/":
            text = text.replace(real, alias)
    return text


# --------------------------------------------------------------------------
# Pinned plugin resolution


def _git(root: Path, *args: str, binary: bool = False):
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout if binary else proc.stdout.decode("utf-8").strip()


def strategy_repo_candidates() -> List[Path]:
    """Ordered candidate roots: $I6DMO_STRATEGY_REPO, then siblings of this checkout
    and of its main repository (for a linked worktree)."""
    out: List[Path] = []
    env = os.environ.get("I6DMO_STRATEGY_REPO")
    if env:
        out.append(Path(env))
    here = Path(__file__).resolve().parents[2]
    out.append(here.parent / "heuristic-strategy")
    common = _git(here, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if common:
        out.append(Path(common).parent.parent / "heuristic-strategy")
    seen, unique = set(), []
    for p in out:
        if str(p) not in seen:
            seen.add(str(p))
            unique.append(p)
    return unique


def _read_pinned(path: str, commit: str, roots: Optional[Sequence[Path]] = None):
    for root in roots or strategy_repo_candidates():
        if not (root / ".git").exists():
            continue
        data = _git(root, "show", f"{commit}:{path}", binary=True)
        if data is not None:
            blob = _git(root, "rev-parse", f"{commit}:{path}")
            return data, blob
    raise StrategyAdapterError("STRATEGY_PIN_UNRESOLVED",
                               f"{path} at {commit} not found in any heuristic-strategy checkout "
                               "(set I6DMO_STRATEGY_REPO and fetch the pinned branch)")


_POLICY_CACHE: Dict[str, types.ModuleType] = {}


def load_pinned_policy(pin: Mapping[str, str] = STRATEGY_PIN,
                       roots: Optional[Sequence[Path]] = None) -> types.ModuleType:
    """Load the pinned policy blob as an isolated module after verifying sha256."""
    data, blob = _read_pinned(pin["path"], pin["commit"], roots)
    digest = hashlib.sha256(data).hexdigest()
    if digest != pin["sha256"] or (pin.get("blob") and blob != pin["blob"]):
        raise StrategyAdapterError("STRATEGY_PIN_MISMATCH",
                                   f"{pin['path']}@{pin['commit'][:12]} sha256 {digest} != {pin['sha256']}")
    if digest in _POLICY_CACHE:
        return _POLICY_CACHE[digest]
    name = f"i6dmo_pinned_business_pair_policy_{digest[:16]}"
    module = types.ModuleType(name)
    module.__file__ = f"<git:{pin['repository'] if 'repository' in pin else 'heuristic-strategy'}@{pin['commit']}:{pin['path']}>"
    sys.modules[name] = module  # dataclasses resolve their module through sys.modules
    exec(compile(data, module.__file__, "exec"), module.__dict__)
    module.__i6dmo_identity__ = {"commit": pin["commit"], "path": pin["path"], "blob": blob,
                                 "sha256": digest, "rules": dict(module.RULES)}
    _POLICY_CACHE[digest] = module
    return module


def policy_identity(policy: types.ModuleType) -> Dict[str, Any]:
    return dict(policy.__i6dmo_identity__)


def _entry_point_report() -> Dict[str, Any]:
    from importlib.metadata import entry_points
    group, name = ENTRY_POINT.split(":")
    eps = [ep for ep in entry_points(group=group) if ep.name == name]
    report: Dict[str, Any] = {"name": ENTRY_POINT, "used_by_executor": False,
                              "interpreter": _sanitize(sys.executable)}
    if not eps:
        report["resolution"] = "NO_ENTRY_POINT_IN_THIS_INTERPRETER"
        return report
    ep = eps[0]
    report["value"] = ep.value
    dist = getattr(ep, "dist", None)
    report["distribution"] = None if dist is None else f"{dist.metadata['Name']} {dist.version}"
    rel = LEGACY_PLUGIN_PATH
    located = None
    if dist is not None:
        for f in dist.files or ():
            if str(f).replace("\\", "/").endswith(rel):
                located = Path(dist.locate_file(f))
                break
    if located is not None and located.exists():
        report["resolution"] = "INSTALLED_FILE"
        report["file"] = _sanitize(str(located))
        report["sha256"] = hashlib.sha256(located.read_bytes()).hexdigest()
    else:
        report["resolution"] = "EDITABLE_OR_UNLOCATED (depends on sys.path; not imported here)"
    return report


def map_strategy_resolution(roots: Optional[Sequence[Path]] = None) -> Dict[str, Any]:
    """Report what each route would load, without importing the legacy code."""
    policy = load_pinned_policy(roots=roots)
    legacy = {}
    for key, path in (("legacy_plugin_at_pin", LEGACY_PLUGIN_PATH),
                      ("legacy_policy_at_pin", LEGACY_POLICY_PATH)):
        data, blob = _read_pinned(path, STRATEGY_PIN["commit"], roots)
        legacy[key] = {"path": path, "blob": blob,
                       "sha256": hashlib.sha256(data).hexdigest()}
    legacy["legacy_plugin_at_pin"]["known_contract_defects"] = [
        "POSITIONAL_PREDICTION_COLUMNS", "MISSING_HEAD_FILLED_WITH_CURRENT_PRICE",
        "MIN_MAX_EARLY_CLOSE_ORDER_BLIND_IGNORES_TP", "STOP_CLOSE_WITHOUT_FILL",
        "NO_CALENDAR_FORCE_FLAT", "SWAP_NOT_DEBITED_TO_BROKER"]
    legacy["legacy_plugin_at_pin"]["evidence"] = (
        "predictor satoshi/i6n-plan-audit-20261011 "
        "docs/audits/evidence/I6N_PLAN_AUDIT_20261011/addendum/strategy/")
    return {"schema": SCHEMA + ".plugin_map", "executor_source": "PINNED_GIT_BLOB",
            "pinned_policy": policy_identity(policy), **legacy,
            "entry_point": _entry_point_report(),
            "decision": ("The I6-DMO executor imports only the pinned business pair policy. "
                         "The legacy plugin/policy remain unchanged for historical identity "
                         "and are not used by I6-DMO.")}


# --------------------------------------------------------------------------
# Named outputs and masks


def _check_family(family: str) -> tuple:
    if family not in FAMILIES:
        raise StrategyAdapterError("FAMILY_INVALID", f"family {family!r}")
    return FAMILIES[family]


def build_family_heads(*, origin: datetime, family: str, predicted: Mapping[int, Any],
                       week: c1.OperationalWeek, role: c1.Role,
                       validation_eligible: Optional[Iterable[int]] = None) -> List[Dict[str, Any]]:
    """One record per declared head, in horizon order, with availability and reason."""
    hs = _check_family(family)
    if not isinstance(predicted, Mapping):
        raise StrategyAdapterError("POSITIONAL_INPUT_REJECTED",
                                   "predictions must be a {horizon_hours: value} mapping")
    keys = list(predicted)
    if any(type(k) is not int for k in keys) or sorted(keys) != sorted(hs) or len(keys) != len(hs):
        raise StrategyAdapterError("HEAD_KEYS_MISMATCH",
                                   f"{family} keys {keys!r} must be exactly {list(hs)}")
    eligible = None if validation_eligible is None else {int(h) for h in validation_eligible}
    decisions = c1.evaluate_heads(origin, hs, role, week)
    records = []
    for index, (h, d) in enumerate(zip(hs, decisions)):
        raw = predicted[h]
        reasons = []
        if not d.operational_eligible:
            reasons.append("OPERATIONALLY_INELIGIBLE:" +
                           ",".join(r.value for r in d.operational_reasons))
        if eligible is not None and h not in eligible:
            reasons.append("HEAD_NOT_ELIGIBLE_IN_VALIDATION")
        if raw is None:
            reasons.append("PREDICTION_MISSING")
        elif isinstance(raw, bool) or not isinstance(raw, (int, float)) or \
                not math.isfinite(float(raw)):
            reasons.append("PREDICTION_NON_FINITE")
        available = not reasons
        records.append({
            "family": family, "index": index, "horizon_hours": h,
            "role": d.role.value, "origin": c1.iso(d.origin),
            "target_timestamp": c1.iso(d.target_timestamp),
            "eligible_until": c1.iso(d.eligible_until),
            "week_id": d.week_id, "operational_eligible": d.operational_eligible,
            "available": available, "value": float(raw) if available else None,
            "reason": None if available else ";".join(reasons)})
    return records


def policy_heads(records: Sequence[Mapping[str, Any]], policy: types.ModuleType):
    return [policy.HeadForecast(r["family"], r["horizon_hours"], r["available"],
                                r["value"], r["reason"]) for r in records]


def parse_named_columns(row: Mapping[str, Any]) -> Dict[str, Dict[int, Any]]:
    """``short_h2`` .. ``long_h72`` columns to family maps; legacy positional names refused."""
    out: Dict[str, Dict[int, Any]] = {"short": {}, "long": {}}
    for col, value in row.items():
        if col.startswith("Prediction_"):
            raise StrategyAdapterError("POSITIONAL_COLUMN_REJECTED",
                                       f"{col}: positional columns carry no horizon_hours")
        for family, hs in FAMILIES.items():
            prefix = f"{family}_h"
            if col.startswith(prefix):
                tail = col[len(prefix):]
                if not tail.isdigit() or int(tail) not in hs:
                    raise StrategyAdapterError("HEAD_KEYS_MISMATCH", f"unknown column {col}")
                out[family][int(tail)] = value
    for family, hs in FAMILIES.items():
        if sorted(out[family]) != list(hs):
            raise StrategyAdapterError("HEAD_KEYS_MISMATCH",
                                       f"{family} columns {sorted(out[family])} != {list(hs)}")
    return out


def family_predictions_from_arrays(*, row_id: Sequence[Any], horizons_hours: Sequence[Any],
                                   predicted_price: Sequence[Sequence[Any]],
                                   family: str) -> Dict[datetime, Dict[int, Optional[float]]]:
    """Model-lane arrays (row_id = origin epoch seconds) to origin -> {h: value|None}.

    Columns are mapped by ``horizons_hours``, never by position. NaN becomes an
    explicit missing head. The label mask of the model lane is NOT consumed: it
    depends on future label existence and must not drive a decision.
    """
    hs = _check_family(family)
    cols = [int(h) for h in horizons_hours]
    if sorted(cols) != list(hs) or len(cols) != len(hs):
        raise StrategyAdapterError("HEAD_KEYS_MISMATCH", f"{family} horizons {cols} != {list(hs)}")
    out: Dict[datetime, Dict[int, Optional[float]]] = {}
    for rid, row in zip(row_id, predicted_price):
        t = datetime.fromtimestamp(int(rid), tz=UTC)
        if t in out:
            raise StrategyAdapterError("DUPLICATE_ORIGIN", c1.iso(t))
        values = [float(v) for v in row]
        if len(values) != len(cols):
            raise StrategyAdapterError("HEAD_KEYS_MISMATCH", "row width differs from horizons")
        out[t] = {h: (v if math.isfinite(v) else None) for h, v in zip(cols, values)}
    return out


def _receipt_heads(records):
    return [{"family": r["family"], "horizon_hours": r["horizon_hours"],
             "target_timestamp": r["target_timestamp"], "value": r["value"]}
            for r in records if r["available"]]


def _skipped(records):
    return [{"family": r["family"], "horizon_hours": r["horizon_hours"], "reason": r["reason"]}
            for r in records if not r["available"]]


def management_decision(policy, *, direction: str, take_profit: float, stop_loss: float,
                        short_records, long_records) -> Dict[str, Any]:
    records = sorted(list(short_records) + list(long_records), key=lambda r: r["horizon_hours"])
    d = policy.early_close_decision(direction=direction, take_profit=take_profit,
                                    stop_loss=stop_loss, heads=policy_heads(records, policy))
    return {"action": d.action, "reason": d.reason,
            "first_crossing_horizon_hours": d.first_crossing_horizon_hours,
            "first_crossing_family": d.first_crossing_family,
            "consulted": _receipt_heads(records), "skipped": _skipped(records),
            "rules_version": d.rules_version}


def entry_decision_from_records(policy, *, current_price: float, long_records,
                                params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if any(r["family"] != "long" for r in long_records):
        raise StrategyAdapterError("SHORT_HEAD_IN_ENTRY", "entry consumes the long family only")
    p = policy.EntryParameters(**(params or DEFAULT_ENTRY_PARAMETERS))
    d = policy.entry_decision(current_price=current_price,
                              long_heads=policy_heads(long_records, policy), params=p)
    return {"action": d.action, "reason": d.reason,
            "geometry": None if d.geometry is None else asdict(d.geometry),
            "consulted": _receipt_heads(long_records), "skipped": _skipped(long_records),
            "rules_version": d.rules_version}


# --------------------------------------------------------------------------
# Pair eligibility (PLAN section 5)


def pair_eligibility(short_report: Mapping[str, Any], long_report: Mapping[str, Any], *,
                     min_valid_long_heads: int = 2, min_valid_short_heads: int = 1) -> Dict[str, Any]:
    """Per-head annual eligibility is MAE_h < naive_h (strict) on matching support."""
    eligible, status = {}, {}
    for family, report in (("short", short_report), ("long", long_report)):
        hs = FAMILIES[family]
        extra = set(report) - {str(h) for h in hs}
        if extra:
            raise StrategyAdapterError("HEAD_KEYS_MISMATCH", f"{family} report has {sorted(extra)}")
        eligible[family], status[family] = [], {}
        for h in hs:
            row = report.get(str(h))
            if row is None:
                status[family][str(h)] = "HEAD_NOT_REPORTED"
                continue
            mae, naive, n = row.get("mae"), row.get("naive"), row.get("n")
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and
                       math.isfinite(float(x)) for x in (mae, naive)) or not n:
                status[family][str(h)] = "NO_SCORE"
                ok = False
            else:
                ok = float(mae) < float(naive)
                status[family][str(h)] = "ELIGIBLE" if ok else "NOT_BETTER_THAN_NAIVE"
            if "eligible" in row and bool(row["eligible"]) != ok:
                raise StrategyAdapterError("ELIGIBILITY_FLAG_MISMATCH",
                                           f"{family} h{h}: flag {row['eligible']} vs MAE<naive {ok}")
            if ok:
                eligible[family].append(h)
    missing = []
    if len(eligible["short"]) < min_valid_short_heads:
        missing.append(f"SHORT_ELIGIBLE_HEADS_{len(eligible['short'])}_LT_{min_valid_short_heads}")
    if len(eligible["long"]) < min_valid_long_heads:
        missing.append(f"LONG_ELIGIBLE_HEADS_{len(eligible['long'])}_LT_{min_valid_long_heads}")
    return {"status": "PAIR_NOT_ELIGIBLE" if missing else "ELIGIBLE", "eligible": eligible,
            "head_status": status, "missing": missing,
            "rule": {"min_valid_long_heads": min_valid_long_heads,
                     "min_valid_short_heads": min_valid_short_heads,
                     "eligibility": "MAE_h < naive_h strictly, same support"}}


# --------------------------------------------------------------------------
# Execution


@dataclass(frozen=True)
class Bar:
    close_time: datetime
    open: float
    high: float
    low: float
    close: float

    def __post_init__(self):
        t = c1.utc_instant("close_time", self.close_time)
        if t.minute or t.second or t.microsecond:
            raise StrategyAdapterError("BAR_INVALID", f"{c1.iso(t)} is not on the hour")
        object.__setattr__(self, "close_time", t)
        vals = (self.open, self.high, self.low, self.close)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or v <= 0 for v in vals):
            raise StrategyAdapterError("BAR_INVALID", f"{c1.iso(t)}: non-positive or non-finite")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise StrategyAdapterError("BAR_INVALID", f"{c1.iso(t)}: high/low inconsistent")


@dataclass(frozen=True)
class ExecutionParameters:
    """Costs default to the legacy plugin's effective values (spread_pips 2.0 and
    slippage_pips 1.0 times pip_cost 0.00001, commission 7 per lot round trip,
    swap 10 per lot per day). Owner ruling pending on these magnitudes."""

    initial_equity: float = 10000.0
    spread_price: float = 2.0 * 0.00001
    slippage_price: float = 1.0 * 0.00001
    commission_per_lot_round_trip: float = 7.0
    swap_per_lot_per_day: float = 10.0
    lot_units: float = 100000.0
    force_close_lead_hours: int = 1
    sharpe_periods_per_year: int = 260
    passive_reference: bool = True

    def __post_init__(self):
        for k in ("initial_equity", "lot_units"):
            if not getattr(self, k) > 0:
                raise StrategyAdapterError("PARAMETER_INVALID", k)
        for k in ("spread_price", "slippage_price", "commission_per_lot_round_trip",
                  "swap_per_lot_per_day"):
            if getattr(self, k) < 0:
                raise StrategyAdapterError("PARAMETER_INVALID", k)
        if type(self.force_close_lead_hours) is not int or self.force_close_lead_hours < 1:
            raise StrategyAdapterError("PARAMETER_INVALID", "force_close_lead_hours >= 1")


def _week_at(weeks, t):
    for w in weeks:
        if w.market_open <= t < w.market_close:
            return w
    return None


class _Book:
    def __init__(self, ex: ExecutionParameters):
        self.ex = ex
        self.cash = ex.initial_equity
        self.position = None
        self.pending = None
        self.trades: List[Dict[str, Any]] = []
        self.cancelled: List[Dict[str, Any]] = []
        self.failures: Dict[str, int] = {}
        self.costs = {"spread": 0.0, "slippage": 0.0, "commission": 0.0, "swap": 0.0}
        self.notional = 0.0

    def fail(self, code):
        self.failures[code] = self.failures.get(code, 0) + 1

    def fill_price(self, reference, buy):
        adj = self.ex.spread_price / 2 + self.ex.slippage_price
        return reference + adj if buy else reference - adj

    def commission_per_fill(self, units):
        return self.ex.commission_per_lot_round_trip / 2 * units / self.ex.lot_units

    def swap(self, units, start, end):
        return self.ex.swap_per_lot_per_day * units / self.ex.lot_units * ((end - start) / HOUR) / 24.0

    def open_position(self, order, bar):
        buy = order["direction"] == "long"
        price = self.fill_price(bar.open, buy)
        self.position = {**order, "entry_reference_price": bar.open, "entry_fill_price": price,
                         "entry_fill_time": c1.iso(bar.close_time - HOUR),
                         "entry_fill_bar_close_time": c1.iso(bar.close_time),
                         "_entry_time": bar.close_time - HOUR, "_fill_bar": bar.close_time}
        self.notional += order["units"] * price

    def close_position(self, *, reference, fill_instant, fill_bar_close, reason,
                       decision_time, deadline):
        pos = self.position
        units, buy_entry = pos["units"], pos["direction"] == "long"
        sign = 1.0 if buy_entry else -1.0
        exit_price = self.fill_price(reference, not buy_entry)
        commission = 2 * self.commission_per_fill(units)
        swap = self.swap(units, pos["_entry_time"], fill_instant)
        gross = sign * units * (exit_price - pos["entry_fill_price"])
        net = gross - commission - swap
        spread = units * self.ex.spread_price
        slippage = units * 2 * self.ex.slippage_price
        self.costs["spread"] += spread
        self.costs["slippage"] += slippage
        self.costs["commission"] += commission
        self.costs["swap"] += swap
        self.notional += units * exit_price
        self.cash += net
        violation = deadline is not None and fill_bar_close > deadline
        if violation:
            self.fail("FORCE_FLAT_FILL_AFTER_DEADLINE")
        trade = {k: v for k, v in pos.items() if not k.startswith("_")}
        trade.update({"exit_reason": reason, "exit_decision_time": decision_time,
                      "exit_reference_price": reference, "exit_fill_price": exit_price,
                      "exit_fill_time": c1.iso(fill_instant),
                      "exit_fill_bar_close_time": c1.iso(fill_bar_close),
                      "gross_pnl_mid": sign * units * (reference - pos["entry_reference_price"]),
                      "spread_cost": spread, "slippage_cost": slippage,
                      "commission": commission, "swap": swap, "net_pnl": net,
                      "force_flat_violation": violation})
        self.trades.append(trade)
        self.position = None

    def equity(self, mark, at):
        if self.position is None:
            return self.cash
        pos = self.position
        sign = 1.0 if pos["direction"] == "long" else -1.0
        return (self.cash + sign * pos["units"] * (mark - pos["entry_fill_price"])
                - self.commission_per_fill(pos["units"])
                - self.swap(pos["units"], pos["_entry_time"], at))


def _protective(book, bar):
    pos = book.position
    if pos is None or pos.get("take_profit") is None:
        return
    tp, sl, long_ = pos["take_profit"], pos["stop_loss"], pos["direction"] == "long"
    first_bar = bar.close_time == pos["_fill_bar"]
    if not first_bar:
        gap_sl = bar.open <= sl if long_ else bar.open >= sl
        gap_tp = bar.open >= tp if long_ else bar.open <= tp
        if gap_sl or gap_tp:
            book.close_position(reference=bar.open, fill_instant=bar.close_time - HOUR,
                                fill_bar_close=bar.close_time,
                                reason="STOP_LOSS" if gap_sl else "TAKE_PROFIT",
                                decision_time=None, deadline=pos["_deadline_close"])
            return
    sl_hit = bar.low <= sl if long_ else bar.high >= sl
    tp_hit = bar.high >= tp if long_ else bar.low <= tp
    if sl_hit or tp_hit:  # both in one bar: stop loss first (declared, conservative)
        book.close_position(reference=sl if sl_hit else tp, fill_instant=bar.close_time,
                            fill_bar_close=bar.close_time,
                            reason="STOP_LOSS" if sl_hit else "TAKE_PROFIT",
                            decision_time=None, deadline=pos["_deadline_close"])


def _simulate(*, bars, weeks, short_predictions, long_predictions, policy, entry_params,
              ex: ExecutionParameters, validation_eligible, passive: bool):
    book = _Book(ex)
    params = policy.EntryParameters(**entry_params)
    lead = ex.force_close_lead_hours * HOUR
    decisions: List[Dict[str, Any]] = []
    curve: List[tuple] = []
    position_at: Dict[str, float] = {}
    entries: List[datetime] = []
    passive_weeks = set()
    counters = {"origins_without_predictions": 0}
    ve = validation_eligible or {}
    for bar in bars:
        t = bar.close_time
        # 1. pending order at this bar's open
        if book.pending is not None:
            order, book.pending = book.pending, None
            if order["kind"] == "ENTRY":
                if t <= order["deadline"]:
                    book.open_position(order["position"], bar)
                else:
                    book.cancelled.append({"decision_time": order["position"]["entry_decision_time"],
                                           "reason": "ENTRY_FILL_AFTER_DEADLINE",
                                           "next_bar_close_time": c1.iso(t)})
            else:
                book.close_position(reference=bar.open, fill_instant=t - HOUR,
                                    fill_bar_close=t, reason=order["reason"],
                                    decision_time=order["decision_time"],
                                    deadline=order["deadline"])
        # 2. protective orders inside the bar
        _protective(book, bar)
        # 3. mark to market at the close
        curve.append((t, book.equity(bar.close, t)))
        pos = book.position
        position_at[c1.iso(t)] = 0.0 if pos is None else \
            (pos["units"] if pos["direction"] == "long" else -pos["units"])
        # 4. decision at origin t
        if book.pending is not None:
            continue
        if pos is not None:
            pweek = pos["_week"]
            if t >= pweek.force_flat - lead:
                book.pending = {"kind": "CLOSE", "reason": "FORCE_FLAT", "decision_time": c1.iso(t),
                                "deadline": pweek.force_flat}
                decisions.append({"origin": c1.iso(t), "role": "MANAGEMENT",
                                  "action": "FORCE_CLOSE", "reason": "FORCE_FLAT_LEAD",
                                  "consulted": [], "skipped": []})
                continue
            if passive or pos.get("take_profit") is None:
                continue
            if t not in short_predictions or t not in long_predictions:
                counters["origins_without_predictions"] += 1
                continue
            week = _week_at(weeks, t)
            if week is None or week.week_id != pweek.week_id:
                continue
            srec = build_family_heads(origin=t, family="short", predicted=short_predictions[t],
                                      week=week, role=c1.Role.MANAGEMENT,
                                      validation_eligible=ve.get("short"))
            lrec = build_family_heads(origin=t, family="long", predicted=long_predictions[t],
                                      week=week, role=c1.Role.MANAGEMENT,
                                      validation_eligible=ve.get("long"))
            d = management_decision(policy, direction=pos["direction"],
                                    take_profit=pos["take_profit"], stop_loss=pos["stop_loss"],
                                    short_records=srec, long_records=lrec)
            decisions.append({"origin": c1.iso(t), "role": "MANAGEMENT", **d})
            if d["action"] == "EARLY_CLOSE":
                book.pending = {"kind": "CLOSE", "reason": "EARLY_CLOSE_SL_FIRST",
                                "decision_time": c1.iso(t), "deadline": pweek.force_flat}
            continue
        week = _week_at(weeks, t)
        if week is None or not (week.operational_start <= t < min(week.entry_cutoff, week.force_flat)):
            continue
        recent = [e for e in entries if t - e < 120 * HOUR]
        if len(recent) >= params.max_trades_per_5days:
            continue
        deadline = min(week.entry_cutoff, week.force_flat - lead)
        if passive:
            if week.week_id in passive_weeks:
                continue
            passive_weeks.add(week.week_id)
            units = min(params.max_order_volume, book.cash * params.rel_volume * params.leverage)
            order = {"direction": "long", "units": units, "take_profit": None, "stop_loss": None,
                     "entry_decision_time": c1.iso(t), "week_id": week.week_id,
                     "_week": week, "_deadline_close": week.force_flat}
            book.pending = {"kind": "ENTRY", "deadline": deadline, "position": order}
            entries.append(t)
            continue
        if t not in long_predictions:
            counters["origins_without_predictions"] += 1
            continue
        lrec = build_family_heads(origin=t, family="long", predicted=long_predictions[t],
                                  week=week, role=c1.Role.ENTRY,
                                  validation_eligible=ve.get("long"))
        d = entry_decision_from_records(policy, current_price=bar.close, long_records=lrec,
                                        params=entry_params)
        decisions.append({"origin": c1.iso(t), "role": "ENTRY", **d})
        if d["action"] == "ABSTAIN":
            continue
        g = d["geometry"]
        units = policy.compute_order_units(reward_risk_ratio=g["reward_risk_ratio"],
                                           available_cash=book.cash, params=params)
        if not units > 0:
            continue
        order = {"direction": g["direction"], "units": units,
                 "take_profit": g["take_profit_price"], "stop_loss": g["stop_loss_price"],
                 "entry_decision_time": c1.iso(t), "week_id": week.week_id,
                 "_week": week, "_deadline_close": week.force_flat}
        book.pending = {"kind": "ENTRY", "deadline": deadline, "position": order}
        entries.append(t)
    # End of data: no fill is invented.
    open_positions = []
    if book.pending is not None and book.pending["kind"] == "ENTRY":
        book.cancelled.append({"decision_time": book.pending["position"]["entry_decision_time"],
                               "reason": "ENTRY_UNFILLED_AT_END_OF_DATA", "next_bar_close_time": None})
    if book.position is not None:
        pos, last = book.position, bars[-1]
        if book.pending is not None and book.pending["kind"] == "CLOSE":
            status = "OPEN_NO_FILL"
            book.fail("CLOSE_ORDER_UNFILLED_AT_END")
            if book.pending["reason"] == "FORCE_FLAT":
                book.fail("FORCE_FLAT_NOT_FILLED")
        else:
            status = "OPEN_AT_END_OF_DATA"
            book.fail("POSITION_OPEN_AT_END_OF_DATA")
        sign = 1.0 if pos["direction"] == "long" else -1.0
        open_positions.append({**{k: v for k, v in pos.items() if not k.startswith("_")},
                               "status": status, "mark_price": last.close,
                               "mark_time": c1.iso(last.close_time),
                               "unrealized_pnl": sign * pos["units"] * (last.close - pos["entry_fill_price"]),
                               "accrued_commission": book.commission_per_fill(pos["units"]),
                               "accrued_swap": book.swap(pos["units"], pos["_entry_time"],
                                                         last.close_time)})
    return book, decisions, curve, position_at, open_positions, counters


def _metrics(book, curve, position_at, open_positions, ex):
    init = ex.initial_equity
    final = curve[-1][1] if curve else init
    peak, mdd, mdd_frac = init, 0.0, 0.0
    for _, e in curve:
        peak = max(peak, e)
        if peak - e > mdd:
            mdd = peak - e
        if peak > 0 and (peak - e) / peak > mdd_frac:
            mdd_frac = (peak - e) / peak
    daily: Dict[str, float] = {}
    for t, e in curve:
        daily[t.date().isoformat()] = e
    returns, prev = [], init
    for day in sorted(daily):
        returns.append(daily[day] / prev - 1.0)
        prev = daily[day]
    sharpe = sortino = None
    sharpe_reason = sortino_reason = None
    ppy = ex.sharpe_periods_per_year
    if len(returns) < 2:
        sharpe_reason = sortino_reason = "INSUFFICIENT_RETURNS"
    else:
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        if var <= 0:
            sharpe_reason = "ZERO_RETURN_VARIANCE"
        else:
            sharpe = mean / math.sqrt(var) * math.sqrt(ppy)
        down = math.sqrt(sum(min(r, 0.0) ** 2 for r in returns) / len(returns))
        if down <= 0:
            sortino_reason = "NO_DOWNSIDE_RETURNS"
        else:
            sortino = mean / down * math.sqrt(ppy)
    pnl = [t["net_pnl"] for t in book.trades]
    gp = sum(p for p in pnl if p > 0)
    gl = -sum(p for p in pnl if p < 0)
    if not pnl:
        pf, pf_reason = None, "NO_CLOSED_TRADES"
    elif gl == 0:
        pf, pf_reason = None, "NO_LOSING_TRADES"
    else:
        pf, pf_reason = gp / gl, None
    exposure = (sum(1 for v in position_at.values() if v) / len(position_at)) if position_at else None
    return {"initial_equity": init, "final_equity": final, "net_pnl": final - init,
            "return": (final - init) / init,
            "realized_pnl": sum(pnl), "closed_trades": len(book.trades),
            "open_trades": len(open_positions),
            "winning_trades": sum(1 for p in pnl if p > 0),
            "losing_trades": sum(1 for p in pnl if p < 0),
            "max_drawdown": mdd, "max_drawdown_fraction": mdd_frac,
            "return_clock": "UTC_DAY_LAST_MARK_TO_MARKET", "sharpe_periods_per_year": ppy,
            "daily_returns": len(returns),
            "sharpe": sharpe, "sharpe_reason": sharpe_reason,
            "sortino": sortino, "sortino_reason": sortino_reason,
            "profit_factor": pf, "profit_factor_reason": pf_reason,
            "exposure": exposure, "turnover": book.notional / init,
            "costs": dict(book.costs)}


def run_backtest(*, bars: Sequence[Bar], weeks: Sequence[c1.OperationalWeek],
                 short_predictions: Mapping[datetime, Mapping[int, Any]],
                 long_predictions: Mapping[datetime, Mapping[int, Any]],
                 policy: Optional[types.ModuleType] = None,
                 entry_params: Optional[Mapping[str, Any]] = None,
                 execution: Optional[ExecutionParameters] = None,
                 validation_eligible: Optional[Mapping[str, Iterable[int]]] = None,
                 allow_test: bool = False) -> Dict[str, Any]:
    """One-asset business backtest of the pair policy; synthetic or validation data only."""
    policy = policy or load_pinned_policy()
    ex = execution or ExecutionParameters()
    entry_params = dict(entry_params or DEFAULT_ENTRY_PARAMETERS)
    bars = list(bars)
    if not bars:
        raise StrategyAdapterError("NO_BARS", "bars are required")
    for a, b in zip(bars, bars[1:]):
        if not a.close_time < b.close_time:
            raise StrategyAdapterError("BAR_ORDER_INVALID", c1.iso(b.close_time))
    if not allow_test:
        late = [t for t in [bars[-1].close_time, *short_predictions, *long_predictions]
                if t >= TEST_START]
        if late:
            raise StrategyAdapterError("TEST_FORBIDDEN", f"{c1.iso(min(late))} >= 2025-01-01")
    for preds in (short_predictions, long_predictions):
        for t in preds:
            c1.utc_instant("prediction origin", t)
    book, decisions, curve, position_at, open_positions, counters = _simulate(
        bars=bars, weeks=weeks, short_predictions=short_predictions,
        long_predictions=long_predictions, policy=policy, entry_params=entry_params, ex=ex,
        validation_eligible=validation_eligible, passive=False)
    result = {"schema": SCHEMA + ".backtest", "policy": policy_identity(policy),
              "entry_parameters": entry_params, "execution": asdict(ex),
              "rules": {**EXECUTION_RULES, "policy": dict(policy.RULES)},
              "trades": book.trades, "open_positions": open_positions,
              "cancelled_orders": book.cancelled, "failures": dict(book.failures),
              "decisions": decisions, "position_at": position_at,
              "equity_curve": [(c1.iso(t), e) for t, e in curve], "counters": counters,
              "metrics": _metrics(book, curve, position_at, open_positions, ex)}
    refs = {"cash": {"final_equity": ex.initial_equity, "net_pnl": 0.0, "return": 0.0}}
    if ex.passive_reference:
        pbook, _, pcurve, ppos_at, popen, _ = _simulate(
            bars=bars, weeks=weeks, short_predictions={}, long_predictions={}, policy=policy,
            entry_params=entry_params, ex=ex, validation_eligible=None, passive=True)
        refs["business_passive_long"] = {
            "label": "BUSINESS_MATCHED_PASSIVE_LONG",
            "rule": "long at the first entry origin of each week, max admissible units, force flat",
            "metrics": _metrics(pbook, pcurve, ppos_at, popen, ex),
            "failures": dict(pbook.failures)}
    units = min(entry_params["max_order_volume"],
                ex.initial_equity * entry_params["rel_volume"] * entry_params["leverage"])
    refs["conventional_buy_and_hold"] = {
        "label": "CONVENTIONAL_BUY_AND_HOLD_VIOLATES_NO_WEEKEND_POSITION",
        "units": units, "net_pnl_before_costs": units * (bars[-1].close - bars[0].open)}
    result["references"] = refs
    return result


# --------------------------------------------------------------------------
# CLI


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.i6dmo.strategy_adapter")
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("map-plugin", help="report the strategy code each route would load")
    m.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "map-plugin":
        report = map_strategy_resolution()
        print(json.dumps(report, indent=1, sort_keys=True) if args.json else
              f"executor_source={report['executor_source']} "
              f"policy={report['pinned_policy']['sha256']} "
              f"entry_point={report['entry_point'].get('resolution')}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
