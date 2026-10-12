"""I6N point-in-time availability gate for input features (owner order 2026-10-12).

Every external input feature must carry a declaration in the sealed contract
``feature_availability.v1.json`` (next to this module):

``event_time``       what instant/date a value refers to (``SOURCE_DATE`` or ``BAR_END``)
``available_time``   when that value became publicly available (rule + basis)
``timezone``         the IANA zone in which a source date is anchored
``availability_source``  VERIFIED_EVIDENCE | DECLARED_CONSERVATIVE | UNVERIFIED
``revision_policy``  how later revisions of the value are treated
``consumption``      the as-of rule the retained input actually applies
                     (the first instant a window can see the value)

The gate's invariant is the one the owner stated: a window may consume a value
only when ``available_time <= t``.  For a retained input this reduces to
``consumed_from(event) >= available_at(event)`` for every event date in the
checked span; any earlier consumption is an ``AVAILABILITY_VIOLATION``.  A
feature without a declaration is refused (``UNDECLARED_FEATURE``) and a
declared-but-excluded feature is refused (``EXCLUDED_FEATURE``).  The gate never
shifts a feature by an assumed lag and never drops a feature silently.

Pure standard library; no I/O besides reading the bundled contract.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA = "i6n.feature_availability.contract.v1"
CONTRACT_PATH = Path(__file__).with_name("feature_availability.v1.json")
SOURCES = ("VERIFIED_EVIDENCE", "DECLARED_CONSERVATIVE", "UNVERIFIED")
STATUSES = ("RETAINED", "EXCLUDED")
EVENT_KINDS = ("SOURCE_DATE", "BAR_END")
AVAILABLE_KINDS = ("DATE_OFFSET", "NEXT_WEEKDAY", "BAR_END")
CONSUMPTION_KINDS = ("DATE_OFFSET", "BAR_END")
REVISION_KINDS = ("NOT_REVISED", "LATEST_VINTAGE_UNVERIFIED", "RETROACTIVE_SPLIT_ADJUSTMENT",
                  "VENDOR_BARS_AS_RETAINED")
REQUIRED_FIELDS = ("feature", "status", "event_time", "available_time", "timezone",
                   "availability_source", "availability_evidence", "revision_policy", "consumption")

#: The sealed 20-feature manifest of plan 2.0.0 (H4 and the superseded H72 identity).
LEGACY_MANIFEST_20 = (
    "fred.inflation_expectations.t5yie.level",
    "fred.rates.dgs3mo.diff_5d",
    "fred.rates.dgs5.level",
    "fred.rates.dgs7.logret_5d",
    "fred.rates.dprime.logret_5d",
    "fred.stress.stlfsi4.level",
    "fx.gbpusd.logret_1h",
    "px.close_loc",
    "px.log_co",
    "px.rv5",
    "rg.atr_ratio",
    "yh.eem.logret_5d",
    "yh.iyr.logret_1d",
    "yh.kc_f.logret_1d",
    "yh.n225.logret_1d",
    "yh.si_f.logret_1d",
    "yh.slv.logret_1d",
    "yh.vnq.logret_1d",
    "yh.xlc.logret_1d",
    "yh.xlre.logret_1d",
)
EXCLUDED_FEATURES = ("fred.stress.stlfsi4.level",)
#: Owner order 2026-10-12: the 20-feature manifest minus STLFSI4, same order, no shift.
MANIFEST_NO_STLFSI4 = tuple(f for f in LEGACY_MANIFEST_20 if f not in EXCLUDED_FEATURES)


class AvailabilityGateError(ValueError):
    """Typed refusal; the message starts with a stable code."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def manifest_sha256(features: Sequence[str]) -> str:
    """Order-sensitive identity of a feature manifest (column order is identity)."""
    return _sha({"schema": "i6n.feature_manifest.v1", "features": list(features)})


def contract_sha256(contract: Mapping[str, Any]) -> str:
    return _sha({k: v for k, v in contract.items() if k != "contract_sha256"})


def _invalid(feature: Any, detail: str) -> AvailabilityGateError:
    return AvailabilityGateError("INVALID_DECLARATION", f"{feature}: {detail}")


def _int_hours(value: Any, feature: str, name: str) -> int:
    if type(value) is not int or not -24 * 14 <= value <= 24 * 14:
        raise _invalid(feature, f"{name} must be an integer number of hours")
    return value


def _validate_declaration(row: Any) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise _invalid(row, "declaration must be an object")
    feature = row.get("feature", "<missing feature>")
    missing = [f for f in REQUIRED_FIELDS if f not in row]
    if missing:
        raise _invalid(feature, f"missing {missing}")
    if not isinstance(feature, str) or not feature.strip():
        raise _invalid(feature, "feature must be a nonempty name")
    if row["status"] not in STATUSES:
        raise _invalid(feature, f"status must be one of {STATUSES}")
    if row["availability_source"] not in SOURCES:
        raise _invalid(feature, f"availability_source must be one of {SOURCES}")
    if not isinstance(row["availability_evidence"], str) or not row["availability_evidence"].strip():
        raise _invalid(feature, "availability_evidence must be nonempty text")
    tz = row["timezone"]
    try:
        if not isinstance(tz, str) or not tz:
            raise ZoneInfoNotFoundError(tz)
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        raise _invalid(feature, f"timezone {tz!r} is not an IANA zone") from None
    event, available, consumption = row["event_time"], row["available_time"], row["consumption"]
    if not isinstance(event, Mapping) or event.get("kind") not in EVENT_KINDS:
        raise _invalid(feature, f"event_time.kind must be one of {EVENT_KINDS}")
    weekdays = event.get("event_weekdays", [0, 1, 2, 3, 4])
    if (not isinstance(weekdays, list) or not weekdays
            or any(type(d) is not int or not 0 <= d <= 6 for d in weekdays)):
        raise _invalid(feature, "event_time.event_weekdays must list weekdays 0..6")
    if not isinstance(available, Mapping) or available.get("kind") not in AVAILABLE_KINDS:
        raise _invalid(feature, f"available_time.kind must be one of {AVAILABLE_KINDS}")
    if not isinstance(available.get("basis"), str) or not available["basis"].strip():
        raise _invalid(feature, "available_time.basis must name the publication fact")
    if available["kind"] == "NEXT_WEEKDAY":
        if type(available.get("weekday")) is not int or not 0 <= available["weekday"] <= 6:
            raise _invalid(feature, "available_time.weekday must be 0..6")
        _int_hours(available.get("hour"), feature, "available_time.hour")
    else:
        _int_hours(available.get("lag_hours"), feature, "available_time.lag_hours")
    if not isinstance(consumption, Mapping) or consumption.get("kind") not in CONSUMPTION_KINDS:
        raise _invalid(feature, f"consumption.kind must be one of {CONSUMPTION_KINDS}")
    _int_hours(consumption.get("lag_hours"), feature, "consumption.lag_hours")
    if consumption.get("source") not in SOURCES:
        raise _invalid(feature, f"consumption.source must be one of {SOURCES}")
    bar = event["kind"] == "BAR_END"
    if bar != (available["kind"] == "BAR_END") or bar != (consumption["kind"] == "BAR_END"):
        raise _invalid(feature, "BAR_END events require BAR_END availability and consumption")
    policy = row["revision_policy"]
    if (not isinstance(policy, Mapping) or policy.get("kind") not in REVISION_KINDS
            or not isinstance(policy.get("note"), str)):
        raise _invalid(feature, f"revision_policy.kind must be one of {REVISION_KINDS}")
    # The weakest evidence governs the declaration's own source label.
    rank = {s: i for i, s in enumerate(SOURCES)}
    if rank[row["availability_source"]] < rank[consumption["source"]]:
        raise _invalid(feature, "availability_source cannot be stronger than its consumption source")
    return dict(row)


def validate_contract(contract: Any, *, seal: bool = True) -> dict[str, Any]:
    """Validate structure (and the self seal unless ``seal=False``) and return a copy."""
    if not isinstance(contract, Mapping) or contract.get("schema") != SCHEMA:
        raise AvailabilityGateError("INVALID_CONTRACT", f"schema must equal {SCHEMA}")
    rows = contract.get("declarations")
    if not isinstance(rows, list) or not rows:
        raise AvailabilityGateError("INVALID_CONTRACT", "declarations must be a nonempty list")
    seen = set()
    for row in rows:
        clean = _validate_declaration(row)
        if clean["feature"] in seen:
            raise _invalid(clean["feature"], "declared twice")
        seen.add(clean["feature"])
    span = contract.get("checked_span")
    try:
        start, end = date.fromisoformat(span["start"]), date.fromisoformat(span["end"])
    except (TypeError, KeyError, ValueError):
        raise AvailabilityGateError("INVALID_CONTRACT", "checked_span needs ISO start/end dates") from None
    if end < start:
        raise AvailabilityGateError("INVALID_CONTRACT", "checked_span end precedes start")
    if seal and contract.get("contract_sha256") != contract_sha256(contract):
        raise AvailabilityGateError("CONTRACT_SEAL_MISMATCH", "contract bytes changed after sealing")
    return deepcopy(dict(contract))


def load_contract(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else CONTRACT_PATH
    return validate_contract(json.loads(path.read_text(encoding="utf-8")))


def declaration(contract: Mapping[str, Any], feature: str) -> dict[str, Any]:
    for row in contract["declarations"]:
        if row["feature"] == feature:
            return deepcopy(row)
    raise AvailabilityGateError("UNDECLARED_FEATURE", feature)


def retained_features(contract: Mapping[str, Any]) -> list[str]:
    return [row["feature"] for row in contract["declarations"] if row["status"] == "RETAINED"]


def _anchor(decl: Mapping[str, Any], event: date | datetime) -> datetime:
    """The event instant: a source date at 00:00 in its zone, or the bar end itself."""
    if decl["event_time"]["kind"] == "BAR_END":
        if not isinstance(event, datetime) or event.tzinfo is None:
            raise ValueError("BAR_END events are timezone-aware datetimes")
        return event.astimezone(timezone.utc)
    day = event.date() if isinstance(event, datetime) else event
    return datetime.combine(day, time(0), ZoneInfo(decl["timezone"])).astimezone(timezone.utc)


def available_at(decl: Mapping[str, Any], event: date | datetime) -> datetime:
    """Instant at which the value for ``event`` became publicly available (UTC)."""
    anchor = _anchor(decl, event)
    rule = decl["available_time"]
    if rule["kind"] == "NEXT_WEEKDAY":
        local = anchor.astimezone(ZoneInfo(decl["timezone"]))
        days = (rule["weekday"] - local.weekday()) % 7 or 7  # strictly after the event date
        target = datetime.combine(local.date() + timedelta(days=days), time(0), ZoneInfo(decl["timezone"]))
        return (target + timedelta(hours=rule["hour"])).astimezone(timezone.utc)
    return anchor + timedelta(hours=rule["lag_hours"])


def consumed_from(decl: Mapping[str, Any], event: date | datetime) -> datetime:
    """First instant at which the retained input exposes the value for ``event``."""
    return _anchor(decl, event) + timedelta(hours=decl["consumption"]["lag_hours"])


def admissible(decl: Mapping[str, Any], event: date | datetime, t: datetime) -> bool:
    """A window ending at ``t`` may consume the value only when available_time <= t."""
    if not isinstance(t, datetime) or t.tzinfo is None:
        raise ValueError("t must be a timezone-aware datetime")
    return available_at(decl, event) <= t


def window_mask(decl: Mapping[str, Any], events: Iterable[date | datetime],
                ts: Iterable[datetime]) -> list[bool]:
    return [admissible(decl, e, t) for e, t in zip(events, ts, strict=True)]


def _events(decl: Mapping[str, Any], start: date, end: date):
    weekdays = set(decl["event_time"].get("event_weekdays", [0, 1, 2, 3, 4]))
    day = start
    while day <= end:
        if day.weekday() in weekdays:
            if decl["event_time"]["kind"] == "BAR_END":
                yield datetime.combine(day, time(12), timezone.utc)
            else:
                yield day
        day += timedelta(days=1)


def verify_declaration(decl: Mapping[str, Any], start: date, end: date) -> dict[str, Any]:
    """Check consumed_from(e) >= available_at(e) for every event in [start, end]."""
    violations, margins, first = 0, [], None
    for event in _events(decl, start, end):
        margin = (consumed_from(decl, event) - available_at(decl, event)).total_seconds() / 3600
        margins.append(margin)
        if margin < 0:
            violations += 1
            first = first or (event.isoformat(), margin)
    return {
        "events": len(margins), "violations": violations,
        "min_margin_hours": min(margins) if margins else None,
        "first_violation": None if first is None else {"event": first[0], "early_by_hours": -first[1]},
    }


def gate(features: Sequence[str], contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Refuse any feature without a passing declaration; return a typed report."""
    contract = load_contract() if contract is None else contract
    if (not isinstance(features, (list, tuple)) or not features
            or any(not isinstance(f, str) or not f.strip() for f in features)
            or len(set(features)) != len(features)):
        raise AvailabilityGateError("INVALID_MANIFEST", "features must be unique nonempty names")
    span = contract["checked_span"]
    start, end = date.fromisoformat(span["start"]), date.fromisoformat(span["end"])
    by_name = {row["feature"]: row for row in contract["declarations"]}
    rows = {}
    for feature in features:
        decl = by_name.get(feature)
        if decl is None:
            raise AvailabilityGateError("UNDECLARED_FEATURE", feature)
        if decl["status"] != "RETAINED":
            raise AvailabilityGateError("EXCLUDED_FEATURE", feature)
        check = verify_declaration(decl, start, end)
        if check["violations"]:
            raise AvailabilityGateError(
                "AVAILABILITY_VIOLATION",
                f"{feature}: {check['violations']} events consumed before available_time; "
                f"first {check['first_violation']}")
        rows[feature] = {
            "availability_source": decl["availability_source"],
            "consumption_source": decl["consumption"]["source"],
            "revision_policy": decl["revision_policy"]["kind"],
            **check,
        }
    return {
        "schema": "i6n.feature_availability.gate_report.v1",
        "status": "PASS",
        "contract_sha256": contract.get("contract_sha256"),
        "manifest_sha256": manifest_sha256(features),
        "checked_span": {"start": span["start"], "end": span["end"]},
        "features": rows,
    }
