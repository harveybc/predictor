"""I6-N C1: pure per-head horizon eligibility and support table.

Candidate implementation of the owner order
``SATOSHI_I6N_BOUNDED_ENGINEERING_2026_10_11.md`` section 4 and of
``BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md`` sections 2-3. It is not wired
into any executor, does not train, and its outputs are not measurements.

Vocabulary (one operational week of a SUPPLIED, versioned calendar)::

    O  market open                       (calendar)
    A  operational start = O + entry_delay_after_open_hours
    K  new-entry cutoff                  (calendar-resolved instant)
    B  force-flat deadline = C - force_flat_before_close_hours
    C  market close                      (calendar)

Two different eligibilities are computed for an origin ``t`` and a head with
horizon ``h`` (elapsed hours, so ``q = t + h`` is a UTC instant):

``E`` operational, known at ``t``: only the calendar is consulted.
    ENTRY:      A <= t < min(K, B)  and  q < B
    MANAGEMENT: A <= t < B          and  q < B
    ``q < B`` is strict (a target at B falls when the book must already be
    flat) and, because B < C of the same week, it also rejects any target in
    the next operational week even when it lands inside that week's [A, B).

``M`` historical label: ``E`` AND input availability at ``t`` AND the exact
    target bar ``q`` observed AND (if declared) the label available by the
    split's ``label_available_by`` AND ``split.start <= q < split.end``.
    Split membership is decided by the TARGET instant ``q`` (market-week
    contract section 2: the objective must belong to the split); the end is
    exclusive. A later missing bar changes ``M`` and never ``E``.

``available`` per prediction (market-week contract section 3): an explicit,
    input-driven field. The caller supplies ``prediction_available`` (inputs,
    checkpoint and execution correct for that head); ``available = E AND
    prediction_available`` with typed :class:`AvailabilityReason`. Not supplied
    => ``available is None`` (never a default True/False). It is recorded apart
    from ``M`` and never changes ``E`` or ``M``; this module does not decide
    whether a model actually produced a prediction.

What this module never does: infer O/C from gaps in prices, choose a broker
or a time zone, fall back to a default (e.g. New York) calendar, fill a value,
move or truncate an endpoint, convert ``h`` into "hours remaining", or drop
the other heads of an origin because one head is ineligible.

Support counts are produced by ENUMERATING the declared origin timestamps per
week/role/head. The closed form ``(B - A) - h`` is only valid for an aligned
hourly grid, a half-open interval, integer durations and no other cuts or
absences; it is deliberately not emitted.

Dependencies: standard library only (no pandas, TensorFlow, network or
database module; ``sqlite3`` is not imported, directly or through helpers).
``_require_plain_int`` is reused from the sealed weekly protocol.

Limits (candidate implementation, not deployed, not a measurement):

* ``prediction_available`` is a caller-supplied fact. Deciding it from real
  inputs, checkpoints and execution belongs to a later adapter.
* A week with ``A >= B`` is rejected (``OPERATIONAL_WINDOW_EMPTY``); a cutoff
  ``K <= A`` is accepted as a week without new entries.
* ``input_unavailable_origins`` must each equal one enumerated in-calendar
  origin and requires label evidence. An origin that matches but is never
  operationally eligible for a head does not change that head's counts,
  because input availability only acts on ``M`` and ``M`` implies ``E``.
* Keras masking is not exercised or validated here.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, ClassVar, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from tools.business_weekly_protocol import _require_plain_int

ELIGIBILITY_VERSION = "i6n.eligibility.v1"
CONFIG_SCHEMA_VERSION = "i6n.eligibility.config.v1"
TABLE_SCHEMA_VERSION = "i6n.eligibility.support_table.v1"
UTC = timezone.utc
HOUR = timedelta(hours=1)
SECOND = timedelta(seconds=1)

#: Design documents read for this candidate (uncommitted working-tree bytes at
#: base 269c2d132b21a55e59cd6faa8242b1ac74f0f7ae; verified against the audit
#: request section 2). Recorded as provenance; this module does not read them.
CONTRACT_SOURCES: Tuple[Mapping[str, str], ...] = (
    {"path": "docs/handoffs/SATOSHI_I6N_BOUNDED_ENGINEERING_2026_10_11.md",
     "sha256": "243fc9e18bc82145962e67e6c2adb361344ba0aeedee978c12320476e1ad2025"},
    {"path": "docs/handoffs/I6N_STRICT_INDEPENDENT_PLAN_AUDIT_REQUEST_2026_10_11.md",
     "sha256": "2e060f7d27b4ee74ead00e3f7987269892f0fa57490dcd8b55530c9cde5c605f"},
    {"path": "docs/tres_temas_entrevista/program_v3/BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md",
     "sha256": "604f94809c41613e7346ed630d698591fadd7ecc9961b9e3cb04fcad00c01c76"},
    {"path": "docs/tres_temas_entrevista/program_v3/BUSINESS_WEEKLY_WALK_FORWARD_CONTRACT_2026_10_03.md",
     "sha256": "d64206b875462c913087ccba387b98422ae51c0ab88c6f490362532867b5a4c1"},
    {"path": "docs/tres_temas_entrevista/program_v3/I6N_JOINT_CONTEXT_CAPACITY_PROPOSAL.md",
     "sha256": "d4341e72f4fc05531cfcffd5036c1968d522a105fa534d2add0ae2d10eb56487"},
)


# --------------------------------------------------------------------------
# Typed vocabulary


class Role(str, Enum):
    """Use of a forecast origin by the strategy."""

    ENTRY = "ENTRY"
    MANAGEMENT = "MANAGEMENT"


class ErrorCode(str, Enum):
    """Typed contract violations (raised, never silently repaired)."""

    NOT_A_DATETIME = "NOT_A_DATETIME"
    NAIVE_DATETIME = "NAIVE_DATETIME"
    TIMESTAMP_INVALID = "TIMESTAMP_INVALID"
    INVALID_HORIZON = "INVALID_HORIZON"
    DUPLICATE_HORIZON = "DUPLICATE_HORIZON"
    INVALID_ROLE = "INVALID_ROLE"
    CALENDAR_REQUIRED = "CALENDAR_REQUIRED"
    CALENDAR_INVALID = "CALENDAR_INVALID"
    CALENDAR_OFFSET_MISMATCH = "CALENDAR_OFFSET_MISMATCH"
    OFFSET_INVALID = "OFFSET_INVALID"
    SPLIT_INVALID = "SPLIT_INVALID"
    INPUT_AVAILABILITY_INVALID = "INPUT_AVAILABILITY_INVALID"
    LABEL_EVIDENCE_INVALID = "LABEL_EVIDENCE_INVALID"
    ORIGIN_GRID_INVALID = "ORIGIN_GRID_INVALID"
    CONFIG_INVALID = "CONFIG_INVALID"
    OPERATIONAL_WINDOW_EMPTY = "OPERATIONAL_WINDOW_EMPTY"
    PREDICTION_AVAILABILITY_INVALID = "PREDICTION_AVAILABILITY_INVALID"


class EligibilityError(ValueError):
    """Contract violation carrying a stable :class:`ErrorCode`."""

    def __init__(self, code: ErrorCode, message: str):
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.detail = message

    def to_json(self) -> Dict[str, str]:
        """Return the typed error as a JSON-ready mapping."""
        return {"code": self.code.value, "message": self.detail}


class OperationalReason(str, Enum):
    """Why ``E`` is false. Listed in precedence order (first = primary)."""

    ORIGIN_BEFORE_OPERATIONAL_START = "ORIGIN_BEFORE_OPERATIONAL_START"
    ORIGIN_AT_OR_AFTER_ENTRY_CUTOFF = "ORIGIN_AT_OR_AFTER_ENTRY_CUTOFF"
    ORIGIN_AT_OR_AFTER_FORCE_FLAT = "ORIGIN_AT_OR_AFTER_FORCE_FLAT"
    TARGET_AT_OR_AFTER_FORCE_FLAT = "TARGET_AT_OR_AFTER_FORCE_FLAT"
    TARGET_AT_OR_AFTER_MARKET_CLOSE = "TARGET_AT_OR_AFTER_MARKET_CLOSE"


class LabelReason(str, Enum):
    """Why ``M`` is false. Listed in precedence order (first = primary)."""

    OPERATIONALLY_INELIGIBLE = "OPERATIONALLY_INELIGIBLE"
    INPUT_UNAVAILABLE = "INPUT_UNAVAILABLE"
    TARGET_NOT_OBSERVED = "TARGET_NOT_OBSERVED"
    TARGET_AVAILABLE_AFTER_LABEL_CUTOFF = "TARGET_AVAILABLE_AFTER_LABEL_CUTOFF"
    TARGET_BEFORE_SPLIT_START = "TARGET_BEFORE_SPLIT_START"
    TARGET_AT_OR_AFTER_SPLIT_END = "TARGET_AT_OR_AFTER_SPLIT_END"
    TARGET_OUTSIDE_DECLARED_SPLITS = "TARGET_OUTSIDE_DECLARED_SPLITS"


class AvailabilityReason(str, Enum):
    """Why a head's prediction is not ``available`` to the strategy adapter."""

    OPERATIONALLY_INELIGIBLE = "OPERATIONALLY_INELIGIBLE"
    PREDICTION_UNAVAILABLE = "PREDICTION_UNAVAILABLE"


class InputReason(str, Enum):
    """Why an input observation may not enter the window of an origin."""

    INPUT_NOT_YET_AVAILABLE = "INPUT_NOT_YET_AVAILABLE"


# --------------------------------------------------------------------------
# Scalar validation


def utc_instant(name: str, value: Any) -> datetime:
    """Return ``value`` as a UTC instant; reject naive or non-datetimes."""
    if not isinstance(value, datetime):
        raise EligibilityError(ErrorCode.NOT_A_DATETIME,
                               f"{name} must be a datetime, not {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise EligibilityError(ErrorCode.NAIVE_DATETIME, f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def parse_instant(name: str, text: Any) -> datetime:
    """Parse an ISO-8601 timestamp WITH offset into a UTC instant."""
    if not isinstance(text, str):
        raise EligibilityError(ErrorCode.TIMESTAMP_INVALID, f"{name} must be an ISO string")
    try:
        value = datetime.fromisoformat(text)
    except ValueError as exc:
        raise EligibilityError(ErrorCode.TIMESTAMP_INVALID, f"{name}: {exc}") from None
    return utc_instant(name, value)


def iso(value: datetime) -> str:
    """Canonical UTC ISO text used in every emitted record."""
    return value.astimezone(UTC).isoformat()


def validate_horizon(horizon_hours: Any) -> int:
    """Return a positive integer horizon in elapsed hours (bool rejected)."""
    try:
        value = _require_plain_int("horizon_hours", horizon_hours)
    except TypeError as exc:
        raise EligibilityError(ErrorCode.INVALID_HORIZON, str(exc)) from None
    if value <= 0:
        raise EligibilityError(ErrorCode.INVALID_HORIZON,
                               f"horizon_hours must be positive, got {value}")
    return value


def _offset_hours(name: str, value: Any) -> int:
    try:
        value = _require_plain_int(name, value)
    except TypeError as exc:
        raise EligibilityError(ErrorCode.OFFSET_INVALID, str(exc)) from None
    if value < 0:
        raise EligibilityError(ErrorCode.OFFSET_INVALID, f"{name} must be >= 0")
    return value


def _role(value: Any) -> Role:
    if not isinstance(value, Role):
        raise EligibilityError(ErrorCode.INVALID_ROLE, f"role must be Role, got {value!r}")
    return value


def canonical_digest(value: Any) -> str:
    """Canonical JSON SHA-256 (same formula as ``tools.i6m_queue.digest``).

    Reimplemented here (five lines) instead of imported, because
    ``tools.i6m_queue`` imports ``sqlite3`` and this component must not
    depend on a database module.
    """
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------
# Calendar, split and evidence objects


@dataclass(frozen=True)
class CalendarRef:
    """Identity of the supplied instrument calendar. Both fields mandatory."""

    calendar_id: str
    version: str

    def __post_init__(self) -> None:
        for name in ("calendar_id", "version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise EligibilityError(ErrorCode.CALENDAR_REQUIRED,
                                       f"calendar {name} is mandatory")


@dataclass(frozen=True)
class OperationalWeek:
    """One operational week; all instants are stored in UTC.

    Build with :meth:`derive`. ``__post_init__`` re-checks every invariant so
    a directly constructed instance cannot bypass the offsets. A week whose
    offsets leave ``A >= B`` is ``OPERATIONAL_WINDOW_EMPTY`` (a calendar/offset
    error, not a horizon incompatibility). A cutoff ``K <= A`` is accepted: it
    declares a week without new entries, and ENTRY rows then carry typed
    exclusion reasons for every origin.
    """

    week_id: str
    calendar: CalendarRef
    market_open: datetime
    operational_start: datetime
    entry_cutoff: datetime
    force_flat: datetime
    market_close: datetime
    entry_delay_after_open_hours: int
    force_flat_before_close_hours: int

    def __post_init__(self) -> None:
        if not isinstance(self.week_id, str) or not self.week_id.strip():
            raise EligibilityError(ErrorCode.CALENDAR_INVALID, "week_id is mandatory")
        if not isinstance(self.calendar, CalendarRef):
            raise EligibilityError(ErrorCode.CALENDAR_REQUIRED, "calendar reference missing")
        for name in ("market_open", "operational_start", "entry_cutoff",
                     "force_flat", "market_close"):
            object.__setattr__(self, name, utc_instant(name, getattr(self, name)))
        delay = _offset_hours("entry_delay_after_open_hours",
                              self.entry_delay_after_open_hours)
        flat = _offset_hours("force_flat_before_close_hours",
                             self.force_flat_before_close_hours)
        if not self.market_open < self.market_close:
            raise EligibilityError(ErrorCode.CALENDAR_INVALID,
                                   f"{self.week_id}: market_open must precede market_close")
        if self.operational_start != self.market_open + delay * HOUR:
            raise EligibilityError(ErrorCode.CALENDAR_OFFSET_MISMATCH,
                                   f"{self.week_id}: A != O + {delay} h")
        if self.force_flat != self.market_close - flat * HOUR:
            raise EligibilityError(ErrorCode.CALENDAR_OFFSET_MISMATCH,
                                   f"{self.week_id}: B != C - {flat} h")
        if not self.operational_start < self.force_flat:
            raise EligibilityError(ErrorCode.OPERATIONAL_WINDOW_EMPTY,
                                   f"{self.week_id}: A must precede B (O + {delay} h >= "
                                   f"C - {flat} h); the week has no operational origin")
        if not self.market_open <= self.entry_cutoff <= self.market_close:
            raise EligibilityError(ErrorCode.CALENDAR_INVALID,
                                   f"{self.week_id}: entry_cutoff outside [O, C]")

    @classmethod
    def derive(cls, *, week_id: str, calendar: CalendarRef, market_open: datetime,
               market_close: datetime, entry_cutoff: datetime,
               entry_delay_after_open_hours: int, force_flat_before_close_hours: int,
               declared_operational_start: Optional[datetime] = None,
               declared_force_flat: Optional[datetime] = None) -> "OperationalWeek":
        """Derive A and B from O, C and the offsets, in elapsed hours.

        ``declared_*`` values, when the calendar also states A or B, must equal
        the derived instants; a mismatch is ``CALENDAR_OFFSET_MISMATCH``.
        """
        o = utc_instant("market_open", market_open)
        c = utc_instant("market_close", market_close)
        delay = _offset_hours("entry_delay_after_open_hours", entry_delay_after_open_hours)
        flat = _offset_hours("force_flat_before_close_hours", force_flat_before_close_hours)
        a, b = o + delay * HOUR, c - flat * HOUR
        for name, declared, derived in (("operational_start", declared_operational_start, a),
                                        ("force_flat", declared_force_flat, b)):
            if declared is not None and utc_instant(name, declared) != derived:
                raise EligibilityError(ErrorCode.CALENDAR_OFFSET_MISMATCH,
                                       f"{week_id}: declared {name} != derived")
        return cls(week_id=week_id, calendar=calendar, market_open=o,
                   operational_start=a, entry_cutoff=entry_cutoff, force_flat=b,
                   market_close=c, entry_delay_after_open_hours=delay,
                   force_flat_before_close_hours=flat)

    def to_json(self) -> Dict[str, Any]:
        """JSON-ready boundaries (UTC ISO)."""
        return {"week_id": self.week_id, "market_open": iso(self.market_open),
                "operational_start": iso(self.operational_start),
                "entry_cutoff": iso(self.entry_cutoff), "force_flat": iso(self.force_flat),
                "market_close": iso(self.market_close)}


@dataclass(frozen=True)
class SplitBounds:
    """Explicit split: ``start <= q < end`` decided by the TARGET instant.

    ``label_available_by`` must be passed explicitly; ``None`` declares that no
    information cutoff applies to this split (e.g. a scored external year).
    Otherwise a label whose ``available_at`` exceeds it is excluded.
    """

    split_id: str
    start: datetime
    end: datetime
    label_available_by: Optional[datetime]
    membership: ClassVar[str] = "TARGET_INSTANT"

    def __post_init__(self) -> None:
        if not isinstance(self.split_id, str) or not self.split_id.strip():
            raise EligibilityError(ErrorCode.SPLIT_INVALID, "split_id is mandatory")
        object.__setattr__(self, "start", utc_instant("split.start", self.start))
        object.__setattr__(self, "end", utc_instant("split.end", self.end))
        if self.label_available_by is not None:
            object.__setattr__(self, "label_available_by",
                               utc_instant("split.label_available_by",
                                           self.label_available_by))
        if not self.start < self.end:
            raise EligibilityError(ErrorCode.SPLIT_INVALID,
                                   f"{self.split_id}: start must precede end")

    def to_json(self) -> Dict[str, Any]:
        """JSON-ready split declaration."""
        return {"split_id": self.split_id, "start": iso(self.start), "end": iso(self.end),
                "label_available_by": None if self.label_available_by is None
                else iso(self.label_available_by), "membership": self.membership}


class ObservedTargets:
    """Exact observed target bars: UTC timestamp -> UTC ``available_at``.

    Built once and reused; a bar can never be available before it occurs.
    """

    __slots__ = ("_bars",)

    def __init__(self, bars: Mapping[datetime, datetime]):
        if not isinstance(bars, Mapping):
            raise EligibilityError(ErrorCode.LABEL_EVIDENCE_INVALID,
                                   "observed targets must be a mapping")
        normalized: Dict[datetime, datetime] = {}
        for stamp, available in bars.items():
            t = utc_instant("observed.timestamp", stamp)
            a = utc_instant("observed.available_at", available)
            if a < t:
                raise EligibilityError(ErrorCode.LABEL_EVIDENCE_INVALID,
                                       f"bar {iso(t)} available before it occurs")
            if t in normalized and normalized[t] != a:
                raise EligibilityError(ErrorCode.LABEL_EVIDENCE_INVALID,
                                       f"conflicting availability for {iso(t)}")
            normalized[t] = a
        self._bars = normalized

    @classmethod
    def coerce(cls, value: Any) -> "ObservedTargets":
        """Return ``value`` if already normalized, else normalize it."""
        return value if isinstance(value, cls) else cls(value)

    def available_at(self, target: datetime) -> Optional[datetime]:
        """Availability of the EXACT bar ``target``; ``None`` if not observed."""
        return self._bars.get(target)

    def __len__(self) -> int:
        return len(self._bars)


@dataclass(frozen=True)
class HeadDecision:
    """Per-head decision. ``label_*`` fields are ``None``/empty when ``M`` was
    not evaluated (no label evidence supplied); that is never a False. The
    same holds for ``available`` when ``prediction_available`` was not given."""

    origin: datetime
    role: Role
    horizon_hours: int
    target_timestamp: datetime
    eligible_until: datetime
    remaining_seconds_to_force_flat: int
    week_id: str
    calendar_id: str
    calendar_version: str
    operational_eligible: bool
    operational_reasons: Tuple[OperationalReason, ...]
    label_evaluated: bool
    label_eligible: Optional[bool]
    label_reasons: Tuple[LabelReason, ...]
    split_id: Optional[str]
    availability_evaluated: bool = False
    prediction_available: Optional[bool] = None
    available: Optional[bool] = None
    availability_reasons: Tuple[AvailabilityReason, ...] = ()

    def to_json(self) -> Dict[str, Any]:
        """JSON-ready record with typed reasons."""
        return {"origin": iso(self.origin), "role": self.role.value,
                "horizon_hours": self.horizon_hours,
                "target_timestamp": iso(self.target_timestamp),
                "eligible_until": iso(self.eligible_until),
                "remaining_seconds_to_force_flat": self.remaining_seconds_to_force_flat,
                "week_id": self.week_id, "calendar_id": self.calendar_id,
                "calendar_version": self.calendar_version,
                "operational_eligible": self.operational_eligible,
                "operational_reasons": [r.value for r in self.operational_reasons],
                "label_evaluated": self.label_evaluated,
                "label_eligible": self.label_eligible,
                "label_reasons": [r.value for r in self.label_reasons],
                "split_id": self.split_id,
                "availability_evaluated": self.availability_evaluated,
                "prediction_available": self.prediction_available,
                "available": self.available,
                "availability_reasons": [r.value for r in self.availability_reasons]}


# --------------------------------------------------------------------------
# Pure eligibility functions


def operational_eligibility(origin: datetime, horizon_hours: int, role: Role,
                            week: OperationalWeek) -> Tuple[bool, Tuple[OperationalReason, ...]]:
    """Return ``E(t, h, role)`` and every applicable typed reason.

    Uses only the supplied calendar week and ``t``; never any bar, price or
    future observation. Reasons are emitted in :class:`OperationalReason`
    order, so ``reasons[0]`` is the primary exclusion.
    """
    t = utc_instant("origin", origin)
    h = validate_horizon(horizon_hours)
    role = _role(role)
    if not isinstance(week, OperationalWeek):
        raise EligibilityError(ErrorCode.CALENDAR_REQUIRED, "an OperationalWeek is mandatory")
    q = t + h * HOUR
    reasons: List[OperationalReason] = []
    if t < week.operational_start:
        reasons.append(OperationalReason.ORIGIN_BEFORE_OPERATIONAL_START)
    if role is Role.ENTRY and t >= week.entry_cutoff:
        reasons.append(OperationalReason.ORIGIN_AT_OR_AFTER_ENTRY_CUTOFF)
    if t >= week.force_flat:
        reasons.append(OperationalReason.ORIGIN_AT_OR_AFTER_FORCE_FLAT)
    if q >= week.force_flat:
        reasons.append(OperationalReason.TARGET_AT_OR_AFTER_FORCE_FLAT)
    if q >= week.market_close:
        reasons.append(OperationalReason.TARGET_AT_OR_AFTER_MARKET_CLOSE)
    return (not reasons, tuple(reasons))


def _prediction_availability(value: Any, heads: Sequence[int]) -> Dict[int, Optional[bool]]:
    """Normalize ``prediction_available``: None, one bool, or {head: bool}.

    A mapping must name exactly the requested heads with plain-int keys and
    bool values; anything else is ``PREDICTION_AVAILABILITY_INVALID``.
    """
    if value is None:
        return {h: None for h in heads}
    if isinstance(value, bool):
        return {h: value for h in heads}
    if not isinstance(value, Mapping):
        raise EligibilityError(ErrorCode.PREDICTION_AVAILABILITY_INVALID,
                               "prediction_available must be None, a bool or {head: bool}")
    keys = list(value)
    if any(type(k) is not int for k in keys) or set(keys) != set(heads):
        raise EligibilityError(ErrorCode.PREDICTION_AVAILABILITY_INVALID,
                               f"prediction_available keys {keys!r} must equal heads {list(heads)}")
    if any(not isinstance(v, bool) for v in value.values()):
        raise EligibilityError(ErrorCode.PREDICTION_AVAILABILITY_INVALID,
                               "prediction_available values must be bool")
    return {h: value[h] for h in heads}


def _decision(t: datetime, h: int, role: Role, week: OperationalWeek,
              e: Tuple[bool, Tuple[OperationalReason, ...]],
              label: Optional[Tuple[bool, Tuple[LabelReason, ...], Optional[str]]],
              prediction_available: Optional[bool] = None) -> HeadDecision:
    if prediction_available is None:
        available, a_reasons = None, ()
    else:
        found: List[AvailabilityReason] = []
        if not e[0]:
            found.append(AvailabilityReason.OPERATIONALLY_INELIGIBLE)
        if not prediction_available:
            found.append(AvailabilityReason.PREDICTION_UNAVAILABLE)
        available, a_reasons = not found, tuple(found)
    return HeadDecision(
        origin=t, role=role, horizon_hours=h, target_timestamp=t + h * HOUR,
        eligible_until=week.force_flat - h * HOUR,
        remaining_seconds_to_force_flat=(week.force_flat - t) // SECOND,
        week_id=week.week_id, calendar_id=week.calendar.calendar_id,
        calendar_version=week.calendar.version,
        operational_eligible=e[0], operational_reasons=e[1],
        label_evaluated=label is not None,
        label_eligible=None if label is None else label[0],
        label_reasons=() if label is None else label[1],
        split_id=None if label is None else label[2],
        availability_evaluated=prediction_available is not None,
        prediction_available=prediction_available, available=available,
        availability_reasons=a_reasons)


def _label_reasons(t: datetime, h: int, e_ok: bool, input_available: Any,
                   observed: ObservedTargets, split: Any) -> Tuple[bool, Tuple[LabelReason, ...]]:
    if not isinstance(input_available, bool):
        raise EligibilityError(ErrorCode.INPUT_AVAILABILITY_INVALID,
                               "input_available must be a bool")
    if not isinstance(split, SplitBounds):
        raise EligibilityError(ErrorCode.SPLIT_INVALID, "explicit SplitBounds is mandatory")
    q = t + h * HOUR
    reasons: List[LabelReason] = []
    if not e_ok:
        reasons.append(LabelReason.OPERATIONALLY_INELIGIBLE)
    if not input_available:
        reasons.append(LabelReason.INPUT_UNAVAILABLE)
    available = observed.available_at(q)
    if available is None:
        reasons.append(LabelReason.TARGET_NOT_OBSERVED)
    elif split.label_available_by is not None and available > split.label_available_by:
        reasons.append(LabelReason.TARGET_AVAILABLE_AFTER_LABEL_CUTOFF)
    if q < split.start:
        reasons.append(LabelReason.TARGET_BEFORE_SPLIT_START)
    if q >= split.end:
        reasons.append(LabelReason.TARGET_AT_OR_AFTER_SPLIT_END)
    return (not reasons, tuple(reasons))


def historical_label_eligibility(origin: datetime, horizon_hours: int, role: Role,
                                 week: OperationalWeek, *, input_available: bool,
                                 observed_targets: Any, split: SplitBounds,
                                 prediction_available: Optional[bool] = None) -> HeadDecision:
    """Return one :class:`HeadDecision` with both ``E`` and ``M``.

    ``E`` is computed exactly as :func:`operational_eligibility`; label
    evidence can only turn ``M`` false and never alters ``E``.
    ``prediction_available`` (optional bool) only fills ``available``.
    """
    t = utc_instant("origin", origin)
    h = validate_horizon(horizon_hours)
    e = operational_eligibility(t, h, role, week)
    observed = ObservedTargets.coerce(observed_targets)
    ok, reasons = _label_reasons(t, h, e[0], input_available, observed, split)
    pa = _prediction_availability(prediction_available, (h,))[h]
    return _decision(t, h, role, week, e, (ok, reasons, split.split_id), pa)


def evaluate_heads(origin: datetime, horizons_hours: Sequence[int], role: Role,
                   week: OperationalWeek, *, input_available: Optional[bool] = None,
                   observed_targets: Any = None,
                   split: Optional[SplitBounds] = None,
                   prediction_available: Any = None) -> Tuple[HeadDecision, ...]:
    """Evaluate every head of one origin; one decision per requested head.

    Label arguments are all-or-nothing: with none, only ``E`` is evaluated;
    with all three, ``M`` too. Ineligible heads are reported, never removed.
    ``prediction_available`` is independent of both: ``None`` (not supplied,
    ``available`` stays ``None``), one bool for every head, or a mapping
    ``{head: bool}`` naming exactly the requested heads.
    """
    t = utc_instant("origin", origin)
    hs = [validate_horizon(h) for h in horizons_hours]
    if not hs:
        raise EligibilityError(ErrorCode.INVALID_HORIZON, "at least one head is required")
    if len(set(hs)) != len(hs):
        raise EligibilityError(ErrorCode.DUPLICATE_HORIZON, f"duplicate heads in {hs}")
    pa = _prediction_availability(prediction_available, hs)
    supplied = [x is not None for x in (input_available, observed_targets, split)]
    if any(supplied) and not all(supplied):
        raise EligibilityError(ErrorCode.LABEL_EVIDENCE_INVALID,
                               "input_available, observed_targets and split go together")
    if not any(supplied):
        return tuple(_decision(t, h, role, week, operational_eligibility(t, h, role, week),
                               None, pa[h])
                     for h in hs)
    observed = ObservedTargets.coerce(observed_targets)
    return tuple(historical_label_eligibility(t, h, role, week, input_available=input_available,
                                              observed_targets=observed, split=split,
                                              prediction_available=pa[h])
                 for h in hs)


def input_admissibility(origin: datetime,
                        observation_available_at: datetime) -> Tuple[bool, Optional[InputReason]]:
    """May an observation enter the input window of ``origin``?

    Only knowledge at ``t`` matters: data from non-trading hours (weekend,
    [O, A), [B, C)) is admissible once available; the operating band is never
    consulted. Future availability is rejected.
    """
    t = utc_instant("origin", origin)
    a = utc_instant("observation_available_at", observation_available_at)
    return (True, None) if a <= t else (False, InputReason.INPUT_NOT_YET_AVAILABLE)


# --------------------------------------------------------------------------
# Configuration loading


_CALENDAR_KEYS = {"calendar_id", "version", "weeks", "description"}
_WEEK_KEYS = {"week_id", "market_open", "market_close", "entry_cutoff",
              "operational_start", "force_flat"}
_CONFIG_KEYS = {"schema_version", "description", "market_calendar",
                "entry_delay_after_open_hours", "force_flat_before_close_hours",
                "roles", "horizons_hours", "origin_grid", "input_unavailable_origins",
                "splits", "label_evidence"}


def load_calendar(config: Mapping[str, Any]) -> Tuple[CalendarRef, Tuple[OperationalWeek, ...]]:
    """Build the supplied calendar. Missing calendar => ``CALENDAR_REQUIRED``.

    There is no default calendar, zone or broker. Weeks must be listed in
    order, with unique ids, and must not overlap (``C_k <= O_{k+1}``).
    """
    if not isinstance(config, Mapping):
        raise EligibilityError(ErrorCode.CONFIG_INVALID, "config must be a JSON object")
    cal = config.get("market_calendar")
    if not isinstance(cal, Mapping):
        raise EligibilityError(ErrorCode.CALENDAR_REQUIRED,
                               "market_calendar is mandatory; no fallback calendar exists")
    ref = CalendarRef(cal.get("calendar_id"), cal.get("version"))
    unknown = set(cal) - _CALENDAR_KEYS
    if unknown:
        raise EligibilityError(ErrorCode.CALENDAR_INVALID, f"unknown calendar keys {sorted(unknown)}")
    if "entry_delay_after_open_hours" not in config or \
            "force_flat_before_close_hours" not in config:
        raise EligibilityError(ErrorCode.OFFSET_INVALID, "both offsets must be explicit")
    delay = _offset_hours("entry_delay_after_open_hours", config["entry_delay_after_open_hours"])
    flat = _offset_hours("force_flat_before_close_hours", config["force_flat_before_close_hours"])
    raw_weeks = cal.get("weeks")
    if not isinstance(raw_weeks, list) or not raw_weeks:
        raise EligibilityError(ErrorCode.CALENDAR_INVALID, "calendar weeks must be a non-empty list")
    weeks: List[OperationalWeek] = []
    for raw in raw_weeks:
        if not isinstance(raw, Mapping):
            raise EligibilityError(ErrorCode.CALENDAR_INVALID, "each week must be an object")
        missing = {"week_id", "market_open", "market_close", "entry_cutoff"} - set(raw)
        unknown = set(raw) - _WEEK_KEYS
        if missing or unknown:
            raise EligibilityError(ErrorCode.CALENDAR_INVALID,
                                   f"week keys missing {sorted(missing)} unknown {sorted(unknown)}")
        wid = raw["week_id"]
        weeks.append(OperationalWeek.derive(
            week_id=wid, calendar=ref,
            market_open=parse_instant(f"{wid}.market_open", raw["market_open"]),
            market_close=parse_instant(f"{wid}.market_close", raw["market_close"]),
            entry_cutoff=parse_instant(f"{wid}.entry_cutoff", raw["entry_cutoff"]),
            entry_delay_after_open_hours=delay, force_flat_before_close_hours=flat,
            declared_operational_start=None if raw.get("operational_start") is None
            else parse_instant(f"{wid}.operational_start", raw["operational_start"]),
            declared_force_flat=None if raw.get("force_flat") is None
            else parse_instant(f"{wid}.force_flat", raw["force_flat"])))
    ids = [w.week_id for w in weeks]
    if len(set(ids)) != len(ids):
        raise EligibilityError(ErrorCode.CALENDAR_INVALID, "duplicate week_id")
    for prev, nxt in zip(weeks, weeks[1:]):
        if not prev.market_close <= nxt.market_open:
            raise EligibilityError(ErrorCode.CALENDAR_INVALID,
                                   f"{prev.week_id} overlaps or follows {nxt.week_id}")
    return ref, tuple(weeks)


def _split(raw: Any) -> SplitBounds:
    if not isinstance(raw, Mapping) or set(raw) != {"split_id", "start", "end",
                                                    "label_available_by"}:
        raise EligibilityError(ErrorCode.SPLIT_INVALID,
                               "split needs exactly split_id, start, end, label_available_by")
    cut = raw["label_available_by"]
    return SplitBounds(raw["split_id"], parse_instant("split.start", raw["start"]),
                       parse_instant("split.end", raw["end"]),
                       label_available_by=None if cut is None
                       else parse_instant("split.label_available_by", cut))


@dataclass(frozen=True)
class SupportPlan:
    """Validated support-table request (see :func:`prepare_support_plan`)."""

    calendar: CalendarRef
    weeks: Tuple[OperationalWeek, ...]
    roles: Tuple[Role, ...]
    horizons: Tuple[int, ...]
    origin_grid: Mapping[str, Any]
    origins_by_week: Mapping[str, Tuple[datetime, ...]]
    origins_outside_calendar_weeks: int
    input_unavailable: frozenset
    splits: Tuple[SplitBounds, ...]
    observed: Optional[ObservedTargets]

    @property
    def units_total(self) -> int:
        """Rows to emit: weeks x roles x heads."""
        return len(self.weeks) * len(self.roles) * len(self.horizons)


def _origin_grid(raw: Any, weeks: Sequence[OperationalWeek]):
    if not isinstance(raw, Mapping) or raw.get("kind") not in ("UTC_WHOLE_HOURS", "EXPLICIT"):
        raise EligibilityError(ErrorCode.ORIGIN_GRID_INVALID,
                               "origin_grid.kind must be UTC_WHOLE_HOURS or EXPLICIT")
    by_week: Dict[str, List[datetime]] = {w.week_id: [] for w in weeks}
    if raw["kind"] == "UTC_WHOLE_HOURS":
        if set(raw) != {"kind"}:
            raise EligibilityError(ErrorCode.ORIGIN_GRID_INVALID, "UTC_WHOLE_HOURS takes no options")
        for w in weeks:
            t = w.market_open.replace(minute=0, second=0, microsecond=0)
            if t < w.market_open:
                t += HOUR
            while t < w.market_close:
                by_week[w.week_id].append(t)
                t += HOUR
        return dict(raw), {k: tuple(v) for k, v in by_week.items()}, 0
    if set(raw) != {"kind", "timestamps"} or not isinstance(raw["timestamps"], list):
        raise EligibilityError(ErrorCode.ORIGIN_GRID_INVALID, "EXPLICIT needs a timestamps list")
    stamps = [parse_instant("origin_grid.timestamp", s) for s in raw["timestamps"]]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise EligibilityError(ErrorCode.ORIGIN_GRID_INVALID,
                               "explicit origins must be strictly increasing")
    opens = [w.market_open for w in weeks]
    outside = 0
    for t in stamps:
        i = bisect.bisect_right(opens, t) - 1
        if i >= 0 and t < weeks[i].market_close:
            by_week[weeks[i].week_id].append(t)
        else:
            outside += 1
    summary = {"kind": "EXPLICIT", "timestamps_count": len(stamps),
               "timestamps_sha256": canonical_digest([iso(t) for t in stamps])}
    return summary, {k: tuple(v) for k, v in by_week.items()}, outside


def prepare_support_plan(config: Mapping[str, Any]) -> SupportPlan:
    """Validate a support-table config (schema ``i6n.eligibility.config.v1``).

    ``input_unavailable_origins`` is applied, never merely counted: every
    entry must be unique and equal one enumerated in-calendar origin, and a
    non-empty list requires ``label_evidence`` (input availability acts only
    on ``M``). Otherwise ``INPUT_AVAILABILITY_INVALID``.
    """
    ref, weeks = load_calendar(config)
    unknown = set(config) - _CONFIG_KEYS
    missing = _CONFIG_KEYS - {"description"} - set(config)
    if unknown or missing:
        raise EligibilityError(ErrorCode.CONFIG_INVALID,
                               f"config keys missing {sorted(missing)} unknown {sorted(unknown)}")
    if config["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise EligibilityError(ErrorCode.CONFIG_INVALID,
                               f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    raw_roles = config["roles"]
    if not isinstance(raw_roles, list) or not raw_roles or len(set(map(str, raw_roles))) != len(raw_roles):
        raise EligibilityError(ErrorCode.INVALID_ROLE, "roles must be a non-empty unique list")
    try:
        roles = tuple(Role(r) for r in raw_roles)
    except ValueError as exc:
        raise EligibilityError(ErrorCode.INVALID_ROLE, str(exc)) from None
    raw_h = config["horizons_hours"]
    if not isinstance(raw_h, list) or not raw_h:
        raise EligibilityError(ErrorCode.INVALID_HORIZON, "horizons_hours must be a non-empty list")
    horizons = tuple(validate_horizon(h) for h in raw_h)
    if len(set(horizons)) != len(horizons):
        raise EligibilityError(ErrorCode.DUPLICATE_HORIZON, "duplicate horizons_hours")
    grid, by_week, outside = _origin_grid(config["origin_grid"], weeks)
    if not isinstance(config["input_unavailable_origins"], list):
        raise EligibilityError(ErrorCode.INPUT_AVAILABILITY_INVALID,
                               "input_unavailable_origins must be an explicit list")
    declared = [parse_instant("input_unavailable_origin", s)
                for s in config["input_unavailable_origins"]]
    unavailable = frozenset(declared)
    if len(unavailable) != len(declared):
        raise EligibilityError(ErrorCode.INPUT_AVAILABILITY_INVALID,
                               "duplicate input_unavailable_origins")
    enumerated = {t for ts in by_week.values() for t in ts}
    unmatched = sorted(unavailable - enumerated)
    if unmatched:
        raise EligibilityError(
            ErrorCode.INPUT_AVAILABILITY_INVALID,
            f"{len(unmatched)} input_unavailable_origins match no enumerated in-calendar "
            f"origin and would have no effect (first: {iso(unmatched[0])})")
    if unavailable and config["label_evidence"] is None:
        raise EligibilityError(ErrorCode.INPUT_AVAILABILITY_INVALID,
                               "input_unavailable_origins only act on M; without "
                               "label_evidence they would have no effect")
    if not isinstance(config["splits"], list):
        raise EligibilityError(ErrorCode.SPLIT_INVALID, "splits must be a list")
    splits = tuple(sorted((_split(s) for s in config["splits"]), key=lambda s: s.start))
    if len({s.split_id for s in splits}) != len(splits):
        raise EligibilityError(ErrorCode.SPLIT_INVALID, "duplicate split_id")
    for a, b in zip(splits, splits[1:]):
        if b.start < a.end:
            raise EligibilityError(ErrorCode.SPLIT_INVALID,
                                   f"splits {a.split_id} and {b.split_id} overlap")
    evidence = config["label_evidence"]
    observed = None
    if evidence is not None:
        if not isinstance(evidence, Mapping) or set(evidence) != {"observed_targets"} \
                or not isinstance(evidence["observed_targets"], list):
            raise EligibilityError(ErrorCode.LABEL_EVIDENCE_INVALID,
                                   "label_evidence must be null or {observed_targets: [...]}")
        bars: Dict[datetime, datetime] = {}
        for item in evidence["observed_targets"]:
            if not isinstance(item, Mapping) or set(item) != {"timestamp", "available_at"}:
                raise EligibilityError(ErrorCode.LABEL_EVIDENCE_INVALID,
                                       "observed target needs timestamp and available_at")
            t = parse_instant("observed.timestamp", item["timestamp"])
            if t in bars:
                raise EligibilityError(ErrorCode.LABEL_EVIDENCE_INVALID,
                                       f"duplicate observed bar {iso(t)}")
            bars[t] = parse_instant("observed.available_at", item["available_at"])
        observed = ObservedTargets(bars)
        if not splits:
            raise EligibilityError(ErrorCode.SPLIT_INVALID,
                                   "label evidence requires at least one explicit split")
    return SupportPlan(calendar=ref, weeks=weeks, roles=roles, horizons=horizons,
                       origin_grid=grid, origins_by_week=by_week,
                       origins_outside_calendar_weeks=outside, input_unavailable=unavailable,
                       splits=splits, observed=observed)


# --------------------------------------------------------------------------
# Support table


def _split_for(splits: Sequence[SplitBounds], q: datetime) -> Optional[SplitBounds]:
    for s in splits:
        if s.start <= q < s.end:
            return s
    return None


def iter_support_rows(plan: SupportPlan) -> Iterator[Dict[str, Any]]:
    """Yield one row per (week, role, head) in calendar/config order.

    Counts come from enumerating ``plan.origins_by_week``; a head with no
    eligible origin is reported as ``NO_ELIGIBLE_ROWS``, never as a score.
    """
    for week in plan.weeks:
        origins = plan.origins_by_week[week.week_id]
        for role in plan.roles:
            for h in plan.horizons:
                eligible: List[datetime] = []
                exclusions: Counter = Counter()
                for t in origins:
                    ok, reasons = operational_eligibility(t, h, role, week)
                    if ok:
                        eligible.append(t)
                    else:
                        exclusions[reasons[0].value] += 1
                row: Dict[str, Any] = {
                    "week_id": week.week_id, "role": role.value, "horizon_hours": h,
                    "calendar_id": plan.calendar.calendar_id,
                    "calendar_version": plan.calendar.version,
                    **{k: v for k, v in week.to_json().items() if k != "week_id"},
                    "origins_enumerated": len(origins),
                    "operational_eligible": len(eligible),
                    "operational_status": "SUPPORTED" if eligible else "NO_ELIGIBLE_ROWS",
                    "operational_exclusions": dict(sorted(exclusions.items())),
                    "first_eligible_origin": iso(eligible[0]) if eligible else None,
                    "last_eligible_origin": iso(eligible[-1]) if eligible else None,
                    "eligible_origins_sha256": canonical_digest([iso(t) for t in eligible]),
                }
                if plan.observed is None:
                    row["label_status"] = "LABEL_EVIDENCE_NOT_SUPPLIED"
                    row["label"] = None
                else:
                    by_split = {s.split_id: 0 for s in plan.splits}
                    label_excl: Counter = Counter()
                    label_origins: List[str] = []
                    for t in eligible:
                        q = t + h * HOUR
                        split = _split_for(plan.splits, q)
                        if split is None:
                            label_excl[LabelReason.TARGET_OUTSIDE_DECLARED_SPLITS.value] += 1
                            continue
                        ok, reasons = _label_reasons(t, h, True, t not in plan.input_unavailable,
                                                     plan.observed, split)
                        if ok:
                            by_split[split.split_id] += 1
                            label_origins.append(iso(t))
                        else:
                            label_excl[reasons[0].value] += 1
                    row["label_status"] = "EVALUATED"
                    row["label"] = {
                        "label_eligible": len(label_origins), "by_split": by_split,
                        "exclusions_among_operational": dict(sorted(label_excl.items())),
                        "label_origins_sha256": canonical_digest(label_origins)}
                yield row


def assemble_support_table(plan: SupportPlan, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Wrap emitted rows with identity, provenance and scope disclaimers."""
    if len(rows) != plan.units_total:
        raise EligibilityError(ErrorCode.CONFIG_INVALID, "row count differs from plan")
    return {
        "schema_version": TABLE_SCHEMA_VERSION,
        "eligibility_version": ELIGIBILITY_VERSION,
        "calendar": {"calendar_id": plan.calendar.calendar_id, "version": plan.calendar.version},
        "entry_delay_after_open_hours": plan.weeks[0].entry_delay_after_open_hours,
        "force_flat_before_close_hours": plan.weeks[0].force_flat_before_close_hours,
        "roles": [r.value for r in plan.roles],
        "horizons_hours": list(plan.horizons),
        "origin_grid": dict(plan.origin_grid),
        "origins_outside_calendar_weeks": plan.origins_outside_calendar_weeks,
        "input_unavailable_origins_applied": len(plan.input_unavailable),
        "splits": [s.to_json() for s in plan.splits],
        "label_evidence_supplied": plan.observed is not None,
        "units_total": plan.units_total,
        "rows": list(rows),
        "support_note": ("Counts enumerate the declared origin timestamps per week/role/head "
                         "under half-open [A,B) and q<B; they are not the closed form (B-A)-h."),
        "contract_sources": [dict(s) for s in CONTRACT_SOURCES],
        "candidate_implementation": True,
        "deployed": False,
        "scientific_measurement": False,
        "keras_masking_validated": False,
    }


def build_support_table(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate ``config`` and return the complete support table."""
    plan = prepare_support_plan(config)
    return assemble_support_table(plan, list(iter_support_rows(plan)))
