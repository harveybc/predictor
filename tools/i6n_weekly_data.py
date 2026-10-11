"""Weekly, per-head data partitions governed by the audited C1 calendar.

This module uses no prices, fits no scalers and reads no files. The caller must
supply a versioned calendar and source-availability evidence. Its digest binds
support, not data values or the authenticity of the supplied source clock.
Source coverage is a caller declaration, not raw-byte verification. Complete
coverage certifies only the supplied calendar span and hourly origin grid.
Outer evaluation is development-only; TEST starts at the explicit sealed
boundary. Off-operation observations remain available as input context but
cannot become training labels merely because a quote exists.
"""

from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Mapping, Sequence

from tools.i6n_contracts.eligibility import (
    ObservedTargets, OperationalWeek, Role, SplitBounds, evaluate_heads, iso,
    utc_instant, validate_horizon,
)


class WeeklyDataError(ValueError):
    """Rejected support with a stable machine-readable ``code``."""

    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class Partition:
    """Ordered origin indices and independent target masks, including exclusions."""

    indices: tuple[int, ...]
    origins: tuple[datetime, ...]
    target_instants: tuple[tuple[datetime, ...], ...]
    mask: tuple[tuple[bool, ...], ...]
    exclusions: Mapping[str, int]

    @property
    def n_eligible(self) -> int:
        """Number of eligible origin-head pairs, not complete multivariate rows."""
        return sum(sum(row) for row in self.mask)


@dataclass(frozen=True)
class WeeklyPartitions:
    """A single weekend refit and the following development week."""

    fit_cutoff: datetime
    history_start: datetime
    inner_start: datetime
    horizons: tuple[int, ...]
    train: Partition
    inner: Partition
    evaluation: Partition
    support_sha256: str
    coverage_mode: str
    coverage_complete: bool


def _positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _years_before(instant, years):
    """Subtract calendar years; Feb 29 maps to Feb 28 in a non-leap year."""
    try:
        return instant.replace(year=instant.year - years)
    except ValueError:
        if instant.month == 2 and instant.day == 29:
            return instant.replace(year=instant.year - years, day=28)
        raise


def _complete_coverage(origins, weeks, evaluation_index, start, cutoff):
    """Check the declared calendar and exact hourly grid, not source bytes."""
    if weeks[0].market_open > start:
        raise WeeklyDataError("CALENDAR_COVERAGE_MISSING", "first week opens after history_start")
    for previous, following in zip(weeks, weeks[1:]):
        if following.market_open - previous.market_close > timedelta(days=7):
            raise WeeklyDataError("CALENDAR_GAP_TOO_LARGE", f"after {previous.week_id}")
    # The final historical endpoint is the immediately preceding week's close,
    # from which the caller derived cutoff; closure hours need no decision row.
    present = set(origins)
    hour = timedelta(hours=1)
    for index, week in enumerate(weeks):
        lower, upper = ((week.market_open, week.market_close) if index == evaluation_index
                        else (max(start, week.market_open), min(cutoff, week.market_close)))
        if lower >= upper:
            continue
        for t in origins[bisect_left(origins, lower):bisect_left(origins, upper)]:
            if (t - week.market_open) % hour:
                raise WeeklyDataError("ORIGIN_GRID_INVALID", iso(t))
        t = week.market_open + ((lower - week.market_open) // hour) * hour
        if t < lower:
            t += hour
        while t < upper:
            if t not in present:
                raise WeeklyDataError("INCOMPLETE_HOURLY_ORIGINS", iso(t))
            t += hour


def build_weekly_partitions(
    *, origins: Sequence[datetime], weeks: Sequence[OperationalWeek],
    observed_targets: Mapping[datetime, datetime], input_available_at: Mapping[datetime, datetime],
    horizons: Sequence[int], evaluation_week_id: str, evaluation_split: SplitBounds,
    source_coverage_start: datetime, history_years: int = 4, inner_days: int = 28,
    history_span_hours: int = 24, role: Role = Role.MANAGEMENT,
    test_start: datetime = datetime(2025, 1, 1, tzinfo=timezone.utc),
    coverage_mode: str = "complete",
) -> WeeklyPartitions:
    """Partition one refit without using prediction success to choose rows.

    Parameters
    ----------
    origins, observed_targets, input_available_at
        UTC-resolvable decision instants, exact target availability and the
        availability time of each complete input window. Missing inputs exclude
        the origin; missing targets exclude only the corresponding head.
    weeks
        Nonoverlapping weeks from one explicitly versioned source. Missing
        calendar coverage for a requested origin is an error, not a holiday.
        Origins within a declared between-week closure of at most seven days
        retain all-false masks and ORIGIN_IN_MARKET_CLOSURE exclusions per head.
    evaluation_week_id, evaluation_split
        Week to forecast and outer development interval. Every supplied origin
        in the evaluation week is retained, even before/after the split bounds;
        only TARGET membership determines split eligibility, as in C1.
        TEST labels are never admitted by this adapter.
    source_coverage_start
        Caller declaration of accessible input history, NOT raw-byte
        verification. Must precede history_start by the full window context.
        Input windows, inclusion of current t, source values and masks remain
        the caller/window adapter's responsibility; no window is built here.
    coverage_mode
        Default 'complete': first supplied week opens at/before history_start,
        no between-week gap exceeds seven days, and every required hourly
        origin is present in weeks intersecting [history_start, cutoff) plus
        the entire evaluation week. Grids are anchored at each market_open;
        off-grid in-market origins are rejected. The preceding week closes at
        cutoff by construction. These checks alone permit coverage_complete=True.
        Explicit 'synthetic_sparse' skips those completeness checks, ALWAYS
        returns coverage_complete=False, and never treats a >7-day hole as a
        known closure. This escape is for synthetic fixtures, not fitted runs.

    Notes
    -----
    Refit cutoff is the preceding supplied market close, not Monday midnight.
    The last ``inner_days`` of the historical interval form inner validation.
    TRAIN targets are purged at that boundary and TRAIN labels must be known
    there. Inner labels must be known by the weekend refit cutoff. A role is
    chosen once to avoid weighting the same head twice through ENTRY/MANAGEMENT.
    All four years are split into TRAIN and inner, not four years plus inner.
    Unlike outer evaluation, TRAIN/inner also partition by ORIGIN as an explicit
    trainer policy: a pre-inner origin with a target in inner is purged, never
    shared across the early-stopping splits. Every partition needs at least one
    eligible pair; otherwise NO_ELIGIBLE_TRAIN/INNER/EVAL is raised in either mode.
    """
    _positive_int("history_years", history_years)
    _positive_int("inner_days", inner_days)
    _positive_int("history_span_hours", history_span_hours)
    if not isinstance(coverage_mode, str) or coverage_mode not in ("complete", "synthetic_sparse"):
        raise ValueError("coverage_mode must be complete or synthetic_sparse")
    if not isinstance(role, Role):
        raise ValueError("explicit Role required")
    ts = tuple(utc_instant("origin", t) for t in origins)
    if any(a >= b for a, b in zip(ts, ts[1:])):
        raise ValueError("origins must be unique and increasing")
    hs = tuple(validate_horizon(h) for h in horizons)
    if not hs or len(set(hs)) != len(hs):
        raise ValueError("horizons must be nonempty and unique")
    ws = tuple(weeks)
    if not ws or any(not isinstance(w, OperationalWeek) for w in ws):
        raise ValueError("explicit operational weeks required")
    if (len({w.week_id for w in ws}) != len(ws)
            or any(w.calendar != ws[0].calendar for w in ws)
            or any(a.market_close > b.market_open for a, b in zip(ws, ws[1:]))):
        raise ValueError("calendar must be ordered, nonoverlapping and single identity")
    candidates = [i for i, w in enumerate(ws) if w.week_id == evaluation_week_id]
    if len(candidates) != 1 or candidates[0] == 0:
        raise ValueError("evaluation week and a preceding market close required")
    index = candidates[0]
    evaluation_week = ws[index]
    cutoff = ws[index - 1].market_close
    if cutoff >= evaluation_week.market_open:
        raise ValueError("refit needs a market-closed interval before evaluation")
    # A missing recent week must not silently shorten the information history.
    if evaluation_week.market_open - cutoff > timedelta(days=7):
        raise WeeklyDataError("CALENDAR_GAP_TOO_LARGE", "preceding market close is stale")
    start = _years_before(cutoff, history_years)
    inner_start = cutoff - timedelta(days=inner_days)
    if inner_start <= start:
        raise ValueError("inner interval consumes the entire training history")
    if coverage_mode == "complete":
        _complete_coverage(ts, ws, index, start, cutoff)
    coverage = utc_instant("source_coverage_start", source_coverage_start)
    if coverage > start - timedelta(hours=history_span_hours):
        raise ValueError("source does not cover full training history and context")
    boundary = utc_instant("test_start", test_start)
    if not isinstance(evaluation_split, SplitBounds):
        raise ValueError("explicit evaluation split required")
    if (evaluation_split.end > boundary or evaluation_split.start >= boundary
            or evaluation_split.split_id.upper() == "TEST"):
        raise ValueError("development split must not include TEST or exclude the week")
    if not (evaluation_week.market_open < evaluation_split.end
            and evaluation_week.market_close > evaluation_split.start):
        raise ValueError("evaluation week has no overlap with development split")
    observed = ObservedTargets(observed_targets)
    available = {utc_instant("input.origin", t): utc_instant("input.available_at", a)
                 for t, a in input_available_at.items()}
    opens = [w.market_open for w in ws]

    def make(bounds, restrict_week=None):
        indices, selected, targets, masks = [], [], [], []
        excluded = Counter()
        for i, t in enumerate(ts):
            if restrict_week is not None:
                if not restrict_week.market_open <= t < restrict_week.market_close:
                    continue
            elif not bounds.start <= t < bounds.end:
                continue
            wi = bisect_right(opens, t) - 1
            if wi < 0 or t >= ws[wi].market_close:
                known_closure = (wi >= 0 and wi + 1 < len(ws)
                                 and ws[wi].market_close <= t < ws[wi + 1].market_open
                                 and ws[wi + 1].market_open - ws[wi].market_close <= timedelta(days=7))
                if not known_closure:
                    raise WeeklyDataError("CALENDAR_COVERAGE_MISSING", iso(t))
                indices.append(i)
                selected.append(t)
                targets.append(tuple(t + timedelta(hours=h) for h in hs))
                masks.append(tuple(False for _ in hs))
                excluded["ORIGIN_IN_MARKET_CLOSURE"] += len(hs)
                continue
            input_ok = t in available and available[t] <= t
            decisions = evaluate_heads(t, hs, role, ws[wi], input_available=input_ok,
                                       observed_targets=observed, split=bounds)
            indices.append(i)
            selected.append(t)
            targets.append(tuple(d.target_timestamp for d in decisions))
            masks.append(tuple(d.label_eligible for d in decisions))
            for d in decisions:
                excluded.update(r.value for r in d.operational_reasons)
                excluded.update(r.value for r in d.label_reasons)
        return Partition(tuple(indices), tuple(selected), tuple(targets), tuple(masks), dict(excluded))

    train = make(SplitBounds("TRAIN", start, inner_start, inner_start))
    inner = make(SplitBounds("INNER", inner_start, cutoff, cutoff))
    evaluation = make(evaluation_split, evaluation_week)
    for name, part in (("TRAIN", train), ("INNER", inner), ("EVAL", evaluation)):
        if not part.n_eligible:
            raise WeeklyDataError(f"NO_ELIGIBLE_{name}")
    coverage_complete = coverage_mode == "complete"
    body = {
        "schema": "i6n.weekly_support.v2", "calendar": vars(ws[0].calendar),
        "coverage_mode": coverage_mode, "coverage_complete": coverage_complete,
        "training_origin_policy": "disjoint_origin_intervals_with_target_purge",
        "weeks": [w.to_json() for w in ws], "fit_cutoff": iso(cutoff),
        "history_start": iso(start), "inner_start": iso(inner_start),
        "history_span_hours": history_span_hours, "source_coverage_start": iso(coverage),
        "role": role.value, "horizons": hs, "evaluation_split": evaluation_split.to_json(),
        "test_start": iso(boundary),
        "partitions": {name: {"origins": [iso(t) for t in part.origins],
                              "mask": part.mask, "exclusions": part.exclusions}
                       for name, part in (("train", train), ("inner", inner), ("evaluation", evaluation))},
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return WeeklyPartitions(cutoff, start, inner_start, hs, train, inner, evaluation,
                            digest, coverage_mode, coverage_complete)
