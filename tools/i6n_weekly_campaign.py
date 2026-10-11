"""I6-N lane C: weekly walk-forward campaign builder (no fits, no prices, no TEST).

Builds, BEFORE any fit, the sealed list of refit units of the business weekly
protocol and the four 2023 inner folds of the search, from a configured
NOMINAL market calendar (identified by id/version, never certified by decree)
and from the TIMESTAMPS of a retained input. Price and feature values are never
read. Targets at or after the TEST boundary (2025-01-01 UTC) are never looked up.

Units
-----
``WEEKLY_REFIT``  one per calendar week whose operational band [A, B) meets the
    development split (every eligible 2024 week; none skipped, none merged).
    Refit cutoff = market close C of the preceding calendar week; the fit
    happens in [C_prev, O) and the model is released at O. TRAIN covers the
    ``history_years`` calendar years before the cutoff, minus the last
    ``inner_days`` days, which form the purged inner split. VALIDATION is the
    week itself with target membership in the development split.
``INNER_FOLD``    one per quarter of ``fold_year``: cutoff = close of the last
    calendar week lying completely inside the quarter (same boundaries in UTC
    and in the calendar zone, else refused). Same TRAIN/inner rule; the outer
    week is NOT scored. The following week is named only because the existing
    historical cell needs a structural evaluation week to locate the cutoff.

Support
-------
Per unit x role x head offset x split, counts of C1 ``M`` pairs: ``E`` from
:func:`tools.i6n_contracts.eligibility.operational_eligibility` (the audited C1
function, evaluated on each distinct week shape and reused by elapsed offset,
which is exact because E compares instants only), input observed at ``t``,
target bar observed exactly at ``t+h`` and the split rule of
``tools.i6n_weekly_data.build_weekly_partitions`` (origin interval AND target
interval, labels known by the split cutoff). Equality with that reference is a
test. Every enumerated origin is either counted or excluded with a primary
typed reason; ``origins_enumerated == n + sum(exclusions)``.

Data coverage
-------------
A unit is READY only when the retained input starts at or before its context
start (history start minus the largest admitted input span) and the clock
scope certificate covers [context start, last needed instant). Otherwise it is
WAIT_DEPENDENCY with the exact missing object and span; history is never
shortened, no other week is substituted and nothing is skipped silently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from bisect import bisect_left
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from tools.i6n_contracts import eligibility as c1
from tools.i6n_weekly_data import _years_before

PLAN_SCHEMA = "i6n.weekly_campaign.plan.v1"
CONFIG_SCHEMA = "i6n.weekly_campaign.config.v1"
UNIT_SCHEMA = "i6n.weekly_unit.v1"
SUPPORT_RULE = "c1.M with build_weekly_partitions split rule (origin AND target interval)"
NOMINAL_CERTIFICATION = "NOMINAL_CONFIGURED_NOT_BROKER_CERTIFIED"
SHORT_HORIZONS = (2, 3, 4, 5, 6, 12)
LONG_HORIZONS = (18, 24, 36, 48, 60, 72)
BUSINESS_HORIZONS = SHORT_HORIZONS + LONG_HORIZONS
TRAJECTORY_K = (2, 3, 4)
UTC = timezone.utc
HOUR = timedelta(hours=1)
TEST_START = datetime(2025, 1, 1, tzinfo=UTC)
SPLITS = ("TRAIN", "INNER", "VALIDATION")
TEST_CLOSED = "TARGET_IN_TEST_CLOSED"
BEFORE_SPLIT = c1.LabelReason.TARGET_BEFORE_SPLIT_START.value
AFTER_SPLIT = c1.LabelReason.TARGET_AT_OR_AFTER_SPLIT_END.value
INPUT_UNAVAILABLE = c1.LabelReason.INPUT_UNAVAILABLE.value
TARGET_NOT_OBSERVED = c1.LabelReason.TARGET_NOT_OBSERVED.value


class CampaignError(ValueError):
    """Typed refusal with a stable ``code``."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _iso(t: datetime) -> str:
    return c1.iso(t)


def _epoch(t: datetime) -> int:
    return int(t.timestamp())


def _at(seconds: int) -> datetime:
    return datetime.fromtimestamp(seconds, UTC)


def file_sha256(path: Path) -> str:
    """SHA-256 of exact file bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Configured nominal calendar


def nominal_fx_calendar(first_sunday: date, last_sunday: date, *, calendar_id: str, version: str,
                        zone: str = "America/New_York", hour: int = 17,
                        entry_delay_after_open_hours: int = 12,
                        force_flat_before_close_hours: int = 12) -> Dict[str, Any]:
    """Nominal FX week: Sunday ``hour`` to Friday ``hour`` local, entry cutoff Thursday 00:00.

    Wall-clock arithmetic in ``zone`` (DST-aware). It reproduces the retained
    ``FX_NY_17_ASSUMED_NOT_BROKER`` calendar byte for byte (a test checks the
    canonical digest). It is a configured nominal schedule, not broker evidence.
    """
    if first_sunday.weekday() != 6 or last_sunday.weekday() != 6 or last_sunday < first_sunday:
        raise CampaignError("CALENDAR_INVALID", "Sunday bounds required")
    tz = ZoneInfo(zone)
    weeks = []
    day = first_sunday
    while day <= last_sunday:
        sunday = datetime(day.year, day.month, day.day, hour, tzinfo=tz)
        weeks.append(dict(week_id=day.isoformat(), market_open=_iso(sunday),
                          market_close=_iso(sunday + timedelta(days=5)),
                          entry_cutoff=_iso((sunday + timedelta(days=4)).replace(hour=0))))
        day += timedelta(days=7)
    return dict(market_calendar=dict(calendar_id=calendar_id, version=version, weeks=weeks),
                entry_delay_after_open_hours=entry_delay_after_open_hours,
                force_flat_before_close_hours=force_flat_before_close_hours)


def head_offsets(horizons: Iterable[int], ks: Sequence[int] = TRAJECTORY_K) -> Dict[int, Tuple[int, ...]]:
    """Endpoint plus every trajectory offset j*h/k (k in ks, h/k integer) per horizon."""
    result = {}
    for h in horizons:
        h = c1.validate_horizon(h)
        offs = {h}
        for k in ks:
            if h % k == 0:
                offs.update(j * (h // k) for j in range(1, k + 1))
        result[h] = tuple(sorted(offs))
    return result


# --------------------------------------------------------------------------
# Units and folds


def _split(name, o_lo, o_hi, q_lo, q_hi, cutoff):
    return {"split_id": name, "origin_start": _iso(o_lo), "origin_end": _iso(o_hi),
            "target_start": _iso(q_lo), "target_end": _iso(q_hi),
            "label_available_by": None if cutoff is None else _iso(cutoff),
            "membership": "ORIGIN_AND_TARGET_INTERVAL"}


def _unit_frame(kind, unit_id, weeks, index, *, history_years, inner_days, max_context_hours,
                validation, structural, quarter=None):
    if index <= 0 or index >= len(weeks):
        raise CampaignError("CALENDAR_COVERAGE_MISSING", f"{unit_id}: no preceding market close")
    week, prev = weeks[index], weeks[index - 1]
    cutoff = prev.market_close
    if not cutoff < week.market_open:
        raise CampaignError("CALENDAR_INVALID", f"{unit_id}: refit needs a closed interval")
    if week.market_open - cutoff > timedelta(days=7):
        raise CampaignError("CALENDAR_GAP_TOO_LARGE", f"{unit_id}: preceding close is stale")
    start = _years_before(cutoff, history_years)
    inner = cutoff - timedelta(days=inner_days)
    if inner <= start:
        raise CampaignError("INNER_CONSUMES_HISTORY", unit_id)
    if weeks[0].market_open > start:
        raise CampaignError("CALENDAR_COVERAGE_MISSING", f"{unit_id}: calendar starts after history")
    context = start - max_context_hours * HOUR
    splits = {"TRAIN": _split("TRAIN", start, inner, start, inner, inner),
              "INNER": _split("INNER", inner, cutoff, inner, cutoff, cutoff),
              "VALIDATION": None}
    if validation is not None:
        splits["VALIDATION"] = _split(validation.split_id, week.market_open, week.market_close,
                                      validation.start, validation.end, validation.label_available_by)
    return {"schema": UNIT_SCHEMA, "unit_id": unit_id, "fold_id": unit_id, "kind": kind,
            "quarter": quarter, "week_id": week.week_id, "evaluation_week": week.to_json(),
            "fit_cutoff": _iso(cutoff), "refit_window": {"start": _iso(cutoff), "end": _iso(week.market_open)},
            "release_at": _iso(week.market_open), "history_start": _iso(start),
            "inner_start": _iso(inner), "context_start": _iso(context),
            "history_years": history_years, "inner_days": inner_days,
            "max_context_hours": max_context_hours, "splits": splits,
            "scored_outer": validation is not None,
            "structural_evaluation_split": None if structural is None else structural.to_json()}


def evaluation_week_indices(weeks: Sequence[c1.OperationalWeek], split: c1.SplitBounds) -> List[int]:
    """Every week whose operational band [A, B) meets the split (targets lie in (A, B))."""
    return [i for i, w in enumerate(weeks)
            if w.operational_start < split.end and w.force_flat > split.start]


def _quarters(year, zone):
    out = []
    for q in range(4):
        m = 3 * q + 1
        nxt = (year + (m + 3 > 12), (m + 3 - 1) % 12 + 1)
        out.append((f"{year}Q{q + 1}", datetime(year, m, 1, tzinfo=zone).astimezone(UTC),
                    datetime(nxt[0], nxt[1], 1, tzinfo=zone).astimezone(UTC)))
    return out


def quarter_fold_indices(weeks: Sequence[c1.OperationalWeek], year: int,
                         zone: str = "America/New_York") -> List[Tuple[str, int]]:
    """(fold_id, index of the week AFTER the cutoff) for each quarter of ``year``.

    Cutoff = close of the last calendar week with O >= quarter start and
    C <= quarter end. Quarter bounds are evaluated in UTC and in the calendar
    zone; a disagreement is refused rather than silently resolved.
    """
    result = []
    for (fid, s_utc, e_utc), (_, s_loc, e_loc) in zip(_quarters(year, UTC), _quarters(year, ZoneInfo(zone))):
        picks = []
        for s, e in ((s_utc, e_utc), (s_loc, e_loc)):
            inside = [i for i, w in enumerate(weeks) if w.market_open >= s and w.market_close <= e]
            if not inside:
                raise CampaignError("FOLD_WEEK_MISSING", fid)
            picks.append(inside[-1])
        if picks[0] != picks[1]:
            raise CampaignError("FOLD_BOUNDARY_AMBIGUOUS", fid)
        if picks[0] + 1 >= len(weeks):
            raise CampaignError("CALENDAR_COVERAGE_MISSING", f"{fid}: no structural week after cutoff")
        result.append((fid, picks[0] + 1))
    return result


# --------------------------------------------------------------------------
# Support index


class SupportIndex:
    """Per role and head offset: sorted origin lists by primary exclusion reason.

    ``E`` comes from C1 ``operational_eligibility`` evaluated once per distinct
    week shape (grid length and A, K, B, C offsets from O) and replayed by
    elapsed offset. Labels: input observed at ``t`` (bar-end clock), target
    observed exactly at ``t+h``; targets at or after ``test_start`` are typed
    ``TARGET_IN_TEST_CLOSED`` and never looked up.
    """

    def __init__(self, weeks: Sequence[c1.OperationalWeek], observed_seconds: Sequence[int], *,
                 roles: Sequence[c1.Role], offsets: Sequence[int], test_start: datetime):
        self.observed = frozenset(observed_seconds)
        self.test_start = _epoch(test_start)
        self.roles, self.offsets = tuple(roles), tuple(offsets)
        self.lists: Dict[Tuple[str, int], Dict[str, List[int]]] = {}
        self.operational: Dict[Tuple[str, int], List[int]] = {}
        self.origins: List[int] = []
        shapes: Dict[tuple, Dict[Tuple[str, int], List[Optional[str]]]] = {}
        for w in weeks:
            o = _epoch(w.market_open)
            # Same grid as the historical cell: whole hours from O up to C (exclusive).
            grid = list(range(o, math.ceil(w.market_close.timestamp()), 3600))
            self.origins.extend(grid)
            shape = (len(grid), _epoch(w.operational_start) - o, _epoch(w.entry_cutoff) - o,
                     _epoch(w.force_flat) - o, _epoch(w.market_close) - o)
            if shape not in shapes:
                shapes[shape] = {(r.value, h): [None if ok else reasons[0].value
                                                for ok, reasons in (c1.operational_eligibility(_at(t), h, r, w)
                                                                    for t in grid)]
                                 for r in self.roles for h in self.offsets}
            table = shapes[shape]
            for key, primaries in table.items():
                h = key[1]
                buckets = self.lists.setdefault(key, {})
                ops = self.operational.setdefault(key, [])
                for t, primary in zip(grid, primaries):
                    if primary is not None:
                        reason = primary
                    else:
                        ops.append(t)
                        q = t + 3600 * h
                        if t not in self.observed:
                            reason = INPUT_UNAVAILABLE
                        elif q >= self.test_start:
                            reason = TEST_CLOSED
                        elif q not in self.observed:
                            reason = TARGET_NOT_OBSERVED
                        else:
                            reason = ""
                    buckets.setdefault(reason, []).append(t)

    @staticmethod
    def _between(values: List[int], lo: int, hi: int) -> int:
        return max(0, bisect_left(values, hi) - bisect_left(values, lo))

    def count(self, role: str, h: int, o_lo: int, o_hi: int, q_lo: int, q_hi: int,
              labels: bool = True) -> Dict[str, Any]:
        """Support of one split: origin in [o_lo, o_hi), target in [q_lo, q_hi)."""
        buckets = self.lists[(role, h)]
        enumerated = self._between(self.origins, o_lo, o_hi)
        ops = self.operational[(role, h)]
        n_operational = self._between(ops, max(o_lo, q_lo - 3600 * h), min(o_hi, q_hi - 3600 * h))
        if not labels:
            return {"label_state": "WAIT_DEPENDENCY", "n": None, "n_operational": n_operational,
                    "origins_enumerated": enumerated, "status": "WAIT_DEPENDENCY",
                    "exclusions": None, "origins_sha256": None}
        exclusions = {}
        for reason, values in buckets.items():
            if reason:
                k = self._between(values, o_lo, o_hi)
                if k:
                    exclusions[reason] = k
        ok = buckets.get("", [])
        lo, hi = bisect_left(ok, o_lo), bisect_left(ok, o_hi)
        # In split <=> q_lo <= t + h < q_hi; the remainder is before/after the split.
        a = min(max(bisect_left(ok, q_lo - 3600 * h), lo), hi)
        b = min(max(bisect_left(ok, q_hi - 3600 * h), a), hi)
        chosen = ok[a:b]
        if a - lo:
            exclusions[BEFORE_SPLIT] = a - lo
        if hi - b:
            exclusions[AFTER_SPLIT] = hi - b
        if enumerated != len(chosen) + sum(exclusions.values()):
            raise CampaignError("SUPPORT_IDENTITY_BROKEN", f"{role} h={h}")
        return {"label_state": "EVALUATED", "n": len(chosen), "n_operational": n_operational,
                "origins_enumerated": enumerated,
                "status": "SUPPORTED" if chosen else "NO_ELIGIBLE_ROWS",
                "exclusions": dict(sorted(exclusions.items())),
                "origins_sha256": c1.canonical_digest(chosen),
                "first_eligible_origin": _iso(_at(chosen[0])) if chosen else None,
                "last_eligible_origin": _iso(_at(chosen[-1])) if chosen else None,
                "last_eligible_target": _iso(_at(chosen[-1] + 3600 * h)) if chosen else None}


# --------------------------------------------------------------------------
# Plan


def _coverage(unit, *, data_start, scope, needed_end):
    ctx = datetime.fromisoformat(unit["context_start"])
    missing = []
    if data_start > ctx:
        missing.append({"object": "SUCCESSOR_INPUT_ROWS", "start": _iso(ctx), "end": _iso(data_start),
                        "detail": "retained input starts after the unit's context start "
                                  "(history start minus the largest admitted input span)"})
    if scope["start"] > ctx or scope["end"] < needed_end:
        lo, hi = (ctx, scope["start"]) if scope["start"] > ctx else (scope["end"], needed_end)
        missing.append({"object": "CLOCK_SCOPE_CERTIFICATE", "start": _iso(lo), "end": _iso(hi),
                        "detail": f"clock scope {scope['scope_id']} does not cover this span"})
    return {"state": "WAIT_DEPENDENCY" if missing else "READY", "missing": missing,
            "needed_start": _iso(ctx), "needed_end": _iso(needed_end)}


def build_plan(*, calendar_config: Mapping[str, Any], observed_seconds: Sequence[int],
               clock_scope: Mapping[str, Any], horizons: Sequence[int] = BUSINESS_HORIZONS,
               roles: Sequence[str] = ("MANAGEMENT", "ENTRY"), fit_role: str = "MANAGEMENT",
               history_years: int = 4, inner_days: int = 28, max_context_hours: int = 432,
               development_split: c1.SplitBounds, fold_year: Optional[int] = 2023,
               test_start: datetime = TEST_START, min_support: Optional[Mapping[str, int]] = None,
               experiment: str, arms_per_week: int = 2,
               provenance: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return the sealed, JSON-ready campaign plan (no I/O)."""
    ref, weeks = c1.load_calendar(calendar_config)
    obs = sorted(int(t) for t in observed_seconds)
    if not obs:
        raise CampaignError("NO_DATA")
    if len(set(obs)) != len(obs):
        raise CampaignError("DUPLICATE_TIMESTAMPS")
    if obs[-1] >= _epoch(test_start):
        raise CampaignError("TEST_ROWS_PRESENT", "retained input contains TEST timestamps")
    if development_split.end > test_start or development_split.split_id.upper() == "TEST":
        raise CampaignError("SPLIT_INCLUDES_TEST")
    hs = tuple(c1.validate_horizon(h) for h in horizons)
    if len(set(hs)) != len(hs):
        raise CampaignError("DUPLICATE_HORIZON")
    role_objs = tuple(c1.Role(r) for r in roles)
    if fit_role not in roles:
        raise CampaignError("FIT_ROLE_NOT_DECLARED")
    minimum = {"TRAIN": 1, "INNER": 1, "VALIDATION": 1, **(min_support or {})}
    if any(type(v) is not int or v < 1 for v in minimum.values()) or set(minimum) != set(SPLITS):
        raise CampaignError("MIN_SUPPORT_INVALID")
    offsets_by_h = head_offsets(hs)
    offsets = sorted({o for v in offsets_by_h.values() for o in v})
    data_start, data_end = _at(obs[0]), _at(obs[-1])
    scope = {"scope_id": str(clock_scope["scope_id"]),
             "start": c1.utc_instant("scope.start", clock_scope["start"]),
             "end": c1.utc_instant("scope.end", clock_scope["end"])}
    frames = [_unit_frame("WEEKLY_REFIT", "W" + weeks[i].week_id, weeks, i,
                          history_years=history_years, inner_days=inner_days,
                          max_context_hours=max_context_hours, validation=development_split,
                          structural=None)
              for i in evaluation_week_indices(weeks, development_split)]
    if not frames:
        raise CampaignError("NO_EVALUATION_WEEKS")
    if fold_year is not None:
        for fid, i in quarter_fold_indices(weeks, fold_year):
            w = weeks[i]
            structural = c1.SplitBounds(f"STRUCTURAL_UNSCORED_{fid}", w.market_open,
                                        min(w.market_close, test_start), None)
            frames.append(_unit_frame("INNER_FOLD", fid, weeks, i, history_years=history_years,
                                      inner_days=inner_days, max_context_hours=max_context_hours,
                                      validation=None, structural=structural, quarter=fid))
    index = SupportIndex(weeks, obs, roles=role_objs, offsets=offsets, test_start=test_start)
    by_h = {str(h): 0 for h in hs}
    units = []
    for u in frames:
        week_close = datetime.fromisoformat(u["evaluation_week"]["market_close"])
        # Scored or structural, the cell reads the unit's week (never past TEST).
        needed_end = min(week_close, test_start)
        u["data"] = _coverage(u, data_start=data_start, scope=scope, needed_end=needed_end)
        labels = u["data"]["state"] == "READY"
        support = {}
        for r in roles:
            support[r] = {}
            for o in offsets:
                entry = {"business_horizon": o in hs}
                for name in SPLITS:
                    sp = u["splits"][name]
                    if sp is None:
                        continue
                    entry[name] = index.count(r, o, *(_epoch(datetime.fromisoformat(sp[k])) for k in
                                                      ("origin_start", "origin_end", "target_start", "target_end")),
                                              labels=labels)
                support[r][str(o)] = entry
        u["support"] = support
        u["horizons"] = {}
        for h in hs:
            if not labels:
                state, cause = "WAIT_DEPENDENCY", "; ".join(m["object"] for m in u["data"]["missing"])
            else:
                short = [s for s in SPLITS if u["splits"][s] is not None
                         and support[fit_role][str(h)][s]["n"] < minimum[s]]
                state = "NO_ELIGIBLE_ROWS" if short else "READY"
                cause = None if not short else "below minimum label support in " + ",".join(short)
            u["horizons"][str(h)] = {"state": state, "cause": cause,
                                     "head_offsets": list(offsets_by_h[h])}
            if u["kind"] == "WEEKLY_REFIT" and state == "READY":
                by_h[str(h)] += 1
        u["unit_sha256"] = c1.canonical_digest(u)
        units.append(u)
    weekly = [u for u in units if u["kind"] == "WEEKLY_REFIT"]
    folds = [u for u in units if u["kind"] == "INNER_FOLD"]
    earliest = min(datetime.fromisoformat(u["context_start"]) for u in units)
    coverage = []
    for w in weeks:
        if w.market_close <= earliest:
            continue
        o, c = _epoch(w.market_open), _epoch(w.market_close)
        bars = bisect_left(obs, c + 1) - bisect_left(obs, o + 1)  # bar ends in (O, C]
        coverage.append({"week_id": w.week_id, "calendar_bars": (c - o) // 3600,
                         "observed_bars": bars})
    body = {
        "schema": PLAN_SCHEMA, "experiment": experiment,
        "calendar": {"calendar_id": ref.calendar_id, "version": ref.version,
                     "sha256": c1.canonical_digest(dict(calendar_config)),
                     "certification": NOMINAL_CERTIFICATION,
                     "weeks": len(weeks), "first_week": weeks[0].week_id, "last_week": weeks[-1].week_id},
        "eligibility_version": c1.ELIGIBILITY_VERSION, "support_rule": SUPPORT_RULE,
        "horizons_hours": list(hs), "head_offsets": {str(h): list(v) for h, v in offsets_by_h.items()},
        "roles": list(roles), "fit_role": fit_role, "history_years": history_years,
        "inner_days": inner_days, "max_context_hours": max_context_hours,
        "min_support": dict(sorted(minimum.items())),
        "development_split": development_split.to_json(), "test_start": _iso(test_start),
        "data": {"first_timestamp": _iso(data_start), "last_timestamp": _iso(data_end),
                 "rows": len(obs), "timestamps_sha256": c1.canonical_digest(obs),
                 "clock_scope": {"scope_id": scope["scope_id"], "start": _iso(scope["start"]),
                                 "end": _iso(scope["end"])},
                 "values_read": False, "test_rows_read": False,
                 "week_coverage": coverage,
                 "weeks_without_bars": [x["week_id"] for x in coverage if x["observed_bars"] == 0]},
        "provenance": dict(provenance or {}),
        "units": units,
        "summary": {
            "weekly_units": len(weekly), "fold_units": len(folds),
            "weekly_units_ready": sum(u["data"]["state"] == "READY" for u in weekly),
            "fold_units_ready": [u["fold_id"] for u in folds if u["data"]["state"] == "READY"],
            "fold_units_waiting": [u["fold_id"] for u in folds if u["data"]["state"] != "READY"],
        },
        "dispositions": {"by_horizon": by_h, "weekly_fits_per_arm": sum(by_h.values()),
                         "arms_per_week": arms_per_week,
                         "weekly_fits_all_arms": arms_per_week * sum(by_h.values()),
                         "plan_ceiling": arms_per_week * len(hs) * 53,
                         "rule": "a (week, h) disposition needs every split of the fit role at "
                                 "or above min_support; others are typed, never scored"},
        "claims": {"fits_run": False, "prices_read": False, "test_opened": False,
                   "calendar_certified": False},
    }
    body["plan_sha256"] = c1.canonical_digest(body)
    return body


# --------------------------------------------------------------------------
# Downstream views (lane B consumes these through INTERFACE_C)


def fold_view(plan: Mapping[str, Any], cells: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, List[dict]]:
    """``folds[h] = [{fold_id, unit_id, state, cause, cutoff_utc, unit_sha256, cell_config}]``."""
    cells = cells or {}
    out: Dict[str, List[dict]] = {}
    for h in plan["horizons_hours"]:
        out[str(h)] = [{"fold_id": u["fold_id"], "unit_id": u["unit_id"], "cutoff_utc": u["fit_cutoff"],
                        "state": u["horizons"][str(h)]["state"], "cause": u["horizons"][str(h)]["cause"],
                        "unit_sha256": u["unit_sha256"],
                        "cell_config": cells.get(u["unit_id"], {}).get(str(h)), "binding": None}
                       for u in plan["units"] if u["kind"] == "INNER_FOLD"]
    return out


def week_view(plan: Mapping[str, Any], cells: Optional[Mapping[str, Mapping[str, Any]]] = None) -> List[dict]:
    """``weeks = [{week_id, unit_id, cutoff_utc, release_at_utc, horizons, eligible, cause, ...}]``."""
    cells = cells or {}
    rows = []
    for u in plan["units"]:
        if u["kind"] != "WEEKLY_REFIT":
            continue
        ready = [int(h) for h, v in u["horizons"].items() if v["state"] == "READY"]
        rows.append({"week_id": u["week_id"], "unit_id": u["unit_id"], "cutoff_utc": u["fit_cutoff"],
                     "release_at_utc": u["release_at"], "horizons": ready, "eligible": bool(ready),
                     "cause": {h: v["cause"] for h, v in u["horizons"].items() if v["state"] != "READY"},
                     "unit_sha256": u["unit_sha256"],
                     "cell_configs": {str(h): cells.get(u["unit_id"], {}).get(str(h)) for h in ready},
                     "binding": None})
    return rows


def cell_overlay(plan: Mapping[str, Any], unit: Mapping[str, Any], h: int) -> Dict[str, Any]:
    """Keys of an ``i6n.historical_cell.v1`` config fixed by the unit (calendar added by caller)."""
    if unit["horizons"][str(h)]["state"] != "READY":
        raise CampaignError("UNIT_NOT_READY", f"{unit['unit_id']} h={h}")
    split = (plan["development_split"] if unit["scored_outer"]
             else unit["structural_evaluation_split"])
    return {"evaluation_week_id": unit["week_id"],
            "evaluation_split": {k: split[k] for k in ("split_id", "start", "end", "label_available_by")},
            "coverage_mode": "complete", "role": plan["fit_role"], "horizon_hours": int(h)}


def support_rows(plan: Mapping[str, Any], business_only: bool = False) -> List[dict]:
    """Flat per unit x role x head offset x split rows for CSV."""
    rows = []
    for u in plan["units"]:
        for role, by_o in u["support"].items():
            for o, entry in by_o.items():
                if business_only and not entry["business_horizon"]:
                    continue
                for name in SPLITS:
                    s = entry.get(name)
                    if s is None:
                        continue
                    rows.append({"unit_id": u["unit_id"], "kind": u["kind"], "week_id": u["week_id"],
                                 "fit_cutoff": u["fit_cutoff"], "data_state": u["data"]["state"],
                                 "role": role, "head_offset_hours": int(o),
                                 "business_horizon": entry["business_horizon"], "split": name,
                                 "n": s["n"], "n_operational": s["n_operational"],
                                 "origins_enumerated": s["origins_enumerated"], "status": s["status"],
                                 "exclusions": json.dumps(s["exclusions"], sort_keys=True),
                                 "origins_sha256": s["origins_sha256"]})
    return rows


# --------------------------------------------------------------------------
# File-based CLI (reads timestamps only)


def _load_json(path: Path, sha256: str) -> Any:
    if file_sha256(path) != sha256:
        raise CampaignError("DIGEST_MISMATCH", str(path.name))
    return json.loads(path.read_text())


def load_timestamps(npz_path: Path, sha256: str) -> List[int]:
    """Decode ONLY the ``ts`` member of a retained NPZ (no prices/features)."""
    import numpy as np
    if file_sha256(npz_path) != sha256:
        raise CampaignError("DIGEST_MISMATCH", npz_path.name)
    with np.load(npz_path, allow_pickle=False) as archive:
        ts = archive["ts"]
    if ts.ndim != 1 or ts.dtype.kind not in "iu":
        raise CampaignError("TIMESTAMPS_INVALID")
    return [int(t) for t in ts]


def _write(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")


def redact(value: Any) -> Any:
    """Replace absolute paths by '<private>/<basename>' (digests are kept)."""
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        return "<private>/" + os.path.basename(value)
    return value


def run_plan(config_path: Path, output: Path) -> Dict[str, Any]:
    """Build the plan from a private config; write private and redacted artifacts."""
    cfg = json.loads(Path(config_path).read_text())
    if cfg.get("schema") != CONFIG_SCHEMA:
        raise CampaignError("CONFIG_INVALID", "schema")
    d = cfg["data"]
    clock = _load_json(Path(d["clock_path"]), d["clock_sha256"])
    scope_doc = _load_json(Path(d["clock_scope_path"]), d["clock_scope_sha256"])
    calendar = _load_json(Path(cfg["calendar_path"]), cfg["calendar_sha256"])
    if clock.get("status") != "RESOLVED" or clock.get("input_sha256") != d["input_sha256"]:
        raise CampaignError("CLOCK_NOT_BOUND")
    if c1.canonical_digest(calendar) != clock.get("calendar_sha256"):
        raise CampaignError("CALENDAR_NOT_BOUND", "calendar differs from the clock manifest")
    ts = load_timestamps(Path(d["input_path"]), d["input_sha256"])
    dev = cfg["development_split"]
    split = c1.SplitBounds(dev["split_id"], c1.parse_instant("dev.start", dev["start"]),
                           c1.parse_instant("dev.end", dev["end"]), None)
    scope = {"scope_id": scope_doc.get("status", "") + ":" + d["clock_scope_sha256"][:12],
             "start": c1.parse_instant("scope.start", scope_doc["data_start"]),
             "end": c1.parse_instant("scope.end", scope_doc["data_end"])}
    plan = build_plan(calendar_config=calendar, observed_seconds=ts, clock_scope=scope,
                      horizons=cfg.get("horizons_hours", BUSINESS_HORIZONS),
                      roles=cfg.get("roles", ["MANAGEMENT", "ENTRY"]),
                      fit_role=cfg.get("fit_role", "MANAGEMENT"),
                      history_years=cfg.get("history_years", 4), inner_days=cfg.get("inner_days", 28),
                      max_context_hours=cfg.get("max_context_hours", 432), development_split=split,
                      fold_year=cfg.get("fold_year", 2023), experiment=cfg["experiment"],
                      arms_per_week=cfg.get("arms_per_week", 2),
                      provenance={"input_sha256": d["input_sha256"], "clock_sha256": d["clock_sha256"],
                                  "clock_scope_sha256": d["clock_scope_sha256"],
                                  "calendar_file_sha256": cfg["calendar_sha256"],
                                  "clock_scope_status": scope_doc.get("status"),
                                  "full_clock_audit_status": scope_doc.get("full_audit_status"),
                                  "inherited_limits": cfg.get("inherited_limits", [])})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cells: Dict[str, Dict[str, Any]] = {}
    template_path = cfg.get("cell_template_path")
    if template_path:
        template = _load_json(Path(template_path), cfg["cell_template_sha256"])
        if template.get("schema") != "i6n.historical_cell.v1":
            raise CampaignError("CELL_TEMPLATE_INVALID")
        if (template["input"]["sha256"] != d["input_sha256"]
                or template["source_clock"]["sha256"] != d["clock_sha256"]
                or c1.canonical_digest(template["calendar"]) != clock["calendar_sha256"]):
            raise CampaignError("CELL_TEMPLATE_NOT_BOUND")
        (output / "cells").mkdir()
        for u in plan["units"]:
            for h, v in u["horizons"].items():
                if v["state"] != "READY":
                    continue
                cell = dict(template, **cell_overlay(plan, u, int(h)))
                path = output / "cells" / f"{u['unit_id']}_h{h}.json"
                _write(path, cell)
                cells.setdefault(u["unit_id"], {})[h] = {"path": str(path), "sha256": file_sha256(path)}
    _write(output / "PLAN.json", plan)
    view = {"schema": "i6n.weekly_campaign.views.v1", "plan_sha256": plan["plan_sha256"],
            "folds": fold_view(plan, cells), "weeks": week_view(plan, cells)}
    _write(output / "VIEWS.private.json", view)
    _write(output / "VIEWS.redacted.json", redact(view))
    rows = support_rows(plan)
    with (output / "SUPPORT.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digests = {p.name: file_sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    _write(output / "ARTIFACTS.json", {"plan_sha256": plan["plan_sha256"], "files": digests,
                                       "cells": len([1 for v in cells.values() for _ in v])})
    return {"plan_sha256": plan["plan_sha256"], **plan["summary"], **plan["dispositions"]}


def verify(output: Path) -> Dict[str, Any]:
    """Recheck artifact digests and the plan seal (reads no data)."""
    output = Path(output)
    manifest = json.loads((output / "ARTIFACTS.json").read_text())
    bad = [n for n, h in manifest["files"].items() if file_sha256(output / n) != h]
    plan = json.loads((output / "PLAN.json").read_text())
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    sealed = c1.canonical_digest(body) == plan["plan_sha256"] == manifest["plan_sha256"]
    units_ok = all(c1.canonical_digest({k: v for k, v in u.items() if k != "unit_sha256"}) == u["unit_sha256"]
                   for u in plan["units"])
    return {"state": "VERIFIED" if (not bad and sealed and units_ok) else "DAMAGED",
            "bad_files": bad, "plan_sealed": sealed, "units_sealed": units_ok}


def export(output: Path, target: Path) -> Dict[str, Any]:
    """Write git-safe evidence: compact unit manifest (fit role, business heads) and counts CSV."""
    output, target = Path(output), Path(target)
    if verify(output)["state"] != "VERIFIED":
        raise CampaignError("PLAN_DAMAGED")
    plan = json.loads((output / "PLAN.json").read_text())
    fit, hs = plan["fit_role"], [str(h) for h in plan["horizons_hours"]]
    units = []
    for u in plan["units"]:
        keep = {k: v for k, v in u.items() if k != "support"}
        keep["support_fit_role_business"] = {
            h: {s: {k: v for k, v in u["support"][fit][h][s].items()
                    if k in ("n", "n_operational", "origins_enumerated", "status", "origins_sha256", "exclusions")}
                for s in SPLITS if s in u["support"][fit][h]}
            for h in hs}
        units.append(keep)
    manifest = {"schema": "i6n.weekly_campaign.redacted_units.v1", "plan_sha256": plan["plan_sha256"],
                "plan_file_sha256": file_sha256(output / "PLAN.json"),
                "note": "unit_sha256 seals the FULL unit (all roles/offsets) in the private PLAN.json",
                **{k: plan[k] for k in ("experiment", "calendar", "horizons_hours", "head_offsets", "roles",
                                        "fit_role", "history_years", "inner_days", "max_context_hours",
                                        "min_support", "development_split", "test_start", "summary",
                                        "dispositions", "claims", "eligibility_version", "support_rule")},
                "data": {k: v for k, v in plan["data"].items() if k != "week_coverage"},
                "provenance": plan["provenance"], "units": units}
    target.mkdir(parents=True, exist_ok=True)
    (target / "WEEKLY_UNITS.redacted.json").write_text(
        json.dumps(redact(manifest), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    fields = ["unit_id", "kind", "data_state", "role", "head_offset_hours", "split", "n",
              "n_operational", "origins_enumerated", "status"]
    with (target / "SUPPORT_BUSINESS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(support_rows(plan, business_only=True))
    with (target / "WEEK_COVERAGE.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["week_id", "calendar_bars", "observed_bars"])
        writer.writeheader()
        writer.writerows(plan["data"]["week_coverage"])
    return {p.name: file_sha256(p) for p in sorted(target.iterdir())
            if p.name in ("WEEKLY_UNITS.redacted.json", "SUPPORT_BUSINESS.csv", "WEEK_COVERAGE.csv")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("export")
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--to", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps(run_plan(args.config, args.output), indent=1, sort_keys=True))
        return 0
    if args.command == "export":
        print(json.dumps(export(args.output, args.to), indent=1, sort_keys=True))
        return 0
    result = verify(args.output)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if result["state"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
