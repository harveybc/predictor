"""Calendar MARKET-TIME sampling candidates MG2/MG12 (order 85ea134e section 2).

CPU-only support study for sampling input slots on the EXPECTED open-market
hour grid of the versioned ASSUMED Forex calendar, separately identified from
the elapsed-time DS2/DS12 grids:

* Expected endpoints are generated from the calendar (every whole bar-END hour
  in ``(open, close]`` of each week), never from dataset rows. A vendor gap in
  an open session is a missing observation with a counted exclusion
  (``INPUT_SLOT_NOT_OBSERVED``), not a compressed closure.
* Short candidate ``MG2_MARKET_EXACT``: 13 slots every 2 expected open-market
  hours (24 open-market-hour span). Long candidate ``MG12_MARKET_EXACT``: 13
  slots every 12 expected open-market hours (144 open-market-hour span).
  ELAPSED spans vary across closures and are never labelled fixed 24/144 h;
  actual UTC timestamps, actual elapsed offsets and closure flags are retained
  for a later model to encode.
* Exact finite retained observations only; no fill, interpolation, synthetic
  prices or target cleaning. Every slot is <= t, so availability holds.
  Input history may cross weekends. Targets remain exact ELAPSED t+h under
  the original C1 business masks and split purge from ``tools.i6dmo.data``.

This module prepares data/contract evidence. It fits nothing, reads no TEST
bar (the loaders refuse one) and declares no sampling rule optimal.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from tools.i6dmo import data, provenance
from tools.i6n_contracts import eligibility as c1

UTC = timezone.utc
HOUR = 3600
STATUS = "SUPPORT_STUDY_NOT_A_PREDICTIVE_MEASUREMENT"


@dataclass(frozen=True)
class MarketWindowSpec:
    """13 slots separated by ``spacing_market_hours`` EXPECTED open-market hours."""

    spacing_market_hours: int
    observations: int = 13

    @property
    def span_market_hours(self) -> int:
        return self.spacing_market_hours * (self.observations - 1)

    @classmethod
    def from_design(cls, design: Mapping[str, Any]) -> "MarketWindowSpec":
        window = data.WindowSpec.from_design(design)
        return cls(window.spacing_hours, window.observations)


def market_rule_name(spec: MarketWindowSpec) -> str:
    return f"MG{spec.spacing_market_hours}_MARKET_EXACT"


def expected_market_hours(bundle: data.InputBundle) -> np.ndarray:
    """Expected bar-END instants from the calendar: every whole hour in (open, close]."""
    _, weeks = data._weeks(bundle)
    out: List[int] = []
    for w in weeks:
        o, c = int(w.market_open.timestamp()), int(w.market_close.timestamp())
        out.extend(range(o + HOUR, c + 1, HOUR))
    grid = np.asarray(out, dtype=np.int64)
    if np.any(np.diff(grid) <= 0):
        raise data.DataContractError("CALENDAR_INVALID", "expected market hours not increasing")
    return grid


@dataclass(frozen=True)
class MarketRecord:
    origin: int
    split: str
    week_id: Optional[str]
    input_ok: bool
    input_reason: Optional[str]
    slot_instants: Tuple[int, ...]
    actual_offsets_s: Tuple[int, ...]
    closure_crossed: Tuple[bool, ...]
    elapsed_span_s: int
    label_ok: Tuple[bool, ...]
    entry_ok: Tuple[bool, ...]


@dataclass
class MarketOriginTable:
    rule: str
    spec: MarketWindowSpec
    horizons: Tuple[int, ...]
    identity: Dict[str, Any]
    origin_ts: np.ndarray
    split: np.ndarray
    week_id: np.ndarray
    input_ok: np.ndarray
    input_reason: List[Optional[str]]
    slot_instants: np.ndarray        # (n, obs) int64; -1 before grid start
    slot_index: np.ndarray           # (n, obs) int64 row in retained input; -1 absent
    closure_crossed: np.ndarray      # (n, obs) bool; slot j reached across a closure from j-1
    label_ok: np.ndarray
    entry_ok: np.ndarray
    counts: Dict[str, Any]
    split_weeks: Dict[str, List[str]]
    index: Dict[int, int] = field(default_factory=dict)

    def record(self, t: int) -> MarketRecord:
        i = self.index[int(t)]
        slots = tuple(int(v) for v in self.slot_instants[i])
        return MarketRecord(
            origin=int(self.origin_ts[i]), split=str(self.split[i]),
            week_id=str(self.week_id[i]) or None, input_ok=bool(self.input_ok[i]),
            input_reason=self.input_reason[i], slot_instants=slots,
            actual_offsets_s=tuple(s - int(t) for s in slots),
            closure_crossed=tuple(bool(v) for v in self.closure_crossed[i]),
            elapsed_span_s=slots[-1] - slots[0],
            label_ok=tuple(bool(v) for v in self.label_ok[i]),
            entry_ok=tuple(bool(v) for v in self.entry_ok[i]))

    def origins_in_week(self, week_id: str) -> List[int]:
        return [int(t) for t, w in zip(self.origin_ts, self.week_id) if w == week_id]

    def materialized(self, split: str) -> np.ndarray:
        return (self.split == split) & self.input_ok & self.label_ok.any(axis=1)


def build_market_table(bundle: data.InputBundle, design: Mapping[str, Any], *,
                       evaluation_splits=None, authorized_evaluation_end=None) -> MarketOriginTable:
    """Evaluate expected market-hour origins + C1 masks; default splits are unchanged.

    Explicit pre-TEST splits support fit-free master tables with later fold
    maturity filtering. Their bounds become part of the returned identity.
    """
    design = data.validate_design(design)
    spec = MarketWindowSpec.from_design(design)
    rule = market_rule_name(spec)
    splits = data.split_bounds(design) if evaluation_splits is None else tuple(evaluation_splits)
    if evaluation_splits is not None:
        limit = data.test_start(design)
        if authorized_evaluation_end is not None:
            if authorized_evaluation_end != datetime(2026, 1, 1, tzinfo=UTC):
                raise data.DataContractError('AUTHORIZED_EVALUATION_END_INVALID')
            limit = authorized_evaluation_end
        if not splits or any(not isinstance(s, c1.SplitBounds) for s in splits) \
                or len({s.split_id for s in splits}) != len(splits) \
                or any(s.end > limit or (s.label_available_by is not None and s.label_available_by > limit)
                       for s in splits) \
                or any(a.end > b.start for a, b in zip(splits, splits[1:])):
            raise data.DataContractError("EVALUATION_SPLITS_INVALID", "ordered disjoint pre-TEST bounds required")
    ref, weeks = data._weeks(bundle)
    hs = data.horizons(design)
    ts = bundle.ts
    pos = {int(v): i for i, v in enumerate(ts)}
    finite = np.isfinite(bundle.x).all(axis=1)
    grid = expected_market_hours(bundle)
    grid_pos = {int(v): i for i, v in enumerate(grid)}
    observed = c1.ObservedTargets({datetime.fromtimestamp(int(v), UTC):
                                   datetime.fromtimestamp(int(v), UTC) for v in ts})
    import bisect
    opens = [w.market_open for w in weeks]
    lo, hi = int(splits[0].start.timestamp()), int(splits[-1].end.timestamp())
    cand = np.flatnonzero((ts >= lo) & (ts < hi))
    if evaluation_splits is not None:
        active = np.zeros(ts.shape, dtype=bool)
        for s in splits:
            active |= (ts >= int(s.start.timestamp())) & (ts < int(s.end.timestamp()))
        cand = np.flatnonzero(active)
    n, nh, obs = len(cand), len(hs), spec.observations
    out_split = np.empty(n, dtype=object)
    out_week = np.empty(n, dtype=object)
    input_ok = np.zeros(n, dtype=bool)
    input_reason: List[Optional[str]] = []
    slot_instants = np.full((n, obs), -1, dtype=np.int64)
    slot_index = np.full((n, obs), -1, dtype=np.int64)
    closure = np.zeros((n, obs), dtype=bool)
    label_ok = np.zeros((n, nh), dtype=bool)
    entry_ok = np.zeros((n, nh), dtype=bool)
    reasons = ("ORIGIN_OUTSIDE_CALENDAR_WEEK", "ORIGIN_NOT_EXPECTED_MARKET_HOUR",
               "INPUT_BEFORE_DATA_START", "INPUT_SLOT_NOT_OBSERVED",
               "INPUT_VALUE_NOT_FINITE", "NO_VALID_HEAD")
    counts = {s: {"origins": {"candidates": 0, "input_ok": 0, **{r: 0 for r in reasons}},
                  "heads": {str(h): {"label_valid": 0, "entry_valid": 0, "reasons": Counter()}
                            for h in hs},
                  "rows_materialized": 0} for s in
              (data.SPLITS if evaluation_splits is None else tuple(s.split_id for s in splits))}
    step = spec.spacing_market_hours
    for k, i in enumerate(cand):
        t = int(ts[i])
        t_dt = datetime.fromtimestamp(t, UTC)
        split = next(s for s in splits if s.start <= t_dt < s.end)
        out_split[k] = split.split_id
        c = counts[split.split_id]
        c["origins"]["candidates"] += 1
        gp = grid_pos.get(t)
        reason = None
        if gp is None:
            reason = "ORIGIN_NOT_EXPECTED_MARKET_HOUR"
        elif gp - step * (obs - 1) < 0:
            reason = "INPUT_BEFORE_DATA_START"
        else:
            slots = grid[gp - step * (obs - 1): gp + 1: step]
            slot_instants[k] = slots
            closure[k, 1:] = np.diff(slots) > step * HOUR
            if int(slots[0]) < int(ts[0]):
                reason = "INPUT_BEFORE_DATA_START"
            else:
                for j, s_ in enumerate(slots):
                    idx = pos.get(int(s_))
                    if idx is None:
                        reason = reason or "INPUT_SLOT_NOT_OBSERVED"
                        continue
                    slot_index[k, j] = idx
                    if not finite[idx]:
                        reason = reason or "INPUT_VALUE_NOT_FINITE"
        input_ok[k] = reason is None
        input_reason.append(reason)
        if reason is None:
            c["origins"]["input_ok"] += 1
        else:
            c["origins"][reason] += 1
        w = bisect.bisect_right(opens, t_dt) - 1
        if w < 0 or not t_dt < weeks[w].market_close:
            out_week[k] = ""
            c["origins"]["ORIGIN_OUTSIDE_CALENDAR_WEEK"] += 1
            for h in hs:
                c["heads"][str(h)]["reasons"]["ORIGIN_OUTSIDE_CALENDAR_WEEK"] += 1
            continue
        week = weeks[w]
        out_week[k] = week.week_id
        mgmt = c1.evaluate_heads(t_dt, hs, c1.Role.MANAGEMENT, week,
                                 input_available=bool(input_ok[k]),
                                 observed_targets=observed, split=split)
        entry = c1.evaluate_heads(t_dt, hs, c1.Role.ENTRY, week,
                                  input_available=bool(input_ok[k]),
                                  observed_targets=observed, split=split)
        for j, h in enumerate(hs):
            label_ok[k, j] = bool(mgmt[j].label_eligible)
            entry_ok[k, j] = bool(entry[j].label_eligible)
            head = c["heads"][str(h)]
            if label_ok[k, j]:
                head["label_valid"] += 1
            else:
                head["reasons"][mgmt[j].label_reasons[0].value] += 1
            head["entry_valid"] += int(entry_ok[k, j])
        if input_ok[k] and not label_ok[k].any():
            c["origins"]["NO_VALID_HEAD"] += 1
    split_arr = out_split.astype(str)
    have = set(int(v) for v in ts)
    split_weeks: Dict[str, List[str]] = {}
    for s in splits:
        split_weeks[s.split_id] = [w.week_id for w in weeks
                                   if w.operational_start < s.end and w.force_flat > s.start]
        s0, s1 = int(s.start.timestamp()), int(s.end.timestamp())
        expected = grid[(grid >= s0) & (grid < s1)]
        missing = int(sum(1 for v in expected.tolist() if v not in have))
        counts[s.split_id]["closures_vs_gaps"] = {
            "expected_open_market_hours": int(len(expected)),
            "observed_bars_at_expected_hours": int(len(expected)) - missing,
            "unplanned_missing_expected_hours": missing,
            "closure_hours_are_not_expected": True}
        sel = (split_arr == s.split_id) & input_ok & label_ok.any(axis=1)
        counts[s.split_id]["rows_materialized"] = int(sel.sum())
        spans = (slot_instants[sel, -1] - slot_instants[sel, 0]) // HOUR
        counts[s.split_id]["elapsed_span_hours_distribution"] = {
            str(int(v)): int(cnt) for v, cnt in zip(*np.unique(spans, return_counts=True))}
        for h in hs:
            counts[s.split_id]["heads"][str(h)]["reasons"] = dict(
                sorted(counts[s.split_id]["heads"][str(h)]["reasons"].items()))
    table = MarketOriginTable(
        rule=rule, spec=spec, horizons=hs,
        identity={"rule": rule, "span_unit": "OPEN_MARKET_HOURS",
                  "span_open_market_hours": spec.span_market_hours,
                  "elapsed_span_hours_fixed": False,
                  "calendar_id": ref.calendar_id, "calendar_version": ref.version,
                  "calendar_status": ("ASSUMED_NOT_BROKER" if "ASSUMED" in ref.calendar_id
                                      or "FIXTURE" in ref.calendar_id else "SUPPLIED_UNVERIFIED"),
                  "targets": "exact ELAPSED t+h under the original C1 business masks",
                  "status": STATUS},
        origin_ts=ts[cand].astype(np.int64), split=split_arr, week_id=out_week.astype(str),
        input_ok=input_ok, input_reason=input_reason, slot_instants=slot_instants,
        slot_index=slot_index, closure_crossed=closure, label_ok=label_ok, entry_ok=entry_ok,
        counts=counts, split_weeks=split_weeks)
    table.index = {int(t): k for k, t in enumerate(table.origin_ts)}
    if evaluation_splits is not None:
        table.identity["evaluation_splits"] = [s.to_json() for s in splits]
        table.identity["evaluation_splits_sha256"] = provenance.canonical_sha256(table.identity["evaluation_splits"])
        if authorized_evaluation_end is not None:
            table.identity['owner_authorized_test_end'] = authorized_evaluation_end.isoformat()
    # synthetic fixture calendars carry FIXTURE in the id; the real assumed one carries ASSUMED
    if "ASSUMED" not in ref.calendar_id and "FIXTURE" not in ref.calendar_id:
        table.identity["calendar_status"] = "SUPPLIED_UNVERIFIED"
    else:
        table.identity["calendar_status"] = "ASSUMED_NOT_BROKER"
    return table


@dataclass
class MarketSplitArrays:
    split: str
    rule: str
    horizons: Tuple[int, ...]
    X: np.ndarray                    # (n, obs, F) exact retained values at market slots
    slot_instants: np.ndarray        # (n, obs) int64 UTC
    actual_offsets_s: np.ndarray     # (n, obs) int64 slot - t (varies across closures)
    closure_crossed: np.ndarray      # (n, obs) bool
    y_price: np.ndarray
    mask: np.ndarray
    entry_mask: np.ndarray
    price_origin: np.ndarray
    origin_ts: np.ndarray
    row_ids: np.ndarray
    week_ids: np.ndarray


def materialize(bundle: data.InputBundle, table: MarketOriginTable, split: str) -> MarketSplitArrays:
    if split not in data.SPLITS:
        raise data.DataContractError("SPLIT_UNKNOWN", f"{split!r}; TEST is never materialized")
    sel = np.flatnonzero(table.materialized(split))
    slots = table.slot_index[sel]
    if (slots < 0).any():
        raise data.DataContractError("INTERNAL_SLOT_MISSING", split)
    pos = {int(v): i for i, v in enumerate(bundle.ts)}
    origin_idx = np.asarray([pos[int(t)] for t in table.origin_ts[sel]], dtype=np.int64)
    hs = table.horizons
    mask = table.label_ok[sel].copy()
    y = np.full(mask.shape, np.nan)
    for j, h in enumerate(hs):
        rows = np.flatnonzero(mask[:, j])
        y[rows, j] = bundle.price[[pos[int(table.origin_ts[sel][r]) + h * HOUR] for r in rows]]
    instants = table.slot_instants[sel]
    return MarketSplitArrays(
        split=split, rule=table.rule, horizons=hs, X=bundle.x[slots],
        slot_instants=instants.copy(),
        actual_offsets_s=(instants - table.origin_ts[sel][:, None]).astype(np.int64),
        closure_crossed=table.closure_crossed[sel].copy(), y_price=y, mask=mask,
        entry_mask=table.entry_ok[sel] & mask, price_origin=bundle.price[origin_idx],
        origin_ts=table.origin_ts[sel].copy(),
        row_ids=np.asarray([data.row_id(t) for t in table.origin_ts[sel]], dtype=str),
        week_ids=table.week_id[sel].copy())


def _weekly(table: MarketOriginTable, split: str) -> Dict[str, Any]:
    out = {}
    sel_split = table.split == split
    for wid in table.split_weeks[split]:
        rows = sel_split & (table.week_id == wid)
        usable = rows & table.input_ok & table.label_ok.any(axis=1)
        out[wid] = {"origins": int(rows.sum()), "usable": int(usable.sum()),
                    **({"s_h": {str(h): int((rows & table.input_ok & table.label_ok[:, j]).sum())
                                for j, h in enumerate(table.horizons)}})}
    return out


def _candidate_report(table: MarketOriginTable) -> Dict[str, Any]:
    splits = {}
    for s in data.SPLITS:
        c = json.loads(json.dumps(table.counts[s]))
        c["weekly"] = _weekly(table, s)
        splits[s] = c
    return {"identity": dict(table.identity), "observations": table.spec.observations,
            "spacing_open_market_hours": table.spec.spacing_market_hours,
            "horizons_hours": list(table.horizons), "splits": splits}


def _usable_sets(table: MarketOriginTable) -> Dict[str, set]:
    out = {}
    for s in data.SPLITS:
        sel = table.materialized(s)
        out[s] = set(int(v) for v in table.origin_ts[sel])
    return out


def support_study(input_dir: str | Path, short_design: Mapping[str, Any]) -> Dict[str, Any]:
    """Full support study: MG2 + MG12 candidates vs the retained DS2/DS12 exact grids."""
    long_design = data.long_data_view(short_design)
    report: Dict[str, Any] = {
        "schema": "i6dmo.market_grid.support_study.v1",
        "status": STATUS,
        "order": "MUSASHI_TO_SATOSHI_STATIC_BASELINE_CLOSE_AND_MARKET_GRID_20261015 section 2",
        "no_rule_is_selected_from_support_counts": True,
        "candidates": {}, "elapsed_grid_reference": {}, "shared_usable_origins": {},
    }
    usable: Dict[str, Dict[str, set]] = {}
    bundles = {}
    for name, design in (("MG2_MARKET_EXACT", short_design), ("MG12_MARKET_EXACT", long_design)):
        bundle = data.load_input(input_dir, design)
        bundles[name] = bundle
        table = build_market_table(bundle, design)
        report["candidates"][name] = _candidate_report(table)
        usable[name] = _usable_sets(table)
        if name == "MG2_MARKET_EXACT":
            report["source_clock"] = {k: bundle.clock.get(k) for k in
                                      ("status", "timestamp_semantics", "availability_policy")}
            report["source_clock"]["clock_policy"] = bundle.clock.get("clock_policy")
            report["calendar"] = {"calendar_id": table.identity["calendar_id"],
                                  "version": table.identity["calendar_version"],
                                  "status": table.identity["calendar_status"],
                                  "sha256": bundle.files.get("CALENDAR.json")}
            report["inputs"] = {k: v for k, v in bundle.files.items()}
            report["examples"] = _examples(bundle, table)
    for name, design in (("DS2_EXACT", short_design), ("DS12_EXACT", long_design)):
        bundle = bundles["MG2_MARKET_EXACT" if name == "DS2_EXACT" else "MG12_MARKET_EXACT"]
        table = data.build_origin_table(bundle, design)
        ref = {}
        for s in data.SPLITS:
            sel = table.materialized(s)
            ref[s] = {"rows_materialized": int(sel.sum()),
                      "origins": dict(table.counts[s]["origins"]),
                      "heads": {h: v["label_valid"] for h, v in table.counts[s]["heads"].items()}}
            usable.setdefault(name, {})[s] = set(int(v) for v in table.origin_ts[sel])
        report["elapsed_grid_reference"][name] = ref
    for s in data.SPLITS:
        report["shared_usable_origins"][s] = {
            "both_market_candidates": len(usable["MG2_MARKET_EXACT"][s]
                                          & usable["MG12_MARKET_EXACT"][s]),
            "mg2_and_ds2": len(usable["MG2_MARKET_EXACT"][s] & usable["DS2_EXACT"][s]),
            "mg2_only_vs_ds2": len(usable["MG2_MARKET_EXACT"][s] - usable["DS2_EXACT"][s]),
            "ds2_only_vs_mg2": len(usable["DS2_EXACT"][s] - usable["MG2_MARKET_EXACT"][s]),
            "mg12_usable": len(usable["MG12_MARKET_EXACT"][s]),
            "ds12_usable": len(usable["DS12_EXACT"][s]),
        }
    return report


def _examples(bundle: data.InputBundle, table: MarketOriginTable) -> Dict[str, Any]:
    """Executed worked cases: Monday weekend crossing and DST closures (real data)."""
    _, weeks = data._weeks(bundle)
    have = table.index
    monday = None
    for w in weeks:
        a = int(w.operational_start.timestamp())
        for t in range(a, a + 6 * HOUR, HOUR):
            if t in have and table.record(t).input_ok and any(table.record(t).closure_crossed):
                monday = table.record(t)
                break
        if monday:
            break
    dst_spans = {}
    for t in table.origin_ts.tolist():
        rec = None
        if t in have:
            r = table.record(t)
            if r.input_ok and any(r.closure_crossed):
                dst_spans.setdefault(r.elapsed_span_s, t)
        if len(dst_spans) >= 2:
            break
    out: Dict[str, Any] = {}
    if monday:
        out["monday_weekend_crossing"] = {
            "origin": data.iso(monday.origin), "week_id": monday.week_id,
            "slot_instants_utc": [data.iso(s) for s in monday.slot_instants],
            "actual_offsets_hours": [o / HOUR for o in monday.actual_offsets_s],
            "closure_crossed_slots": int(sum(monday.closure_crossed)),
            "elapsed_span_hours": monday.elapsed_span_s / HOUR}
    out["dst"] = {"distinct_closure_spans_hours": sorted(s / HOUR for s in dst_spans),
                  "spans_differ": len(dst_spans) >= 2}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.i6dmo.market_grid")
    parser.add_argument("--design", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = support_study(args.input_dir, json.loads(Path(args.design).read_text()))
    report["report_sha256"] = provenance.canonical_sha256(report)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({"status": report["status"], "out": str(args.out),
                      "report_sha256": report["report_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# =========================================================================== #
# Training adapter (order 50d20d01 section 1.1; plan sections 2, 4, 6)        #
# =========================================================================== #

DESIGN_SCHEMA = "i6dmo.market_design.v1"
BUNDLE_SCHEMA = "i6dmo.market_bundle.v1"
#: Support study this adapter must reconcile against before any fit.
STUDY_SHA256 = "d2b6a2380380cfcc6eaafd0ece7b3e3c3375bf28f178d43a8e0ca05163ee0475"
#: Reference counts from that study. Plan section 1: reference counts for unchanged
#: input/calendar identities, NEVER quotas to force by filling gaps.
STUDY_REFERENCE = {
    "MG2_MARKET_EXACT": {"fit": 16496, "inner": 715, "validation": 4772},
    "MG12_MARKET_EXACT": {"fit": 13174, "inner": 546, "validation": 3804},
    "shared": {"fit": 13123, "inner": 546, "validation": 3760},
}
#: Per-window model inputs (plan section 3). Raw hours; the /24 conversion is the model's.
TIME_ENCODER = {
    "offsets_unit": "HOURS",
    "offsets_sign": "SIGNED_FROM_T",
    "last_offset_exactly_zero": True,
    "adjacent_delta_unit": "HOURS",
    "adjacent_delta_first_slot": 0,
    "closure_crossed_first_slot": 0,
    "closure_crossed_values": [0, 1],
    "model_applies_division_by": 24,
    "model_requirement": ("sinusoidal position code evaluated on ACTUAL signed offsets (never "
                          "ordinal slot indices), then two context channels appended before the "
                          "core: adjacent elapsed-hours/24 and closure-crossed (0/1)"),
    "value_channels": "19 branches x 4 filters = 76 encoded value channels plus the 2 context channels",
    "dtypes": {"offsets_hours": "float64", "adjacent_delta_hours": "float64",
               "closure_crossed": "int64", "slot_instants": "int64"},
}
MARKET_ARRAYS = ("row_id", "x", "origin_price", "target_price", "mask",
                 "offsets_hours", "adjacent_delta_hours", "closure_crossed", "slot_instants")
MARKET_AUX = ("row_id", "row_ids", "entry_mask", "week_ids")


def market_design(design: Mapping[str, Any]) -> Dict[str, Any]:
    """Typed MG design (data-lane sealed inputs) derived from a validated elapsed design.

    Market spacing replaces the elapsed ``input`` block entirely: no
    ``history_span_hours`` survives, so a fixed elapsed span can never be read
    out of this document. The runtime/model seal is added by its owner lane.
    """
    base = data.validate_design(design)
    spec = MarketWindowSpec.from_design(base)
    doc: Dict[str, Any] = {
        "schema": DESIGN_SCHEMA,
        "status": "DATA_LANE_SEALED_INPUTS_NOT_A_RUNTIME_CONFIG",
        "experiment_id": ("I6-DMO-SHORT-MG2-STATIC-R0" if spec.spacing_market_hours == 2
                          else "I6-DMO-LONG-MG12-STATIC-R0"),
        "derives_from_design_sha256": provenance.canonical_sha256(base),
        "asset": base["asset"], "seed": base["seed"], "repetitions": base["repetitions"],
        "evaluation_mode": base["evaluation_mode"],
        "retraining_frequency": base["retraining_frequency"],
        "input_sampling": {
            "rule": market_rule_name(spec), "observations": spec.observations,
            "spacing_open_market_hours": spec.spacing_market_hours,
            "span_open_market_hours": spec.span_market_hours,
            "elapsed_span_hours_fixed": False,
            "include_origin": True, "issuance_hours": 1, "value_mode": "RAW",
            "grid": ("expected whole-hour bar-END instants in (open, close] of the versioned "
                     "calendar; never dataset row counts"),
            "exactness": ("exact finite retained observation required at every expected endpoint; "
                          "unplanned open-session gaps are counted exclusions"),
            "no_asof_no_smoothing_no_interpolation_no_synthetic_prices": True,
        },
        "horizons_hours": list(data.horizons(base)),
        "targets": "exact ELAPSED t+h, unmodified, under the original C1 business masks",
        "naive": "p(t) vs p(t+h) in ELAPSED hours on exactly the model's rows, original price",
        "splits": json.loads(json.dumps(base["splits"])),
        "business": json.loads(json.dumps(base["business"])),
        "feature_policy": json.loads(json.dumps(base["feature_policy"])),
        "features": list(provenance.MANIFEST),
        "feature_manifest_sha256": provenance.EXPECTED_MANIFEST_SHA256,
        "availability_contract_sha256": provenance.EXPECTED_CONTRACT_SHA256,
        "preprocessing": "scalers and loss scales fit on FIT rows only; selection is not redone",
        "model": json.loads(json.dumps(base["model"])),
        "training": json.loads(json.dumps(base["training"])),
        "metrics": json.loads(json.dumps(base["metrics"])),
        "time_encoder": dict(TIME_ENCODER),
        "support_study_sha256": STUDY_SHA256,
        "support_study_reference_counts_are_not_quotas": True,
    }
    doc["data_identity_sha256"] = data_identity_sha256(doc)
    validate_market_design(doc)
    return doc


_IDENTITY_KEYS = ("experiment_id", "asset", "input_sampling", "horizons_hours", "targets",
                  "naive", "splits", "business", "feature_policy", "features",
                  "feature_manifest_sha256", "availability_contract_sha256", "preprocessing",
                  "time_encoder")


def data_identity_sha256(doc: Mapping[str, Any]) -> str:
    """Identity of everything that determines the materialized arrays."""
    return provenance.canonical_sha256({k: doc[k] for k in _IDENTITY_KEYS})


def validate_market_design(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """Typed validation; refuses contradictory fixed-elapsed versus market-time fields."""
    if not isinstance(doc, Mapping) or doc.get("schema") != DESIGN_SCHEMA:
        raise data.DataContractError("MARKET_DESIGN_INVALID", f"schema must be {DESIGN_SCHEMA}")
    for key in ("input", "history_span_hours", "spacing_hours"):
        if key in doc:
            raise data.DataContractError(
                "CONTRADICTORY_TIME_FIELDS",
                f"{key!r} is an elapsed-grid field and cannot appear in a market-time design")
    s = doc.get("input_sampling")
    if not isinstance(s, Mapping):
        raise data.DataContractError("MARKET_DESIGN_INVALID", "input_sampling is mandatory")
    if set(s) & {"history_span_hours", "spacing_hours"}:
        raise data.DataContractError("CONTRADICTORY_TIME_FIELDS",
                                     "elapsed span/spacing inside input_sampling")
    if s.get("elapsed_span_hours_fixed") is not False:
        raise data.DataContractError("CONTRADICTORY_TIME_FIELDS",
                                     "elapsed spans vary across closures and DST")
    if s.get("observations") != 13 or s.get("spacing_open_market_hours") not in (2, 12):
        raise data.DataContractError("MARKET_DESIGN_INVALID", "13 slots, spacing 2 or 12")
    if s.get("span_open_market_hours") != s["spacing_open_market_hours"] * 12:
        raise data.DataContractError("MARKET_DESIGN_INVALID", "span must be spacing * 12")
    if s.get("rule") != f"MG{s['spacing_open_market_hours']}_MARKET_EXACT":
        raise data.DataContractError("MARKET_DESIGN_INVALID", "rule disagrees with spacing")
    if doc.get("features") != list(provenance.MANIFEST) or \
            doc.get("feature_manifest_sha256") != provenance.EXPECTED_MANIFEST_SHA256:
        raise data.DataContractError("MARKET_DESIGN_INVALID", "19-feature manifest required")
    if doc.get("time_encoder", {}).get("model_applies_division_by") != 24:
        raise data.DataContractError("MARKET_DESIGN_INVALID", "time encoder contract altered")
    if "data_identity_sha256" in doc and doc["data_identity_sha256"] != data_identity_sha256(doc):
        raise data.DataContractError("MARKET_DESIGN_SEAL", "data identity does not verify")
    return dict(doc)


def _elapsed_design(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """The elapsed-schema document the audited data loaders consume (inputs excepted)."""
    s = doc["input_sampling"]
    base = {
        "schema": "i6dmo.design.v1", "status": "DESIGN_NOT_RUNTIME_CONFIG",
        "experiment_id": doc["experiment_id"], "asset": doc["asset"], "seed": doc["seed"],
        "repetitions": doc["repetitions"], "evaluation_mode": doc["evaluation_mode"],
        "retraining_frequency": doc["retraining_frequency"],
        "input": {"value_mode": "RAW", "history_span_hours": s["span_open_market_hours"],
                  "spacing_hours": s["spacing_open_market_hours"], "observations": 13,
                  "include_origin": True, "issuance_hours": 1, "smoothing": False,
                  "require_exact_timestamps": True},
        "horizons_hours": list(doc["horizons_hours"]), "feature_policy": doc["feature_policy"],
        "splits": doc["splits"], "business": doc["business"], "model": doc["model"],
        "training": doc["training"], "metrics": doc["metrics"],
        "execution": {"first_scientific_gpu": "preferred_5090", "one_heavy_job_per_host": True,
                      "secondary_gpu": "4090_after_admission",
                      "preferred_aux_gpu": "disabled_for_parallel_fits",
                      "adopt_first_full_fit_as_cost_measurement": True, "cpu_fallback": False,
                      "new_scientific_identity": True, "resume_retired_doin_campaign": False},
    }
    return data.validate_design(base)


def authenticate_market_bundle(manifest: Mapping[str, Any], design: Mapping[str, Any]) -> str:
    """Refuse a bundle whose schema or market rule differs from the design's."""
    validate_market_design(design)
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise data.DataContractError("MARKET_BUNDLE_SCHEMA", str(manifest.get("schema")))
    expected = design["input_sampling"]["rule"]
    got = (manifest.get("input_sampling") or {}).get("rule")
    if got != expected:
        raise data.DataContractError("MARKET_RULE_MISMATCH", f"{got!r} != {expected!r}")
    if manifest.get("data_identity_sha256") != design.get("data_identity_sha256"):
        raise data.DataContractError("MARKET_DATA_IDENTITY_MISMATCH",
                                     str(manifest.get("data_identity_sha256")))
    return expected


def time_inputs(arrays: MarketSplitArrays) -> Dict[str, np.ndarray]:
    """Per-window model inputs: signed offsets (h), adjacent deltas (h), closure flags."""
    off = arrays.actual_offsets_s.astype(np.float64) / HOUR
    adj = np.zeros_like(off)
    adj[:, 1:] = np.diff(off, axis=1)
    flags = arrays.closure_crossed.astype(np.int64)
    flags[:, 0] = 0
    if off.size and not np.all(off[:, -1] == 0):
        raise data.DataContractError("INTERNAL_OFFSET_NOT_ZERO", "the last offset must be 0")
    return {"offsets_hours": off, "adjacent_delta_hours": adj, "closure_crossed": flags,
            "slot_instants": arrays.slot_instants.astype(np.int64)}


def market_fit_statistics(bundle: data.InputBundle, table: MarketOriginTable) -> dict:
    """FIT-only feature mean/std over the slot source rows and per-head fit naive MAE."""
    import math
    sel = np.flatnonzero(table.materialized("FIT"))
    if not len(sel):
        return {"status": "NO_FIT_ROWS"}
    bars = np.unique(table.slot_index[sel].reshape(-1))
    xs = bundle.x[bars]
    pos = {int(v): i for i, v in enumerate(bundle.ts)}
    naive_mae, unsupported = {}, []
    for j, h in enumerate(table.horizons):
        rows = sel[table.label_ok[sel, j]]
        if not len(rows):
            naive_mae[str(h)] = None
            unsupported.append(h)
            continue
        origins = table.origin_ts[rows]
        p0 = bundle.price[[pos[int(t)] for t in origins]]
        p1 = bundle.price[[pos[int(t) + h * HOUR] for t in origins]]
        value = math.fsum(np.abs(p1 - p0).tolist()) / len(rows)
        naive_mae[str(h)] = value
        if value == 0.0:
            unsupported.append(h)
    std = xs.std(axis=0)
    zero = [f for f, v in zip(bundle.features, std) if v == 0.0]
    status = "UNSUPPORTED_HEAD" if unsupported else ("ZERO_SCALE" if zero else "OK")
    return {"status": status,
            "basis": "FIT rows only; unique exact slot source bars; std ddof=0",
            "fit_rows": int(len(sel)), "fit_window_bars": int(len(bars)),
            "feature_mean": {f: float(v) for f, v in zip(bundle.features, xs.mean(axis=0))},
            "feature_std": {f: float(v) for f, v in zip(bundle.features, std)},
            "zero_scale_features": zero, "naive_mae_fit": naive_mae,
            "unsupported_heads": unsupported}


def _reconciliation(rule: str, rows: Mapping[str, int], shared: Mapping[str, int]) -> dict:
    reference = STUDY_REFERENCE[rule]
    per_split, status = {}, "RECONCILED"
    for split in ("fit", "inner", "validation"):
        got, want = int(rows[split]), int(reference[split])
        per_split[split] = {"materialized": got, "study_rows": want, "difference": got - want,
                            "status": "MATCH" if got == want else "DIFFERENCE_RETAINED"}
        if got != want:
            status = "DIFFERENCE_RETAINED_BEFORE_ANY_FIT"
    return {"study_sha256": STUDY_SHA256, "rule": rule,
            "basis": "materialized rows (input_ok and >= 1 valid head) per split",
            "per_split": per_split, "status": status,
            "shared_usable_origins": dict(shared),
            "shared_study_reference": dict(STUDY_REFERENCE["shared"]),
            "counts_are_not_quotas": True,
            "note": ("a difference is retained and reported, never closed by filling gaps, "
                     "ASOF substitution or relabelling closures")}


def build_market_bundle(input_dir: str | Path, market_doc: Mapping[str, Any],
                        out_dir: str | Path, *, code_commit: Optional[str] = None,
                        shared_with: Optional[str | Path] = None) -> dict:
    """Materialize MG2/MG12 training bundles with sealed identities and reconciliation."""
    doc = validate_market_design(market_doc)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise data.DataContractError("BUNDLE_EXISTS", "a frozen bundle is never overwritten")
    elapsed = _elapsed_design(doc)
    bundle = data.load_input(input_dir, elapsed)
    table = build_market_table(bundle, elapsed)
    gate = provenance.availability_gate(list(bundle.features))
    out.mkdir(parents=True, exist_ok=True)
    splits_json, files, rows, usable = {}, {}, {}, {}
    for split, bounds in zip(data.SPLITS, data.split_bounds(elapsed)):
        arrays = materialize(bundle, table, split)
        key = split.lower()
        model_npz = {"row_id": arrays.origin_ts.astype(np.int64), "x": arrays.X,
                     "origin_price": arrays.price_origin, "target_price": arrays.y_price,
                     "mask": arrays.mask, **time_inputs(arrays)}
        aux_npz = {"row_id": arrays.origin_ts.astype(np.int64), "row_ids": arrays.row_ids,
                   "entry_mask": arrays.entry_mask, "week_ids": arrays.week_ids}
        name, aux = f"{key}.npz", f"{key}_aux.npz"
        np.savez(out / name, **model_npz)
        np.savez(out / aux, **aux_npz)
        files[name] = provenance.file_sha256(out / name)
        files[aux] = provenance.file_sha256(out / aux)
        rows[key] = int(len(arrays.row_ids))
        usable[key] = set(int(v) for v in arrays.origin_ts)
        splits_json[key] = {
            **bounds.to_json(), "path": name, "aux_path": aux, "n_rows": rows[key],
            "row_ids_sha256": provenance.canonical_sha256(arrays.row_ids.tolist()),
            "arrays_sha256": provenance.array_sha256(model_npz),
            "aux_sha256": provenance.array_sha256(aux_npz),
            "slot_instants_sha256": provenance.array_sha256(
                {"slot_instants": model_npz["slot_instants"]}),
            "offsets_sha256": provenance.array_sha256(
                {"offsets_hours": model_npz["offsets_hours"],
                 "adjacent_delta_hours": model_npz["adjacent_delta_hours"],
                 "closure_crossed": model_npz["closure_crossed"]}),
            "head_valid": {str(h): int(arrays.mask[:, j].sum())
                           for j, h in enumerate(arrays.horizons)},
            "head_entry_valid": {str(h): int(arrays.entry_mask[:, j].sum())
                                 for j, h in enumerate(arrays.horizons)},
            "counts": json.loads(json.dumps(table.counts[split])),
            "elapsed_span_hours_distribution": table.counts[split]["elapsed_span_hours_distribution"],
            "first_origin": data.iso(arrays.origin_ts.min()) if rows[key] else None,
            "last_origin": data.iso(arrays.origin_ts.max()) if rows[key] else None,
        }
    shared = {k: len(v) for k, v in usable.items()}
    if shared_with is not None:
        other = json.loads((Path(shared_with) / "BUNDLE.json").read_text())
        for key in rows:
            with np.load(Path(shared_with) / other["splits"][key]["path"],
                         allow_pickle=False) as z:
                shared[key] = len(usable[key] & set(int(v) for v in z["row_id"]))
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "status": "TRAINING_ADAPTER_BUNDLE_NO_FIT_HAS_RUN",
        "experiment_id": doc["experiment_id"],
        "input_sampling": dict(doc["input_sampling"]),
        "offsets_open_market_hours": list(range(-doc["input_sampling"]["span_open_market_hours"],
                                                1, doc["input_sampling"]["spacing_open_market_hours"])),
        "elapsed_offsets_vary_per_row": True,
        "per_row_time_inputs": dict(TIME_ENCODER),
        "model_arrays": list(MARKET_ARRAYS), "aux_arrays": list(MARKET_AUX),
        "horizons_hours": list(doc["horizons_hours"]),
        "targets": doc["targets"], "naive": doc["naive"],
        "design_sha256": provenance.canonical_sha256(doc),
        "data_identity_sha256": doc["data_identity_sha256"],
        "derives_from_design_sha256": doc["derives_from_design_sha256"],
        "features": list(bundle.features),
        "feature_manifest_sha256": gate["manifest_sha256"],
        "availability_contract_sha256": gate["contract_sha256"],
        "availability_gate": gate, "producer_status": provenance.producer_status(),
        "confirmatory_test_blocked": gate["confirmatory_test_blocked"],
        "calendar": {**{k: table.identity[k] for k in ("calendar_id", "calendar_version")},
                     "status": table.identity["calendar_status"],
                     "sha256": bundle.files.get("CALENDAR.json")},
        "source_clock": {k: bundle.clock.get(k) for k in
                         ("status", "timestamp_semantics", "availability_policy", "clock_policy")},
        "inputs": {name: {"name": name, "sha256": digest}
                   for name, digest in bundle.files.items()},
        "history_provenance": data.history_provenance(bundle, data.build_origin_table(
            bundle, elapsed)) if False else {"earliest_consumed_bar": data.iso(
                int(bundle.ts[table.slot_index[table.slot_index >= 0].min()]))},
        "splits": splits_json,
        "fit_statistics": market_fit_statistics(bundle, table),
        "support_hashes": {"study_sha256": STUDY_SHA256,
                           "row_ids": {k: v["row_ids_sha256"] for k, v in splits_json.items()},
                           "arrays": {k: v["arrays_sha256"] for k, v in splits_json.items()},
                           "slot_instants": {k: v["slot_instants_sha256"]
                                             for k, v in splits_json.items()},
                           "offsets": {k: v["offsets_sha256"] for k, v in splits_json.items()}},
        "reconciliation": _reconciliation(doc["input_sampling"]["rule"], rows, shared),
        "test_materialized": False,
        "development_selection_used_2024": True,
        "selection_redone": False,
        "code": {"files_sha256": data._code_digests(), "commit": code_commit,
                 "market_grid_sha256": provenance.file_sha256(Path(__file__))},
    }
    manifest["bundle_sha256"] = data._manifest_digest(manifest)
    for info in splits_json.values():
        info["sha256"] = files[info["path"]]
    manifest["files"] = files
    (out / "BUNDLE.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def load_market_manifest(out_dir: str | Path) -> dict:
    data.verify_bundle(out_dir)
    return json.loads((Path(out_dir) / "BUNDLE.json").read_text())


def load_market_split(out_dir: str | Path, split: str, *,
                      design: Optional[Mapping[str, Any]] = None) -> Dict[str, np.ndarray]:
    """Verified read of one market split (model + aux arrays), optionally authenticated."""
    manifest = load_market_manifest(out_dir)
    if design is not None:
        authenticate_market_bundle(manifest, design)
    info = manifest["splits"][split.lower()]
    merged: Dict[str, np.ndarray] = {}
    for key in ("path", "aux_path"):
        with np.load(Path(out_dir) / info[key], allow_pickle=False) as z:
            merged.update({k: z[k] for k in z.files})
    return merged


def paired_metrics(out_dir: str | Path, split: str, predictions: np.ndarray, *,
                   input_dir: str | Path, checkpoint: str = "unfitted-contract-check") -> dict:
    """C2 paired metrics on exactly the bundle's rows, elapsed horizons, original price."""
    from tools.i6dmo import naive as naive_mod
    manifest = load_market_manifest(out_dir)
    arrays = load_market_split(out_dir, split)
    info = manifest["splits"][split.lower()]

    class _A:
        pass
    view = _A()
    view.split = split.upper()
    view.horizons = tuple(manifest["horizons_hours"])
    view.features = tuple(manifest["features"])
    view.mask = arrays["mask"]
    view.entry_mask = arrays["entry_mask"]
    view.y_price = arrays["target_price"]
    view.price_origin = arrays["origin_price"]
    view.origin_ts = arrays["row_id"]
    view.row_ids = arrays["row_ids"]
    view.week_ids = arrays["week_ids"]
    view.expected_weeks = tuple(sorted(set(str(w) for w in arrays["week_ids"])))
    view.calendar_id = manifest["calendar"]["calendar_id"]
    design = {"experiment_id": manifest["experiment_id"],
              "horizons_hours": manifest["horizons_hours"],
              "data_identity_sha256": manifest["data_identity_sha256"]}
    report = naive_mod.paired_metrics(view, predictions, design, input_dir=input_dir,
                                      checkpoint=checkpoint,
                                      sealed_at="2026-10-15T00:00:00+00:00")
    report["bundle_sha256"] = manifest["bundle_sha256"]
    report["rule"] = manifest["input_sampling"]["rule"]
    report["row_ids_sha256"] = info["row_ids_sha256"]
    return report
