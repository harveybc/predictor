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


def build_market_table(bundle: data.InputBundle, design: Mapping[str, Any]) -> MarketOriginTable:
    """Evaluate every candidate origin on the expected market-hour grid + C1 masks."""
    design = data.validate_design(design)
    spec = MarketWindowSpec.from_design(design)
    rule = market_rule_name(spec)
    splits = data.split_bounds(design)
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
                  "rows_materialized": 0} for s in data.SPLITS}
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
