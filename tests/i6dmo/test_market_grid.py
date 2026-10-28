"""Calendar MARKET-TIME sampling candidates (order 85ea134e section 2): MG2 / MG12.

Synthetic fixtures only. This is a support study contract: no fit, no GPU, no
TEST year, no rule declared optimal. Slots are EXPECTED open-market hour
endpoints generated from the versioned ASSUMED calendar; retained observations
must exist EXACTLY there. Closures vary the elapsed span; vendor gaps in an
open session are counted exclusions, never compressed closures.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from tests.i6dmo import _synth
from tools.i6dmo import data, market_grid

UTC = timezone.utc
H = 3600
SHORT_HEADS = (2, 3, 4, 5, 6, 12)
LONG_HEADS = (18, 24, 36, 48, 60, 72)


def long_design():
    return data.long_data_view(_synth.load_design())


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    root = tmp_path_factory.mktemp("mg")
    return _synth.make_input(root / "input",
                             missing={_synth.epoch("2022-06-15T13:00:00+00:00")},
                             nonfinite=[(_synth.epoch("2021-09-08T09:00:00+00:00"), 5)])


@pytest.fixture(scope="module")
def short(src):
    design = _synth.load_design()
    bundle = data.load_input(src, design)
    return {"design": design, "bundle": bundle,
            "table": market_grid.build_market_table(bundle, design)}


@pytest.fixture(scope="module")
def lng(src):
    design = long_design()
    bundle = data.load_input(src, design)
    return {"design": design, "bundle": bundle,
            "table": market_grid.build_market_table(bundle, design)}


def cal_weeks(src):
    return json.loads((src / "CALENDAR.json").read_text())["market_calendar"]["weeks"]


def test_rule_names_and_market_span_labelling(short, lng):
    assert short["table"].rule == "MG2_MARKET_EXACT"
    assert lng["table"].rule == "MG12_MARKET_EXACT"
    for t, span in ((short["table"], 24), (lng["table"], 144)):
        assert t.spec.span_market_hours == span
        assert t.spec.observations == 13
        assert t.identity["span_unit"] == "OPEN_MARKET_HOURS"
        assert t.identity["elapsed_span_hours_fixed"] is False
    assert short["table"].identity["rule"] != "DS2_EXACT"
    assert short["table"].identity["calendar_status"] == "ASSUMED_NOT_BROKER"


def test_expected_grid_is_calendar_hours_not_dataset_rows(src, short):
    weeks = cal_weeks(src)
    grid = market_grid.expected_market_hours(short["bundle"])
    inside = set(grid.tolist())
    for w in weeks[:40]:
        o, c = _synth.epoch(w["market_open"]), _synth.epoch(w["market_close"])
        for t in range(o + H, c + 1, H):
            assert t in inside
        assert o not in inside                          # bar-end convention: (open, close]
        assert c + H not in inside                      # closure hours are not expected
    hole = _synth.epoch("2022-06-15T13:00:00+00:00")
    assert hole in inside                               # vendor gap stays an EXPECTED hour


def test_monday_origin_crosses_weekend_with_exact_friday_slots(src, short):
    weeks = cal_weeks(src)
    i = next(k for k, w in enumerate(weeks) if w["week_id"] == "2021-02-07")
    prev_close = _synth.epoch(weeks[i - 1]["market_close"])
    t = _synth.epoch(weeks[i]["market_open"]) + 12 * H  # Monday, 12 market hours in
    rec = short["table"].record(t)
    assert rec.input_ok is True                         # DS2_EXACT refused this origin
    slots = rec.slot_instants
    assert len(slots) == 13 and slots[-1] == t
    # bar-END grid: 12 open-market hours back from open+12h is the previous Friday close
    assert slots[6] == prev_close
    assert slots[5] == prev_close - 2 * H               # earlier slots stay on Friday
    gap = _synth.epoch(weeks[i]["market_open"]) - prev_close
    assert rec.closure_crossed[7] is True and sum(rec.closure_crossed) == 1
    assert rec.elapsed_span_s == 24 * H + gap           # varies: never a fixed 24 h label
    assert list(rec.actual_offsets_s) == [s - t for s in slots]
    arrays_ok = all(s in set(int(v) for v in short["bundle"].ts) for s in slots)
    assert arrays_ok


def test_dst_closures_change_elapsed_span(src, short):
    weeks = cal_weeks(src)
    spans = {}
    for wid in ("2021-03-14", "2021-11-07"):            # spring-forward / fall-back
        i = next(k for k, w in enumerate(weeks) if w["week_id"] == wid)
        gap = _synth.epoch(weeks[i]["market_open"]) - _synth.epoch(weeks[i - 1]["market_close"])
        t = _synth.epoch(weeks[i]["market_open"]) + 12 * H
        rec = short["table"].record(t)
        assert rec.input_ok is True
        assert rec.elapsed_span_s == 24 * H + gap
        spans[wid] = rec.elapsed_span_s
    assert spans["2021-03-14"] != spans["2021-11-07"]   # 47 h vs 49 h closures


def test_unexpected_midweek_gap_is_counted_exclusion_never_compressed(src, short):
    hole = _synth.epoch("2022-06-15T13:00:00+00:00")
    table = short["table"]
    rec = table.record(hole + 2 * H)                    # its t-2 market hours slot IS the hole
    assert rec.input_ok is False
    assert rec.input_reason == "INPUT_SLOT_NOT_OBSERVED"
    # the grid did NOT slide to the previous observed bar
    assert (hole) in rec.slot_instants
    counts = table.counts["FIT"]
    assert counts["origins"]["INPUT_SLOT_NOT_OBSERVED"] >= 1
    assert counts["closures_vs_gaps"]["unplanned_missing_expected_hours"] >= 1
    nf = _synth.epoch("2021-09-08T09:00:00+00:00")
    rec = table.record(nf + 2 * H)
    assert rec.input_ok is False and rec.input_reason == "INPUT_VALUE_NOT_FINITE"


def test_current_t_and_t_plus_1_perturbations(src, short, tmp_path):
    design = short["design"]
    arrays = market_grid.materialize(short["bundle"], short["table"], "FIT")
    r = len(arrays.origin_ts) // 3
    t = int(arrays.origin_ts[r])
    pos = np.searchsorted(short["bundle"].ts, t)
    for shift, must_change in ((0, True), (1, False)):  # perturb t itself, then t+1h only
        p = _synth.make_input(tmp_path / f"p{shift}",
                              missing={_synth.epoch("2022-06-15T13:00:00+00:00")},
                              nonfinite=[(_synth.epoch("2021-09-08T09:00:00+00:00"), 5)])
        with np.load(p / "INPUT.npz") as z:
            ts, x = z["ts"], z["x"].copy()
        x[pos + shift] += 5.0
        _synth.rewrite_input(p, x=x)
        b2 = data.load_input(p, design)
        a2 = market_grid.materialize(b2, market_grid.build_market_table(b2, design), "FIT")
        r2 = list(a2.row_ids).index(arrays.row_ids[r])
        changed = not np.array_equal(a2.X[r2], arrays.X[r])
        assert changed is must_change


def test_long_candidate_has_support_where_ds12_exact_has_none(src, lng):
    table = lng["table"]
    for split in ("FIT", "INNER", "VALIDATION"):
        assert table.counts[split]["rows_materialized"] > 0
        for h in LONG_HEADS:
            assert table.counts[split]["heads"][str(h)]["label_valid"] > 0
    exact = data.build_origin_table(lng["bundle"], lng["design"])
    assert all(exact.counts[s]["rows_materialized"] == 0 for s in ("FIT", "INNER", "VALIDATION"))
    arrays = market_grid.materialize(lng["bundle"], table, "FIT")
    assert (np.diff(arrays.actual_offsets_s[:, :, 0] if arrays.actual_offsets_s.ndim == 3
                    else arrays.actual_offsets_s) > 0).all()
    assert arrays.actual_offsets_s.shape[1] == 13
    assert (arrays.actual_offsets_s[:, -1] == 0).all()  # current t retained
    spans = arrays.actual_offsets_s[:, -1] - arrays.actual_offsets_s[:, 0]
    assert (spans >= 144 * H).all() and (spans > 144 * H).any()


def test_targets_and_masks_are_the_original_business_masks(src, short):
    """Market-time sampling changes INPUTS only; per-head C1 masks and exact elapsed targets stay."""
    from tools.i6n_contracts import eligibility as c1
    bundle, table = short["bundle"], short["table"]
    cal = json.loads((src / "CALENDAR.json").read_text())
    ref, weeks = c1.load_calendar(cal)
    observed = c1.ObservedTargets({datetime.fromtimestamp(int(v), UTC):
                                   datetime.fromtimestamp(int(v), UTC) for v in bundle.ts})
    splits = data.split_bounds(short["design"])
    checked = 0
    for wid in ("2022-05-15", "2024-03-10"):
        week = next(w for w in weeks if w.week_id == wid)
        for t in table.origins_in_week(wid)[:40]:
            rec = table.record(t)
            t_dt = datetime.fromtimestamp(t, UTC)
            split = next(s for s in splits if s.start <= t_dt < s.end)
            ds = c1.evaluate_heads(t_dt, SHORT_HEADS, c1.Role.MANAGEMENT, week,
                                   input_available=rec.input_ok, observed_targets=observed,
                                   split=split)
            assert tuple(d.label_eligible for d in ds) == tuple(rec.label_ok)
            checked += 1
    assert checked > 40
    arrays = market_grid.materialize(bundle, table, "VALIDATION")
    pos = {int(v): i for i, v in enumerate(bundle.ts)}
    for j, h in enumerate(SHORT_HEADS):
        rows = np.flatnonzero(arrays.mask[:, j])[:20]
        for r in rows:
            q = int(arrays.origin_ts[r]) + h * H        # exact ELAPSED target, not market time
            assert arrays.y_price[r, j] == bundle.price[pos[q]]


def test_support_study_report_structure(src, tmp_path):
    report = market_grid.support_study(src, _synth.load_design())
    assert report["status"] == "SUPPORT_STUDY_NOT_A_PREDICTIVE_MEASUREMENT"
    assert set(report["candidates"]) == {"MG2_MARKET_EXACT", "MG12_MARKET_EXACT"}
    assert set(report["elapsed_grid_reference"]) == {"DS2_EXACT", "DS12_EXACT"}
    assert "optimal" not in json.dumps(report).lower()
    for name, cand in report["candidates"].items():
        for split in ("FIT", "INNER", "VALIDATION"):
            c = cand["splits"][split]
            assert {"origins", "heads", "weekly", "closures_vs_gaps",
                    "elapsed_span_hours_distribution"} <= set(c)
            for head in c["heads"].values():
                assert {"label_valid", "entry_valid", "reasons"} <= set(head)
    shared = report["shared_usable_origins"]
    for split in ("FIT", "INNER", "VALIDATION"):
        assert shared[split]["both_market_candidates"] <= min(
            report["candidates"]["MG2_MARKET_EXACT"]["splits"][split]["rows_materialized"],
            report["candidates"]["MG12_MARKET_EXACT"]["splits"][split]["rows_materialized"])
    assert report["source_clock"]["timestamp_semantics"] == "UTC_BAR_END"
    assert report["calendar"]["status"] == "ASSUMED_NOT_BROKER"
    assert report["examples"]["monday_weekend_crossing"]["closure_crossed_slots"] >= 1
    assert report["examples"]["dst"]["spans_differ"] is True
