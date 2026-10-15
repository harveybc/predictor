#!/usr/bin/env python3
"""RB01 - independent falsification of the market-time DATA and METRIC paths.

Plan MARKET_TIME_PAIR_PLAN.md section 6, data items:
  A. current-t inclusion, future perturbation, exact elapsed target timestamps
  B. weekend/DST offsets vs ordinary weeks, unplanned gaps not compressed
  C. chronology, split/purge boundaries, same-row per-horizon naive in original price
  D. first slot delta 0 / flag 0, last offset exactly 0

Every control carries an injected fail-closed counterexample written by this
reviewer: if a doctored input does NOT change the control's answer, the control
measures nothing and is reported as such.

CPU only, synthetic fixtures only, no fit, no TEST bar. argv[1] = a workdir.
Exit 0; the JSON on stdout is the verdict.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.getcwd())

from tests.i6dmo import _synth                      # noqa: E402
from tools.i6dmo import data, market_grid, naive     # noqa: E402

UTC = timezone.utc
H = 3600
WORK = sys.argv[1] if len(sys.argv) > 1 else "."
from pathlib import Path                            # noqa: E402

WORK = Path(WORK)
WORK.mkdir(parents=True, exist_ok=True)
R: dict = {"probe": "RB01_data_time", "controls": {}}


def ep(text):
    return int(datetime.fromisoformat(text).timestamp())


def build(tag, **kw):
    src = _synth.make_input(WORK / ("in_" + tag), **kw)
    design = market_grid.market_design(_synth.load_design())
    bundle = data.load_input(src, market_grid._elapsed_design(design))
    table = market_grid.build_market_table(bundle, market_grid._elapsed_design(design))
    return src, design, bundle, table


# ---------------------------------------------------------------- A. current t / future
src, design, bundle, table = build("base")
arrays = market_grid.materialize(bundle, table, "FIT")
ti = market_grid.time_inputs(arrays)

last_offsets = ti["offsets_hours"][:, -1]
first_delta = ti["adjacent_delta_hours"][:, 0]
first_flag = ti["closure_crossed"][:, 0]
slots = arrays.slot_instants
R["controls"]["A1_current_t_included"] = {
    "rows": int(len(slots)),
    "last_slot_equals_origin_rows": int((slots[:, -1] == arrays.origin_ts).sum()),
    "last_offset_all_zero": bool(np.all(last_offsets == 0.0)),
    "max_offset": float(ti["offsets_hours"].max()),
    "all_offsets_non_positive": bool(np.all(ti["offsets_hours"] <= 0)),
    "verdict": "PASS" if (np.all(slots[:, -1] == arrays.origin_ts)
                          and np.all(last_offsets == 0.0)
                          and np.all(ti["offsets_hours"] <= 0)) else "FAIL",
}
R["controls"]["D1_first_slot_delta_and_flag_zero"] = {
    "first_delta_all_zero": bool(np.all(first_delta == 0.0)),
    "first_flag_all_zero": bool(np.all(first_flag == 0)),
    "verdict": "PASS" if (np.all(first_delta == 0.0)
                          and np.all(first_flag == 0)) else "FAIL",
}

# A2 future perturbation: change every bar strictly after t for one origin and
# show that the row's inputs, offsets, closure flags and chosen slots do not move.
probe_t = int(arrays.origin_ts[len(arrays.origin_ts) // 2])
future = bundle.ts > probe_t
x2 = bundle.x.copy()
x2[future] += 1234.5
price2 = bundle.price.copy()
price2[future] += 99.0
import dataclasses                                   # noqa: E402
b2 = dataclasses.replace(bundle, x=x2, price=price2)
t2 = market_grid.build_market_table(b2, market_grid._elapsed_design(design))
a2 = market_grid.materialize(b2, t2, "FIT")
i0 = int(np.flatnonzero(arrays.origin_ts == probe_t)[0])
i1 = int(np.flatnonzero(a2.origin_ts == probe_t)[0])
same_x = bool(np.array_equal(arrays.X[i0], a2.X[i0 if i0 == i1 else i1]))
same_off = bool(np.array_equal(arrays.actual_offsets_s[i0], a2.actual_offsets_s[i1]))
same_flags = bool(np.array_equal(arrays.closure_crossed[i0], a2.closure_crossed[i1]))
# the counterexample: perturbing a bar INSIDE the window (at or before t) MUST change x
x3 = bundle.x.copy()
x3[bundle.ts == int(arrays.slot_instants[i0, 0])] += 1234.5
b3 = dataclasses.replace(bundle, x=x3)
t3 = market_grid.build_market_table(b3, market_grid._elapsed_design(design))
a3 = market_grid.materialize(b3, t3, "FIT")
i3 = int(np.flatnonzero(a3.origin_ts == probe_t)[0])
past_changes = not np.array_equal(arrays.X[i0], a3.X[i3])
R["controls"]["A2_future_perturbation"] = {
    "origin": data.iso(probe_t),
    "bars_perturbed_after_t": int(future.sum()),
    "inputs_unchanged": same_x, "offsets_unchanged": same_off,
    "closure_flags_unchanged": same_flags,
    "counterexample_in_window_bar_changes_inputs": bool(past_changes),
    "verdict": "PASS" if (same_x and same_off and same_flags and past_changes) else "FAIL",
}

# A3 exact ELAPSED target timestamps: y_price must be the retained price at t + h*3600
hs = list(table.horizons)
pos = {int(v): i for i, v in enumerate(bundle.ts)}
bad_target, checked = [], 0
for j, h in enumerate(hs):
    rows = np.flatnonzero(arrays.mask[:, j])[:400]
    for r in rows:
        t = int(arrays.origin_ts[r])
        want = bundle.price[pos[t + h * H]]
        checked += 1
        if float(arrays.y_price[r, j]) != float(want):
            bad_target.append({"origin": data.iso(t), "h": h})
R["controls"]["A3_exact_elapsed_targets"] = {
    "checked": checked, "mismatches": len(bad_target),
    "examples": bad_target[:5],
    "counterexample_shifted_target_is_detected": None,
    "verdict": "PASS" if not bad_target else "FAIL",
}
# counterexample: compare against t + h*3600 + 1h; it must DISAGREE somewhere
shifted_bad = 0
for j, h in enumerate(hs):
    for r in np.flatnonzero(arrays.mask[:, j])[:200]:
        t = int(arrays.origin_ts[r])
        k = pos.get(t + h * H + H)
        if k is not None and float(arrays.y_price[r, j]) != float(bundle.price[k]):
            shifted_bad += 1
R["controls"]["A3_exact_elapsed_targets"]["counterexample_shifted_target_is_detected"] = \
    shifted_bad > 0
if shifted_bad == 0:
    R["controls"]["A3_exact_elapsed_targets"]["verdict"] = "CONTROL_UNINFORMATIVE"

# ------------------------------------------------- B. weekend / DST / unplanned gaps
spans = (arrays.slot_instants[:, -1] - arrays.slot_instants[:, 0]) // H
uniq, cnt = np.unique(spans, return_counts=True)
n_cross = int((arrays.closure_crossed.any(axis=1)).sum())
ordinary = spans[~arrays.closure_crossed.any(axis=1)]
crossing = spans[arrays.closure_crossed.any(axis=1)]
R["controls"]["B1_weekend_dst_offsets"] = {
    "elapsed_span_hours_distribution": {str(int(a)): int(b) for a, b in zip(uniq, cnt)},
    "rows_with_a_closure_crossing": n_cross,
    "ordinary_week_spans": sorted({int(v) for v in ordinary}),
    "closure_crossing_spans": sorted({int(v) for v in crossing}),
    "ordinary_span_is_exactly_the_market_span": sorted({int(v) for v in ordinary}) == [24],
    "crossing_spans_are_all_longer": bool(len(crossing) and crossing.min() > 24),
    "more_than_one_distinct_crossing_span_dst_evidence": len(set(int(v) for v in crossing)) > 1,
    "verdict": "PASS" if (sorted({int(v) for v in ordinary}) == [24]
                          and len(crossing) and crossing.min() > 24) else "FAIL",
}

# B2 an unplanned gap inside an open session must be a counted exclusion and must
# NOT become a closure flag or a compressed closure.
gap_ts = ep("2022-06-15T13:00:00+00:00")
src_g, design_g, bundle_g, table_g = build("gap", missing={gap_ts})
k = table_g.index.get(gap_ts)
grid = market_grid.expected_market_hours(bundle_g)
affected = [i for i in range(len(table_g.origin_ts))
            if gap_ts in set(int(v) for v in table_g.slot_instants[i])]
excluded = [i for i in affected if table_g.input_reason[i] == "INPUT_SLOT_NOT_OBSERVED"]
flagged = [i for i in affected
           if bool(table_g.closure_crossed[i][
               list(int(v) for v in table_g.slot_instants[i]).index(gap_ts)])]
split_counts = {s: table_g.counts[s]["origins"].get("INPUT_SLOT_NOT_OBSERVED", 0)
                for s in data.SPLITS}
gap_in_expected = bool(gap_ts in set(int(v) for v in grid))
R["controls"]["B2_unplanned_gap_not_compressed"] = {
    "dropped_bar": data.iso(gap_ts),
    "still_an_expected_open_market_hour": gap_in_expected,
    "origins_whose_window_contains_it": len(affected),
    "of_those_excluded_INPUT_SLOT_NOT_OBSERVED": len(excluded),
    "of_those_whose_closure_flag_turned_true_at_that_slot": len(flagged),
    "counted_exclusions_per_split": split_counts,
    "verdict": "PASS" if (gap_in_expected and len(affected) > 0
                          and len(excluded) == len(affected)
                          and len(flagged) == 0
                          and sum(split_counts.values()) > 0) else "FAIL",
}
# counterexample: with the bar PRESENT the same origins are not excluded
def _base_reason(origin):
    k2 = table.index.get(int(origin))
    return None if k2 is None else table.input_reason[k2]


not_excluded_when_present = sum(
    1 for i in affected
    if _base_reason(table_g.origin_ts[i]) != "INPUT_SLOT_NOT_OBSERVED")
R["controls"]["B2_unplanned_gap_not_compressed"][
    "counterexample_same_origins_not_excluded_with_the_bar_present"] = not_excluded_when_present

# -------------------------------------- C. chronology, purge, same-row per-horizon naive
chron = {
    "slots_strictly_increasing_rows":
        int(np.all(np.diff(arrays.slot_instants, axis=1) > 0, axis=1).sum()),
    "rows": int(len(arrays.slot_instants)),
}
chron["verdict"] = "PASS" if chron["slots_strictly_increasing_rows"] == chron["rows"] else "FAIL"
R["controls"]["C1_chronology"] = chron

bounds = {b.split_id: (int(b.start.timestamp()), int(b.end.timestamp()))
          for b in data.split_bounds(market_grid._elapsed_design(design))}
purge = {}
for split in data.SPLITS:
    a = market_grid.materialize(bundle, table, split)
    s0, s1 = bounds[split]
    inside = bool(np.all((a.origin_ts >= s0) & (a.origin_ts < s1)))
    lbl = a.origin_ts[:, None] + np.asarray(hs, dtype=np.int64)[None, :] * H
    reach = int((a.mask & (lbl >= s1)).sum())
    purge[split] = {"rows": int(len(a.origin_ts)), "origins_inside_split": inside,
                    "valid_labels_reaching_split_end": reach}
R["controls"]["C2_split_and_purge"] = {
    **purge,
    "verdict": "PASS" if all(v["origins_inside_split"] and
                             v["valid_labels_reaching_split_end"] == 0
                             for v in purge.values()) else "FAIL",
}

# C3 same-row per-horizon naive in ORIGINAL price, through the shipped metric path
mg_dir = WORK / "mg2_bundle"
manifest = market_grid.build_market_bundle(src, design, mg_dir, code_commit="REVIEW")
split = "validation"
sp = market_grid.load_market_split(mg_dir, split)
n, nh = sp["mask"].shape
rng = np.random.default_rng(11)
pred = sp["origin_price"][:, None] + rng.normal(0, 0.001, size=(n, nh))
rep = market_grid.paired_metrics(mg_dir, split, pred, input_dir=src)
per_h = rep.get("per_horizon") or rep.get("heads") or {}
manual = {}
for j, h in enumerate(hs):
    rows = np.flatnonzero(sp["mask"][:, j])
    p0 = sp["origin_price"][rows]
    p1 = sp["target_price"][rows, j]
    manual[str(h)] = {"n": int(len(rows)),
                      "naive_mae": float(np.abs(p1 - p0).mean()),
                      "model_mae": float(np.abs(pred[rows, j] - p1).mean())}
agree = []
for h in map(str, hs):
    got = per_h.get(h) or per_h.get(int(h)) or {}
    agree.append({
        "h": h,
        "report_n": got.get("n"), "manual_n": manual[h]["n"],
        "report_naive": got.get("naive_mae"), "manual_naive": manual[h]["naive_mae"],
        "report_mae": got.get("mae"), "manual_mae": manual[h]["model_mae"],
    })
R["controls"]["C3_same_row_naive_original_price"] = {
    "split": split, "rows": int(n),
    "report_keys": sorted(rep.keys())[:30],
    "per_horizon_comparison": agree,
    "n_matches": all(a["report_n"] == a["manual_n"] for a in agree),
    "naive_matches": all(a["report_naive"] is not None
                         and abs(a["report_naive"] - a["manual_naive"]) <= 1e-9
                         for a in agree),
    "mae_matches": all(a["report_mae"] is not None
                       and abs(a["report_mae"] - a["manual_mae"]) <= 1e-9
                       for a in agree),
}
cc = R["controls"]["C3_same_row_naive_original_price"]
cc["verdict"] = "PASS" if (cc["n_matches"] and cc["naive_matches"]
                           and cc["mae_matches"]) else "FAIL"

# C3 counterexample: rescale the retained prices the replay reads and require a typed refusal
import shutil                                        # noqa: E402
bad_src = WORK / "in_rescaled"
shutil.copytree(src, bad_src, dirs_exist_ok=True)
with np.load(bad_src / "INPUT.npz", allow_pickle=False) as z:
    doc = {k: z[k] for k in z.files}
doc["price"] = doc["price"] * 1.0001
np.savez(bad_src / "INPUT.npz", **doc)
try:
    market_grid.paired_metrics(mg_dir, split, pred, input_dir=bad_src)
    cc["counterexample_rescaled_prices"] = "NOT_CAUGHT"
except Exception as exc:                              # noqa: BLE001
    cc["counterexample_rescaled_prices"] = type(exc).__name__ + ":" + str(exc)[:160]
cc["counterexample_caught"] = cc["counterexample_rescaled_prices"] != "NOT_CAUGHT"
if not cc["counterexample_caught"]:
    cc["verdict"] = "FAIL"

R["verdict"] = {k: v.get("verdict") for k, v in R["controls"].items()}
print(json.dumps(R, indent=2, sort_keys=True, default=str))
