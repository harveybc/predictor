"""P3 (SC01/SC03/SC04/SC05 data side): INDEPENDENT recomputation from the raw files only.

Reads the frozen population's npz/JSON and the retained INPUT.npz + CALENDAR.json with numpy and
the standard library; it imports NOTHING from tools.i6dmo, so it does not share code with the
adapters under review. No fit, no TEST (the input ends before 2025-01-01; asserted).

Checks:
  * expected market-hour grid from CALENDAR.json weeks (bar-END whole hours in (open, close]):
    every common row's MG1 slots = t and the previous 24 expected instants, MG2m slots = every
    second one (13); weekend/DST gaps follow the calendar (offsets are actual UTC differences);
    closure flags equal (adjacent gap > spacing).
  * exactness: every slot of a common row is an observed input bar; inputs equal INPUT x at
    the slot instants; origin/target prices equal INPUT price at t and t+h; NaN off-mask.
  * masks: the (origin, head) pairs are a subset of both arms' exact support; DEVELOPMENT and
    INNER and FIT are disjoint and inside their sealed bounds; labels t+h < split end.
  * missing open-session bars: for every candidate origin typed INPUT_SLOT_NOT_OBSERVED the
    independently computed expected grid really misses an observed bar.
  * scaler: mean/std (ddof=0) over the union of both arms' FIT slot bars recomputed; row set
    equals SCALER_ROWS.npz; all bars before the INNER start.
  * naive loss scale per horizon recomputed on the common FIT mask.

Run:
  CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 1800s -n i6dmo-review-p3 -- \
    python -I docs/i6n/direct_multi_output_static/matched_short/review/p3_independent.py
"""
import bisect
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402  (paths/redaction only; it does not import tools.i6dmo)

import numpy as np  # noqa: E402

H = 3600
HORIZONS = (2, 3, 4, 5, 6, 12)
SPLITS = ("FIT", "INNER", "DEVELOPMENT")
SPACING = {"mg1": 1, "mg2m": 2}
SLOTS = {"mg1": 25, "mg2m": 13}
TEST_START = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())


def epoch(text):
    return int(datetime.fromisoformat(text).timestamp())


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    pop = C.POPULATION
    body = json.loads((pop / "POPULATION.json").read_text())
    with np.load(C.INPUT / "INPUT.npz", allow_pickle=False) as z:
        ts, x, price = z["ts"], z["x"], z["price"]
    assert int(ts.max()) < TEST_START, "input holds a TEST bar"
    pos = {int(v): i for i, v in enumerate(ts)}
    cal = json.loads((C.INPUT / "CALENDAR.json").read_text())["market_calendar"]
    grid = []
    for w in cal["weeks"]:
        o, c = epoch(w["market_open"]), epoch(w["market_close"])
        first = (o // H + 1) * H
        grid.extend(range(first, c + 1, H))
    grid = np.asarray(sorted(set(grid)), dtype=np.int64)
    gidx = {int(v): i for i, v in enumerate(grid)}
    res = {"input_last_bar": datetime.fromtimestamp(int(ts.max()), timezone.utc).isoformat(),
           "expected_grid_hours": int(len(grid)), "splits": {}}
    bounds = {n: (epoch(body["identities"]["splits"][n]["start"]), epoch(body["identities"]["splits"][n]["end"]))
              for n in SPLITS}
    all_ok = True
    weekend_gaps = {}
    for split in SPLITS:
        sp = npz(pop / f"POPULATION_{split}.npz")
        t = sp["origin_row_ids"]
        head = sp["head_mask"]
        lo, hi = bounds[split]
        r = {"origins": int(len(t)), "counts": {str(h): int(head[:, j].sum()) for j, h in enumerate(HORIZONS)},
             "counts_match_seal": {str(h): int(head[:, j].sum()) for j, h in enumerate(HORIZONS)}
             == C.SEAL["splits"][split]["counts"],
             "origins_in_bounds": bool(((t >= lo) & (t < hi)).all()),
             "labels_before_split_end": bool(((t[:, None] + np.asarray(HORIZONS) * H)[head] < hi).all()),
             "labels_before_test": bool(((t[:, None] + np.asarray(HORIZONS) * H)[head] < TEST_START).all())}
        for arm in ("mg1", "mg2m"):
            a = npz(pop / f"ARM_{arm}_{split}.npz")
            si = a["slot_instants"]
            # independent expected grid: t and the previous expected open-market instants
            k = np.asarray([gidx.get(int(v), -1) for v in t])
            grid_ok = bool((k >= 0).all())
            step = SPACING[arm]
            want = np.stack([grid[k - step * (SLOTS[arm] - 1 - s)] for s in range(SLOTS[arm])], axis=1) \
                if grid_ok and (k - step * (SLOTS[arm] - 1) >= 0).all() else None
            slots_ok = want is not None and np.array_equal(want, si)
            observed = all(int(v) in pos for v in np.unique(si))
            xi = np.vectorize(lambda v: pos[int(v)])(si)
            x_ok = np.array_equal(x[xi], a["x"]) and a["x"].tobytes() == x[xi].tobytes()
            off_ok = np.array_equal(a["offsets_hours"], (si - t[:, None]).astype(np.float64) / H)
            gap = np.diff(si, axis=1) / H
            adj_ok = np.array_equal(a["adjacent_delta_hours"][:, 1:], gap) and not a["adjacent_delta_hours"][:, 0].any()
            clos_ok = np.array_equal(a["closure_crossed"][:, 1:].astype(bool), gap > step) and \
                not a["closure_crossed"][:, 0].any()
            p0_ok = np.array_equal(a["origin_price"], price[[pos[int(v)] for v in t]])
            tgt_ok, nan_ok = True, True
            for j, h in enumerate(HORIZONS):
                m = head[:, j]
                tgt_ok &= np.array_equal(a["target_price"][m, j], price[[pos[int(v) + h * H] for v in t[m]]])
                nan_ok &= bool(np.isnan(a["target_price"][~m, j]).all())
            same_masks = all(np.array_equal(a[key], sp[key]) for key in ("head_mask", "entry_mask", "management_mask"))
            if arm == "mg1":
                for row in range(len(t)):
                    for g in gap[row][gap[row] > 1]:
                        wk = datetime.fromtimestamp(int(si[row][np.flatnonzero(gap[row] > 1)[0] + 1]),
                                                    timezone.utc).date().isoformat()
                        weekend_gaps.setdefault(wk, set()).add(float(g))
            r[arm] = {"expected_grid_slots_equal": bool(slots_ok), "every_slot_observed": bool(observed),
                      "x_equals_input_at_slots_bytes": bool(x_ok), "offsets_actual_utc": bool(off_ok),
                      "adjacent_delta_actual": bool(adj_ok), "closure_flag_equals_gap_gt_spacing": bool(clos_ok),
                      "origin_price_replay": bool(p0_ok), "target_replay_exact_t_plus_h": bool(tgt_ok),
                      "nan_off_mask": bool(nan_ok), "masks_equal_population": bool(same_masks),
                      "last_slot_is_t": bool(np.array_equal(si[:, -1], t))}
            all_ok &= all(r[arm].values())
        a1, a2 = npz(pop / f"ARM_mg1_{split}.npz"), npz(pop / f"ARM_mg2m_{split}.npz")
        r["mg2m_slots_equal_mg1_every_second"] = bool(np.array_equal(a1["slot_instants"][:, ::2], a2["slot_instants"]))
        # exclusions typed INPUT_SLOT_NOT_OBSERVED really miss an expected observed bar
        exc = npz(pop / f"EXCLUSIONS_{split}.npz")
        codes = body["report"]["input_reason_codes"]
        miss_code = codes.index("INPUT_SLOT_NOT_OBSERVED")
        checked, wrong = 0, 0
        for arm in ("mg1", "mg2m"):
            rows = np.flatnonzero(exc[f"input_reason_{arm}"] == miss_code)
            for i in rows:
                tt = int(exc["candidate_row_ids"][i])
                kk = gidx.get(tt)
                if kk is None or kk - SPACING[arm] * (SLOTS[arm] - 1) < 0:
                    continue
                want = [int(grid[kk - SPACING[arm] * s]) for s in range(SLOTS[arm])]
                checked += 1
                wrong += all(v in pos and np.isfinite(x[pos[v]]).all() for v in want)
        r["slot_not_observed_exclusions_checked"] = checked
        r["slot_not_observed_exclusions_contradicted"] = wrong
        all_ok &= r["mg2m_slots_equal_mg1_every_second"] and wrong == 0 and r["origins_in_bounds"] and \
            r["labels_before_split_end"] and r["labels_before_test"] and r["counts_match_seal"]
        res["splits"][split] = r
    # split disjointness
    ids = [set(npz(pop / f"POPULATION_{s}.npz")["origin_row_ids"].tolist()) for s in SPLITS]
    res["splits_disjoint"] = not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    # scaler
    f1, f2 = npz(pop / "ARM_mg1_FIT.npz"), npz(pop / "ARM_mg2m_FIT.npz")
    union = np.unique(np.concatenate([f1["slot_instants"].reshape(-1), f2["slot_instants"].reshape(-1)]))
    sealed_rows = npz(pop / "SCALER_ROWS.npz")["fit_source_bar_ts"]
    xs = x[[pos[int(v)] for v in union]]
    mean, std = xs.mean(axis=0), xs.std(axis=0)
    res["scaler"] = {"union_rows": int(len(union)), "equals_SCALER_ROWS": bool(np.array_equal(union, sealed_rows)),
                     "all_before_inner_start": bool(union.max() < bounds["INNER"][0]),
                     "mean_equal": bool(np.array_equal(mean, np.asarray(body["scaler"]["mean"]))),
                     "std_equal": bool(np.array_equal(std, np.asarray(body["scaler"]["std"]))),
                     "max_abs_mean_diff": float(np.max(np.abs(mean - np.asarray(body["scaler"]["mean"])))),
                     "max_abs_std_diff": float(np.max(np.abs(std - np.asarray(body["scaler"]["std"]))))}
    # naive loss scale
    fit = npz(pop / "POPULATION_FIT.npz")
    naive = {}
    for j, h in enumerate(HORIZONS):
        tt = fit["origin_row_ids"][fit["head_mask"][:, j]]
        p0 = price[[pos[int(v)] for v in tt]]
        p1 = price[[pos[int(v) + h * H] for v in tt]]
        naive[str(h)] = math.fsum(np.abs(p1 - p0).tolist()) / len(tt)
    res["naive_loss_scale"] = {h: {"recomputed": v, "sealed": body["naive_loss_scale"][h],
                                   "equal": v == body["naive_loss_scale"][h]} for h, v in naive.items()}
    dst = {wk: sorted(weekend_gaps.get(wk, [])) for wk in
           ("2024-03-03", "2024-03-10", "2024-03-17", "2024-10-20", "2024-10-27", "2024-11-03", "2024-11-10")}
    res["weekend_gap_hours_by_reopen_day_mg1"] = dst
    res["dst_weekends_differ_by_one_hour"] = bool(
        dst["2024-03-10"] and dst["2024-03-03"] and dst["2024-03-17"] and dst["2024-11-03"]
        and dst["2024-10-20"] and dst["2024-11-10"]
        and dst["2024-03-10"][0] == dst["2024-03-03"][0] - 1 == dst["2024-03-17"][0] - 1
        and dst["2024-11-03"][0] == dst["2024-10-20"][0] + 1 == dst["2024-11-10"][0] + 1)
    # the EU/US DST-mismatch Sunday (2024-10-27): the calendar expects the 22:00Z bar; is it observed?
    t2200 = int(datetime(2024, 10, 27, 22, tzinfo=timezone.utc).timestamp())
    exc = npz(pop / "EXCLUSIONS_DEVELOPMENT.npz")
    codes = body["report"]["input_reason_codes"]
    win = (exc["candidate_row_ids"] >= t2200) & (exc["candidate_row_ids"] < t2200 + 26 * H)
    res["oct27_expected_2200Z_bar_observed"] = t2200 in pos
    res["oct27_candidate_input_reasons_mg1"] = {codes[c]: int((exc["input_reason_mg1"][win] == c).sum())
                                              for c in set(exc["input_reason_mg1"][win].tolist())}
    all_ok &= res["splits_disjoint"] and all(v for k, v in res["scaler"].items() if isinstance(v, bool)) and \
        all(v["equal"] for v in res["naive_loss_scale"].values()) and res["dst_weekends_differ_by_one_hour"]
    res["verdict"] = "PASS" if all_ok else "DEFECT_OR_MISMATCH"
    C.write("p3_independent.json", res)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
