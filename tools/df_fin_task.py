#!/usr/bin/env python3
"""RP70: the financial task as an executable population — horizons by ELAPSED TIME, folds by week identity,
Huber deltas from the admissible train pairs at full precision, candidates enumerated, receivers frozen.

What Musashi measured against the v1 design and what changes here:

  horizons     "6 h" was six retained rows: a Friday 23:00 origin got Monday 05:00 (54 h) as its "6 h" target.
               Here a target is the bar whose timestamp is EXACTLY origin + h hours. If that bar is absent
               (weekend, holiday, missing intraday row) the origin is EXCLUDED with its reason and counted;
               an event-time horizon would be another task and is not silently substituted.
  availability the file is a retrospective archive: the label of an hourly bar is taken as the time the bar is
               complete (declared, the lake's contract says event time = available time); intrabar finality,
               producer publication delay and timezone are NOT established by the file and are recorded as
               limitations, never invented. The decision at t reads bars with label <= t only.
  folds        DEV = the last `dev_weeks` complete weeks (Monday 00:00 .. Sunday 23:00 on the label's clock)
               strictly before the reserve (2025-01-01, never read). Each fold: test week k, validation week
               k-1, train = the `history_weeks` weeks before the validation week; a purge of h hours between
               train targets and validation origins, and between validation targets and test origins.
  deltas       the residual scale is the median |y(t+h) - y(t)| / sigma_train over the ADMISSIBLE TRAIN PAIRS
               of the fold (timestamp-mapped), full precision. Rules declared BEFORE any outcome: fewer than
               MIN_PAIRS pairs -> FALLBACK_FIXED_GRID; sigma not finite or <= 0 (flat train) -> NONIDENTIFIABLE
               (the Huber family is NOT_RUN in that fold); a zero median scale with a positive sigma -> the
               first positive residual quantile above the median, else FALLBACK_FIXED_GRID. Every candidate is
               finite and strictly positive; nothing is rounded.
  candidates   exactly 12 DEV fits per loss family per (receiver, horizon, fold, seed), enumerated by id; the
               four installed-default arms are a separate, declared population (not inside the 12).
  receivers    both receivers' width, depth and input set are frozen here (parameters measured by building
               them), not "chosen at pilot".
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAKE, RESOURCE = "financial_files", "market_data/forex/g10/eurusd/1h.parquet"
TIME_COLUMN, HOLDOUT = "datetime", "2025-01-01"
INPUT_COLUMNS = ["open", "high", "low", "close", "volume"]
TARGET = "close"
HORIZONS = {"h6": {"hours": 6, "why": "one quarter of a trading day at this resolution; declared before any score"},
            "h72": {"hours": 72, "why": "about three days of elapsed time; declared before any score"}}
HOUR_NS = 3_600_000_000_000
WEEK_HOURS = 168
MIN_PAIRS = 100
FIXED_GRID_Z = (0.1, 0.5, 1.0, 2.0)
SCALE_FRACTIONS = (0.25, 0.5, 1.0, 2.0)
LEARNING_RATES = (0.0005, 0.001, 0.003)
WEIGHT_DECAYS = (0.001, 0.004, 0.01)
DEFAULTS = {"adam": {"lr": 0.001, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-7},
            "adamw": {"lr": 0.001, "weight_decay": 0.004, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-7},
            "huber": {"delta_z": 1.0}}


class TaskRefusal(SystemExit):
    pass


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- bars -----------------------------------------------------------------------------------------------------

def _instants(series, *, name: str, timestamp_interpretation: str | None) -> np.ndarray:
    """Labels as int64 ns. A tz-aware column is interpreted ONLY under a declared rule (UTC_OFFSET_AWARE converts to UTC);
    an offset is never discarded silently (RP75)."""
    import pandas as pd
    ts = pd.to_datetime(series) if not pd.api.types.is_datetime64_any_dtype(series) else series
    if getattr(ts.dt, "tz", None) is not None:
        if timestamp_interpretation != "UTC_OFFSET_AWARE":
            raise TaskRefusal(f"REFUSED: {name!r} carries a UTC offset and the design declares no interpretation "
                              "(timestamp_interpretation='UTC_OFFSET_AWARE' converts to UTC); an offset is not discarded")
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    if ts.isna().any():
        raise TaskRefusal(f"REFUSED: {int(ts.isna().sum())} labels of {name!r} are missing or unparseable")
    return ts.to_numpy().astype("datetime64[ns]").astype(np.int64)


def parse_bars(frame, *, time_column: str = TIME_COLUMN, holdout: str = HOLDOUT, columns: list = INPUT_COLUMNS, target: str = TARGET,
               available_time_column: str | None = None, timestamp_interpretation: str | None = None, availability: dict | None = None) -> dict:
    """The bars as served: strictly increasing labels, declared columns present, nothing at or after the reserve.
    `available_time_column` (from the delivered producer contract) gives each bar's availability instant; without it the
    label is used AND the record says the basis is the archive label, never observed availability."""
    if time_column not in frame.columns:
        raise TaskRefusal(f"REFUSED: the time column {time_column!r} is absent")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise TaskRefusal(f"REFUSED: declared input columns absent from the resource: {missing}")
    ts_ns = _instants(frame[time_column], name=time_column, timestamp_interpretation=timestamp_interpretation)
    if available_time_column and available_time_column != time_column:
        if available_time_column not in frame.columns:
            raise TaskRefusal(f"REFUSED: the declared available-time column {available_time_column!r} is absent")
        avail_ns = _instants(frame[available_time_column], name=available_time_column, timestamp_interpretation=timestamp_interpretation)
        if (avail_ns < ts_ns).any():
            raise TaskRefusal("REFUSED: a bar is declared available BEFORE its own label; the producer contract is inconsistent")
        basis = "PRODUCER_AVAILABLE_TIME_COLUMN"
    else:
        avail_ns = ts_ns.copy()
        basis = "ARCHIVE_LABEL_AS_AVAILABLE (UNDECLARED availability: no point-in-time or live inference is drawn)"
    if ts_ns.size == 0:
        raise TaskRefusal("REFUSED: no bars")
    d = np.diff(ts_ns)
    if (d <= 0).any():
        raise TaskRefusal(f"REFUSED: {int((d <= 0).sum())} bars are not strictly increasing in time (duplicates or disorder)")
    limit = np.datetime64(holdout).astype("datetime64[ns]").astype(np.int64)
    if ts_ns.max() >= limit:
        raise TaskRefusal(f"REFUSED: bars at or after the reserve {holdout} were delivered; the reserve is never read")
    X = frame[columns].to_numpy(dtype=np.float64)
    return {"ts_ns": ts_ns, "avail_ns": avail_ns, "X": X, "y": frame[target].to_numpy(dtype=np.float64), "columns": list(columns), "target": target,
            "target_channel": columns.index(target), "n": int(ts_ns.size),
            "availability": {"basis": basis, "delivery_contract": availability or {"availability_use": "UNDECLARED"},
                             "not_established_by_the_file": ["intrabar finality", "producer publication delay", "timezone of the labels",
                                                             "release cutoffs of the producer"],
                             "decision_rule": "the decision at t reads bars whose AVAILABLE instant is <= t (label when no availability is declared)",
                             "scope": "archive-only evidence; a hardcoded sentence is not an availability contract"}}


def admissible_pairs(bars: dict, origins: np.ndarray, targets: np.ndarray, *, W: int) -> dict:
    """The pairs whose CONSUMED support meets the missingness policy: W retained bars ending at the origin, every input role
    finite on every window row, finite origin and label, and every window row AVAILABLE at the origin (RP75/A3)."""
    X, y, ts, avail = bars["X"], bars["y"], bars["ts_ns"], bars["avail_ns"]
    n = X.shape[0]
    row_ok = np.isfinite(X).all(axis=1)
    c_ok = np.concatenate([[0], np.cumsum(row_ok)])
    o, t = np.asarray(origins, dtype=np.int64), np.asarray(targets, dtype=np.int64)
    reasons = {}
    keep = o >= W-1
    reasons["WINDOW_BEFORE_FIRST_BAR"] = int((~keep).sum())
    win_fin = np.zeros(o.size, dtype=bool)
    win_fin[keep] = (c_ok[o[keep]+1] - c_ok[o[keep]-W+1]) == W
    reasons["NONFINITE_INPUT_IN_WINDOW"] = int((keep & ~win_fin).sum())
    lab_fin = np.isfinite(y[t]) & np.isfinite(y[o])
    reasons["NONFINITE_LABEL_OR_ORIGIN"] = int((keep & win_fin & ~lab_fin).sum())
    # availability: the latest availability instant among the window rows must not exceed the origin's label
    late = np.zeros(o.size, dtype=bool)
    if not np.array_equal(avail, ts):
        cm = np.maximum.accumulate(avail)                                  # monotone envelope is not enough for windows: check exactly
        for i in np.flatnonzero(keep & win_fin & lab_fin):
            late[i] = avail[o[i]-W+1:o[i]+1].max() > ts[o[i]]
    reasons["ROW_NOT_AVAILABLE_AT_ORIGIN"] = int(late.sum())
    ok = keep & win_fin & lab_fin & ~late
    return {"origins": o[ok], "targets": t[ok], "n": int(ok.sum()), "excluded": reasons, "candidates": int(o.size),
            "policy": "declared missingness: no imputation; a window with any non-finite input role, a non-finite label/origin or a row "
                      "not yet available at the origin is excluded with its reason"}



def map_targets(ts_ns: np.ndarray, y: np.ndarray, *, hours: int) -> dict:
    """The target of origin i is the bar labelled EXACTLY ts[i] + hours; absent -> excluded with its reason."""
    if hours <= 0:
        raise TaskRefusal("REFUSED: a horizon is a positive number of hours")
    want = ts_ns + hours*HOUR_NS
    pos = np.searchsorted(ts_ns, want)
    found = (pos < ts_ns.size)
    found[found] = ts_ns[pos[found]] == want[found]
    tgt = np.where(found, pos, -1)
    reasons = {"NO_BAR_AT_ORIGIN_PLUS_H": int((~found).sum())}
    fin_t = np.zeros(ts_ns.size, dtype=bool)
    fin_t[found] = np.isfinite(y[tgt[found]])
    reasons["NONFINITE_TARGET"] = int((found & ~fin_t).sum())
    fin_o = np.isfinite(y)
    reasons["NONFINITE_ORIGIN"] = int((found & fin_t & ~fin_o).sum())
    ok = found & fin_t & fin_o
    elapsed_next_rows = (ts_ns[np.minimum(np.arange(ts_ns.size)+int(hours), ts_ns.size-1)] - ts_ns)/HOUR_NS
    return {"hours": int(hours), "origins": np.flatnonzero(ok), "targets": tgt[ok], "excluded": reasons,
            "admissible": int(ok.sum()), "candidates": int(ts_ns.size),
            "row_offset_would_have_been_wrong_for": int(((elapsed_next_rows != hours) & ok).sum()),
            "reading": f"{int(ok.sum())} origins have a bar exactly {hours} h later; {int((~found).sum())} have none (weekend, "
                       "holiday or missing bar) and are excluded, never mapped to the next retained row"}


# --- folds by week identity ---------------------------------------------------------------------------------------

def week_start_ns(t_ns: int) -> int:
    day = np.datetime64(np.datetime64(int(t_ns), "ns"), "D")
    monday = day - np.timedelta64(int((day.astype("datetime64[D]").astype(int) - 4) % 7), "D")    # 1970-01-01 was a Thursday (4)
    return int(monday.astype("datetime64[ns]").astype(np.int64))


def dev_folds(ts_ns: np.ndarray, *, holdout: str = HOLDOUT, dev_weeks: int = 26, history_weeks: int = 52, purge_hours: int = 72) -> list:
    """Fold identities (week starts on the label's clock) strictly before the reserve; nothing after it exists here."""
    limit = np.datetime64(holdout).astype("datetime64[ns]").astype(np.int64)
    if ts_ns.max() >= limit:
        raise TaskRefusal("REFUSED: bars at or after the reserve")
    last_complete = week_start_ns(limit) - WEEK_HOURS*HOUR_NS            # the last week fully before the reserve
    first = week_start_ns(int(ts_ns.min()))
    folds = []
    for k in range(dev_weeks):
        test = last_complete - k*WEEK_HOURS*HOUR_NS
        val = test - WEEK_HOURS*HOUR_NS
        train_lo = val - history_weeks*WEEK_HOURS*HOUR_NS
        if train_lo < first:
            raise TaskRefusal(f"REFUSED: fold {k} needs history before the first bar; declare fewer weeks")
        folds.append({"fold": k, "test_week": str(np.datetime64(test, "ns"))[:10], "validation_week": str(np.datetime64(val, "ns"))[:10],
                      "train_weeks": [str(np.datetime64(train_lo, "ns"))[:10], str(np.datetime64(val, "ns"))[:10]],
                      "bounds_ns": {"train": [int(train_lo), int(val)], "validation": [int(val), int(test)],
                                    "test": [int(test), int(test+WEEK_HOURS*HOUR_NS)]},
                      "purge_hours": int(purge_hours), "history_weeks": int(history_weeks)})
    return folds


def fold_pairs(fold: dict, ts_ns: np.ndarray, mapping: dict) -> dict:
    """Admissible (origin, target) pairs per split of one fold: origin AND target inside the split's bounds, with a purge of
    `purge_hours` between the train targets and the validation origins, and between the validation targets and the test origins."""
    o, t = mapping["origins"], mapping["targets"]
    b = fold["bounds_ns"]
    purge = fold["purge_hours"]*HOUR_NS
    out = {}
    for split, (lo, hi) in b.items():
        keep = (ts_ns[o] >= lo) & (ts_ns[o] < hi) & (ts_ns[t] < hi)          # origin AND target inside the split
        if split == "train":
            keep &= ts_ns[t] < b["validation"][0] - purge
        if split == "validation":
            keep &= ts_ns[t] < b["test"][0] - purge
        out[split] = {"origins": o[keep], "targets": t[keep], "n": int(keep.sum())}
    return out


# --- residual scale and Huber deltas ---------------------------------------------------------------------------------

def delta_candidates(y: np.ndarray, train_pairs: dict, *, sigma: float, fractions=SCALE_FRACTIONS, min_pairs: int = MIN_PAIRS) -> dict:
    """Huber deltas in z from the admissible train pairs; every rule is declared and every candidate is finite and > 0."""
    rules = {"min_pairs": min_pairs, "fractions_of_scale": list(fractions), "fixed_grid_z": list(FIXED_GRID_Z),
             "insufficient_pairs": "FALLBACK_FIXED_GRID", "flat_train_target": "NONIDENTIFIABLE: the Huber family is NOT_RUN in this fold",
             "zero_median_scale": "the first positive residual quantile above the median (q60..q99), else FALLBACK_FIXED_GRID",
             "precision": "full float64; nothing rounded"}
    o, t = np.asarray(train_pairs["origins"]), np.asarray(train_pairs["targets"])
    if not (isinstance(sigma, (int, float)) and math.isfinite(sigma) and sigma > 0):
        return {"status": "NONIDENTIFIABLE", "reason": "FLAT_TRAIN_TARGET", "candidates": [], "rules": rules, "pairs": int(o.size)}
    if o.size < min_pairs:
        return {"status": "FALLBACK_FIXED_GRID", "reason": f"INSUFFICIENT_PAIRS ({o.size} < {min_pairs})", "rules": rules, "pairs": int(o.size),
                "candidates": [{"delta_z": float(g), "origin": "fixed_grid"} for g in FIXED_GRID_Z]}
    r = np.abs(y[t]-y[o])/sigma
    r = r[np.isfinite(r)]
    if r.size < min_pairs:
        return {"status": "FALLBACK_FIXED_GRID", "reason": f"INSUFFICIENT_FINITE_RESIDUALS ({r.size} < {min_pairs})", "rules": rules,
                "pairs": int(o.size), "candidates": [{"delta_z": float(g), "origin": "fixed_grid"} for g in FIXED_GRID_Z]}
    scale = float(np.median(r))
    origin = "median_absolute_h_step_change_over_train_pairs"
    if not scale > 0:
        qs = [(q, float(np.quantile(r, q/100))) for q in range(60, 100)]
        pos = next(((q, v) for q, v in qs if v > 0), None)
        if pos is None:
            return {"status": "FALLBACK_FIXED_GRID", "reason": "ZERO_RESIDUAL_SCALE (no positive residual quantile)", "rules": rules,
                    "pairs": int(o.size), "candidates": [{"delta_z": float(g), "origin": "fixed_grid"} for g in FIXED_GRID_Z]}
        scale, origin = pos[1], f"first positive residual quantile above the median: q{pos[0]}"
    cands = [{"delta_z": scale*f, "origin": f"{f} x scale", "fraction_of_scale": f} for f in fractions]
    for c in cands:
        if not (math.isfinite(c["delta_z"]) and c["delta_z"] > 0):
            raise TaskRefusal("REFUSED: a Huber delta candidate is not finite and positive")
        c["fraction_of_train_residuals_below_delta"] = float(np.mean(r <= c["delta_z"]))
        c["delta_in_target_units"] = c["delta_z"]*sigma
    return {"status": "MEASURED", "scale_z": scale, "scale_origin": origin, "sigma_train": float(sigma), "pairs": int(o.size),
            "candidates": cands, "rules": rules}


# --- the enumerated candidate population ----------------------------------------------------------------------------------------

def candidate_allocation() -> dict:
    """Exactly 12 DEV fits per loss family per (receiver, horizon, fold, seed); the defaults are a separate population."""
    mae, huber = [], []
    for lr in LEARNING_RATES:
        mae.append({"id": f"mae_adam_lr{lr:g}", "loss": "mae", "optimizer": "adam", "lr": lr, "weight_decay": 0.0, "delta": None})
        for wd in WEIGHT_DECAYS:
            mae.append({"id": f"mae_adamw_lr{lr:g}_wd{wd:g}", "loss": "mae", "optimizer": "adamw", "lr": lr, "weight_decay": wd, "delta": None})
    for lr in (0.001, 0.003):
        for f in (0.5, 1.0, 2.0):
            huber.append({"id": f"huber_adam_lr{lr:g}_d{f:g}s", "loss": "huber", "optimizer": "adam", "lr": lr, "weight_decay": 0.0,
                          "delta": {"kind": "scale_fraction", "fraction": f}})
            huber.append({"id": f"huber_adamw_lr{lr:g}_wd0.004_d{f:g}s", "loss": "huber", "optimizer": "adamw", "lr": lr, "weight_decay": 0.004,
                          "delta": {"kind": "scale_fraction", "fraction": f}})
    defaults = [{"id": "default_mae_adam", "loss": "mae", "optimizer": "adam", "lr": DEFAULTS["adam"]["lr"], "weight_decay": 0.0, "delta": None},
                {"id": "default_mae_adamw", "loss": "mae", "optimizer": "adamw", "lr": DEFAULTS["adamw"]["lr"], "weight_decay": DEFAULTS["adamw"]["weight_decay"], "delta": None},
                {"id": "default_huber_adam", "loss": "huber", "optimizer": "adam", "lr": DEFAULTS["adam"]["lr"], "weight_decay": 0.0,
                 "delta": {"kind": "fixed", "delta_z": DEFAULTS["huber"]["delta_z"]}},
                {"id": "default_huber_adamw", "loss": "huber", "optimizer": "adamw", "lr": DEFAULTS["adamw"]["lr"], "weight_decay": DEFAULTS["adamw"]["weight_decay"],
                 "delta": {"kind": "fixed", "delta_z": DEFAULTS["huber"]["delta_z"]}}]
    assert len(mae) == 12 and len(huber) == 12 and len({c["id"] for c in mae+huber+defaults}) == 28
    return {"per_family": 12, "mae": mae, "huber": huber, "defaults": defaults,
            "trade_off": "the Huber family spends its 12 on 2 LR x 3 deltas x {adam, adamw@0.004}; the MAE family on 3 LR x {adam, 3 decays}: "
                         "equal fit counts and observed budgets, NOT identical LR/decay coverage — declared",
            "budget_per_fit": "the recipe's update ceiling and cadence, identical for every candidate; observed updates are recorded per cell",
            "selection": "validation MAE_z of the fold (never the test week); every candidate's test score is retained, no best-test"}


def resolve_delta(candidate: dict, deltas: dict) -> float | None:
    """The delta_z a candidate uses in a fold, or None when the family is not identifiable there."""
    d = candidate.get("delta")
    if d is None:
        return None
    if d["kind"] == "fixed":
        return float(d["delta_z"])
    if deltas["status"] == "NONIDENTIFIABLE":
        return None
    if deltas["status"] == "MEASURED":
        return float(deltas["scale_z"]*d["fraction"])
    grid = [c["delta_z"] for c in deltas["candidates"]]
    return float(grid[min(range(len(grid)), key=lambda i: abs(math.log(grid[i]) - math.log(d["fraction"])))])


# --- receivers, frozen ----------------------------------------------------------------------------------------------------------------

def receivers() -> dict:
    K = _module("df_e1_block")
    P = _module("df_e1_pilot")
    compact_asg, larger_asg = [0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 2, 2, 2, 2]
    compact = K.build_modular(compact_asg, 60, 5, 3, 1)
    larger = K.build_modular(larger_asg, 144, 9, 3, 1)
    return {"compact_modular": {"window": 60, "input_columns": INPUT_COLUMNS, "channels": 5, "assignment": compact_asg,
                                "groups": "price (OHLC) | volume", "core": "tcn_w, 16 filters, kernel 3", "dilations": P.core_dilations(60),
                                "parameters": K.n_params(compact), "reach_declared": P.model_reach(60, "tcn_w"),
                                "head": "Dense(p) increment over the last observation; the target channel (close) is read out"},
            "larger_business_receiver": {"window": 144, "input_columns": INPUT_COLUMNS + ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"],
                                         "channels": 9, "assignment": larger_asg, "groups": "price | volume | calendar",
                                         "core": "tcn_w, 16 filters, kernel 3", "dilations": P.core_dilations(144),
                                         "parameters": K.n_params(larger), "reach_declared": P.model_reach(144, "tcn_w"),
                                         "head": "as compact"},
            "rule": "both receivers run every candidate; no candidate is compared across receivers; width, depth and inputs are frozen here"}


def resolution_check(min_delta: float, intended_resolution: float, observed_value: float) -> dict:
    """Whether the stopping rule and the float64 floor can resolve the intended difference at the observed magnitude."""
    floor = float(np.spacing(np.float64(observed_value)))
    return {"min_delta": float(min_delta), "intended_resolution": float(intended_resolution), "float64_floor_at_value": floor,
            "min_delta_hides_intended": bool(min_delta > 0 and min_delta >= intended_resolution),
            "floor_resolves_intended": bool(floor < intended_resolution),
            "reading": "a min_delta at or above the intended resolution would treat such an improvement as no improvement; the float64 "
                       "floor is what survives aggregation of a mean over n rows"}
