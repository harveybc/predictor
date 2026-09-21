#!/usr/bin/env python3
"""RP68/RP72: the block runner — the ACTUAL feature paths, enumeration, scaling, cadence and governance.

A block is a set of paired cells (arm x seed) on the household DEV task (W60/h60, the successor run's
rows), sealed before anything runs and executed under governance, one campaign and delivery per unit,
one terminal with artifacts per unit. Everything the phase-2 design declares happens HERE, in code
that the tests drive through this consumer:

  features       base (the 7 declared channels), calendar (+4: hour/weekday sin-cos from each row's own
                 label, tools/df_e1_calendar.py), randomised_calendar_control (the same 4 channels from
                 the label plus a per-row hashed random offset: same capacity, deterministic, prefix-
                 stable, chance association measured), daily_lag (+1: y(t+h-1440), a row at or before
                 t, read from the panel rows before the slice)
  windows        W=60 or W=1440 over the SAME origins: the panel is read with a left pad so a long
                 window never withdraws an origin the short one keeps
  crop           `crop=60` cuts the raw input to its last 60 rows BEFORE the extractor: an exact
                 information null for the long window (same weights, same padding, same 60 rows).
                 The clamped-dilation long model is NOT a null: it reaches 67 samples (measured) and is
                 named local_support_67 — extra context, declared as such
  enumeration    admissible origins from ROW IDENTITIES: grid-consecutive support, finite inputs on the
                 window rows, finite label at t+h, finite lag where the arm uses it; one COMMON
                 admissible evaluation mask across the block's arms, derived before any score
  scaling        the COMMON train-only scaler of the source run (28 d, W60 windows) for every arm and
                 every volume tier; calendar channels are on the circle and take mean 0 / sd 1; the lag
                 takes the target's scaler. One evaluation sigma for the whole block
  cadence        validation every `validate_every` OBSERVED optimizer updates, patience counted in
                 validation events, restore best, both fixed across arms; reaching the update ceiling
                 is CENSORED wherever the best checkpoint fell
  volume         train history grown BACKWARDS from the fixed DEV validation week; counts of unique
                 support rows, train-only rows, targets, windows and exposures from the identities
  cost pilot     on a declared subset INSIDE train (validation = the last 7 train days, purged); it never
                 reads the DEV validation; it measures seconds/update, seconds/validation event, peak RSS
  closure        arrays vs source rows, reload parity, terminal artifacts vs warehouse, three baselines on
                 identical rows (persistence, daily seasonal, train-only constant); no fit at closure

    seal     python tools/df_e1_block.py seal --block DEV_MATCHED --out DESIGN.json
    prepare  python tools/df_e1_block.py prepare --root ROOT --run-id ID --api-key-file KEY
    pilot    python tools/df_e1_block.py pilot   --root ROOT --run-id ID --api-key-file KEY
    execute  python tools/df_e1_block.py execute --root ROOT --run-id ID --api-key-file KEY
    close    python tools/df_e1_block.py close   --root ROOT --warehouse-token-file T
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = "df_e1_block_design.v1"
SCHEMA_DATA = "df_e1_block_data.v1"
DAY = 1440
W0, H0 = 60, 60
SEEDS = (1, 2, 3)
LAKE, RESOURCE = "public_panels", "uci_235_individual_household_power/panel.parquet"
SOURCE_RUN = Path("~/.local/state/crispdm-data-foundation/e1_household_successor_v3").expanduser()

# every arm the phase can run; a block picks some of them. `features`, `window`, `dilations`, `crop`
# and `train_days` are the ONLY things an arm may move; the recipe, rows, scaler and seeds are the block's
ARMS = {
    "modular_w60":            {"family": "modular", "window": 60,   "features": "base"},
    "gru_adapted_w60":        {"family": "gru",     "window": 60,   "features": "base"},
    "calendar":               {"family": "modular", "window": 60,   "features": "calendar"},
    "randomised_calendar_control": {"family": "modular", "window": 60, "features": "randomised_calendar",
                               "role": "CAPACITY_CONTROL: the same 4 channels built from each row's label plus a per-row random offset "
                                       "drawn from a hash of (seed, panel row id): deterministic, prefix-stable, no calendar information; "
                                       "its finite-sample association with the true clock is measured, not assumed zero"},
    "daily_lag":              {"family": "modular", "window": 60,   "features": "daily_lag"},
    "long_window_own_depth":  {"family": "modular", "window": 1440, "features": "base"},
    "long_window_crop60":     {"family": "modular", "window": 1440, "features": "base", "crop": 60,
                               "role": "EXACT_INFORMATION_NULL: the raw input is cropped to its last 60 rows before the extractor"},
    "long_window_local_support_67": {"family": "modular", "window": 1440, "features": "base", "dilations": [1, 2, 4, 8, 16],
                               "role": "EXTRA_CONTEXT_67_SAMPLES (measured): NOT a null; the clamped core still reaches "
                                       "branch 5 + core 63 - 1 = 67 raw samples"},
    "short_window_deep_core": {"family": "modular", "window": 60,   "features": "base", "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]},
    "volume_56d":             {"family": "modular", "window": 60,   "features": "base", "train_days": 56},
    "volume_112d":            {"family": "modular", "window": 60,   "features": "base", "train_days": 112},
}
BLOCKS = {
    "DEV_MATCHED": {"arms": ["modular_w60", "gru_adapted_w60"], "question": "matched architecture comparison: the modular continuity "
                                                                       "model against the adapted literature GRU on identical rows, "
                                                                       "inputs, split, transform and monitoring schedule"},
    "Q1_CALENDAR": {"arms": ["calendar", "randomised_calendar_control"], "question": "does the wall-clock position add information at h60? "
                                                                          "baseline modular_w60 reused from DEV_MATCHED when its contract matches"},
    "Q2_CONTEXT":  {"arms": ["daily_lag", "long_window_own_depth", "long_window_crop60", "long_window_local_support_67",
                             "short_window_deep_core"], "question": "information beyond the hour, separated from depth and from padding"},
    "Q3_VOLUME":   {"arms": ["volume_56d", "volume_112d"], "question": "more history with the evaluation, scaler and cadence FIXED"},
}
RECIPE = {"loss": "mae", "optimizer": "adam", "learning_rate": 0.003, "batch": 64, "max_updates": 4000,
          "validate_every_updates": 200, "patience_events": 3, "restore_best": True, "min_delta": 0.0,
          "monitor": "validation MAE in scaled units (equals MAE_z with the common sigma)",
          "checkpoint_opportunities": 20,
          "role": "HOUSEHOLD CONTINUITY REFERENCE (MAE+Adam) with validation in OBSERVED updates; it selects no trading loss"}
PILOT = {"max_updates": 200, "validate_every_updates": 50, "patience_events": 3,
         "population": "TRAIN_SUBSET: trains on train origins before the last 7 train days (purged W+h); validates on the last 7 "
                       "train days; the DEV validation is never read by a pilot"}
LIMITS = {"child_cpu_seconds": 3600, "child_wall_seconds": 4800, "parallel_children": 3, "campaign_cpu_seconds": 14400,
          "closure_reserve_seconds": 2000}


class BlockRefusal(SystemExit):
    pass


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_obj(obj) -> str:
    return _module("df_mod_e0").sha_obj(obj)


def write(path: Path, value):
    _module("df_d3_campaign").write_once(Path(path), value)


# --- the sealed design ---------------------------------------------------------------------------------------

def arm_spec(arm: str) -> dict:
    if arm not in ARMS:
        raise BlockRefusal(f"REFUSED: unknown arm {arm!r}")
    return {"arm": arm, "crop": None, "dilations": None, "train_days": 28, "role": "ARM", **ARMS[arm]}


def seal(block: str, *, source_run: Path = SOURCE_RUN, seeds=SEEDS, reuse: dict | None = None, contract=None,
         limits: dict | None = None) -> dict:
    """The block, sealed by content: arms, seeds, recipe, rows, scaler identity and the benchmark contract.
    `contract` replaces the household registry contract ONLY for tests on synthetic panels (recorded as such)."""
    if block not in BLOCKS:
        raise BlockRefusal(f"REFUSED: unknown block {block!r}")
    B = _module("df_benchmark_contract")
    src_design = json.loads((Path(source_run)/"DESIGN.json").read_text())
    data_json = json.loads((Path(source_run)/"DATA.json").read_text())
    ours = contract if contract is not None else B.household_ours()
    arms = [arm_spec(a) for a in BLOCKS[block]["arms"]]
    if block == "DEV_MATCHED":
        comp = {**B.decide(ours, B.gasparin_2019()), "against": "gasparin_2019",
                "rule": "the adapted GRU is measured under OUR contract; Table 5 stays in the source notes"}
    else:
        contrast = ("permitted_inputs" if block in ("Q1_CALENDAR", "Q2_CONTEXT") else "split_rule")
        theirs = B.replace(ours, task_id=f"{ours.task_id}.{block}", varying_factors=(contrast,),
                           estimand=BLOCKS[block]["question"])
        ours_c = B.replace(ours, varying_factors=(contrast,), estimand=BLOCKS[block]["question"])
        comp = {**B.decide(ours_c, theirs), "against": "the block's own baseline under one declared estimand"}
        ours = ours_c
    pad = max([a["window"]-W0 for a in arms] + [DAY-H0 if any(a["features"] == "daily_lag" for a in arms) else 0] + [0])
    max_days = max(a["train_days"] for a in arms)
    lo, hi = data_json["slice_rows"]
    train_end = lo + 28*DAY
    if hi != train_end + 7*DAY:
        raise BlockRefusal("REFUSED: the source slice is not 28 d train + 7 d validation")
    design = {
        "schema": SCHEMA, "purpose": f"E1_BLOCK_{block}", "block": block, "phase": "DEVELOPMENT",
        "state": "SEALED_NOT_EXECUTED", "question": BLOCKS[block]["question"],
        "source_run": {"root": str(source_run), "design_sha256": src_design["design_sha256"],
                       "data_sha256": data_json["data_sha256"], "panel_sha256": data_json["panel_sha256"],
                       "slice_rows": [lo, hi], "train_end_row": train_end, "evaluation_origins": data_json["enumerator"]["validation"]["admissible"],
                       "input_columns": data_json["input_columns"], "target_channel": data_json["target_channel"],
                       "scaler": data_json["scaler"], "graph_assignment": src_design["graph"]["assignment"]},
        "rows": {"lo": lo - max_days*DAY + 28*DAY - pad, "pad_rows": pad, "widest_train_days": max_days, "train_end": train_end, "hi": hi,
                 "reading": "the panel is read from lo (pad + the widest volume tier before the DEV train span) to hi; origins are "
                            "enumerated inside [train span, validation week] from row identities; padded rows are support only"},
        "arms": arms, "seeds": list(seeds), "recipe": RECIPE, "pilot": PILOT, "limits": limits or LIMITS,
        "contract_source": "REGISTRY household_W60_h60" if contract is None else "DECLARED_OVERRIDE (synthetic test panel)",
        "scaler_rule": "COMMON: the source run's train-only scaler (28 d, W60 windows) for every arm and tier; calendar channels "
                       "mean 0 / sd 1; the lag channel takes the target's scaler; one evaluation sigma = the target's train sd",
        "common_evaluation_rule": "the intersection over the block's arms of admissible validation origins, with a finite label "
                                  "and a finite daily lookup, derived at prepare BEFORE any score; every arm scores on it",
        "cells": [{"cell_id": f"{a['arm']}_s{s}", "arm": a["arm"], "seed": s} for s in seeds for a in arms],
        "pilots": [{"cell_id": f"pilot_{a['arm']}", "arm": a["arm"], "seed": int(seeds[0]), **PILOT, "role": "COST_PILOT"} for a in arms],
        "reuse": reuse or {},
        "baselines": {"persistence": "y(t)", "daily_seasonal": "y(t+h-1440)", "train_constant": "mean of the train labels of the "
                      "28 d tier; computed on the common evaluation set at closure, no fit, no terminal"},
        "benchmark_contract": ours.to_design_block(comparability=comp),
        "source_code": {name: sha_file(HERE/name) for name in ("df_e1_block.py", "df_gru_reference.py", "df_e1_calendar.py",
                                                             "df_e1_pilot.py", "df_mod_e0.py", "df_e1_governed.py")},
        "reading_rules": ["three seeds on one task are development evidence", "a fit that reached the update ceiling is CENSORED "
                          "wherever its best checkpoint fell", "no cell is removed after its score is seen",
                          "a published number under another protocol never enters the comparison column"],
    }
    design["design_sha256"] = sha_obj(design)
    return design


def reuse_record(root: Path) -> dict:
    """The baseline reused from a CLOSED block: its design digest, recipe and the accepted terminals of modular_w60.
    Reuse is legitimate only when the complete contract matches (rows, inputs, scaler, recipe, cadence, seeds)."""
    root = Path(root)
    d = json.loads((root/"DESIGN.json").read_text())
    rep = json.loads((root/"REPORT.json").read_text())
    receipts = json.loads((root/"TERMINAL_RECEIPTS.json").read_text())["units"]
    if not rep.get("verified"):
        raise BlockRefusal("REFUSED: the block to reuse from is not closed and verified")
    if d["recipe"] != RECIPE or d["source_run"]["data_sha256"] != json.loads((Path(d["source_run"]["root"])/"DATA.json").read_text())["data_sha256"]:
        raise BlockRefusal("REFUSED: the baseline's recipe or rows differ; it cannot be reused")
    units = {c["cell_id"]: {"terminal_sha256": receipts[c["cell_id"]]["terminal_sha256"], "campaign_sha256": receipts[c["cell_id"]]["campaign_sha256"]}
             for c in d["cells"] if c["arm"] == "modular_w60"}
    return {"baseline_arm": "modular_w60", "run_root": str(root), "block": d["block"], "design_sha256": d["design_sha256"],
            "recipe": d["recipe"], "seeds": d["seeds"], "units": units,
            "rule": "reused because rows, inputs, scaler, recipe, cadence and seeds are identical; the baseline is NOT re-trained"}


def validate(design: dict) -> dict:
    """The design, checked against its own digest, its arms, the code and the prepared source it binds to."""
    B = _module("df_benchmark_contract")
    body = {k: v for k, v in design.items() if k != "design_sha256"}
    if design.get("schema") != SCHEMA or sha_obj(body) != design.get("design_sha256"):
        raise BlockRefusal("REFUSED: design digest/schema mismatch")
    for a in design["arms"]:
        if arm_spec(a["arm"]) != a:
            raise BlockRefusal(f"REFUSED: arm {a['arm']} differs from the registry")
    for name, digest in design["source_code"].items():
        if sha_file(HERE/name) != digest:
            raise BlockRefusal(f"REFUSED: scientific source changed: {name}")
    contract = B.require(design, purpose=f"the {design['block']} block")
    B.bind(contract, json.loads((Path(design["source_run"]["root"])/"DATA.json").read_text()), purpose=f"the {design['block']} block")
    return contract


# --- features and enumeration from row identities ----------------------------------------------------------------

def calendar_channels(frame, ts_format: str) -> np.ndarray:
    """The 4 calendar channels through the production path (each row's own label, nothing after it)."""
    C = _module("df_e1_calendar")
    spec = C.CalendarSpec(timestamp_column="timestamp_label", ts_format=ts_format, step_seconds=60)
    return C.build(frame, spec)["features"]


def hash_uniform(ids: np.ndarray, seed: int, stream: int) -> np.ndarray:
    """A deterministic uniform in [0, 1) per (seed, stream, id): a row's draw never depends on any other row."""
    with np.errstate(over="ignore"):                                              # 64-bit wraparound is the mixer
        s = np.asarray([seed], dtype=np.uint64)*np.uint64(0x9E3779B97F4A7C15) + np.asarray([stream], dtype=np.uint64)*np.uint64(0xBF58476D1CE4E5B9)
        x = ids.astype(np.uint64) + s
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def randomised_calendar_channels(ts, panel_rows: np.ndarray, seed: int = 20260921) -> np.ndarray:
    """The CAPACITY control: hour/weekday sin-cos of (label + a per-row random offset). The offset is a hash of
    (seed, panel row id), so the channel is deterministic, stable under prefix extension and carries no calendar
    information; its chance association with the true clock on a finite slice is measured, never declared zero."""
    hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy()/60.0
    weekday = ts.dt.dayofweek.to_numpy() + hour/24.0
    hour = (hour + 24.0*hash_uniform(panel_rows, seed, 1)) % 24.0
    weekday = (weekday + 7.0*hash_uniform(panel_rows, seed, 2)) % 7.0
    two_pi = 2.0*math.pi
    return np.stack([np.sin(two_pi*hour/24.0), np.cos(two_pi*hour/24.0),
                     np.sin(two_pi*weekday/7.0), np.cos(two_pi*weekday/7.0)], axis=1)


def daily_lag_channel(Y: np.ndarray, *, h: int, lag: int = DAY) -> np.ndarray:
    """At row r the channel holds y(r + h - lag): a row at or before r for h <= lag; NaN where that row is not in the array."""
    if h > lag:
        raise BlockRefusal(f"REFUSED: a lag of {lag} at horizon {h} would read a row AFTER the origin")
    out = np.full(Y.shape[0], np.nan)
    shift = lag - h
    out[shift:] = Y[:Y.shape[0]-shift]
    return out


def admissible_origins(ts_ns: np.ndarray, inputs_finite: np.ndarray, label_finite: np.ndarray, *, W: int, h: int,
                       lo: int, hi: int, step_seconds: int = 60) -> np.ndarray:
    """Origins t in [lo, hi) whose support rows t-W+1..t+h are grid-consecutive, whose window inputs are finite
    and whose label at t+h is finite — from the rows' own identities, nothing assumed."""
    n = ts_ns.shape[0]
    ok = np.ones(n, dtype=bool)
    ok[1:] = np.diff(ts_ns) == step_seconds*1_000_000_000
    ok[0] = True
    cg = np.concatenate([[0], np.cumsum(ok)])
    cf = np.concatenate([[0], np.cumsum(inputs_finite)])
    t = np.arange(max(lo, W-1), min(hi, n-h))
    if t.size == 0:
        return t
    span_ok = (cg[t+h+1]-cg[t-W+2]) == (W+h-1)
    fin = (cf[t+1]-cf[t-W+1]) == W
    return t[span_ok & fin & label_finite[t+h]]


def _coverage(origins: np.ndarray, back: int, forward: int, n: int) -> np.ndarray:
    """How many windows [t-back, t+forward] cover each row, by a difference array (O(n), no W x n index)."""
    d = np.zeros(n+1, dtype=np.int64)
    np.add.at(d, np.clip(origins-back, 0, n), 1)
    np.add.at(d, np.clip(origins+forward+1, 0, n), -1)
    return np.cumsum(d)[:n]


def counts_from_identities(train_origins: np.ndarray, eval_origins: np.ndarray, W: int, h: int, n_rows: int) -> dict:
    """Unique support rows, train-only rows, targets, windows and exposures — from the identities."""
    train_origins, eval_origins = np.asarray(train_origins, dtype=np.int64), np.asarray(eval_origins, dtype=np.int64)
    cov = _coverage(train_origins, W-1, 0, n_rows)
    support = cov > 0
    targets = np.zeros(n_rows, dtype=bool)
    targets[train_origins+h] = True
    val_support = _coverage(eval_origins, W-1, h, n_rows) > 0 if eval_origins.size else np.zeros(n_rows, dtype=bool)
    train_rows = support | targets
    train_only = train_rows & ~val_support
    return {"distinct_windows": int(train_origins.size), "labels": int(targets.sum()),
            "unique_support_rows": int(support.sum()), "unique_train_rows_incl_targets": int(train_rows.sum()),
            "train_only_rows_excluding_validation_support": int(train_only.sum()),
            "rows_shared_with_validation_support": int((train_rows & val_support).sum()),
            "mean_exposures_per_support_row": float(cov[support].mean()) if support.any() else 0.0,
            "reading": "consecutive windows overlap by W-1 rows; distinct windows are NOT independent observations; validation "
                       "support rows are excluded from the train-only count"}


def prepare(design: dict, root: Path, *, frame=None) -> dict:
    """Read the delivered panel rows the block declares, build every channel, enumerate every arm's origins from
    row identities, derive the COMMON evaluation mask, apply the COMMON scaler — before any score.
    `frame` is a pandas frame for tests on synthetic panels; the governed path reads the delivery."""
    import pandas as pd
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    validate(design)
    src = design["source_run"]
    rows = design["rows"]
    if frame is None:
        G = _module("df_e1_governed")
        delivered = G.require_delivery(root, design, "prepare")["delivery"]
        if delivered["sha256"] != src["panel_sha256"]:
            raise BlockRefusal("REFUSED: the delivered panel is not the source run's panel")
        frame = pd.read_parquet(delivered["path"])
    lo, hi, pad = int(rows["lo"]), int(rows["hi"]), int(rows["pad_rows"])
    if lo < 0 or hi > len(frame):
        raise BlockRefusal(f"REFUSED: rows [{lo}, {hi}) are not inside the panel of {len(frame)} rows")
    part = frame.iloc[lo:hi].reset_index(drop=True)
    cols = src["input_columns"]
    j = int(src["target_channel"])
    ts_format = "%d/%m/%Y %H:%M:%S"
    C = _module("df_e1_calendar")
    ts = C.parse_labels(part, C.CalendarSpec(timestamp_column="timestamp_label", ts_format=ts_format))
    ts_ns = ts.to_numpy().astype("datetime64[ns]").astype(np.int64)
    X = part[cols].to_numpy(dtype=np.float64)
    Y = X[:, j].copy()
    mean, sd = np.asarray(src["scaler"]["mean"], dtype=np.float64), np.asarray(src["scaler"]["sd"], dtype=np.float64)
    if mean.shape != (len(cols),) or not (np.isfinite(sd).all() and (sd > 0).all()):
        raise BlockRefusal("REFUSED: the common scaler is not a finite positive per-channel scaler of the declared columns")
    Xs = ((X-mean)/sd).astype(np.float32)
    cal = calendar_channels(part, ts_format).astype(np.float32)
    cal_rand = randomised_calendar_channels(ts, np.arange(lo, hi)).astype(np.float32)
    chance = {C.FEATURES[k]: float(np.corrcoef(cal[:, k], cal_rand[:, k])[0, 1]) for k in range(4)}
    lag_raw = daily_lag_channel(Y, h=H0)
    lag = ((lag_raw-mean[j])/sd[j]).astype(np.float32)
    inputs_finite = np.isfinite(X).all(axis=1)
    label_finite = np.isfinite(Y)
    train_end_local = int(rows["train_end"]) - lo
    hi_local = hi - lo
    origins, counts, feasibility = {}, {}, {}
    for a in design["arms"]:
        W = int(a["window"])
        need_lag = a["features"] == "daily_lag"
        fin = inputs_finite & np.isfinite(lag_raw) if need_lag else inputs_finite
        t_lo = train_end_local - int(a["train_days"])*DAY
        train = admissible_origins(ts_ns, fin, label_finite, W=W, h=H0, lo=t_lo + W0 - 1, hi=train_end_local - (W0+H0))
        val = admissible_origins(ts_ns, fin, label_finite, W=W, h=H0, lo=train_end_local, hi=hi_local)
        origins[a["arm"]] = {"train": train, "validation": val}
        feasibility[a["arm"]] = {"train_candidates": [t_lo + W0 - 1, train_end_local - (W0+H0)], "train_admissible": int(train.size),
                                 "validation_admissible": int(val.size), "window": W, "needs_lag": need_lag}
    # the COMMON evaluation set: admissible for every arm, finite label, finite daily lookup — as the source did
    common = None
    for a in design["arms"]:
        v = origins[a["arm"]]["validation"]
        common = v if common is None else np.intersect1d(common, v)
    lookup = common + H0 - DAY
    common = common[(lookup >= 0) & np.isfinite(Y[np.clip(lookup, 0, None)])]
    # binding to the source run: the 28 d baseline enumeration must reproduce the source's populations exactly
    with np.load(Path(src["root"])/"DATA.npz", allow_pickle=False) as z:
        src_train = z["train_origins"] + (src["slice_rows"][0]-lo)
        src_eval = z["eval_origins"] + (src["slice_rows"][0]-lo)
        src_Y = z["Y"]
    base = next((a for a in design["arms"] if a["window"] == W0 and a["train_days"] == 28 and a["features"] in ("base", "calendar", "randomised_calendar")), None)
    binding = {"baseline_arm_checked": base["arm"] if base else None}
    if base is not None:
        binding["train_origins_equal_source"] = bool(np.array_equal(origins[base["arm"]]["train"], src_train))
        if not binding["train_origins_equal_source"]:
            raise BlockRefusal("REFUSED: the block's enumeration does not reproduce the source run's train origins")
    if not np.array_equal(src_Y, Y[src["slice_rows"][0]-lo: src["slice_rows"][1]-lo], equal_nan=True):
        raise BlockRefusal("REFUSED: the target rows are not the source run's")
    binding["common_evaluation_equals_source"] = bool(np.array_equal(common, src_eval))
    binding["common_evaluation_subset_of_source"] = bool(np.isin(common, src_eval).all())
    if not binding["common_evaluation_subset_of_source"]:
        raise BlockRefusal("REFUSED: the common evaluation set is not inside the source run's evaluation origins")
    for a in design["arms"]:
        counts[a["arm"]] = counts_from_identities(origins[a["arm"]]["train"], common, int(a["window"]), H0, Y.shape[0])
    payload = {"Xs": Xs, "Y": Y, "calendar": cal, "calendar_randomised": cal_rand, "lag": lag, "lag_raw": lag_raw, "ts_ns": ts_ns,
               "scaler_mean": mean, "scaler_sd": sd, "target_channel": np.array([j]), "horizon": np.array([H0]),
               "row_offset": np.array([lo]), "train_end_local": np.array([train_end_local]), "common_eval": common}
    for arm, o in origins.items():
        payload[f"train_origins__{arm}"] = o["train"]
        payload[f"validation_origins__{arm}"] = o["validation"]
    np.savez(root/"BLOCK_DATA.npz", **payload)
    grid = C.grid_report(ts, C.CalendarSpec(timestamp_column="timestamp_label", ts_format=ts_format))
    rec = {"schema": SCHEMA_DATA, "design_sha256": design["design_sha256"], "rows": {"lo": lo, "hi": hi, "pad": pad, "n": int(hi-lo)},
           "input_columns": cols, "target_channel": j, "horizon": H0, "scaler": {"mean": mean.tolist(), "sd": sd.tolist(),
           "source": "COMMON: the source run's train-only scaler; calendar mean 0 / sd 1; lag = target scaler"},
           "sigma_evaluation": float(sd[j]), "grid": grid,
           "randomised_calendar_control": {"seed": 20260921, "chance_association_corr_with_true_calendar": chance,
                                           "reading": "finite-sample association of the control with the true clock; reported, not assumed zero"}, "feasibility": feasibility, "counts_from_identities": counts,
           "common_evaluation": {"n": int(common.size), "first_row": int(common.min()) if common.size else None,
                                 "last_row": int(common.max()) if common.size else None},
           "binding_to_source": binding, "data_sha256": sha_file(root/"BLOCK_DATA.npz")}
    (root/"BLOCK_DATA.json").write_text(json.dumps(rec, indent=1))
    return rec


def load_data(root: Path, design: dict) -> dict:
    rec = json.loads((Path(root)/"BLOCK_DATA.json").read_text())
    if rec["design_sha256"] != design["design_sha256"] or sha_file(Path(root)/"BLOCK_DATA.npz") != rec["data_sha256"]:
        raise BlockRefusal("REFUSED: BLOCK_DATA belongs to another design or was altered")
    with np.load(Path(root)/"BLOCK_DATA.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def arm_inputs(data: dict, spec: dict, *, assignment: list) -> tuple:
    """The channel matrix an arm consumes and its group assignment; the extra channels are declared groups."""
    Xs = data["Xs"]
    p0 = Xs.shape[1]
    j = int(data["target_channel"][0])
    if spec["features"] == "base":
        return Xs, list(assignment)
    if spec["features"] == "calendar":
        return np.concatenate([Xs, data["calendar"]], axis=1), list(assignment) + [max(assignment)+1]*4
    if spec["features"] == "randomised_calendar":
        return np.concatenate([Xs, data["calendar_randomised"]], axis=1), list(assignment) + [max(assignment)+1]*4
    if spec["features"] == "daily_lag":
        return np.concatenate([Xs, data["lag"][:, None]], axis=1), list(assignment) + [assignment[j]]
    raise BlockRefusal(f"REFUSED: unknown feature set {spec['features']!r}")


# --- models ---------------------------------------------------------------------------------------------------

def build_modular(assignment: list, W: int, p: int, j: int, seed: int, *, dilations: list | None = None, crop: int | None = None):
    """ARCH-A branches + tcn_w core, with EXPLICIT dilations and an optional raw-input crop before the extractor."""
    E = _module("df_mod_e0")
    P = _module("df_e1_pilot")
    tf = E._tf()
    tf.keras.utils.set_random_seed(int(seed))
    dil = list(dilations) if dilations is not None else P.core_dilations(crop or W)
    inp = tf.keras.Input(shape=(W, p), name="x")
    x = tf.keras.layers.Lambda(lambda t, c=crop: t[:, -c:, :], name=f"crop_last_{crop}")(inp) if crop else inp
    groups = sorted(set(assignment))
    branches = []
    for g in groups:
        idx = [k for k in range(p) if assignment[k] == g]
        sub = tf.keras.layers.Lambda(lambda t, idx=idx: tf.gather(t, idx, axis=2), name=f"g{g}_select")(x)
        branches.append(E.branch_extractor(tf, sub, f"g{g}", "A"))
    joint = tf.keras.layers.Concatenate(axis=2, name="fusion_seq")(branches) if len(branches) > 1 else branches[0]
    h = joint
    for i, d in enumerate(dil, start=1):
        h = E._tcn_block(tf, h, f"core_tcn{i}", d)
    read = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="core_last")(h)
    delta = tf.keras.layers.Dense(p, name="head")(read)
    last_x = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="last_observation")(x)
    full = tf.keras.layers.Add(name="persistence_skip")([last_x, delta])
    out = tf.keras.layers.Lambda(lambda t: t[:, j:j+1], name="target_readout")(full)
    return tf.keras.Model(inp, out, name=f"modular_A_tcnw_W{W}_crop{crop}_d{len(dil)}")


def build_model(spec: dict, assignment: list, p: int, j: int, seed: int):
    if spec["family"] == "gru":
        return _module("df_gru_reference").build(int(spec["window"]), p, j, seed)
    return build_modular(assignment, int(spec["window"]), p, j, seed, dilations=spec.get("dilations"), crop=spec.get("crop"))


def n_params(model) -> int:
    return int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))


def weight_hash(model) -> str:
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(str((w.shape, str(w.dtype))).encode())
        h.update(np.ascontiguousarray(w).tobytes())
    return h.hexdigest()


# --- training in observed updates ------------------------------------------------------------------------------

def _gather(Xs, origins, W):
    idx = origins[:, None] - W + 1 + np.arange(W)[None, :]
    return Xs[idx]


class Batches:
    """Windows gathered per batch from the scaled channel matrix; reshuffled per pass from (seed, pass)."""
    def __init__(self, Xs, Y, origins, W, h, j, batch, *, mean, sd, shuffle, seed):
        self.Xs, self.Y, self.o, self.W, self.h, self.j, self.batch = Xs, Y, np.asarray(origins), W, h, j, batch
        self.m, self.s, self.shuffle, self.seed, self.epoch = float(mean), float(sd), shuffle, int(seed), 0
        self.perm = np.arange(self.o.size)
        self._reshuffle()

    def _reshuffle(self):
        if self.shuffle:
            self.perm = np.random.default_rng([self.seed, self.epoch]).permutation(self.o.size)

    def __len__(self):
        return math.ceil(self.o.size/self.batch)

    def __getitem__(self, i):
        o = self.o[self.perm[i*self.batch:(i+1)*self.batch]]
        y = ((self.Y[o+self.h]-self.m)/self.s).astype(np.float32)[:, None]
        return _gather(self.Xs, o, self.W), y

    def on_epoch_end(self):
        self.epoch += 1
        self._reshuffle()


def predict(model, ds) -> np.ndarray:
    return np.concatenate([np.asarray(model.predict_on_batch(ds[i][0])) for i in range(len(ds))], axis=0)[:, 0]


def evaluate_mae(model, ds) -> float:
    pred = predict(model, ds)
    y = np.concatenate([ds[i][1][:, 0] for i in range(len(ds))])
    return float(np.mean(np.abs(pred.astype(np.float64)-y.astype(np.float64))))


def fit_by_updates(model, train, val, *, max_updates: int, validate_every: int, patience: int, lr: float, seed: int,
                   loss="mae", min_delta: float = 0.0, optimizer=None) -> dict:
    """The loop: one optimizer update per batch, validation every `validate_every` OBSERVED updates, patience in
    validation events, restore the best weights; the ceiling is CENSORING wherever the best event fell.
    The monitor is the validation MAE of the predictions (scaled units), whatever loss the arm trains on."""
    tf = _module("df_mod_e0")._tf()
    tf.keras.utils.set_random_seed(int(seed))
    opt = optimizer if optimizer is not None else tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss)
    updates, i, events = 0, 0, []
    best, best_weights, best_event = math.inf, model.get_weights(), 0
    running, stop = [], None
    t_fit, t_val = time.process_time(), 0.0
    while updates < max_updates:
        x, y = train[i]
        logs = model.train_on_batch(x, y, return_dict=True)
        running.append(float(logs["loss"]))
        i += 1
        updates += 1
        if i >= len(train):
            train.on_epoch_end()
            i = 0
        if updates % validate_every == 0 or updates == max_updates:
            t0 = time.process_time()
            v = evaluate_mae(model, val)
            t_val += time.process_time()-t0
            events.append({"update": updates, "val_mae_scaled": v, "train_loss_mean_since_last": float(np.mean(running))})
            running = []
            if v < best - min_delta:
                best, best_weights, best_event = v, model.get_weights(), len(events)
            elif len(events)-best_event >= patience:
                stop = "EARLY_STOPPING"
                break
    iterations = int(opt.iterations.numpy())
    if stop is None:
        stop = "UPDATE_BUDGET"
    model.set_weights(best_weights)
    restored = evaluate_mae(model, val)
    return {"updates": updates, "optimizer_iterations": iterations, "updates_are_optimizer_iterations": iterations == updates,
            "validate_every_updates": validate_every, "validation_events": len(events), "events": events,
            "best_event": best_event, "best_update": events[best_event-1]["update"] if best_event else None,
            "best_val_mae_scaled": best, "restored_val_mae_scaled": restored,
            "restore_verified": bool(abs(restored-best) <= 1e-6*max(1.0, abs(best))),
            "stop_reason": stop, "patience_events": patience, "min_delta": min_delta,
            "censoring": {"verdict": "CENSORED_BY_BUDGET" if stop == "UPDATE_BUDGET" else "STOPPED_ON_VALIDATION",
                          "rule": "reaching the update ceiling is censoring wherever the best checkpoint fell; what more budget "
                                  "would reach is unknown in either direction"},
            "fit_cpu_seconds": time.process_time()-t_fit, "validation_cpu_seconds": t_val}


# --- one cell (child process) ------------------------------------------------------------------------------------

def run_cell(design: dict, data: dict, cell: dict, out_dir: Path, *, pilot: bool) -> dict:
    cpu0, wall0 = time.process_time(), time.monotonic()
    spec = next(a for a in design["arms"] if a["arm"] == cell["arm"])
    R = design["recipe"]
    W, h, j = int(spec["window"]), int(data["horizon"][0]), int(data["target_channel"][0])
    X, assignment = arm_inputs(data, spec, assignment=design["source_run"]["graph_assignment"])
    p = X.shape[1]
    Y = data["Y"]
    m, sd = float(data["scaler_mean"][j]), float(data["scaler_sd"][j])
    train_o = data[f"train_origins__{spec['arm']}"]
    eval_o = data["common_eval"]
    train_end = int(data["train_end_local"][0])
    if pilot:
        # the cost pilot lives INSIDE train: it validates on the last 7 train days and never reads the DEV validation
        cut = train_end - 7*DAY
        pilot_train, pilot_val = train_o[train_o + h < cut - W], train_o[train_o >= cut]      # purge = the arm's own window
        train_o, eval_o = pilot_train, pilot_val
        max_updates, every, patience = int(cell["max_updates"]), int(cell["validate_every_updates"]), int(cell["patience_events"])
    else:
        max_updates, every, patience = int(R["max_updates"]), int(R["validate_every_updates"]), int(R["patience_events"])
    seed = int(cell["seed"])
    tr = Batches(X, Y, train_o, W, h, j, int(R["batch"]), mean=m, sd=sd, shuffle=True, seed=seed)
    va = Batches(X, Y, eval_o, W, h, j, int(R["batch"]), mean=m, sd=sd, shuffle=False, seed=seed)
    model = build_model(spec, assignment, p, j, seed)
    initial = weight_hash(model)
    training = fit_by_updates(model, tr, va, max_updates=max_updates, validate_every=every, patience=patience,
                              lr=float(R["learning_rate"]), seed=seed, loss=R["loss"], min_delta=float(R["min_delta"]))
    pred = predict(model, va).astype(np.float64)*sd + m
    y, naive = Y[eval_o+h], Y[eval_o]
    H = _module("df_e1_huber")
    score = H.metrics(pred, y, naive, sd)
    if not np.isclose(score["mae_z"], training["best_val_mae_scaled"], rtol=2e-5, atol=2e-6):
        raise BlockRefusal("REFUSED: the restored predictions do not reproduce the best validation MAE")
    out_dir.mkdir(parents=True, exist_ok=False)
    model.save_weights(out_dir/"weights.weights.h5")
    fresh = build_model(spec, assignment, p, j, seed)
    fresh.load_weights(out_dir/"weights.weights.h5")
    reload_pred = predict(fresh, va).astype(np.float64)*sd + m
    np.testing.assert_allclose(pred, reload_pred, atol=1e-6, rtol=1e-6)
    absolute = eval_o + int(data["row_offset"][0])
    np.savez(out_dir/"arrays.npz", pred=pred, y=y, naive=naive, origins=eval_o, origins_panel_rows=absolute, reload_pred=reload_pred)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    record = {"schema": "df_e1_block_cell.v1", "cell": cell, "arm_spec": spec, "design_sha256": design["design_sha256"],
              "pilot": pilot, "population": {"train_origins": int(train_o.size), "evaluation_origins": int(eval_o.size),
                                             "evaluation_is_common_dev_set": not pilot},
              "channels": p, "assignment": assignment, "parameters": n_params(model), "initial_weights_sha256": initial,
              "final_weights_sha256": weight_hash(model), "training": training, "scores": score,
              "target_mean": m, "target_sd": sd, "reload_max_error": float(np.max(np.abs(pred-reload_pred))),
              "cost": {"cpu_seconds": time.process_time()-cpu0, "wall_seconds": time.monotonic()-wall0,
                       "seconds_per_update": training["fit_cpu_seconds"]/max(1, training["updates"]),
                       "seconds_per_validation_event": training["validation_cpu_seconds"]/max(1, training["validation_events"]),
                       "peak_rss_bytes": int(ru.ru_maxrss)*1024, "host": os.uname().nodename},
              "arrays_sha256": sha_file(out_dir/"arrays.npz"), "weights_file_sha256": sha_file(out_dir/"weights.weights.h5")}
    write(out_dir/"cell.json", record)
    return record


def child(root: Path, unit: str) -> dict:
    design = json.loads((Path(root)/"DESIGN.json").read_text())
    validate(design)
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["child_cpu_seconds"], LIMITS["child_cpu_seconds"]+5))
    G = _module("df_e1_governed")
    delivered = G.require_delivery(root, design, unit)["delivery"]
    if delivered["sha256"] != design["source_run"]["panel_sha256"]:
        raise BlockRefusal("REFUSED: the unit's delivery is not the source panel")
    data = load_data(root, design)
    cell = next(c for c in design["pilots"] + design["cells"] if c["cell_id"] == unit)
    return run_cell(design, data, cell, Path(root)/"attempts"/unit, pilot=cell.get("role") == "COST_PILOT")


# --- governance: prepare, pilot, execute --------------------------------------------------------------------------

def _acquire(a, design, unit):
    G = _module("df_e1_governed")
    return G.acquire(run_id=a.run_id, root=a.root, lake=a.lake, resource=a.resource, unit_id=unit, gov_url=a.gov_url,
                     api_key_file=a.api_key_file, design_sha256=design["design_sha256"], cache_dir=Path(a.root)/"cache",
                     expect_sha256=design["source_run"]["panel_sha256"])


def _terminal_for(a, design, unit, cell, started, ok, rec, exit_code, wall):
    U = _module("df_utility_run")
    root = Path(a.root)
    cost = {"wall_seconds": wall}
    if rec:
        cost["cpu_seconds"] = rec["cost"]["cpu_seconds"]
    ms = [] if not rec else [U._metric(f"e1.block.{k}", float(v), "kW" if k.endswith("kw") else "dimensionless",
                                       split="validation" if not rec["pilot"] else "train_subset_pilot", horizon=H0)
                             for k, v in rec["scores"].items() if k != "rows"]
    artifacts = [] if not rec else [{"role": role, "sha256": sha_file(root/"attempts"/unit/f), "bytes": (root/"attempts"/unit/f).stat().st_size}
                                    for role, f in (("predictions", "arrays.npz"), ("weights", "weights.weights.h5"), ("record", "cell.json"))]
    terminal = U._terminal(status="COMPLETED" if ok else "FAILED", reason=None if ok else f"child exited {exit_code}; see the retained log",
                           cost=cost, metrics=ms, started=started, finished=U._z(U.now_iso()),
                           tags={"purpose": design["purpose"], "classification": "NON_GOVERNING", "phase": "DEVELOPMENT", "unit": unit,
                                 "arm": cell["arm"], "seed": str(cell["seed"]), "role": cell.get("role", "ARM"),
                                 "design_sha256": design["design_sha256"], "contract_sha256": design["benchmark_contract"]["contract_sha256"]})
    terminal["artifacts"] = artifacts
    return terminal


def governance_modules():
    """The legacy dynamic loader publishes a module before executing its body: complete the governance imports on
    the parent thread BEFORE children acquire in parallel (the huber runner learned this the hard way)."""
    G = _module("df_e1_governed")
    G._load("governed_run")
    G._load("df_e1_receipts")
    return G, _module("df_utility_run")


def run_units(a, design, units, *, parallel: int) -> list:
    G, U = governance_modules()
    root = Path(a.root)
    results = []

    def one(cell):
        unit = cell["cell_id"]
        term_path, rec_path = root/"TERMINALS"/f"{unit}.json", root/"attempts"/unit/"cell.json"
        if term_path.exists():
            # an accepted, recorded unit is REUSED, never re-trained to repair a receipt; anything else is a refusal
            receipts = json.loads((root/"TERMINAL_RECEIPTS.json").read_text()).get("units", {}) if (root/"TERMINAL_RECEIPTS.json").is_file() else {}
            held = json.loads(term_path.read_text())
            if held.get("status") == "COMPLETED" and rec_path.is_file() and unit in receipts:
                print(json.dumps({"unit": unit, "reused": True}), flush=True)
                return {"unit": unit, "ok": True, "record": json.loads(rec_path.read_text()), "terminal_sha256": receipts[unit].get("terminal_sha256"), "reused": True}
            raise BlockRefusal(f"REFUSED: {unit} already has a terminal that is not an accepted COMPLETED record; experiments are not silently repeated")
        started = U._z(U.now_iso())
        _acquire(a, design, unit)
        wall = time.monotonic()
        with open(root/f"{unit}.log", "x") as log:
            try:
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child", "--root", str(root), "--unit", unit],
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                                           "TF_CPP_MIN_LOG_LEVEL": "3"}, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=LIMITS["child_wall_seconds"])
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = "WALL_TIMEOUT"
        rec_path = root/"attempts"/unit/"cell.json"
        ok = code == 0 and rec_path.exists()
        rec = json.loads(rec_path.read_text()) if ok else None
        terminal = _terminal_for(a, design, unit, cell, started, ok, rec, code, time.monotonic()-wall)
        (root/"TERMINALS").mkdir(exist_ok=True)
        write(root/"TERMINALS"/f"{unit}.json", terminal)
        reported = G.report_terminal(root, unit, terminal, gov_url=a.gov_url, api_key_file=a.api_key_file,
                                     outbox_dir=str(root/"outbox"), started_at=started)
        if reported["flushed"]["pending"] or reported["flushed"]["failures"]:
            raise BlockRefusal(f"REFUSED: the terminal of {unit} was not accepted: {reported['flushed']['failures']}")
        print(json.dumps({"unit": unit, "ok": ok, "mae_z": rec["scores"]["mae_z"] if rec else None,
                          "stop": rec["training"]["stop_reason"] if rec else None,
                          "cpu": rec["cost"]["cpu_seconds"] if rec else None}), flush=True)
        return {"unit": unit, "ok": ok, "record": rec, "terminal_sha256": (reported.get("receipt") or {}).get("terminal_sha256")}

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for start in range(0, len(units), parallel):
            batch = list(pool.map(one, units[start:start+parallel]))
            results += batch
            if not all(r["ok"] for r in batch):
                raise BlockRefusal("REFUSED: a child failed; the remaining cells are not started")
    return results


def spent_cpu(root: Path) -> float:
    return float(sum(json.loads(p.read_text())["cost"]["cpu_seconds"] for p in (Path(root)/"attempts").glob("*/cell.json")))


def projection(design: dict, pilots: list) -> dict:
    """What the block would cost at the ceiling, from each arm's OWN pilot; the ceiling is never assumed reached early."""
    R = design["recipe"]
    events = math.ceil(R["max_updates"]/R["validate_every_updates"])
    per_arm = {}
    for r in pilots:
        rec = r["record"]
        per_arm[rec["cell"]["arm"]] = {"seconds_per_update": rec["cost"]["seconds_per_update"],
                                       "seconds_per_validation_event_pilot": rec["cost"]["seconds_per_validation_event"],
                                       "pilot_validation_rows": rec["population"]["evaluation_origins"],
                                       "peak_rss_bytes": rec["cost"]["peak_rss_bytes"]}
    n_eval = design["source_run"]["evaluation_origins"]
    cells = {}
    for c in design["cells"]:
        pa = per_arm[c["arm"]]
        val_scale = n_eval/max(1, pa["pilot_validation_rows"])
        cells[c["cell_id"]] = pa["seconds_per_update"]*R["max_updates"] + pa["seconds_per_validation_event_pilot"]*val_scale*events
    total = sum(cells.values())
    return {"per_arm": per_arm, "per_cell_at_ceiling_seconds": cells, "total_at_ceiling_seconds": total,
            "with_headroom_25_percent": total*1.25, "closure_reserve_seconds": LIMITS["closure_reserve_seconds"],
            "assumption": "every cell runs to the ceiling with every validation event; early stopping can only lower it"}


def cmd_prepare(a):
    design = json.loads((Path(a.root)/"DESIGN.json").read_text())
    validate(design)
    G, U = governance_modules()
    root = Path(a.root)
    started = U._z(U.now_iso())
    _acquire(a, design, "prepare")
    t0 = time.process_time()
    try:
        rec = prepare(design, root)
    except BaseException as exc:
        G.report_failed(root, "prepare", f"prepare refused: {str(exc)[:200]}", gov_url=a.gov_url, api_key_file=a.api_key_file)
        raise
    # the prepare unit closes like any other: a COMPLETED terminal with the prepared data's digests as artifacts
    terminal = U._terminal(status="COMPLETED", reason=None, cost={"wall_seconds": time.process_time()-t0, "cpu_seconds": time.process_time()-t0},
                           metrics=[U._metric("e1.block.common_evaluation_rows", rec["common_evaluation"]["n"], "rows", split="validation", horizon=H0)],
                           started=started, finished=U._z(U.now_iso()),
                           tags={"purpose": design["purpose"], "classification": "NON_GOVERNING", "phase": "DEVELOPMENT", "unit": "prepare",
                                 "role": "PREPARATION", "design_sha256": design["design_sha256"]})
    terminal["artifacts"] = [{"role": r, "sha256": sha_file(root/f), "bytes": (root/f).stat().st_size} for r, f in (("data", "BLOCK_DATA.npz"), ("record", "BLOCK_DATA.json"))]
    (root/"TERMINALS").mkdir(exist_ok=True)
    write(root/"TERMINALS"/"prepare.json", terminal)
    reported = G.report_terminal(root, "prepare", terminal, gov_url=a.gov_url, api_key_file=a.api_key_file, outbox_dir=str(root/"outbox"), started_at=started)
    if reported["flushed"]["pending"] or reported["flushed"]["failures"]:
        raise BlockRefusal(f"REFUSED: the prepare terminal was not accepted: {reported['flushed']['failures']}")
    print(json.dumps({k: rec[k] for k in ("common_evaluation", "binding_to_source", "feasibility")}, indent=1))


def cmd_pilot(a):
    design = json.loads((Path(a.root)/"DESIGN.json").read_text())
    validate(design)
    load_data(a.root, design)
    results = run_units(a, design, design["pilots"], parallel=min(LIMITS["parallel_children"], len(design["pilots"])))
    proj = projection(design, results)
    fits = proj["with_headroom_25_percent"] + spent_cpu(a.root) + LIMITS["closure_reserve_seconds"] <= LIMITS["campaign_cpu_seconds"]
    doc = {"schema": "df_e1_block_pilot_report.v1", "design_sha256": design["design_sha256"], "spent_cpu_seconds": spent_cpu(a.root),
           "projection": proj, "fits_the_ceiling": fits, "decision": "EXECUTE" if fits else "BUDGET_LIMITED_BEFORE_ANY_OUTCOME"}
    (Path(a.root)/"REPORT.pilot.json").write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps(doc, indent=1))
    return 0 if fits else 2


def cmd_execute(a):
    design = json.loads((Path(a.root)/"DESIGN.json").read_text())
    validate(design)
    load_data(a.root, design)
    pilot = json.loads((Path(a.root)/"REPORT.pilot.json").read_text())
    if pilot["decision"] != "EXECUTE":
        raise BlockRefusal("REFUSED: the cost pilot did not project inside the ceiling")
    run_units(a, design, design["cells"], parallel=LIMITS["parallel_children"])
    print(json.dumps({"spent_cpu_seconds": spent_cpu(a.root)}))


# --- closure -------------------------------------------------------------------------------------------------------

def baselines(data: dict, design: dict) -> dict:
    """Three DISTINCT references on the common evaluation rows: persistence, daily seasonal persistence, train-only constant."""
    H = _module("df_e1_huber")
    h, j = int(data["horizon"][0]), int(data["target_channel"][0])
    Y, o = data["Y"], data["common_eval"]
    sd = float(data["scaler_sd"][j])
    y, naive = Y[o+h], Y[o]
    base_arm = next(a["arm"] for a in design["arms"] if a["train_days"] == 28)
    train_o = data[f"train_origins__{base_arm}"]
    const = float(np.mean(Y[train_o+h]))
    out = {}
    for name, pred in (("persistence", naive), ("daily_seasonal", Y[o+h-DAY]), ("train_constant", np.full(o.size, const))):
        out[name] = {**H.metrics(pred, y, naive, sd), "definition": design["baselines"][name]}
    return out


def close(a) -> dict:
    root = Path(a.root)
    design = json.loads((root/"DESIGN.json").read_text())
    validate(design)
    data = load_data(root, design)
    C = _module("df_mod_e0_close")
    H = _module("df_e1_huber")
    receipts = json.loads((root/"TERMINAL_RECEIPTS.json").read_text())["units"]
    expected = {c["cell_id"] for c in design["cells"]}
    problems, rows, inits = [], [], {}
    if not expected <= set(receipts):
        problems.append(f"population: units without an accepted terminal: {sorted(expected-set(receipts))}")
    token = Path(a.warehouse_token_file).read_text().strip().strip('"').strip("'") if a.warehouse_token_file else None
    h, j = int(data["horizon"][0]), int(data["target_channel"][0])
    o = data["common_eval"]
    truth, naive = data["Y"][o+h], data["Y"][o]
    for cell in design["cells"]:
        unit = cell["cell_id"]
        folder = root/"attempts"/unit
        if not (folder/"cell.json").is_file():
            problems.append(f"{unit}: no record")
            continue
        rec = json.loads((folder/"cell.json").read_text())
        if rec["design_sha256"] != design["design_sha256"] or rec["cell"] != cell:
            problems.append(f"{unit}: cell/design mismatch")
        if sha_file(folder/"arrays.npz") != rec["arrays_sha256"]:
            problems.append(f"{unit}: arrays digest mismatch")
        with np.load(folder/"arrays.npz", allow_pickle=False) as z:
            for name, actual, wanted in (("origins", z["origins"], o), ("y", z["y"], truth), ("naive", z["naive"], naive)):
                if not np.array_equal(actual, wanted):
                    problems.append(f"{unit}: {name} are not the common evaluation rows")
            if not np.allclose(z["pred"], z["reload_pred"], rtol=1e-6, atol=1e-6):
                problems.append(f"{unit}: reload parity")
            score = H.metrics(z["pred"], z["y"], z["naive"], rec["target_sd"])
        if score != rec["scores"]:
            problems.append(f"{unit}: record does not match arrays")
        inits.setdefault((cell["seed"], rec["arm_spec"]["family"], rec["channels"], rec["arm_spec"]["window"]), set()).add(rec["initial_weights_sha256"])
        terminal = json.loads((root/"TERMINALS"/f"{unit}.json").read_text())
        if token and unit in receipts:
            held = C.warehouse_terminals(a.warehouse_url, token, receipts[unit]["campaign_sha256"])["current"]
            row = held.get(unit, {})
            if row.get("terminal_sha256") != receipts[unit]["terminal_sha256"] or row.get("status") != "COMPLETED":
                problems.append(f"{unit}: warehouse terminal digest/status")
            if sorted((x["role"], x["sha256"], x["bytes"]) for x in row.get("artifacts", [])) != \
                    sorted((x["role"], x["sha256"], x["bytes"]) for x in terminal["artifacts"]):
                problems.append(f"{unit}: warehouse artifacts")
        rows.append({**cell, **score, "parameters": rec["parameters"], "updates": rec["training"]["updates"],
                     "validation_events": rec["training"]["validation_events"], "best_update": rec["training"]["best_update"],
                     "stop": rec["training"]["stop_reason"], "censoring": rec["training"]["censoring"]["verdict"],
                     "cpu_seconds": rec["cost"]["cpu_seconds"], "peak_rss_bytes": rec["cost"]["peak_rss_bytes"],
                     "reload_max_error": rec["reload_max_error"]})
    unpaired = [k for k, v in inits.items() if len(v) != 1]
    if unpaired:
        problems.append(f"unpaired initial weights within a seed for the same graph: {unpaired}")
    arms = sorted({r["arm"] for r in rows})
    summary = {}
    for arm in arms:
        v = [r["mae_z"] for r in rows if r["arm"] == arm]
        summary[arm] = {"n_seeds": len(v), "mean_mae_z": float(np.mean(v)), "sd_mae_z": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                        "mean_mae_kw": float(np.mean([r["mae_kw"] for r in rows if r["arm"] == arm])),
                        "censored_fits": sum(1 for r in rows if r["arm"] == arm and r["censoring"] == "CENSORED_BY_BUDGET")}
    paired = {}
    if len(arms) == 2:
        a0, a1 = arms
        d = [next(r["mae_z"] for r in rows if r["arm"] == a1 and r["seed"] == s) -
             next(r["mae_z"] for r in rows if r["arm"] == a0 and r["seed"] == s) for s in design["seeds"]
             if any(r["arm"] == a1 and r["seed"] == s for r in rows) and any(r["arm"] == a0 and r["seed"] == s for r in rows)]
        paired = {"difference": f"{a1} - {a0} in MAE_z, paired by seed", "values": d, "mean": float(np.mean(d)) if d else None,
                  "signs": {"positive": sum(x > 0 for x in d), "negative": sum(x < 0 for x in d)},
                  "reading": "three seeds; both signs are reported; no interval is claimed from n=3"}
    report = {"schema": "df_e1_block_report.v1", "design_sha256": design["design_sha256"], "block": design["block"],
              "common_evaluation_rows": int(o.size), "sigma_evaluation": float(data["scaler_sd"][j]),
              "rows": rows, "summary": summary, "paired": paired, "baselines": baselines(data, design),
              "problems": problems, "verified": not problems, "spent_cpu_seconds": spent_cpu(root),
              "scope": "DEVELOPMENT; paired seeds; one previously inspected DEV validation week; no test rows read"}
    (root/"REPORT.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ("summary", "paired", "baselines", "problems", "verified")}, indent=1, default=str))
    if problems:
        raise BlockRefusal("REFUSED: closure failed")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["seal", "prepare", "pilot", "execute", "child", "close"])
    ap.add_argument("--block")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--unit")
    ap.add_argument("--run-id")
    ap.add_argument("--gov-url", default="http://127.0.0.1:5055")
    ap.add_argument("--api-key-file", type=Path)
    ap.add_argument("--lake", default=LAKE)
    ap.add_argument("--resource", default=RESOURCE)
    ap.add_argument("--warehouse-url", default="http://127.0.0.1:5057")
    ap.add_argument("--warehouse-token-file", type=Path)
    ap.add_argument("--reuse-from", type=Path, help="a CLOSED block root whose modular_w60 cells are this block's baseline (contract must match)")
    a = ap.parse_args(argv)
    if a.command == "seal":
        d = seal(a.block, source_run=a.source_run, reuse=reuse_record(a.reuse_from) if a.reuse_from else None)
        write(a.out, d)
        print(json.dumps({"design_sha256": d["design_sha256"], "block": d["block"], "cells": len(d["cells"]), "pilots": len(d["pilots"])}, indent=1))
        return 0
    if a.command == "child":
        child(a.root, a.unit)
        return 0
    if a.command == "prepare":
        cmd_prepare(a)
        return 0
    if a.command == "pilot":
        return cmd_pilot(a)
    if a.command == "execute":
        cmd_execute(a)
        return 0
    close(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
