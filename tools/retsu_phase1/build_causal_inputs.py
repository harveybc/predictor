#!/usr/bin/env python
"""Causal inputs for the seasonal-increment repeat.

Two corrections, and nothing else:

1. Kalman (q_level, q_slope, r), the GMM, and the z-scaler at bar t use only
   rows strictly before t. Kalman and the GMM refit at each calendar month on
   rows before that month, then a forward filter / row-wise posterior keeps
   that month. The scaler is an expanding mean and std (ddof 0) of rows
   strictly before t, with a zero-std denominator of 1. Burn-in is 2000 finite
   past rows. No fit uses H2 2023 or 2024 as a block that contains the bar.

2. The column set is every output of the declared causal producers. It is not
   the 61-column subset, and it is not ranked. The pretrained model.pt is not
   read. Weights are a fresh seed-7 build of the same temporal architecture.

The seasonal target is the saved causal-STL endpoint. This script does not
refit STL. A future perturbation on a prefix must leave the past unchanged,
and the bar at the cut must move.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PREDICTOR = Path("/var/tmp/cpu398.Xn9J2L/predictor")
RUNTIME = Path("/var/tmp/sac399-series/runtime")
sys.path.insert(0, str(PREDICTOR))
sys.path.insert(0, str(RUNTIME))

from tools.layer1_pool_h1 import candle_pips, causal_ssa  # noqa: E402
from tools.layer2_pool_h1 import (  # noqa: E402
    CFG,
    apply_gmm,
    fam_hilbert,
    fam_wavelet,
    fit_gmm,
    parents,
)
from tools.lib import kalman_llt  # noqa: E402
from tools.timing_event_clock import event_clock  # noqa: E402
from tools.timing_price_pool import price_pool  # noqa: E402
from app.rl_subset import seal_recipe  # noqa: E402

RAW = Path("/tmp/retsu-causal/raw")
OUT = Path("/tmp/retsu-causal/out")
RECIPE_TEMPLATE = Path("/var/tmp/sac399-series/model/61/bundle/RECIPE.json")
HOUR = 3_600_000_000_000
PIP = 1e-4
MIN_COUNT = 2000
ATR_RATIO_CONSTANT = 0.00131325
OFFSETS = np.arange(-24, 1, 2, dtype=np.int64)
PROBE_BARS = 4200
PROBE_CUT = 3800
FORBIDDEN_IDENTITY = {"recipe_b8fa85625fe025a7", "recipe_b5fb7bcdfa1dbec8"}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_grid() -> tuple[pd.DataFrame, list]:
    hist = pd.read_parquet(RAW / "OHLC.parquet")
    hist.index = pd.DatetimeIndex(hist.index)
    if hist.index.tz is None:
        hist.index = hist.index.tz_localize("UTC")
    else:
        hist.index = hist.index.tz_convert("UTC")
    bars = pd.read_parquet(RAW / "bars.parquet")
    live = pd.DataFrame(
        {
            "OPEN": bars["BidOpen"].to_numpy(float),
            "HIGH": bars["BidHigh"].to_numpy(float),
            "LOW": bars["BidLow"].to_numpy(float),
            "CLOSE": bars["BidClose"].to_numpy(float),
        },
        index=pd.DatetimeIndex(pd.to_datetime(bars["timestamp"], utc=True)),
    )
    grid = pd.concat([hist[["OPEN", "HIGH", "LOW", "CLOSE"]], live], axis=0)
    grid = grid[grid.index < "2025-01-01"].sort_index()
    grid = grid.astype(float)
    if grid.index.has_duplicates:
        raise SystemExit("DUPLICATE_TIMESTAMPS")
    if not grid.index.is_monotonic_increasing:
        raise SystemExit("UNSORTED_GRID")
    if not np.isfinite(grid.to_numpy(float)).all():
        raise SystemExit("NONFINITE_OHLC")
    if grid.index[0] >= pd.Timestamp("2016-01-05", tz="UTC"):
        raise SystemExit("GRID_DOES_NOT_COVER_2016")
    closures = json.loads((RAW / "CLOSURES.json").read_text())
    print(
        f"GRID {len(grid)} {grid.index[0]} -> {grid.index[-1]}",
        flush=True,
    )
    return grid, closures


def _parents(grid: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": grid.index,
            "BidOpen": grid["OPEN"].to_numpy(float),
            "BidHigh": grid["HIGH"].to_numpy(float),
            "BidLow": grid["LOW"].to_numpy(float),
            "BidClose": grid["CLOSE"].to_numpy(float),
        }
    )
    return parents(frame)


def _indicators(grid: pd.DataFrame) -> pd.DataFrame:
    import pandas_ta as ta

    high, low, close = grid["HIGH"], grid["LOW"], grid["CLOSE"]
    atr = ta.atr(high, low, close, length=14).reindex(grid.index)
    bands = ta.bbands(close, length=20, std=2.0).reindex(grid.index)
    upper = bands[[c for c in bands.columns if c.startswith("BBU_")][0]]
    lower = bands[[c for c in bands.columns if c.startswith("BBL_")][0]]
    mid = bands[[c for c in bands.columns if c.startswith("BBM_")][0]]
    width = (upper - lower) / mid.replace(0.0, np.nan)
    position = (close - lower) / (upper - lower).replace(0.0, np.nan)
    stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3).reindex(grid.index)
    stoch_d = stoch[[c for c in stoch.columns if c.startswith("STOCHd_")][0]]
    cci = ta.cci(high, low, close, length=20).reindex(grid.index)
    rolling = close.rolling(24, min_periods=24).std(ddof=1)
    return pd.DataFrame(
        {
            "ATR": atr.to_numpy(float),
            "BB_Width": width.to_numpy(float),
            "BB_position": position.to_numpy(float),
            "Stochastic_D": stoch_d.to_numpy(float),
            "CCI": cci.to_numpy(float),
            "rolling_std_24": rolling.to_numpy(float),
            "ATR_ratio": ATR_RATIO_CONSTANT / atr.to_numpy(float),
        },
        index=grid.index,
    )


def _raw_returns(grid: pd.DataFrame) -> pd.DataFrame:
    opened = grid["OPEN"]
    close = grid["CLOSE"]
    return pd.DataFrame(
        {
            "OPEN@diff": opened.diff().to_numpy(float),
            "RAW_BAR_LOG_RETURN": np.log(close.to_numpy(float) / opened.to_numpy(float)),
            "RAW_LOG_RANGE": np.log(grid["HIGH"].to_numpy(float) / grid["LOW"].to_numpy(float)),
            "RAW_CLOSE_LOG_RETURN": np.log(close / close.shift(1)).to_numpy(float),
        },
        index=grid.index,
    )


def _learned(parent: pd.DataFrame, min_count: int) -> tuple[pd.DataFrame, list]:
    """Month-expanding Kalman and GMM. Params for month M see only rows before M."""
    index = parent.index
    n = len(parent)
    y = parent["log_close"].to_numpy(float)
    months = index.tz_convert("UTC").to_period("M")
    change = np.flatnonzero(months.astype(str)[1:] != months.astype(str)[:-1]) + 1
    bounds = np.r_[0, change, n]
    level = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    innovation = np.full(n, np.nan)
    innovation_std = np.full(n, np.nan)
    posterior = np.full((n, 4), np.nan)
    schedule = []
    cfg_k = CFG["kalman"]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if int(a) < min_count:
            continue
        prm = kalman_llt.fit_llt(y[: int(a)], burn_in=cfg_k["burn_in"], maxiter=cfg_k["maxiter"])
        lvl, slp, inn, inv = kalman_llt.filter_llt(y[: int(b)], prm["q_level"], prm["q_slope"], prm["r"])
        level[a:b] = lvl[a:b] - y[a:b]
        slope[a:b] = slp[a:b]
        innovation[a:b] = inn[a:b]
        with np.errstate(divide="ignore", invalid="ignore"):
            innovation_std[a:b] = inn[a:b] / np.sqrt(inv[a:b])
        gmm = fit_gmm(parent.iloc[: int(a)], np.ones(int(a), dtype=bool))
        post, _ = apply_gmm(parent.iloc[int(a) : int(b)], gmm)
        posterior[a:b] = post.to_numpy(float)
        schedule.append(
            {
                "month": str(index[int(a)].tz_convert("UTC").to_period("M")),
                "fit_rows_before_month": int(a),
                "month_rows": int(b - a),
                "q_level": prm["q_level"],
                "q_slope": prm["q_slope"],
                "r": prm["r"],
                "gmm_fit_rows": int(gmm["fit_rows"]),
                "gmm_weights": gmm["weights"],
            }
        )
        print(
            f"LEARNED {schedule[-1]['month']} fit_before {a} rows {b - a}",
            flush=True,
        )
    frame = pd.DataFrame(
        {
            "kf.lc.level_minus_close": level,
            "kf.lc.slope": slope,
            "kf.lc.innovation": innovation,
            "kf.lc.innovation_std": innovation_std,
            "rg.gmm.p0": posterior[:, 0],
            "rg.gmm.p1": posterior[:, 1],
            "rg.gmm.p2": posterior[:, 2],
            "rg.gmm.entropy": posterior[:, 3],
        },
        index=index,
    )
    return frame, schedule


def raw_features(grid: pd.DataFrame, closures) -> tuple[pd.DataFrame, list]:
    """Complete outputs of the declared causal producers, in group order."""
    started = time.time()
    px = price_pool(grid, clock="utc", closures=closures)
    print(f"PRICE {time.time() - started:.1f}s cols {px.shape[1]}", flush=True)
    parent = _parents(grid)
    started = time.time()
    waves, _ = fam_wavelet(parent)
    print(f"WAVELET {time.time() - started:.1f}s cols {waves.shape[1]}", flush=True)
    started = time.time()
    hilbert, _ = fam_hilbert(parent)
    print(f"HILBERT {time.time() - started:.1f}s cols {hilbert.shape[1]}", flush=True)
    started = time.time()
    ssa = causal_ssa(grid["CLOSE"].to_numpy(float))
    ssa_frame = pd.DataFrame(
        {
            "ssa.trend_minus_close_pips": (ssa[:, 0] - grid["CLOSE"].to_numpy(float)) / PIP,
            "ssa.c2_pips": ssa[:, 1] / PIP,
            "ssa.c3_pips": ssa[:, 2] / PIP,
        },
        index=grid.index,
    )
    print(f"SSA {time.time() - started:.1f}s", flush=True)
    started = time.time()
    learned, schedule = _learned(parent, MIN_COUNT)
    print(f"KALMAN_GMM {time.time() - started:.1f}s months {len(schedule)}", flush=True)
    candles = candle_pips(grid)
    events = event_clock(grid.index)
    inventory = _indicators(grid)
    returns = _raw_returns(grid)
    for name, frame in (("inventory", inventory), ("returns", returns), ("candles", candles), ("events", events)):
        if not frame.index.equals(grid.index):
            raise SystemExit(f"INDEX_{name}")

    def take(columns):
        missing = [c for c in columns if c not in source.columns]
        if missing:
            raise SystemExit("MISSING_COLUMNS " + " ".join(missing))
        return source[columns]

    source = pd.concat([px, candles, events, inventory, returns, waves, hilbert, ssa_frame, learned], axis=1)
    source = source.copy()
    families = [
        ("inv.raw", "pointwise", ["OPEN@diff", "RAW_BAR_LOG_RETURN", "RAW_LOG_RANGE", "RAW_CLOSE_LOG_RETURN"]),
        ("inv.volatility", "conv", ["ATR", "BB_Width", "ATR_ratio", "rolling_std_24"]),
        ("inv.momentum", "pointwise", ["Stochastic_D", "CCI", "BB_position"]),
        ("calendar", "conv", [c for c in px.columns if c.startswith("cal.")]),
        ("px.logret", "conv", [c for c in px.columns if c.startswith("px.logret_")]),
        ("px.range", "conv", [c for c in px.columns if c.startswith("px.range_") and not c.startswith("px.range_ratio")]),
        ("px.close_loc", "conv", [c for c in px.columns if c.startswith("px.close_loc_")]),
        ("px.er", "conv", [c for c in px.columns if c.startswith("px.er_")]),
        ("px.rv", "conv", [c for c in px.columns if c.startswith("px.rv_")]),
        ("px.atr", "conv", [c for c in px.columns if c.startswith("px.atr_")]),
        ("px.range_ratio", "conv", [c for c in px.columns if c.startswith("px.range_ratio")]),
        ("px.bar_body", "conv", ["px.bar_body"]),
        ("px.elapsed_hours", "conv", ["px.elapsed_hours"]),
        ("px.hours_since_session_open", "conv", ["px.hours_since_session_open"]),
        ("px.unexpected_gap_hours", "conv", ["px.unexpected_gap_hours"]),
        ("cdl", "conv", list(candles.columns)),
        ("ev", "conv", list(events.columns)),
        ("ssa/BidClose", "conv", list(ssa_frame.columns)),
        ("wavelet/log_close", "conv", [c for c in waves.columns if c.startswith("wv.lc.")]),
        ("wavelet/logret_1", "conv", [c for c in waves.columns if c.startswith("wv.r1.")]),
        ("hilbert/logret_1", "conv", [c for c in hilbert.columns if c.startswith("hb.r1.")]),
        ("hilbert/log_close", "conv", [c for c in hilbert.columns if c.startswith("hb.lc_detr.")]),
        ("kalman/log_close", "conv", [c for c in learned.columns if c.startswith("kf.")]),
        ("gmm/logret_1+rv_24+log_range", "conv", [c for c in learned.columns if c.startswith("rg.gmm.")]),
    ]
    ordered = []
    group_rows = []
    cursor = 0
    for name, kind, columns in families:
        if not columns:
            raise SystemExit(f"EMPTY_FAMILY {name}")
        if kind == "pointwise" and name not in ("inv.raw", "inv.momentum"):
            raise SystemExit(f"POINTWISE_FAMILY {name}")
        if kind != "pointwise" and kind != "conv":
            raise SystemExit(f"KIND {name}")
        block = take(columns)
        ordered.append(block)
        group_rows.append(
            {
                "name": name,
                "kind": kind,
                "columns": list(columns),
                "indices": list(range(cursor, cursor + len(columns))),
                "units": 2,
            }
        )
        cursor += len(columns)
    frame = pd.concat(ordered, axis=1)
    frame.index = grid.index
    # 1-bar efficiency is 1 wherever the bar moved and undefined (0/0) on a flat bar.
    # The undefined rows are filled with 1, the only value the producer emits, so a flat
    # bar does not punch a hole in the window. The column is then constant.
    if "px.er_1" in frame.columns:
        frame["px.er_1"] = frame["px.er_1"].fillna(1.0)
    if list(frame.columns) != [c for group in group_rows for c in group["columns"]]:
        raise SystemExit("COLUMN_ORDER")
    if frame.columns.duplicated().any():
        raise SystemExit("DUPLICATE_COLUMNS")
    used = set(frame.columns)
    produced = set(source.columns)
    if used != produced:
        raise SystemExit("PRODUCER_COVERAGE " + " ".join(sorted(produced - used)[:20]))
    return frame, group_rows, schedule


def expanding_z(raw: np.ndarray, min_count: int) -> np.ndarray:
    """z[t] uses finite rows strictly before t. A zero std becomes 1. Count below min_count is NaN."""
    finite = np.isfinite(raw)
    values = np.where(finite, raw, 0.0)
    count = np.cumsum(finite, axis=0)
    total = np.cumsum(values, axis=0)
    total2 = np.cumsum(values * values, axis=0)
    count_b = np.vstack([np.zeros((1, raw.shape[1]), float), count[:-1]])
    total_b = np.vstack([np.zeros((1, raw.shape[1]), float), total[:-1]])
    total2_b = np.vstack([np.zeros((1, raw.shape[1]), float), total2[:-1]])
    mean = np.divide(total_b, count_b, out=np.zeros_like(total_b), where=count_b > 0)
    var = np.maximum(total2_b / np.maximum(count_b, 1.0) - mean * mean, 0.0)
    std = np.sqrt(var)
    std = np.where(std > 0.0, std, 1.0)
    z = (raw - mean) / std
    z[count_b < min_count] = np.nan
    z[~finite] = np.nan
    return z


def _diff_report(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    a = left.to_numpy(float)
    b = right.to_numpy(float)
    if a.shape != b.shape:
        raise SystemExit(f"PROBE_SHAPE {a.shape} {b.shape}")
    nan_mismatch = np.isnan(a) ^ np.isnan(b)
    delta = np.abs(np.where(np.isnan(a) | np.isnan(b), 0.0, a - b))
    delta[nan_mismatch] = np.inf
    if not delta.size:
        return {"max_abs": 0.0, "cells": 0, "column": None}
    worst = np.unravel_index(np.argmax(delta), delta.shape)
    material = delta > 1e-8
    return {
        "max_abs": float(delta[worst]) if np.isfinite(delta[worst]) else None,
        "nonfinite_mismatch": bool(nan_mismatch.any()),
        "cells": int((delta > 0).sum()),
        "cells_above_1e-8": int(material.sum()),
        "column": str(left.columns[worst[1]]),
        "row": int(worst[0]),
    }


def probe(grid: pd.DataFrame, closures) -> dict:
    cut = PROBE_CUT
    prefix = grid.iloc[:PROBE_BARS]
    if cut >= len(prefix) - 10:
        raise SystemExit("PROBE_CUT")
    print(f"PROBE start prefix {len(prefix)} cut {cut}", flush=True)
    full, _, _ = raw_features(prefix, closures)
    short, _, _ = raw_features(prefix.iloc[:cut], closures)
    full_z = pd.DataFrame(expanding_z(full.to_numpy(float), MIN_COUNT), index=full.index, columns=full.columns)
    short_z = pd.DataFrame(expanding_z(short.to_numpy(float), MIN_COUNT), index=short.index, columns=short.columns)
    prefix_gap = _diff_report(full_z.iloc[:cut], short_z)
    rng = np.random.default_rng(0)
    pert = prefix.copy()
    m = len(prefix) - cut
    base = float(prefix["CLOSE"].iloc[cut - 1])
    close = base * np.exp(rng.normal(0, 8e-4, m).cumsum())
    width = np.abs(rng.normal(0, 4e-4, m)) * base
    opened = np.r_[base, close[:-1]]
    pert.iloc[cut:, pert.columns.get_loc("CLOSE")] = close
    pert.iloc[cut:, pert.columns.get_loc("OPEN")] = opened
    pert.iloc[cut:, pert.columns.get_loc("HIGH")] = np.maximum(close, opened) + width
    pert.iloc[cut:, pert.columns.get_loc("LOW")] = np.minimum(close, opened) - width
    moved, _, _ = raw_features(pert, closures)
    moved_z = pd.DataFrame(expanding_z(moved.to_numpy(float), MIN_COUNT), index=moved.index, columns=moved.columns)
    future_gap = _diff_report(full_z.iloc[:cut], moved_z.iloc[:cut])
    body = abs(float(full_z["cdl.body_pips"].iloc[cut]) - float(moved_z["cdl.body_pips"].iloc[cut]))
    report = {"prefix": prefix_gap, "future": future_gap, "positive_control_body_abs": body, "cut": cut, "bars": len(prefix)}
    print("PROBE " + json.dumps(report), flush=True)
    # Prefix recomputation can move by one ulp. A future bar must move nothing at all.
    if (
        prefix_gap["cells_above_1e-8"]
        or prefix_gap["nonfinite_mismatch"]
        or future_gap["cells"]
        or future_gap["nonfinite_mismatch"]
        or not np.isfinite(body)
        or body == 0.0
    ):
        raise SystemExit("PROBE_FAILED")
    print("PROBE_OK", flush=True)
    return report


def _check_scaler(raw: np.ndarray, z: np.ndarray) -> None:
    column = 0
    for row in (MIN_COUNT + 10, len(raw) // 2, len(raw) - 1):
        past = raw[:row, column]
        past = past[np.isfinite(past)]
        if len(past) < MIN_COUNT or not np.isfinite(raw[row, column]):
            raise SystemExit(f"SCALER_CHECK_ROW {row}")
        mean = float(past.mean())
        std = float(past.std(ddof=0))
        std = std if std > 0.0 else 1.0
        expect = (float(raw[row, column]) - mean) / std
        if abs(expect - float(z[row, column])) > 1e-8:
            raise SystemExit(f"SCALER_CHECK {row} {expect} {z[row, column]}")
    print("SCALER_CHECK_OK", flush=True)


def persistence(y: np.ndarray, mask: np.ndarray, horizons: list[int]) -> list[dict]:
    rows = []
    for i, horizon in enumerate(horizons):
        ok = mask & np.isfinite(y[:, i])
        rows.append({"horizon": int(horizon), "rows": int(ok.sum()), "mae": float(np.mean(np.abs(y[ok, i]))) if ok.any() else None})
    return rows


def build_recipe(columns: list[str], groups: list[dict], store_sha: str, manifest_sha: str) -> dict:
    template = json.loads(RECIPE_TEMPLATE.read_text())
    names = list(columns)
    rec = {
        "schema": "expert_guided_consumer_recipe.v2",
        "scientific_release": False,
        "custody": template["custody"],
        "business": template["business"],
        "regularization": template["regularization"],
        "labels": template["labels"],
        "store": {
            "file": "causal_features_raw.parquet",
            "sha256": store_sha,
            "manifest_sha256": manifest_sha,
            "note": "raw producer outputs; the model consumes expanding-z windows, not this scaler",
        },
        "list": {"columns": names, "n": len(names), "file": "FEATURE_LIST.json"},
        "tensor": {
            "window": {
                "observations": 13,
                "stride_rows": 2,
                "includes_current_completed_bar": True,
                "slot_offsets_rows": OFFSETS.tolist(),
            },
            "columns": [
                {
                    "consumer_ops": [],
                    "store_column": name,
                    "tensor_name": name,
                    "then": "expanding z of rows strictly before t, already applied in the window tensor",
                    "upstream": {"applied_in_store": False, "decision": "NONE", "store_column": name, "suffix": None},
                }
                for name in names
            ],
        },
        "scaler": {
            "policy": "fixed_train",
            "ddof": 0,
            "constant_columns": [],
            "features": names,
            "mean": [0.0] * len(names),
            "std": [1.0] * len(names),
            "derivation": {
                "fit": "identity_placeholder",
                "applied_in_windows": "expanding mean and std of finite rows strictly before t",
                "min_count": MIN_COUNT,
                "ddof": 0,
                "zero_std_denominator": 1.0,
                "recipe_arrays_are_not_reapplied": True,
            },
        },
        "representation": {
            "resolved": {
                "names": [g["name"] for g in groups],
                "indices": [g["indices"] for g in groups],
                "units": [int(g["units"]) for g in groups],
                "kinds": [g["kind"] for g in groups],
            },
            "groups": {
                "name": "one_group_per_causal_family",
                "groups": {g["name"]: g["columns"] for g in groups},
                "kinds": {g["name"]: g["kind"] for g in groups},
            },
        },
        "split": {
            "arm": "causal_recompute_closed_before_2024",
            "periods": {
                "TRAIN": ["2016-01-01", "2023-07-01"],
                "VALIDATION": ["2023-07-01", "2024-01-01"],
            },
            "train": ["2016-01-01", "2023-07-01"],
            "validation": ["2023-07-01", "2024-01-01"],
            "selection_period": None,
            "status": "parameters_and_columns_closed_before_2024",
            "support_status": "causal_producer_recompute",
            "test_closed_from": "2024-01-01",
        },
        "scope": (
            "Full causal producer outputs. Kalman, GMM and the z-scaler use only rows "
            "strictly before the bar. Column set was not chosen with 2024. Not the 61-column subset."
        ),
        "provenance": {
            "derivation": "causal_recompute_past_only_parameters_full_producer_outputs",
            "not_a_subset_of": "recipe_b8fa85625fe025a7",
            "pretrained_weights": "not_loaded",
        },
    }
    sealed = seal_recipe(rec)
    if sealed["identity"] in FORBIDDEN_IDENTITY:
        raise SystemExit("SEALED_THE_EXPOSED_RECIPE")
    if sealed["list"]["n"] <= 61:
        raise SystemExit("COLUMN_COUNT")
    return sealed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid, closures = load_grid()
    probe_report = probe(grid, closures)
    print("FULL_FEATURES", flush=True)
    raw, groups, schedule = raw_features(grid, closures)
    raw_np = raw.to_numpy(float)
    _check_scaler(raw_np, expanding_z(raw_np, MIN_COUNT))
    z = expanding_z(raw_np, MIN_COUNT)
    labels = np.load(RAW / "retsu-causal-labels.npz")
    lab_ts = labels["ts"].astype(np.int64)
    lookup = {int(ts): i for i, ts in enumerate(grid.index.asi8)}
    pos = np.full(len(lab_ts), -1, np.int64)
    matched_as = {"bar_start": 0, "raw_timestamp": 0, "missing": 0}
    for i, ts in enumerate(lab_ts):
        found = lookup.get(int(ts) - HOUR)
        if found is not None:
            pos[i] = found
            matched_as["bar_start"] += 1
            continue
        found = lookup.get(int(ts))
        if found is not None:
            pos[i] = found
            matched_as["raw_timestamp"] += 1
            continue
        matched_as["missing"] += 1
    print("ALIGN " + json.dumps(matched_as), flush=True)
    for name in ("monitor", "holdout"):
        if np.any(pos[labels[name]] < 0):
            raise SystemExit(f"MISSING_{name.upper()}_TIMESTAMP")
    slots = pos[:, None] + OFFSETS[None, :]
    in_range = (pos >= 0)[:, None] & (slots >= 0) & (slots < len(z))
    gathered = z[np.clip(slots, 0, len(z) - 1)]
    finite_window = in_range.all(axis=1) & np.isfinite(gathered).all(axis=(1, 2))
    windows = np.where(finite_window[:, None, None], gathered, 0.0).astype(np.float32)
    origin = labels["seasonal_origin"].astype(np.float64)
    usable = np.isfinite(origin)
    out = {key: labels[key] for key in ("ts", "pool", "monitor", "holdout")}
    horizons = {"y_short": [1, 2, 3, 4, 5, 6], "y_long": [24, 48]}
    for key, ok_name in (("y_short", "ok_short"), ("y_long", "ok_long")):
        level = labels[key].astype(np.float64)
        increment = level - origin[:, None]
        admitted = np.isfinite(level) & usable[:, None]
        increment[~admitted] = np.nan
        out[key] = increment.astype(np.float32)
        out[ok_name] = labels[ok_name] & finite_window & np.isfinite(increment).all(axis=1)
    out["seasonal_origin"] = np.zeros(len(origin), np.float32)
    for split in ("pool", "monitor", "holdout"):
        for ok_name in ("ok_short", "ok_long"):
            old = int((labels[split] & labels[ok_name]).sum())
            new = int((out[split] & out[ok_name]).sum())
            if old and new / old < 0.9 and split in ("monitor", "holdout"):
                raise SystemExit(f"SURVIVAL {split} {ok_name} {new} {old}")
            print(f"ROWS {split} {ok_name} {new} of {old}", flush=True)
    raw_path = OUT / "causal_features_raw.parquet"
    store = raw.reset_index().rename(columns={"index": "timestamp"})
    store.to_parquet(raw_path, index=False)
    manifest = {
        "columns": list(raw.columns),
        "rows": int(len(raw)),
        "parquet_sha256": sha_file(raw_path),
        "groups": groups,
    }
    manifest_path = OUT / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    recipe = build_recipe(list(raw.columns), groups, manifest["parquet_sha256"], sha_file(manifest_path))
    (OUT / "RECIPE.json").write_text(json.dumps(recipe, indent=2) + "\n")
    (OUT / "learned_params.json").write_text(json.dumps(schedule, indent=2) + "\n")
    np.savez_compressed(OUT / "windows.npz", ts=out["ts"].astype(np.int64), windows=windows)
    np.savez_compressed(OUT / "dataset.npz", **out)
    naive = {}
    for split in ("pool", "monitor", "holdout"):
        naive[split] = {
            "short": persistence(out["y_short"], out[split] & out["ok_short"], horizons["y_short"]),
            "long": persistence(out["y_long"], out[split] & out["ok_long"], horizons["y_long"]),
        }
    protocol = {
        "target": "causal_stl_seasonal_future_minus_origin",
        "level_forecast": "seasonal_origin + predicted_increment",
        "naive": "zero increment, seasonal-level persistence, recomputed on these rows",
        "model_mae_2024": "not computed yet; one predict after champion restore",
        "exceptions_corrected": {
            "parameters": "Kalman, GMM and expanding z use rows strictly before the bar",
            "columns": "full causal producer outputs; 61-column 2024-exposed subset not used",
            "weights": "seed 7 build of the temporal architecture; model.pt not loaded",
        },
        "scaler": {
            "rows": "strictly before t",
            "ddof": 0,
            "min_count": MIN_COUNT,
            "zero_std_denominator": 1.0,
            "recipe_mean_std": "identity, not reapplied",
        },
        "learned": "calendar month, fit on timestamp < month start, forward filter or row posterior, keep that month",
        "excluded_producers": ["usd_strength", "eur_strength", "hmm", "multitaper", "fracdiff"],
        "window": {"observations": 13, "stride_rows": 2, "offsets": OFFSETS.tolist()},
        "architecture": "MODEL_CONFIG.json branch 8 core 16 blocks 2 fixed_fusion_offset causal_patch; fresh seed 7",
        "split": recipe["split"],
        "identity": recipe["identity"],
        "n_columns": recipe["list"]["n"],
        "columns": recipe["list"]["columns"],
        "groups": [{"name": g["name"], "kind": g["kind"], "n": len(g["columns"])} for g in groups],
        "probe": probe_report,
        "alignment": matched_as,
        "horizons_short": horizons["y_short"],
        "horizons_long": horizons["y_long"],
        "naive_mae": naive,
        "grid_start": str(grid.index[0]),
        "grid_end": str(grid.index[-1]),
        "grid_rows": int(len(grid)),
        "champion_protocol": {
            "loss": "mean absolute error of the standardized increment on the H2 2023 monitor",
            "min_delta": 1e-8,
            "optimizer": "AdamW",
            "lr": 3e-4,
            "weight_decay": 0.004,
            "reduce_lr": {"grace": 20, "patience": 20, "factor": 0.2, "floor": 4.8e-7},
            "early_stopping": {"patience": 80, "start_epoch": 100},
            "restore": "best weights and Adam moments; keep the already-reduced learning rate",
            "test": "2024 holdout, one predict after restore",
        },
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print("BUILD_OK", recipe["identity"], recipe["list"]["n"], flush=True)


if __name__ == "__main__":
    main()
