#!/usr/bin/env python3
"""Phase-2 steps 1 and 2. The mini work plan is the source of the constants.

Reads tools/retsu_phase1/MINI_WORK_PLAN.md. Applies decide() on the finished
111 causal columns, then appends causal STL seasonal endpoints. Does not
rebuild Kalman or GMM, does not refit the target STL, and does not write
into the finished causal directory.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

PREDICTOR = Path("/var/tmp/cpu398.Xn9J2L/predictor")
RUNTIME = Path("/var/tmp/sac399-series/runtime")
sys.path.insert(0, str(PREDICTOR))
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.causal_stl_inputs import endpoint_series, fit_history  # noqa: E402
from app.rl_subset import seal_recipe  # noqa: E402
from build_causal_inputs import (  # noqa: E402
    OFFSETS,
    _diff_report,
    expanding_z,
    persistence,
    sha_file,
)
from tools.layer1_pool_h1 import decide, diagnostics  # noqa: E402

HOUR = 3_600_000_000_000
RECIPE_TEMPLATE = Path("/var/tmp/sac399-series/model/61/bundle/RECIPE.json")


def load_plan(path: Path) -> dict:
    text = path.read_text()
    marker = "```phase2-plan\n"
    start = text.index(marker) + len(marker)
    end = text.index("\n```", start)
    plan = json.loads(text[start:end])
    if plan.get("schema") != "retsu_phase2_mini_plan.v1":
        raise SystemExit("PLAN_SCHEMA")
    if plan.get("acf_tie") != "smaller_lag":
        raise SystemExit("PLAN_TIE")
    if int(plan["min_count"]) != 2000:
        raise SystemExit("PLAN_MIN_COUNT")
    if list(OFFSETS) != list(range(-24, 1, 2)):
        raise SystemExit("PLAN_OFFSETS")
    return plan


def blocked(name: str, plan: dict) -> bool:
    if name in plan["never_transform_exact"]:
        return True
    return any(name.startswith(prefix) for prefix in plan["never_transform_prefixes"])


def apply_decision(values: np.ndarray, decision: str, scale: float | None) -> np.ndarray:
    if decision == "NONE":
        return np.array(values, dtype=float, copy=True)
    if decision == "DIFF":
        return np.r_[np.nan, np.diff(np.asarray(values, dtype=float))]
    with np.errstate(divide="ignore", invalid="ignore"):
        if decision == "ASINH":
            out = np.arcsinh(values)
        elif decision == "LOG":
            out = np.log(values)
        elif decision == "ASINHS":
            out = np.arcsinh(np.asarray(values, dtype=float) / float(scale))
        else:
            raise SystemExit(f"DECISION {decision}")
    return np.where(np.isfinite(out), out, np.nan)


def acf_peaks(values: np.ndarray, train_end: int, plan: dict) -> list[tuple[int, float]]:
    from statsmodels.tsa.stattools import acf

    segment = np.asarray(values[:train_end], dtype=float)
    finite = np.isfinite(segment)
    if not finite.any():
        return []
    suffix = segment[int(np.argmax(finite)) :]
    if suffix.size < int(plan["acf_min_rows"]) or not np.isfinite(suffix).all():
        return []
    corr = acf(suffix, nlags=50, fft=True)
    threshold = float(plan["acf_z"]) / np.sqrt(suffix.size)
    hits = []
    for lag in plan["acf_lags_bars"]:
        lag = int(lag)
        if not np.isfinite(corr[lag]):
            continue
        if corr[lag] > threshold and corr[lag] > corr[lag - 1] and corr[lag] >= corr[lag + 1]:
            hits.append((lag, float(corr[lag])))
    return hits


def choose_period(hits: list[tuple[int, float]], plan: dict) -> int:
    if not hits:
        raise SystemExit("PERIOD")
    best = max(hits, key=lambda item: (item[1], -item[0]))
    return int(best[0])


def choose_stl(values: np.ndarray, train_end: int, period: int, plan: dict) -> dict | None:
    histories = [max(int(plan["stl_history_floor"]), int(multiple) * period) for multiple in plan["stl_history_multiples"]]
    configs = [
        {"period": int(period), "seasonal": int(seasonal), "robust": bool(robust), "history": int(history)}
        for history in histories
        for seasonal in plan["stl_seasonal"]
        for robust in plan["stl_robust"]
    ]
    horizon = int(plan["stl_forecast_horizon_bars"])
    series = np.asarray(values[:train_end], dtype=float)
    context = max(config["history"] for config in configs)
    finite = np.isfinite(series)
    invalid = np.r_[0, np.cumsum(~finite)]
    eligible = [
        t
        for t in range(context - 1, len(series) - horizon)
        if invalid[t + horizon + 1] == invalid[t - context + 1]
    ]
    if len(eligible) < int(plan["stl_origins"]):
        return None
    origins = np.array(eligible)[np.linspace(0, len(eligible) - 1, int(plan["stl_origins"]), dtype=int)]
    best = None
    for config in configs:
        losses = []
        for origin in origins:
            history = series[origin - config["history"] + 1 : origin + 1]
            fit = fit_history(history, config)
            slope = np.polyfit(np.arange(period), fit.trend[-period:], 1)[0]
            trend = fit.trend[-1] + slope * np.arange(1, horizon + 1)
            forecast = trend + np.resize(fit.seasonal[-period:], horizon)
            target = series[origin + 1 : origin + horizon + 1]
            common = series[origin - context + 1 : origin + 1]
            scale = max(
                float(np.abs(np.diff(common)).mean()),
                np.finfo(float).eps * max(1.0, float(np.abs(common).max())),
            )
            losses.append(float(np.abs(forecast - target).mean() / scale))
        rank = (float(np.mean(losses)), config["history"], config["seasonal"], int(config["robust"]))
        if best is None or rank < best[0]:
            best = (rank, dict(config), rank[0])
    assert best is not None
    return {"config": best[1], "score": best[2], "origins": int(len(origins))}


def _endpoint_job(payload: tuple[str, np.ndarray, dict]) -> tuple[str, np.ndarray]:
    os.environ["OMP_NUM_THREADS"] = "1"
    name, values, config = payload
    sys.path.insert(0, "/var/tmp/sac399-series/runtime")
    from app.causal_stl_inputs import endpoint_series as fit_endpoint

    return name, fit_endpoint(values, config)


def endpoints(jobs: list[tuple[str, np.ndarray, dict]], workers: int) -> dict[str, np.ndarray]:
    if not jobs:
        return {}
    started = time.time()
    print(f"STL_ENDPOINTS {len(jobs)} workers {workers}", flush=True)
    out = {}
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for done, (name, values) in enumerate(pool.map(_endpoint_job, jobs), start=1):
            out[name] = values
            print(f"STL_DONE {done}/{len(jobs)} {name} {time.time() - started:.0f}s", flush=True)
    return out


def cached_endpoints(out_dir: Path, jobs: list[tuple[str, np.ndarray, dict]], workers: int) -> dict[str, np.ndarray]:
    cache = out_dir / "endpoints"
    cache.mkdir(parents=True, exist_ok=True)
    pending = []
    found: dict[str, np.ndarray] = {}
    for name, values, config in jobs:
        digest = hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()[:16]
        path = cache / f"{name.replace('/', '_')}.{digest}.npy"
        meta = cache / f"{name.replace('/', '_')}.{digest}.json"
        if path.exists() and meta.exists() and json.loads(meta.read_text()) == config:
            found[name] = np.load(path)
            print(f"STL_CACHE {name}", flush=True)
            continue
        pending.append((name, values, config, path, meta, digest))
    fresh = endpoints([(name, values, config) for name, values, config, *_ in pending], workers)
    for name, _values, config, path, meta, _digest in pending:
        np.save(path, fresh[name])
        meta.write_text(json.dumps(config))
        found[name] = fresh[name]
    return found


def assemble(raw: np.ndarray, names: list[str], specs: list[dict], close: np.ndarray, seasonal: dict[str, np.ndarray], jobs: list[dict]) -> tuple[list[str], np.ndarray]:
    columns = [apply_decision(raw[:, index], spec["decision"], spec.get("scale")) for index, spec in enumerate(specs)]
    out_names = list(names)
    for job in jobs:
        column = seasonal[job["name"]]
        if column.shape != (len(close),):
            raise SystemExit(f"STL_SHAPE {job['name']}")
        out_names.append(job["name"])
        columns.append(column)
    if any(name.startswith("Prediction_") for name in out_names):
        raise SystemExit("PREDICTION_COLUMN")
    return out_names, np.column_stack(columns)


def seasonal_jobs(names: list[str], transformed: np.ndarray, close: np.ndarray, train_end: int, plan: dict) -> list[dict]:
    jobs = []
    hits = acf_peaks(close, train_end, plan)
    period = choose_period(hits, plan) if hits else int(plan["close_period_default_bars"])
    chosen = choose_stl(close, train_end, period, plan)
    if chosen is None:
        raise SystemExit("CLOSE_STL_NO_ORIGINS")
    jobs.append({"name": "stl.close", "source": "CLOSE", "period": period, "peaks": hits, **chosen})
    print(f"STL_PARAM stl.close period {period} {json.dumps(chosen['config'])} score {chosen['score']:.6f}", flush=True)
    for index, name in enumerate(names):
        if blocked(name, plan):
            continue
        column_hits = acf_peaks(transformed[:, index], train_end, plan)
        if not column_hits:
            continue
        column_period = choose_period(column_hits, plan)
        column_choice = choose_stl(transformed[:, index], train_end, column_period, plan)
        if column_choice is None:
            print(f"STL_SKIP {name} no eligible train origins", flush=True)
            continue
        job_name = "stl." + name
        jobs.append({"name": job_name, "source": name, "period": column_period, "peaks": column_hits, **column_choice})
        print(
            f"STL_PARAM {job_name} period {column_period} {json.dumps(column_choice['config'])} score {column_choice['score']:.6f}",
            flush=True,
        )
    return jobs


def source_matrix(names: list[str], transformed: np.ndarray, close: np.ndarray, jobs: list[dict]) -> list[tuple[str, np.ndarray, dict]]:
    by_name = {name: transformed[:, index] for index, name in enumerate(names)}
    by_name["CLOSE"] = close
    return [(job["name"], np.ascontiguousarray(by_name[job["source"]], dtype=np.float64), job["config"]) for job in jobs]


def compare(left: np.ndarray, right: np.ndarray, columns: list[str]) -> dict:
    frame_l = pd.DataFrame(left, columns=columns)
    frame_r = pd.DataFrame(right, columns=columns)
    return _diff_report(frame_l, frame_r)


def run_probe(raw: np.ndarray, names: list[str], specs: list[dict], close: np.ndarray, jobs: list[dict], plan: dict) -> dict:
    bars = int(plan["probe_bars"])
    cut = int(plan["probe_cut"])
    if cut >= bars - 10 or bars > len(raw):
        raise SystemExit("PROBE_CUT")
    max_history = max(job["config"]["history"] for job in jobs)
    if max_history >= cut:
        raise SystemExit("PROBE_HISTORY")
    print(f"PROBE start prefix {bars} cut {cut}", flush=True)
    workers = min(16, max(1, (os.cpu_count() or 4) - 2))
    full_t = np.column_stack([apply_decision(raw[:bars, index], spec["decision"], spec.get("scale")) for index, spec in enumerate(specs)])
    short_raw_t = np.column_stack([apply_decision(raw[:cut, index], spec["decision"], spec.get("scale")) for index, spec in enumerate(specs)])
    pert_raw = raw[:bars].copy()
    pert_close = close[:bars].copy()
    rng = np.random.default_rng(0)
    pert_raw[cut:] = pert_raw[cut:] + rng.normal(0.0, 1.0, size=pert_raw[cut:].shape)
    pert_close[cut:] = pert_close[cut:] * np.exp(rng.normal(0.0, 8e-4, size=pert_close[cut:].shape))
    pert_t = np.column_stack([apply_decision(pert_raw[:, index], spec["decision"], spec.get("scale")) for index, spec in enumerate(specs)])
    full_s = endpoints(source_matrix(names, full_t, close[:bars], jobs), workers)
    short_s = endpoints(source_matrix(names, short_raw_t, close[:cut], jobs), workers)
    pert_s = endpoints(source_matrix(names, pert_t, pert_close, jobs), workers)

    def pack(matrix: np.ndarray, series: np.ndarray, seasonal_map: dict[str, np.ndarray]) -> np.ndarray:
        extra = [seasonal_map[job["name"]] for job in jobs]
        return np.column_stack([matrix, *extra]) if extra else matrix

    packed_names = names + [job["name"] for job in jobs]
    full = pack(full_t, close[:bars], full_s)
    short = pack(short_raw_t, close[:cut], short_s)
    moved = pack(pert_t, pert_close, pert_s)
    count = int(plan["min_count"])
    full_z = expanding_z(full, count)
    short_z = expanding_z(short, count)
    moved_z = expanding_z(moved, count)
    prefix_gap = compare(full_z[:cut], short_z, packed_names)
    future_gap = compare(full_z[:cut], moved_z[:cut], packed_names)
    body = names.index("cdl.body_pips")
    control = abs(float(full_z[cut, body]) - float(moved_z[cut, body]))
    # The future probe above does not change the cut row of body by itself if noise starts at cut
    # and body is pointwise: it does change row `cut`. Require that move.
    report = {
        "prefix": prefix_gap,
        "future": future_gap,
        "positive_control_body_abs": control,
        "cut": cut,
        "bars": bars,
    }
    print("PROBE " + json.dumps(report), flush=True)
    if (
        prefix_gap["cells_above_1e-8"]
        or prefix_gap["nonfinite_mismatch"]
        or future_gap["cells"]
        or future_gap["nonfinite_mismatch"]
        or not np.isfinite(control)
        or control == 0.0
    ):
        raise SystemExit("PROBE_FAILED")
    print("PROBE_OK", flush=True)
    return report


def load_close(timestamps: pd.DatetimeIndex, plan: dict) -> np.ndarray:
    hist = pd.read_parquet(plan["hist_ohlc"])
    hist.index = pd.DatetimeIndex(hist.index)
    if hist.index.tz is None:
        hist.index = hist.index.tz_localize("UTC")
    else:
        hist.index = hist.index.tz_convert("UTC")
    bars = pd.read_parquet(plan["bars_parquet"])
    live = pd.DataFrame(
        {"CLOSE": bars["BidClose"].to_numpy(float)},
        index=pd.DatetimeIndex(pd.to_datetime(bars["timestamp"], utc=True)),
    )
    grid = pd.concat([hist[["CLOSE"]], live], axis=0)
    grid = grid[grid.index < "2025-01-01"].sort_index()
    if not grid.index.equals(timestamps):
        raise SystemExit("GRID_MISMATCH")
    close = grid["CLOSE"].to_numpy(float)
    if not np.isfinite(close).all():
        raise SystemExit("NONFINITE_CLOSE")
    return close


def extend_groups(groups: list[dict], jobs: list[dict]) -> list[dict]:
    out = [dict(group) for group in groups]
    cursor = sum(len(group["columns"]) for group in out)
    for job in jobs:
        out.append(
            {
                "name": job["name"],
                "kind": "conv",
                "columns": [job["name"]],
                "indices": [cursor],
                "units": 2,
            }
        )
        cursor += 1
    return out


def build_recipe(columns: list[str], groups: list[dict], specs_by_name: dict, store_sha: str, manifest_sha: str, plan: dict) -> dict:
    template = json.loads(RECIPE_TEMPLATE.read_text())
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
            "note": "post-decide values plus causal STL endpoints; the model consumes expanding-z windows",
        },
        "list": {"columns": list(columns), "n": len(columns), "file": "FEATURE_LIST.json"},
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
                    "upstream": {
                        "applied_in_store": bool(specs_by_name.get(name, {}).get("decision") not in (None, "NONE")),
                        "decision": specs_by_name.get(name, {}).get("decision", "STL_ENDPOINT"),
                        "store_column": name,
                        "suffix": None,
                    },
                }
                for name in columns
            ],
        },
        "scaler": {
            "policy": "fixed_train",
            "ddof": 0,
            "constant_columns": [],
            "features": list(columns),
            "mean": [0.0] * len(columns),
            "std": [1.0] * len(columns),
            "derivation": {
                "fit": "identity_placeholder",
                "applied_in_windows": "expanding mean and std of finite rows strictly before t",
                "min_count": int(plan["min_count"]),
                "ddof": 0,
                "zero_std_denominator": 1.0,
                "recipe_arrays_are_not_reapplied": True,
            },
        },
        "representation": {
            "resolved": {
                "names": [group["name"] for group in groups],
                "indices": [group["indices"] for group in groups],
                "units": [int(group["units"]) for group in groups],
                "kinds": [group["kind"] for group in groups],
            },
            "groups": {
                "name": "causal_family_plus_one_block_per_stl_endpoint",
                "groups": {group["name"]: group["columns"] for group in groups},
                "kinds": {group["name"]: group["kind"] for group in groups},
            },
        },
        "split": {
            "arm": "phase2_transforms_not_yet_selected",
            "periods": {"TRAIN": ["2016-01-01", "2023-07-01"], "VALIDATION": ["2023-07-01", "2024-01-01"]},
            "train": ["2016-01-01", "2023-07-01"],
            "validation": ["2023-07-01", "2024-01-01"],
            "selection_period": None,
            "status": "phase2_transforms_closed_before_2023-07_selection_open",
            "support_status": "phase2_decide_and_causal_stl_endpoint",
            "test_closed_from": "2024-01-01",
        },
        "scope": (
            "Phase-2 steps 1 and 2 only. Stationarity decisions and STL parameters use rows "
            "before 2023-07-01. Selection is not done. Not the 61-column subset. Target STL was not refit."
        ),
        "provenance": {
            "derivation": "phase2_decide_then_causal_stl_endpoint_beside_feature",
            "parent_recipe": plan["parent_recipe"],
            "not_a_subset_of": "recipe_b8fa85625fe025a7",
            "pretrained_weights": "not_loaded",
            "plan": "tools/retsu_phase1/MINI_WORK_PLAN.md",
        },
    }
    sealed = seal_recipe(rec)
    if sealed["identity"] in set(plan["forbidden_recipe"]) or sealed["identity"] == plan["parent_recipe"]:
        raise SystemExit("SEALED_A_CLOSED_RECIPE")
    if sealed["list"]["n"] <= 61:
        raise SystemExit("COLUMN_COUNT")
    return sealed


def main() -> None:
    plan_path = Path(sys.argv[sys.argv.index("--plan") + 1] if "--plan" in sys.argv else Path(__file__).with_name("MINI_WORK_PLAN.md"))
    plan = load_plan(plan_path)
    finished = Path(plan["finished_dir"]).resolve()
    out = Path(plan["out_dir"]).resolve()
    if out == finished or str(out).startswith(str(finished) + os.sep):
        raise SystemExit("REFUSING_FINISHED_DIR")
    if (out / "PHASE2_BUILD_OK").exists():
        raise SystemExit("ALREADY_BUILT")
    parent = json.loads((finished / "protocol.json").read_text())
    if parent.get("identity") != plan["parent_recipe"]:
        raise SystemExit("PARENT_RECIPE")
    out.mkdir(parents=True, exist_ok=True)
    raw_frame = pd.read_parquet(finished / "causal_features_raw.parquet")
    timestamps = pd.DatetimeIndex(pd.to_datetime(raw_frame["timestamp"], utc=True))
    manifest = json.loads((finished / "MANIFEST.json").read_text())
    names = [column for column in raw_frame.columns if column != "timestamp"]
    if names != manifest["columns"]:
        raise SystemExit("COLUMN_ORDER")
    grouped = [column for group in manifest["groups"] for column in group["columns"]]
    if grouped != names:
        raise SystemExit("GROUP_ORDER")
    if len(names) != 111:
        raise SystemExit(f"EXPECTED_111 {len(names)}")
    available = timestamps + pd.Timedelta(plan["bar_to_available"])
    cutoff = pd.Timestamp(plan["decision_available_at_before"])
    train_end = int(np.searchsorted(available.asi8, cutoff.value, side="left"))
    if not (0 < train_end < len(timestamps)):
        raise SystemExit("TRAIN_END")
    if available[train_end - 1] >= cutoff or available[train_end] < cutoff:
        raise SystemExit("TRAIN_END_EDGE")
    print(f"PLAN_OK train_end {train_end} {timestamps[0]} -> {timestamps[-1]}", flush=True)
    close = load_close(timestamps, plan)
    raw = raw_frame[names].to_numpy(float)
    train = np.zeros(len(raw), dtype=bool)
    train[:train_end] = True
    specs = []
    for index, name in enumerate(names):
        column = raw[:, index]
        if blocked(name, plan):
            specs.append({"name": name, "decision": "NONE", "reason": "plan: calendar, event clock or regime probability"})
            print(f"DECIDE {name} NONE blocked", flush=True)
            continue
        measured = diagnostics(column[train])
        decision, reason, _transformed = decide(name, column, measured, train=train)
        spec = {"name": name, "decision": decision, "reason": reason, "diagnostics": {key: measured.get(key) for key in ("n", "adf_p", "kpss_p", "skew", "min", "max")}}
        if "transform_scale" in measured:
            spec["scale"] = measured["transform_scale"]
        specs.append(spec)
        print(f"DECIDE {name} {decision}", flush=True)
    (out / "decisions.json").write_text(json.dumps(specs, indent=2) + "\n")
    transformed = np.column_stack([apply_decision(raw[:, index], spec["decision"], spec.get("scale")) for index, spec in enumerate(specs)])
    jobs = seasonal_jobs(names, transformed, close, train_end, plan)
    (out / "stl_jobs.json").write_text(json.dumps(jobs, indent=2, default=str) + "\n")
    probe = run_probe(raw, names, specs, close, jobs, plan)
    workers = min(16, max(1, (os.cpu_count() or 4) - 2))
    seasonal = cached_endpoints(out, source_matrix(names, transformed, close, jobs), workers)
    out_names, matrix = assemble(raw, names, specs, close, seasonal, jobs)
    if out_names != names + [job["name"] for job in jobs]:
        raise SystemExit("ASSEMBLE_ORDER")
    z = expanding_z(matrix, int(plan["min_count"]))
    labels = np.load(finished / "dataset.npz")
    if not np.array_equal(labels["seasonal_origin"], np.zeros(len(labels["seasonal_origin"]), np.float32)):
        raise SystemExit("ORIGIN_NOT_ZERO")
    lab_ts = labels["ts"].astype(np.int64)
    lookup = {int(ts): i for i, ts in enumerate(timestamps.asi8)}
    pos = np.array([lookup.get(int(ts) - HOUR, -1) for ts in lab_ts], dtype=np.int64)
    if np.any(pos < 0):
        raise SystemExit("ALIGN_MISSING")
    slots = pos[:, None] + OFFSETS[None, :]
    in_range = (slots >= 0) & (slots < len(z))
    gathered = z[np.clip(slots, 0, len(z) - 1)]
    finite_window = in_range.all(axis=1) & np.isfinite(gathered).all(axis=(1, 2))
    windows = np.where(finite_window[:, None, None], gathered, 0.0).astype(np.float32)
    dataset = {key: labels[key] for key in ("ts", "pool", "monitor", "holdout", "y_short", "y_long")}
    dataset["seasonal_origin"] = np.zeros(len(lab_ts), np.float32)
    for ok_name in ("ok_short", "ok_long"):
        dataset[ok_name] = np.asarray(labels[ok_name]) & finite_window
        for split in ("pool", "monitor", "holdout"):
            old = int((labels[split] & labels[ok_name]).sum())
            new = int((dataset[split] & dataset[ok_name]).sum())
            print(f"ROWS {split} {ok_name} {new} of {old}", flush=True)
            if split in ("monitor", "holdout") and old and new / old < 0.9:
                raise SystemExit(f"SURVIVAL {split} {ok_name} {new} {old}")
    store = pd.DataFrame(matrix, columns=out_names)
    store.insert(0, "timestamp", timestamps)
    raw_path = out / "causal_features_raw.parquet"
    store.to_parquet(raw_path, index=False)
    groups = extend_groups(manifest["groups"], jobs)
    manifest_out = {"columns": out_names, "rows": int(len(store)), "parquet_sha256": sha_file(raw_path), "groups": groups, "parent_recipe": plan["parent_recipe"]}
    manifest_path = out / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest_out, indent=2) + "\n")
    specs_by_name = {spec["name"]: spec for spec in specs}
    for job in jobs:
        specs_by_name[job["name"]] = {"decision": "STL_ENDPOINT", "config": job["config"]}
    recipe = build_recipe(out_names, groups, specs_by_name, manifest_out["parquet_sha256"], sha_file(manifest_path), plan)
    (out / "RECIPE.json").write_text(json.dumps(recipe, indent=2) + "\n")
    np.savez_compressed(out / "windows.npz", ts=dataset["ts"].astype(np.int64), windows=windows)
    np.savez_compressed(out / "dataset.npz", **dataset)
    horizons = {"y_short": plan["short_horizons"], "y_long": plan["long_horizons"]}
    naive = {}
    for split in ("pool", "monitor", "holdout"):
        naive[split] = {
            "short": persistence(dataset["y_short"], dataset[split] & dataset["ok_short"], horizons["y_short"]),
            "long": persistence(dataset["y_long"], dataset[split] & dataset["ok_long"], horizons["y_long"]),
        }
    counts = {spec["decision"]: sum(item["decision"] == spec["decision"] for item in specs) for spec in specs}
    protocol = {
        "plan": "tools/retsu_phase1/MINI_WORK_PLAN.md",
        "plan_schema": plan["schema"],
        "parent_recipe": plan["parent_recipe"],
        "identity": recipe["identity"],
        "status": recipe["split"]["status"],
        "selection_period": None,
        "model_mae_2024": "not computed yet; one predict after the final champion restore",
        "target": "saved causal STL seasonal increment; not refit",
        "naive": "zero increment, seasonal-level persistence, recomputed on these rows",
        "naive_mae": naive,
        "decisions": counts,
        "stl_columns": [job["name"] for job in jobs],
        "n_columns": len(out_names),
        "train_end_row": train_end,
        "probe": probe,
        "finished_dir_untouched": str(finished),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "PHASE2_BUILD_OK").write_text(recipe["identity"] + "\n")
    print(f"PHASE2_BUILD_OK {recipe['identity']} columns {len(out_names)} stl {len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
