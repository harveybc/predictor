#!/usr/bin/env python3
"""Bounded block computation of feature-feature dependence for phase 2.

One worker process loads the TRAIN feature matrix named by the manifest exactly once
(digest-checked, the only file it opens), restricts itself to a declared number of
numeric threads, and computes every expected disposition for the pairs of one shard:

* identity/equivalence gate on shared rows (byte equality, numeric equality within a
  declared tolerance, exact affine relation, near-perfect monotone relation, mask
  agreement, and source/underlying/transform/unit/availability metadata agreement);
* per fold (TRAIN and every chronological inner-TRAIN fold), on aligned finite rows:
  Pearson, Spearman, Kendall tau-b, mutual information (quantile-bin plug-in estimator
  with declared bins and seed), distance correlation (declared even-stride subsample),
  and cross-correlation at elapsed-hour lags in both lead directions;
* per (metric, lag) fold stability: mean, sd, sign consistency, min, max, valid folds;
* an explicit state for every expected cell: MEASURED, INSUFFICIENT_SUPPORT,
  NOT_APPLICABLE or FAILED, always with effective support and a reason.

Nothing here opens targets, validation or test data; fold results depend only on rows
inside the fold's TRAIN range, so mutating a later row cannot change them.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from tools.fs_phase23_manifest import canonical_bytes, sha256_file

METHOD = "pairwise_v1"
SCHEMA_TERMINAL = "fs_phase23.pairwise_terminal.v1"

DEFAULT_PARAMS: dict[str, Any] = {
    "method": METHOD,
    "lags_hours": [0, 1, 2, 6, 24, 48, 168],
    "min_support": 100,
    "mi_estimator": "quantile_bins_plugin",
    "mi_bins": 8,
    "mi_seed": 0,
    "mi_edges": "quantile_fit_on_aligned_rows_within_fold",
    "dcor_max_rows": 2000,
    "dcor_subsample": "even_stride",
    "dcor_seed": 0,
    "numeric_rtol": 1e-9,
    "numeric_atol": 1e-12,
    "affine_rel_tol": 1e-8,
    "monotone_abs_spearman": 0.999,
    "kendall_variant": "b",
    "redundancy_abs_spearman": 0.9,
    "redundancy_linkage": "average",
}

BASE_METRICS = ("pearson", "spearman", "kendall_tau", "mutual_information", "distance_correlation")
GATE_ALIAS_STATES = ("BYTE_IDENTICAL", "NUMERIC_EQUAL", "EXACT_AFFINE")


class WorkerError(RuntimeError):
    pass


def code_sha256() -> str:
    return sha256_file(Path(__file__))


def params_sha256(params: dict) -> str:
    return hashlib.sha256(canonical_bytes(params)).hexdigest()


def metric_slots(params: dict) -> list[tuple[str, int]]:
    """The declared (metric, lag) cells every pair must carry per fold, in canonical order."""
    slots = [(m, 0) for m in BASE_METRICS]
    lags = sorted({int(l) for l in params["lags_hours"]})
    if 0 in lags:
        slots.append(("xcorr", 0))
    for lag in lags:
        if lag > 0:
            slots.append(("xcorr_x_leads", lag))
            slots.append(("xcorr_y_leads", lag))
    return slots


def row_key(run_id: str, *parts: Any) -> str:
    return hashlib.sha256("|".join([run_id, *[str(p) for p in parts]]).encode()).hexdigest()


def _finite_float(value: float) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


# ----------------------------------------------------------------------------- data

def timestamps_to_seconds(series) -> np.ndarray:
    """Epoch seconds (int64) from a tz-aware or naive timestamp column; identical across pandas 2.x and 3.x."""
    import pandas as pd
    ts = pd.to_datetime(series, utc=True)
    naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return (naive.to_numpy().astype("datetime64[ns]").astype(np.int64) // 1_000_000_000).astype(np.int64)


class PopulationData:
    """The TRAIN feature matrix in canonical (manifest) column order."""

    def __init__(self, manifest: dict, data_root: Path, column_block: int = 32):
        import pandas as pd  # local import keeps numeric-thread env settable before numpy loads

        data_root = Path(data_root)
        self.manifest = manifest
        name = manifest["data"]["features_file"]
        if any(tok in name.lower() for tok in ("validation", "test", "holdout")):
            raise WorkerError(f"refusing to open non-TRAIN file {name}")
        path = data_root / name
        observed = sha256_file(path)
        if observed != manifest["data"]["features_sha256"]:
            raise WorkerError(f"features digest {observed[:16]} != manifest {manifest['data']['features_sha256'][:16]}")
        self.features = [f["feature_id"] for f in manifest["features"]]
        ts_col = manifest["data"]["timestamp_column"]
        n = int(manifest["data"]["train_rows"])
        ts_frame = pd.read_parquet(path, columns=[ts_col])
        if len(ts_frame) != n:
            raise WorkerError(f"TRAIN file has {len(ts_frame)} rows, manifest declares {n}")
        self.ts_seconds = timestamps_to_seconds(ts_frame[ts_col])
        if np.any(np.diff(self.ts_seconds) <= 0):
            raise WorkerError("decision timestamps must be strictly increasing")
        del ts_frame
        # one shared read per column block; the matrix is held once as p x n float64
        self.X = np.empty((len(self.features), n), dtype=np.float64)
        for start in range(0, len(self.features), column_block):
            block = self.features[start:start + column_block]
            frame = pd.read_parquet(path, columns=block)
            for k, f in enumerate(block):
                self.X[start + k] = frame[f].to_numpy(dtype=np.float64)
            del frame
        self.n = n
        self.index = {f: i for i, f in enumerate(self.features)}
        self.meta = {f["feature_id"]: f for f in manifest["features"]}

    def folds(self) -> list[tuple[str, int]]:
        out = [("TRAIN", self.n)]
        for fold in self.manifest["folds"]:
            out.append((fold["fold_id"], int(fold["train_rows"][1])))
        return out


# ----------------------------------------------------------------------------- estimators

def _rank(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(values, method="average")


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    xm = x - x.mean()
    ym = y - y.mean()
    den = math.sqrt(float(xm @ xm) * float(ym @ ym))
    if den == 0.0:
        return None
    return float(xm @ ym) / den


def mutual_information_bins(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Plug-in MI (nats) with quantile edges fit on the given aligned rows."""
    def codes(v: np.ndarray) -> np.ndarray:
        edges = np.quantile(v, np.linspace(0.0, 1.0, bins + 1))
        inner = np.unique(edges[1:-1])
        return np.searchsorted(inner, v, side="right")
    cx, cy = codes(x), codes(y)
    joint = np.bincount(cx * bins + cy, minlength=bins * bins).reshape(bins, bins).astype(np.float64)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log(joint[nz] / (px @ py)[nz])))


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    n = len(x)
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a -= a.mean(axis=0, keepdims=True)
    a -= a.mean(axis=1, keepdims=True)
    a += a.mean()
    b -= b.mean(axis=0, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    b += b.mean()
    dcov2 = float((a * b).mean())
    dvx = float((a * a).mean())
    dvy = float((b * b).mean())
    den = math.sqrt(dvx * dvy)
    if den <= 0.0:
        return None
    return math.sqrt(max(dcov2, 0.0) / den)


def even_stride_subsample(n: int, max_rows: int) -> np.ndarray:
    if n <= max_rows:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_rows).round().astype(np.int64))


# ----------------------------------------------------------------------------- pair computation

def _cell(metric: str, lag: int, value: float | None, support: int, state: str, reason: str | None,
          estimator: str, params: dict) -> dict:
    return {"metric": metric, "lag_hours": int(lag), "value": _finite_float(value) if state == "MEASURED" else None,
            "support_n": int(support), "state": state, "reason": reason, "estimator": estimator, "params": params}


def pair_fold_cells(x: np.ndarray, y: np.ndarray, ts: np.ndarray, params: dict,
                    rank_x: np.ndarray | None = None, rank_y: np.ndarray | None = None) -> list[dict]:
    """All (metric, lag) cells for one pair inside one fold.  ``x``/``y`` are the fold rows (may hold NaN).

    ``rank_x``/``rank_y`` are optional precomputed average ranks over the fold rows; they are used only
    when the pair shares every row (both columns complete), where they equal the per-pair ranks exactly.
    """
    min_support = int(params["min_support"])
    bins = int(params["mi_bins"])
    mi_params = {"bins": bins, "seed": int(params["mi_seed"]), "edges": params["mi_edges"]}
    dcor_params = {"max_rows": int(params["dcor_max_rows"]), "subsample": params["dcor_subsample"], "seed": int(params["dcor_seed"])}
    kendall_params = {"variant": params["kendall_variant"]}
    shared = np.isfinite(x) & np.isfinite(y)
    n = int(shared.sum())
    cells: list[dict] = []
    base_est = {"pearson": ("pearson_product_moment", {}), "spearman": ("spearman_average_ranks", {}),
                "kendall_tau": ("scipy_kendalltau", kendall_params),
                "mutual_information": (params["mi_estimator"], mi_params),
                "distance_correlation": ("szekely_dcor_subsample", dcor_params)}
    if n < min_support:
        reason = f"aligned finite rows {n} < min_support {min_support}"
        for m in BASE_METRICS:
            est, p = base_est[m]
            cells.append(_cell(m, 0, None, n, "INSUFFICIENT_SUPPORT", reason, est, p))
    else:
        xs, ys = x[shared], y[shared]
        if xs.std() == 0.0 or ys.std() == 0.0:
            reason = "zero variance on aligned rows"
            for m in BASE_METRICS:
                est, p = base_est[m]
                cells.append(_cell(m, 0, None, n, "NOT_APPLICABLE", reason, est, p))
        else:
            try:
                cells.append(_cell("pearson", 0, pearson(xs, ys), n, "MEASURED", None, *base_est["pearson"]))
                if rank_x is not None and rank_y is not None and n == len(x):
                    rho = pearson(rank_x.astype(np.float64), rank_y.astype(np.float64))
                else:
                    rho = pearson(_rank(xs), _rank(ys))
                cells.append(_cell("spearman", 0, rho, n, "MEASURED", None, *base_est["spearman"]))
                from scipy.stats import kendalltau
                tau = float(kendalltau(xs, ys, variant=params["kendall_variant"]).statistic)
                cells.append(_cell("kendall_tau", 0, tau, n, "MEASURED", None, *base_est["kendall_tau"]))
                cells.append(_cell("mutual_information", 0, mutual_information_bins(xs, ys, bins), n, "MEASURED", None,
                                   *base_est["mutual_information"]))
                idx = even_stride_subsample(n, int(params["dcor_max_rows"]))
                dc = distance_correlation(xs[idx], ys[idx])
                cells.append(_cell("distance_correlation", 0, dc, len(idx), "MEASURED" if dc is not None else "NOT_APPLICABLE",
                                   None if dc is not None else "zero distance variance", *base_est["distance_correlation"]))
            except Exception as exc:  # noqa: BLE001 - a failed estimator is a disposition, not a crash
                have = {c["metric"] for c in cells}
                for m in BASE_METRICS:
                    if m not in have:
                        est, p = base_est[m]
                        cells.append(_cell(m, 0, None, n, "FAILED", f"{type(exc).__name__}: {exc}", est, p))
    # cross-correlation at elapsed-hour lags
    lags = sorted({int(l) for l in params["lags_hours"]})
    if 0 in lags:
        pz = next(c for c in cells if c["metric"] == "pearson")
        cells.append(_cell("xcorr", 0, pz["value"], pz["support_n"], pz["state"], pz["reason"], "pearson_product_moment",
                           {"direction": "contemporaneous"}))
    for lag in lags:
        if lag <= 0:
            continue
        # position of the row observed exactly `lag` elapsed hours before each row
        target = ts - lag * 3600
        pos = np.searchsorted(ts, target)
        pos = np.clip(pos, 0, len(ts) - 1)
        ok = ts[pos] == target
        later = np.nonzero(ok)[0]
        earlier = pos[ok]
        for metric, lead, lagged in (("xcorr_x_leads", x, y), ("xcorr_y_leads", y, x)):
            # lead[t - lag] against lagged[t]
            a = lead[earlier]
            b = lagged[later]
            m = np.isfinite(a) & np.isfinite(b)
            k = int(m.sum())
            est = "pearson_product_moment"
            p = {"lag_hours": lag, "alignment": "elapsed_hours_exact_timestamp"}
            if len(later) == 0:
                cells.append(_cell(metric, lag, None, 0, "NOT_APPLICABLE", "no rows at this elapsed lag on the bar grid", est, p))
            elif k < min_support:
                cells.append(_cell(metric, lag, None, k, "INSUFFICIENT_SUPPORT", f"aligned finite rows {k} < min_support {min_support}", est, p))
            else:
                v = pearson(a[m], b[m])
                if v is None:
                    cells.append(_cell(metric, lag, None, k, "NOT_APPLICABLE", "zero variance on aligned rows", est, p))
                else:
                    cells.append(_cell(metric, lag, v, k, "MEASURED", None, est, p))
    return cells


def gate_pair(x: np.ndarray, y: np.ndarray, meta_l: dict, meta_r: dict, params: dict) -> dict:
    fx, fy = np.isfinite(x), np.isfinite(y)
    shared = fx & fy
    n = int(shared.sum())
    union = int((fx | fy).sum())
    disagreement = int((fx ^ fy).sum())
    out: dict[str, Any] = {
        "shared_support": n, "left_finite": int(fx.sum()), "right_finite": int(fy.sum()),
        "mask_disagreement": disagreement, "mask_jaccard": (n / union) if union else None,
        "byte_identical": False, "numeric_equal": False, "exact_affine": False, "affine_a": None, "affine_b": None,
        "affine_max_rel_residual": None, "near_perfect_monotone": False, "abs_spearman": None,
    }
    for key, name in (("source", "same_source"), ("family", "same_underlying"), ("transform", "same_transform"),
                      ("unit", "same_unit"), ("availability_time", "same_availability"), ("clock", "same_clock")):
        out[name] = (meta_l.get(key) == meta_r.get(key)) if (meta_l.get(key) is not None or meta_r.get(key) is not None) else None
    out["coverage_left"], out["coverage_right"] = meta_l.get("train_coverage"), meta_r.get("train_coverage")
    out["support_h_left"], out["support_h_right"] = meta_l.get("support_h"), meta_r.get("support_h")
    state = "DISTINCT"
    if n == 0:
        state = "NO_SHARED_SUPPORT"
    else:
        xs, ys = x[shared], y[shared]
        if np.array_equal(xs.view(np.uint64), ys.view(np.uint64)):
            out["byte_identical"] = True
            out["numeric_equal"] = True
            state = "BYTE_IDENTICAL"
        elif np.allclose(xs, ys, rtol=float(params["numeric_rtol"]), atol=float(params["numeric_atol"])):
            out["numeric_equal"] = True
            state = "NUMERIC_EQUAL"
        else:
            sx = xs.std()
            if n >= int(params["min_support"]) and sx > 0 and ys.std() > 0:
                a, b = np.polyfit(xs, ys, 1)
                resid = np.max(np.abs(ys - (a * xs + b)))
                scale = float(np.max(np.abs(ys))) or 1.0
                rel = float(resid / scale)
                out["affine_a"], out["affine_b"], out["affine_max_rel_residual"] = float(a), float(b), rel
                if a != 0.0 and rel <= float(params["affine_rel_tol"]):
                    out["exact_affine"] = True
                    state = "EXACT_AFFINE"
                else:
                    rho = pearson(_rank(xs), _rank(ys))
                    out["abs_spearman"] = abs(rho) if rho is not None else None
                    if rho is not None and abs(rho) >= float(params["monotone_abs_spearman"]):
                        out["near_perfect_monotone"] = True
                        state = "NEAR_PERFECT_MONOTONE"
            elif n < int(params["min_support"]):
                state = "INSUFFICIENT_SUPPORT"
    out["gate_state"] = state
    out["is_alias_candidate"] = state in GATE_ALIAS_STATES
    return out


def stability_row(values_by_fold: list[tuple[str, dict]]) -> dict:
    measured = [c["value"] for _, c in values_by_fold if c["state"] == "MEASURED" and c["value"] is not None]
    valid = len(measured)
    if valid == 0:
        reasons = sorted({c["state"] for _, c in values_by_fold})
        return {"mean": None, "sd": None, "sign_consistent": None, "min": None, "max": None, "valid_folds": 0,
                "state": "INSUFFICIENT_SUPPORT" if "INSUFFICIENT_SUPPORT" in reasons else reasons[0],
                "fold_states": {f: c["state"] for f, c in values_by_fold}}
    arr = np.asarray(measured, dtype=np.float64)
    signs = np.sign(arr)
    return {"mean": float(arr.mean()), "sd": float(arr.std()), "sign_consistent": bool(np.all(signs == signs[0])),
            "min": float(arr.min()), "max": float(arr.max()), "valid_folds": valid, "state": "MEASURED",
            "fold_states": {f: c["state"] for f, c in values_by_fold}}


# ----------------------------------------------------------------------------- shard

def compute_shard(data: PopulationData, plan: dict, shard_index: int, pairs: list[tuple[str, str]], host_id: str) -> dict:
    """Unsealed terminal for one shard: every expected row for every pair."""
    params = plan["params"]
    run_id = plan["identity"]
    pop = plan["population_id"]
    unit = plan["shards"][shard_index]["unit_id"]
    code = code_sha256()
    psha = plan["params_sha256"]
    started = time.time()
    folds = data.folds()
    slots = metric_slots(params)
    metric_rows: list[dict] = []
    stability_rows: list[dict] = []
    gate_rows: list[dict] = []
    common = {"run_id": run_id, "population_id": pop, "unit_id": unit, "method": params["method"],
              "params_sha256": psha, "code_sha256": code}
    columns = sorted({data.index[f] for pair in pairs for f in pair})
    per_pair: dict[tuple[str, str], dict[str, dict[tuple[str, int], dict]]] = {pair: {} for pair in pairs}
    # fold-outer loop: average ranks of every complete column are computed once per fold (float32 holds
    # ranks up to 2**24 exactly) and shared by every pair of complete columns in the shard
    for fold_id, end in folds:
        ranks: dict[int, np.ndarray] = {}
        for i in columns:
            col = data.X[i, :end]
            if np.isfinite(col).all():
                ranks[i] = _rank(col).astype(np.float32)
        ts = data.ts_seconds[:end]
        for left, right in pairs:
            i, j = data.index[left], data.index[right]
            cells = pair_fold_cells(data.X[i, :end], data.X[j, :end], ts, params, ranks.get(i), ranks.get(j))
            per_pair[(left, right)][fold_id] = {(c["metric"], c["lag_hours"]): c for c in cells}
        del ranks
    for left, right in pairs:
        i, j = data.index[left], data.index[right]
        xf, yf = data.X[i], data.X[j]
        per_fold = per_pair[(left, right)]
        for fold_id, _ in folds:
            for c in per_fold[fold_id].values():
                metric_rows.append({**common, "left": left, "right": right, "fold_id": fold_id, **c,
                                    "row_key": row_key(run_id, "metric", left, right, fold_id, c["metric"], c["lag_hours"])})
        inner = [f for f, _ in folds if f != "TRAIN"]
        for metric, lag in slots:
            st = stability_row([(f, per_fold[f][(metric, lag)]) for f in inner])
            stability_rows.append({**common, "left": left, "right": right, "metric": metric, "lag_hours": lag,
                                   "fold_count": len(inner), **st,
                                   "row_key": row_key(run_id, "stability", left, right, metric, lag)})
        g = gate_pair(xf, yf, data.meta[left], data.meta[right], params)
        gate_rows.append({**common, "left": left, "right": right, "fold_id": "TRAIN", **g,
                          "row_key": row_key(run_id, "gate", left, right)})
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    finished = time.time()
    return {
        "schema": SCHEMA_TERMINAL, "unit_id": unit, "population_id": pop, "identity": run_id, "shard_index": shard_index,
        "method": params["method"], "params_sha256": psha, "plan_sha256": plan["plan_sha256"], "code_sha256": code,
        "host_id": host_id, "pair_count": len(pairs), "started_at": started, "finished_at": finished,
        "wall_seconds": finished - started, "peak_rss_bytes": int(peak_kb) * 1024,
        "rows": {"feature_pair_metrics": metric_rows, "feature_pair_stability": stability_rows, "feature_pair_gate": gate_rows},
    }


def expected_row_counts(pair_count: int, fold_count: int, params: dict) -> dict:
    slots = len(metric_slots(params))
    return {"feature_pair_metrics": pair_count * (fold_count + 1) * slots,
            "feature_pair_stability": pair_count * slots,
            "feature_pair_gate": pair_count}


def expected_row_keys(run_id: str, pairs: list[tuple[str, str]], fold_ids: list[str], params: dict) -> dict[str, set[str]]:
    slots = metric_slots(params)
    folds = ["TRAIN", *fold_ids]
    metrics, stab, gate = set(), set(), set()
    for left, right in pairs:
        for fold in folds:
            for metric, lag in slots:
                metrics.add(row_key(run_id, "metric", left, right, fold, metric, lag))
        for metric, lag in slots:
            stab.add(row_key(run_id, "stability", left, right, metric, lag))
        gate.add(row_key(run_id, "gate", left, right))
    return {"feature_pair_metrics": metrics, "feature_pair_stability": stab, "feature_pair_gate": gate}


def limit_numeric_threads(threads: int = 1) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(int(threads))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
