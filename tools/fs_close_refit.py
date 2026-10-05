#!/usr/bin/env python3
"""FS-CLOSE paired refits at common K on identical TRAIN inner folds (order 2026-10-05 section 8).

Worker-side engine. It receives a sealed plan of candidate feature sets (ALL_ADMISSIBLE,
RANDOM_K, the best predictive/redundancy selector, +causal, +representation, knockoff) and
refits, for every set x K x head x target x horizon x inner fold, the SAME small head on the
SAME rows with the SAME initialisation, updates and seed, and reports the error next to the
same-row naive. Nothing here reads VALIDATION (2024) or TEST (2025): the loader refuses any
decision row at or after the declared TRAIN end.

Heads (declared, fixed, one seed):
  ridge  closed-form ridge, alpha 1.0 on fit-row standardised inputs, intercept = fit-row mean
         (regression targets); multinomial logistic regression, C = 1.0, lbfgs, max_iter 200,
         seed 0 (barrier targets, needs scikit-learn).
  hgb    HistGradientBoosting, max_iter 100, max_leaf_nodes 15, learning_rate 0.1, seed 0,
         no early stopping (needs scikit-learn).

Rows: a cell's rows are every fold row whose target is finite (barrier: has support). They
depend on the target and the fold only, never on the feature set, so every set is scored on
identical rows; the digest of those rows is written in every metric row. Missing feature values
are imputed with the fit-row median of that feature (declared; identical for every set).

Naive, same rows: zero return and fit-row mean for regression; fit-row class prior for the
barrier. Skill = 1 - loss_model / loss_naive; strictly positive means the head beat the naive.

Output: a long table (parquet) with one row per metric x naive pair, plus a receipt with input
digests, plan digest, code digest, seed, head parameters and the whole-process peak RSS.
Cells already present for an unchanged set definition are not recomputed (idempotent).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

PLAN_SCHEMA = "fs_close_refit_plan.v1"
METRICS_SCHEMA = "fs_close_paired_refit_metrics.v1"
RECEIPT_SCHEMA = "fs_close_refit_receipt.v1"
TRAIN_END_UTC = "2024-01-01T00:00:00+00:00"  # decision rows must be strictly before this
REGRESSION_TARGETS = {
    "Y_s": (1, 2, 3, 4, 5, 6),
    "Y_l": (24, 48, 72, 96, 120, 144),
}
BARRIER_TARGETS = {"Y_b": ((6, "Y_b_s6"), (144, "Y_b_l144"))}
HEAD_PARAMS = {
    "ridge": {"alpha": 1.0, "standardise": "fit-row mean/std", "intercept": "fit-row mean",
              "barrier": {"estimator": "LogisticRegression", "C": 1.0, "solver": "lbfgs",
                          "max_iter": 200, "random_state": 0}},
    "hgb": {"max_iter": 100, "max_leaf_nodes": 15, "learning_rate": 0.1, "random_state": 0,
            "early_stopping": False},
}
SEED = 0
METRIC_COLUMNS = (
    "set_id", "set_kind", "set_sha256", "k", "head", "target", "horizon", "fold", "metric",
    "value", "naive_kind", "naive_value", "skill", "n_fit", "n_rows", "rows_sha256",
    "population_sha256", "features_sha256", "n_features", "seed", "fit_seconds",
    "predict_ms_per_row", "peak_rss_bytes", "plan_sha256", "code_sha256", "state", "reason",
)


class RefitError(RuntimeError):
    """A plan, input or guard violation that must stop the run explicitly."""


# ----------------------------------------------------------------------------- digests
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def names_sha256(names) -> str:
    return sha256_bytes("\n".join(names).encode())


def code_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


# ----------------------------------------------------------------------------- guards
def assert_train_only(timestamps_s: np.ndarray, train_end_utc: str = TRAIN_END_UTC) -> dict:
    """Refuse rows at/after the TRAIN end; return the bound evidence for the receipt."""
    import datetime as dt

    end = dt.datetime.fromisoformat(train_end_utc).timestamp()
    ts = np.asarray(timestamps_s, dtype="int64")
    if ts.size == 0:
        raise RefitError("NO_ROWS")
    if int(ts.max()) >= end:
        raise RefitError(f"TEST_OR_VALIDATION_READ_REFUSED: max decision time {int(ts.max())} >= "
                         f"train end {train_end_utc}")
    return {"rows": int(ts.size), "min_ts": int(ts.min()), "max_ts": int(ts.max()),
            "train_end_utc": train_end_utc}


# ----------------------------------------------------------------------------- loading
def load_inputs(feature_files, targets_file, folds_file, population):
    """Join PS1 feature batches and the target file on row_id; return arrays and digests.

    Columns are copied one at a time into a preallocated float64 matrix so the peak stays close to
    one copy of X (declared footprint, measured in the receipt).
    """
    import pyarrow.parquet as pq

    digests = {}
    col_index = {name: i for i, name in enumerate(population)}
    X = None
    row_ids = None
    ts = None
    seen = set()
    for fp in feature_files:
        fp = Path(fp)
        digests[fp.name + "@" + fp.parent.name] = sha256_file(fp)
        pf = pq.ParquetFile(fp)
        names = pf.schema.names
        ids = pf.read(columns=["row_id"]).column("row_id").to_numpy()
        if row_ids is None:
            row_ids = ids
            tcol = pf.read(columns=["t_decision_utc"]).column("t_decision_utc")
            ts = (tcol.cast("int64").to_numpy() // 10**9) if "ns" in str(tcol.type) else tcol.cast("int64").to_numpy()
            X = np.empty((len(row_ids), len(population)), dtype="float64")
        elif not np.array_equal(ids, row_ids):
            raise RefitError(f"ROW_ID_MISMATCH between feature batches: {fp}")
        wanted = [n for n in names if n in col_index]
        for start in range(0, len(wanted), 32):
            chunk = wanted[start:start + 32]
            tbl = pf.read(columns=chunk)
            for name in chunk:
                X[:, col_index[name]] = tbl.column(name).to_numpy(zero_copy_only=False).astype("float64", copy=False)
                seen.add(name)
            del tbl
    missing = [c for c in population if c not in seen]
    if missing:
        raise RefitError(f"POPULATION_NOT_IN_INPUTS: {len(missing)} first {missing[:5]}")
    tp = Path(targets_file)
    digests[tp.name] = sha256_file(tp)
    tdf = pq.read_table(tp).to_pandas()
    if not np.array_equal(tdf["row_id"].to_numpy(), row_ids):
        raise RefitError("ROW_ID_MISMATCH between features and targets")
    bound = assert_train_only(ts)
    folds = json.loads(Path(folds_file).read_text())
    digests["folds.json"] = sha256_file(Path(folds_file))
    return X, row_ids, tdf, folds, digests, bound


def target_vector(tdf, target: str, horizon: int):
    """Return (y, kind) for one target cell. kind in {regression, barrier}."""
    if target in REGRESSION_TARGETS:
        col = f"{target}_{horizon}h"
        y = tdf[col].to_numpy(dtype="float64")
        return y, "regression"
    for h, col in BARRIER_TARGETS["Y_b"]:
        if target == "Y_b" and h == horizon:
            raw = tdf[col].to_numpy(dtype="float64")
            y = np.full(raw.shape, np.nan)
            ok = np.isfinite(raw) & np.isin(raw, (-1.0, 0.0, 1.0))
            y[ok] = raw[ok] + 1.0  # 0 = SL first, 1 = timeout, 2 = TP first
            return y, "barrier"
    raise RefitError(f"UNKNOWN_TARGET {target} h{horizon}")


def all_cells():
    for t, hs in REGRESSION_TARGETS.items():
        for h in hs:
            yield t, h
    for h, _ in BARRIER_TARGETS["Y_b"]:
        yield "Y_b", h


# ----------------------------------------------------------------------------- heads
def _standardise(Xf, Xv):
    """In place on the fold copies (they are already private copies of X)."""
    mu = Xf.mean(axis=0)
    sd = Xf.std(axis=0)
    sd[sd == 0] = 1.0
    Xf -= mu
    Xf /= sd
    Xv -= mu
    Xv /= sd
    return Xf, Xv


def _impute(Xf, Xv):
    """Fit-row median imputation, in place, column by column."""
    for j in range(Xf.shape[1]):
        cf = Xf[:, j]
        bad = ~np.isfinite(cf)
        if bad.any():
            med = np.nanmedian(cf) if (~bad).any() else 0.0
            med = med if np.isfinite(med) else 0.0
            cf[bad] = med
        else:
            med = None
        cv = Xv[:, j]
        badv = ~np.isfinite(cv)
        if badv.any():
            cv[badv] = med if med is not None else (np.nanmedian(cf) if np.isfinite(np.nanmedian(cf)) else 0.0)
    return Xf, Xv


def fit_predict(head: str, kind: str, Xf, yf, Xv):
    """Fit one head on fit rows and predict val rows.

    Returns (predictions or class probabilities, fit_seconds, predict_seconds).
    """
    Xf, Xv = _impute(Xf, Xv)
    if head == "ridge":
        Zf, Zv = _standardise(Xf, Xv)
        if kind == "regression":
            alpha = HEAD_PARAMS["ridge"]["alpha"]
            t0 = time.time()
            ym = yf.mean()
            A = Zf.T @ Zf + alpha * np.eye(Zf.shape[1])
            w = np.linalg.solve(A, Zf.T @ (yf - ym))
            t1 = time.time()
            pred = Zv @ w + ym
            return pred, t1 - t0, time.time() - t1
        from sklearn.linear_model import LogisticRegression

        p = HEAD_PARAMS["ridge"]["barrier"]
        clf = LogisticRegression(C=p["C"], solver=p["solver"], max_iter=p["max_iter"],
                                 random_state=p["random_state"])
        t0 = time.time()
        clf.fit(Zf, yf.astype(int))
        t1 = time.time()
        pred = _full_proba(clf, Zv)
        return pred, t1 - t0, time.time() - t1
    if head == "hgb":
        p = HEAD_PARAMS["hgb"]
        if kind == "regression":
            from sklearn.ensemble import HistGradientBoostingRegressor

            m = HistGradientBoostingRegressor(max_iter=p["max_iter"], max_leaf_nodes=p["max_leaf_nodes"],
                                              learning_rate=p["learning_rate"],
                                              random_state=p["random_state"], early_stopping=False)
            t0 = time.time()
            m.fit(Xf, yf)
            t1 = time.time()
            pred = m.predict(Xv)
            return pred, t1 - t0, time.time() - t1
        from sklearn.ensemble import HistGradientBoostingClassifier

        m = HistGradientBoostingClassifier(max_iter=p["max_iter"], max_leaf_nodes=p["max_leaf_nodes"],
                                           learning_rate=p["learning_rate"],
                                           random_state=p["random_state"], early_stopping=False)
        t0 = time.time()
        m.fit(Xf, yf.astype(int))
        t1 = time.time()
        pred = _full_proba(m, Xv)
        return pred, t1 - t0, time.time() - t1
    raise RefitError(f"UNKNOWN_HEAD {head}")


def _full_proba(model, Z):
    pr = model.predict_proba(Z)
    out = np.full((Z.shape[0], 3), 1e-12)
    for j, c in enumerate(model.classes_):
        out[:, int(c)] = pr[:, j]
    out /= out.sum(axis=1, keepdims=True)
    return out


# ----------------------------------------------------------------------------- metrics
def regression_metrics(y, pred, y_fit_mean):
    err = pred - y
    mae, mse = float(np.mean(np.abs(err))), float(np.mean(err**2))
    zero_mae, zero_mse = float(np.mean(np.abs(y))), float(np.mean(y**2))
    mean_mae = float(np.mean(np.abs(y - y_fit_mean)))
    mean_mse = float(np.mean((y - y_fit_mean) ** 2))
    return [("mae", mae, "zero", zero_mae), ("mse", mse, "zero", zero_mse),
            ("mae", mae, "fit_mean", mean_mae), ("mse", mse, "fit_mean", mean_mse)]


def barrier_metrics(y, proba, prior):
    yi = y.astype(int)
    n = yi.size
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), yi] = 1.0
    eps = 1e-12
    ll = float(-np.mean(np.log(np.clip(proba[np.arange(n), yi], eps, 1.0))))
    prior_ll = float(-np.mean(np.log(np.clip(prior[yi], eps, 1.0))))
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    prior_brier = float(np.mean(np.sum((prior[None, :] - onehot) ** 2, axis=1)))
    return [("log_loss", ll, "fit_prior", prior_ll), ("brier", brier, "fit_prior", prior_brier)]


def skill(value, naive):
    return float(1.0 - value / naive) if naive and np.isfinite(naive) and naive > 0 else float("nan")


# ----------------------------------------------------------------------------- plan
def resolve_features(set_entry: dict, target_key: str, fold: str):
    """Features for one cell: 'target|fold', then 'target', then '*|fold', then flat list."""
    by = set_entry.get("features_by")
    if isinstance(by, dict):
        for key in (f"{target_key}|{fold}", target_key, f"*|{fold}"):
            if key in by:
                return list(by[key])
        if "*" in by:
            return list(by["*"])
        return None
    feats = set_entry.get("features")
    return list(feats) if isinstance(feats, list) else None


def load_plan(path: Path) -> dict:
    plan = json.loads(Path(path).read_text())
    if plan.get("schema") != PLAN_SCHEMA:
        raise RefitError(f"WRONG_PLAN_SCHEMA {plan.get('schema')!r}")
    pop = plan.get("population")
    if not isinstance(pop, list) or len(pop) != len(set(pop)):
        raise RefitError("PLAN_POPULATION_INVALID")
    if plan.get("population_sha256") != names_sha256(sorted(pop)):
        raise RefitError("PLAN_POPULATION_DIGEST_MISMATCH")
    popset = set(pop)
    for s in plan.get("sets", []):
        for key in ("set_id", "set_kind"):
            if not s.get(key):
                raise RefitError(f"SET_MISSING_{key.upper()}")
        lists = list(s["features_by"].values()) if isinstance(s.get("features_by"), dict) else [s.get("features")]
        for lst in lists:
            if not isinstance(lst, list) or not lst:
                raise RefitError(f"SET_{s['set_id']}_EMPTY_FEATURE_LIST")
            bad = [f for f in lst if f not in popset]
            if bad:
                raise RefitError(f"SET_{s['set_id']}_OUTSIDE_POPULATION: {bad[:3]}")
            k = s.get("k")
            if k is not None and len(lst) != k:
                raise RefitError(f"SET_{s['set_id']}_K_MISMATCH: declared {k}, got {len(lst)}")
    return plan


def set_sha256(s: dict) -> str:
    body = {k: v for k, v in s.items() if k not in ("set_sha256",)}
    return canonical_sha256(body)


# ----------------------------------------------------------------------------- run
def existing_keys(out_path: Path) -> set:
    if not out_path.is_file():
        return set()
    import pyarrow.parquet as pq

    t = pq.read_table(out_path, columns=["set_sha256", "k", "head", "target", "horizon", "fold"])
    d = t.to_pydict()
    return {(d["set_sha256"][i], d["k"][i], d["head"][i], d["target"][i], d["horizon"][i], d["fold"][i])
            for i in range(t.num_rows)}


def append_rows(out_path: Path, rows: list):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        return
    table = pa.Table.from_pylist(rows)
    if out_path.is_file():
        old = pq.read_table(out_path)
        table = pa.concat_tables([old, table.select(old.column_names)], promote_options="default")
    tmp = out_path.with_suffix(".tmp.parquet")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, out_path)


def run_plan(plan_path: Path, feature_files, targets_file, folds_file, out_dir: Path,
             heads=("ridge",), hgb_max_k=24, fold_names=None, progress=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = load_plan(plan_path)
    plan_digest = sha256_file(Path(plan_path))
    population = list(plan["population"])
    pop_digest = plan["population_sha256"]
    X, row_ids, tdf, folds, digests, bound = load_inputs(feature_files, targets_file, folds_file, population)
    col = {name: i for i, name in enumerate(population)}
    out_path = out_dir / "paired_refit_metrics.parquet"
    done = existing_keys(out_path)
    code_digest = code_sha256()
    started = time.time()
    cells_done = cells_skipped = cells_failed = 0
    fold_list = [f for f in folds["folds"] if not fold_names or f["name"] in fold_names]
    rows_buffer = []
    _write_progress(out_dir, progress, started, 0, 0, 0)   # RUNNING from the first second, before any fit
    for head, s in [(h, s) for h in heads for s in plan["sets"] if h in (s.get("heads") or heads)]:
        s_digest = set_sha256(s)
        k = s.get("k")
        k_val = int(k) if k is not None else -1
        if True:
            if head == "hgb" and k is not None and k > hgb_max_k:
                continue
            for target, horizon in all_cells():
                y_all, kind = target_vector(tdf, target, horizon)
                target_key = f"{target}_{horizon}h"
                for fd in fold_list:
                    fold = fd["name"]
                    key = (s_digest, k_val, head, target, horizon, fold)
                    if key in done:
                        cells_skipped += 1
                        continue
                    feats = resolve_features(s, target_key, fold)
                    if feats is None:
                        cells_failed += 1
                        rows_buffer.append(_failed_row(s, s_digest, k_val, head, target, horizon, fold,
                                                       pop_digest, plan_digest, code_digest,
                                                       "NO_FEATURE_LIST_FOR_CELL"))
                        continue
                    a, b = fd["train_rows"]
                    c, d = fd["val_rows"]
                    yf_all, yv_all = y_all[a:b], y_all[c:d]
                    fit_mask, val_mask = np.isfinite(yf_all), np.isfinite(yv_all)
                    fit_idx = np.arange(a, b)[fit_mask]
                    val_idx = np.arange(c, d)[val_mask]
                    if fit_idx.size < 100 or val_idx.size < 10:
                        cells_failed += 1
                        rows_buffer.append(_failed_row(s, s_digest, k_val, head, target, horizon, fold,
                                                       pop_digest, plan_digest, code_digest,
                                                       f"TOO_FEW_ROWS fit={fit_idx.size} val={val_idx.size}"))
                        continue
                    idx = [col[f] for f in feats]
                    Xf, Xv = X[np.ix_(fit_idx, idx)], X[np.ix_(val_idx, idx)]
                    yf, yv = y_all[fit_idx], y_all[val_idx]
                    rows_digest = sha256_bytes(row_ids[val_idx].astype("int64").tobytes())
                    try:
                        pred, fit_seconds, predict_seconds = fit_predict(head, kind, Xf, yf, Xv)
                    except Exception as exc:  # retained as FAILED, never dropped
                        cells_failed += 1
                        rows_buffer.append(_failed_row(s, s_digest, k_val, head, target, horizon, fold,
                                                       pop_digest, plan_digest, code_digest,
                                                       f"{type(exc).__name__}: {exc}"[:300]))
                        continue
                    predict_ms = predict_seconds * 1000.0 / max(1, val_idx.size)
                    if kind == "regression":
                        metrics = regression_metrics(yv, pred, float(yf.mean()))
                    else:
                        prior = np.bincount(yf.astype(int), minlength=3) / yf.size
                        metrics = barrier_metrics(yv, pred, prior)
                    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                    for metric, value, naive_kind, naive_value in metrics:
                        rows_buffer.append({
                            "set_id": s["set_id"], "set_kind": s["set_kind"], "set_sha256": s_digest,
                            "k": k_val, "head": head, "target": target, "horizon": int(horizon),
                            "fold": fold, "metric": metric, "value": float(value),
                            "naive_kind": naive_kind, "naive_value": float(naive_value),
                            "skill": skill(value, naive_value), "n_fit": int(fit_idx.size),
                            "n_rows": int(val_idx.size), "rows_sha256": rows_digest,
                            "population_sha256": pop_digest, "features_sha256": names_sha256(feats),
                            "n_features": len(feats), "seed": SEED, "fit_seconds": float(fit_seconds),
                            "predict_ms_per_row": float(predict_ms), "peak_rss_bytes": int(peak),
                            "plan_sha256": plan_digest, "code_sha256": code_digest,
                            "state": "MEASURED", "reason": "",
                        })
                    cells_done += 1
                    if len(rows_buffer) >= 2000:
                        append_rows(out_path, rows_buffer)
                        rows_buffer = []
                        _write_progress(out_dir, progress, started, cells_done, cells_skipped, cells_failed)
    append_rows(out_path, rows_buffer)
    receipt = {
        "schema": RECEIPT_SCHEMA, "plan_sha256": plan_digest, "population_sha256": pop_digest,
        "inputs_sha256": digests, "train_bound": bound, "heads": list(heads),
        "head_params": {h: HEAD_PARAMS[h] for h in heads}, "hgb_max_k": hgb_max_k, "seed": SEED,
        "code_sha256": code_digest, "started_utc": _iso(started), "finished_utc": _iso(time.time()),
        "wall_seconds": time.time() - started, "cells_done": cells_done, "cells_skipped": cells_skipped,
        "cells_failed": cells_failed,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "children_peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
        "output": str(out_path), "output_sha256": sha256_file(out_path) if out_path.is_file() else None,
        "imputation": "fit-row median per feature (identical for every set)",
        "rows_rule": "every fold row with a finite target (barrier: supported label); independent of the set",
    }
    (out_dir / "refit_receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    _write_progress(out_dir, progress, started, cells_done, cells_skipped, cells_failed, final=True)
    return receipt


def _failed_row(s, s_digest, k_val, head, target, horizon, fold, pop_digest, plan_digest, code_digest, reason):
    return {"set_id": s["set_id"], "set_kind": s["set_kind"], "set_sha256": s_digest, "k": k_val,
            "head": head, "target": target, "horizon": int(horizon), "fold": fold, "metric": "",
            "value": float("nan"), "naive_kind": "", "naive_value": float("nan"), "skill": float("nan"),
            "n_fit": 0, "n_rows": 0, "rows_sha256": "", "population_sha256": pop_digest,
            "features_sha256": "", "n_features": 0, "seed": SEED, "fit_seconds": 0.0,
            "predict_ms_per_row": 0.0, "peak_rss_bytes": 0, "plan_sha256": plan_digest,
            "code_sha256": code_digest, "state": "FAILED", "reason": reason}


def _iso(t: float) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()


def _write_progress(out_dir, progress, started, done, skipped, failed, final=False):
    p = Path(progress) if progress else out_dir / "progress.json"
    body = {"schema": "fs_close_refit_progress.v1", "updated_utc": _iso(time.time()),
            "cells_done": done, "cells_skipped": skipped, "cells_failed": failed,
            "elapsed_seconds": time.time() - started, "state": "FINISHED" if final else "RUNNING"}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=1))
    os.replace(tmp, p)


# ----------------------------------------------------------------------------- closure step (VALIDATION read once)
CLOSURE_SCHEMA = "fs_close_closure_record.v1"
VALIDATION_START_UTC = "2024-01-01T00:00:00+00:00"
VALIDATION_END_UTC = "2025-01-01T00:00:00+00:00"


def assert_validation_window(timestamps_s: np.ndarray) -> dict:
    """Refuse any row outside [VALIDATION_START, VALIDATION_END): TEST (2025) is never read."""
    import datetime as dt

    lo = dt.datetime.fromisoformat(VALIDATION_START_UTC).timestamp()
    hi = dt.datetime.fromisoformat(VALIDATION_END_UTC).timestamp()
    ts = np.asarray(timestamps_s, dtype="int64")
    if ts.size == 0:
        raise RefitError("NO_VALIDATION_ROWS")
    if int(ts.max()) >= hi:
        raise RefitError(f"TEST_READ_REFUSED: max validation decision time {int(ts.max())} >= {VALIDATION_END_UTC}")
    if int(ts.min()) < lo:
        raise RefitError(f"VALIDATION_WINDOW_VIOLATED: min decision time {int(ts.min())} < {VALIDATION_START_UTC}")
    return {"rows": int(ts.size), "min_ts": int(ts.min()), "max_ts": int(ts.max()),
            "window": [VALIDATION_START_UTC, VALIDATION_END_UTC]}


def load_validation(feature_files, targets_file, population):
    """Same layout as the TRAIN inputs, bounded to the external validation year."""
    import pyarrow.parquet as pq

    col_index = {name: i for i, name in enumerate(population)}
    X = None
    row_ids = None
    ts = None
    digests = {}
    seen = set()
    for fp in feature_files:
        fp = Path(fp)
        digests["validation:" + fp.name + "@" + fp.parent.name] = sha256_file(fp)
        pf = pq.ParquetFile(fp)
        ids = pf.read(columns=["row_id"]).column("row_id").to_numpy()
        if row_ids is None:
            row_ids = ids
            tcol = pf.read(columns=["t_decision_utc"]).column("t_decision_utc")
            ts = (tcol.cast("int64").to_numpy() // 10**9) if "ns" in str(tcol.type) else tcol.cast("int64").to_numpy()
            X = np.empty((len(row_ids), len(population)), dtype="float64")
        elif not np.array_equal(ids, row_ids):
            raise RefitError(f"ROW_ID_MISMATCH between validation feature batches: {fp}")
        for name in [n for n in pf.schema.names if n in col_index]:
            X[:, col_index[name]] = pf.read(columns=[name]).column(name).to_numpy(zero_copy_only=False).astype("float64", copy=False)
            seen.add(name)
    missing = [c for c in population if c not in seen]
    if missing:
        raise RefitError(f"VALIDATION_POPULATION_NOT_IN_INPUTS: {len(missing)} first {missing[:5]}")
    tp = Path(targets_file)
    digests["validation:" + tp.name] = sha256_file(tp)
    tdf = pq.read_table(tp).to_pandas()
    if not np.array_equal(tdf["row_id"].to_numpy(), row_ids):
        raise RefitError("ROW_ID_MISMATCH between validation features and targets")
    bound = assert_validation_window(ts)
    return X, row_ids, tdf, digests, bound


def run_closure(plan_path: Path, feature_files, targets_file, folds_file, val_feature_files, val_targets_file,
                out_dir: Path, k_primary: int = 24, head: str = "ridge") -> dict:
    """Score the frozen K=k_primary candidate sets once on VALIDATION under the declared rule.

    Fit on ALL TRAIN rows (the union of every fold's rows), score on the 2024 rows, same head, same
    seed, same naives. Writes closure_record.json with every candidate's cells and the winner.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = load_plan(plan_path)
    population = list(plan["population"])
    X, row_ids, tdf, folds, digests, bound = load_inputs(feature_files, targets_file, folds_file, population)
    Xv, vrow_ids, vdf, vdigests, vbound = load_validation(val_feature_files, val_targets_file, population)
    col = {name: i for i, name in enumerate(population)}
    candidates = [s for s in plan["sets"] if s["set_kind"] in ("ALL_ADMISSIBLE", "PRED_BEST", "PLUS_CAUSAL", "PLUS_REP", "KNOCKOFF")
                  and (s.get("k") in (None, k_primary))]
    results = []
    for s in candidates:
        cells = []
        for target, horizon in all_cells():
            y_all, kind = target_vector(tdf, target, horizon)
            yv_all, _ = target_vector(vdf, target, horizon)
            target_key = f"{target}_{horizon}h"
            feats = resolve_features(s, target_key, "*") or resolve_features(s, target_key, folds["folds"][-1]["name"])
            if feats is None:
                # per-fold sets: the union order of the last fold is the frozen candidate for the closure
                by = s.get("features_by", {})
                feats = next((v for k, v in by.items() if k.startswith(target_key)), None)
            if feats is None:
                cells.append({"target": target, "horizon": horizon, "state": "FAILED", "reason": "NO_FEATURE_LIST"})
                continue
            idx = [col[f] for f in feats]
            fit_idx = np.where(np.isfinite(y_all))[0]
            val_idx = np.where(np.isfinite(yv_all))[0]
            Xf, Xvv = X[np.ix_(fit_idx, idx)], Xv[np.ix_(val_idx, idx)]
            yf, yv = y_all[fit_idx], yv_all[val_idx]
            pred, fit_s, pred_s = fit_predict(head, kind, Xf, yf, Xvv)
            if kind == "regression":
                metrics = regression_metrics(yv, pred, float(yf.mean()))
            else:
                prior = np.bincount(yf.astype(int), minlength=3) / yf.size
                metrics = barrier_metrics(yv, pred, prior)
            primary = [m for m in metrics if m[0] in ("mae", "log_loss")]
            stricter = min(skill(v, nv) for _, v, _, nv in primary)
            cells.append({"target": target, "horizon": horizon, "state": "MEASURED", "n_fit": int(fit_idx.size),
                          "n_rows": int(val_idx.size), "rows_sha256": sha256_bytes(vrow_ids[val_idx].astype("int64").tobytes()),
                          "metrics": [{"metric": m, "value": v, "naive_kind": nk, "naive_value": nv, "skill": skill(v, nv)} for m, v, nk, nv in metrics],
                          "skill_vs_stricter_naive": stricter, "fit_seconds": fit_s, "n_features": len(feats)})
        measured = [c for c in cells if c["state"] == "MEASURED"]
        score = float(np.mean([c["skill_vs_stricter_naive"] for c in measured])) if measured else float("nan")
        short_long = [c for c in measured if c["target"] in ("Y_s", "Y_l")]
        results.append({"set_id": s["set_id"], "set_kind": s["set_kind"], "set_sha256": set_sha256(s), "k": s.get("k"),
                        "score": score, "cells_measured": len(measured), "cells": cells,
                        "n_features": max((c.get("n_features", 0) for c in measured), default=0),
                        "fit_seconds_total": float(sum(c.get("fit_seconds", 0.0) for c in measured)),
                        "beats_naive_every_short_long_cell": bool(short_long) and all(c["skill_vs_stricter_naive"] > 0 for c in short_long),
                        "beats_naive_every_cell": bool(measured) and all(c["skill_vs_stricter_naive"] > 0 for c in measured)})
    order = {"ALL_ADMISSIBLE": 0, "PRED_BEST": 1, "PLUS_CAUSAL": 2, "PLUS_REP": 3, "KNOCKOFF": 4}
    complete = [r for r in results if r["cells_measured"] == len(list(all_cells()))]
    winner = None
    if complete:
        best = sorted(complete, key=lambda r: (-r["score"], r["n_features"], r["fit_seconds_total"], order.get(r["set_kind"], 9)))[0]
        winner = {"set_id": best["set_id"], "set_kind": best["set_kind"], "set_sha256": best["set_sha256"], "score": best["score"],
                  "k": best["k"], "n_features": best["n_features"]}
    record = {"schema": CLOSURE_SCHEMA, "head": head, "head_params": HEAD_PARAMS[head], "seed": SEED, "k_primary": k_primary,
              "train_bound": bound, "validation_bound": vbound, "validation_read_count": 1,
              "inputs_sha256": {**digests, **vdigests}, "plan_sha256": sha256_file(Path(plan_path)), "code_sha256": code_sha256(),
              "candidates": results, "winner": winner,
              "strategy_eligible": bool(winner) and next(r for r in results if r["set_id"] == winner["set_id"])["beats_naive_every_short_long_cell"],
              "sensitivities": {r["set_id"]: {"score": r["score"], "beats_naive_every_cell": r["beats_naive_every_cell"]} for r in results},
              "finished_utc": _iso(time.time())}
    (out_dir / "closure_record.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--features", nargs="+", required=True, help="PS1 features_train.parquet files")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--heads", default="ridge")
    ap.add_argument("--hgb-max-k", type=int, default=24)
    ap.add_argument("--folds-only", nargs="*", default=None)
    ap.add_argument("--closure", action="store_true", help="run the VALIDATION closure step instead of the fold refits")
    ap.add_argument("--val-features", nargs="*", default=None)
    ap.add_argument("--val-targets", default=None)
    a = ap.parse_args(argv)
    if a.closure:
        rec = run_closure(Path(a.plan), a.features, a.targets, a.folds, a.val_features, a.val_targets, Path(a.out_dir))
        print(json.dumps({"winner": rec["winner"], "strategy_eligible": rec["strategy_eligible"]}))
        return 0
    receipt = run_plan(Path(a.plan), a.features, a.targets, a.folds, Path(a.out_dir),
                       heads=tuple(a.heads.split(",")), hgb_max_k=a.hgb_max_k, fold_names=a.folds_only)
    print(json.dumps({k: receipt[k] for k in ("cells_done", "cells_skipped", "cells_failed",
                                              "peak_rss_bytes", "wall_seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
