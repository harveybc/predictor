#!/usr/bin/env python3
"""Timing-target feature selection driver (JSON config) producing a TIMING_PROXY_SELECTION manifest.

Expert-demonstration / hindsight-guided learning: the label is the optimized ZigZag expert's FUTURE pivot
(``tools/timing_pivot_targets.py``, semantics timing_targets.v2 with a per-row label-support ledger); every
input is causal.  The pinned phase-2/3 engines run UNCHANGED on the timing population; this driver adds the
availability gate, the ordered inventory, quality profiles, prefix/future-perturbation checks, the
univariate FDR screen, fold stability, within-day variance, redundancy, the causal stage through
causal-inference (pack EURUSD_TIMING; labelled CONDITIONAL_PREDICTIVE_ASSOCIATION), the evaluator inputs,
an invalidation DAG, a read ledger, a gate-conformant sealed manifest, OLAP publication and snapshots.
Usefulness established downstream is for the pivot-classification PROXY only, not for an RL policy.

    python tools/timing_selection.py run         --config CFG.json --out OUT [--reuse-from PREV_OUT]
    python tools/timing_selection.py causal      --config CFG.json --out OUT      (needs tigramite)
    python tools/timing_selection.py attach-eval --out OUT --eval EVAL_DIR
    python tools/timing_selection.py tables      --out OUT
    python tools/timing_selection.py publish     --out OUT --db OLAP.sqlite
    python tools/timing_selection.py snapshot    --out OUT --dest DIR [--dest DIR2 ...]

Rules: TEST rows (2025-01-01+) are never read; the engines see TRAIN 2020-2022 only; INNER 2023 is joint
development (the expert was chosen with it); 2024 is previously observed and read once by the evaluator
after its freeze; no host name, address or private path is written into any artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import fs_phase23_manifest as man  # noqa: E402
from tools import timing_pivot_targets as tp  # noqa: E402
from tools import timing_price_pool as pool  # noqa: E402

SCHEMA = "predictor.timing_selection.v1"
MANIFEST_SCHEMA = "feature_selection.manifest.v1"
SELECTION_NAME = "TIMING_PROXY_SELECTION"
SEAL_STATUS = "TIMING_PROXY_SELECTION_SEALED_PENDING_INDEPENDENT_DECISION"
FRACTAL_COLUMNS = ("hurst_rs", "variogram_hurst", "hurst_spectrum_width", "box_counting_dimension")
RAW_LEVEL_COLUMNS = ("OPEN", "HIGH", "LOW", "CLOSE")
INNER_A_END = "2023-07-01"
BAR_CONVENTION = "BAR_START_LABEL_LAKE_CLOCK_AS_LABELLED"


class DriverError(ValueError):
    pass


def _utc(s):
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, default=_jsonable) + "\n")


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (pd.Timestamp,)):
        return str(v)
    raise TypeError(type(v).__name__)


def gate_digest(doc: dict, drop: str) -> str:
    """The selected-manifest gate's canonical form (sort_keys, compact separators) without ``drop``."""
    body = {k: v for k, v in doc.items() if k != drop}
    return hashlib.sha256(json.dumps(json.loads(json.dumps(body, default=_jsonable)), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _redact(value):
    """Artifacts never carry private paths: any string with a path separator is reduced to its basename."""
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, str) and "/" in value:
        return Path(value).name
    return value


def _load_by_path(name: str, path, sha256: str | None):
    path = Path(path)
    if sha256 and tp.sha256_file(path) != sha256:
        raise DriverError(f"MODULE_DIGEST: {path.name} differs from its pin")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_jsonable).encode()).hexdigest()


# ----------------------------------------------------------------------------- candidate pool
def inventory_candidates(bars: pd.DataFrame, spec: dict):
    """The admitted-core candidate population computed by its owners' code loaded by path: financial-data
    ``core_selection_bundle.candidate_frame`` and feature-eng ``normalize_phase1b.add_derived_features``."""
    bundle = _load_by_path("core_selection_bundle", spec["bundle_script"], spec.get("bundle_sha256"))
    gen = _load_by_path("core_selection_technical_generator", spec["generator_module"], spec.get("generator_sha256"))

    shift = pd.Timedelta(hours=1) if spec.get("datetime_convention") == "availability_instant" else pd.Timedelta(0)

    def fn(b):
        frame = b.reset_index()
        frame["DATE_TIME"] = pd.DatetimeIndex(frame["DATE_TIME"]) + shift      # the bundle's own convention when declared
        x = bundle.candidate_frame(frame, gen.add_derived_features, gen.FEATURE_COLS)
        cols = [c for c in x.columns if c != "DATE_TIME"]
        out = x[cols].copy()
        out.index = b.index
        return out

    x = fn(bars)
    meta = []
    for c in x.columns:
        known = c in ("hod_sin", "hod_cos", "dow_sin", "dow_cos")
        meta.append({"feature_id": c, "family": "calendar_known" if known else ("raw_price" if c in RAW_LEVEL_COLUMNS or c.startswith("RAW_") else "technical"),
                     "source": "inventory: financial-data core_selection_bundle.candidate_frame + feature-eng normalize_phase1b",
                     "transform": "trailing indicator (pandas_ta / rolling / ewm) of bars <= t" if not known else
                     ("UTC calendar encoding of the bar's availability instant (bar start + 1 h)" if spec.get("datetime_convention") == "availability_instant"
                      else "bar label encoding (lake clock)"),
                     "unit": "price" if c in RAW_LEVEL_COLUMNS else "dimensionless", "clock": "KNOWN_IN_ADVANCE" if known else "OBSERVED",
                     "availability_time": "known in advance" if known else "timestamp", "support_h": 0 if known else 1,
                     "train_coverage": float(x[c].notna().mean()), "source_bytes": int(x[c].notna().sum()) * 8})
    return x, meta, fn


def fractal_candidates(bars: pd.DataFrame, spec: dict, lake_sha256: str):
    """The causal fractal columns.  ``compute_with_module`` + ``windows``: recomputed on THIS source's closes with
    the owner's plugin (feature-eng ``app/plugins/causal_fractal.py``), columns ``<feature>_w<window>``.
    Otherwise from the fractal lane's store, keyed by (asset, timestamp, bar convention, source): the store
    MANIFEST must name the same lake bytes and asset, and the store's own close must equal the lake close on every
    joined row (supporting evidence, never the key)."""
    note = {}
    if spec.get("compute_with_module"):
        mod = _load_by_path("causal_fractal_plugin", spec["compute_with_module"], spec.get("module_sha256"))
        close = bars["CLOSE"].to_numpy(float)
        frames = []
        for w in spec.get("windows", [200]):
            f = mod.causal_fractal_frame(close, window=int(w))[list(FRACTAL_COLUMNS)]
            f.columns = [f"{c}_w{int(w)}" for c in f.columns]
            frames.append(f)
        x = pd.concat(frames, axis=1)
        x.index = bars.index
        note = {"source": "computed_with_owner_plugin", "module_sha256": tp.sha256_file(spec["compute_with_module"]),
                "module_commit": spec.get("module_commit"), "windows_bars": [int(w) for w in spec.get("windows", [200])],
                "input": "closes of the configured bar source (observed bars; windows in observed bars)"}
        meta = [{"feature_id": c, "family": "fractal", "source": "feature-eng causal_fractal plugin (recomputed on this source)",
                 "transform": "trailing-window fractal estimator of closes <= t (window in observed bars)", "unit": "dimensionless",
                 "clock": "OBSERVED", "availability_time": "timestamp", "support_h": 1, "train_coverage": float(x[c].notna().mean()),
                 "source_bytes": int(x[c].notna().sum()) * 8} for c in x.columns]
        return x, meta, note
    if spec.get("manifest"):
        if tp.sha256_file(spec["manifest"]) != spec.get("manifest_sha256"):
            raise DriverError("FRACTAL_MANIFEST_DIGEST")
        fm = json.loads(Path(spec["manifest"]).read_text())
        if fm.get("source_sha256") != lake_sha256 or fm.get("instrument") != "EURUSD" or fm.get("bar") != "1h":
            raise DriverError("FRACTAL_SOURCE_IDENTITY: the store is not computed from the pinned lake EURUSD H1 bytes")
        note["store_identity"] = {"instrument": fm.get("instrument"), "bar": fm.get("bar"), "source_sha256": fm.get("source_sha256"),
                                  "window_is_observed_bars": fm.get("window_is_observed_bars_not_calendar_time")}
    parts = []
    for f in spec["files"]:
        raw = pd.read_parquet(Path(f))
        tcol = next(c for c in ("date_time", "timestamp", "datetime", "DATE_TIME") if c in raw.columns)
        idx = pd.DatetimeIndex(pd.to_datetime(raw[tcol]))
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        raw = raw.set_index(idx)
        if len(raw) and raw.index.max() >= tp.TEST_START:
            raise DriverError("TEST_ROWS_FORBIDDEN: fractal file reaches TEST rows")
        if raw.index.duplicated().any():
            raise DriverError("FRACTAL_DUPLICATE_TIMESTAMPS")
        parts.append(raw)
    joined = pd.concat(parts, axis=0)
    close = joined["close"].groupby(level=0).first() if "close" in joined else None
    cols = [c for c in joined.columns if any(c.startswith(b + "_w") for b in FRACTAL_COLUMNS)]
    x = joined[cols].groupby(level=0).first().reindex(bars.index)
    if close is not None:
        cl = close.reindex(bars.index)
        both = cl.notna()
        mism = int((np.abs(cl[both] - bars.CLOSE[both]) > 1e-9).sum())
        note["close_agreement"] = {"joined_rows": int(both.sum()), "close_mismatches": mism}
        if mism:
            raise DriverError(f"FRACTAL_CLOSE_DISAGREES: {mism} joined rows differ from the lake close")
    note.update(source="files", files_sha256={Path(f).name: tp.sha256_file(f) for f in spec["files"]},
                manifest_sha256=spec.get("manifest_sha256"), columns=list(x.columns),
                key=["EURUSD", "timestamp", BAR_CONVENTION, f"lake:{lake_sha256}"])
    meta = [{"feature_id": c, "family": "fractal", "source": "fractal store (feature-eng causal_fractal plugin, ported from agent-multi)",
             "transform": "trailing-window fractal estimator of closes <= t (window in observed bars)", "unit": "dimensionless",
             "clock": "OBSERVED", "availability_time": "timestamp", "support_h": 1, "train_coverage": float(x[c].notna().mean()),
             "source_bytes": int(x[c].notna().sum()) * 8} for c in x.columns]
    return x, meta, note


def assemble_pool(bars: pd.DataFrame, cfg: dict, closures=None):
    cand = cfg["candidates"]
    frames, meta, notes, gens = [], [], {}, {}
    if cand.get("inventory"):
        x, m, fn = inventory_candidates(bars, cand["inventory"])
        frames.append(x)
        meta += m
        notes["inventory"] = {"columns": list(x.columns)}
        gens["inventory"] = fn
    if cand.get("price_pool", True):
        clock = cand.get("price_pool_clock", "label")
        cls = closures if cand.get("price_pool_closure_features") else None
        x = pool.price_pool(bars, clock=clock, closures=cls)
        frames.append(x)
        meta += pool.pool_metadata(x)
        gens["price_pool"] = (lambda b, _c=clock, _cl=cls: pool.price_pool(b, clock=_c, closures=_cl))
    if cand.get("fractal"):
        x, m, n = fractal_candidates(bars, cand["fractal"], source_sha256(cfg))
        frames.append(x)
        meta += m
        notes["fractal"] = n
    if not frames:
        raise DriverError("EMPTY_POOL: no candidate source declared")
    X = pd.concat(frames, axis=1)
    if X.columns.duplicated().any():
        raise DriverError(f"DUPLICATE_FEATURE_IDS: {sorted(set(X.columns[X.columns.duplicated()]))}")
    return X, meta, notes, gens


def inventory_record(meta: list[dict], X: pd.DataFrame, notes: dict, alias_groups: dict | None = None) -> dict:
    fam = {}
    for m in meta:
        src = "inventory" if m["source"].startswith("inventory") else ("fractal" if m["family"] == "fractal" else "price_pool")
        fam.setdefault(src, []).append(m["feature_id"])
    cols = list(X.columns)
    return {"ordered_columns": cols, "count": len(cols), "counts_by_source": {k: len(v) for k, v in fam.items()},
            "count_reconciliation": " + ".join(f"{len(v)} {k}" for k, v in fam.items()) + f" = {len(cols)}",
            "columns_sha256": _digest(cols), "by_source": fam,
            "exact_alias_groups_phase2": alias_groups or {},
            "units": {c: pool.window_unit(c) if not c.startswith(tuple(f + "_w" for f in FRACTAL_COLUMNS)) else "observed bars" for c in cols},
            "clock": BAR_CONVENTION, "clock_note": "calendar encodings (cal.*, hod_*, dow_*) encode the lake label clock; its UTC "
                                                   "mapping is under repair and no hand shift is applied", "notes": _redact(notes)}


def source_sha256(cfg: dict) -> str:
    return cfg["source"]["sha256"] if cfg.get("source") else cfg["lake"]["sha256"]


def load_bars(cfg: dict, start=None, end=None):
    """(bars, closures, source identity).  ``source.kind == "admitted"``: admitted UTC bar starts through the
    clock lane's ``bar_source`` (pinned), closures from the admitted CLOSURE_INTERVALS.json.  Else the legacy lake."""
    src = cfg.get("source")
    if src and src.get("kind") == "admitted":
        a, b = start or src["load"][0], end or src["load"][1]
        bars = tp.load_admitted_bars(src, a, b)
        closures = tp.load_closures(src)
        ident = {"kind": "admitted", "sha256": src["sha256"], "admission_sha256": src["admission"]["sha256"],
                 "closures_sha256": src["closures"]["sha256"], "bar_convention": "UTC_BAR_START",
                 "availability": "UTC_BAR_START_PLUS_ONE_HOUR", "bar_source_sha256": src["bar_source_sha256"],
                 "file": Path(src["path"]).name, "closures_file": Path(src["closures"]["path"]).name}
        return bars, closures, ident
    lake = cfg["lake"]
    if tp.sha256_file(lake["path"]) != lake["sha256"]:
        raise DriverError("LAKE_DIGEST: lake file differs from its pin")
    a, b = start or lake["load"][0], end or lake["load"][1]
    return tp.load_lake_bars(lake["path"], a, b), None, {"kind": "lake_label", "sha256": lake["sha256"], "bar_convention": "LAKE_LABEL"}


TARGET_LEDGER = {"timing_pivot_k{k}": "t1", "timing_valley_k{k}": "t1", "timing_peak_k{k}": "t1",
                 "timing_action_confirmed": "t2", "timing_expert_leg_remaining_pips": "t3"}


# ----------------------------------------------------------------------------- screens and reports
def _bh(p, q):
    p = np.asarray(p, float)
    out = np.zeros(len(p), bool)
    idx = np.flatnonzero(np.isfinite(p))
    if not len(idx):
        return out
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(idx) / (np.arange(len(idx)) + 1)
    below = np.flatnonzero(ranked <= q)
    if len(below):
        out[order[: below.max() + 1]] = True
    return out


def univariate_screen(X: pd.DataFrame, y, *, head: str, q: float = 0.1, min_support: int = 100) -> dict:
    """Per feature: Kruskal-Wallis H across classes (classification) or Spearman (regression), BH at ``q``."""
    from scipy import stats
    y = np.asarray(y, float)
    rep, pvals, names = {}, [], []
    for col in X.columns:
        x = X[col].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        rec = {"n": int(m.sum()), "p": np.nan, "statistic": np.nan, "pass_fdr": False}
        if m.sum() >= min_support and np.nanstd(x[m]) > 0:
            if head == "classification":
                groups = [x[m][y[m] == c] for c in np.unique(y[m])]
                if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                    s, p = stats.kruskal(*groups)
                    rec.update(statistic=float(s), p=float(p), test="kruskal_wallis")
            else:
                r, p = stats.spearmanr(x[m], y[m])
                rec.update(statistic=float(r), p=float(p), test="spearman")
        rep[col] = rec
        pvals.append(rec["p"])
        names.append(col)
    for name, ok in zip(names, _bh(pvals, q)):
        rep[name]["pass_fdr"] = bool(ok)
        rep[name]["fdr_q"] = q
    return rep


def blocked_permutation_screen(X: pd.DataFrame, y, *, q: float = 0.1, n_perm: int = 4999, min_shift: int = 168,
                               seed: int = 0, min_support: int = 100) -> dict:
    """Dependence-aware univariate screen (plan audit v3, V3-05).  Rows are the eligible TRAIN origins in time
    order.  Statistic: Kruskal-Wallis H across the label classes (rank-based; the tie correction depends only on
    the feature, so it is constant under the null and omitted).  Null: the label sequence circularly shifted
    against the feature by a uniform offset in [min_shift, n - min_shift] rows, which keeps the serial
    dependence of BOTH series and breaks only their alignment.  p = (1 + #{H_null >= H_obs}) / (n_perm + 1);
    resolution 1 / (n_perm + 1).  BH at ``q`` over the features with a defined statistic.  Effect size
    epsilon^2 = H / (n - 1)."""
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    rep, pvals, names = {}, [], []
    for col in X.columns:
        x = X[col].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        rec = {"n": int(m.sum()), "p": np.nan, "H": np.nan, "epsilon2": np.nan, "pass_fdr": False,
               "test": "kruskal_H_blocked_circular_shift_null", "n_perm": int(n_perm), "min_shift_rows": int(min_shift),
               "p_resolution": 1.0 / (n_perm + 1)}
        nn = int(m.sum())
        if nn >= max(min_support, 4 * min_shift) and np.nanstd(x[m]) > 0:
            from scipy.stats import rankdata
            r = rankdata(x[m])
            g = y[m].astype(np.int64)
            k = int(g.max()) + 1
            cnt = np.bincount(g, minlength=k).astype(float)
            if (cnt > 0).sum() >= 2:
                def H(gg):
                    rs = np.bincount(gg, weights=r, minlength=k)
                    c = np.bincount(gg, minlength=k).astype(float)
                    ok = c > 0
                    return 12.0 / (nn * (nn + 1)) * np.sum(rs[ok] ** 2 / c[ok]) - 3.0 * (nn + 1)
                h0 = H(g)
                shifts = rng.integers(min_shift, nn - min_shift + 1, size=n_perm)
                exceed = sum(1 for sh in shifts if H(np.roll(g, int(sh))) >= h0)
                rec.update(H=float(h0), epsilon2=float(h0 / (nn - 1)), p=float((1 + exceed) / (n_perm + 1)))
        rep[col] = rec
        pvals.append(rec["p"])
        names.append(col)
    for name, ok in zip(names, _bh(pvals, q)):
        rep[name]["pass_fdr"] = bool(ok)
        rep[name]["fdr_q"] = q
    return rep


def fold_stability(X: pd.DataFrame, y, folds: list[dict], *, head: str, q: float) -> dict:
    """The univariate screen on every prefix fold; rank agreement (Spearman of -log p) between folds and with TRAIN."""
    from scipy import stats
    per_fold = {}
    for f in folds:
        a, b = f["train_rows"]
        per_fold[f["fold_id"]] = univariate_screen(X.iloc[a:b], np.asarray(y)[a:b], head=head, q=q)
    full = univariate_screen(X, y, head=head, q=q)
    names = list(X.columns)

    def score(rep):
        return np.array([-np.log10(max(rep[n]["p"], 1e-300)) if np.isfinite(rep[n]["p"]) else 0.0 for n in names])

    out = {"folds": {}, "pass_agreement": {}}
    s_full = score(full)
    for fid, rep in per_fold.items():
        s = score(rep)
        rho = float(stats.spearmanr(s, s_full).statistic) if np.std(s) > 0 and np.std(s_full) > 0 else np.nan
        out["folds"][fid] = {"rank_spearman_vs_train": rho, "passed": [n for n in names if rep[n]["pass_fdr"]]}
    for n in names:
        out["pass_agreement"][n] = int(sum(1 for rep in per_fold.values() if rep[n]["pass_fdr"]) + (1 if full[n]["pass_fdr"] else 0))
    return out


def vif(X: pd.DataFrame) -> dict:
    x = X.to_numpy(float)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < 10 or x.shape[1] < 2:
        return {c: np.nan for c in X.columns}
    xz = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)
    out = {}
    for j, c in enumerate(X.columns):
        a = np.column_stack([np.ones(len(xz)), np.delete(xz, j, axis=1)])
        beta, *_ = np.linalg.lstsq(a, xz[:, j], rcond=None)
        r2 = 1.0 - (xz[:, j] - a @ beta).var() / max(xz[:, j].var(), 1e-12)
        out[c] = float(1.0 / max(1.0 - r2, 1e-12))
    return out


def redundancy_report(X: pd.DataFrame, clusters: list[dict], selected: list[str]) -> dict:
    corr = X.corr(method="spearman")
    fract = [c for c in X.columns if any(c == b or c.startswith(b + "_w") for b in FRACTAL_COLUMNS)]
    rv = [c for c in X.columns if c.startswith("px.rv_") or c in ("rolling_std_24", "ATR", "px.atr_14")]
    fr_vs_rv = {f: {r: float(corr.loc[f, r]) for r in rv} for f in fract}
    sel_corr = {a: {b: float(corr.loc[a, b]) for b in selected} for a in selected}
    return {"clusters": [{"cluster_id": c["cluster_id"], "members": c["members"]} for c in clusters],
            "vif_selected": vif(X[selected]) if selected else {}, "fractal_vs_realized_vol_spearman": fr_vs_rv,
            "selected_pairwise_spearman": sel_corr}


# ----------------------------------------------------------------------------- causal stage
def causal_stage(X: pd.DataFrame, T: pd.DataFrame, bars: pd.DataFrame, cfg: dict, k: int, features: list[str], *,
                 mixed: bool) -> dict:
    """Owned by causal-inference (``fs_timing_pack``, pack EURUSD_TIMING).  All results are labelled
    CONDITIONAL_PREDICTIVE_ASSOCIATION; NOT_IDENTIFIED / NO_LINK never rejects a feature.
      mixed CI  PCMCI+ with tigramite RegressionCI on [x, T1 (one categorical), H], controls, temporal halves
      HAC       lagged linear-probability screen on the 0/1 indicators (association comparator)
      matched   q80 crossings vs hour/weekday/pre-vol matched controls (association comparator)"""
    src = Path(cfg["provider_src"])
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from causal_inference_provider import fs_timing_pack as tpk  # noqa: E402
    hist = [h for h in cfg.get("history", ["px.logret_24", "px.rv_24"]) if h in X.columns]
    cols = [f for f in features if f in X.columns and f not in hist] + hist
    Xs = X[cols]
    primary = f"timing_pivot_k{k}"
    out = {"pack": tpk.PACK_ID, "provider_sha256": tp.sha256_file(Path(tpk.__file__)), "history": hist,
           "evidence_label": tpk.EVIDENCE_LABEL, "lag0_mapping": tpk.LAG0_MAPPING, "interpreter": sys.version.split()[0],
           "hac_lag_screen": {}, "matched_control_contrast": {}, "mixed_ci": None}
    for t in (f"timing_valley_k{k}", f"timing_peak_k{k}"):
        out["hac_lag_screen"][t] = tpk.hac_lag_screen(Xs, T[t].to_numpy(float), k=k, history=hist, q=float(cfg.get("q", 0.1)))
    bars_ei = tpk.bars_table(bars.index, bars["CLOSE"].to_numpy(float), bars["HIGH"].to_numpy(float), bars["LOW"].to_numpy(float))
    piv = T[primary].to_numpy(float)
    piv_any = np.where(np.isfinite(piv), (piv > 0).astype(float), np.nan)
    for f in [c for c in cols if c not in hist]:
        out["matched_control_contrast"][f] = tpk.matched_control_contrast(
            bars_ei, X[f].to_numpy(float), piv_any, k=k, event_quantile=float(cfg.get("event_quantile", 0.8)),
            n_controls=int(cfg.get("n_controls", 5)), min_events=int(cfg.get("min_events", 20)),
            n_boot=int(cfg.get("n_boot", 400)), seed=int(cfg.get("seed", 0)))
    if mixed:
        try:
            import tigramite  # noqa: F401
            import importlib.metadata as md
            half = len(Xs) // 2
            out["mixed_ci"] = tpk.mixed_ci_stage(Xs, piv, k=k, history=hist, target_name=primary,
                                                 reference=cfg.get("negative_control_reference", "px.logret_2"),
                                                 halves=(slice(0, half), slice(half, len(Xs))), pc_alpha=cfg.get("pc_alpha", 0.05),
                                                 alpha=cfg.get("alpha", 0.01), seed=int(cfg.get("seed", 0)),
                                                 workers=int(cfg.get("workers", 4)))
            out["mixed_ci"]["pin"] = {"library": "tigramite", "version": md.version("tigramite")}
        except ModuleNotFoundError as exc:
            out["mixed_ci"] = {"state": "NOT_RUN", "reason": f"{exc}", "next": "run the `causal` verb under an interpreter with tigramite"}
    else:
        out["mixed_ci"] = {"state": "PENDING_CAUSAL_VERB"}
    return out


# ----------------------------------------------------------------------------- selection recipe
def apply_recipe(candidates: dict, clusters: list[dict], rankings: dict, recipe: dict, screen_pass: set[str]) -> dict:
    """Union of the engine's ``method``/``k`` subsets over the recipe targets (the PRIMARY T1 task only: the
    3-class label and its two direction indicators), intersected with the FDR screen, pruned to one
    representative per TRAIN Spearman redundancy cluster (max engine relevance for the primary target)."""
    method, k = recipe.get("method", "MRMR"), int(recipe.get("k", 12))
    tids = recipe["targets"]
    union, per_target = set(), {}
    for c in candidates["candidates"]:
        if c["method"] == method and c["k"] == k and c["target_id"] in tids:
            per_target[c["target_id"]] = list(c["members"])
            union |= set(c["members"])
    kept = union & screen_pass if recipe.get("require_fdr_pass", True) else set(union)
    cluster_of = {m: c["cluster_id"] for c in clusters for m in c["members"]}
    by_cluster: dict = {}
    for f in kept:
        by_cluster.setdefault(cluster_of.get(f, f"single:{f}"), []).append(f)
    selected = sorted(max(ms, key=lambda f: (rankings.get(f, 0.0), f)) for ms in by_cluster.values())
    return {"method": method, "k": k, "targets": tids, "per_target": per_target, "union": sorted(union),
            "dropped_by_fdr_screen": sorted(union - kept), "pruned_redundant": sorted(kept - set(selected)), "selected": selected}


def comparator_subsets(candidates: dict, method: str, k: int, tids: list[str]) -> dict:
    return {c["target_id"]: list(c["members"]) for c in candidates["candidates"]
            if c["method"] == method and c["k"] == k and c["target_id"] in tids}


# ----------------------------------------------------------------------------- DAG (F08)
DAG_NODES = {
    "pool_features": {"inputs": ["lake", "inventory_generator", "inventory_bundle", "fractal_store", "pool_code", "candidates_config"], "target_dependent": False},
    "quality_profiles": {"inputs": ["pool_features"], "target_dependent": False},
    "perturbation_checks": {"inputs": ["pool_features"], "target_dependent": False},
    "within_day_variance": {"inputs": ["pool_features", "splits"], "target_dependent": False},
    "targets_ledger": {"inputs": ["lake", "zigzag_module", "expert_params", "k_bars", "target_semantics", "splits"], "target_dependent": True},
    "finality_check": {"inputs": ["targets_ledger"], "target_dependent": True},
    "population_manifest": {"inputs": ["pool_features", "targets_ledger", "splits"], "target_dependent": True},
    "phase2_pairwise": {"inputs": ["population_manifest", "pairwise_params"], "target_dependent": True,
                        "note": "pair metrics are target-independent in CONTENT, but the pinned engine binds its run identity to the "
                                "population manifest (which includes targets) and refuses foreign identities, so they are recomputed"},
    "phase3_filters": {"inputs": ["phase2_pairwise", "filter_params"], "target_dependent": True},
    "univariate_screen": {"inputs": ["pool_features", "targets_ledger"], "target_dependent": True},
    "fold_stability": {"inputs": ["univariate_screen"], "target_dependent": True},
    "recipe": {"inputs": ["phase3_filters", "univariate_screen", "phase2_pairwise", "recipe_config"], "target_dependent": True},
    "causal": {"inputs": ["pool_features", "targets_ledger", "causal_config", "recipe"], "target_dependent": True},
    "eval_inputs": {"inputs": ["pool_features", "targets_ledger", "recipe", "comparison_sets"], "target_dependent": True},
    "evaluator": {"inputs": ["eval_inputs", "adapter_revision"], "target_dependent": True},
    "selection_manifest": {"inputs": ["recipe", "evaluator"], "target_dependent": True},
}


def dag_identities(leaves: dict) -> dict:
    """Identity of every node = digest of its inputs' identities (leaves are content digests)."""
    ids = dict(leaves)
    pending = dict(DAG_NODES)
    while pending:
        progressed = False
        for name, node in list(pending.items()):
            if all(i in ids for i in node["inputs"]):
                ids[name] = _digest({i: ids[i] for i in node["inputs"]})
                pending.pop(name)
                progressed = True
        if not progressed:
            raise DriverError(f"DAG_UNRESOLVED: {sorted(pending)}")
    return {n: ids[n] for n in DAG_NODES}


def plan_invalidation(prev: dict | None, new: dict) -> dict:
    """REUSE a node only when its identity is unchanged (identical inputs, masks, generation identities)."""
    out = {}
    for n in DAG_NODES:
        same = bool(prev) and prev.get(n) == new[n]
        out[n] = "REUSE" if same and n in ("pool_features", "quality_profiles", "perturbation_checks", "within_day_variance") else (
            "REUSE_ELIGIBLE" if same else "RECOMPUTE")
    return out


# ----------------------------------------------------------------------------- read ledger (F02)
def read_ledger(cfg: dict) -> dict:
    return {"schema": "timing_selection.read_ledger.v1",
            "prior_reads": [
                {"actor": "ZigZag expert selection (heuristic-strategy docs/short_horizon/results/zigzag/CHOICE.json)",
                 "periods": {"2020-2022": "frequency-rule acceptance of the NSGA-II front", "2023": "chose the expert by best 2023 Sharpe",
                             "2024": "reported confirm_2024 metrics"},
                 "consequence": "2023 is JOINT development for every downstream use of this expert; 2024 is PREVIOUSLY OBSERVED"},
                {"actor": "pivot detector (heuristic-strategy satoshi/pivot-detector-20261019 @cead861c)",
                 "periods": {"2023": "temperature and operating threshold", "2024": "reported once"}, "consequence": "2024 observed again"},
                {"actor": "this lane, provisional v1 (targets timing_targets.v1, 2026-10-19)",
                 "periods": {"2020-2022": "phase-2/3 selection, screens", "2023-2024": "evaluator loaded rows and began fits; stopped before any "
                             "metric was computed or written"}, "consequence": "no 2023/2024 metric of v1 exists or influenced v2"}],
            "this_run": {"2020-2022 TRAIN": "every selection stage (engines, screens, recipe, causal)",
                         "2023H1 INNER_A": "evaluator: early stopping, temperature, regularization cell, threshold",
                         "2023H2 INNER_B": "evaluator: frozen scoring only",
                         "2024": "evaluator `reconfirm` after FROZEN.json; previously observed; no decision may change on it",
                         "2025+": "never read"},
            "prior_runs_declared_in_config": cfg.get("read_ledger_extra", []),
            "freeze_rule": "the candidate subset is frozen in SELECTION_MANIFEST before the evaluator's first read of 2023; "
                           "the evaluator's choices are frozen in FROZEN.json before its read of 2024"}


# ----------------------------------------------------------------------------- driver
def _cuts(sp: dict) -> list[str]:
    return sorted({str(_utc(c)) for c in [sp["train"][1], sp.get("inner_a_end", INNER_A_END), sp["inner"][1], sp["confirm"][1],
                                          *sp.get("folds", [])]})


def run(config_path, out, reuse_from=None) -> dict:
    t0 = time.time()
    cfg = json.loads(Path(config_path).read_text())
    if cfg.get("schema") != SCHEMA:
        raise DriverError("CONFIG_SCHEMA")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config_sha = man.digest(cfg)
    prepared = out / "PREPARED.json"
    if prepared.exists() and json.loads(prepared.read_text())["config_sha256"] != config_sha:
        raise DriverError("CONFIG_CHANGED: use another output directory")
    bars, closures, src_ident = load_bars(cfg)
    lake = {"sha256": src_ident["sha256"]}
    zz = tp.load_zigzag_module(cfg["zigzag"]["module"], cfg["zigzag"]["sha256"])
    zp = cfg["zigzag"]["params"]
    params = zz.Params(depth=int(zp["depth"]), deviation_pips=float(zp["deviation_pips"]), backstep=int(zp["backstep"]))
    k = int(cfg["zigzag"]["k_bars"])
    conf_def = cfg["zigzag"].get("confirmation", "confirmation_bars")
    sp = cfg["splits"]
    cand = cfg["candidates"]
    leaves = {"lake": _digest({k_: v for k_, v in src_ident.items() if k_ not in ("file", "closures_file")}), "zigzag_module": cfg["zigzag"]["sha256"], "expert_params": _digest(zp), "k_bars": str(k),
              "target_semantics": tp.SEMANTICS if conf_def == "confirmation_bars" else f"{tp.SEMANTICS}+{conf_def}", "splits": _digest({"inner_a_end": INNER_A_END, **sp}),
              "inventory_generator": (cand.get("inventory") or {}).get("generator_sha256", "none"),
              "inventory_bundle": (cand.get("inventory") or {}).get("bundle_sha256", "none"),
              "fractal_store": (cand.get("fractal") or {}).get("manifest_sha256", "none"),
              "pool_code": tp.sha256_file(Path(pool.__file__)), "candidates_config": _digest(_redact(cand)),
              "pairwise_params": _digest(cfg.get("pairwise_params", {})), "filter_params": _digest(cfg["filter"]),
              "recipe_config": _digest(cfg["filter"].get("recipe", {})), "causal_config": _digest(_redact(cfg.get("causal", {}))),
              "comparison_sets": _digest(cfg.get("comparison_sets", {})), "adapter_revision": cfg.get("adapter_revision", "unpinned")}
    dag = dag_identities(leaves)
    prev = json.loads((Path(reuse_from) / "DAG.json").read_text())["identities"] if reuse_from and (Path(reuse_from) / "DAG.json").exists() else None
    plan = plan_invalidation(prev, dag)
    # ---- targets with the support ledger (target-dependent)
    cuts = _cuts(sp)
    arrays = tp.label_arrays(bars.HIGH.to_numpy(float), bars.LOW.to_numpy(float), bars.CLOSE.to_numpy(float), zz, params, k_bars=k,
                             confirmation=conf_def)
    T, ledger = tp.build_targets(bars, zz, params, k_bars=k, cuts=cuts, arrays=arrays, with_ledger=True, confirmation=conf_def)
    T.index = bars.index
    ia_end = sp.get("inner_a_end", INNER_A_END)
    periods = {"TRAIN_2020_2022": sp["train"], "INNER_A_2023H1": [sp["inner"][0], ia_end],
               "INNER_B_2023H2": [ia_end, sp["inner"][1]], "REOBSERVED_2024": sp["confirm"]}
    status = tp.status_table(ledger, periods)
    cut_pos = tp._cut_positions(bars.index, cuts)
    if cfg.get("finality_all_targets_all_cuts", conf_def != "confirmation_bars"):
        fin_all = tp.finality_check(bars, zz, params, k_bars=k, cut_positions=cut_pos, full=arrays, check_t2=True, confirmation=conf_def)
        finality = {"t1_t2_t3_all_cuts": fin_all, "cuts_checked": len(cut_pos), "targets_checked": ["t1", "t2", "t3"],
                    "all_final_rows_unchanged": bool(fin_all["all_final_rows_unchanged"])}
    else:   # legacy (slow v1 confirmation replay): T2 only at three cuts; the block names say exactly what was checked
        fin_cheap = tp.finality_check(bars, zz, params, k_bars=k, cut_positions=cut_pos, full=arrays, check_t2=False, confirmation=conf_def)
        t2_cuts = [tp._cut_positions(bars.index, [c])[0] for c in (sp["train"][1], sp["inner"][1], (sp.get("folds") or [sp["train"][1]])[0])]
        fin_t2 = tp.finality_check(bars, zz, params, k_bars=k, cut_positions=sorted(set(t2_cuts)), full=arrays, check_t2=True,
                                   confirmation=conf_def)
        finality = {"t1_only_all_cuts_t2_t3_skipped": fin_cheap, "t1_t2_t3_three_cuts": fin_t2,
                    "all_final_rows_unchanged": bool(fin_cheap["all_final_rows_unchanged"] and fin_t2["all_final_rows_unchanged"])}
    # ---- the ONE shared eligibility mask (business calendar + target-specific label support); applied to every
    # target BEFORE any stage reads it, and handed to the evaluator with its digest
    elig = None
    if closures is not None:
        from tools import timing_eligibility as EL
        bc = cfg.get("business_calendar", {})
        elig = EL.eligibility(bars.index, closures, k_bars=k, reopen_hours=float(bc.get("reopen_hours", 12)),
                              preclose_hours=float(bc.get("preclose_hours", 12)),
                              label_status={"t1": ledger["t1_status"].to_numpy(), "t2": ledger["t2_status"].to_numpy(),
                                            "t3": ledger["t3_status"].to_numpy()})
        for col in list(T.columns):
            key = next((v for tk, v in TARGET_LEDGER.items() if tk.format(k=k) == col), None)
            if key:
                T.loc[~elig[f"{key}_eligible"].to_numpy(), col] = np.nan
        elig_counts = {t_: EL.counts(elig, periods, f"{t_}_reason") for t_ in ("t1", "t2", "t3")}
        elig_counts["business"] = EL.counts(elig, periods, "business_reason")
        elig_digest = EL.digest(elig)
    # ---- pool (target-independent; reusable)
    reused_pool = False
    if plan["pool_features"] == "REUSE":
        src = Path(reuse_from) / "data" / "pool_all.parquet"
        X = pd.read_parquet(src)
        X.index = bars.index
        meta = json.loads((Path(reuse_from) / "POOL_META.json").read_text())
        notes, gens = meta["notes"], {}
        meta = meta["meta"]
        reused_pool = True
    else:
        X, meta, notes, gens = assemble_pool(bars, cfg, closures)
    admitted = pool.availability_gate(meta)
    X = X[admitted]
    meta = [m for m in meta if m["feature_id"] in admitted]
    data = out / "data"
    data.mkdir(exist_ok=True)
    if not reused_pool:
        X.reset_index(drop=True).to_parquet(data / "pool_all.parquet", index=False)
        _write_json(out / "POOL_META.json", {"meta": meta, "notes": _redact(notes)})
    else:
        shutil.copyfile(Path(reuse_from) / "data" / "pool_all.parquet", data / "pool_all.parquet")
        shutil.copyfile(Path(reuse_from) / "POOL_META.json", out / "POOL_META.json")
    ledger.to_parquet(out / "LABEL_LEDGER.parquet", index=False)
    if elig is not None:
        elig.to_parquet(out / "ELIGIBILITY.parquet", index=False)
    # ---- quality + perturbation (target-independent)
    tr0, tr1 = _utc(sp["train"][0]), _utc(sp["train"][1])
    train = (bars.index >= tr0) & (bars.index < tr1)
    if plan["quality_profiles"] == "REUSE":
        quality = json.loads((Path(reuse_from) / "QUALITY_PROFILES.json").read_text())
        perturb = json.loads((Path(reuse_from) / "PERTURBATION.json").read_text())
    else:
        quality = {"lake_grid": pool.quality_profile(X), "train_only": pool.quality_profile(X.loc[train])}
        quality["profile_series"] = provider_profiles(X.loc[train], cfg, [m["feature_id"] for m in meta if not m["source"].startswith("inventory")])
        perturb = perturbation_suite(bars, gens, cfg)
    _write_json(out / "QUALITY_PROFILES.json", quality)
    _write_json(out / "PERTURBATION.json", perturb)
    # ---- TRAIN split for the engines, prefix folds at the declared cuts
    Xt, Tt = X.loc[train].reset_index(drop=True), T.loc[train].reset_index(drop=True)
    n = len(Xt)
    ts_train = bars.index[train]
    folds = [{"fold_id": f"prefix_before_{str(_utc(c))[:10]}", "train_rows": [0, int(ts_train.searchsorted(_utc(c)))]} for c in sp.get("folds", [])]
    feat_cols = list(Xt.columns)
    fx = pd.concat([pd.DataFrame({"row_id": np.arange(n), "timestamp": ts_train}), Xt], axis=1)
    tt = pd.concat([pd.DataFrame({"row_id": np.arange(n), "timestamp": ts_train}), Tt.drop(columns=["row_id", "timestamp"])], axis=1)
    if not prepared.exists():
        fx.to_parquet(data / "features_train.parquet", index=False)
        tt.to_parquet(data / "targets_train.parquet", index=False)
    targets = tp.manifest_targets(k)
    for m in meta:
        m["train_coverage"] = float(Xt[m["feature_id"]].notna().mean())
    manifest = man.build_manifest(
        population_id=cfg["population_id"], identity=f"timing-selection:{config_sha[:16]}",
        identity_source={"kind": "TIMING_SELECTION_CONFIG", "config_sha256": config_sha, "target_identity": dag["targets_ledger"]},
        phase1={"state": "QUALITY_PROFILED_NEW_FAMILIES", "scope": "QUALITY_PROFILES.json; causal ladder rungs 2-3 not run"},
        data={"features_file": "features_train.parquet", "targets_file": "targets_train.parquet",
              "features_sha256": man.sha256_file(data / "features_train.parquet"),
              "targets_sha256": man.sha256_file(data / "targets_train.parquet"), "timestamp_column": "timestamp",
              "row_id_column": "row_id", "bar_hours": 1, "train_rows": n},
        features=[m for m in meta if m["feature_id"] in feat_cols], targets=[{k_: v for k_, v in t.items() if k_ != "role"} for t in targets],
        folds=folds, causal_supported=[], contract_source="timing_pivot targets (timing_targets.v2) from the sealed ZigZag expert (labels only)",
        data_root=data)
    man.write_manifest(manifest, out / "MANIFEST.json")
    targets_doc = tp.targets_document(T, params=params, k_bars=k, lake_sha256=lake["sha256"], zigzag_sha256=cfg["zigzag"]["sha256"],
                                      confirmation=conf_def)
    targets_doc.update(status_by_period=status, finality=finality, ledger_sha256=man.sha256_file(out / "LABEL_LEDGER.parquet"),
                       cuts=cuts, roles={t["target_id"]: t["role"] for t in targets}, source=src_ident)
    if elig is not None:
        targets_doc["eligibility"] = {"schema": "timing_selection.eligibility.v1", "digest": elig_digest,
                                      "rules": {"reopen_hours": float(cfg.get("business_calendar", {}).get("reopen_hours", 12)),
                                                "preclose_hours": float(cfg.get("business_calendar", {}).get("preclose_hours", 12)),
                                                "k_window_inside_one_session": True, "label_support": "ledger status LABELLED"},
                                      "counts_by_period_and_reason": elig_counts,
                                      "note": "applied to every target before selection, causal stage and evaluation; inputs keep all historical bars"}
    _write_json(out / "TARGETS.json", targets_doc)
    _write_json(prepared, {"config_sha256": config_sha, "config": _redact(cfg), "manifest_sha256": manifest["manifest_sha256"],
                           "pool_notes": _redact(notes), "admitted": admitted, "train_rows": n})
    # ---- phases 2 and 3: the existing engines, unchanged, local warehouse (duckdb, else the adapter's sqlite3)
    from tools import feature_pairwise_campaign as campaign
    try:
        import duckdb  # noqa: F401
        from tools.fs_phase23_warehouse import Warehouse
    except ModuleNotFoundError:
        from tools.fs_phase23_warehouse_adapter import WarehouseAdapter as Warehouse
    state = out / "campaign"
    plan_path = state / "PLAN.json"
    if not plan_path.exists():
        campaign.plan(manifest_path=out / "MANIFEST.json", state_root=state, n_shards=int(cfg.get("n_shards", 1)),
                      hosts=[{"host_id": "local", "size_class": "small"}], params=cfg.get("pairwise_params", {}))
    worker = campaign.run_worker(plan_path=plan_path, state_root=state, data_root=data, host_id="local")
    wh = Warehouse(out / "metrics.duckdb")
    try:
        closure = campaign.follow_once(plan_path=plan_path, state_root=state, terminal_dirs=[state / "terminals"],
                                       warehouse_path=out / "metrics.duckdb", data_root=data, warehouse=wh,
                                       phase3_workers=int(cfg.get("phase3_workers", 1)))
        if not closure["phase2_closed"] or not closure["phase3_closed"]:
            raise DriverError(f"ENGINES_INCOMPLETE: {closure}")
        readback = wh.reconcile(manifest["identity"])["tables"]
    finally:
        wh.close()
    candidates = json.loads((state / "CANDIDATES_FOR_VALIDATION.json").read_text())
    p2 = json.loads((state / "PHASE_2_COMPLETE.json").read_text())
    mats = np.load(state / "PHASE2_MATRICES.npz", allow_pickle=False)
    feats = [str(f) for f in mats["features"]]
    reps = [str(f) for f in mats["alias_representative"]]
    alias_groups = {}
    for f, r in zip(feats, reps):
        if f != r:
            alias_groups.setdefault(r, [r]).append(f)
    clusters = campaign.redundancy_clusters(feats, mats["spearman"], [bool(v) for v in mats["admissible"]], manifest,
                                            json.loads((state / "PLAN.json").read_text())["params"])
    inventory = inventory_record(meta, Xt, notes, alias_groups)
    _write_json(out / "INVENTORY.json", inventory)
    # ---- timing-specific stages (TRAIN only)
    primary = f"timing_pivot_k{k}"
    q = float(cfg.get("fdr_q", 0.1))
    screens = {t["target_id"]: univariate_screen(Xt, Tt[t["column"]].to_numpy(float), head=t["head"], q=q) for t in targets}
    sc = cfg.get("screen", {"method": "kruskal_iid"})
    if sc.get("method") == "blocked_circular_shift":
        screen = blocked_permutation_screen(Xt, Tt[primary].to_numpy(float), q=q, n_perm=int(sc.get("n_perm", 4999)),
                                            min_shift=int(sc.get("min_shift_rows", 168)), seed=int(sc.get("seed", 0)))
    else:
        screen = screens[primary]
    stability = fold_stability(Xt, Tt[primary].to_numpy(float), folds, head="classification", q=q) if folds else {}
    within_day = pool.within_day_variance(Xt.set_index(ts_train))
    rankings = {}
    for path in sorted((state / "phase3" / "terminals").glob("*.json.gz")):
        doc = campaign.load_terminal(path)
        if doc.get("target_id") == primary:
            for r in doc["rows"]["feature_filter_rankings"]:
                if r["method"] == "MRMR":
                    rankings[r["feature_id"]] = float((r.get("terms") or {}).get("relevance") or r.get("score") or 0.0)
    screen_pass = {f for f, r in screen.items() if r["pass_fdr"]}
    rec_cfg = dict(cfg["filter"].get("recipe", {}))
    rec_cfg.setdefault("targets", [primary, f"timing_valley_k{k}", f"timing_peak_k{k}"])
    recipe = apply_recipe(candidates, clusters, rankings, rec_cfg, screen_pass)
    selected = recipe["selected"]
    comparators = comparator_subsets(candidates, recipe["method"], recipe["k"], ["timing_action_confirmed", "timing_expert_leg_remaining_pips"])
    redundancy = redundancy_report(Xt, clusters, selected)
    causal = causal_stage(Xt, Tt, bars.loc[train], cfg["causal"], k, sorted(screen_pass | set(selected)), mixed=False) \
        if cfg.get("causal", {}).get("enabled") else {"state": "NOT_ENABLED"}
    # ---- the candidate subset is FROZEN here, before any 2023 read by the evaluator
    frozen_subset = {"selected": selected, "recipe": recipe, "frozen_at": pd.Timestamp.utcnow().isoformat(), "reads_so_far": ["TRAIN_2020_2022"]}
    frozen_subset["subset_sha256"] = _digest({"selected": selected, "recipe": recipe})
    _write_json(out / "FROZEN_SUBSET.json", frozen_subset)
    # ---- evaluator input: all loaded rows (TRAIN + INNER + 2024), features and labels; TEST never loaded
    ev = out / "eval_input"
    ev.mkdir(exist_ok=True)
    pd.concat([pd.DataFrame({"timestamp": bars.index}), X.reset_index(drop=True)], axis=1).to_parquet(ev / "features_all.parquet", index=False)
    T.reset_index(drop=True).to_parquet(ev / "targets_all.parquet", index=False)
    if elig is not None:
        elig.to_parquet(ev / "ELIGIBILITY.parquet", index=False)
    sets = {"timing_selected": selected}
    for name, cols in (cfg.get("comparison_sets") or {}).items():
        sets[name] = list(cols)
    missing = {n_: [c for c in cols if c not in X.columns] for n_, cols in sets.items()}
    if any(missing.values()):
        raise DriverError(f"COMPARISON_SET_COLUMNS_MISSING: {missing}")
    _write_json(ev / "FEATURE_SETS.json", {"schema": "timing_eval_input.v2" if elig is not None else "timing_eval_input.v1",
                                            "source": src_ident,
                                            "eligibility": ({"file": "ELIGIBILITY.parquet", "digest": elig_digest, "column": "t1_eligible",
                                                             "file_sha256": man.sha256_file(ev / "ELIGIBILITY.parquet")} if elig is not None else None),
                                            "sets": sets, "features_sha256": man.sha256_file(ev / "features_all.parquet"),
                                            "targets_sha256": man.sha256_file(ev / "targets_all.parquet"), "k_bars": k,
                                            "splits": sp, "inner_a_end": INNER_A_END, "subset_sha256": frozen_subset["subset_sha256"],
                                            "expert": {"params": zp, "module_sha256": cfg["zigzag"]["sha256"], "confirmation": conf_def,
                                                       "label": cfg["zigzag"].get("label")},
                                            "strategy": cfg.get("strategy", {}),
                                            "fractal_manifest_sha256": (cand.get("fractal") or {}).get("manifest_sha256")})
    _write_json(out / "READ_LEDGER.json", read_ledger(cfg))
    _write_json(out / "DAG.json", {"nodes": DAG_NODES, "leaves": leaves, "identities": dag, "plan": plan,
                                   "reused": [n_ for n_, v in plan.items() if v == "REUSE"]})
    report = {"schema": "predictor.timing_selection.report.v1", "config_sha256": config_sha, "manifest_sha256": manifest["manifest_sha256"],
              "identity": manifest["identity"], "train_rows": n, "features": len(feat_cols), "admitted": admitted,
              "inventory": {k_: inventory[k_] for k_ in ("count", "counts_by_source", "count_reconciliation", "columns_sha256")},
              "targets": targets_doc, "worker": _redact(worker), "phase2": {"closure_sha256": p2["closure_sha256"]},
              "phase3": {"methods": json.loads((state / "PHASE_3_FILTER_COMPLETE.json").read_text())["methods"],
                         "candidates": len(candidates["candidates"])}, "readback": readback, "recipe": recipe,
              "comparator_subsets_T2_T3": comparators, "univariate_screen": screens, "screen_gate": {"method": sc.get("method"), "results": screen},
              "fold_stability": stability,
              "within_day_variance": within_day, "redundancy": redundancy, "causal": causal, "dag_plan": plan,
              "wall_seconds": time.time() - t0, "test_read": False}
    _write_json(out / "REPORT.json", report)
    _write_json(out / "CAUSAL_STAGE.json", causal)
    _write_json(out / "SELECTION_MANIFEST.json", selection_manifest(cfg, manifest, targets, feat_cols, selected, recipe, within_day,
                                                                    dag, frozen_subset))
    return report


def provider_profiles(Xtrain: pd.DataFrame, cfg: dict, cols: list[str]) -> dict:
    """causal-inference ``feature_selection_profile.profile_series`` (the existing phase-1 numeric profiler) on
    every NEW-family column, TRAIN rows, observed-bar calendar."""
    src = (cfg.get("causal") or {}).get("provider_src")
    if not src:
        return {"state": "NOT_RUN", "reason": "no provider_src"}
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from causal_inference_provider.feature_selection_profile import METRICS_VERSION, profile_series
    except ModuleNotFoundError as exc:
        return {"state": "NOT_RUN", "reason": str(exc)}
    frame = Xtrain.reset_index().rename(columns={Xtrain.index.name or "index": "timestamp"})
    out = {"profiler": "causal_inference_provider.feature_selection_profile.profile_series", "metrics_version": METRICS_VERSION,
           "calendar": "observed_only", "frequency": "1h", "columns": {}}
    for c in cols:
        rec = profile_series(c, frame, "timestamp", c, "1h", "observed_only")
        out["columns"][c] = rec
    return out


def perturbation_suite(bars: pd.DataFrame, gens: dict, cfg: dict) -> dict:
    """Prefix invariance + future perturbation for every generator in the pool and for the evaluator's TRAIN-only
    scaler, on a bounded window (the fractal store is checked through its owner module when configured)."""
    out = {}
    sub = bars.iloc[: min(len(bars), 6000)]
    cut = len(sub) - min(1500, len(sub) // 4)
    for name, fn in gens.items():
        out[name] = pool.perturbation_check(fn, sub, cut, seed=1)
    fr = (cfg["candidates"].get("fractal") or {})
    mod_path = fr.get("owner_module") or fr.get("compute_with_module")
    mod_sha = fr.get("owner_module_sha256") or fr.get("module_sha256")
    if mod_path:
        mod = _load_by_path("causal_fractal_features", mod_path, mod_sha)
        small = bars.iloc[: 1800]

        def ffn(b):
            frames = []
            for w in (48, 96, 200):
                f = mod.causal_fractal_frame(b["CLOSE"].to_numpy(float), window=w)[list(FRACTAL_COLUMNS)]
                f.columns = [f"{c}_w{w}" for c in f.columns]
                frames.append(f)
            o = pd.concat(frames, axis=1)
            o.index = b.index
            return o

        out["fractal_owner_module"] = pool.perturbation_check(ffn, small, 1400, seed=2)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1000, 3))
    mu, sd = x[:600].mean(0), x[:600].std(0)
    x2 = x.copy()
    x2[600:] = rng.normal(5, 3, size=(400, 3))
    out["evaluator_train_only_scaler"] = {"passed": bool(np.allclose(mu, x2[:600].mean(0)) and np.allclose(sd, x2[:600].std(0))),
                                          "rule": "mean/sd fitted on TRAIN identical rows only; later rows cannot move it"}
    out["all_passed"] = all(v.get("passed") for v in out.values() if isinstance(v, dict))
    return out


def selection_manifest(cfg, manifest, targets, feat_cols, selected, recipe, within_day, dag, frozen_subset) -> dict:
    features = []
    for f in feat_cols:
        if f in selected:
            state, reasons = "selected", [f"{recipe['method']}_K{recipe['k']}_UNION_T1", "FDR_PASS_T1", "CLUSTER_REPRESENTATIVE"]
        else:
            reasons = []
            if f in recipe["pruned_redundant"]:
                reasons.append("REDUNDANT_WITH_SELECTED_REPRESENTATIVE")
            if f in recipe["dropped_by_fdr_screen"]:
                reasons.append("FDR_FAIL_T1")
            if f not in recipe["union"]:
                reasons.append(f"NOT_IN_{recipe['method']}_K{recipe['k']}_UNION_T1")
            state = "rejected"
        if within_day.get(f, {}).get("constant_within_day"):
            reasons.append("CONSTANT_WITHIN_DAY")
        features.append({"name": f, "state": state, "reason_codes": reasons})
    order = [f for f in selected]
    cand = cfg["candidates"]
    src = {("admitted_source" if cfg.get("source") else "lake"): source_sha256(cfg), "zigzag_module": cfg["zigzag"]["sha256"],
           "population_manifest": manifest["manifest_sha256"],
           "features_train": manifest["data"]["features_sha256"], "targets_train": manifest["data"]["targets_sha256"],
           "config": man.digest(cfg), "target_identity": dag["targets_ledger"], "subset": frozen_subset["subset_sha256"]}
    if cfg.get("source"):
        src.update(admission_receipt=cfg["source"]["admission"]["sha256"], closure_intervals=cfg["source"]["closures"]["sha256"],
                   bar_source_module=cfg["source"]["bar_source_sha256"])
    for key, val in (("fractal_plugin", (cand.get("fractal") or {}).get("module_sha256")),
                     ("fractal_store_manifest", (cand.get("fractal") or {}).get("manifest_sha256")),
                     ("inventory_generator", (cand.get("inventory") or {}).get("generator_sha256")),
                     ("inventory_bundle_script", (cand.get("inventory") or {}).get("bundle_sha256"))):
        if val:
            src[key] = val
    doc = {"schema": MANIFEST_SCHEMA, "manifest_version": 1, "selection_name": SELECTION_NAME, "dataset_id": cfg["population_id"],
           "status": SEAL_STATUS, "final_selection": False, "proxy_only": True,
           "scope": "usefulness for the pivot-classification proxy and its trivial reversal strategy only; NOT an RL-selected subset; "
                    "a fixed-budget SAC comparison against the admitted-core 15 is a later, separately authorized integration",
           "learning_paradigm": "expert-demonstration / hindsight-guided learning (not human-preference RLHF)",
           "producer": {"role": "producer", "lane": "SELECTION"},
           "independent_decision_record": {"decision_id": f"{SELECTION_NAME}:{manifest['identity']}:decision", "decider_role": "auditor",
                                           "state": "PENDING"},
           "targets": [{"name": t["target_id"], "horizons": [int(t["horizon_hours"])], "family": t["family"], "head": t["head"],
                        "role": t["role"], "horizon_unit": "observed market bars"} for t in targets],
           "target_params": {"zigzag": cfg["zigzag"]["params"], "k_bars": cfg["zigzag"]["k_bars"], "semantics": tp.SEMANTICS,
                             "confirmation": cfg["zigzag"].get("confirmation", "confirmation_bars"), "expert_label": cfg["zigzag"].get("label"),
                             "expert_kind": "HINDSIGHT_NON_CAUSAL_EXPERT"},
           "split": {**cfg["splits"], "inner_a_end": INNER_A_END,
                     "roles": {"2020-2022": "selection (TRAIN)", "2023": "JOINT development (expert chosen with it)",
                               "2024": "PREVIOUSLY OBSERVED reconfirmation", "2025+": "never read"}},
           "features": features, "selected_features": order,
           "source_digests": src,
           "rl_observation_contract": {"schema": "gym_fx.observation_contract.v1", "feature_columns": order,
                                       "note": "column order only; scaling and window follow the consuming RL contract"},
           "predictive_validation": None, "test_read": False}
    doc["manifest_sha256"] = gate_digest(doc, "manifest_sha256")
    return doc


def causal_only(config_path, out) -> dict:
    """Re-run the causal stage (including the mixed-data CI stage) from the saved TRAIN data and targets under an
    interpreter that has tigramite; rewrites CAUSAL_STAGE.json and REPORT.json['causal'] only."""
    cfg = json.loads(Path(config_path).read_text())
    out = Path(out).resolve()
    prep = json.loads((out / "PREPARED.json").read_text())
    if prep["config_sha256"] != man.digest(cfg):
        raise DriverError("CONFIG_CHANGED: the causal stage must use the run's config")
    k = int(cfg["zigzag"]["k_bars"])
    fx = pd.read_parquet(out / "data" / "features_train.parquet")
    tt = pd.read_parquet(out / "data" / "targets_train.parquet")
    bars_all, _, _ = load_bars(cfg)
    bars = bars_all.loc[(bars_all.index >= _utc(cfg["splits"]["train"][0])) & (bars_all.index < _utc(cfg["splits"]["train"][1]))]
    if len(bars) != len(fx):
        raise DriverError("TRAIN_GRID_MISMATCH")
    Xt, Tt = fx.drop(columns=["row_id", "timestamp"]), tt.drop(columns=["row_id", "timestamp"])
    report = json.loads((out / "REPORT.json").read_text())
    gate = (report.get("screen_gate") or {}).get("results") or report["univariate_screen"][f"timing_pivot_k{k}"]
    screen_pass = {f for f, r in gate.items() if r["pass_fdr"]}
    causal = causal_stage(Xt, Tt, bars, cfg["causal"], k, sorted(screen_pass | set(report["recipe"]["selected"])), mixed=True)
    report["causal"] = causal
    _write_json(out / "CAUSAL_STAGE.json", causal)
    _write_json(out / "REPORT.json", report)
    return causal


def attach_eval(out, eval_dir) -> dict:
    out, eval_dir = Path(out), Path(eval_dir)
    doc = json.loads((out / "SELECTION_MANIFEST.json").read_text())
    seal = json.loads((eval_dir / "SEAL.json").read_text())
    doc["predictive_validation"] = {"seal_sha256": tp.sha256_file(eval_dir / "SEAL.json"), "evaluator": seal.get("evaluator"),
                                    "freeze_sha256": seal.get("freeze_sha256"), "roles": seal.get("roles"), "proxy_only": True}
    doc["source_digests"]["evaluator_seal"] = doc["predictive_validation"]["seal_sha256"]
    doc["status"] = SEAL_STATUS
    doc.pop("manifest_sha256", None)
    doc["manifest_sha256"] = gate_digest(doc, "manifest_sha256")
    _write_json(out / "SELECTION_MANIFEST.json", doc)
    return doc


# ----------------------------------------------------------------------------- OLAP publication + snapshots (A12)
OLAP_TABLES = {
    "timing_selection_manifest": "row_key TEXT PRIMARY KEY, run_id TEXT, manifest_sha256 TEXT, status TEXT, body TEXT",
    "timing_selection_features": "row_key TEXT PRIMARY KEY, run_id TEXT, ordinal INTEGER, feature TEXT, state TEXT, reasons TEXT",
    "timing_selection_eval": "row_key TEXT PRIMARY KEY, run_id TEXT, feature_set TEXT, arm TEXT, period TEXT, role TEXT, metric TEXT, value REAL",
}


def _olap_rows(out: Path) -> dict:
    out = Path(out)
    m = json.loads((out / "SELECTION_MANIFEST.json").read_text())
    run_id = m["manifest_sha256"]
    rows = {"timing_selection_manifest": [(f"{run_id}:manifest", run_id, m["manifest_sha256"], m["status"],
                                           json.dumps(m, sort_keys=True))],
            "timing_selection_features": [(f"{run_id}:feature:{f['name']}", run_id, i, f["name"], f["state"], ",".join(f["reason_codes"]))
                                          for i, f in enumerate(m["features"])],
            "timing_selection_eval": []}
    summ = out / "eval" / "summary.csv"
    if summ.exists():
        s = pd.read_csv(summ)
        keep = [c for c in s.columns if c not in ("set", "arm", "cell", "threshold", "period", "role") and pd.api.types.is_numeric_dtype(s[c])]
        for _, r in s.iterrows():
            for c in keep:
                v = r[c]
                if pd.notna(v):
                    rows["timing_selection_eval"].append((f"{run_id}:{r['set']}:{r['arm']}:{r['period']}:{c}", run_id, str(r["set"]), str(r["arm"]),
                                                          str(r["period"]), str(r.get("role", "")), c, float(v)))
    return rows


def publish(out, db) -> dict:
    """Idempotent publication into a local sqlite OLAP file: INSERT OR IGNORE on deterministic row keys, then an
    exact readback (count + sorted-row digest) per table; a second publication inserts nothing."""
    rows = _olap_rows(out)
    con = sqlite3.connect(str(db))
    try:
        for t, ddl in OLAP_TABLES.items():
            con.execute(f"CREATE TABLE IF NOT EXISTS {t} ({ddl})")
        inserted = {}
        for t, rs in rows.items():
            before = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            if rs:
                con.executemany(f"INSERT OR IGNORE INTO {t} VALUES ({','.join('?' * len(rs[0]))})", rs)
            con.commit()
            inserted[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] - before
        run_id = rows["timing_selection_manifest"][0][1]
        readback = {}
        for t, rs in rows.items():
            got = con.execute(f"SELECT * FROM {t} WHERE run_id = ? ORDER BY row_key", (run_id,)).fetchall()
            want = sorted(rs, key=lambda r: r[0])
            readback[t] = {"count": len(got), "expected": len(want),
                           "match": [tuple(g) for g in got] == [tuple(w) for w in want]}
    finally:
        con.close()
    doc = {"db": Path(db).name, "run_id": run_id, "inserted": inserted, "readback": readback,
           "all_match": all(v["match"] for v in readback.values())}
    _write_json(Path(out) / "OLAP_PUBLICATION.json", doc)
    return doc


SNAPSHOT_FILES = ("SELECTION_MANIFEST.json", "MANIFEST.json", "TARGETS.json", "LABEL_LEDGER.parquet", "INVENTORY.json",
                  "QUALITY_PROFILES.json", "PERTURBATION.json", "REPORT.json", "CAUSAL_STAGE.json", "DAG.json", "READ_LEDGER.json",
                  "FROZEN_SUBSET.json", "TABLES.md", "OLAP_PUBLICATION.json", "PREPARED.json", "POOL_META.json",
                  "campaign/PHASE_2_COMPLETE.json", "campaign/PHASE_3_FILTER_COMPLETE.json", "campaign/CANDIDATES_FOR_VALIDATION.json",
                  "eval_input/FEATURE_SETS.json", "eval/SEAL.json", "eval/FROZEN.json", "eval/summary.csv", "eval/UNCERTAINTY.json",
                  "eval/scores_develop.csv", "eval/scores_reobserved_2024.csv", "eval/RECONFIRM.json", "eval/trials_threshold_inner_a.csv",
                  "eval/COMPUTE_LEDGER.json", "READ_LEDGER_ADDENDUM.json", "GATE_CHECK.json", "CLOCK_AND_COVERAGE_WITNESS.json", "SNAPSHOT.json")


def payload_files(out: Path) -> list:
    """The FULL replay payload: every file under the run directory (feature/target matrices, scalers in the
    develop records, saved models, prediction archives, ledgers, the phase-2/3 terminals and the metrics database)."""
    out = Path(out)
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name not in ("SNAPSHOT.json",))


def snapshot(out, dests, payload: str = "full") -> dict:
    """Copy the run to each destination with SHA256SUMS, re-hash every copy, and record each copy's device id (a
    second copy on the same device is reported as such, never as an independent physical copy).  ``payload``:
    ``full`` (default; every file, see ``payload_files``) or ``metadata`` (the legacy sealed-artifact list)."""
    out = Path(out)
    files = payload_files(out) if payload == "full" else [f for f in SNAPSHOT_FILES if (out / f).exists()]
    sums = {f: tp.sha256_file(out / f) for f in files}
    copies = []
    for d in dests:
        d = Path(d)
        for f in files:
            (d / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out / f, d / f)
        (d / "SHA256SUMS").write_text("".join(f"{sums[f]}  {f}\n" for f in files))
        bad = [f for f in files if tp.sha256_file(d / f) != sums[f]]
        copies.append({"dest": d.name, "device": os.stat(d).st_dev, "files": len(files), "verified": not bad, "mismatches": bad})
    devices = {c["device"] for c in copies} | {os.stat(out).st_dev}
    kinds = {"models": sum(f.endswith((".keras", ".pkl")) for f in files), "npz": sum(f.endswith(".npz") for f in files),
             "parquet": sum(f.endswith(".parquet") for f in files), "databases": sum(f.endswith((".duckdb", ".sqlite")) for f in files)}
    doc = {"payload": payload, "files": len(files), "kinds": kinds, "bytes": int(sum((out / f).stat().st_size for f in files)), "sha256sums": sums, "copies": [{k: v for k, v in c.items() if k != "device"} for c in copies],
           "distinct_devices_including_source": len(devices),
           "independent_physical_copy": False,
           "device_note": "device ids are filesystem ids on ONE host; partitions of one physical disk are not independent copies"}
    _write_json(out / "SNAPSHOT.json", doc)
    return doc


# ----------------------------------------------------------------------------- tables
def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def tables(out) -> str:
    """Render the run's evidence into TABLES.md."""
    out = Path(out)
    rep = json.loads((out / "REPORT.json").read_text())
    sel = json.loads((out / "SELECTION_MANIFEST.json").read_text())
    inv = json.loads((out / "INVENTORY.json").read_text())
    k = int(sel["target_params"]["k_bars"])
    primary = f"timing_pivot_k{k}"
    tg = rep["targets"]
    parts = [f"# {SELECTION_NAME} tables ({sel['dataset_id']}, identity {rep['identity']})\n",
             f"Status `{sel['status']}`; manifest sha256 `{sel['manifest_sha256']}`; TRAIN rows {rep['train_rows']}; features {rep['features']} "
             f"({inv['count_reconciliation']}).\n",
             "## Targets and support ledger\n",
             f"Semantics `{tg['semantics']}`; pivots {tg['pivots']}, confirmed {tg['confirmed_pivots']}, never confirmed {tg['unconfirmed_pivots']}; "
             f"finality: all rows declared final unchanged under truncation and future perturbation = {tg['finality']['all_final_rows_unchanged']}.\n"]
    rows = []
    for p, v in tg["status_by_period"].items():
        for t in ("t1", "t2", "t3"):
            rows.append([p, t, v["origins"], *[v[t][s] for s in tp.STATUS]])
    parts.append(_md_table(["period", "target", "origins", *tp.STATUS], rows))
    rec = rep["recipe"]
    parts.append("\n## Selection recipe (primary T1 task only)\n")
    parts.append(_md_table(["step", "features"], [
        [f"{rec['method']} K{rec['k']} union over {', '.join(rec['targets'])}", f"{len(rec['union'])}: " + ", ".join(rec["union"])],
        ["dropped by the T1 FDR screen", ", ".join(rec["dropped_by_fdr_screen"]) or "-"],
        ["pruned as redundant", ", ".join(rec["pruned_redundant"]) or "-"],
        ["SELECTED", f"{len(rec['selected'])}: " + ", ".join(rec["selected"])]]))
    parts.append("\nComparator subsets (reported, not used): " + "; ".join(f"{t}: {', '.join(m)}" for t, m in rep["comparator_subsets_T2_T3"].items()))
    scr = rep["univariate_screen"][primary]
    rows = sorted(((f, r) for f, r in scr.items() if r["p"] is not None), key=lambda x: x[1]["p"])
    parts.append("\n## Univariate screen vs T1 (Kruskal-Wallis, BH q 0.1)\n")
    parts.append(_md_table(["feature", "H", "p", "pass_fdr", "n"], [[f, r["statistic"], r["p"], r["pass_fdr"], r["n"]] for f, r in rows]))
    st = rep.get("fold_stability") or {}
    if st:
        parts.append("\n## Stability across prefix folds\n")
        parts.append(_md_table(["fold", "rank Spearman vs TRAIN", "passed"], [[f, v["rank_spearman_vs_train"], len(v["passed"])] for f, v in st["folds"].items()]))
        parts.append("\nSelected: FDR pass count over (folds + TRAIN) = " + ", ".join(f"{f}: {st['pass_agreement'].get(f)}" for f in rec["selected"]))
    wd = rep["within_day_variance"]
    parts.append("\n## Within-day variance\n")
    parts.append("Constant within a day: " + (", ".join(f for f, v in wd.items() if v["constant_within_day"]) or "none") + ".\n")
    parts.append(_md_table(["selected feature", "within-day variance ratio"], [[f, wd[f]["within_day_variance_ratio"]] for f in rec["selected"]]))
    red = rep["redundancy"]
    parts.append("\n## Redundancy\n")
    parts.append(_md_table(["selected feature", "VIF"], [[f, v] for f, v in red["vif_selected"].items()]))
    multi = [c for c in red["clusters"] if len(c["members"]) > 1]
    parts.append(f"\nTRAIN Spearman clusters (|rho| >= 0.9, average linkage): {len(red['clusters'])} clusters, {len(multi)} with > 1 member.\n")
    parts.append(_md_table(["cluster", "members"], [[c["cluster_id"], ", ".join(c["members"])] for c in multi]))
    fr = red["fractal_vs_realized_vol_spearman"]
    if fr:
        cols = sorted(next(iter(fr.values())).keys())
        parts.append("\nFractal columns vs the realized-vol family (Spearman, TRAIN):\n")
        parts.append(_md_table(["fractal", *cols], [[f, *[v[c] for c in cols]] for f, v in fr.items()]))
    ca = rep.get("causal") or {}
    if ca.get("mixed_ci") and isinstance(ca["mixed_ci"], dict) and ca["mixed_ci"].get("records"):
        mc = ca["mixed_ci"]
        parts.append(f"\n## Causal stage: {mc['evidence_label']} (tigramite {mc.get('pin', {}).get('version')} RegressionCI, PCMCI+, tau_max {mc['tau_max']})\n")
        parts.append(f"History H = {mc['history']}. {ca['lag0_mapping']}. Controls: {mc['control_verdicts']} (behave: {mc['controls_behave']}).\n")
        rows = []
        for f, r in mc["records"].items():
            best = min(r["links_x_to_y"], key=lambda lk: lk["p"]) if r["links_x_to_y"] else None
            stab = mc["temporal_resampling"].get(f, {})
            rows.append([f, r["verdict"], best and best["lag"], best and best["edge"], best and best["p"], stab.get("first_half"), stab.get("second_half")])
        rows.sort(key=lambda r: (r[1] == "NO_LINK", r[4] if r[4] is not None else 1.0))
        parts.append(_md_table(["feature", "verdict", "lag", "edge", "p", "verdict H1", "verdict H2"], rows))
    for t, v in (ca.get("hac_lag_screen") or {}).items():
        parts.append(f"\nHAC lag screen (association comparator), {t}: parents at lags 1..{k} = {len(v['parents_at_lags_1_to_k'])}: "
                     + ", ".join(v["parents_at_lags_1_to_k"]))
    mcc = ca.get("matched_control_contrast") or {}
    rows = sorted([(f, r["events"], r["excess_pivot_rate"], r["ci95"][0], r["ci95"][1], r["significant"])
                   for f, r in mcc.items() if r.get("state") == "ESTIMATED"], key=lambda r: -abs(r[2]))
    if rows:
        parts.append("\n### Matched-control contrast (association comparator; outcome = pivot within k)\n")
        parts.append(_md_table(["feature", "events", "excess", "ci lo", "ci hi", "significant"], rows[:30]))
    ev = out / "eval" / "SEAL.json"
    if ev.exists():
        seal = json.loads(ev.read_text())
        parts.append("\n## Predictive validation (proxy; identical rows)\n")
        rows = [[s["set"], s["arm"], s.get("cell"), s.get("threshold"), s["period"], s.get("role"),
                 s.get("valley_long_pr_auc"), s.get("peak_short_pr_auc"), s.get("valley_long_base_rate_incl_none"),
                 s.get("log_loss"), s.get("valley_long_precision"), s.get("peak_short_precision"),
                 s.get("sharpe_daily_sqrt365"), s.get("net_profit_usd"), s.get("max_drawdown_pct"), s.get("entries_filled"), s.get("win_rate"), s.get("valid")]
                for s in seal["summary"] if s["period"] != "TRAIN_2020_2022"]
        parts.append(_md_table(["set", "arm", "cell", "tau", "period", "role", "PR-AUC valley", "PR-AUC peak", "base valley",
                                "log-loss", "prec long", "prec short", "Sharpe", "net USD", "MDD %", "entries", "win %", "valid"], rows))
    text = "\n".join(parts) + "\n"
    (out / "TABLES.md").write_text(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--reuse-from", default=None, help="a previous run directory; target-independent stages with identical identities are reused")
    a = sub.add_parser("attach-eval")
    a.add_argument("--out", required=True)
    a.add_argument("--eval", required=True)
    c = sub.add_parser("causal")
    c.add_argument("--config", required=True)
    c.add_argument("--out", required=True)
    t = sub.add_parser("tables")
    t.add_argument("--out", required=True)
    pb = sub.add_parser("publish")
    pb.add_argument("--out", required=True)
    pb.add_argument("--db", required=True)
    sn = sub.add_parser("snapshot")
    sn.add_argument("--out", required=True)
    sn.add_argument("--dest", action="append", required=True)
    sn.add_argument("--payload", choices=("full", "metadata"), default="full")
    args = ap.parse_args(argv)
    if args.verb == "run":
        rep = run(args.config, args.out, args.reuse_from)
        print(json.dumps({"identity": rep["identity"], "selected": rep["recipe"]["selected"], "wall_seconds": rep["wall_seconds"],
                          "dag": rep["dag_plan"]}))
    elif args.verb == "causal":
        causal = causal_only(args.config, args.out)
        mc = causal.get("mixed_ci") or {}
        print(json.dumps({"mixed_ci_controls": mc.get("control_verdicts"), "behave": mc.get("controls_behave"),
                          "hac_parents": {t_: len(v["parents_at_lags_1_to_k"]) for t_, v in causal["hac_lag_screen"].items()}}))
    elif args.verb == "attach-eval":
        print(json.dumps({"manifest_sha256": attach_eval(args.out, args.eval)["manifest_sha256"]}))
    elif args.verb == "tables":
        print(tables(args.out)[:300])
    elif args.verb == "publish":
        print(json.dumps(publish(args.out, args.db)))
    else:
        print(json.dumps({k_: v for k_, v in snapshot(args.out, args.dest, args.payload).items() if k_ != "sha256sums"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
