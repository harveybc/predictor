#!/usr/bin/env python3
"""Timing-target feature selection driver (JSON config): the existing phase-2/3 engines run unchanged on a
population whose targets are the ZigZag hindsight expert's timing labels; the timing-specific stages
(availability gate, univariate FDR screen, causal stage, redundancy / stability / within-day reports)
are added around them, and the result is a sealed selection manifest reusable by RL (gym-fx observation
contract) and by heuristic strategies.  Predictive validation is the strategy's pivot-detector harness
(heuristic-strategy ``tools/short_horizon_timing_eval.py``); its SEAL is attached with ``attach-eval``.

    python tools/timing_selection.py run --config CFG.json --out OUT
    python tools/timing_selection.py attach-eval --out OUT --eval EVAL_DIR

Rules: TEST rows (2025-01-01+) are never read; TRAIN = the decision years only; the engines see TRAIN
only; 2023 (INNER) is read by the evaluator for the operating point; 2024 is read once by the evaluator
as the confirmation; no host name, address or private path is written into any artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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
FRACTAL_COLUMNS = ("hurst_rs", "variogram_hurst", "hurst_spectrum_width", "box_counting_dimension")
RAW_LEVEL_COLUMNS = ("OPEN", "HIGH", "LOW", "CLOSE")


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
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (pd.Timestamp,)):
        return str(v)
    raise TypeError(type(v).__name__)


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


# ----------------------------------------------------------------------------- candidate pool
def inventory_candidates(bars: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, list[dict]]:
    """The admitted-core candidate population computed by its owners' code loaded by path: financial-data
    ``core_selection_bundle.candidate_frame`` and feature-eng ``normalize_phase1b.add_derived_features``."""
    bundle = _load_by_path("core_selection_bundle", spec["bundle_script"], spec.get("bundle_sha256"))
    gen = _load_by_path("core_selection_technical_generator", spec["generator_module"], spec.get("generator_sha256"))
    frame = bars.reset_index().rename(columns={"DATE_TIME": "DATE_TIME"})
    x = bundle.candidate_frame(frame, gen.add_derived_features, gen.FEATURE_COLS)
    cols = [c for c in x.columns if c != "DATE_TIME"]
    x = x.set_index(pd.DatetimeIndex(x["DATE_TIME"]))[cols]
    x.index = bars.index
    meta = []
    for c in cols:
        known = c in ("hod_sin", "hod_cos", "dow_sin", "dow_cos")
        meta.append({"feature_id": c, "family": "calendar_known" if known else ("raw_price" if c in RAW_LEVEL_COLUMNS or c.startswith("RAW_") else "technical"),
                     "source": "derived:lake_ohlc_h1 via financial-data core_selection_bundle + feature-eng normalize_phase1b",
                     "transform": "trailing indicator (pandas_ta / rolling / ewm) of bars <= t" if not known else "bar label encoding",
                     "unit": "price" if c in RAW_LEVEL_COLUMNS else "dimensionless", "clock": "KNOWN_IN_ADVANCE" if known else "OBSERVED",
                     "availability_time": "known in advance" if known else "timestamp", "support_h": 0 if known else 1,
                     "train_coverage": float(x[c].notna().mean()), "source_bytes": int(x[c].notna().sum()) * 8})
    return x, meta


def fractal_candidates(bars: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, list[dict], dict]:
    """The causal fractal columns.  Preferred: the files the fractal lane provides (``files``: several parquet
    files, columns ``<feature>_w<window>``, joined on ``date_time``; TEST never present) with their MANIFEST
    sha256 cited.  Fallback until they land: computed with agent-multi's ``causal_fractal_frame`` loaded by path."""
    note = {}
    if spec.get("files"):
        parts = []
        for f in spec["files"]:
            raw = pd.read_parquet(Path(f))
            tcol = next(c for c in ("date_time", "timestamp", "datetime", "DATE_TIME") if c in raw.columns)
            idx = pd.DatetimeIndex(pd.to_datetime(raw[tcol]))
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            raw = raw.set_index(idx)
            if len(raw) and raw.index.max() >= tp.TEST_START:
                raise DriverError("TEST_ROWS_FORBIDDEN: fractal file reaches TEST rows")
            cols = [c for c in raw.columns if any(c.startswith(b + "_w") for b in FRACTAL_COLUMNS)]
            parts.append(raw[cols])
        joined = pd.concat(parts, axis=0)
        joined = joined.groupby(level=0).first()           # one row per timestamp across files/windows
        x = joined.reindex(bars.index)
        note = {"source": "files", "files_sha256": {Path(f).name: tp.sha256_file(f) for f in spec["files"]},
                "manifest_sha256": spec.get("manifest_sha256"), "columns": list(x.columns),
                "joined_rows": int(x.notna().any(axis=1).sum())}
        if note["joined_rows"] == 0:
            raise DriverError("FRACTAL_JOIN_EMPTY: the fractal files share no timestamp with the lake window")
    elif spec.get("path"):
        p = Path(spec["path"])
        raw = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        tcol = next(c for c in ("timestamp", "date_time", "datetime", "DATE_TIME") if c in raw.columns)
        idx = pd.DatetimeIndex(pd.to_datetime(raw[tcol]))
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        raw = raw.set_index(idx)
        cols = [c for c in FRACTAL_COLUMNS if c in raw.columns]
        x = raw[cols].reindex(bars.index)
        note = {"source": "file", "sha256": tp.sha256_file(p), "joined_rows": int(x.notna().all(axis=1).sum())}
        if note["joined_rows"] == 0:
            raise DriverError("FRACTAL_JOIN_EMPTY: the fractal file shares no timestamp with the lake window")
    else:
        mod = _load_by_path("causal_fractal_features", spec["compute_with_module"], spec.get("module_sha256"))
        window = int(spec.get("window", mod.WINDOW))
        x = mod.causal_fractal_frame(bars["CLOSE"].to_numpy(float), window=window)[list(FRACTAL_COLUMNS)]
        x.index = bars.index
        note = {"source": "computed_locally_with_owner_module", "module_sha256": tp.sha256_file(spec["compute_with_module"]),
                "window_bars": window, "pending": "the fractal lane's official EURUSD H1 files replace this when they land"}
    meta = [{"feature_id": c, "family": "fractal", "source": "feature-eng causal_fractal plugin (ported from agent-multi tools/causal_fractal_features.py)",
             "transform": "trailing-window fractal estimator of closes <= t", "unit": "dimensionless",
             "clock": "OBSERVED", "availability_time": "timestamp", "support_h": 1, "train_coverage": float(x[c].notna().mean()),
             "source_bytes": int(x[c].notna().sum()) * 8} for c in x.columns]
    return x, meta, note


def assemble_pool(bars: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict], dict]:
    cand = cfg["candidates"]
    frames, meta, notes = [], [], {}
    if cand.get("inventory"):
        x, m = inventory_candidates(bars, cand["inventory"])
        frames.append(x)
        meta += m
        notes["inventory"] = {"columns": list(x.columns)}
    if cand.get("price_pool", True):
        x = pool.price_pool(bars)
        frames.append(x)
        meta += pool.pool_metadata(x)
    if cand.get("fractal"):
        x, m, n = fractal_candidates(bars, cand["fractal"])
        frames.append(x)
        meta += m
        notes["fractal"] = n
    if not frames:
        raise DriverError("EMPTY_POOL: no candidate source declared")
    X = pd.concat(frames, axis=1)
    if X.columns.duplicated().any():
        raise DriverError(f"DUPLICATE_FEATURE_IDS: {sorted(set(X.columns[X.columns.duplicated()]))}")
    return X, meta, notes


# ----------------------------------------------------------------------------- screens and reports
def _bh(p, q):
    p = np.asarray(p, float)
    n = len(p)
    out = np.zeros(n, bool)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    idx = np.flatnonzero(ok)
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
    passed = _bh(pvals, q)
    for name, ok in zip(names, passed):
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
    m = np.isfinite(x).all(axis=1)
    x = x[m]
    out = {}
    if len(x) < 10 or x.shape[1] < 2:
        return {c: np.nan for c in X.columns}
    xz = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)
    for j, c in enumerate(X.columns):
        others = np.delete(xz, j, axis=1)
        a = np.column_stack([np.ones(len(xz)), others])
        beta, *_ = np.linalg.lstsq(a, xz[:, j], rcond=None)
        resid = xz[:, j] - a @ beta
        r2 = 1.0 - resid.var() / max(xz[:, j].var(), 1e-12)
        out[c] = float(1.0 / max(1.0 - r2, 1e-12))
    return out


def redundancy_report(X: pd.DataFrame, clusters: list[dict], selected: list[str]) -> dict:
    corr = X.corr(method="spearman")
    fract = [c for c in X.columns if c in FRACTAL_COLUMNS]
    rv = [c for c in X.columns if c.startswith("px.rv_") or c in ("rolling_std_24", "ATR", "px.atr_14")]
    fr_vs_rv = {f: {r: float(corr.loc[f, r]) for r in rv} for f in fract}
    return {"clusters": [{"cluster_id": c["cluster_id"], "members": c["members"]} for c in clusters],
            "vif_selected": vif(X[selected]) if selected else {}, "fractal_vs_realized_vol_spearman": fr_vs_rv}


# ----------------------------------------------------------------------------- causal stage
def causal_stage(X: pd.DataFrame, T: pd.DataFrame, bars: pd.DataFrame, cfg: dict, k: int, features: list[str]) -> dict:
    """The causal stage is owned by causal-inference (``fs_timing_pack``, EURUSD_TIMING pack): PCMCI+ through the
    existing comparator (NOT_RUN without tigramite), the lagged HAC partial screen at lags 0..k (BH per target)
    and the matched-control contrast in the style of event_impact.match_controls.  Loaded from ``provider_src``."""
    src = Path(cfg["provider_src"])
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from causal_inference_provider import fs_timing_pack as tpk  # noqa: E402
    hist = [h for h in cfg.get("history", ["px.logret_24", "px.rv_24"]) if h in X.columns]
    tnames = [t for t in cfg.get("targets", [f"timing_valley_k{k}", f"timing_peak_k{k}"]) if t in T.columns]
    cols = [f for f in features if f in X.columns] + [h for h in hist if h not in features]
    Xs = X[cols]
    out = {"pack": tpk.PACK_ID, "provider_sha256": tp.sha256_file(Path(tpk.__file__)), "history": hist, "targets": tnames,
           "hac_lag_screen": {}, "pcmci": {}, "matched_control_contrast": {}}
    for t in tnames:
        y = T[t].to_numpy(float)
        out["hac_lag_screen"][t] = tpk.hac_lag_screen(Xs, y, k=k, history=hist, q=float(cfg.get("q", 0.1)))
        out["pcmci"][t] = tpk.pcmci_parents(Xs, y, k=k, history=hist, target_name=t, pc_alpha=cfg.get("pc_alpha", 0.05),
                                            alpha=cfg.get("alpha", 0.01), max_features=cfg.get("pcmci_max_features"))
    bars_ei = tpk.bars_table(bars.index, bars["CLOSE"].to_numpy(float), bars["HIGH"].to_numpy(float), bars["LOW"].to_numpy(float))
    piv = T[f"timing_pivot_k{k}"].to_numpy(float)
    piv_any = np.where(np.isfinite(piv), (piv > 0).astype(float), np.nan)
    for f in [c for c in cols if c not in hist]:
        out["matched_control_contrast"][f] = tpk.matched_control_contrast(
            bars_ei, X[f].to_numpy(float), piv_any, k=k, event_quantile=float(cfg.get("event_quantile", 0.8)),
            n_controls=int(cfg.get("n_controls", 5)), min_events=int(cfg.get("min_events", 20)),
            n_boot=int(cfg.get("n_boot", 400)), seed=int(cfg.get("seed", 0)))
    return out


# ----------------------------------------------------------------------------- selection recipe
def apply_recipe(candidates: dict, clusters: list[dict], rankings: dict, recipe: dict, screen_pass: set[str]) -> dict:
    """Union of the engine's ``method``/``k`` subsets over the recipe targets, intersected with the FDR screen,
    pruned to one representative per TRAIN Spearman redundancy cluster (max relevance for the primary target)."""
    method, k = recipe.get("method", "MRMR"), int(recipe.get("k", 12))
    tids = recipe.get("targets") or sorted({c["target_id"] for c in candidates["candidates"]})
    union, per_target = set(), {}
    for c in candidates["candidates"]:
        if c["method"] == method and c["k"] == k and c["target_id"] in tids:
            per_target[c["target_id"]] = list(c["members"])
            union |= set(c["members"])
    kept_screen = union & screen_pass if recipe.get("require_fdr_pass", True) else set(union)
    cluster_of = {}
    for c in clusters:
        for m in c["members"]:
            cluster_of[m] = c["cluster_id"]
    rel = rankings
    by_cluster: dict = {}
    for f in kept_screen:
        by_cluster.setdefault(cluster_of.get(f, f"single:{f}"), []).append(f)
    selected = sorted(max(ms, key=lambda f: (rel.get(f, 0.0), f)) for ms in by_cluster.values())
    pruned = sorted(kept_screen - set(selected))
    return {"method": method, "k": k, "targets": tids, "per_target": per_target, "union": sorted(union),
            "dropped_by_fdr_screen": sorted(union - kept_screen), "pruned_redundant": pruned, "selected": selected}


# ----------------------------------------------------------------------------- driver
def run(config_path, out) -> dict:
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
    lake = cfg["lake"]
    if tp.sha256_file(lake["path"]) != lake["sha256"]:
        raise DriverError("LAKE_DIGEST: lake file differs from its pin")
    bars = tp.load_lake_bars(lake["path"], lake["load"][0], lake["load"][1])
    zz = tp.load_zigzag_module(cfg["zigzag"]["module"], cfg["zigzag"]["sha256"])
    zp = cfg["zigzag"]["params"]
    params = zz.Params(depth=int(zp["depth"]), deviation_pips=float(zp["deviation_pips"]), backstep=int(zp["backstep"]))
    k = int(cfg["zigzag"]["k_bars"])
    sp = cfg["splits"]
    cuts = [sp["train"][1], sp["inner"][1], sp["confirm"][1]] + list(sp.get("folds", []))
    cuts = sorted({str(_utc(c)) for c in cuts})
    T = tp.build_targets(bars, zz, params, k_bars=k, cuts=cuts)
    T.index = bars.index
    X, meta, notes = assemble_pool(bars, cfg)
    admitted = pool.availability_gate(meta)
    X = X[admitted]
    # ---- TRAIN split for the engines (decision years only), prefix folds at the declared cuts
    tr0, tr1 = _utc(sp["train"][0]), _utc(sp["train"][1])
    train = (bars.index >= tr0) & (bars.index < tr1)
    Xt, Tt = X.loc[train].reset_index(drop=True), T.loc[train].reset_index(drop=True)
    n = len(Xt)
    ts_train = bars.index[train]
    folds = []
    for c in sp.get("folds", []):
        stop = int(ts_train.searchsorted(_utc(c)))
        folds.append({"fold_id": f"prefix_before_{str(_utc(c))[:10]}", "train_rows": [0, stop]})
    data = out / "data"
    data.mkdir(exist_ok=True)
    feat_cols = list(Xt.columns)
    fx = pd.concat([pd.DataFrame({"row_id": np.arange(n), "timestamp": ts_train}), Xt], axis=1)
    tt = pd.concat([pd.DataFrame({"row_id": np.arange(n), "timestamp": ts_train}), Tt.drop(columns=["row_id", "timestamp"])], axis=1)
    if not prepared.exists():
        fx.to_parquet(data / "features_train.parquet", index=False)
        tt.to_parquet(data / "targets_train.parquet", index=False)
    targets = tp.manifest_targets(k)
    for m in meta:
        m["train_coverage"] = float(Xt[m["feature_id"]].notna().mean()) if m["feature_id"] in Xt else 0.0
    manifest = man.build_manifest(
        population_id=cfg["population_id"], identity=f"timing-selection:{config_sha[:16]}",
        identity_source={"kind": "TIMING_SELECTION_CONFIG", "config_sha256": config_sha},
        phase1={"state": "NOT_RUN", "scope": "timing targets; phase-1 profiles not part of this driver"},
        data={"features_file": "features_train.parquet", "targets_file": "targets_train.parquet",
              "features_sha256": man.sha256_file(data / "features_train.parquet"),
              "targets_sha256": man.sha256_file(data / "targets_train.parquet"), "timestamp_column": "timestamp",
              "row_id_column": "row_id", "bar_hours": 1, "train_rows": n},
        features=[m for m in meta if m["feature_id"] in feat_cols], targets=targets, folds=folds, causal_supported=[],
        contract_source="timing_pivot targets from the sealed ZigZag expert (labels only)", data_root=data)
    man.write_manifest(manifest, out / "MANIFEST.json")
    targets_doc = tp.targets_document(T, params=params, k_bars=k, lake_sha256=lake["sha256"], zigzag_sha256=cfg["zigzag"]["sha256"])
    _write_json(out / "TARGETS.json", targets_doc)
    _write_json(prepared, {"config_sha256": config_sha, "config": _redact(cfg), "manifest_sha256": manifest["manifest_sha256"],
                           "pool_notes": notes, "admitted": admitted, "train_rows": n})
    # ---- phases 2 and 3: the existing engines, unchanged, local warehouse (duckdb or sqlite fallback)
    from tools import feature_pairwise_campaign as campaign
    try:
        import duckdb  # noqa: F401
        from tools.fs_phase23_warehouse import Warehouse
    except ModuleNotFoundError:                       # the adapter's sqlite3 backend, reported in every receipt
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
    from tools import feature_filter_selection as sel
    feats = [str(f) for f in mats["features"]]
    clusters = campaign.redundancy_clusters(feats, mats["spearman"], [bool(v) for v in mats["admissible"]], manifest,
                                            json.loads((state / "PLAN.json").read_text())["params"])
    # ---- timing-specific stages
    primary = f"timing_pivot_k{k}"
    y1 = Tt[primary].to_numpy(float)
    screen = univariate_screen(Xt, y1, head="classification", q=float(cfg.get("fdr_q", 0.1)))
    screens = {primary: screen}
    for t in targets:
        if t["target_id"] != primary:
            screens[t["target_id"]] = univariate_screen(Xt, Tt[t["column"]].to_numpy(float), head=t["head"], q=float(cfg.get("fdr_q", 0.1)))
    stability = fold_stability(Xt, y1, folds, head="classification", q=float(cfg.get("fdr_q", 0.1))) if folds else {}
    within_day = pool.within_day_variance(Xt.set_index(ts_train))
    # relevance for cluster representatives: MI(feature; primary target) from the engine's rankings
    rankings = {}
    p3 = state / "phase3"
    for path in sorted((p3 / "terminals").glob("*.json.gz")):
        doc = campaign.load_terminal(path)
        if doc.get("target_id") == primary:
            for r in doc["rows"]["feature_filter_rankings"]:
                if r["method"] == "MRMR":
                    rankings[r["feature_id"]] = float((r.get("terms") or {}).get("relevance") or r.get("score") or 0.0)
    screen_pass = {f for f, r in screen.items() if r["pass_fdr"]}
    recipe = apply_recipe(candidates, clusters, rankings, cfg["filter"].get("recipe", {}), screen_pass)
    selected = recipe["selected"]
    redundancy = redundancy_report(Xt, clusters, selected)
    causal = None
    if cfg.get("causal", {}).get("enabled"):
        causal = causal_stage(Xt, Tt, bars.loc[train], cfg["causal"], k, sorted(screen_pass | set(selected)))
    # ---- evaluator input: all loaded rows (TRAIN + INNER + CONFIRM), features and labels; TEST never loaded
    ev = out / "eval_input"
    ev.mkdir(exist_ok=True)
    full = pd.concat([pd.DataFrame({"timestamp": bars.index}), X.reset_index(drop=True)], axis=1)
    full.to_parquet(ev / "features_all.parquet", index=False)
    T.reset_index(drop=True).to_parquet(ev / "targets_all.parquet", index=False)
    sets = {"timing_selected": selected, "timing_union_before_prune": recipe["union"]}
    for name, cols in (cfg.get("comparison_sets") or {}).items():
        sets[name] = [c for c in cols]
    _write_json(ev / "FEATURE_SETS.json", {"sets": sets, "features_sha256": man.sha256_file(ev / "features_all.parquet"), "fractal_manifest_sha256": (cfg["candidates"].get("fractal") or {}).get("manifest_sha256"),
                                            "targets_sha256": man.sha256_file(ev / "targets_all.parquet"), "k_bars": k,
                                            "splits": sp, "missing_in_pool": {n_: [c for c in cols if c not in X.columns] for n_, cols in sets.items()}})
    # ---- reports and the sealed (DRAFT) manifest
    report = {"schema": "predictor.timing_selection.report.v1", "config_sha256": config_sha, "manifest_sha256": manifest["manifest_sha256"],
              "identity": manifest["identity"], "train_rows": n, "features": len(feat_cols), "admitted": admitted,
              "pool_notes": _redact(notes), "targets": targets_doc, "worker": _redact(worker), "phase2": {"closure_sha256": p2["closure_sha256"]},
              "phase3": {"methods": candidates and json.loads((state / "PHASE_3_FILTER_COMPLETE.json").read_text())["methods"],
                         "candidates": len(candidates["candidates"])}, "readback": readback, "recipe": recipe,
              "univariate_screen": screens, "fold_stability": stability, "within_day_variance": within_day,
              "redundancy": redundancy, "causal": causal, "wall_seconds": time.time() - t0, "test_read": False}
    _write_json(out / "REPORT.json", report)
    _write_json(out / "CAUSAL_STAGE.json", causal or {"state": "NOT_ENABLED"})
    sel_manifest = selection_manifest(cfg, manifest, targets, feat_cols, selected, recipe, screen, within_day, out)
    _write_json(out / "SELECTION_MANIFEST.json", sel_manifest)
    return report


def selection_manifest(cfg, manifest, targets, feat_cols, selected, recipe, screen, within_day, out: Path) -> dict:
    features = []
    for f in feat_cols:
        if f in selected:
            state, reasons = "selected", [f"{recipe['method']}_K{recipe['k']}_UNION", "FDR_PASS", "CLUSTER_REPRESENTATIVE"]
        else:
            reasons = []
            if f in recipe["pruned_redundant"]:
                reasons.append("REDUNDANT_WITH_SELECTED_REPRESENTATIVE")
            if f in recipe["dropped_by_fdr_screen"]:
                reasons.append("FDR_FAIL")
            if f not in recipe["union"]:
                reasons.append(f"NOT_IN_{recipe['method']}_K{recipe['k']}_UNION")
            state = "rejected"
        if within_day.get(f, {}).get("constant_within_day"):
            reasons.append("CONSTANT_WITHIN_DAY")
        features.append({"feature_id": f, "state": state, "reasons": reasons})
    doc = {"schema": MANIFEST_SCHEMA, "dataset_id": cfg["population_id"], "producer": "timing_selection_driver",
           "role": "producer", "independent_decision_record": None,
           "status": "PRELIMINARY_PENDING_PLAN_AUDIT", "final_selection": False,
           "status_note": "preliminary: the method is under plan audit; stages affected by its findings are rerun before a FINAL seal; "
                          "an independent decision record is still required by selected_manifest_gate",
           "targets": [{"target_id": t["target_id"], "family": t["family"], "head": t["head"], "horizons": [t["horizon_hours"]]} for t in targets],
           "target_params": {"zigzag": cfg["zigzag"]["params"], "k_bars": cfg["zigzag"]["k_bars"], "expert_label": "HINDSIGHT_NON_CAUSAL_EXPERT"},
           "split": cfg["splits"], "features": features, "selected_features": selected,
           "source_digests": {"lake": cfg["lake"]["sha256"], "zigzag_module": cfg["zigzag"]["sha256"],
                              "population_manifest": manifest["manifest_sha256"], "features_train": manifest["data"]["features_sha256"],
                              "targets_train": manifest["data"]["targets_sha256"], "config": man.digest(cfg),
                              "fractal_candidates_manifest": (cfg["candidates"].get("fractal") or {}).get("manifest_sha256"),
                              "inventory_generator": (cfg["candidates"].get("inventory") or {}).get("generator_sha256"),
                              "inventory_bundle_script": (cfg["candidates"].get("inventory") or {}).get("bundle_sha256")},
           "rl_observation_contract": {"schema": "gym_fx.observation_contract.v1", "feature_columns": selected,
                                       "scaling": "rolling_zscore", "note": "ordered columns; scaling window per the RL contract"},
           "predictive_validation": None, "test_read": False}
    doc["manifest_sha256"] = man.digest(doc)
    return doc


def causal_only(config_path, out) -> dict:
    """Re-run the causal stage alone (e.g. under an interpreter that has tigramite) from the saved TRAIN data and
    targets; rewrites CAUSAL_STAGE.json and REPORT.json["causal"].  No engine output is touched."""
    cfg = json.loads(Path(config_path).read_text())
    out = Path(out).resolve()
    prep = json.loads((out / "PREPARED.json").read_text())
    if prep["config_sha256"] != man.digest(cfg):
        raise DriverError("CONFIG_CHANGED: the causal stage must use the run's config")
    k = int(cfg["zigzag"]["k_bars"])
    fx = pd.read_parquet(out / "data" / "features_train.parquet")
    tt = pd.read_parquet(out / "data" / "targets_train.parquet")
    lake = cfg["lake"]
    if tp.sha256_file(lake["path"]) != lake["sha256"]:
        raise DriverError("LAKE_DIGEST: lake file differs from its pin")
    bars = tp.load_lake_bars(lake["path"], cfg["splits"]["train"][0], cfg["splits"]["train"][1])
    if len(bars) != len(fx):
        raise DriverError("TRAIN_GRID_MISMATCH")
    Xt = fx.drop(columns=["row_id", "timestamp"])
    Tt = tt.drop(columns=["row_id", "timestamp"])
    report = json.loads((out / "REPORT.json").read_text())
    screen_pass = {f for f, r in report["univariate_screen"][f"timing_pivot_k{k}"].items() if r["pass_fdr"]}
    feats = sorted(screen_pass | set(report["recipe"]["selected"]))
    causal = causal_stage(Xt, Tt, bars, cfg["causal"], k, feats)
    causal["interpreter"] = sys.version.split()[0]
    report["causal"] = causal
    _write_json(out / "CAUSAL_STAGE.json", causal)
    _write_json(out / "REPORT.json", report)
    return causal


def attach_eval(out, eval_dir) -> dict:
    out, eval_dir = Path(out), Path(eval_dir)
    doc = json.loads((out / "SELECTION_MANIFEST.json").read_text())
    seal = json.loads((eval_dir / "SEAL.json").read_text())
    doc["predictive_validation"] = {"seal_sha256": hashlib.sha256((eval_dir / "SEAL.json").read_bytes()).hexdigest(),
                                    "evaluator": seal.get("evaluator"), "summary": seal.get("summary"), "chosen_on": seal.get("chosen_on"),
                                    "confirm": seal.get("confirm")}
    doc["status"] = "PRELIMINARY_PENDING_PLAN_AUDIT"
    doc.pop("manifest_sha256", None)
    doc["manifest_sha256"] = man.digest(doc)
    _write_json(out / "SELECTION_MANIFEST.json", doc)
    return doc


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def tables(out) -> str:
    """Render the run's evidence into TABLES.md (selection, screens, stability, within-day, redundancy, causal, eval)."""
    out = Path(out)
    rep = json.loads((out / "REPORT.json").read_text())
    sel = json.loads((out / "SELECTION_MANIFEST.json").read_text())
    k = int(sel["target_params"]["k_bars"])
    primary = f"timing_pivot_k{k}"
    parts = [f"# Timing selection tables ({sel['dataset_id']}, identity {rep['identity']})\n",
             f"Status `{sel['status']}`; manifest sha256 `{sel['manifest_sha256']}`; TRAIN rows {rep['train_rows']}; features {rep['features']}.\n"]
    rec = rep["recipe"]
    parts.append("## Selection recipe\n")
    parts.append(_md_table(["step", "features"], [
        [f"{rec['method']} K{rec['k']} union over {', '.join(rec['targets'])}", f"{len(rec['union'])}: " + ", ".join(rec["union"])],
        ["dropped by FDR screen (primary target)", ", ".join(rec["dropped_by_fdr_screen"]) or "-"],
        ["pruned as redundant (cluster representative kept)", ", ".join(rec["pruned_redundant"]) or "-"],
        ["SELECTED", f"{len(rec['selected'])}: " + ", ".join(rec["selected"])]]))
    parts.append("\n## Univariate screen vs the primary target (Kruskal-Wallis, BH q=%s)\n" % rep["univariate_screen"][primary][next(iter(rep["univariate_screen"][primary]))].get("fdr_q"))
    scr = rep["univariate_screen"][primary]
    rows = sorted(((f, r) for f, r in scr.items() if r["p"] is not None), key=lambda x: x[1]["p"])
    parts.append(_md_table(["feature", "H", "p", "pass_fdr", "n"], [[f, r["statistic"], r["p"], r["pass_fdr"], r["n"]] for f, r in rows]))
    st = rep.get("fold_stability") or {}
    if st:
        parts.append("\n## Stability across prefix folds (univariate screen)\n")
        parts.append(_md_table(["fold", "rank Spearman vs TRAIN", "passed"], [[f, v["rank_spearman_vs_train"], len(v["passed"])] for f, v in st["folds"].items()]))
        agree = st["pass_agreement"]
        parts.append("\nSelected features: FDR pass count over (folds + TRAIN) = " + ", ".join(f"{f}: {agree.get(f)}" for f in rec["selected"]))
    wd = rep["within_day_variance"]
    parts.append("\n## Within-day variance\n")
    const = [f for f, v in wd.items() if v["constant_within_day"]]
    parts.append(f"Constant within a day: {', '.join(const) or 'none'}. Selected features' within-day variance ratio: " +
                 ", ".join(f"{f}: {wd[f]['within_day_variance_ratio']:.3f}" for f in rec["selected"]))
    red = rep["redundancy"]
    parts.append("\n## Redundancy\n")
    parts.append("VIF of the selected set: " + ", ".join(f"{f}: {v:.2f}" for f, v in red["vif_selected"].items()))
    multi = [c for c in red["clusters"] if len(c["members"]) > 1]
    parts.append(f"\nTRAIN Spearman clusters (|rho| >= 0.9, average linkage): {len(red['clusters'])} clusters, {len(multi)} with > 1 member.")
    parts.append(_md_table(["cluster", "members"], [[c["cluster_id"], ", ".join(c["members"])] for c in multi]))
    fr = red["fractal_vs_realized_vol_spearman"]
    if fr:
        cols = sorted(next(iter(fr.values())).keys())
        parts.append("\nFractal columns vs realized-vol family (Spearman, TRAIN):\n")
        parts.append(_md_table(["fractal", *cols], [[f, *[v[c] for c in cols]] for f, v in fr.items()]))
    ca = rep.get("causal")
    if ca:
        parts.append("\n## Causal stage (causal-inference EURUSD_TIMING pack)\n")
        parts.append(f"History H = {ca['history']}; PCMCI+: " + "; ".join(f"{t}: {v.get('state')}" + (f" ({v.get('reason')})" if v.get('reason') else "") for t, v in ca["pcmci"].items()))
        for t, v in ca["hac_lag_screen"].items():
            rows = [r for r in v["rows"] if r.get("pass_fdr") and r["lag"] >= 1]
            rows.sort(key=lambda r: r["p"])
            parts.append(f"\n### HAC lag screen, target {t}: parents at lags 1..{k} = {len(v['parents_at_lags_1_to_k'])}\n")
            parts.append(_md_table(["feature", "lag", "coef", "t", "p"], [[r["feature"], r["lag"], r["coef"], r["t"], r["p"]] for r in rows[:40]]))
        for t, v in ca["pcmci"].items():
            if v.get("state") == "RUN":
                rows = [(f, r.get("verdict")) for f, r in v["records"].items()]
                parts.append(f"\n### PCMCI+ verdicts, target {t}\n")
                parts.append(_md_table(["feature", "verdict"], rows))
        mc = ca["matched_control_contrast"]
        rows = [(f, r["events"], r["event_pivot_rate"], r["control_pivot_rate"], r["excess_pivot_rate"], r["ci95"][0], r["ci95"][1], r["significant"])
                for f, r in mc.items() if r.get("state") == "ESTIMATED"]
        rows.sort(key=lambda r: -abs(r[4]))
        parts.append("\n### Matched-control contrast (q80 upward crossings vs matched controls; outcome = pivot within k)\n")
        parts.append(_md_table(["feature", "events", "event rate", "control rate", "excess", "ci lo", "ci hi", "significant"], rows[:40]))
    ev = out / "eval" / "SEAL.json"
    if ev.exists():
        seal = json.loads(ev.read_text())
        parts.append("\n## Predictive validation (pivot-detector harness; identical rows: %s)\n" % seal["identical_rows"])
        rows = [[s["set"], s["arm"], s["n_features"], s["threshold"], s["period"], s["auc_valley_mean"], s["auc_peak_mean"], s["log_loss_mean"],
                 s["log_loss_base_rate_mean"], s["precision_at_threshold_mean"], s["sharpe_daily_sqrt365_mean"], s["trades_per_week_mean"], s["valid_all"]]
                for s in seal["summary"] if s["period"] != "DECISION_2020_2022"]
        parts.append(_md_table(["set", "arm", "F", "tau", "period", "AUC valley", "AUC peak", "log-loss", "base-rate ll", "precision", "Sharpe", "orders/wk", "valid"], rows))
    text = "\n".join(parts) + "\n"
    (out / "TABLES.md").write_text(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    a = sub.add_parser("attach-eval")
    a.add_argument("--out", required=True)
    a.add_argument("--eval", required=True)
    c = sub.add_parser("causal")
    c.add_argument("--config", required=True)
    c.add_argument("--out", required=True)
    t = sub.add_parser("tables")
    t.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.verb == "tables":
        print(tables(args.out)[:400])
        return 0
    if args.verb == "causal":
        causal = causal_only(args.config, args.out)
        print(json.dumps({"pcmci": {t: v.get("state") for t, v in causal["pcmci"].items()},
                          "hac_parents": {t: len(v["parents_at_lags_1_to_k"]) for t, v in causal["hac_lag_screen"].items()}}))
        return 0
    if args.verb == "run":
        rep = run(args.config, args.out)
        print(json.dumps({"identity": rep["identity"], "selected": rep["recipe"]["selected"], "wall_seconds": rep["wall_seconds"]}))
    else:
        doc = attach_eval(args.out, args.eval)
        print(json.dumps({"manifest_sha256": doc["manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
