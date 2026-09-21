#!/usr/bin/env python3
"""RP71: the governed financial runner — FIN-LOSS-OPT executed as a population, on synthetic acceptance
fixtures in this round (no financial scientific selection is authorised).

What runs, in order, per unit under governance (one campaign and one bounded-range delivery per unit,
one terminal with artifacts per unit):

  prepare   the delivered bars (a CUT strictly before the reserve) -> strictly increasing labels -> targets by
            ELAPSED TIME (tools/df_fin_task.py) -> weekly fold identities -> per fold: admissible train /
            validation / test pairs with purges, the TRAIN-ONLY input scaler and target sigma, the Huber delta
            candidates from the train pairs; everything digested in FIN_DATA before any fit
  cell      one (fold, candidate, seed): the receiver frozen in the design, the loss (MAE or Huber at the fold's
            delta), the optimizer (Adam or AdamW with decoupled decay, biases excluded), validation every K observed
            updates with patience in events, restore best; predictions on the validation AND test pairs saved at
            full precision; the float64 floor at the observed MAE_z recorded (min_delta 0)
  select    per fold and loss family, the candidate with the lowest VALIDATION MAE_z; every candidate's test score
            is retained; paired differences across folds (temporal blocks) with a moving-block bootstrap interval
            and the tuning multiplicity declared
  close     arrays vs FIN_DATA pairs, reload parity, record vs arrays, warehouse terminal and artifacts by digest
"""

from __future__ import annotations

import argparse
import hashlib
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
SCHEMA = "df_fin_runner_design.v1"
SCHEMA_DATA = "df_fin_runner_data.v1"
RECIPE = {"batch": 64, "max_updates": 1500, "validate_every_updates": 100, "patience_events": 3, "min_delta": 0.0,
          "monitor": "validation MAE in scaled units (MAE_z with the fold's train sigma)", "restore_best": True}
LIMITS = {"child_cpu_seconds": 1800, "child_wall_seconds": 2400, "parallel_children": 2}


class FinRefusal(SystemExit):
    pass


def _module(name: str):
    import importlib.util
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


# --- the sealed design -----------------------------------------------------------------------------------------

def seal(*, lake: str, resource: str, time_column: str, holdout: str, range_from: str, range_to: str,
         receiver: str = "compact_modular", horizon: str = "h6", dev_weeks: int = 2, history_weeks: int = 4,
         candidate_ids: tuple = ("default_mae_adam", "default_huber_adam"), seeds=(1,), recipe: dict | None = None,
         contract=None, purpose: str = "FIN_LOSS_OPT_ACCEPTANCE", columns: list | None = None) -> dict:
    T = _module("df_fin_task")
    B = _module("df_benchmark_contract")
    alloc = T.candidate_allocation()
    by_id = {c["id"]: c for c in alloc["mae"] + alloc["huber"] + alloc["defaults"]}
    unknown = [c for c in candidate_ids if c not in by_id]
    if unknown:
        raise FinRefusal(f"REFUSED: unknown candidates {unknown}")
    recv = T.receivers()[receiver]
    ours = contract if contract is not None else B.fx_eurusd_1h_ours()
    hours = T.HORIZONS[horizon]["hours"]
    ours = B.replace(ours, horizon_steps=hours, horizon_seconds=hours*3600, input_window_steps=int(recv["window"]),
                     task_id=f"{ours.task_id.rsplit('.', 1)[0]}.{horizon}")
    design = {"schema": SCHEMA, "purpose": purpose, "phase": "DEVELOPMENT", "state": "SEALED_NOT_EXECUTED",
              "source": {"lake": lake, "resource": resource, "time_column": time_column, "holdout": holdout,
                         "range": {"from": range_from, "to": range_to}, "columns": columns or T.INPUT_COLUMNS, "target": T.TARGET},
              "horizon": {"id": horizon, "hours": hours, "rule": "the target is the bar labelled exactly origin + hours; absent -> excluded"},
              "folds": {"dev_weeks": dev_weeks, "history_weeks": history_weeks, "purge_hours": hours,
                        "rule": "weekly identities on the label's clock, strictly before the reserve"},
              "receiver": {"id": receiver, **recv}, "recipe": recipe or RECIPE, "limits": LIMITS,
              "candidates": [by_id[c] for c in candidate_ids], "allocation": alloc, "seeds": list(seeds),
              "cells": [{"cell_id": f"f{k}_{cid}_s{s}", "fold": k, "candidate_id": cid, "seed": s}
                        for k in range(dev_weeks) for cid in candidate_ids for s in seeds],
              "benchmark_contract": ours.to_design_block(comparability=B.planned_reference(
                  ours, why="no literature reference has been re-executed on this task; nothing is comparable yet")),
              "source_code": {n: sha_file(HERE/n) for n in ("df_fin_runner.py", "df_fin_task.py", "df_e1_block.py", "df_mod_e0.py", "df_e1_governed.py")},
              "what_this_is_not": "not a financial recipe selection; acceptance on synthetic bars unless the purpose says otherwise"}
    design["design_sha256"] = sha_obj(design)
    return design


def validate(design: dict):
    B = _module("df_benchmark_contract")
    body = {k: v for k, v in design.items() if k != "design_sha256"}
    if design.get("schema") != SCHEMA or sha_obj(body) != design.get("design_sha256"):
        raise FinRefusal("REFUSED: design digest/schema mismatch")
    for n, d in design["source_code"].items():
        if sha_file(HERE/n) != d:
            raise FinRefusal(f"REFUSED: scientific source changed: {n}")
    return B.require(design, purpose="the financial factorial")


# --- prepare ------------------------------------------------------------------------------------------------------

def calendar_of(ts_ns: np.ndarray) -> np.ndarray:
    import pandas as pd
    ts = pd.Series(pd.to_datetime(ts_ns))
    hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy()/60.0
    wd = ts.dt.dayofweek.to_numpy() + hour/24.0
    tp = 2*math.pi
    return np.stack([np.sin(tp*hour/24), np.cos(tp*hour/24), np.sin(tp*wd/7), np.cos(tp*wd/7)], axis=1)


def prepare(design: dict, root: Path, *, frame=None) -> dict:
    import pandas as pd
    T = _module("df_fin_task")
    B = _module("df_benchmark_contract")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    contract = validate(design)
    src = design["source"]
    delivered_sha = None
    if frame is None:
        G = _module("df_e1_governed")
        delivered = G.require_delivery(root, design, "prepare")["delivery"]
        if (delivered.get("range") or {}) != src["range"]:
            raise FinRefusal("REFUSED: the delivered range is not the design's")
        delivered_sha = delivered["sha256"]
        frame = pd.read_parquet(delivered["path"])
    bars = T.parse_bars(frame, time_column=src["time_column"], holdout=src["holdout"], columns=src["columns"], target=src["target"])
    ts, X, y = bars["ts_ns"], bars["X"], bars["y"]
    hours = int(design["horizon"]["hours"])
    mapping = T.map_targets(ts, y, hours=hours)
    folds = T.dev_folds(ts, holdout=src["holdout"], dev_weeks=int(design["folds"]["dev_weeks"]),
                        history_weeks=int(design["folds"]["history_weeks"]), purge_hours=int(design["folds"]["purge_hours"]))
    W = int(design["receiver"]["window"])
    cal = calendar_of(ts) if design["receiver"]["channels"] == 9 else None
    payload = {"ts_ns": ts, "X": X, "y": y, "target_channel": np.array([bars["target_channel"]]), "hours": np.array([hours]),
               "mapping_origins": mapping["origins"], "mapping_targets": mapping["targets"]}
    if cal is not None:
        payload["calendar"] = cal
    rec_folds = []
    for f in folds:
        pairs = T.fold_pairs(f, ts, mapping)
        lo, hi = f["bounds_ns"]["train"]
        train_rows = (ts >= lo) & (ts < hi) & np.isfinite(X).all(axis=1) & np.isfinite(y)
        if train_rows.sum() < W+1:
            raise FinRefusal(f"REFUSED: fold {f['fold']} has too few train bars for a window of {W}")
        mean, sd = X[train_rows].mean(axis=0), X[train_rows].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        y_mean, sigma = float(y[train_rows].mean()), float(y[train_rows].std())
        deltas = T.delta_candidates(y, pairs["train"], sigma=sigma)
        k = f["fold"]
        for split in ("train", "validation", "test"):
            o, t = pairs[split]["origins"], pairs[split]["targets"]
            keep = o >= W-1
            payload[f"f{k}_{split}_origins"], payload[f"f{k}_{split}_targets"] = o[keep], t[keep]
        payload[f"f{k}_scaler_mean"], payload[f"f{k}_scaler_sd"] = mean, sd
        payload[f"f{k}_target_mean_sigma"] = np.array([y_mean, sigma])
        rec_folds.append({**f, "pairs": {s: int(payload[f'f{k}_{s}_origins'].size) for s in ("train", "validation", "test")},
                          "train_bars": int(train_rows.sum()), "scaler": {"mean": mean.tolist(), "sd": sd.tolist(), "fitted_on": "train bars of the fold"},
                          "target_mean": y_mean, "sigma_train": sigma, "deltas": deltas})
    np.savez(root/"FIN_DATA.npz", **payload)
    # the typed contract binds to what was actually prepared (fold 0's scaler stands for the population's sd domain)
    data_json = {"input_columns": bars["columns"], "target_channel": bars["target_channel"], "horizon": hours, "window": W,
                 "panel_sha256": delivered_sha, "scaler": {"sd": rec_folds[0]["scaler"]["sd"]},
                 "enumerator": {"validation": {"admissible": rec_folds[0]["pairs"]["validation"]}}}
    data_json["scaler"]["sd"][bars["target_channel"]] = rec_folds[0]["sigma_train"]
    B.bind(contract, data_json, purpose="the financial factorial")
    rec = {"schema": SCHEMA_DATA, "design_sha256": design["design_sha256"], "delivered_sha256": delivered_sha, "bars": bars["n"],
           "availability": bars["availability"], "mapping": {k: v for k, v in mapping.items() if k not in ("origins", "targets")},
           "folds": rec_folds, "data_sha256": sha_file(root/"FIN_DATA.npz")}
    (root/"FIN_DATA.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def load_data(root: Path, design: dict) -> tuple:
    rec = json.loads((Path(root)/"FIN_DATA.json").read_text())
    if rec["design_sha256"] != design["design_sha256"] or sha_file(Path(root)/"FIN_DATA.npz") != rec["data_sha256"]:
        raise FinRefusal("REFUSED: FIN_DATA belongs to another design or was altered")
    with np.load(Path(root)/"FIN_DATA.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, rec


# --- one cell -----------------------------------------------------------------------------------------------------

class PairBatches:
    """Windows of the last W retained bars ending at each origin; the target is the pair's mapped bar (elapsed time)."""
    def __init__(self, Xs, yz, origins, targets, W, batch, *, shuffle, seed):
        self.Xs, self.yz, self.o, self.t, self.W, self.batch = Xs, yz, np.asarray(origins), np.asarray(targets), W, batch
        self.shuffle, self.seed, self.epoch = shuffle, int(seed), 0
        self.perm = np.arange(self.o.size)
        self._reshuffle()

    def _reshuffle(self):
        if self.shuffle:
            self.perm = np.random.default_rng([self.seed, self.epoch]).permutation(self.o.size)

    def __len__(self):
        return math.ceil(self.o.size/self.batch)

    def __getitem__(self, i):
        sel = self.perm[i*self.batch:(i+1)*self.batch]
        o = self.o[sel]
        idx = o[:, None] - self.W + 1 + np.arange(self.W)[None, :]
        return self.Xs[idx], self.yz[self.t[sel]].astype(np.float32)[:, None]

    def on_epoch_end(self):
        self.epoch += 1
        self._reshuffle()


def components(candidate: dict, delta_z, tf):
    if candidate["loss"] == "mae":
        loss = tf.keras.losses.MeanAbsoluteError()
    elif candidate["loss"] == "huber":
        if delta_z is None:
            raise FinRefusal("REFUSED: the Huber family is not identifiable in this fold (no delta)")
        loss = tf.keras.losses.Huber(delta=float(delta_z))
    else:
        raise FinRefusal(f"REFUSED: unknown loss {candidate['loss']!r}")
    kw = dict(learning_rate=float(candidate["lr"]), beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    if candidate["optimizer"] == "adam":
        opt = tf.keras.optimizers.Adam(**kw)
    elif candidate["optimizer"] == "adamw":
        opt = tf.keras.optimizers.AdamW(**kw, weight_decay=float(candidate["weight_decay"]))
        opt.exclude_from_weight_decay(var_names=["bias"])
    else:
        raise FinRefusal(f"REFUSED: unknown optimizer {candidate['optimizer']!r}")
    return loss, opt


def run_cell(design: dict, data: dict, rec: dict, cell: dict, out_dir: Path) -> dict:
    K = _module("df_e1_block")
    T = _module("df_fin_task")
    H = _module("df_e1_huber")
    E = _module("df_mod_e0")
    tf = E._tf()
    cpu0, wall0 = time.process_time(), time.monotonic()
    k, seed = int(cell["fold"]), int(cell["seed"])
    cand = next(c for c in design["candidates"] if c["id"] == cell["candidate_id"])
    fold = rec["folds"][k]
    R = design["recipe"]
    recv = design["receiver"]
    W, j = int(recv["window"]), int(data["target_channel"][0])
    X = data["X"]
    if recv["channels"] == 9:
        X = np.concatenate([X, data["calendar"]], axis=1)
    mean, sd = data[f"f{k}_scaler_mean"], data[f"f{k}_scaler_sd"]
    if recv["channels"] == 9:
        mean, sd = np.concatenate([mean, np.zeros(4)]), np.concatenate([sd, np.ones(4)])
    y_mean, sigma = (float(v) for v in data[f"f{k}_target_mean_sigma"])
    Xs = ((X-mean)/sd).astype(np.float32)
    yz = (data["y"]-y_mean)/sigma
    delta_z = T.resolve_delta(cand, fold["deltas"])
    loss, opt = components(cand, delta_z, tf)
    tr = PairBatches(Xs, yz, data[f"f{k}_train_origins"], data[f"f{k}_train_targets"], W, int(R["batch"]), shuffle=True, seed=seed)
    va = PairBatches(Xs, yz, data[f"f{k}_validation_origins"], data[f"f{k}_validation_targets"], W, int(R["batch"]), shuffle=False, seed=seed)
    te = PairBatches(Xs, yz, data[f"f{k}_test_origins"], data[f"f{k}_test_targets"], W, int(R["batch"]), shuffle=False, seed=seed)
    model = K.build_modular(recv["assignment"], W, X.shape[1], j, seed)
    initial = K.weight_hash(model)
    training = K.fit_by_updates(model, tr, va, max_updates=int(R["max_updates"]), validate_every=int(R["validate_every_updates"]),
                                patience=int(R["patience_events"]), lr=float(cand["lr"]), seed=seed, loss=loss, min_delta=float(R["min_delta"]),
                                optimizer=opt)
    out = {}
    y = data["y"]
    for split, ds in (("validation", va), ("test", te)):
        pred = K.predict(model, ds).astype(np.float64)*sigma + y_mean
        o, t = data[f"f{k}_{split}_origins"], data[f"f{k}_{split}_targets"]
        out[split] = {"pred": pred, "y": y[t], "naive": y[o], "origins": o, "targets": t,
                      "scores": H.metrics(pred, y[t], y[o], sigma) if o.size else None}
    out_dir.mkdir(parents=True, exist_ok=False)
    model.save_weights(out_dir/"weights.weights.h5")
    fresh = K.build_modular(recv["assignment"], W, X.shape[1], j, seed)
    fresh.load_weights(out_dir/"weights.weights.h5")
    reload_pred = K.predict(fresh, va).astype(np.float64)*sigma + y_mean
    np.testing.assert_allclose(out["validation"]["pred"], reload_pred, atol=1e-6, rtol=1e-6)
    arrays = {f"{s}_{kk}": v for s, d in out.items() for kk, v in d.items() if kk != "scores"}
    arrays["validation_reload_pred"] = reload_pred
    arrays["validation_ts_ns"] = data["ts_ns"][out["validation"]["origins"]]
    arrays["test_ts_ns"] = data["ts_ns"][out["test"]["origins"]]
    np.savez(out_dir/"arrays.npz", **arrays)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    val_mae_z = out["validation"]["scores"]["mae_z"]
    record = {"schema": "df_fin_runner_cell.v1", "cell": cell, "candidate": cand, "delta_z_used": delta_z, "fold_identity": {kk: fold[kk] for kk in ("fold", "test_week", "validation_week", "train_weeks")},
              "design_sha256": design["design_sha256"], "sigma_train": sigma, "target_mean": y_mean, "parameters": K.n_params(model),
              "initial_weights_sha256": initial, "final_weights_sha256": K.weight_hash(model), "training": training,
              "scores": {"validation": out["validation"]["scores"], "test": out["test"]["scores"]},
              "resolution": T.resolution_check(float(R["min_delta"]), 1e-6, val_mae_z),
              "reload_max_error": float(np.max(np.abs(out["validation"]["pred"]-reload_pred))),
              "cost": {"cpu_seconds": time.process_time()-cpu0, "wall_seconds": time.monotonic()-wall0, "peak_rss_bytes": int(ru.ru_maxrss)*1024,
                       "seconds_per_update": training["fit_cpu_seconds"]/max(1, training["updates"]), "host": os.uname().nodename},
              "arrays_sha256": sha_file(out_dir/"arrays.npz"), "weights_file_sha256": sha_file(out_dir/"weights.weights.h5")}
    (out_dir/"cell.json").write_text(json.dumps(record, indent=1, default=str))
    return record


def child(root: Path, unit: str) -> dict:
    design = json.loads((Path(root)/"DESIGN.json").read_text())
    validate(design)
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["child_cpu_seconds"], LIMITS["child_cpu_seconds"]+5))
    G = _module("df_e1_governed")
    G.require_delivery(root, design, unit)
    data, rec = load_data(root, design)
    cell = next(c for c in design["cells"] if c["cell_id"] == unit)
    return run_cell(design, data, rec, cell, Path(root)/"attempts"/unit)


# --- governance --------------------------------------------------------------------------------------------------------

def acquire(a, design, unit):
    G = _module("df_e1_governed")
    s = design["source"]
    return G.acquire(run_id=a.run_id, root=a.root, lake=s["lake"], resource=s["resource"], unit_id=unit, role="bars", gov_url=a.gov_url,
                     api_key_file=a.api_key_file, design_sha256=design["design_sha256"], cache_dir=Path(a.root)/"cache",
                     start=s["range"]["from"], end=s["range"]["to"])


def terminal_for(design, unit, cell, started, ok, rec, exit_code, wall, root: Path):
    U = _module("df_utility_run")
    cost = {"wall_seconds": wall}
    if rec:
        cost["cpu_seconds"] = rec["cost"]["cpu_seconds"]
    ms = []
    if rec:
        for split in ("validation", "test"):
            for k, v in (rec["scores"][split] or {}).items():
                if k != "rows":
                    ms.append(U._metric(f"fin.{split}.{k}", float(v), "z_train" if k.endswith("_z") else "dimensionless" if k.startswith("skill") else "target_units",
                                        split=split, horizon=int(design["horizon"]["hours"])))
    artifacts = [] if not rec else [{"role": r, "sha256": sha_file(root/"attempts"/unit/f), "bytes": (root/"attempts"/unit/f).stat().st_size}
                                    for r, f in (("predictions", "arrays.npz"), ("weights", "weights.weights.h5"), ("record", "cell.json"))]
    terminal = U._terminal(status="COMPLETED" if ok else "FAILED", reason=None if ok else f"child exited {exit_code}; see the retained log",
                           cost=cost, metrics=ms, started=started, finished=U._z(U.now_iso()),
                           tags={"purpose": design["purpose"], "classification": "NON_GOVERNING", "phase": "DEVELOPMENT", "unit": unit,
                                 "fold": str(cell["fold"]), "candidate": cell["candidate_id"], "seed": str(cell["seed"]),
                                 "design_sha256": design["design_sha256"], "contract_sha256": design["benchmark_contract"]["contract_sha256"]})
    terminal["artifacts"] = artifacts
    return terminal


def governance_modules():
    G = _module("df_e1_governed")
    G._load("governed_run")                     # complete the governance imports on the parent thread before parallel children
    G._load("df_e1_receipts")
    return G, _module("df_utility_run")


def run_units(a, design, units, *, parallel: int) -> list:
    G, U = governance_modules()
    root = Path(a.root)
    results = []

    def one(cell):
        unit = cell["cell_id"]
        if (root/"TERMINALS"/f"{unit}.json").exists():
            raise FinRefusal(f"REFUSED: {unit} already has a terminal")
        started = U._z(U.now_iso())
        acquire(a, design, unit)
        wall = time.monotonic()
        with open(root/f"{unit}.log", "x") as log:
            try:
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child", "--root", str(root), "--unit", unit],
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "TF_CPP_MIN_LOG_LEVEL": "3"},
                                      stdout=log, stderr=subprocess.STDOUT, timeout=LIMITS["child_wall_seconds"])
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = "WALL_TIMEOUT"
        rec_path = root/"attempts"/unit/"cell.json"
        ok = code == 0 and rec_path.exists()
        rec = json.loads(rec_path.read_text()) if ok else None
        terminal = terminal_for(design, unit, cell, started, ok, rec, code, time.monotonic()-wall, root)
        (root/"TERMINALS").mkdir(exist_ok=True)
        (root/"TERMINALS"/f"{unit}.json").write_text(json.dumps(terminal, indent=1, default=str))
        reported = G.report_terminal(root, unit, terminal, gov_url=a.gov_url, api_key_file=a.api_key_file, outbox_dir=str(root/"outbox"), started_at=started)
        if reported["flushed"]["pending"] or reported["flushed"]["failures"]:
            raise FinRefusal(f"REFUSED: the terminal of {unit} was not accepted: {reported['flushed']['failures']}")
        return {"unit": unit, "ok": ok, "record": rec, "terminal_sha256": (reported.get("receipt") or {}).get("terminal_sha256")}

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for start in range(0, len(units), parallel):
            batch = list(pool.map(one, units[start:start+parallel]))
            results += batch
            if not all(r["ok"] for r in batch):
                raise FinRefusal("REFUSED: a child failed; the remaining cells are not started")
    return results


# --- selection and inference over temporal blocks ------------------------------------------------------------------------

def block_bootstrap(values: np.ndarray, *, block_len: int, n_boot: int = 2000, seed: int = 0) -> dict:
    """Moving-block bootstrap of the mean over consecutive folds (temporal blocks); an iid resample beside it for the record."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    rng = np.random.default_rng(seed)
    if n < 2:
        return {"mean": float(v.mean()) if n else None, "interval_95": None, "n": int(n), "block_len": int(block_len), "why": "fewer than two blocks"}
    L = max(1, min(block_len, n))
    starts = np.arange(n-L+1)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(starts, size=math.ceil(n/L), replace=True)
        sample = np.concatenate([v[s:s+L] for s in pick])[:n]
        means.append(sample.mean())
    iid = [v[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return {"mean": float(v.mean()), "n": int(n), "block_len": int(L), "interval_95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            "iid_interval_95_for_the_record": [float(np.quantile(iid, .025)), float(np.quantile(iid, .975))],
            "signs": {"positive": int((v > 0).sum()), "negative": int((v < 0).sum()), "zero": int((v == 0).sum())},
            "reading": "a moving-block interval respects the week-to-week dependence; both signs are reported; n is the number of folds"}


def select(root: Path, design: dict) -> dict:
    root = Path(root)
    records = {}
    for c in design["cells"]:
        p = root/"attempts"/c["cell_id"]/"cell.json"
        if p.is_file():
            records[c["cell_id"]] = json.loads(p.read_text())
    families = sorted({c["loss"] for c in design["candidates"]})
    per_fold = {}
    for k in range(int(design["folds"]["dev_weeks"])):
        per_fold[k] = {}
        for fam in families:
            rows = [(cid, r) for cid, r in records.items() if r["cell"]["fold"] == k and r["candidate"]["loss"] == fam]
            if not rows:
                per_fold[k][fam] = {"status": "NOT_RUN"}
                continue
            best = min(rows, key=lambda x: x[1]["scores"]["validation"]["mae_z"])
            per_fold[k][fam] = {"selected": best[0], "by": "validation MAE_z", "validation_mae_z": best[1]["scores"]["validation"]["mae_z"],
                                "test_mae_z_of_selected": (best[1]["scores"]["test"] or {}).get("mae_z"),
                                "all_candidates_test_mae_z": {cid: (r["scores"]["test"] or {}).get("mae_z") for cid, r in rows},
                                "n_candidates_compared": len(rows)}
    paired = {}
    if len(families) == 2:
        f0, f1 = families
        diffs = [per_fold[k][f1]["test_mae_z_of_selected"] - per_fold[k][f0]["test_mae_z_of_selected"] for k in per_fold
                 if per_fold[k][f0].get("test_mae_z_of_selected") is not None and per_fold[k][f1].get("test_mae_z_of_selected") is not None]
        paired = {"difference": f"{f1} - {f0}, test MAE_z of each fold's validation-selected candidate", "values": diffs,
                  **block_bootstrap(np.asarray(diffs), block_len=2),
                  "multiplicity": {"candidates_per_family": {fam: len([c for c in design["candidates"] if c["loss"] == fam]) for fam in families},
                                   "reading": "each family's selection compared this many candidates on validation; the test difference is "
                                              "conditional on that selection and is not a confirmatory test"}}
    doc = {"schema": "df_fin_runner_selection.v1", "design_sha256": design["design_sha256"], "per_fold": per_fold, "paired": paired,
           "rule": "selection by validation MAE_z only; every candidate's test score retained; no best-test"}
    (root/"SELECTION.json").write_text(json.dumps(doc, indent=1, default=str))
    return doc


# --- closure ----------------------------------------------------------------------------------------------------------------

def close(a) -> dict:
    root = Path(a.root)
    design = json.loads((root/"DESIGN.json").read_text())
    validate(design)
    data, rec = load_data(root, design)
    C = _module("df_mod_e0_close")
    H = _module("df_e1_huber")
    receipts = json.loads((root/"TERMINAL_RECEIPTS.json").read_text())["units"] if (root/"TERMINAL_RECEIPTS.json").is_file() else {}
    token = Path(a.warehouse_token_file).read_text().strip().strip('"').strip("'") if getattr(a, "warehouse_token_file", None) else None
    problems, rows = [], []
    for cell in design["cells"]:
        unit = cell["cell_id"]
        folder = root/"attempts"/unit
        if not (folder/"cell.json").is_file():
            problems.append(f"{unit}: no record")
            continue
        r = json.loads((folder/"cell.json").read_text())
        k = cell["fold"]
        if sha_file(folder/"arrays.npz") != r["arrays_sha256"]:
            problems.append(f"{unit}: arrays digest mismatch")
        with np.load(folder/"arrays.npz", allow_pickle=False) as z:
            for split in ("validation", "test"):
                if not (np.array_equal(z[f"{split}_origins"], data[f"f{k}_{split}_origins"]) and np.array_equal(z[f"{split}_targets"], data[f"f{k}_{split}_targets"])):
                    problems.append(f"{unit}: {split} pairs are not the fold's")
                if z[f"{split}_origins"].size:
                    sc = H.metrics(z[f"{split}_pred"], z[f"{split}_y"], z[f"{split}_naive"], r["sigma_train"])
                    if sc != r["scores"][split]:
                        problems.append(f"{unit}: {split} record does not match arrays")
            if not np.allclose(z["validation_pred"], z["validation_reload_pred"], rtol=1e-6, atol=1e-6):
                problems.append(f"{unit}: reload parity")
        if unit not in receipts:
            problems.append(f"{unit}: no accepted terminal receipt")
        elif token:
            held = C.warehouse_terminals(a.warehouse_url, token, receipts[unit]["campaign_sha256"])["current"]
            row = held.get(unit, {})
            terminal = json.loads((root/"TERMINALS"/f"{unit}.json").read_text())
            if row.get("terminal_sha256") != receipts[unit]["terminal_sha256"] or row.get("status") != "COMPLETED":
                problems.append(f"{unit}: warehouse terminal digest/status")
            if sorted((x["role"], x["sha256"], x["bytes"]) for x in row.get("artifacts", [])) != sorted((x["role"], x["sha256"], x["bytes"]) for x in terminal["artifacts"]):
                problems.append(f"{unit}: warehouse artifacts")
            wanted = {(m["metric"], m["split"]): m["value"] for m in terminal["metrics"]}
            got = {(m.get("metric"), m.get("split")): m.get("value") for m in row.get("metrics", [])}
            if any(got.get(key) != val for key, val in wanted.items()):
                problems.append(f"{unit}: warehouse metric values differ from the terminal's (precision lost?)")
        rows.append({**cell, "validation_mae_z": r["scores"]["validation"]["mae_z"], "test_mae_z": (r["scores"]["test"] or {}).get("mae_z"),
                     "updates": r["training"]["updates"], "stop": r["training"]["stop_reason"], "delta_z": r["delta_z_used"]})
    selection = select(root, design)
    report = {"schema": "df_fin_runner_report.v1", "design_sha256": design["design_sha256"], "rows": rows, "selection": selection["per_fold"],
              "paired": selection["paired"], "problems": problems, "verified": not problems,
              "scope": design["purpose"] + "; DEVELOPMENT; synthetic acceptance unless the purpose says otherwise"}
    (root/"REPORT.json").write_text(json.dumps(report, indent=1, default=str))
    if problems:
        raise FinRefusal(f"REFUSED: closure failed: {problems[:5]}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "execute", "child", "close"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--unit")
    ap.add_argument("--run-id")
    ap.add_argument("--gov-url", default="http://127.0.0.1:5055")
    ap.add_argument("--api-key-file", type=Path)
    ap.add_argument("--warehouse-url", default="http://127.0.0.1:5057")
    ap.add_argument("--warehouse-token-file", type=Path)
    a = ap.parse_args(argv)
    design = json.loads((a.root/"DESIGN.json").read_text())
    if a.command == "child":
        child(a.root, a.unit)
        return 0
    if a.command == "prepare":
        validate(design)
        G, U = governance_modules()
        started = U._z(U.now_iso())
        acquire(a, design, "prepare")
        t0 = time.process_time()
        try:
            rec = prepare(design, a.root)
        except BaseException as exc:
            G.report_failed(a.root, "prepare", f"prepare refused: {str(exc)[:200]}", gov_url=a.gov_url, api_key_file=a.api_key_file)
            raise
        terminal = U._terminal(status="COMPLETED", reason=None, cost={"wall_seconds": time.process_time()-t0, "cpu_seconds": time.process_time()-t0},
                               metrics=[U._metric("fin.bars", rec["bars"], "rows", split="prepare", horizon=int(design["horizon"]["hours"]))],
                               started=started, finished=U._z(U.now_iso()),
                               tags={"purpose": design["purpose"], "classification": "NON_GOVERNING", "phase": "DEVELOPMENT", "unit": "prepare",
                                     "role": "PREPARATION", "design_sha256": design["design_sha256"]})
        terminal["artifacts"] = [{"role": r, "sha256": sha_file(a.root/f), "bytes": (a.root/f).stat().st_size} for r, f in (("data", "FIN_DATA.npz"), ("record", "FIN_DATA.json"))]
        (a.root/"TERMINALS").mkdir(exist_ok=True)
        (a.root/"TERMINALS"/"prepare.json").write_text(json.dumps(terminal, indent=1, default=str))
        reported = G.report_terminal(a.root, "prepare", terminal, gov_url=a.gov_url, api_key_file=a.api_key_file, outbox_dir=str(a.root/"outbox"), started_at=started)
        if reported["flushed"]["pending"] or reported["flushed"]["failures"]:
            raise FinRefusal(f"REFUSED: the prepare terminal was not accepted: {reported['flushed']['failures']}")
        print(json.dumps({k: v for k, v in rec.items() if k in ("bars", "mapping")}, indent=1, default=str))
        return 0
    if a.command == "execute":
        validate(design)
        load_data(a.root, design)
        run_units(a, design, design["cells"], parallel=LIMITS["parallel_children"])
        return 0
    close(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
