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
         candidate_ids: tuple = ("A_mae_adam", "A_huber_adam"), seeds=(1,), recipe: dict | None = None,
         contract=None, purpose: str = "FIN_LOSS_OPT_ACCEPTANCE", columns: list | None = None) -> dict:
    T = _module("df_fin_task")
    B = _module("df_benchmark_contract")
    alloc = T.candidate_allocation()
    by_id = {c["id"]: c for c in alloc["A_fixed_default"] + alloc["B_equal_budget_lr"] + alloc["C_decay_factor"] + alloc["D_delta_factor"]}
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
    delivery_meta = None
    if frame is not None and delivered_sha is None:
        delivery_meta = {"availability_use": "SYNTHETIC_FIXTURE_UNDECLARED"}
    else:
        delivery_meta = {k: delivered.get(k) for k in ("availability_use", "availability_label", "availability_contract_sha256")}
    bars = T.parse_bars(frame, time_column=src["time_column"], holdout=src["holdout"], columns=src["columns"], target=src["target"],
                        available_time_column=src.get("available_time_column"), timestamp_interpretation=src.get("timestamp_interpretation"),
                        availability=delivery_meta)
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
    min_pairs = int(design.get("folds", {}).get("min_pairs_per_split", 32))
    for f in folds:
        pairs = T.fold_pairs(f, ts, mapping)
        lo, hi = f["bounds_ns"]["train"]
        k = f["fold"]
        admissible, excluded = {}, {}
        for split in ("train", "validation", "test"):
            a = T.admissible_pairs(bars, pairs[split]["origins"], pairs[split]["targets"], W=W)      # the CONSUMED support, per pair
            admissible[split], excluded[split] = a, a["excluded"]
        if any(admissible[sp]["n"] < min_pairs for sp in ("train", "validation", "test")):
            rec_folds.append({**f, "status": "INSUFFICIENT_POPULATION", "pairs": {sp: admissible[sp]["n"] for sp in admissible},
                              "excluded": excluded, "why": f"a split has fewer than {min_pairs} admissible pairs; the fold produces NO score"})
            continue
        # the scaler and sigma come from the ADMISSIBLE train pairs' support rows only (finite by construction)
        tr_o = admissible["train"]["origins"]
        support = np.zeros(ts.size, dtype=bool)
        for w in range(W):
            support[tr_o - w] = True
        mean, sd = X[support].mean(axis=0), X[support].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        y_mean, sigma = float(y[support].mean()), float(y[support].std())
        deltas = T.delta_candidates(y, admissible["train"], sigma=sigma)                              # the SAME admissible train population
        for split in ("train", "validation", "test"):
            payload[f"f{k}_{split}_origins"], payload[f"f{k}_{split}_targets"] = admissible[split]["origins"], admissible[split]["targets"]
        payload[f"f{k}_scaler_mean"], payload[f"f{k}_scaler_sd"] = mean, sd
        payload[f"f{k}_target_mean_sigma"] = np.array([y_mean, sigma])
        rec_folds.append({**f, "status": "SCORABLE", "pairs": {sp: admissible[sp]["n"] for sp in admissible}, "excluded": excluded,
                          "train_support_rows": int(support.sum()), "scaler": {"mean": mean.tolist(), "sd": sd.tolist(), "fitted_on": "support rows of the admissible train pairs"},
                          "target_mean": y_mean, "sigma_train": sigma, "deltas": deltas,
                          "population_rule": "one shared admissible population per split for every candidate of the fold"})
    scorable = [f for f in rec_folds if f.get("status") == "SCORABLE"]
    if not scorable:
        raise FinRefusal("REFUSED: no fold has a sufficient admissible population; nothing is scored")
    np.savez(root/"FIN_DATA.npz", **payload)
    # the typed contract binds to what was actually prepared (fold 0's scaler stands for the population's sd domain)
    first = scorable[0]
    data_json = {"input_columns": bars["columns"], "target_channel": bars["target_channel"], "horizon": hours, "window": W,
                 "panel_sha256": delivered_sha, "scaler": {"sd": list(first["scaler"]["sd"])},
                 "enumerator": {"validation": {"admissible": first["pairs"]["validation"]}}}
    data_json["scaler"]["sd"][bars["target_channel"]] = first["sigma_train"]
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
        x, yz = self.Xs[idx], self.yz[self.t[sel]].astype(np.float32)[:, None]
        if not (np.isfinite(x).all() and np.isfinite(yz).all()):            # the last line: a non-finite tensor never reaches the model
            raise FinRefusal("REFUSED: a consumed window or label is not finite; the admissible population was not applied")
        return x, yz

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
    if fold.get("status") != "SCORABLE":
        raise FinRefusal(f"REFUSED: fold {k} is {fold.get('status')}: it produces no score")
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

MIN_BLOCKS = 5                       # declared minimum resampling support: distinct complete blocks
COVERAGE_CERTIFICATES = {}           # (n_positions, block_len) -> measured null coverage at 95 % nominal (RP85: predeclared, filled by tests/evidence)


def block_bootstrap(values, *, block_len: int, n_boot: int = 2000, seed: int = 0, min_blocks: int = MIN_BLOCKS) -> dict:
    """Moving-block bootstrap of the mean over CALENDAR positions (RP76/RP85). ESTIMAND: the mean paired effect over the
    USABLE week population = the observed weeks that can enter at least one complete block. An observed week that cannot
    enter any block is a NAMED support limitation: the result is then DESCRIPTIVE over all observed weeks (no interval),
    never an interval over a silently different population. Fewer than `min_blocks` complete blocks -> DESCRIPTIVE too."""
    v = np.asarray(values, dtype=np.float64)
    present = np.isfinite(v)
    n_present = int(present.sum())
    L = max(1, int(block_len))
    starts = [s0 for s0 in range(v.size-L+1) if present[s0:s0+L].all()]
    covered = np.zeros(v.size, dtype=bool)
    for s0 in starts:
        covered[s0:s0+L] = True
    isolated = [int(i) for i in np.flatnonzero(present & ~covered)]
    desc = {"estimand": "mean paired effect over the usable week population (weeks that can enter a complete block)",
            "mean_all_observed_weeks": float(v[present].mean()) if n_present else None, "n_folds_present": n_present, "n_positions": int(v.size),
            "block_len": L, "complete_blocks": len(starts), "isolated_observed_weeks": isolated,
            "signs": {"positive": int((v[present] > 0).sum()), "negative": int((v[present] < 0).sum()), "zero": int((v[present] == 0).sum())},
            "sample_sd_ddof1": float(v[present].std(ddof=1)) if n_present > 1 else None}
    if isolated:
        return {**desc, "mean": desc["mean_all_observed_weeks"], "interval_95": None, "status": "INSUFFICIENT_SUPPORT_ISOLATED_WEEKS",
                "why": f"observed weeks at positions {isolated} count in the mean but could enter no complete block of length {L}; "
                       "an interval over the remaining weeks would estimate another population",
                "reading": "descriptive only over all observed weeks"}
    if len(starts) < min_blocks or n_present < 2*L:
        return {**desc, "mean": desc["mean_all_observed_weeks"], "interval_95": None, "status": "INSUFFICIENT_RESAMPLING_SUPPORT",
                "why": f"{len(starts)} complete blocks of length {L} over {n_present} folds; at least {min_blocks} blocks and 2 x block length are required",
                "reading": "descriptive only: mean, signs and sample SD; no coverage is claimed"}
    rng = np.random.default_rng(seed)
    k = math.ceil(n_present/L)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(starts, size=k, replace=True)
        sample = np.concatenate([v[s0:s0+L] for s0 in pick])[:n_present]
        means.append(sample.mean())
    cert = COVERAGE_CERTIFICATES.get((int(v.size), L))
    return {**desc, "mean": desc["mean_all_observed_weeks"], "interval_95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            "status": "RESAMPLED" if cert and cert.get("confirmatory") else "RESAMPLED_DESCRIPTIVE",
            "n_boot": n_boot, "coverage_certificate": cert,
            "reading": ("a moving-block percentile interval over calendar positions; CONFIRMATORY only where a predeclared coverage certificate "
                        "for (positions, block length) exists (tests/evidence); otherwise a descriptive interval, not 95 % coverage")}


def select(root: Path, design: dict, *, verification: dict | None = None) -> dict:
    """RP84 (v4). Populations are kept apart: A = four fixed defaults, REPORTED; B = the same three LRs tuned SEPARATELY within
    EACH loss x optimizer combination, selection at configuration level (validation MAE_z averaged over the paired seeds);
    C = decay sensitivity contrasts paired to their B anchor (same loss, AdamW, default LR); D = delta sensitivity contrasts
    paired to their anchor (same optimizer, default LR). Nothing outside B enters B's search. A record is consumed only when
    its identity (design, candidate, fold, seed) agrees with the design cell, its scores are finite, it is not duplicated, and —
    when a verification is given — its custody verified."""
    root = Path(root)
    seeds = sorted(design.get("seeds") or [])
    T = _module("df_fin_task")
    alloc = T.candidate_allocation()
    pops = {c["id"]: c["population"] for key in ("A_fixed_default", "B_equal_budget_lr", "C_decay_factor", "D_delta_factor") for c in alloc[key]}
    candidates = {c["id"]: {**c, "population": c.get("population") or pops.get(c["id"], "UNKNOWN")} for c in design["candidates"]}
    verified = None if verification is None else set(verification.get("verified_units") or [])
    consumed, rejected, seen = {}, [], set()
    for c in design["cells"]:
        p = root/"attempts"/c["cell_id"]/"cell.json"
        if not p.is_file():
            continue
        r = json.loads(p.read_text())
        cell = r.get("cell") or {}
        cid = (r.get("candidate") or {}).get("id")
        key = (int(cell.get("fold", -1)), cid, int(cell.get("seed", -1)))
        why = None
        if cid not in candidates or cid != c["candidate_id"] or int(cell.get("seed", -1)) != int(c["seed"]) or int(cell.get("fold", -1)) != int(c["fold"]) \
                or r.get("design_sha256") not in (None, design.get("design_sha256")):
            why = "contradictory identity (design/candidate/fold/seed)"
        elif key in seen:
            why = "duplicate record"
        else:
            v = ((r.get("scores") or {}).get("validation") or {}).get("mae_z")
            t = ((r.get("scores") or {}).get("test") or {}).get("mae_z")
            if not (isinstance(v, (int, float)) and math.isfinite(v)) or (t is not None and not math.isfinite(t)):
                why = "non-finite score"
            elif verified is not None and c["cell_id"] not in verified:
                why = "custody not verified"
        if why:
            rejected.append({"unit": c["cell_id"], "why": why}); continue
        seen.add(key)
        consumed[key] = r
    # RP90 (Musashi RP89 #2): under a verification the fold population comes from its rows (the ACCEPTED preparation record);
    # the local FIN_DATA.json is consulted only when no verification was given (development use, declared as such)
    if verification is not None:
        folds_rec = {int(r["fold"]): ("FOLD_NOT_SCORABLE" if r.get("status") == "FOLD_NOT_SCORABLE" else "SCORABLE") for r in verification.get("rows") or []}
    else:
        folds_rec = {f["fold"]: f.get("status", "SCORABLE") for f in json.loads((root/"FIN_DATA.json").read_text())["folds"]} if (root/"FIN_DATA.json").is_file() else {}
    n_folds = int(design["folds"]["dev_weeks"])
    strata = [(l, o) for l in ("mae", "huber") for o in ("adam", "adamw")]

    def config(k, cid):
        cells = {s: consumed.get((k, cid, s)) for s in seeds}
        if not seeds or any(cells[s] is None for s in seeds):
            return {"status": "INCOMPLETE", "seeds_present": [s for s in seeds if cells[s]]}
        vals = [cells[s]["scores"]["validation"]["mae_z"] for s in seeds]
        return {"status": "COMPLETE", "validation_mae_z_by_seed": {str(s): x for s, x in zip(seeds, vals)}, "validation_mean_over_seeds": float(np.mean(vals)),
                "test_mae_z_by_seed": {str(s): (cells[s]["scores"]["test"] or {}).get("mae_z") for s in seeds}}

    per_fold, counts = {}, {"A": 0, "B": 0, "C": 0, "D": 0}
    for k in range(n_folds):
        status = folds_rec.get(k, "SCORABLE")
        per_fold[k] = {"status": status}
        if status != "SCORABLE":
            per_fold[k]["why"] = "no score in this fold; its calendar position is kept as a gap"
            continue
        # A: reported, never selected
        A = {cid: config(k, cid) for cid, c in candidates.items() if c["population"] == "A_fixed_default"}
        counts["A"] += sum(1 for a in A.values() if a["status"] == "COMPLETE")
        # B: within each (loss, optimizer) stratum
        B = {}
        for l, o in strata:
            cands = {cid: config(k, cid) for cid, c in candidates.items() if c["population"] == "B_equal_budget_lr" and c["loss"] == l and c["optimizer"] == o}
            complete = {cid: x for cid, x in cands.items() if x["status"] == "COMPLETE"}
            counts["B"] += len(complete)
            if not complete:
                B[f"{l}_{o}"] = {"status": "NOT_RUN_OR_INCOMPLETE", "configs": cands}; continue
            best = min(complete, key=lambda cid: complete[cid]["validation_mean_over_seeds"])
            B[f"{l}_{o}"] = {"selected": best, "by": "validation MAE_z averaged over the paired seeds, within this loss x optimizer stratum",
                             "n_candidates_compared": len(complete), "lrs_compared": sorted(candidates[cid]["lr"] for cid in complete),
                             "validation_mean_over_seeds": complete[best]["validation_mean_over_seeds"],
                             "test_mae_z_by_seed_of_selected": complete[best]["test_mae_z_by_seed"], "configs": cands}
        # C and D: paired sensitivity contrasts against their anchors (never competitors)
        def anchor(loss, opt):
            cid = next((c for c, x in candidates.items() if x["population"] == "B_equal_budget_lr" and x["loss"] == loss and x["optimizer"] == opt
                        and abs(x["lr"]-T.DEFAULTS[opt]["lr"]) < 1e-12), None)
            return cid, (config(k, cid) if cid else {"status": "INCOMPLETE"})
        CD = {}
        for cid, c in candidates.items():
            if c["population"] not in ("C_decay_factor", "D_delta_factor"):
                continue
            me = config(k, cid)
            a_id, a_cfg = anchor(c["loss"], c["optimizer"])
            counts[c["population"][0]] += 1 if me["status"] == "COMPLETE" else 0
            if me["status"] != "COMPLETE" or a_cfg.get("status") != "COMPLETE":
                CD[cid] = {"status": "INCOMPLETE", "anchor": a_id}; continue
            d = {s: me["test_mae_z_by_seed"][s]-a_cfg["test_mae_z_by_seed"][s] for s in me["test_mae_z_by_seed"]}
            CD[cid] = {"status": "PAIRED_CONTRAST", "anchor": a_id, "factor": c["population"], "test_difference_by_seed": d,
                       "validation_difference_by_seed": {s: me["validation_mae_z_by_seed"][s]-a_cfg["validation_mae_z_by_seed"][s] for s in me["validation_mae_z_by_seed"]}}
        per_fold[k].update({"A_fixed_default": A, "B_equal_budget_lr": B, "CD_sensitivity_contrasts": CD})
    # primary contrasts from B: loss within optimizer and optimizer within loss, paired by (fold, seed)
    contrasts = {}
    for name, (x, y) in {"huber_minus_mae_adam": ("huber_adam", "mae_adam"), "huber_minus_mae_adamw": ("huber_adamw", "mae_adamw"),
                         "adamw_minus_adam_mae": ("mae_adamw", "mae_adam"), "adamw_minus_adam_huber": ("huber_adamw", "huber_adam")}.items():
        by_seed = {str(s): [] for s in seeds}
        for k in range(n_folds):
            bx, by = (per_fold[k].get("B_equal_budget_lr") or {}).get(x, {}), (per_fold[k].get("B_equal_budget_lr") or {}).get(y, {})
            for s in seeds:
                tx, ty = (bx.get("test_mae_z_by_seed_of_selected") or {}).get(str(s)), (by.get("test_mae_z_by_seed_of_selected") or {}).get(str(s))
                by_seed[str(s)].append(tx-ty if (tx is not None and ty is not None) else float("nan"))
        contrasts[name] = {"by_seed": {s: {"values_by_fold_position": [None if not np.isfinite(v) else v for v in vals], **block_bootstrap(np.asarray(vals), block_len=int(design.get("inference", {}).get("block_len", 2)))} for s, vals in by_seed.items()},
                           "reading": "test MAE_z difference between the two strata's B-selected configurations, paired by fold and seed; seeds retained as replicates"}
    doc = {"schema": "df_fin_runner_selection.v4", "design_sha256": design["design_sha256"], "per_fold": per_fold, "contrasts": contrasts,
           "consumed": {"records": len(consumed), "by_population_complete_configs": counts, "strata": [f"{l}_{o}" for l, o in strata], "rejected": rejected},
           "rule": "A reported; B searched within each loss x optimizer over the same LRs at configuration level; C/D paired sensitivity "
                   "contrasts; identity from records; no best-of-seeds, no best-test, no cross-population competition"}
    (root/"SELECTION.json").write_text(json.dumps(doc, indent=1, default=str))
    return doc


# --- closure ----------------------------------------------------------------------------------------------------------------

def close(a) -> dict:
    """RP82: the financial closure CONSUMES tools/df_closure_table.verify_fin_run (the same authority as the table and the
    block closure); selection runs only on verified cells; a failed closure emits no selection and no contrast."""
    root = Path(a.root)
    design = json.loads((root/"DESIGN.json").read_text())
    validate(design)
    T = _module("df_closure_table")
    C = _module("df_mod_e0_close")
    token = Path(a.warehouse_token_file).read_text().strip().strip('"').strip("'") if getattr(a, "warehouse_token_file", None) else None
    warehouse = (lambda campaign: C.warehouse_terminals(a.warehouse_url, token, campaign)) if token else None
    ver = T.verify_fin_run(root, warehouse=warehouse)
    problems = list(ver["problems"])
    if warehouse is None:
        problems.append("closure without a warehouse read: no accepted custody, nothing is verified")
    scorable = [r for r in ver["rows"] if r.get("status") != "FOLD_NOT_SCORABLE"]
    verified_all = not problems and scorable and all(r["verified"] for r in scorable)
    B = _module("df_benchmark_contract")
    disposition = B.disposition(design)
    report = {"schema": "df_fin_runner_report.v3", "design_sha256": design["design_sha256"],
              # RP90: the financial task is a deferred mandatory stage (or, for a synthetic/household design, historical);
              # a selection below is the declared factorial's bookkeeping, never an active recommendation
              "disposition": disposition, "active_selection": None,
              "verification": {k: ver[k] for k in ("design_identity", "preparation_custody", "verified_units", "unverified_units", "required_folds_from")},
              "rows": [{"unit": r["unit"], "fold": r["fold"], "candidate_id": r.get("candidate_id"), "seed": r.get("seed"),
                        "validation_mae_z": (r.get("scores") or {}).get("validation", {}).get("mae_z"), "test_mae_z": (r.get("scores") or {}).get("test", {}).get("mae_z"),
                        "verified": r["verified"], "custody": r.get("custody")} for r in ver["rows"]],
              "problems": problems, "verified": bool(verified_all),
              "scope": design["purpose"] + "; DEVELOPMENT; synthetic acceptance unless the purpose says otherwise"}
    if verified_all:
        selection = select(root, design, verification=ver)
        report["selection"] = selection["per_fold"]
        report["contrasts"] = selection["contrasts"]
        report["consumed"] = selection["consumed"]
    else:
        report["selection"] = None
        report["contrasts"] = None
        report["reading"] = "closure FAILED: no selection, no contrast and no proposal are emitted"
    (root/"REPORT.json").write_text(json.dumps(report, indent=1, default=str))
    if not verified_all:
        raise FinRefusal(f"REFUSED: closure failed: {problems[:5]}")
    return report


def run_prepare(a, design: dict) -> dict:
    """Acquire the prepare unit's bounded-range delivery, prepare, and close the prepare unit with a terminal whose artifacts
    are the prepared data's digests (the accepted preparation evidence the verifier binds to)."""
    validate(design)
    G, U = governance_modules()
    root = Path(a.root)
    started = U._z(U.now_iso())
    acquire(a, design, "prepare")
    t0 = time.process_time()
    try:
        rec = prepare(design, root)
    except BaseException as exc:
        G.report_failed(root, "prepare", f"prepare refused: {str(exc)[:200]}", gov_url=a.gov_url, api_key_file=a.api_key_file)
        raise
    terminal = U._terminal(status="COMPLETED", reason=None, cost={"wall_seconds": time.process_time()-t0, "cpu_seconds": time.process_time()-t0},
                           metrics=[U._metric("fin.bars", rec["bars"], "rows", split="prepare", horizon=int(design["horizon"]["hours"]))],
                           started=started, finished=U._z(U.now_iso()),
                           tags={"purpose": design["purpose"], "classification": "NON_GOVERNING", "phase": "DEVELOPMENT", "unit": "prepare",
                                 "role": "PREPARATION", "design_sha256": design["design_sha256"]})
    terminal["artifacts"] = [{"role": r, "sha256": sha_file(root/f), "bytes": (root/f).stat().st_size} for r, f in (("data", "FIN_DATA.npz"), ("record", "FIN_DATA.json"))]
    (root/"TERMINALS").mkdir(exist_ok=True)
    (root/"TERMINALS"/"prepare.json").write_text(json.dumps(terminal, indent=1, default=str))
    reported = G.report_terminal(root, "prepare", terminal, gov_url=a.gov_url, api_key_file=a.api_key_file, outbox_dir=str(root/"outbox"), started_at=started)
    if reported["flushed"]["pending"] or reported["flushed"]["failures"]:
        raise FinRefusal(f"REFUSED: the prepare terminal was not accepted: {reported['flushed']['failures']}")
    print(json.dumps({k: v for k, v in rec.items() if k in ("bars", "mapping")}, indent=1, default=str))
    return rec


# --- RP88: the train-only cost diagnostic -----------------------------------------------------------------------------

COST_PILOT = {"max_updates": 200, "validate_every_updates": 50, "patience_events": 3, "batch": 64, "min_delta": 0.0,
              "purpose": "COST_DIAGNOSTIC_TRAIN_ONLY: seconds, memory and samples; NOT adequacy, convergence or a winner"}


def seal_cost_pilot(*, lake: str, resource: str, time_column: str, holdout: str, range_from: str, range_to: str, dev_start: str,
                    receivers=("compact_modular", "larger_business_receiver"), horizons=("h6", "h72"), columns: list | None = None,
                    contract=None, purpose: str = "FIN_LOSS_OPT_COST_PILOT") -> dict:
    """The pre-DEV training slice, the receivers, the horizons and the four fixed-default recipes, sealed BEFORE any byte is read.
    The slice ends before `dev_start` (the first DEV week): the DEV weeks and the reserve are never read by the pilot."""
    T = _module("df_fin_task")
    B = _module("df_benchmark_contract")
    if not (range_to < dev_start):
        raise FinRefusal("REFUSED: the cost pilot's slice must end before the first DEV week")
    alloc = T.candidate_allocation()
    recv = T.receivers()
    ours = contract if contract is not None else B.fx_eurusd_1h_ours()
    configs = [{"config_id": f"{r}__{h}__{c['id']}", "receiver": r, "horizon": h, "candidate": c}
               for r in receivers for h in horizons for c in alloc["A_fixed_default"]]
    design = {"schema": "df_fin_cost_pilot_design.v1", "purpose": purpose, "phase": "DEVELOPMENT", "state": "SEALED_NOT_EXECUTED",
              "source": {"lake": lake, "resource": resource, "time_column": time_column, "holdout": holdout, "range": {"from": range_from, "to": range_to},
                         "columns": columns or T.INPUT_COLUMNS, "target": T.TARGET, "dev_start": dev_start,
                         "rule": "pre-DEV training slice only; internal purged validation = the last week of the slice; DEV and reserve never read"},
              "receivers": {r: recv[r] for r in receivers}, "horizons": {h: T.HORIZONS[h] for h in horizons},
              "recipe": COST_PILOT, "configs": configs, "cells": [{"cell_id": c["config_id"], **c} for c in configs],
              "history_alternatives_weeks": [52, 104, 208],
              "benchmark_contract": ours.to_design_block(comparability=B.planned_reference(ours, why="cost diagnostic; no comparator involved")),
              "source_code": {n: sha_file(HERE/n) for n in ("df_fin_runner.py", "df_fin_task.py", "df_e1_block.py", "df_mod_e0.py", "df_e1_governed.py")},
              "what_this_is_not": "not a scientific comparison, not model adequacy, not a winner; 200 observed updates per configuration"}
    design["design_sha256"] = sha_obj(design)
    return design


def cost_pilot(design: dict, root: Path, *, frame=None, delivered: dict | None = None) -> dict:
    """RP88: on the delivered pre-DEV slice, for every (receiver, horizon, default recipe): setup, warm-up, train updates, validation,
    replay and peak RSS measured SEPARATELY; usable examples counted from the admissible populations; projections for the full
    scientific allocation are arithmetic on these measurements (declared as such)."""
    import pandas as pd
    T = _module("df_fin_task")
    K = _module("df_e1_block")
    E = _module("df_mod_e0")
    tf = E._tf()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    src = design["source"]
    if frame is None:
        G = _module("df_e1_governed")
        delivered = G.require_delivery(root, design, "prepare")["delivery"]
        frame = pd.read_parquet(delivered["path"])
    try:
        bars = T.parse_bars(frame, time_column=src["time_column"], holdout=src["holdout"], columns=src["columns"], target=src["target"])
    except T.TaskRefusal as exc:
        doc = {"schema": "df_fin_cost_pilot.v1", "design_sha256": design["design_sha256"], "status": "REFUSED_AT_SCHEMA", "reason": str(exc)[:400],
               "served_columns": list(frame.columns), "rows": int(len(frame)), "reading": "the served schema differs from the declared one: the design is "
               "amended before any fit; nothing was measured"}
        (root/"COST_PILOT.json").write_text(json.dumps(doc, indent=1, default=str))
        return doc
    ts, X, y = bars["ts_ns"], bars["X"], bars["y"]
    limit = np.datetime64(src["dev_start"]).astype("datetime64[ns]").astype(np.int64)
    if ts.max() >= limit:
        raise FinRefusal("REFUSED: the delivered slice reaches the DEV weeks")
    val_start = T.week_start_ns(int(ts.max())) if T.week_start_ns(int(ts.max())) > ts.min() else int(ts.max())
    R = design["recipe"]
    results, cpu0 = [], time.process_time()
    for cfg in design["configs"]:
        recv = design["receivers"][cfg["receiver"]]
        hours = design["horizons"][cfg["horizon"]]["hours"]
        W = int(recv["window"])
        m = T.map_targets(ts, y, hours=hours)
        purge = hours*T.HOUR_NS
        train_keep = (ts[m["targets"]] < val_start - purge)
        val_keep = (ts[m["origins"]] >= val_start)
        tr_p = T.admissible_pairs(bars, m["origins"][train_keep], m["targets"][train_keep], W=W)
        va_p = T.admissible_pairs(bars, m["origins"][val_keep], m["targets"][val_keep], W=W)
        Xc = X if recv["channels"] == 5 else np.concatenate([X, calendar_of(ts)], axis=1)
        entry = {"config_id": cfg["config_id"], "receiver": cfg["receiver"], "horizon": cfg["horizon"], "candidate": cfg["candidate"]["id"],
                 "samples": {"train_pairs": tr_p["n"], "validation_pairs": va_p["n"], "train_excluded": tr_p["excluded"], "bars": bars["n"]}}
        if tr_p["n"] < 2*R["batch"] or va_p["n"] < 8:
            entry.update(status="INSUFFICIENT_SAMPLES"); results.append(entry); continue
        t0 = time.process_time()
        support = np.zeros(ts.size, dtype=bool)
        for w in range(W):
            support[tr_p["origins"]-w] = True
        mean, sd = Xc[support].mean(axis=0), np.where(Xc[support].std(axis=0) > 0, Xc[support].std(axis=0), 1.0)
        y_mean, sigma = float(y[support].mean()), float(y[support].std())
        deltas = T.delta_candidates(y, tr_p, sigma=sigma)
        Xs = ((Xc-mean)/sd).astype(np.float32); yz = (y-y_mean)/sigma
        delta_z = T.resolve_delta(cfg["candidate"], deltas)
        try:
            loss, opt = components(cfg["candidate"], delta_z, tf)
        except FinRefusal as exc:
            entry.update(status="NOT_IDENTIFIABLE", reason=str(exc)[:200]); results.append(entry); continue
        tr = PairBatches(Xs, yz, tr_p["origins"], tr_p["targets"], W, int(R["batch"]), shuffle=True, seed=1)
        va = PairBatches(Xs, yz, va_p["origins"], va_p["targets"], W, int(R["batch"]), shuffle=False, seed=1)
        model = K.build_modular(recv["assignment"], W, Xc.shape[1], bars["target_channel"], 1)
        setup = time.process_time()-t0
        model.compile(optimizer=opt, loss=loss)
        t0 = time.process_time(); xb, yb = tr[0]; model.train_on_batch(xb, yb); warm = time.process_time()-t0      # graph tracing, measured apart
        model = K.build_modular(recv["assignment"], W, Xc.shape[1], bars["target_channel"], 1)                      # a fresh model for the timed loop
        loss, opt = components(cfg["candidate"], delta_z, tf)
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        training = K.fit_by_updates(model, tr, va, max_updates=int(R["max_updates"]), validate_every=int(R["validate_every_updates"]),
                                    patience=int(R["patience_events"]), lr=float(cfg["candidate"]["lr"]), seed=1, loss=loss, min_delta=0.0, optimizer=opt)
        t0 = time.process_time(); K.predict(model, va); replay = time.process_time()-t0
        cpu = training["cpu"]
        entry.update(status="MEASURED", parameters=K.n_params(model), delta_z=delta_z,
                     cpu={"setup_seconds": setup, "warm_up_seconds": warm, "train_update_seconds": cpu["train_update_seconds"], "validation_seconds": cpu["validation_seconds"],
                          "restore_seconds": cpu["restore_seconds"], "replay_predict_seconds": replay, "updates": training["updates"], "validation_events": training["validation_events"],
                          "seconds_per_update": cpu["train_update_seconds"]/max(1, training["updates"]),
                          "validation_seconds_per_event": cpu["validation_seconds"]/max(1, training["validation_events"]),
                          "validation_seconds_per_sample": cpu["validation_seconds"]/max(1, training["validation_events"]*va_p["n"])},
                     peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024, stop=training["stop_reason"], triggers=training["triggers"],
                     val_mae_z_first_last=[training["events"][0]["val_mae_scaled"], training["events"][-1]["val_mae_scaled"]] if training["events"] else None)
        results.append(entry)
    measured = [r for r in results if r["status"] == "MEASURED"]
    # projections: arithmetic on the measurements, for the scientific recipe (RECIPE) and the 26-population allocation
    proj = {}
    for r in measured:
        events = math.ceil(RECIPE["max_updates"]/RECIPE["validate_every_updates"])
        per_cell = r["cpu"]["seconds_per_update"]*RECIPE["max_updates"] + r["cpu"]["validation_seconds_per_event"]*events + r["cpu"]["setup_seconds"] + r["cpu"]["warm_up_seconds"] + r["cpu"]["replay_predict_seconds"]
        proj[r["config_id"]] = {"seconds_per_full_cell_at_ceiling": per_cell}
    by_rh = {}
    for r in measured:
        by_rh.setdefault((r["receiver"], r["horizon"]), []).append(proj[r["config_id"]]["seconds_per_full_cell_at_ceiling"])
    allocation = {}
    for (rc, hz), secs in by_rh.items():
        mean_cell = float(np.mean(secs))
        for folds in (26,):
            for hist in design["history_alternatives_weeks"]:
                n_cells = 26*3*folds
                allocation[f"{rc}__{hz}__folds{folds}__history{hist}w"] = {"cells": n_cells, "mean_seconds_per_cell": mean_cell, "cpu_seconds_at_ceiling": mean_cell*n_cells,
                                                                            "usable_train_pairs_estimate": int(np.mean([r["samples"]["train_pairs"] for r in measured if r["receiver"] == rc and r["horizon"] == hz])*hist/52),
                                                                            "history_note": "pairs scale linearly with history weeks from the delivered 52-week slice: an ESTIMATE, not a count"}
    doc = {"schema": "df_fin_cost_pilot.v1", "design_sha256": design["design_sha256"], "status": "MEASURED" if measured else "NOTHING_MEASURED",
           "slice": {"bars": bars["n"], "first": str(np.datetime64(int(ts.min()), "ns"))[:16], "last": str(np.datetime64(int(ts.max()), "ns"))[:16],
                     "internal_validation_from": str(np.datetime64(val_start, "ns"))[:16], "availability": bars["availability"]},
           "configs": results, "projection_per_config": proj, "scientific_allocation_projection": allocation,
           "total_pilot_cpu_seconds": time.process_time()-cpu0,
           "reading": "cost, memory and samples of 200 observed updates per configuration on the pre-DEV slice; no adequacy, convergence or winner; "
                      "a patience of 300 updates is not adequate training because the dataset is large — adequacy is judged by learning curves in the scientific run"}
    (root/"COST_PILOT.json").write_text(json.dumps(doc, indent=1, default=str))
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "execute", "child", "close", "cost-pilot"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--unit")
    ap.add_argument("--run-id")
    ap.add_argument("--gov-url", default="http://127.0.0.1:5055")
    ap.add_argument("--api-key-file", type=Path)
    ap.add_argument("--warehouse-url", default="http://127.0.0.1:5057")
    ap.add_argument("--warehouse-token-file", type=Path)
    a = ap.parse_args(argv)
    design = json.loads((a.root/"DESIGN.json").read_text())
    if a.command == "cost-pilot":
        # the prepare unit's delivery (bounded range) and one terminal for the whole diagnostic, with the measured cost
        G, U = governance_modules()
        started = U._z(U.now_iso())
        acquire(a, design, "prepare")
        t0 = time.process_time()
        try:
            doc = cost_pilot(design, a.root)
        except BaseException as exc:
            G.report_failed(a.root, "prepare", f"cost pilot refused: {str(exc)[:200]}", gov_url=a.gov_url, api_key_file=a.api_key_file)
            raise
        # RP90 (Musashi RP89 #3): a typed refusal returned as a document (REFUSED_AT_SCHEMA, NOTHING_MEASURED) is NOT a completed
        # diagnostic: the unit closes FAILED with the reason, under its own campaign identity, and the command exits non-zero
        if doc.get("status") != "MEASURED":
            reported = G.report_failed(a.root, "prepare", f"cost pilot {doc.get('status')}: {str(doc.get('reason') or doc.get('reading'))[:160]}",
                                       gov_url=a.gov_url, api_key_file=a.api_key_file, outbox_dir=str(a.root/"outbox"))
            print(json.dumps({"status": doc.get("status"), "closed_as": "FAILED", "reason": doc.get("reason"), "sent": reported["flushed"]["sent"],
                              "pending": reported["flushed"]["pending"], "failures": reported["flushed"]["failures"]}))
            raise FinRefusal(f"REFUSED: the cost pilot did not measure ({doc.get('status')}): {str(doc.get('reason'))[:200]}; "
                             f"the unit was closed FAILED (sent {reported['flushed']['sent']}, pending {reported['flushed']['pending']})")
        ru = resource.getrusage(resource.RUSAGE_SELF)
        terminal = U._terminal(status="COMPLETED", reason=None, cost={"wall_seconds": time.process_time()-t0, "cpu_seconds": ru.ru_utime+ru.ru_stime},
                               metrics=[U._metric("fin.cost_pilot.configs_measured", sum(1 for c in doc.get("configs", []) if c.get("status") == "MEASURED"), "configs", split="train_only", horizon=0)],
                               started=started, finished=U._z(U.now_iso()),
                               tags={"purpose": design["purpose"], "classification": "NON_GOVERNING", "phase": "DEVELOPMENT", "unit": "prepare", "role": "COST_PILOT",
                                     "design_sha256": design["design_sha256"], "status": doc.get("status")})
        terminal["artifacts"] = [{"role": "record", "sha256": sha_file(a.root/"COST_PILOT.json"), "bytes": (a.root/"COST_PILOT.json").stat().st_size}]
        (a.root/"TERMINALS").mkdir(exist_ok=True); (a.root/"TERMINALS"/"prepare.json").write_text(json.dumps(terminal, indent=1, default=str))
        reported = G.report_terminal(a.root, "prepare", terminal, gov_url=a.gov_url, api_key_file=a.api_key_file, outbox_dir=str(a.root/"outbox"), started_at=started)
        print(json.dumps({"status": doc.get("status"), "configs": len(doc.get("configs", [])), "cpu": doc.get("total_pilot_cpu_seconds"), "sent": reported["flushed"]["sent"],
                          "pending": reported["flushed"]["pending"], "failures": reported["flushed"]["failures"]}))
        if reported["flushed"]["pending"] or reported["flushed"]["failures"]:
            raise FinRefusal(f"REFUSED: the cost-pilot terminal was not accepted (pending {reported['flushed']['pending']}): {reported['flushed']['failures']}")
        return 0
    if a.command == "child":
        child(a.root, a.unit)
        return 0
    if a.command == "prepare":
        run_prepare(a, design)
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
