#!/usr/bin/env python3
"""Paired DEVELOPMENT factorial, with governed deliveries and a common MAE monitor."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("df_e1_phase1", HERE / "df_e1_phase1.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
RECIPES = {f"{loss}_{opt}": {"loss": loss, "optimizer": opt, "monitor": "val_mae",
                           "delta": 1., "weight_decay": .004 if opt == "adamw" else 0.}
           for loss in ("mae", "huber") for opt in ("adam", "adamw")}


def write(path, value):
    P._module("df_d3_campaign").write_once(Path(path), value)


def components(recipe, lr):
    if recipe not in RECIPES.values():
        raise ValueError("unknown or changed factorial recipe")
    tf = P._module("df_mod_e0")._tf()
    loss = tf.keras.losses.Huber(delta=recipe["delta"]) if recipe["loss"] == "huber" else "mae"
    kw = dict(learning_rate=lr, beta_1=.9, beta_2=.999, epsilon=1e-7)
    opt = (tf.keras.optimizers.AdamW(**kw, weight_decay=recipe["weight_decay"])
           if recipe["optimizer"] == "adamw" else tf.keras.optimizers.Adam(**kw))
    return loss, opt


def metrics(pred, y, naive, sd):
    pred, y, naive = (np.asarray(a, dtype=float).reshape(-1) for a in (pred, y, naive))
    if not len(y) or pred.shape != y.shape or naive.shape != y.shape or not all(
            np.isfinite(a).all() for a in (pred, y, naive)) or not np.isfinite(sd) or sd <= 0:
        raise ValueError("empty, partial, nonfinite or invalid-scale metric input")
    mae = float(np.mean(np.abs(pred-y)))
    baseline = float(np.mean(np.abs(naive-y)))
    if baseline <= 0:
        raise ValueError("naive denominator is not positive")
    return {"mae_kw": mae, "mae_z": mae/sd, "rmse_kw": float(np.sqrt(np.mean((pred-y)**2))),
            "naive_mae_kw": baseline, "skill_vs_naive": 1-mae/baseline, "rows": len(y)}


def weight_hash(model):
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(str((w.shape, str(w.dtype))).encode())
        h.update(w.tobytes())
    return h.hexdigest()


def seal(source):
    d = P.seal(source)
    d.pop("design_sha256")
    d.update(schema="df_e1_huber_design.v1", purpose="HUBER_ADAMW_FACTORIAL_DEVELOPMENT",
             arms=list(RECIPES), recipes=RECIPES, pilots=[],
             cells=[{"cell_id": f"{arm}_s{s}", "arm": arm, "seed": s}
                    for s in (1, 2, 3) for arm in RECIPES],
             factors_moved=["loss", "optimizer"],
             factors_held=["model", "data", "rows", "scaler", "seeds", "monitor", "learning_rate",
                           "batch", "update_ceiling", "early_stopping_patience"],
             resource_limits={"child_cpu_seconds": 1800, "campaign_cpu_seconds": 14400,
                              "parallel_children": 4},
             source_code={name: P.sha_file(HERE/name) for name in
                          ("df_e1_huber.py", "df_e1_phase1.py", "df_e1_pilot.py", "df_mod_e0.py",
                           "df_e1_governed.py")})
    d["training"]["monitor"] = "val_mae"
    d["graph"]["loss"] = "per-arm recipe; common validation MAE monitor"
    d["graph"]["optimizer"] = "per-arm Adam or AdamW recipe"
    d["graph"]["regimes"] = {"R0": "random detector, trainable; no pretraining in this factorial"}
    d["design_sha256"] = P._module("df_mod_e0").sha_obj(d)
    return d


def validate(d):
    # BENCHMARK-CONTRACTS (RP67): the contract is typed, its own digest recomputes, and it BINDS to the
    # prepared data this factorial consumes; a foreign target/horizon with a re-digested outer design refuses here
    B = P._module("df_benchmark_contract")
    contract = B.require(d, purpose="the loss/optimizer factorial")
    source_root = Path(d["source_run"]["root"])
    B.bind(contract, json.loads((source_root/"DATA.json").read_text()), purpose="the loss/optimizer factorial")
    E = P._module("df_mod_e0")
    if d["schema"] != "df_e1_huber_design.v1" or E.sha_obj(
            {k: v for k, v in d.items() if k != "design_sha256"}) != d["design_sha256"]:
        raise ValueError("design digest/schema mismatch")
    expected = [{"cell_id": f"{arm}_s{s}", "arm": arm, "seed": s}
                for s in (1, 2, 3) for arm in RECIPES]
    if d["recipes"] != RECIPES or d["cells"] != expected or d["training"]["monitor"] != "val_mae":
        raise ValueError("factorial population mismatch")
    for name, digest in d["source_code"].items():
        if P.sha_file(HERE/name) != digest:
            raise ValueError(f"scientific source changed: {name}")


def child(root, unit):
    cpu_start = time.process_time()
    d = json.loads((root/"DESIGN.json").read_text())
    validate(d)
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1805))
    G = P._module("df_e1_governed")
    delivery = G.require_delivery(root, d, unit)["delivery"]
    if delivery["sha256"] != d["source_run"]["panel_sha256"]:
        raise ValueError("delivered panel is not the prepared data's source")
    source = Path(d["source_run"]["root"])
    if P.sha_file(source/"DATA.npz") != d["source_run"]["data_sha256"]:
        raise ValueError("prepared data digest mismatch")
    with np.load(source/"DATA.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    cell = next(c for c in d["cells"] if c["cell_id"] == unit)
    out = root/"attempts"/unit
    out.mkdir(parents=True, exist_ok=False)
    E, B = P._module("df_mod_e0"), P._module("df_e1_pilot")
    tf = E._tf()
    tf.keras.utils.set_random_seed(cell["seed"])
    W, h, j = (int(data[k][0]) for k in ("window", "horizon", "target_channel"))
    m, sd = float(data["scaler_mean"][j]), float(data["scaler_sd"][j])
    Dataset = B._dataset_class()
    kw = dict(scaler_mean=data["scaler_mean"], scaler_sd=data["scaler_sd"])
    tr = Dataset(data["Xs"], data["Y"], data["train_origins"], W, h, j, 64,
                 shuffle=True, seed=cell["seed"], **kw)
    va = Dataset(data["Xs"], data["Y"], data["eval_origins"], W, h, j, 64,
                 shuffle=False, seed=cell["seed"], **kw)
    model = P._arm_model("core_mae", d, data, cell["seed"])
    initial = weight_hash(model)
    # Match the phase-1 loop's post-construction RNG reset, in every arm.
    tf.keras.utils.set_random_seed(cell["seed"])
    loss, opt = components(d["recipes"][cell["arm"]], d["training"]["learning_rate"])
    model.compile(optimizer=opt, loss=loss, metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    ceiling = d["training"]["max_updates"]

    class Budget(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if int(self.model.optimizer.iterations.numpy()) >= ceiling:
                self.model.stop_training = True

    es = tf.keras.callbacks.EarlyStopping(monitor="val_mae", mode="min", patience=3,
                                          restore_best_weights=True)
    t, wall = time.process_time(), time.monotonic()
    hist = model.fit(tr, validation_data=va, epochs=math.ceil(ceiling/len(tr)),
                     callbacks=[Budget(), es], verbose=0)
    pred = B._predict(model, va).reshape(-1)*sd+m
    y, naive = data["Y"][data["eval_origins"]+h], data["Y"][data["eval_origins"]]
    score = metrics(pred, y, naive, sd)
    if not np.isclose(score["mae_z"], min(hist.history["val_mae"]), rtol=2e-5, atol=2e-6):
        raise ValueError("restored predictions do not match best validation MAE")
    model.save_weights(out/"weights.weights.h5")
    restored = P._arm_model("core_mae", d, data, cell["seed"])
    restored.load_weights(out/"weights.weights.h5")
    reload_pred = B._predict(restored, va).reshape(-1)*sd+m
    np.testing.assert_allclose(pred, reload_pred, atol=1e-6, rtol=1e-6)
    np.savez(out/"arrays.npz", pred=pred, y=y, naive=naive, origins=data["eval_origins"], reload_pred=reload_pred)
    record = {"cell": cell, "design_sha256": d["design_sha256"], "scores": score,
              "recipe": d["recipes"][cell["arm"]], "optimizer": opt.get_config(),
              "loss": tf.keras.losses.serialize(tf.keras.losses.get(loss)),
              "initial_weights_sha256": initial, "final_weights_sha256": weight_hash(model),
              "curve": {k: [float(x) for x in v] for k, v in hist.history.items()},
              "best_epoch": int(np.argmin(hist.history["val_mae"]))+1,
              "updates": int(opt.iterations.numpy()), "early_stopped": bool(es.stopped_epoch),
              "budget_reached": int(opt.iterations.numpy()) >= ceiling,
              "parameters": model.count_params(), "target_mean": m, "target_sd": sd,
              "cpu_seconds": time.process_time()-cpu_start, "fit_and_verify_cpu_seconds": time.process_time()-t,
              "wall_seconds": time.monotonic()-wall,
              "reload_max_error": float(np.max(np.abs(pred-reload_pred))),
              "arrays_sha256": P.sha_file(out/"arrays.npz"),
              "weights_file_sha256": P.sha_file(out/"weights.weights.h5"),
              "environment": E.numeric_environment() if hasattr(E, "numeric_environment") else
                  {"python": sys.version, "numpy": np.__version__, "tensorflow": tf.__version__}}
    write(out/"cell.json", record)


def governance_modules():
    # The legacy dynamic loader publishes modules before executing their bodies.
    # Complete its imports on the parent thread, before simultaneous deliveries.
    G = P._module("df_e1_governed")
    G._load("governed_run")
    G._load("df_e1_receipts")
    return G, P._module("df_utility_run")


def execute(a):
    root = a.root
    d = json.loads((root/"DESIGN.json").read_text())
    validate(d)
    G, U = governance_modules()

    def one(cell):
        unit = cell["cell_id"]
        if (root/"TERMINALS"/f"{unit}.json").exists():
            raise ValueError("existing attempt: do not silently repeat experiments")
        started = U._z(U.now_iso())
        G.acquire(run_id=a.run_id, root=root, lake="public_panels",
                  resource="uci_235_individual_household_power/panel.parquet", unit_id=unit,
                  gov_url=a.gov_url, api_key_file=a.api_key_file, design_sha256=d["design_sha256"],
                  cache_dir=root/"cache", expect_sha256=d["source_run"]["panel_sha256"])
        wall = time.monotonic()
        with open(root/f"{unit}.log", "x") as log:
            try:
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--cell", unit],
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                                       "OPENBLAS_NUM_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "3"},
                                  stdout=log, stderr=subprocess.STDOUT, timeout=2400)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                exit_code = "WALL_TIMEOUT"
        record_path = root/"attempts"/unit/"cell.json"
        ok = exit_code == 0 and record_path.exists()
        rec = json.loads(record_path.read_text()) if ok else None
        cost = {"wall_seconds": time.monotonic()-wall}
        if rec:
            cost["cpu_seconds"] = rec["cpu_seconds"]
        ms = [] if not rec else [U._metric(f"e1.huber.{k}", float(v),
                               "kW" if k.endswith("kw") else "dimensionless", split="validation", horizon=60)
                               for k, v in rec["scores"].items() if k != "rows"]
        artifacts = [] if not rec else [
            {"role": role, "sha256": P.sha_file(root/"attempts"/unit/file),
             "bytes": (root/"attempts"/unit/file).stat().st_size}
            for role, file in (("predictions", "arrays.npz"), ("weights", "weights.weights.h5"), ("record", "cell.json"))]
        terminal = U._terminal(status="COMPLETED" if ok else "FAILED",
                     reason=None if ok else f"child exited {exit_code}; see retained log; CPU cost unavailable",
                     cost=cost, metrics=ms, started=started, finished=U._z(U.now_iso()),
                     tags={"purpose": d["purpose"], "classification": "NON_GOVERNING", "phase": "DEVELOPMENT",
                           "unit": unit, "arm": cell["arm"], "seed": str(cell["seed"]),
                           "design_sha256": d["design_sha256"]})
        terminal["artifacts"] = artifacts
        (root/"TERMINALS").mkdir(exist_ok=True)
        write(root/"TERMINALS"/f"{unit}.json", terminal)
        reported = G.report_terminal(root, unit, terminal, gov_url=a.gov_url, api_key_file=a.api_key_file,
                                     outbox_dir=str(root/"outbox"), started_at=started)
        if reported["flushed"]["pending"] or reported["flushed"]["failures"]:
            raise ValueError(f"terminal not delivered: {unit}")
        print(json.dumps({"unit": unit, "ok": ok, "scores": rec["scores"] if rec else None}), flush=True)
        return ok

    # Batches bound concurrency and stop the next batch if the previous one fails.
    workers = d["resource_limits"]["parallel_children"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(d["cells"]), workers):
            if not all(list(pool.map(one, d["cells"][start:start+workers]))):
                raise ValueError("failed child; remaining cells not started")
            spent = sum(json.loads(p.read_text())["cpu_seconds"] for p in (root/"attempts").glob("*/cell.json"))
            if spent > 14400-workers*1800 and start+workers < len(d["cells"]):
                raise ValueError("insufficient remaining CPU allowance for next worst-case batch")


def close(a):
    root = a.root
    d = json.loads((root/"DESIGN.json").read_text())
    validate(d)
    receipts = json.loads((root/"TERMINAL_RECEIPTS.json").read_text())["units"]
    expected = {c["cell_id"] for c in d["cells"]}
    if set(receipts) != expected:
        raise ValueError("accepted population differs from design")
    C = P._module("df_mod_e0_close")
    token = a.warehouse_token_file.read_text().strip()
    source = Path(d["source_run"]["root"])
    if P.sha_file(source/"DATA.npz") != d["source_run"]["data_sha256"]:
        raise ValueError("source data changed")
    with np.load(source/"DATA.npz", allow_pickle=False) as z:
        origins = z["eval_origins"]
        truth = z["Y"][origins+int(z["horizon"][0])]
        naive = z["Y"][origins]
    rows, problems, inits = [], [], {}
    for cell in d["cells"]:
        unit = cell["cell_id"]
        folder = root/"attempts"/unit
        rec = json.loads((folder/"cell.json").read_text())
        if rec["design_sha256"] != d["design_sha256"] or rec["cell"] != cell:
            raise ValueError("cell/design mismatch")
        if P.sha_file(folder/"arrays.npz") != rec["arrays_sha256"]:
            raise ValueError("array digest mismatch")
        with np.load(folder/"arrays.npz", allow_pickle=False) as z:
            for actual, wanted in ((z["origins"], origins), (z["y"], truth), (z["naive"], naive)):
                np.testing.assert_array_equal(actual, wanted)
            np.testing.assert_allclose(z["pred"], z["reload_pred"], rtol=1e-6, atol=1e-6)
            score = metrics(z["pred"], z["y"], z["naive"], rec["target_sd"])
        if score != rec["scores"]:
            raise ValueError("record does not match arrays")
        inits.setdefault(cell["seed"], set()).add(rec["initial_weights_sha256"])
        receipt = receipts[unit]
        held = C.warehouse_terminals(a.warehouse_url, token, receipt["campaign_sha256"])["current"]
        terminal = json.loads((root/"TERMINALS"/f"{unit}.json").read_text())
        row = held.get(unit, {})
        if set(held) != {unit} or row.get("terminal_sha256") != receipt["terminal_sha256"]:
            problems.append(f"{unit}: population/digest")
        if row.get("status") != "COMPLETED" or row.get("config_sha256") != d["design_sha256"]:
            problems.append(f"{unit}: status/design")
        if sorted(map(C._metric_key, row.get("metrics", []))) != sorted(map(C._metric_key, terminal["metrics"])):
            problems.append(f"{unit}: warehouse metric content")
        if sorted((x["role"], x["sha256"], x["bytes"]) for x in row.get("artifacts", [])) != sorted(
                (x["role"], x["sha256"], x["bytes"]) for x in terminal["artifacts"]):
            problems.append(f"{unit}: warehouse artifacts")
        rows.append({**cell, **score, "updates": rec["updates"], "best_epoch": rec["best_epoch"],
                     "budget_reached": rec["budget_reached"], "cpu_seconds": rec["cpu_seconds"],
                     "reload_max_error": rec["reload_max_error"]})
    if any(len(v) != 1 for v in inits.values()):
        problems.append("unpaired initial weights")
    summary = {arm: {"mean_mae_kw": float(np.mean(v := [r["mae_kw"] for r in rows if r["arm"] == arm])),
                     "sd_mae_kw": float(np.std(v, ddof=1))} for arm in RECIPES}
    report = {"design_sha256": d["design_sha256"], "rows": rows, "summary": summary,
              "warehouse_problems": problems, "verified": not problems,
              "scope": "DEVELOPMENT; 3 paired seeds; one previously inspected validation slice"}
    write(root/"REPORT.json", report)
    print(json.dumps(report, indent=1))
    if problems:
        raise ValueError("closure failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=Path.home()/".local/state/crispdm-data-foundation/e1_household_successor_v3")
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--cell")
    ap.add_argument("--close", action="store_true")
    ap.add_argument("--run-id", default="musashi-huber-adamw-20260921")
    ap.add_argument("--api-key-file", type=Path)
    ap.add_argument("--gov-url", default="http://127.0.0.1:5055")
    ap.add_argument("--warehouse-url", default="http://127.0.0.1:5057")
    ap.add_argument("--warehouse-token-file", type=Path)
    a = ap.parse_args()
    if a.seal:
        a.root.mkdir(parents=True, exist_ok=True)
        write(a.root/"DESIGN.json", seal(a.source))
    elif a.cell:
        child(a.root, a.cell)
    elif a.close:
        close(a)
    else:
        execute(a)


if __name__ == "__main__":
    main()
