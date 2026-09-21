#!/usr/bin/env python3
"""RP62/RP63: the first diagnostic phase — one factor at a time, on the rows the run already used.

The order of contrasts is fixed and only the first one is executed here:

    (a) training and loss      MSE in z-space, monitored on validation MSE (what the run did)
                               against MAE aligned with the measure the question is judged by
    (d) architecture           our core against the reference TCN block, parameter-matched
    (b) input information      NOT MOVED in this phase
    (c) train volume           NOT MOVED in this phase
    (e) pretraining            NOT MOVED in this phase — R0/R1/R2 return only after (a) and (d)

Everything a comparison must hold fixed is held fixed by construction: the same prepared DATA (the
successor run's, by digest), the same train origins, the same evaluation origins, the same labels,
the same scaler, the same batch, the same update ceiling, the same patience and the same three
seeds. What differs is named in the cell's own identifier and nowhere else.

Each cell is governed like any other unit: its campaign is registered and its delivery of the panel
verified BEFORE it runs, its terminal is reported through the outbox and its campaign reconciled.
The metrics are recomputed from the saved arrays at closure, never trusted from the record.

    python tools/df_e1_phase1.py --seal DESIGN.json
    python tools/df_e1_phase1.py --design DESIGN.json --root ROOT --run-id ID --source-run RUN \\
        --api-key-file KEY [--cost-pilot] [--cpu-cap-seconds N]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = "df_e1_phase1_design.v1"
ARMS = (
    ("core_mse", "our core, MSE loss in z-space, early stopping on validation MSE — the run's own recipe"),
    ("core_mae", "our core, MAE loss in z-space, early stopping on validation MAE — aligned with the "
                 "measure the question is judged by"),
    ("tcn_mse", "the reference TCN block (Bai/locuslab, ported and parameter-matched), MSE loss, "
                "early stopping on validation MSE — the architecture contrast at the run's own recipe"),
)
SEEDS = (1, 2, 3)


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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def seal(source_run: Path, *, max_updates: int = 4000, batch: int = 64, patience: int = 3,
         lr: float = 0.003, pilot_updates: int = 200) -> dict:
    """The phase, fixed before it runs, digested by its own content."""
    E = _module("df_mod_e0")
    source = Path(source_run)
    source_design = json.loads((source/"DESIGN.json").read_text())
    data_json = json.loads((source/"DATA.json").read_text())
    design = {
        "schema": SCHEMA, "purpose": "E1_DIAGNOSTIC_PHASE_1",
        "phase": "DEVELOPMENT",
        "what_this_is": "a diagnostic contrast of training objective and architecture on the rows a "
                        "finished run already used; it is not a new experiment on new data and it "
                        "does not revisit pretraining",
        "factors_moved": ["training_and_loss", "architecture"],
        "factors_held": ["input_information", "train_volume", "pretraining", "data", "rows",
                         "labels", "scaler", "batch", "update_ceiling", "patience", "seeds"],
        "source_run": {"root": str(source), "design_sha256": source_design["design_sha256"],
                       "data_sha256": data_json["data_sha256"], "panel_sha256": data_json["panel_sha256"],
                       "reading": "the prepared DATA of that run is consumed by digest; nothing is "
                                  "re-enumerated, so every arm sees exactly the rows it saw"},
        "task": source_design["task"], "graph": source_design["graph"],
        "training": {"batch": batch, "max_updates": max_updates, "learning_rate": lr,
                     "early_stopping": {"patience_epochs": patience, "restore_best": True},
                     "monitor": "each arm monitors the loss it trains on; the arm's identifier says which"},
        "reference_model": _module("df_tcn_reference").REFERENCE,
        "arms": [{"arm": a, "what_differs": why} for a, why in ARMS],
        "replicas": {"seeds": list(SEEDS), "pairing": "every arm of a seed starts from the same "
                                                      "initial weights where the graph is the same; "
                                                      "the reference block is a different graph and "
                                                      "cannot share them, which is stated, not hidden"},
        "cells": [{"cell_id": f"{a}_s{s}", "arm": a, "seed": s} for a, _ in ARMS for s in SEEDS],
        "pilots": [{"cell_id": f"pilot_{a}", "arm": a, "seed": SEEDS[0], "max_updates": pilot_updates}
                   for a, _ in ARMS],
        "metrics": {"mae": "kW on the common evaluation set, recomputed from arrays at closure",
                    "persistence_scaled_error_horizon_train": "MAE / mean|Y[t+h]-Y[t]| over train origins"},
        "governance": "one campaign and one verified delivery per unit before it runs; a terminal "
                      "through the outbox and a reconciled campaign after it",
        "reading_rules": ["three seeds on one task are development evidence, not a confirmation",
                          "an arm that stops at the update ceiling is CENSORED: what it would reach "
                          "with more budget is unknown, in either direction",
                          "no cell is removed after its score is seen"],
    }
    design["design_sha256"] = E.sha_obj(design)
    return design


def _arm_model(arm: str, design: dict, data: dict, seed: int):
    P = _module("df_e1_pilot")
    W = int(data["window"][0])
    j = int(data["target_channel"][0])
    p = int(data["Xs"].shape[1])
    if arm.startswith("tcn"):
        return _module("df_tcn_reference").build(W, p, seed=seed)
    return P._model_for_target(design["graph"]["assignment"], W, p, j, seed,
                               core=design["graph"]["core_kind"])


def _fit(arm: str, model, train_ds, val_ds, *, max_updates, patience, lr, seed) -> dict:
    """The run's own loop, with the arm's own loss and monitor."""
    E = _module("df_mod_e0")
    tf = E._tf()
    import math
    tf.keras.utils.set_random_seed(int(seed))
    loss = "mae" if arm.endswith("mae") else "mse"
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss)
    steps = len(train_ds)
    max_epochs = max(1, math.ceil(max_updates/steps))

    class Counter(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.updates, self.budget_stop = 0, False

        def on_train_batch_end(self, b, logs=None):
            self.updates += 1
            if self.updates >= max_updates:
                self.budget_stop = True
                self.model.stop_training = True

    counter = Counter()
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(patience),
                                          restore_best_weights=True)
    t0 = time.process_time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=max_epochs, verbose=0,
                     callbacks=[counter, es])
    fit_s = time.process_time()-t0
    va = [float(v) for v in hist.history["val_loss"]]
    stop = "UPDATE_BUDGET" if counter.budget_stop else ("EARLY_STOPPING" if es.stopped_epoch else "EPOCH_BUDGET")
    iterations = int(model.optimizer.iterations.numpy())
    return {"loss_trained": loss, "monitor": f"validation {loss}",
            "updates": int(counter.updates), "optimizer_iterations": iterations,
            "updates_are_optimizer_iterations": iterations == counter.updates,
            "epochs": len(va), "steps_per_epoch": steps,
            "curve": {"train": [float(v) for v in hist.history["loss"]], "validation": va},
            "stop_reason": stop, "restored_checkpoint_epoch": int(np.argmin(va))+1,
            "censoring": {"budget_reached": bool(counter.budget_stop), "stopped_by": stop,
                          "best_epoch": int(np.argmin(va))+1, "epochs": len(va),
                          "updates_used_of_ceiling": [int(counter.updates), int(max_updates)],
                          "verdict": "CENSORED_BY_BUDGET" if counter.budget_stop else
                                     "STOPPED_ON_VALIDATION" if es.stopped_epoch else "EPOCH_BUDGET_REACHED",
                          "criterion": "a fit that stops at the ceiling says what was explored, not "
                                       "what is reachable"},
            "fit_seconds": round(fit_s, 3)}


def run_cell(design: dict, data: dict, cell: dict, out_dir: Path, *, max_updates: int) -> dict:
    """One arm, one seed: fit, predict, score, and save the arrays the closure will recompute from."""
    P = _module("df_e1_pilot")
    E = _module("df_mod_e0")
    out_dir.mkdir(parents=True, exist_ok=True)
    W, h = int(data["window"][0]), int(data["horizon"][0])
    j = int(data["target_channel"][0])
    Dataset = P._dataset_class()
    kw = dict(scaler_mean=data["scaler_mean"], scaler_sd=data["scaler_sd"])
    seed = int(cell["seed"])
    train = Dataset(data["Xs"], data["Y"], data["train_origins"], W, h, j, design["training"]["batch"],
                    shuffle=True, seed=seed, **kw)
    val = Dataset(data["Xs"], data["Y"], data["eval_origins"], W, h, j, design["training"]["batch"],
                  shuffle=False, seed=seed, **kw)
    model = _arm_model(cell["arm"], design, data, seed)
    params = int(sum(int(w.shape.num_elements()) for w in model.trainable_weights))
    t0 = time.process_time()
    training = _fit(cell["arm"], model, train, val, max_updates=max_updates,
                    patience=design["training"]["early_stopping"]["patience_epochs"],
                    lr=design["training"]["learning_rate"], seed=seed)
    pred_s = P._predict(model, val).reshape(-1)
    m, s = float(data["scaler_mean"][j]), float(data["scaler_sd"][j])
    pred = pred_s*s + m
    y = data["Y"][data["eval_origins"]+h]
    scores = E.mase(pred[:, None], y[:, None], data["denominator"].tolist())
    np.savez(out_dir/"arrays.npz", validation_pred=pred[:, None], validation_y=y[:, None],
             eval_origins=data["eval_origins"], denominator=data["denominator"])
    model.save_weights(out_dir/"weights.weights.h5")
    record = {"schema": "df_e1_phase1_cell.v1", "cell_id": cell["cell_id"], "arm": cell["arm"],
              "seed": seed, "design_sha256": design["design_sha256"], "parameters": params,
              "training": training,
              "scores": {"validation": {"model": {"mae_mean": scores["mae_mean"],
                                                  "persistence_scaled_error_horizon_train": scores["mase_mean"],
                                                  "mase_mean": scores["mase_mean"]}}},
              "cost": {"cpu_seconds_process": round(time.process_time()-t0, 3)},
              "arrays_sha256": sha_file(out_dir/"arrays.npz")}
    (out_dir/"cell.json").write_text(json.dumps(record, indent=1, default=str))
    return record


def run(design: dict, *, root: Path, run_id: str, source_run: Path, gov_url: str, api_key_file: Path,
        lake: str, resource: str, cost_pilot_only: bool = False, cap_seconds: float = 8000.0,
        outbox_dir: str | None = None) -> dict:
    G = _module("df_e1_governed")
    U = _module("df_utility_run")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root/"DESIGN.json").write_text(json.dumps(design, indent=1))
    source = Path(source_run)
    # the prepared DATA of the source run, consumed by digest: the same rows, the same labels
    data_path = root/"DATA.npz"
    if not data_path.exists():
        data_path.write_bytes((source/"DATA.npz").read_bytes())
    digest = sha_file(data_path)
    if digest != design["source_run"]["data_sha256"]:
        raise SystemExit(f"REFUSED: DATA.npz is {digest}, the sealed design names "
                         f"{design['source_run']['data_sha256']}")
    with np.load(data_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    report = {"schema": "df_e1_phase1_report.v1", "run_id": run_id, "at": now_iso(),
              "design_sha256": design["design_sha256"], "host": os.uname().nodename,
              "data_sha256": digest, "cells": {}, "terminals": [], "stopped": None,
              "cap_seconds": cap_seconds}
    units = design["pilots"] if cost_pilot_only else design["cells"]
    spent = 0.0
    for cell in units:
        max_updates = cell.get("max_updates", design["training"]["max_updates"])
        started = U._z(U.now_iso())
        G.acquire(run_id=run_id, root=root, lake=lake, resource=resource, unit_id=cell["cell_id"],
                  gov_url=gov_url, api_key_file=api_key_file,
                  design_sha256=design["design_sha256"], cache_dir=root/"cache",
                  expect_sha256=design["source_run"]["panel_sha256"])
        record = run_cell(design, data, cell, root/"attempts"/cell["cell_id"], max_updates=max_updates)
        spent += float(record["cost"]["cpu_seconds_process"])
        report["cells"][cell["cell_id"]] = {
            "arm": cell["arm"], "seed": cell["seed"], "parameters": record["parameters"],
            "mae": record["scores"]["validation"]["model"]["mae_mean"],
            "persistence_scaled_error_horizon_train":
                record["scores"]["validation"]["model"]["persistence_scaled_error_horizon_train"],
            "updates": record["training"]["updates"], "epochs": record["training"]["epochs"],
            "stop": record["training"]["stop_reason"],
            "verdict": record["training"]["censoring"]["verdict"],
            "cpu_seconds": record["cost"]["cpu_seconds_process"]}
        terminal = U._terminal(
            status="COMPLETED", reason=None,
            cost={"wall_seconds": record["cost"]["cpu_seconds_process"],
                  "cpu_seconds": record["cost"]["cpu_seconds_process"]},
            metrics=[U._metric("e1.phase1.mae_validation", record["scores"]["validation"]["model"]["mae_mean"],
                               "kW", split="validation", horizon=int(data["horizon"][0])),
                     U._metric("e1.phase1.persistence_scaled_error",
                               record["scores"]["validation"]["model"]["persistence_scaled_error_horizon_train"],
                               "ratio_persistence_h", split="validation", horizon=int(data["horizon"][0]))],
            started=started, finished=U._z(U.now_iso()),
            tags={"purpose": "E1_DIAGNOSTIC_PHASE_1", "classification": "NON_GOVERNING",
                  "phase": "DEVELOPMENT", "unit": cell["cell_id"], "arm": cell["arm"],
                  "seed": str(cell["seed"]), "design_sha256": design["design_sha256"]})
        reported = G.report_terminal(root, cell["cell_id"], terminal, gov_url=gov_url,
                                     api_key_file=api_key_file, outbox_dir=outbox_dir,
                                     started_at=started)
        report["terminals"].append({"unit_id": cell["cell_id"],
                                    "terminal_sha256": (reported.get("receipt") or {}).get("terminal_sha256"),
                                    "reconciliation": reported["reconciliation"],
                                    "sent": reported["flushed"]["sent"]})
        if reported["flushed"]["pending"] or reported["flushed"]["failures"]:
            raise SystemExit(f"REFUSED: the terminal of {cell['cell_id']} was not accepted: "
                             f"{reported['flushed']['failures']}")
        if spent > cap_seconds:
            report["stopped"] = f"CPU_CAP_REACHED after {cell['cell_id']}"
            break
    report["spent_cpu_seconds"] = round(spent, 3)
    if cost_pilot_only:
        report["projection"] = _project(design, report)
    _module("df_d3_campaign").write_once(root/("REPORT.pilot.json" if cost_pilot_only else "REPORT.json"),
                                         report)
    return report


def _project(design: dict, report: dict) -> dict:
    """What the full phase would cost, from the pilots' measured cost per update."""
    per_update = {}
    for cell_id, entry in report["cells"].items():
        arm = entry["arm"]
        per_update[arm] = entry["cpu_seconds"]/max(1, entry["updates"])
    ceiling = design["training"]["max_updates"]
    per_arm = {arm: rate*ceiling for arm, rate in per_update.items()}
    total = sum(per_arm.get(c["arm"], 0.0) for c in design["cells"])
    return {"cpu_seconds_per_update_by_arm": per_update,
            "projected_cpu_seconds_per_cell_at_the_ceiling": per_arm,
            "cells": len(design["cells"]),
            "projected_total_cpu_seconds": round(total, 1),
            "with_headroom_25_percent": round(total*1.25, 1),
            "assumption": "every cell runs to the full ceiling; early stopping can only lower it"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seal", type=Path)
    ap.add_argument("--design", type=Path)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--run-id")
    ap.add_argument("--source-run", type=Path,
                    default=Path("~/.local/state/crispdm-data-foundation/e1_household_successor_v3").expanduser())
    ap.add_argument("--gov-url", default="http://127.0.0.1:5055")
    ap.add_argument("--api-key-file", type=Path)
    ap.add_argument("--lake", default="public_panels")
    ap.add_argument("--resource", default="uci_235_individual_household_power/panel.parquet")
    ap.add_argument("--outbox-dir", default=None)
    ap.add_argument("--cost-pilot", action="store_true")
    ap.add_argument("--cpu-cap-seconds", type=float, default=8000.0)
    a = ap.parse_args(argv)
    if a.seal:
        design = seal(a.source_run)
        _module("df_d3_campaign").write_once(a.seal, design)
        print(json.dumps({"design_sha256": design["design_sha256"], "cells": len(design["cells"]),
                          "pilots": len(design["pilots"])}, indent=1))
        return 0
    design = json.loads(a.design.read_text())
    E = _module("df_mod_e0")
    if design.get("schema") != SCHEMA or \
            E.sha_obj({k: v for k, v in design.items() if k != "design_sha256"}) != design["design_sha256"]:
        raise SystemExit("REFUSED: the design is not a sealed phase-1 design")
    report = run(design, root=a.root, run_id=a.run_id, source_run=a.source_run, gov_url=a.gov_url,
                 api_key_file=a.api_key_file, lake=a.lake, resource=a.resource,
                 cost_pilot_only=a.cost_pilot, cap_seconds=a.cpu_cap_seconds, outbox_dir=a.outbox_dir)
    print(json.dumps({"cells": report["cells"], "spent": report["spent_cpu_seconds"],
                      "projection": report.get("projection"), "stopped": report["stopped"]},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
