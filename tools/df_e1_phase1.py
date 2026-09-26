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

# --- the monitor defect, and its repair -----------------------------------------------------------
#
# RP63 monitored `val_loss` for every arm. `val_loss` IS the arm's own trained loss, so changing the
# loss silently changed TWO further things that no contrast is allowed to move:
#
#   (1) WHEN early stopping fired, hence how many optimiser updates each arm was given. The retained
#       run: `core_mae` and `tcn_mse` 11 762 updates each, `core_mse` 10 270 -- a 14.5% budget
#       difference between the winning arm and the arm it beat, named in
#       SATOSHI_RP57_RP64_DISPOSITION_2026_09_26 section 5.1.
#   (2) WHICH checkpoint was restored: `restore_best_weights` restored the argmin of a DIFFERENT
#       curve in each arm, and for the `*_mae` arms that curve is the very measure the comparison is
#       judged on. The MAE arm was allowed to select its checkpoint on the judged measure; the MSE
#       arms were not. This leg is not named anywhere in either disposition.
#
# Both legs come from one line. The repair is one fixed, arm-independent monitor for every arm, plus
# a budget that is matched by construction rather than by luck, and a runner that REFUSES a design
# which declares neither.
FIXED_MONITOR = "val_mae"                       # the measure the question is judged by, in every arm
MONITORS = (FIXED_MONITOR, "val_mse")           # any arm-independent metric name may be fixed
LEGACY_MONITOR = "val_loss"                     # arm-DEPENDENT: the defect, reachable only by name
BUDGET_FIXED = "FIXED_UPDATES"                  # every cell runs exactly the ceiling; matched by construction
BUDGET_EARLY = "EARLY_STOPPING"                 # the budget is an outcome; arms may and did drift
LEGACY_PROTOCOL = "LEGACY_PER_ARM_MONITOR_UNMATCHED_BUDGET"
MONITOR_DECLARATION = ("one fixed, arm-independent monitor for every arm: the measure the question is "
                       "judged by. The trained loss never chooses the stopping epoch or the restored "
                       "checkpoint")


class ProtocolRefusal(ValueError):
    """A design whose monitor or budget would confound the contrast it is built to measure."""


def resolve_protocol(design: dict) -> dict:
    """The monitor and budget rule this design declares -- or a refusal naming the defect.

    A design that says nothing is REFUSED rather than defaulted, because silence is exactly how the
    retained phase-1 design reached execution: it declared 'each arm monitors the loss it trains on'
    and nothing about the budget, and both legs of the confound followed from that sentence.
    """
    t = design.get("training") or {}
    monitor = t.get("monitor")
    budget = t.get("budget_match")
    if budget == LEGACY_PROTOCOL:
        # reachable, but only by naming it: reproducing the defective run is a legitimate act
        return {"monitor": LEGACY_MONITOR, "budget_match": BUDGET_EARLY,
                "monitor_is_arm_independent": False, "budget_is_matched_by_construction": False,
                "declared": LEGACY_PROTOCOL,
                "defect": ("this protocol lets the trained loss choose both the stopping epoch and the "
                           "restored checkpoint; its arms are not budget-matched and its contrasts are "
                           "confounded with both")}
    if monitor not in MONITORS:
        raise ProtocolRefusal(
            f"REFUSED: this design declares the monitor {monitor!r}, which is not one of the "
            f"arm-independent monitors {MONITORS}. An arm-dependent monitor makes the trained loss "
            f"choose the stopping epoch and the restored checkpoint, so the contrast measures the "
            f"recipe and its own budget together. Declare training.monitor={FIXED_MONITOR!r}, or "
            f"training.budget_match={LEGACY_PROTOCOL!r} to reproduce the defective run on purpose")
    if budget not in (BUDGET_FIXED, BUDGET_EARLY):
        raise ProtocolRefusal(
            f"REFUSED: this design declares no budget rule (training.budget_match={budget!r}). "
            f"With {BUDGET_EARLY!r} the optimiser-update count is an outcome and the arms may drift "
            f"apart, as they did by 1 492 updates in RP63; {BUDGET_FIXED!r} gives every cell the same "
            f"count by construction")
    return {"monitor": monitor, "budget_match": budget,
            "monitor_is_arm_independent": True,
            "budget_is_matched_by_construction": budget == BUDGET_FIXED,
            "declared": f"{monitor} / {budget}"}


def budget_audit(cells: dict) -> dict:
    """Per-arm optimiser-update totals, and whether the arms are matched.

    `cells` is a mapping of cell_id -> {"arm", "updates", ...}, which is the shape both the runner's
    own report and the retained RP63 report carry, so this rule reads a published run without
    reformatting it.
    """
    per_arm: dict = {}
    for cid, c in cells.items():
        arm = c.get("arm")
        u = int(c.get("updates") or 0)
        e = per_arm.setdefault(arm, {"cells": [], "updates_per_cell": [], "total_updates": 0})
        e["cells"].append(cid)
        e["updates_per_cell"].append(u)
        e["total_updates"] += u
    totals = {a: e["total_updates"] for a, e in per_arm.items()}
    per_cell_sets = {a: sorted(e["updates_per_cell"]) for a, e in per_arm.items()}
    matched_totals = len(set(totals.values())) <= 1
    matched_per_seed = len({tuple(v) for v in per_cell_sets.values()}) <= 1
    spread = (max(totals.values()) - min(totals.values())) if totals else 0
    return {"per_arm": per_arm, "total_updates_by_arm": totals,
            "matched_on_totals": matched_totals, "matched_per_seed": matched_per_seed,
            "largest_total_minus_smallest": int(spread),
            "relative_spread": (spread / min(totals.values())) if totals and min(totals.values()) else None,
            "rule": ("two arms compared on their mean error must have been given the same number of "
                     "optimiser updates; an arm that ran longer has an advantage that is not its recipe")}


def require_budget_match(cells: dict) -> dict:
    """Refuse to report a contrast whose arms were given different budgets."""
    a = budget_audit(cells)
    if not a["matched_on_totals"]:
        raise ProtocolRefusal(
            "REFUSED: the arms of this contrast are not budget-matched: "
            + ", ".join(f"{arm} {tot} updates" for arm, tot in sorted(a["total_updates_by_arm"].items()))
            + f" (a spread of {a['largest_total_minus_smallest']} updates). The difference between "
              "these arms mixes the recipe with the budget and is not a recipe effect")
    return a


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
         lr: float = 0.003, pilot_updates: int = 200, monitor: str = FIXED_MONITOR,
         budget_match: str = BUDGET_FIXED) -> dict:
    """The phase, fixed before it runs, digested by its own content.

    `monitor` and `budget_match` are part of the sealed identity, so two runs that differ in either
    cannot share a design digest. The defaults are the REPAIRED protocol; the defective one is
    reachable by passing ``budget_match=LEGACY_PROTOCOL`` and is then named in the design itself.
    """
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
        "factors_held": (["input_information", "train_volume", "pretraining", "data", "rows",
                          "labels", "scaler", "batch", "update_ceiling", "patience", "seeds",
                          "monitor", "optimiser_updates"] if budget_match == BUDGET_FIXED else
                         ["input_information", "train_volume", "pretraining", "data", "rows",
                          "labels", "scaler", "batch", "update_ceiling", "patience", "seeds"]),
        "source_run": {"root": str(source), "design_sha256": source_design["design_sha256"],
                       "data_sha256": data_json["data_sha256"], "panel_sha256": data_json["panel_sha256"],
                       "reading": "the prepared DATA of that run is consumed by digest; nothing is "
                                  "re-enumerated, so every arm sees exactly the rows it saw"},
        "task": source_design["task"], "graph": source_design["graph"],
        "training": {"batch": batch, "max_updates": max_updates, "learning_rate": lr,
                     "early_stopping": {"patience_epochs": patience, "restore_best": True},
                     # the two fields the RP63 defect lived in: an arm-independent monitor, and a
                     # budget rule that says whether the update count is fixed or is an outcome
                     "monitor": monitor, "budget_match": budget_match,
                     "monitor_declaration": (MONITOR_DECLARATION if budget_match != LEGACY_PROTOCOL else
                                             "LEGACY: each arm monitors the loss it trains on, and its "
                                             "budget is whatever early stopping gave it")},
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
                          ("every arm ran the same number of optimiser updates, so no difference "
                           "between arms carries a budget the others did not get"
                           if budget_match == BUDGET_FIXED else
                           "the arms' budgets are outcomes of early stopping and MAY DIFFER; a "
                           "difference between arms mixes the recipe with the budget"),
                          "an arm that stops at the update ceiling is CENSORED: what it would reach "
                          "with more budget is unknown, in either direction",
                          "no cell is removed after its score is seen"],
        # BENCHMARK-CONTRACTS: the task's contract travels with the design, and comparability is
        # decided from its fields BEFORE any score; the runner refuses a design without it
        "benchmark_contract": _benchmark_contract(),
    }
    design["design_sha256"] = E.sha_obj(design)
    return design


def _benchmark_contract() -> dict:
    """The household task's typed contract and its comparability, decided from fields (RP67)."""
    B = _module("df_benchmark_contract")
    ours = B.household_ours()
    decision = {**B.decide(ours, B.gasparin_2019()), "against": "gasparin_2019",
                "rule": "decided from the identity fields, never from a score"}
    return ours.to_design_block(comparability=decision)


def _arm_model(arm: str, design: dict, data: dict, seed: int):
    P = _module("df_e1_pilot")
    W = int(data["window"][0])
    j = int(data["target_channel"][0])
    p = int(data["Xs"].shape[1])
    if arm.startswith("tcn"):
        return _module("df_tcn_reference").build(W, p, seed=seed)
    return P._model_for_target(design["graph"]["assignment"], W, p, j, seed,
                               core=design["graph"]["core_kind"])


def _fit(arm: str, model, train_ds, val_ds, *, max_updates, patience, lr, seed,
         monitor: str, budget_match: str, ckpt_dir: Path | None = None) -> dict:
    """One cell's fit, with the arm's own loss and an explicitly declared monitor and budget rule.

    `monitor` and `budget_match` are REQUIRED. There is no default, because the defect this signature
    repairs was a default: the previous version hard-coded ``monitor="val_loss"``, which is the arm's
    own trained loss, and let the stopping epoch and the restored checkpoint follow the recipe under
    test. A caller must now say which arm-independent quantity decides those two things.

    Under ``budget_match=FIXED_UPDATES`` early stopping is not installed at all: every cell runs
    exactly `max_updates` optimiser updates and the monitor's only job is to choose which checkpoint
    is restored. That is what makes two arms comparable -- not an equal *ceiling*, which RP63 already
    had, but an equal *count*.
    """
    E = _module("df_mod_e0")
    tf = E._tf()
    import math
    tf.keras.utils.set_random_seed(int(seed))
    loss = "mae" if arm.endswith("mae") else "mse"
    if monitor == LEGACY_MONITOR:
        metrics = []                                     # the defect, reproduced only when named
    elif monitor in MONITORS:
        name = monitor[len("val_"):]
        metrics = [tf.keras.metrics.MeanAbsoluteError(name="mae") if name == "mae"
                   else tf.keras.metrics.MeanSquaredError(name="mse")]
    else:
        raise ProtocolRefusal(f"REFUSED: {monitor!r} is not an arm-independent monitor; "
                              f"one of {MONITORS} or the named legacy {LEGACY_MONITOR!r}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=metrics)
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
    callbacks = [counter]
    es = None
    if budget_match == BUDGET_EARLY:
        es = tf.keras.callbacks.EarlyStopping(monitor=monitor, mode="min", patience=int(patience),
                                             restore_best_weights=True)
        callbacks.append(es)
    elif budget_match != BUDGET_FIXED:
        raise ProtocolRefusal(f"REFUSED: {budget_match!r} is not a budget rule; "
                              f"one of {(BUDGET_FIXED, BUDGET_EARLY)}")
    else:
        # the monitor still chooses the checkpoint; it no longer chooses the budget
        ckpt = Path(ckpt_dir or ".")/"_monitor_best.weights.h5"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt), monitor=monitor, mode="min", save_best_only=True,
            save_weights_only=True, verbose=0))
    t0 = time.process_time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=max_epochs, verbose=0,
                     callbacks=callbacks)
    fit_s = time.process_time()-t0
    if monitor not in hist.history:
        raise ProtocolRefusal(f"REFUSED: the monitor {monitor!r} is not in the fitted history "
                              f"{sorted(hist.history)}; a monitor that is not measured cannot select")
    curve = [float(v) for v in hist.history[monitor]]
    best = int(np.argmin(curve))+1
    # The update count is READ BEFORE any checkpoint is restored. Keras 3's `save_weights` carries the
    # optimizer's variables, so restoring the argmin checkpoint also rewinds `optimizer.iterations` to
    # the value it held at that epoch: reading it afterwards would report the restored epoch's count as
    # the run's budget, and two arms restoring different epochs would look budget-mismatched when they
    # are not. The counter is the number of updates PERFORMED; the accounting must come from it.
    iterations = int(model.optimizer.iterations.numpy())
    if budget_match == BUDGET_FIXED:
        restored_iterations = None
        if ckpt.is_file():
            model.load_weights(ckpt)                     # the argmin of the FIXED monitor, every arm
            restored_iterations = int(model.optimizer.iterations.numpy())
            ckpt.unlink()
        stop = "FIXED_UPDATE_BUDGET"
        early_stopped = False
    else:
        restored_iterations = None
        stop = "UPDATE_BUDGET" if counter.budget_stop else ("EARLY_STOPPING" if es.stopped_epoch else "EPOCH_BUDGET")
        early_stopped = bool(es.stopped_epoch)
    return {"loss_trained": loss, "monitor": monitor,
            "monitor_is_arm_independent": monitor != LEGACY_MONITOR,
            "budget_match": budget_match,
            "budget_is_matched_by_construction": budget_match == BUDGET_FIXED,
            "monitor_chose_the_budget": budget_match == BUDGET_EARLY,
            "updates": int(counter.updates), "optimizer_iterations": iterations,
            "updates_are_optimizer_iterations": iterations == counter.updates,
            "optimizer_iterations_after_restore": restored_iterations,
            "restore_rewound_the_optimizer": (restored_iterations is not None
                                              and restored_iterations != iterations),
            "epochs": len(curve), "steps_per_epoch": steps,
            "curve": {"train": [float(v) for v in hist.history["loss"]],
                      "validation": [float(v) for v in hist.history["val_loss"]],
                      "monitor": curve, "monitor_name": monitor},
            "stop_reason": stop, "restored_checkpoint_epoch": best,
            "restored_checkpoint_chosen_on": monitor,
            "censoring": {"budget_reached": bool(counter.budget_stop), "stopped_by": stop,
                          "best_epoch": best, "epochs": len(curve),
                          "updates_used_of_ceiling": [int(counter.updates), int(max_updates)],
                          "verdict": ("BUDGET_MATCHED_BY_CONSTRUCTION" if budget_match == BUDGET_FIXED
                                      else "CENSORED_BY_BUDGET" if counter.budget_stop
                                      else "STOPPED_ON_VALIDATION" if early_stopped else "EPOCH_BUDGET_REACHED"),
                          "criterion": ("every cell ran the same number of updates, so no arm's mean "
                                        "carries a budget the others did not get"
                                        if budget_match == BUDGET_FIXED else
                                        "a fit that stops at the ceiling says what was explored, not "
                                        "what is reachable")},
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
    protocol = resolve_protocol(design)
    training = _fit(cell["arm"], model, train, val, max_updates=max_updates,
                    patience=design["training"]["early_stopping"]["patience_epochs"],
                    lr=design["training"]["learning_rate"], seed=seed,
                    monitor=protocol["monitor"], budget_match=protocol["budget_match"],
                    ckpt_dir=out_dir)
    training["protocol"] = protocol
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
    # no contract, no run: the refusal lives in the runner, before any acquisition or fit — and the
    # contract must BIND to the data this run will actually consume (RP67), not only to itself
    B = _module("df_benchmark_contract")
    contract = B.require(design, purpose="the phase-1 diagnostic training")
    B.bind(contract, json.loads((Path(source_run)/"DATA.json").read_text()), purpose="the phase-1 diagnostic training")
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
