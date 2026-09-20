#!/usr/bin/env python3
"""RP30: the household DEV pilot of E1 — modular reference ARCH-A under the three learning regimes
(R0 random trainable detector; R1 pre-trained frozen detector; R2 the same pre-trained detector
adjustable), with persistence, seasonal-naive (daily) and linear controls on ONE task/horizon, three
paired replicas (same initial checkpoint per seed across regimes; same AE per seed for R1/R2).

    seal      DESIGN sealed BEFORE any outcome: task/horizon/context (from E1_TASKS), DEV sub-partition
              (rows), graph (units/layers/activations/reach), grouping (held constant), seeds, budget
              (batch, update ceiling, validation stopping/checkpoint), pilot size, headroom
    prepare   the DEV bytes through the real loader (RP27): contract -> enumerator -> train-only scaler;
              memory preflight; DATA.npz (scaled inputs, raw targets, origins, common evaluation set,
              MASE denominator) with digests; windows are gathered per batch, never materialised
    pilot     cost pilot on the DEV sub-partition (AE and one fit at a small update ceiling): seconds per
              update, overhead, peak memory -> projection with headroom against the remaining ceiling
    run       every unit as an isolated child (memory/wall/CPU ceilings, own scope): ae_s, R0_s, R1_s,
              R2_s (R1/R2 depend on ae_s), controls; verified scores recomputed from the child's arrays;
              a terminal document (governed_terminal.v1) per unit in the run root
    close     RESULTS.json/.md: per regime x seed, paired differences, controls, costs by phase, curves
              and censoring, identity proofs (shared initial checkpoint, detector digests, gradients)

Governance disposition (declared, not hidden): the public panels are NOT resources of any lake the
deployed data-gov serves (lakes: financial_files, olap_cube, predictor_examples), and a DATASETS
campaign completes a unit only through governed deliveries of every declared dataset; a SYNTHETIC
campaign would misdeclare real bytes. The pilot therefore records its terminals locally with the exact
governed schema and writes CAMPAIGN_PROPOSAL.json (the DATASETS campaign that reports them once the
panel is a lake resource). The missing object is one lake registration of the public panels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
_LOAD_LOCK = __import__("threading").RLock()


def _load(name: str, where: Path = HERE):
    with _LOAD_LOCK:
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module


E = _load("df_mod_e0")
L = _load("df_e1_loader")
RG = _load("df_e1_regimes")

SCHEMA_DESIGN = "df_e1_pilot_design.v1"
SCHEMA_CELL = "df_e1_cell.v1"
SCHEMA_DATA = "df_e1_pilot_data.v1"
TASKS_FILE = REPO / "docs/tres_temas_entrevista/program_v3/E1_TASKS.json"
FAMILY = "uci_235"
DAY = 1440                                   # rows per day at one-minute sampling
MEMORY_PREFLIGHT_CAP = 1 << 30               # bytes the prepared DATA may occupy in a child
REGIMES = ("R0", "R1", "R2")
#: RP37: the controls are named by the INFORMATION they use, because a control with more information
#: than the network is a reference, not a peer.
CONTROLS = ("persistence", "seasonal_naive_daily", "linear_ridge", "linear_reach")
CONTROL_SUPPORT = {
    "persistence": "one sample: the target's last observation (inside every receiver's reach)",
    "seasonal_naive_daily": "the target one day before the target row: OUTSIDE the context window; a HIGHER-INFORMATION "
                            "reference, not an equal-information peer",
    "linear_ridge": "the whole window (W x p): the same information the declared context gives",
    "linear_reach": "the last `model_reach_steps` rows of the window: the same information a local receiver can use",
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# --- seal ---------------------------------------------------------------------------------------------------

def seal(*, window: int = 60, horizon: int = 60, dev_train_days: int = 28, dev_val_days: int = 7, seeds=(1, 2, 3), max_updates: int = 4000,
         ae_updates: int = 1500, batch: int = 64, patience_epochs: int = 3, pilot_updates: int = 300, headroom: float = 0.25,
         learning_rate: float = 3e-3, mask_ratio: float = 0.3, ridge_lambda: float = 1.0, task_memory_bytes: int = 2 << 30,
         internal_validation_fraction: float = 0.15, validation_bank_seed: int = 20260919, core_kind: str = "conv3",
         data_access: str = "LOCAL_CHARACTERISED_FILE",
         wall_seconds: float = 2400.0, cpu_seconds: int = 2400, declared_task: dict | None = None) -> dict:
    """`declared_task`: a task declaration used INSTEAD of the E1_TASKS lookup (tests on synthetic panels only; recorded)."""
    tasks = json.loads(TASKS_FILE.read_text())
    fam = tasks["families"][FAMILY]
    key = f"W{window}_h{horizon}"
    if declared_task is None and key not in fam["windows"]:
        raise SystemExit(f"REFUSED: task {key} is not declared in E1_TASKS for {FAMILY}")
    task = declared_task if declared_task is not None else fam["windows"][key]
    train_end = int(fam["splits_by_time"]["train"][1])
    dev_rows = [train_end - dev_train_days * DAY, train_end + dev_val_days * DAY]
    contract = L.household_contract(window, horizon, history=True)
    contract.splits = {"train": dev_train_days / (dev_train_days + dev_val_days), "validation": dev_val_days / (dev_train_days + dev_val_days)}
    cols = contract.input_columns()
    physical = {"Global_reactive_power": 0, "Global_intensity": 0, "Global_active_power": 0, "Voltage": 1,
                "Sub_metering_1": 2, "Sub_metering_2": 2, "Sub_metering_3": 2}
    assignment = [physical[c] for c in cols]
    reach = model_reach(window, core_kind)
    cells, pilots = [], []
    for s in seeds:
        cells.append({"cell_id": f"ae_s{s}", "kind": "ae", "seed": s, "max_updates": ae_updates})
        for r in REGIMES:
            cells.append({"cell_id": f"{r}_s{s}", "kind": "fit", "regime": r, "seed": s, "max_updates": max_updates,
                          **({"depends_on": f"ae_s{s}"} if r != "R0" else {})})
    cells.append({"cell_id": "controls", "kind": "controls", "seed": 0})
    pilots = [{"cell_id": "pilot_ae", "kind": "ae", "seed": int(seeds[0]), "max_updates": pilot_updates, "role": "COST_PILOT"},
              {"cell_id": "pilot_fit", "kind": "fit", "regime": "R0", "seed": int(seeds[0]), "max_updates": pilot_updates, "role": "COST_PILOT"}]
    design = {
        "schema": SCHEMA_DESIGN, "purpose": "E1_DEV_PILOT_HOUSEHOLD", "phase": "DEVELOPMENT", "data_access": data_access,
        "what_this_is": "the first E1 development pilot: three learning regimes on one public task; it does not confirm H1, does not select a "
                        "universal model and does not replace the benchmark of the proposal (families/comparators stay pending in the matrix)",
        "family": FAMILY, "dataset_id": fam["dataset_id"], "governed_bytes": fam["governed_bytes"],
        "task_source": "E1_TASKS" if declared_task is None else "DECLARED_OVERRIDE (synthetic test)",
        "task": {"id": key, "window_steps": window, "horizon_steps": horizon, "context_physical_seconds": task["context_physical_seconds"],
                 "horizon_physical_seconds": task["horizon_physical_seconds"], "purge": task["purge"],
                 "model_reach_steps": reach, "model_reach_physical_seconds": reach * 60,
                 "reach_measured_by": "perturbation and gradient per input row (tools/df_e1_receiver.py), not asserted",
                 "reach_note": "ARCH-A with the sequence fusion depends on the last `model_reach_steps` samples of the window (branch 5 + core 3 - 1); "
                               "the window is the declared context, the reach is what the prediction can depend on (RP28: full real reach declared)",
                 "target": contract.targets, "features": contract.features, "target_history_as_feature": True,
                 "family_usable_windows": task["usable_windows_all_targets_valid"]},
        "contract": json.loads(json.dumps(contract.__dict__)), "contract_sha256": contract.sha256(),
        "dev_subpartition": {"rows": dev_rows, "rows_note": f"the last {dev_train_days} days of the family's train split and the first {dev_val_days} days "
                                                            f"of its validation split (family train end row {train_end}); test rows never read",
                             "train_days": dev_train_days, "validation_days": dev_val_days, "splits_within_slice": contract.splits,
                             "purge_between_splits": window + horizon},
        "graph": {"arch": "A", "fusion": "sequence", "core_kind": core_kind, "assignment": assignment, "input_columns": cols,
                  "core": ("Conv1D(16, k=3, causal, ELU), reach 3 positions" if core_kind == "conv3" else
                           f"causal dilated Conv1D stack with dilations {core_dilations(window)} (kernel 3, 16 filters, "
                           f"residual), reach {core_reach(window, 'tcn_w')} positions: derived from W, not searched"),
                  "grouping": "held constant: physical groups (power: active history, reactive, intensity | voltage | sub-metering 1-3); "
                              "profile-based grouping is a separate factor not varied here",
                  "layers": {"detector (per branch)": "2 residual causal Conv1D blocks: Conv1D(16, k=3, d=1, ELU) + 1x1 projection skip",
                             "integrator": "identity (ARCH-A)", "adapter": "TimeDistributed Dense(8), linear",
                             "core": "Conv1D(16, k=3, causal, ELU) on the channel-concatenated branch sequences, last position",
                             "head": "Dense(p) increment over the last observation (persistence skip); the target channel is read out",
                             "readout": "the model output at the target's input channel; only that channel is trained and scored"},
                  "loss": "mse on the train-only-scaled target", "optimizer": f"adam lr {learning_rate}",
                  "initial_weights": "keras set_random_seed(seed) before build: the SAME initial checkpoint for R0/R1/R2 of a seed (digest recorded per unit)",
                  "regimes": {"R0": "random detector, trainable", "R1": "detector from ae_s (masked AE, train windows only), frozen",
                              "R2": "detector from ae_s, adjustable (same imported weights as R1)"}},
        "pretraining": {"objective": "masked reconstruction (mask ratio %.2f of time x channel positions) on DEV TRAIN windows; the internal "
                                     "validation is a PURGED TAIL of those same train origins, so the supervised DEV validation is never "
                                     "consumed by pre-training; decoder separate and never connected at inference" % mask_ratio,
                        "mask_ratio": mask_ratio, "max_updates": ae_updates, "patience_epochs": patience_epochs,
                        "internal_validation_fraction": internal_validation_fraction,
                        "internal_validation_purge": "window + horizon rows between the pre-training origins and the internal validation ones",
                        "validation_bank_seed": validation_bank_seed,
                        "validation_mask_policy": "FIXED BANK by window identity for validation; epoch-varying noise for training, as declared"},
        "training": {"batch": batch, "max_updates": max_updates, "learning_rate": learning_rate, "loss": "mse",
                     "early_stopping": {"monitor": "validation loss (mse)", "patience_epochs": patience_epochs, "restore_best": True},
                     "epoch_note": "an epoch is one pass over the DEV train origins in batches; the update counter binds exactly",
                     "same_task_budget": "every regime gets the same update ceiling; the AE cost is reported apart and added for the total-cost reading"},
        "controls": {"persistence": "y_hat(t+h) = y(t)", "seasonal_naive_daily": "y_hat(t+h) = y(t+h-1440)",
                     "linear_ridge": f"ridge (lambda {ridge_lambda} x n_train on standardised inputs) on the flattened scaled window (W x p) + bias, fitted on DEV train windows",
                     "linear_reach": "the same ridge on the last `model_reach_steps` rows only: the information a local receiver can use",
                     "support": CONTROL_SUPPORT},
        "metrics": {"mae": "kW on the common evaluation set", "mase": "MAE / mean |y(t+h) - y(t)| over DEV train origins (persistence at the horizon)",
                    "common_evaluation_set": "DEV validation origins admissible for the model AND with a finite label AND a finite daily lookup; identical for every unit",
                    "test": "NOT SCORED (the benchmark's final test stays unscored)"},
        "replicas": {"seeds": list(seeds), "pairing": "R0/R1/R2 of a seed share the initial checkpoint; R1/R2 share the AE of the seed",
                     "purpose": "optimisation variability, not a power calculation"},
        "budget": {"headroom": headroom, "pilot_updates": pilot_updates, "task_memory_bytes": task_memory_bytes, "wall_seconds": wall_seconds,
                   "cpu_seconds": cpu_seconds, "memory_preflight_cap_bytes": MEMORY_PREFLIGHT_CAP,
                   "rule": "projection = sum over remaining units of (seconds/update x update ceiling + overhead) x (1 + headroom); "
                           "if spent + projection exceeds the ceiling the pilot stops BUDGET_LIMITED before any outcome (arms/seeds are never cut after results)"},
        "cells": cells, "pilots": pilots, "ridge_lambda": ridge_lambda,
        "governance": {"disposition": "LOCAL_TERMINALS_PUBLIC_PANEL_NOT_A_LAKE_RESOURCE",
                       "why": "the deployed data-gov serves no lake holding the public panels; DATASETS campaigns complete units only through deliveries of "
                              "every declared dataset; SYNTHETIC would misdeclare real bytes",
                       "missing_object": "one lake registration of public_panels_c126_v2 in data-gov (config + service restart: an owner action)",
                       "what_is_recorded": "governed_terminal.v1 documents per unit in the run root + CAMPAIGN_PROPOSAL.json"},
    }
    design["design_sha256"] = E.sha_obj(design)
    return design


# --- prepare -----------------------------------------------------------------------------------------------

def prepare(design: dict, root: Path) -> dict:
    """Prepare the run's tensors from the panel. Where the panel comes from is decided by the design:
    `data_access: "GOVERNED_DELIVERY"` reads ONLY the bytes data-gov delivered to this run (RP33), and
    refuses to start when there is none; the historical setting reads the characterised file in place
    and is what the preserved 2026-09-19 pilot used."""
    import pandas as pd
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    data_json, data_npz = root / "DATA.json", root / "DATA.npz"
    access = design.get("data_access", "LOCAL_CHARACTERISED_FILE")
    if data_json.is_file():
        rec = json.loads(data_json.read_text())
        if rec["design_sha256"] != design["design_sha256"] or sha_file(data_npz) != rec["data_sha256"]:
            raise SystemExit("REFUSED: DATA in this root belongs to another design or was altered")
        if access == "GOVERNED_DELIVERY":
            # RP42: a cached DATA file is not a licence to skip the delivery. The bytes it was built
            # from must still be the ones this run was delivered, now.
            G = _load("df_e1_governed")
            delivered = G.require_delivery(root, design)["delivery"]
            if delivered.get("sha256") != rec.get("panel_sha256"):
                raise G.GovernanceUnavailable(
                    f"REFUSED: the cached DATA came from {rec.get('panel_sha256')} but this run was delivered "
                    f"{delivered.get('sha256')}")
            rec["delivery_recheck"] = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                       "delivery_id": delivered.get("delivery_id")}
        return rec
    t0 = time.process_time()
    delivery = None
    if access == "GOVERNED_DELIVERY":
        G = _load("df_e1_governed")
        delivery = G.require_delivery(root, design)
        panel = Path(delivery["delivery"]["path"])
    else:
        panel = Path(design["governed_bytes"]["path"]).expanduser()
    digest = sha_file(panel)
    if digest != design["governed_bytes"]["sha256"]:
        raise SystemExit("REFUSED: the panel's bytes are not the governed bytes of E1_TASKS")
    lo, hi = design["dev_subpartition"]["rows"]
    frame = pd.read_parquet(panel)
    n_panel = len(frame)
    frame = frame.iloc[lo:hi].reset_index(drop=True)
    c = L.TaskContract(**design["contract"])
    resolved = L.resolve(frame, c)
    enum = L.enumerate_windows(resolved, c)
    if c.sha256() != design["contract_sha256"] or enum["contract_sha256"] != design["contract_sha256"]:
        raise SystemExit("REFUSED: contract digest disagreement")
    W, h, p = enum["window"], enum["horizon"], len(c.input_columns())
    tr, va = enum["splits"]["train"], enum["splits"]["validation"]
    # memory preflight: the scaled input slice + a materialised TRAIN tensor for the scaler (the only materialisation, train only)
    bytes_slice = int(frame.shape[0] * p * 4)
    bytes_train_tensor = int(tr["admissible"] * W * p * 8)
    preflight = {"slice_bytes": bytes_slice, "train_tensor_bytes_for_scaler": bytes_train_tensor, "cap_bytes": MEMORY_PREFLIGHT_CAP,
                 "batch_window_bytes": int(design["training"]["batch"] * W * p * 4),
                 "policy": "windows are gathered per batch from the scaled slice; only the train tensor for the scaler is materialised once, here"}
    if bytes_slice + bytes_train_tensor > MEMORY_PREFLIGHT_CAP:
        raise SystemExit(f"REFUSED: memory preflight {bytes_slice + bytes_train_tensor} B exceeds {MEMORY_PREFLIGHT_CAP} B")
    Ttr = L.build_tensors(resolved, enum, "train", c, None)
    scaler = L.fit_scaler(Ttr)
    del Ttr
    X = resolved["inputs"].astype(np.float64)
    Y = resolved["targets"][:, 0].astype(np.float64)
    Xs = ((X - scaler["mean"]) / scaler["sd"]).astype(np.float32)
    j_t = c.input_columns().index(c.targets[0])
    tr_o, va_o = np.asarray(tr["origin_ids"], dtype=np.int64), np.asarray(va["origin_ids"], dtype=np.int64)
    tr_m, va_m = np.asarray(tr["target_mask"], dtype=bool)[:, 0], np.asarray(va["target_mask"], dtype=bool)[:, 0]
    tr_o = tr_o[tr_m]                                                       # train windows with a finite label
    denom = float(np.mean(np.abs(Y[tr_o + h] - Y[tr_o])))                    # persistence at the horizon, train only
    lookup = va_o + h - DAY
    seasonal_ok = (lookup >= 0)
    seasonal_ok[seasonal_ok] = np.isfinite(Y[lookup[seasonal_ok]])
    ev = va_o[va_m & seasonal_ok]
    coverage = {"validation_admissible": int(va_o.size), "with_finite_label": int(va_m.sum()), "with_finite_daily_lookup": int((va_m & seasonal_ok).sum()),
                "common_evaluation_set": int(ev.size)}
    np.savez(data_npz, Xs=Xs, Y=Y, train_origins=tr_o, eval_origins=ev, denominator=np.array([denom]), scaler_mean=scaler["mean"], scaler_sd=scaler["sd"],
             target_channel=np.array([j_t]), window=np.array([W]), horizon=np.array([h]))
    rec = {"schema": SCHEMA_DATA, "design_sha256": design["design_sha256"], "panel_sha256": digest, "panel_rows": n_panel, "slice_rows": [lo, hi],
           "data_access": access, "delivery": (delivery or {}).get("delivery"),
           "campaign_sha256": (delivery or {}).get("campaign_sha256"),
           "contract_sha256": c.sha256(), "input_columns": c.input_columns(), "target_channel": j_t, "window": W, "horizon": h, "p": p,
           "enumerator": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in enum["splits"].items()},
           "grid_rows_not_ok": enum["grid_rows_not_ok"], "input_rows_non_finite": enum["input_rows_non_finite"],
           "scaler": {"mean": scaler["mean"].tolist(), "sd": scaler["sd"].tolist(), "fitted_on": scaler["fitted_on"], "n_rows": scaler["n_rows"]},
           "train_origins_with_label": int(tr_o.size), "mase_denominator_persistence_h": denom, "coverage": coverage,
           "memory_preflight": preflight, "cpu_seconds": round(time.process_time() - t0, 3)}
    rec["data_sha256"] = sha_file(data_npz)
    data_json.write_text(json.dumps(rec, indent=1))
    return rec


# --- batched windows -----------------------------------------------------------------------------------------

def _gather(Xs: np.ndarray, origins: np.ndarray, W: int) -> np.ndarray:
    idx = origins[:, None] - W + 1 + np.arange(W)[None, :]
    return Xs[idx]


def _mask_for(origins: np.ndarray, W: int, p: int, ratio: float, *, bank_seed: int, epoch: int | None) -> np.ndarray:
    """The mask of each window, derived from the WINDOW'S OWN IDENTITY (its origin), never from its
    position in a batch (RP36, dictum F3).

    `epoch=None` is the fixed VALIDATION BANK: the same origin always receives the same mask, so the
    criterion a stopping rule compares across epochs is the same stimulus, and a reordering, a resume
    or a second evaluation cannot move it. An integer `epoch` is TRAINING noise, which the design
    declares as varying: it still depends on the window's identity, so shuffling does not change what
    a given window sees within an epoch.
    """
    out = np.empty((origins.size, W, p), dtype=bool)
    for k, origin in enumerate(origins):
        key = [int(bank_seed), int(origin)] if epoch is None else [int(bank_seed), int(epoch), int(origin)]
        out[k] = np.random.default_rng(key).random((W, p)) < ratio
    return out


def mask_bank_digest(origins: np.ndarray, W: int, p: int, ratio: float, *, bank_seed: int) -> str:
    """The identity of a validation bank: its origins and the masks they receive."""
    m = _mask_for(np.asarray(origins, dtype=np.int64), W, p, ratio, bank_seed=bank_seed, epoch=None)
    h = hashlib.sha256()
    h.update(np.asarray(origins, dtype=np.int64).tobytes())
    h.update(m.tobytes())
    h.update(json.dumps({"W": W, "p": p, "ratio": ratio, "bank_seed": bank_seed}, sort_keys=True).encode())
    return h.hexdigest()


def _dataset_class():
    tf = E._tf()

    class WindowBatches(tf.keras.utils.PyDataset):
        """Windows gathered per batch from the scaled slice (never materialised).

        `masked` turns it into the auto-encoder's stream. `mask_mode` says which stimulus it serves:
        "bank" is the fixed validation bank (invariant to epoch, batch order, resume and repetition)
        and "training" is the declared training noise. Shuffling only reorders windows; it never
        changes what a window is asked to reconstruct.
        """
        def __init__(self, Xs, Y, origins, W, h, j_t, batch, *, scaler_mean, scaler_sd, shuffle, seed, masked=None,
                     mask_mode="training", bank_seed=20260919):
            super().__init__(workers=1, use_multiprocessing=False)
            self.Xs, self.Y, self.origins, self.W, self.h, self.j, self.batch = Xs, Y, np.asarray(origins), W, h, j_t, batch
            self.m, self.s = float(scaler_mean[j_t]), float(scaler_sd[j_t])
            self.shuffle, self.seed, self.epoch, self.masked = shuffle, seed, 0, masked
            self.mask_mode, self.bank_seed = mask_mode, bank_seed
            if masked is not None and mask_mode not in ("bank", "training"):
                raise SystemExit(f"unknown mask mode {mask_mode!r}")
            self.perm = np.arange(self.origins.size)
            self._reshuffle()

        def _reshuffle(self):
            if self.shuffle:
                self.perm = np.random.default_rng([self.seed, self.epoch]).permutation(self.origins.size)

        def __len__(self):
            return math.ceil(self.origins.size / self.batch)

        def __getitem__(self, i):
            o = self.origins[self.perm[i * self.batch:(i + 1) * self.batch]]
            X = _gather(self.Xs, o, self.W)
            if self.masked is not None:                                           # AE: masked input -> [x, mask] target
                epoch = None if self.mask_mode == "bank" else self.epoch
                seed = self.bank_seed if self.mask_mode == "bank" else self.seed
                m = _mask_for(o, self.W, X.shape[2], self.masked, bank_seed=seed, epoch=epoch)
                return np.where(m, 0.0, X).astype(np.float32), np.concatenate([X, m.astype(np.float32)], axis=2)
            y = ((self.Y[o + self.h] - self.m) / self.s).astype(np.float32)[:, None]
            return X, y

        def on_epoch_end(self):
            self.epoch += 1
            self._reshuffle()
    return WindowBatches


def core_dilations(window: int) -> list:
    """RP37: the dilations a causal kernel-3 stack needs for its reach to cover `window`, derived from
    W and nothing else. Reach of a stack with dilations d1..dk is 1 + 2*sum(d), so powers of two are
    added until that reaches W. No search, no tuning: the rule is the window."""
    dils, reach = [], 1
    d = 1
    while reach < window:
        dils.append(d)
        reach += 2 * d
        d *= 2
    return dils


def core_reach(window: int, core: str) -> int:
    """Samples of the FUSED sequence the readout can depend on, per core."""
    if core == "conv3":
        return 3
    if core == "tcn_w":
        return min(window, 1 + 2 * sum(core_dilations(window)))
    raise SystemExit(f"unknown core {core!r}")


def model_reach(window: int, core: str, arch: str = "A") -> int:
    """Branch reach and core reach compose: the branch's last position sees `branch` samples and the
    readout sees `core` positions of the branch output."""
    branch = E.branch_reach(arch, window)
    return min(window, branch + core_reach(window, core) - 1)


def _model_for_target(assignment, W, p, j_t, seed, core: str = "conv3"):
    """ARCH-A's detector and adapter are PRESERVED; only the common core changes, and only between the
    two declared options: `conv3` (the local kernel-3 core of E0) and `tcn_w` (a causal dilated stack
    whose reach covers W). The core is never pre-trained here: H-CORE remains a later question."""
    tf = E._tf()
    tf.keras.utils.set_random_seed(int(seed))
    if core == "conv3":
        m = E.build_modular(assignment, W, p, fusion="sequence", seed=seed, arch="A")
        out = tf.keras.layers.Lambda(lambda t: t[:, j_t:j_t + 1], name="target_readout")(m.output)
        return tf.keras.Model(m.input, out, name="modular_A_conv3_target")
    if core != "tcn_w":
        raise SystemExit(f"unknown core {core!r}")
    inp = tf.keras.Input(shape=(W, p), name="x")
    groups = sorted(set(assignment))
    branches = []
    for g in groups:
        idx = [k for k in range(p) if assignment[k] == g]
        sub = tf.keras.layers.Lambda(lambda t, idx=idx: tf.gather(t, idx, axis=2), name=f"g{g}_select")(inp)
        branches.append(E.branch_extractor(tf, sub, f"g{g}", "A"))          # the SAME detector and adapter as ARCH-A
    joint = tf.keras.layers.Concatenate(axis=2, name="fusion_seq")(branches) if len(branches) > 1 else branches[0]
    h = joint
    for i, d in enumerate(core_dilations(W), start=1):
        h = E._tcn_block(tf, h, f"core_tcn{i}", d)
    read = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="core_last")(h)
    delta = tf.keras.layers.Dense(p, name="head")(read)
    last_x = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="last_observation")(inp)
    full = tf.keras.layers.Add(name="persistence_skip")([last_x, delta])
    out = tf.keras.layers.Lambda(lambda t: t[:, j_t:j_t + 1], name="target_readout")(full)
    return tf.keras.Model(inp, out, name="modular_A_tcnw_target")


def _fit_batched(model, train_ds, val_ds, *, max_updates, patience, lr, seed) -> dict:
    tf = E._tf()
    tf.keras.utils.set_random_seed(int(seed))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    steps = len(train_ds)
    max_epochs = max(1, math.ceil(max_updates / steps))

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
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(patience), restore_best_weights=True)
    t0 = time.process_time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=max_epochs, verbose=0, callbacks=[counter, es])
    fit_s = time.process_time() - t0
    va = [float(v) for v in hist.history["val_loss"]]
    stop = "UPDATE_BUDGET" if counter.budget_stop else ("EARLY_STOPPING" if es.stopped_epoch else "EPOCH_BUDGET")
    restored = float(model.evaluate(val_ds, verbose=0))
    return {"updates": int(counter.updates), "epochs": len(va), "steps_per_epoch": steps, "max_epochs_allowed": max_epochs, "max_updates": int(max_updates),
            "curve": {"train": [float(v) for v in hist.history["loss"]], "validation": va}, "stop_reason": stop,
            "restored_checkpoint_epoch": int(np.argmin(va)) + 1, "restored_validation_loss": restored,
            "restore_verified": bool(abs(restored - min(va)) <= 1e-4 * max(1.0, abs(min(va)))),
            # RP36: reaching the budget IS optimisation censoring, whatever the argmin says. The reading
            # reports the slope at the end, how long the run lasted and the adequacy criterion; it never
            # concludes "not truncated" from a best epoch that happens not to be the last.
            "censoring": {"budget_reached": bool(counter.budget_stop),
                          "stopped_by": stop,
                          "best_epoch": int(np.argmin(va)) + 1, "epochs": len(va),
                          "validation_slope_last_two": (float(va[-1] - va[-2]) if len(va) > 1 else None),
                          "validation_slope_last_third": (float(va[-1] - va[max(0, int(len(va) * 2 / 3) - 1)])
                                                          if len(va) > 2 else None),
                          "improvement_since_best": (float(va[-1] - min(va)) if va else None),
                          "updates_used_of_ceiling": [int(counter.updates), int(max_updates)],
                          "adequacy_criterion": "a fit is adequate for comparison when it stopped by EARLY_STOPPING with a "
                                                "non-improving slope over the last third; a run that stopped at the update "
                                                "ceiling is CENSORED and its comparison is a lower bound on what the arm could reach",
                          "verdict": ("CENSORED_BY_BUDGET" if counter.budget_stop else
                                      "STOPPED_ON_VALIDATION" if es.stopped_epoch else "EPOCH_BUDGET_REACHED")},
            "fit_seconds": round(fit_s, 3)}


def _load_data(job: dict) -> dict:
    path = Path(job["data_npz"])
    if sha_file(path) != job["data_sha256"]:
        raise SystemExit("REFUSED: DATA.npz is not the prepared bytes the design bound")
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    d["W"], d["h"], d["j"] = int(d["window"][0]), int(d["horizon"][0]), int(d["target_channel"][0])
    return d


def _predict(model, ds) -> np.ndarray:
    return np.concatenate([np.asarray(model.predict_on_batch(ds[i][0])) for i in range(len(ds))], axis=0)[:, 0]


# --- units ----------------------------------------------------------------------------------------------------

def run_unit(job: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = _load_data(job)
    design = job["design"]
    W, h, j, p = d["W"], d["h"], d["j"], d["Xs"].shape[1]
    batch, lr = int(design["training"]["batch"]), float(design["training"]["learning_rate"])
    patience = int(design["training"]["early_stopping"]["patience_epochs"])
    assignment = design["graph"]["assignment"]
    WB = _dataset_class()
    kw = dict(scaler_mean=d["scaler_mean"], scaler_sd=d["scaler_sd"])
    cost = {}
    rec = {"schema": SCHEMA_CELL, "cell_id": job["cell_id"], "kind": job["kind"], "seed": int(job["seed"]), "design_sha256": design["design_sha256"],
           "data_sha256": job["data_sha256"], "task": design["task"]["id"], "arch": "A", "fusion": "sequence", "exposure": "NO_TEST_ACCESS",
           "common_evaluation_set_size": int(d["eval_origins"].size), "max_updates": int(job.get("max_updates", 0)), "role": job.get("role", "CELL")}
    if job["kind"] == "ae":
        tf = E._tf()
        tf.keras.utils.set_random_seed(int(job["seed"]))
        ae, dec_names = RG.build_autoencoder(assignment, W, p, arch="A", seed=int(job["seed"]), mask_ratio=design["pretraining"]["mask_ratio"])

        def masked_mse(y_true, y_pred):
            x, m = y_true[..., :p], y_true[..., p:]
            return tf.reduce_sum(m * tf.square(x - y_pred)) / (tf.reduce_sum(m) + 1e-8)
        ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=masked_mse)
        # RP36: the pre-training's internal validation is a PURGED TAIL OF THE TRAIN ORIGINS. The
        # supervised DEV validation is never read here, so selecting the auto-encoder cannot consume
        # the split the regimes are later compared on.
        tr_all = np.asarray(d["train_origins"], dtype=np.int64)
        frac = float(design["pretraining"]["internal_validation_fraction"])
        cut = max(1, int(round(tr_all.size * (1.0 - frac))))
        purge = W + h
        ae_tr = tr_all[:max(1, cut - purge)]
        ae_va = tr_all[cut:]
        ratio = design["pretraining"]["mask_ratio"]
        bank_seed = int(design["pretraining"]["validation_bank_seed"])
        tr_ds = WB(d["Xs"], d["Y"], ae_tr, W, h, j, batch, shuffle=True, seed=int(job["seed"]), masked=ratio,
                   mask_mode="training", **kw)
        va_ds = WB(d["Xs"], d["Y"], ae_va, W, h, j, 256, shuffle=False, seed=0, masked=ratio,
                   mask_mode="bank", bank_seed=bank_seed, **kw)
        steps = len(tr_ds)
        max_updates = int(job["max_updates"])
        max_epochs = max(1, math.ceil(max_updates / steps))
        counter = {"updates": 0, "stop": False}

        class Stop(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, b, logs=None):
                counter["updates"] += 1
                if counter["updates"] >= max_updates:
                    counter["stop"] = True
                    self.model.stop_training = True
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        t0 = time.process_time()
        hist = ae.fit(tr_ds, validation_data=va_ds, epochs=max_epochs, verbose=0, callbacks=[Stop(), es])
        cost["pretrain_seconds"] = round(time.process_time() - t0, 3)
        det = RG.detector_layer_names(ae)
        np.savez(out_dir / "detector_pretrained.npz", **{f"{n}__{i}": w for n in det for i, w in enumerate(ae.get_layer(n).get_weights())})
        np.savez(out_dir / "decoder.npz", **{f"{n}__{i}": w for n in dec_names for i, w in enumerate(ae.get_layer(n).get_weights())})
        va = [float(v) for v in hist.history["val_loss"]]
        t1 = time.process_time()
        rec_val = float(ae.evaluate(va_ds, verbose=0))
        rec_val_again = float(ae.evaluate(va_ds, verbose=0))
        cost["metrics_seconds"] = round(time.process_time() - t1, 3)
        stop_reason = "UPDATE_BUDGET" if counter["stop"] else ("EARLY_STOPPING" if es.stopped_epoch else "EPOCH_BUDGET")
        best = float(min(va)) if va else None
        rec.update(pretraining={"updates": counter["updates"], "epochs": len(va), "steps_per_epoch": steps, "stop_reason": stop_reason,
                                "curve": {"train": [float(v) for v in hist.history["loss"]], "validation": va},
                                "best_epoch": int(np.argmin(va)) + 1, "best_validation_loss": best,
                                "reconstruction_val_mse_masked": rec_val,
                                "restored_reproduces_best": bool(best is not None and abs(rec_val - best) <= 1e-4 * max(1.0, abs(best))),
                                "evaluation_is_repeatable": bool(abs(rec_val - rec_val_again) <= 1e-9),
                                "internal_validation": "a PURGED TAIL of the DEV train origins; the supervised DEV validation "
                                                       "and the final test are never read during pre-training",
                                "internal_validation_windows": int(ae_va.size), "pretrain_windows": int(ae_tr.size),
                                "purge_between": int(purge), "mask_ratio": ratio,
                                "validation_mask": "a FIXED BANK keyed by the window's origin: invariant to epoch, batch order, "
                                                   "resume and repeated evaluation (training noise varies by epoch, as declared)",
                                "validation_bank_sha256": mask_bank_digest(ae_va, W, p, ratio, bank_seed=bank_seed),
                                "decoder_layers": dec_names, "detector_layers": det, "detector_digest": RG.weights_digest(ae, det),
                                "decoder_never_at_inference": True, "diagnostic_only": True},
                   output_files={"detector": "detector_pretrained.npz", "decoder": "decoder.npz"},
                   detector_sha256=sha_file(out_dir / "detector_pretrained.npz"),
                   training={"updates": counter["updates"], "stop_reason": stop_reason},
                   parameters=E.count_params(ae), scores={"validation": {"model": {"mase_mean": None, "mae_mean": None, "status": "NO_APLICA"}}})
        ae.save_weights(str(out_dir / "ae.weights.h5"))
        np.savez(out_dir / "arrays.npz", denominator=d["denominator"], eval_origins=d["eval_origins"],
                 ae_validation_origins=ae_va, ae_train_origins=ae_tr)
    elif job["kind"] == "fit":
        regime = job["regime"]
        model = _model_for_target(assignment, W, p, j, int(job["seed"]), core=design["graph"].get("core_kind", "conv3"))
        det = RG.detector_layer_names(model)
        rec["initial_checkpoint"] = {"seed": int(job["seed"]), "full_digest": RG.weights_digest(model, det + RG.non_detector_weighted_layer_names(model)),
                                     "detector_digest": RG.weights_digest(model, det), "note": "digest of the model right after build, BEFORE any import: shared by R0/R1/R2 of the seed"}
        t0 = time.process_time()
        info = RG.apply_regime(model, regime, Path(job["pretrained_npz"]) if regime != "R0" else None)
        cost["reuse_seconds"] = round(time.process_time() - t0, 3)
        if regime != "R0":
            rec["pretrained_sha256"] = sha_file(Path(job["pretrained_npz"]))
        tr_ds = WB(d["Xs"], d["Y"], d["train_origins"], W, h, j, batch, shuffle=True, seed=int(job["seed"]), **kw)
        va_ds = WB(d["Xs"], d["Y"], d["eval_origins"], W, h, j, 256, shuffle=False, seed=0, **kw)
        Xb, yb = tr_ds[0]
        grad = RG.gradient_report(model, Xb, yb)
        before = {n: [w.copy() for w in model.get_layer(n).get_weights()] for n in det + RG.non_detector_weighted_layer_names(model)}
        training = _fit_batched(model, tr_ds, va_ds, max_updates=int(job["max_updates"]), patience=patience, lr=lr, seed=int(job["seed"]))
        cost["fit_seconds"] = training["fit_seconds"]
        change = {n: float(sum(np.linalg.norm(a - b) for a, b in zip(model.get_layer(n).get_weights(), before[n]))) for n in before}
        t1 = time.process_time()
        pred_s = _predict(model, va_ds)
        cost["inference_seconds"] = round(time.process_time() - t1, 3)
        t2 = time.process_time()
        pred = pred_s * float(d["scaler_sd"][j]) + float(d["scaler_mean"][j])
        y = d["Y"][d["eval_origins"] + h]
        scores = E.mase(pred[:, None], y[:, None], d["denominator"].tolist())
        cost["metrics_seconds"] = round(time.process_time() - t2, 3)
        model.save_weights(str(out_dir / "weights.weights.h5"))
        np.savez(out_dir / "arrays.npz", validation_pred=pred[:, None], validation_y=y[:, None], denominator=d["denominator"], eval_origins=d["eval_origins"])
        rec.update(regime=regime, regime_setup={k: v for k, v in info.items() if k != "description"}, gradient_proof={k: v for k, v in grad.items() if k != "per_variable"},
                   training=training, weight_change_by_layer=change, detector_digest_after_fit=RG.weights_digest(model, det),
                   detector_unchanged=bool(RG.weights_digest(model, det) == info["detector_digest_after_setup"]), parameters=info["params"],
                   scores={"validation": {"model": scores}})
    elif job["kind"] == "controls":
        t0 = time.process_time()
        Y, ev = d["Y"], d["eval_origins"]
        y = Y[ev + h]
        preds = {"persistence": Y[ev], "seasonal_naive_daily": Y[ev + h - DAY]}
        reach = int(design["task"]["model_reach_steps"])
        tr = d["train_origins"]
        m_t, s_t = float(d["scaler_mean"][j]), float(d["scaler_sd"][j])
        nfeat = W * p + 1
        G, b = np.zeros((nfeat, nfeat)), np.zeros(nfeat)
        for i in range(0, tr.size, 1024):
            o = tr[i:i + 1024]
            Xf = np.concatenate([_gather(d["Xs"], o, W).reshape(o.size, -1).astype(np.float64), np.ones((o.size, 1))], axis=1)
            yt = (Y[o + h] - m_t) / s_t
            G += Xf.T @ Xf
            b += Xf.T @ yt
        lam = float(design["ridge_lambda"]) * tr.size
        reg = lam * np.eye(nfeat)
        reg[-1, -1] = 0.0
        beta = np.linalg.solve(G + reg, b)
        Xe = np.concatenate([_gather(d["Xs"], ev, W).reshape(ev.size, -1).astype(np.float64), np.ones((ev.size, 1))], axis=1)
        preds["linear_ridge"] = (Xe @ beta) * s_t + m_t
        # the information-matched linear control: only the rows a local receiver can reach
        nfeat_r = reach * p + 1
        Gr, br = np.zeros((nfeat_r, nfeat_r)), np.zeros(nfeat_r)
        for i in range(0, tr.size, 1024):
            o = tr[i:i + 1024]
            Xf = np.concatenate([_gather(d["Xs"], o, W)[:, W - reach:, :].reshape(o.size, -1).astype(np.float64),
                                 np.ones((o.size, 1))], axis=1)
            Gr += Xf.T @ Xf
            br += Xf.T @ ((Y[o + h] - m_t) / s_t)
        reg_r = float(design["ridge_lambda"]) * tr.size * np.eye(nfeat_r)
        reg_r[-1, -1] = 0.0
        beta_r = np.linalg.solve(Gr + reg_r, br)
        Xer = np.concatenate([_gather(d["Xs"], ev, W)[:, W - reach:, :].reshape(ev.size, -1).astype(np.float64),
                              np.ones((ev.size, 1))], axis=1)
        preds["linear_reach"] = (Xer @ beta_r) * s_t + m_t
        cost["fit_seconds"] = round(time.process_time() - t0, 3)
        t1 = time.process_time()
        scores = {k: E.mase(v[:, None], y[:, None], d["denominator"].tolist()) for k, v in preds.items()}
        cost["metrics_seconds"] = round(time.process_time() - t1, 3)
        np.savez(out_dir / "arrays.npz", validation_y=y[:, None], denominator=d["denominator"], eval_origins=ev, **{f"validation_pred_{k}": v[:, None] for k, v in preds.items()})
        rec.update(scores={"validation": {k: v for k, v in scores.items()}},
                   ridge={"lambda_scaled": lam, "features": nfeat, "reach_features": nfeat_r, "reach_steps": reach},
                   control_support=CONTROL_SUPPORT, training={"updates": 0, "stop_reason": "CLOSED_FORM"},
                   parameters={"total": nfeat, "trainable": nfeat, "frozen": 0})
    else:
        raise SystemExit(f"unknown unit kind {job['kind']!r}")
    rec["cost"] = {**cost, "cpu_seconds_process": round(time.process_time(), 3)}
    rec["arrays_sha256"] = sha_file(out_dir / "arrays.npz")
    body = json.dumps(rec, indent=1, sort_keys=True, default=float)
    (out_dir / "cell.json").write_text(body)
    return rec


def worker_main(job_file: Path) -> int:
    job = json.loads(Path(job_file).read_text())
    adir = Path(job["attempt_dir"])
    run_unit(job, adir)
    body = (adir / "cell.json").read_bytes()
    result = {"status": "COMPLETED", "reason": "", "output_file": "cell.json", "output_sha256": hashlib.sha256(body).hexdigest(),
              "rows_written": body.count(b"\n") + (0 if body.endswith(b"\n") else 1), "outcome": "COMPLETED"}
    tmp = adir / "result.json.tmp"
    tmp.write_text(json.dumps(result))
    os.replace(tmp, adir / "result.json")
    return 0


# --- verification, terminals, coordinator --------------------------------------------------------------------

def verified_unit(attempt_dir: Path, result: dict, verified: dict) -> tuple:
    """RP43: kept as the thin bytes-and-schema gate ONLY; the scientific verdict is the closure's
    (tools/df_e1_close.py), which run_isolated calls below so a resumed or fresh unit is judged by the
    same code that closes it. Nothing here decides that a score may be consumed."""
    name = (result or {}).get("output_file")
    path = Path(attempt_dir) / str(name)
    if not name or not path.is_file():
        return None, {"outcome": "SCORE_UNVERIFIED", "why": f"{name!r} absent"}
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != result.get("output_sha256") or digest != (verified or {}).get("output_sha256"):
        return None, {"outcome": "SCORE_UNVERIFIED", "why": "the output's bytes are not the ones the child declared and the runner verified"}
    rec = json.loads(body)
    if rec.get("schema") != SCHEMA_CELL:
        return None, {"outcome": "SCORE_UNVERIFIED", "why": f"schema {rec.get('schema')!r}"}
    arrays = Path(attempt_dir) / "arrays.npz"
    if not arrays.is_file() or sha_file(arrays) != rec.get("arrays_sha256"):
        return None, {"outcome": "SCORE_UNVERIFIED", "why": "arrays absent or altered"}
    z = np.load(arrays)
    if rec["kind"] == "fit":
        a = E.mase(z["validation_pred"], z["validation_y"], z["denominator"].tolist())["mase_mean"]
        b = rec["scores"]["validation"]["model"]["mase_mean"]
        if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):
            return None, {"outcome": "SCORE_UNVERIFIED", "why": "validation MASE in the record is not the one recomputed from the arrays"}
    elif rec["kind"] == "controls":
        for k in CONTROLS:
            a = E.mase(z[f"validation_pred_{k}"], z["validation_y"], z["denominator"].tolist())["mase_mean"]
            if abs(a - rec["scores"]["validation"][k]["mase_mean"]) > 1e-9:
                return None, {"outcome": "SCORE_UNVERIFIED", "why": f"{k} MASE not reproduced from the arrays"}
    return rec, None


def _closure_verdict(attempt_dir: Path, job: dict, score: dict) -> tuple:
    """RP43: the SAME verification the closure runs, applied the moment a unit finishes or is resumed,
    so a score is never consumed on a weaker check than the one that will judge it later."""
    CL = _load("df_e1_close")
    root = Path(attempt_dir).parent.parent
    try:
        register = CL.register(root)
    except BaseException as exc:                                # noqa: BLE001
        return None, {"outcome": "SCORE_UNVERIFIED", "why": f"the run's register is unreadable: {exc}"[:200]}
    if register["problems"]:
        return None, {"outcome": "SCORE_UNVERIFIED", "why": f"the run's register refuses: {register['problems'][:2]}"}
    verdict = CL.verify_unit(register, job["cell_id"], do_replay=False)
    if verdict["metrics"] != CL.VERIFIED or verdict["problems"]:
        return None, {"outcome": "SCORE_UNVERIFIED", "why": "; ".join(verdict["problems"][:3]) or "metrics refused",
                      "closure": {k: verdict[k] for k in ("metrics", "inference", "regime", "scope")}}
    return score, None


def run_isolated(job: dict, *, attempt_dir: Path, assigned_bytes: int, wall_seconds: float, cpu_seconds: float) -> dict:
    H = _load("df_utility_harness")
    IR = _load("df_isolated_runner")
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    prior = attempt_dir / "outcome.json"
    if prior.is_file():
        recorded = json.loads(prior.read_text())
        refusal = H._job_binding_refusal(attempt_dir, job)
        score = None
        if refusal is None and recorded.get("status") == "COMPLETED":
            result = json.loads((attempt_dir / "result.json").read_text()) if (attempt_dir / "result.json").is_file() else None
            score, refusal = verified_unit(attempt_dir, result, recorded.get("verified"))
            if score is not None:
                score, refusal = _closure_verdict(attempt_dir, job, score)
        history = dict(recorded.get("summary") or {})
        if refusal is not None:
            return {"outcome": "SCORE_UNVERIFIED", "reason": refusal["why"], "cost": history.get("cost", {}), "score": None, "resumed": True, "refusal": refusal}
        return {**history, "score": score, "resumed": True}
    job_file = attempt_dir / "job.json"
    job_file.write_text(json.dumps({**job, "attempt_dir": str(attempt_dir)}, default=H._jsonable))
    task = IR.Task(argv=[sys.executable, "-B", str(HERE / "df_e1_pilot.py"), "--worker", str(job_file)], name=f"e1-{job['cell_id']}",
                   attempt_dir=attempt_dir, assigned_bytes=assigned_bytes, wall_seconds=wall_seconds, cpu_seconds=cpu_seconds,
                   mechanism=IR.detect_mechanism(), extra_env={"OMP_NUM_THREADS": "2"})
    task.start()
    task.wait()
    status, reason, verified = IR.classify(task.outcome, attempt_dir)
    result = json.loads((attempt_dir / "result.json").read_text()) if (attempt_dir / "result.json").is_file() else None
    cost = {"cpu_seconds": task.outcome.get("cpu_seconds"), "wall_seconds": task.outcome.get("wall_seconds"), "peak_rss_bytes": task.outcome.get("child_maxrss_bytes"),
            "cgroup_memory_peak": task.outcome.get("cgroup_memory_peak"), "started_at": task.outcome.get("started_at"), "ended_at": task.outcome.get("ended_at"),
            "host": os.uname().nodename}
    if status != "COMPLETED":
        summary = {"outcome": "RESOURCE_EXCEEDED" if status == "RESOURCE_EXCEEDED" else "UNCERTAIN", "reason": reason, "cost": cost, "score": None}
    else:
        score, refusal = verified_unit(attempt_dir, result, verified)
        if score is not None:
            score, refusal = _closure_verdict(attempt_dir, job, score)
        summary = {"outcome": "COMPLETED" if score else "SCORE_UNVERIFIED", "reason": reason, "cost": cost, "score": score, **({"refusal": refusal} if refusal else {})}
    prior.write_text(json.dumps({"status": status, "verified": verified, "summary": {k: v for k, v in summary.items() if k != "score"}}, default=H._jsonable))
    return summary


def _governed_summary(root: Path, report: dict) -> dict:
    """What the governed run can claim: every unit delivered, every terminal accepted, every campaign
    reconciled. A unit missing any of the three is named."""
    deliveries = json.loads((Path(root) / "DELIVERIES.json").read_text())
    per_unit = {}
    for record in report["terminals"]:
        unit = record["unit_id"]
        rec = record.get("reconciliation") or {}
        per_unit[unit] = {"delivered": unit in (deliveries.get("units") or {}),
                          "terminal_accepted": bool(record.get("terminal_sent")) and not record.get("terminal_pending"),
                          "reconciled": bool(rec.get("http") == 200 and not rec.get("missing_units")
                                             and not rec.get("accounting_only") and not rec.get("lake_only"))}
    incomplete = sorted(u for u, v in per_unit.items() if not all(v.values()))
    return {"per_unit": per_unit, "units_incomplete": incomplete,
            "transfer": deliveries.get("transfer"),
            "all_units_governed": not incomplete,
            "rule": "delivered before working, terminal accepted, campaign reconciled: all three or the unit is named"}


def _terminal(unit_id: str, out: dict, design: dict, tags: dict) -> dict:
    R = _load("df_utility_run")
    rec, cost = out.get("score"), out["cost"]
    if out["outcome"] == "RESOURCE_EXCEEDED":
        status, reason = "FAILED", f"{out['outcome']}: {out.get('reason') or ''}"[:300]
    elif rec is not None:
        status, reason = "COMPLETED", None
    else:
        status, reason = "INCONCLUSIVE", f"{out['outcome']}: {out.get('reason') or ''}"[:300]
    rows, states = [], {}

    def add(name, value, unit):
        if value is None or not np.isfinite(value):
            states[name] = "NO_APLICA"
            return
        states[name] = "MEDIDO"
        rows.append(R._metric(name, value, unit, split="validation", horizon=int(design["task"]["horizon_steps"])))
    if rec:
        add("e1.updates", rec.get("training", {}).get("updates"), "count")
        add("e1.params_trainable", (rec.get("parameters") or {}).get("trainable"), "count")
        for k, v in (rec.get("cost") or {}).items():
            add(f"e1.cost.{k}", v, "seconds")
        if rec["kind"] == "fit":
            add("e1.mase_validation", rec["scores"]["validation"]["model"]["mase_mean"], "mase")
            add("e1.mae_validation", rec["scores"]["validation"]["model"]["mae_mean"], "mae")
        elif rec["kind"] == "controls":
            for k in CONTROLS:
                add(f"e1.{k}.mase_validation", rec["scores"]["validation"][k]["mase_mean"], "mase")
                add(f"e1.{k}.mae_validation", rec["scores"]["validation"][k]["mae_mean"], "mae")
        elif rec["kind"] == "ae":
            add("e1.ae.reconstruction_val_mse_masked", rec["pretraining"]["reconstruction_val_mse_masked"], "mse")
    return R._terminal(status=status, reason=reason, cost=cost, metrics=rows, started=cost.get("started_at") or R.now_iso(), finished=cost.get("ended_at") or R.now_iso(),
                       tags={"purpose": "E1_DEV_PILOT", "proposal": "P-E1", "grants": "NONE", "classification": "NON_GOVERNING", "phase": "DEVELOPMENT",
                             "outcome": str(out["outcome"]), "design_sha256": design["design_sha256"], "unit_id": unit_id, "family": design["family"],
                             "task": design["task"]["id"], "host": cost.get("host", ""), "metric_states": json.dumps(states, sort_keys=True), **tags})


def spent_cpu(root: Path) -> float:
    DEV = _load("df_utility_dev_run")
    return DEV.spent_cpu(root)


def run(design: dict, *, root: Path, run_id: str, cap_seconds: float, already_spent: float, parallel: int = 1,
        pilot_only: bool = False, trace=print, gov_url: str | None = None, api_key_file: Path | None = None,
        lake: str = "public_panels", resource: str | None = None, outbox_dir: str | None = None) -> dict:
    """RP42: when the design's `data_access` is GOVERNED_DELIVERY this is the ONLY path: every unit's
    campaign is registered and its delivery verified BEFORE it reads or computes anything, and every
    unit ends with a terminal that reaches the accounting. Nothing writes a local terminal document."""
    campaign = _load("df_d3_campaign")
    GR = _load("governed_run")
    G = _load("df_e1_governed")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "DESIGN.json").is_file():
        if json.loads((root / "DESIGN.json").read_text())["design_sha256"] != design["design_sha256"]:
            raise SystemExit("REFUSED: this root was frozen under another design")
    else:
        campaign.write_once(root / "DESIGN.json", design)
    code_identity = GR.strict_code_identity(REPO)
    governed = design.get("data_access") == "GOVERNED_DELIVERY"
    resource = resource or f"{design['family'].replace('uci_235', 'uci_235_individual_household_power').replace('uci_321', 'uci_321_electricityloaddiagrams20112014')}/panel.parquet"
    if governed:
        if not api_key_file:
            raise G.GovernanceUnavailable("REFUSED: a governed run needs the service key; nothing is prepared")
        units = ["prepare"] + [c["cell_id"] for c in design["pilots"] + design["cells"]]
        for unit_id in units:                       # the campaign of EVERY unit, before any of them works
            G.acquire(run_id=run_id, root=root, lake=lake, resource=resource, unit_id=unit_id,
                      gov_url=gov_url or G.DEFAULT_GOV, api_key_file=api_key_file,
                      design_sha256=design["design_sha256"], cache_dir=Path(root) / "cache",
                      expect_sha256=design["governed_bytes"]["sha256"])
        trace(json.dumps({"event": "acquired", "units": len(units), "resource": resource}), flush=True)
    data = prepare(design, root)
    budgets = design["budget"]
    report = {"schema": "df_e1_pilot_report.v1", "run_id": run_id, "design_sha256": design["design_sha256"], "code_identity": code_identity, "data": data,
              "cap_seconds": cap_seconds, "already_spent_seconds": already_spent, "cost_pilot": {}, "projection": None, "cells": {}, "terminals": [],
              "stopped": None, "host": os.uname().nodename, "parallel": int(parallel), "governance": design["governance"]}
    spent = lambda: already_spent + spent_cpu(root)
    tdir = root / "TERMINALS"
    tdir.mkdir(exist_ok=True)

    def job_for(c, **extra):
        job = {"kind": c["kind"], "cell_id": c["cell_id"], "seed": c["seed"], "max_updates": c.get("max_updates", 0), "design": design, "data_npz": str(root / "DATA.npz"),
               "data_sha256": data["data_sha256"], "run_id": run_id, "role": c.get("role", "CELL"), **({"regime": c["regime"]} if "regime" in c else {}), **extra}
        if c.get("depends_on"):
            job["pretrained_npz"] = str(root / "attempts" / c["depends_on"] / "detector_pretrained.npz")
        return job

    def child(c, need=0.0):
        attempt = root / "attempts" / c["cell_id"]
        resumed = (attempt / "outcome.json").is_file()
        if not resumed and spent() + need > cap_seconds:
            raise SystemExit(f"CPU_CAP_EXHAUSTED: spent {spent():.0f} s + next child up to {need:.0f} s exceeds {cap_seconds:.0f} s")
        delivery = None
        if governed:
            # this unit's OWN delivery, verified now: a cached DATA file never skips this
            delivery = G.require_delivery(root, design, c["cell_id"])["delivery"]
            if delivery.get("sha256") != data["panel_sha256"]:
                raise G.GovernanceUnavailable(
                    f"REFUSED: unit {c['cell_id']} was delivered {delivery.get('sha256')} but the prepared data came "
                    f"from {data['panel_sha256']}")
        trace(json.dumps({"event": "child", "unit": c["cell_id"], "resumed": resumed,
                          "delivery": (delivery or {}).get("delivery_id")}), flush=True)
        out = run_isolated(job_for(c), attempt_dir=attempt, assigned_bytes=int(budgets["task_memory_bytes"]), wall_seconds=float(budgets["wall_seconds"]),
                           cpu_seconds=int(budgets["cpu_seconds"]))
        trace(json.dumps({"event": "child-done", "unit": c["cell_id"], "outcome": out["outcome"], "cpu": out["cost"].get("cpu_seconds")}), flush=True)
        term = _terminal(c["cell_id"], out, design, {"role": c.get("role", "CELL"), "kind": c["kind"], "seed": str(c["seed"]), **({"regime": c["regime"]} if "regime" in c else {})})
        record = {"unit_id": c["cell_id"], "status": term["status"], "outcome": out["outcome"], "cost": out["cost"],
                  "resumed": out.get("resumed", False)}
        if governed:
            reported = G.report_terminal(root, c["cell_id"], term, gov_url=gov_url or G.DEFAULT_GOV,
                                         api_key_file=api_key_file, outbox_dir=outbox_dir)
            record.update(governed=True, delivery_id=(delivery or {}).get("delivery_id"),
                          campaign_sha256=reported.get("campaign_sha256"),
                          terminal_sent=reported["flushed"]["sent"], terminal_pending=reported["flushed"]["pending"],
                          reconciliation=reported["reconciliation"])
            if reported["flushed"]["pending"] or reported["flushed"]["failures"]:
                raise SystemExit(f"REFUSED: the terminal of {c['cell_id']} was not accepted: {reported['flushed']['failures']}")
        else:
            tpath = tdir / f"{c['cell_id']}.json"
            if not tpath.exists():
                tpath.write_text(json.dumps({"unit_id": c["cell_id"], "campaign": "NOT SUBMITTED: this run is not governed",
                                             "terminal": term}, indent=1, default=str))
            record.update(governed=False)
        report["terminals"].append(record)
        return out

    def summary(out):
        rec = out.get("score")
        s = {"outcome": out["outcome"], "cost": out.get("cost"), "resumed": out.get("resumed", False)}
        if rec and rec["kind"] == "fit":
            s.update(regime=rec["regime"], seed=rec["seed"], mase_validation=rec["scores"]["validation"]["model"]["mase_mean"], mae_validation=rec["scores"]["validation"]["model"]["mae_mean"],
                     updates=rec["training"]["updates"], stop_reason=rec["training"]["stop_reason"], censoring=rec["training"]["censoring"], phases=rec["cost"],
                     detector_unchanged=rec["detector_unchanged"], detector_receives_gradient=rec["gradient_proof"]["detector_receives_gradient"],
                     initial_full_digest=rec["initial_checkpoint"]["full_digest"], parameters=rec["parameters"])
        elif rec and rec["kind"] == "ae":
            s.update(seed=rec["seed"], updates=rec["pretraining"]["updates"], stop_reason=rec["pretraining"]["stop_reason"], reconstruction=rec["pretraining"]["reconstruction_val_mse_masked"],
                     phases=rec["cost"], detector_sha256=rec["detector_sha256"])
        elif rec and rec["kind"] == "controls":
            s.update(controls={k: {"mase": rec["scores"]["validation"][k]["mase_mean"], "mae": rec["scores"]["validation"][k]["mae_mean"]} for k in CONTROLS}, phases=rec["cost"])
        return s
    # --- cost pilot ---
    measured = {}
    for c in design["pilots"]:
        out = child(c)
        rec = out.get("score")
        report["cost_pilot"][c["cell_id"]] = summary(out)
        if rec is None:
            report.update(stopped=f"COST_PILOT_FAILED: {c['cell_id']} {out['outcome']}", spent_cpu_seconds=spent())
            campaign.write_once(root / "REPORT.json", report)
            return report
        cpu = float(out["cost"].get("cpu_seconds") or rec["cost"]["cpu_seconds_process"])
        fit_s = float(rec["cost"].get("fit_seconds") or rec["cost"].get("pretrain_seconds") or 0.0)
        updates = max(1, int(rec["training"]["updates"]))
        measured[c["kind"]] = {"cpu_seconds": cpu, "fit_seconds": fit_s, "overhead_seconds": max(0.0, cpu - fit_s), "updates": updates, "seconds_per_update": fit_s / updates,
                               "peak_rss_bytes": out["cost"].get("peak_rss_bytes"), "cgroup_memory_peak": out["cost"].get("cgroup_memory_peak")}
        report["cost_pilot"][c["cell_id"]].update(measured[c["kind"]])
    headroom = float(budgets["headroom"])
    per_cell = {}
    for c in design["cells"]:
        m = measured["fit"] if c["kind"] in ("fit", "controls") else measured["ae"]
        per_cell[c["cell_id"]] = (m["seconds_per_update"] * int(c.get("max_updates", 0)) + m["overhead_seconds"]) if c["kind"] != "controls" else m["overhead_seconds"] * 2
    remaining = [c for c in design["cells"] if not (root / "attempts" / c["cell_id"] / "outcome.json").is_file()]
    total = sum(per_cell[c["cell_id"]] for c in remaining)
    report["projection"] = {"per_cell": per_cell, "remaining_cells": len(remaining), "projected_remaining_cpu_seconds": total, "headroom": headroom,
                            "projected_with_headroom": total * (1 + headroom), "spent_so_far": spent(), "cap_seconds": cap_seconds, "fits": spent() + total * (1 + headroom) <= cap_seconds,
                            "assumption": "every unit at its full update ceiling (early stopping only lowers it); per-update cost from the pilot of its kind; overhead per child from the pilots"}
    trace(json.dumps({"event": "projection", **{k: v for k, v in report["projection"].items() if k != "per_cell"}}), flush=True)
    if not report["projection"]["fits"]:
        campaign.write_once(root / "PLAN.json", {"schema": "df_e1_pilot_plan.v1", "verdict": "BUDGET_LIMITED_NOT_LAUNCHED", "projection": report["projection"], "measured": measured})
        report.update(stopped="BUDGET_LIMITED_NOT_LAUNCHED", spent_cpu_seconds=spent())
        campaign.write_once(root / "REPORT.json", report)
        return report
    if pilot_only:
        report.update(stopped="PILOT_ONLY", spent_cpu_seconds=spent())
        campaign.write_once(root / "REPORT.pilot.json", report)
        return report
    # --- the units in waves (dependencies respected) ---
    import concurrent.futures
    import threading
    lock = threading.RLock()
    done = {}
    pending = list(design["cells"])
    stop = None
    while pending and stop is None:
        ready = [c for c in pending if not c.get("depends_on") or c["depends_on"] in done]
        later = [c for c in pending if c not in ready]
        runnable = []
        for c in ready:
            dep = c.get("depends_on")
            if dep and done[dep].get("score") is None:
                report["cells"][c["cell_id"]] = {"outcome": "INCONCLUSIVE_DEPENDENCY", "depends_on": dep, "why": "its AE unit did not complete and verify; not re-run silently"}
                done[c["cell_id"]] = {"score": None, "outcome": "INCONCLUSIVE_DEPENDENCY"}
            else:
                runnable.append(c)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(parallel))) as pool:
            futures = {pool.submit(child, c, per_cell[c["cell_id"]] * (1 + headroom)): c for c in runnable}
            for fut in concurrent.futures.as_completed(futures):
                c = futures[fut]
                try:
                    out = fut.result()
                except SystemExit as e:
                    stop = str(e)
                    continue
                with lock:
                    done[c["cell_id"]] = out
                    report["cells"][c["cell_id"]] = summary(out)
        pending = later
    report["stopped"] = stop
    report["spent_cpu_seconds"] = spent()
    if governed:
        report["governance_result"] = _governed_summary(root, report)
        campaign.write_once(root / ("REPORT.json" if not (root / "REPORT.json").exists() else f"REPORT.{int(time.time())}.json"), report)
        return report
    proposal = {"schema": "governed_campaign.v1", "campaign_key": f"{run_id}-e1-household-dev-pilot", "classification": "NON_GOVERNING", "project": "predictor",
                "code_identity": code_identity, "config_sha256": design["design_sha256"], "input_mode": "DATASETS", "synthetic_spec_sha256": None,
                "units": [c["cell_id"] for c in design["pilots"] + design["cells"]],
                "datasets": [{"lake": "public_panels_c126_v2 (NOT YET A LAKE)", "resource": "uci_235_individual_household_power/panel.parquet", "role": "panel", "from": None, "to": None}],
                "terminal_lake": "olap_cube", "status": "PROPOSAL_NOT_SUBMITTED", "why": design["governance"]["why"], "missing_object": design["governance"]["missing_object"],
                "panel_sha256": data["panel_sha256"]}
    if not (root / "CAMPAIGN_PROPOSAL.json").exists():
        campaign.write_once(root / "CAMPAIGN_PROPOSAL.json", proposal)
    campaign.write_once(root / ("REPORT.json" if not (root / "REPORT.json").exists() else f"REPORT.{int(time.time())}.json"), report)
    return report


# --- close --------------------------------------------------------------------------------------------------

def close(root: Path) -> dict:
    root = Path(root)
    design = json.loads((root / "DESIGN.json").read_text())
    reports = [root / "REPORT.json"] if (root / "REPORT.json").is_file() else sorted(root.glob("REPORT*.json"))
    if not reports:
        raise SystemExit("REFUSED: no REPORT in the root")
    report = json.loads(reports[-1].read_text())          # the full run's report governs; REPORT.pilot.json is only the cost pilot
    cells = {}
    for c in design["cells"] + design["pilots"]:
        p = root / "attempts" / c["cell_id"] / "cell.json"
        oc = root / "attempts" / c["cell_id"] / "outcome.json"
        if p.is_file() and oc.is_file():
            o = json.loads(oc.read_text())
            rec = json.loads(p.read_text())
            res = json.loads((root / "attempts" / c["cell_id"] / "result.json").read_text())
            score, refusal = verified_unit(root / "attempts" / c["cell_id"], res, o.get("verified"))
            cells[c["cell_id"]] = {"rec": rec, "outcome": o, "verified": refusal is None, "refusal": refusal}
    seeds = design["replicas"]["seeds"]
    table, paired = {}, {}
    ae_by_seed = {}
    for s in seeds:
        for r in REGIMES:
            k = f"{r}_s{s}"
            if k in cells and cells[k]["verified"]:
                rec = cells[k]["rec"]
                table[k] = {"regime": r, "seed": s, "mase": rec["scores"]["validation"]["model"]["mase_mean"], "mae": rec["scores"]["validation"]["model"]["mae_mean"],
                            "updates": rec["training"]["updates"], "epochs": rec["training"]["epochs"], "stop": rec["training"]["stop_reason"], "censoring": rec["training"]["censoring"],
                            "best_epoch": rec["training"]["restored_checkpoint_epoch"], "curve_val": rec["training"]["curve"]["validation"], "curve_train": rec["training"]["curve"]["train"],
                            "phases": rec["cost"], "child_cpu": cells[k]["outcome"]["summary"]["cost"].get("cpu_seconds"), "peak_rss": cells[k]["outcome"]["summary"]["cost"].get("peak_rss_bytes"),
                            "cgroup_peak": cells[k]["outcome"]["summary"]["cost"].get("cgroup_memory_peak"), "detector_unchanged": rec["detector_unchanged"],
                            "detector_gradient": rec["gradient_proof"]["detector_receives_gradient"], "initial_full_digest": rec["initial_checkpoint"]["full_digest"],
                            "detector_after_setup": rec["regime_setup"]["detector_digest_after_setup"], "params_trainable": rec["parameters"]["trainable"],
                            "pretrained_sha256": rec.get("pretrained_sha256")}
            else:
                table[k] = {"regime": r, "seed": s, "outcome": (cells.get(k) or {}).get("outcome", {}).get("summary", {}).get("outcome", report["cells"].get(k, {}).get("outcome", "ABSENT"))}
        a = f"ae_s{s}"
        if a in cells and cells[a]["verified"]:
            rec = cells[a]["rec"]
            ae_by_seed[s] = {"updates": rec["pretraining"]["updates"], "stop": rec["pretraining"]["stop_reason"], "reconstruction": rec["pretraining"]["reconstruction_val_mse_masked"],
                             "phases": rec["cost"], "child_cpu": cells[a]["outcome"]["summary"]["cost"].get("cpu_seconds"), "detector_sha256": rec["detector_sha256"],
                             "curve_val": rec["pretraining"]["curve"]["validation"]}
        if all(f"{r}_s{s}" in table and "mase" in table[f"{r}_s{s}"] for r in REGIMES):
            b = table[f"R0_s{s}"]["mase"]
            paired[s] = {"R1_minus_R0": table[f"R1_s{s}"]["mase"] - b, "R2_minus_R0": table[f"R2_s{s}"]["mase"] - b, "R2_minus_R1": table[f"R2_s{s}"]["mase"] - table[f"R1_s{s}"]["mase"]}
    identity = {}
    for s in seeds:
        digs = {r: table.get(f"{r}_s{s}", {}).get("initial_full_digest") for r in REGIMES}
        dets = {r: table.get(f"{r}_s{s}", {}).get("detector_after_setup") for r in REGIMES}
        identity[s] = {"shared_initial_checkpoint": len({d for d in digs.values() if d}) == 1 and all(digs.values()),
                       "R1_R2_same_imported_detector": dets.get("R1") is not None and dets.get("R1") == dets.get("R2"),
                       "R0_detector_is_the_random_initial": dets.get("R0") is not None and dets.get("R0") != dets.get("R1"),
                       "R1_detector_frozen": table.get(f"R1_s{s}", {}).get("detector_unchanged") is True and table.get(f"R1_s{s}", {}).get("detector_gradient") is False,
                       "R0_R2_detector_learns": all(table.get(f"{r}_s{s}", {}).get("detector_unchanged") is False and table.get(f"{r}_s{s}", {}).get("detector_gradient") is True for r in ("R0", "R2"))}
    controls = None
    if "controls" in cells and cells["controls"]["verified"]:
        rec = cells["controls"]["rec"]
        controls = {k: {"mase": rec["scores"]["validation"][k]["mase_mean"], "mae": rec["scores"]["validation"][k]["mae_mean"]} for k in CONTROLS}
        controls["phases"] = rec["cost"]
    means = {}
    for r in REGIMES:
        vals = [table[f"{r}_s{s}"]["mase"] for s in seeds if "mase" in table.get(f"{r}_s{s}", {})]
        means[r] = {"n": len(vals), "mase_mean": float(np.mean(vals)) if vals else None, "mase_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
    pdiff = {}
    for key in ("R1_minus_R0", "R2_minus_R0", "R2_minus_R1"):
        vals = [paired[s][key] for s in paired]
        pdiff[key] = {"n": len(vals), "mean": float(np.mean(vals)) if vals else None, "per_seed": {s: paired[s][key] for s in paired},
                      "min": float(min(vals)) if vals else None, "max": float(max(vals)) if vals else None}
    cost_total = {"ae_cpu": sum(v.get("child_cpu") or 0 for v in ae_by_seed.values()), "fit_cpu_by_regime": {r: sum((table[f"{r}_s{s}"].get("child_cpu") or 0) for s in seeds if "mase" in table.get(f"{r}_s{s}", {})) for r in REGIMES},
                  "pilot_cpu": sum((cells[c]["outcome"]["summary"]["cost"].get("cpu_seconds") or 0) for c in ("pilot_ae", "pilot_fit") if c in cells),
                  "controls_cpu": (cells["controls"]["outcome"]["summary"]["cost"].get("cpu_seconds") if "controls" in cells else None),
                  "spent_cpu_seconds_root": spent_cpu(root), "reading": "same task budget per regime (fit CPU); total cost = fit + the AE of the seed for R1/R2 (reported apart)"}
    doc = {"schema": "df_e1_pilot_results.v1", "design_sha256": design["design_sha256"], "run_id": report["run_id"], "stopped": report.get("stopped"), "task": design["task"],
           "dev": design["dev_subpartition"], "data": report["data"]["coverage"], "denominator": report["data"]["mase_denominator_persistence_h"], "cells": table, "ae": ae_by_seed,
           "controls": controls, "means": means, "paired": pdiff, "identity": identity, "costs": cost_total, "projection": report.get("projection"), "cost_pilot": report.get("cost_pilot"),
           "verified_units": {k: v["verified"] for k, v in cells.items()}, "governance": design["governance"], "host": report.get("host"),
           "reading_rules": ["development pilot: no H1 confirmation, no model selection, no benchmark substitute", "BUDGET_LIMITED / truncated curves are not evidence against pre-training",
                             "no row or task was chosen after seeing regime results (design sealed before outcomes)"]}
    (root / "RESULTS.json").write_text(json.dumps(doc, indent=1, default=float))
    lines = [f"# E1 household DEV pilot — results (run {report['run_id']})", "", f"Task {design['task']['id']}: context {design['task']['context_physical_seconds']} s, horizon {design['task']['horizon_physical_seconds']} s, model reach {design['task']['model_reach_physical_seconds']} s. "
             f"DEV rows {design['dev_subpartition']['rows']}. Common evaluation set {report['data']['coverage']['common_evaluation_set']} origins. MASE denominator (persistence h, train) {report['data']['mase_denominator_persistence_h']:.4f} kW.", "",
             "| unit | regime | seed | MASE | MAE (kW) | updates | epochs | stop | best epoch | censoring | fit s | child CPU s | peak RSS MB |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in table.items():
        if "mase" in v:
            trunc = v["censoring"]["verdict"]
            lines.append(f"| {k} | {v['regime']} | {v['seed']} | {v['mase']:.4f} | {v['mae']:.4f} | {v['updates']} | {v['epochs']} | {v['stop']} | {v['best_epoch']} | {trunc} | {v['phases'].get('fit_seconds')} | {v['child_cpu']} | {(v['peak_rss'] or 0) / 2**20:.0f} |")
        else:
            lines.append(f"| {k} | {v['regime']} | {v['seed']} | {v.get('outcome')} | | | | | | | | | |")
    if controls:
        lines += ["", "| control | MASE | MAE (kW) |", "|---|---|---|"] + [f"| {k} | {controls[k]['mase']:.4f} | {controls[k]['mae']:.4f} |" for k in CONTROLS]
    lines += ["", "| regime | n | MASE mean | MASE sd |", "|---|---|---|---|"] + [f"| {r} | {m['n']} | {m['mase_mean'] if m['mase_mean'] is None else round(m['mase_mean'], 4)} | {m['mase_sd'] if m['mase_sd'] is None else round(m['mase_sd'], 4)} |" for r, m in means.items()]
    lines += ["", "| paired difference (MASE) | n | mean | min | max | per seed |", "|---|---|---|---|---|---|"] + [f"| {k} | {v['n']} | {v['mean'] if v['mean'] is None else round(v['mean'], 4)} | {v['min'] if v['min'] is None else round(v['min'], 4)} | {v['max'] if v['max'] is None else round(v['max'], 4)} | {json.dumps({str(s): round(x, 4) for s, x in v['per_seed'].items()})} |" for k, v in pdiff.items()]
    lines += ["", "| seed | AE updates | AE stop | masked val MSE | AE child CPU s |", "|---|---|---|---|---|"] + [f"| {s} | {v['updates']} | {v['stop']} | {v['reconstruction']:.4f} | {v['child_cpu']} |" for s, v in ae_by_seed.items()]
    lines += ["", "Identity proofs per seed: " + json.dumps(identity), "", "Costs: " + json.dumps(cost_total), "", "Governance: " + design["governance"]["disposition"] + " — " + design["governance"]["missing_object"]]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", type=Path, default=None)
    ap.add_argument("--seal", type=Path, default=None, help="write the sealed design here (refuses to overwrite)")
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--pilot-only", action="store_true")
    ap.add_argument("--close", action="store_true")
    ap.add_argument("--cpu-cap-seconds", type=float, default=14400.0)
    ap.add_argument("--already-spent", type=float, default=0.0)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--gov-url", default=None)
    ap.add_argument("--api-key-file", type=Path, default=None)
    ap.add_argument("--lake", default="public_panels")
    ap.add_argument("--resource", default=None)
    ap.add_argument("--outbox-dir", default=None)
    a = ap.parse_args(argv)
    if a.worker is not None:
        return worker_main(a.worker)
    if a.seal is not None:
        d = seal()
        _load("df_d3_campaign").write_once(a.seal, d)
        print(json.dumps({"design_sha256": d["design_sha256"], "cells": len(d["cells"]), "pilots": len(d["pilots"])}))
        return 0
    if a.close:
        doc = close(a.root)
        print(json.dumps({"means": doc["means"], "paired": {k: v["mean"] for k, v in doc["paired"].items()}, "controls": doc["controls"], "stopped": doc["stopped"]}, indent=1))
        return 0
    design = json.loads(a.design.read_text())
    if design.get("schema") != SCHEMA_DESIGN or E.sha_obj({k: v for k, v in design.items() if k != "design_sha256"}) != design["design_sha256"]:
        raise SystemExit("REFUSED: the design is not a sealed E1 pilot design")
    if a.prepare_only:
        rec = prepare(design, a.root)
        print(json.dumps({k: v for k, v in rec.items() if k != "scaler"}, indent=1))
        return 0
    report = run(design, root=a.root, run_id=a.run_id, cap_seconds=a.cpu_cap_seconds, already_spent=a.already_spent,
                 parallel=a.parallel, pilot_only=a.pilot_only, gov_url=a.gov_url, api_key_file=a.api_key_file,
                 lake=a.lake, resource=a.resource, outbox_dir=a.outbox_dir)
    print(json.dumps({"stopped": report["stopped"], "spent": report.get("spent_cpu_seconds"), "projection": {k: v for k, v in (report.get("projection") or {}).items() if k != "per_cell"},
                      "cells": {k: {kk: v.get(kk) for kk in ("outcome", "mase_validation", "updates", "stop_reason")} for k, v in report["cells"].items()}}, indent=1, default=str))
    return 0 if report["stopped"] is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
