#!/usr/bin/env python3
"""Lane F2: ETH 4h financial forecast campaign on the modular branch+core engine.

Reuses M04's persistent queue (``tools.modular_doin_campaign.Campaign``: queue.sqlite,
attempts, incumbent) and search space, with a LOCAL executor: every attempt is admitted
through ``crispdm-run`` on the worker that runs this loop (one GPU attempt at a time),
training through ``tools/eth_cell_runner.py`` and verifying through the independent
``tools/modular_checkpoint_scorer.py`` in a fresh process (exact match required).

Cells (named, R0): architecture {grouped32, per_feature, control_mlp} x loss {mae, huber}
x optimizer {adam (weight_decay 0), adamw (weight_decay 1e-4)} x paired seeds. The
control is the non-branching base architecture (flatten + MLP) at the same target, rows,
budget and selection rule; it is enqueued as an ordinary candidate row whose nested
config carries ``control`` instead of a modular ``model.branches`` layout.

Subcommands: ``declare`` (CAMPAIGN.json from the NPZ manifest and the pinned checkout;
the freeze lives in it), ``materialize`` (nested candidate for a named cell, e.g. for cost
pilots), ``amend`` (append an amendment: caps from measurement, control sizing),
``enqueue`` (named cells x seeds), ``run`` (the loop; STATUS.json every claim and at a
fixed cadence; STOP file for a graceful stop), ``status``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import modular_doin_campaign as mdc  # noqa: E402
from tools import modular_search_space as ss  # noqa: E402

SCHEMA = "f2.eth_forecast_campaign.v1"
HORIZONS = [1, 2, 3, 4, 5, 6]
TARGET_FEATURE = "log_return_1"
SEASONAL_PERIOD = 6

# Lane D corrected default (da4ce7b4 design), minus the axes this campaign names explicitly.
DEFAULT_FLAT = {
    "branch.channels": 16, "branch.dilation_rate": 1, "branch.kernel_size": 3, "branch.plugin": "causal_conv1d",
    "branch.regime": "R0", "core.blocks": 2, "core.d_model": 64, "core.dropout": 0.0, "core.ff_dim": 128,
    "core.heads": 4, "core.kernel_size": 3, "core.regime": "R0", "core.stage_channels_0": 32,
    "core.stage_channels_1": 16, "core.stage_count": 3, "core.time_factor_0": 2, "core.time_factor_1": 2,
    "core.time_factor_2": 1, "model.branch_steps": 24, "model.output_channels": 8, "model.output_steps": 6,
    "train.batch_size": 64, "train.learning_rate": 0.001, "train.max_epochs": 30, "train.min_delta": 0.0,
    "train.patience": 5,
}
ARCHITECTURES = {"grouped32": {"branch.grouping_size": 32}, "per_feature": {"branch.grouping_size": 1}}
LOSSES = {"mae": {"train.loss": "mae"}, "huber": {"train.loss": "huber", "train.huber_delta": 1.0}}
OPTIMIZERS = {"adam": {"train.weight_decay": 0.0}, "adamw": {"train.weight_decay": 0.0001}}
CONTROL = "control_mlp"


def search_space(feature_count, seeds):
    return {"schema": ss.SCHEMA, "engine": "modular_temporal.v2_full_grid", "bounds": {
        "branch.channels": {"choices": [8, 16, 32]}, "branch.dilation_rate": {"choices": [1]},
        "branch.grouping_size": {"choices": [1, 8, 32, feature_count]}, "branch.kernel_size": {"choices": [3, 5, 7]},
        "branch.plugin": {"choices": ["causal_conv1d"]}, "branch.regime": {"choices": ["R0", "R1", "R2"]},
        "core.blocks": {"choices": [1, 2, 3]}, "core.d_model": {"choices": [32, 64, 128]},
        "core.dropout": {"type": "float", "low": 0.0, "high": 0.3}, "core.ff_dim": {"choices": [64, 128, 256]},
        "core.heads": {"choices": [2, 4, 8]}, "core.kernel_size": {"choices": [3]},
        "core.regime": {"choices": ["R0", "R1", "R2"]},
        **{f"core.stage_channels_{i}": {"choices": [96, 64, 48, 32, 24, 16, 12]} for i in range(4)},
        "core.stage_count": {"choices": [3, 4]},
        **{f"core.time_factor_{i}": {"choices": [1, 2, 4, 8]} for i in range(4)},
        "model.branch_steps": {"choices": [24]}, "model.output_channels": {"choices": [4, 8, 16]},
        "model.output_steps": {"choices": [3, 6]}, "train.batch_size": {"choices": [32, 64, 128]},
        "train.huber_delta": {"type": "float", "low": 0.05, "high": 2.0, "log": True},
        "train.learning_rate": {"type": "float", "low": 0.0001, "high": 0.003, "log": True},
        "train.loss": {"choices": ["huber", "mae", "mse"]}, "train.max_epochs": {"choices": [30]},
        "train.min_delta": {"choices": [0.0, 0.0001]}, "train.patience": {"choices": [3, 5, 8]},
        "train.seed": {"choices": list(seeds)},
        # Adam is AdamW with zero decoupled decay; the two named optimizers are a discrete axis.
        "train.weight_decay": {"choices": [0.0, 0.0001]}}}


def parse_size(text):
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    text = str(text).strip()
    return int(float(text[:-1]) * units[text[-1].upper()]) if text[-1].upper() in units else int(text)


RESIDUAL_SUFFIX = "_sres"


def residual_spec():
    """M01's seasonal_naive_cumulative (a-engine 3b073d2e): exact for the cumulative standardized target."""
    return {"kind": "seasonal_naive_cumulative", "period": SEASONAL_PERIOD, "target_features": [TARGET_FEATURE]}


def with_residual(nested):
    out = json.loads(ss.canonical(nested))
    out["model"]["target_residual"] = residual_spec()
    out["modular_candidate"]["f2_variant"] = "seasonal_residual_cumulative_p6"
    return json.loads(ss.canonical(out))


REGIME_SUFFIXES = {"_r1": "R1", "_r2": "R2"}


def regime_of(cell):
    for suffix, regime in REGIME_SUFFIXES.items():
        if cell.endswith(suffix):
            return regime
    return "R0"


def donor_declaration(donor_dir, feature_names, grouping=1):
    """base.donors map and donor_binding from an M02-format donor directory (schema-2 manifests embed provenance)."""
    import hashlib
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    d = Path(donor_dir)
    donors, bound = {}, {}
    names = [f"branch_{i}" for i in range(len(feature_names))] if grouping == 1 else None
    if names is None:
        raise ValueError("donor_declaration supports per-feature donors (grouping 1) only")
    for name in names:
        path = d / f"{name}.keras"
        donors[f"{grouping}:{name}"] = str(path)
        bound[str(path)] = {"keras": sha(path), "manifest": sha(d / f"{name}.manifest.json"),
                            "provenance": sha(d / f"{name}.provenance.json")}
    core = d / "core.keras"
    donors[f"core:{grouping}"] = str(core)
    bound[str(core)] = {"keras": sha(core), "manifest": sha(d / "core.manifest.json"),
                        "provenance": sha(d / "core.provenance.json")}
    binding = {"index_sha256": sha(d.parent / "PRETRAIN_RECEIPT.json") if (d.parent / "PRETRAIN_RECEIPT.json").exists()
               else sha(d / "PRETRAIN.json"),
               "amendment_sha256": "NOT_APPLICABLE_SCHEMA2_NATIVE_PROVENANCE", "required_contract": "OPERATIONAL",
               "donors": bound}
    return donors, binding


def parse_cell(cell):
    """'<architecture>_<loss>_<optimizer>' -> (architecture, loss, optimizer), refusing unknown names."""
    for suffix in (RESIDUAL_SUFFIX, *REGIME_SUFFIXES):
        if cell.endswith(suffix):
            cell = cell[:-len(suffix)]
    for arch in sorted([*ARCHITECTURES, "grouped_all", CONTROL], key=len, reverse=True):
        if cell.startswith(arch + "_"):
            rest = cell[len(arch) + 1:].split("_")
            if len(rest) == 2 and rest[0] in LOSSES and rest[1] in OPTIMIZERS:
                return arch, rest[0], rest[1]
    raise ValueError(f"unknown cell name {cell!r}; expected <architecture>_<loss>_<optimizer>")


def cell_flat(architecture, loss, optimizer, feature_count=None, regime="R0"):
    if architecture == "grouped_all":  # one branch over every input channel (FX campaigns, few inputs)
        if not feature_count:
            raise ValueError("grouped_all needs the feature count")
        arch = {"branch.grouping_size": feature_count}
    elif architecture in ARCHITECTURES:
        arch = ARCHITECTURES[architecture]
    else:
        raise ValueError(f"unknown architecture {architecture}")
    flat = {**DEFAULT_FLAT, **arch, **LOSSES[loss], **OPTIMIZERS[optimizer]}
    if regime != "R0":
        flat["branch.regime"] = flat["core.regime"] = regime
    return flat


def control_hidden_for(parameter_target, window, features, horizons, targets):
    """Equal hidden widths h (two layers) whose MLP parameter count is closest to the target."""
    inputs, outputs = window * features, horizons * targets

    def count(h):
        return inputs * h + h + h * h + h + h * outputs + outputs
    best = min(range(4, 4096), key=lambda h: abs(count(h) - parameter_target))
    return [best, best], count(best)


def control_candidate(base, loss, optimizer, seed, hidden):
    evaluator = {"learning_rate": DEFAULT_FLAT["train.learning_rate"], "patience": DEFAULT_FLAT["train.patience"],
                 "min_delta": DEFAULT_FLAT["train.min_delta"], "max_epochs": DEFAULT_FLAT["train.max_epochs"],
                 "batch_size": DEFAULT_FLAT["train.batch_size"], "seed": seed, **base["evaluator_fixed"],
                 "loss": loss, "weight_decay": OPTIMIZERS[optimizer]["train.weight_decay"]}
    if loss == "huber":
        evaluator["huber_delta"] = LOSSES["huber"]["train.huber_delta"]
    nested = {"modular_candidate": {"schema": "f2.control_candidate.v1"},
              "control": {"kind": "flatten_mlp", "hidden": list(hidden), "activation": "relu"},
              "model": {"window": base["window"], "sample_hours": base["sample_hours"],
                        "feature_names": list(base["feature_names"]), "horizons": list(base["horizons"]),
                        "target_count": len(base["target_feature_indices"])},
              "evaluator": evaluator, "target_feature_indices": list(base["target_feature_indices"]),
              "objective": dict(base["objective"])}
    return json.loads(ss.canonical(nested))


# ------------------------------------------------------------------ declaration --

def declare(args):
    manifest = json.loads(Path(args.data_manifest).read_text())
    import numpy as np
    with np.load(Path(args.data_dir) / "validation.npz", allow_pickle=False) as z:
        features = z["feature_names"].astype(str).tolist()
        targets = z["target_names"].astype(str).tolist()
    horizons = list(manifest["horizons"])
    if targets != [TARGET_FEATURE] or horizons != sorted(set(horizons)):
        raise ValueError("NPZ target/horizons disagree with the campaign")
    revision = args.predictor_revision
    if len(revision) != 40:
        raise ValueError("predictor_revision must be a full commit id")
    seeds = [2021, 2022]
    donors, binding = ({}, None)
    if getattr(args, "donor_dir", None):
        donors, binding = donor_declaration(args.donor_dir, features)
    base = {"feature_names": features, "window": manifest["window"], "sample_hours": manifest["sample_hours"],
            "horizons": horizons, "target_feature_indices": [features.index(TARGET_FEATURE)],
            "objective": {"metric": "MAE", "split": "validation", "higher_is_better": False, "unit": "z_train"},
            "evaluator_fixed": {"max_updates": 1000000, "max_seconds": 5400.0}, "donors": donors}
    if binding:
        base["donor_binding"] = binding
    data_dir = Path(args.data_dir)
    declaration = {
        "schema": SCHEMA, "campaign_id": args.campaign_id, "label": "DEVELOPMENT",
        "development_note": "the 2024 validation reserve is consulted by every cell and by selection; it is a "
                            "DEVELOPMENT reserve. The 2025 test rows are untouched for a later confirmatory pass.",
        "asset": "ETHUSDT 4h spot bars (view predictor b1f8a74f)",
        "base": base, "search_space": search_space(len(features), [2021, 2022, 2023, 2024]),
        "paired_seeds": seeds, "default_candidate": cell_flat("per_feature", "mae", "adamw"),
        "default_huber_delta": 1.0,
        "cells": {"architectures": list(ARCHITECTURES) + [CONTROL], "losses": list(LOSSES),
                  "optimizers": {k: v["train.weight_decay"] for k, v in OPTIMIZERS.items()},
                  "naming": "<architecture>_<loss>_<optimizer> x seed", "regime": "R0"},
        "control": {"kind": "flatten_mlp", "activation": "relu", "hidden": None, "parameter_target": None,
                    "sizing": "two equal hidden layers sized to the grouped32 modular model's trainable "
                              "parameter count measured in the cost pilot (amended before enqueue)"},
        "data": {"train": {"path": str(data_dir / "train.npz"), "sha256": manifest["splits"]["train"]["sha256"]},
                 "validation": {"path": str(data_dir / "validation.npz"),
                                "sha256": manifest["splits"]["validation"]["sha256"]},
                 "manifest": {"path": str(Path(args.data_manifest).resolve()),
                              "sha256": mdc.sha_file(args.data_manifest)}},
        "data_location": "workers",
        "seasonal_period": getattr(args, "seasonal_period", None) or SEASONAL_PERIOD,
        "data_manifest": {k: manifest.get(k) for k in ("schema", "dataset_id", "source_sha256", "source_commit",
                                                    "feature_order_sha256", "feature_manifest", "declared_split",
                                                    "purge_bars", "purge_seconds", "clock", "scaler_identity", "target", "metric_space",
                                                    "split_sha256")},
        "data_manifest_sha256": mdc.sha_file(args.data_manifest),
        "naives": {"persistence_last_value": "evaluator baseline: last observed standardized 1-bar return at the "
                                             "origin repeated per horizon (same rows)",
                   "zero_return": "MANDATORY first bar: CLOSE[t+h] = CLOSE[t], i.e. Y_h = -h*mu/sigma (same rows)",
                   "train_mean": "Y_h = 0 (the train drift h*mu)",
                   "seasonal_6": f"{SEASONAL_PERIOD}-bar (24 h) seasonal naive on the cumulative target: "
                                 "Y_h(t) := Y_h(t-6), read from the input window (same rows)",
                   "strict_minimum": "per horizon the lowest-MAE naive among the above decides the reading"},
        "lane_b_probe": {"commits": ["3f40cea", "678665e"], "targets": "Y_s@4h, Y_l@24h, Y_l@144h elapsed-second log "
                         "returns; h=1 and h=6 coincide with Y_s@4h and Y_l@24h on windows without irregular steps "
                         "(all scored windows); Y_l@144h NOT_COVERED by this campaign",
                         "result": "every feature subset lost to the zero-return naive in the ridge probe"},
        "resources": {"train": {"cap": None, "wall": "2h", "queue_seconds": 7200, "timeout_seconds": 6600},
                      "verify": {"cap": None, "wall": "20m", "queue_seconds": 7200, "timeout_seconds": 1100}},
        "executor": {"kind": "f2.local_crispdm", "predictor_checkout": args.predictor_checkout,
                     "predictor_python": args.predictor_python, "predictor_revision": revision,
                     "crispdm_run": args.crispdm_run, "cpu_threads": 4, "cuda_visible_devices": args.gpu_uuid,
                     "extra_env": {"LD_LIBRARY_PATH": args.ld_library_path, "TF_DETERMINISTIC_OPS": "1",
                                   "TF_CPP_MIN_LOG_LEVEL": "1", "TF_NUM_INTRAOP_THREADS": "4",
                                   "CUDA_CACHE_MAXSIZE": "2147483648"}},
        "hosts": {args.host_role: {}},
        "verification": {"require_exact_match": True,
                         "rule": "VERIFIED only on bitwise-equal rescoring at the receipt batch under "
                                 "TF_DETERMINISTIC_OPS=1 in a fresh admitted process; a tolerance-only pass is "
                                 "FINDING_NOT_EXACT"},
        "require_pin": True,
        "freeze": {"performed_by": "M07 (lane F2) under the owner standing order of 2026-10-01; no external signature",
                   "source": manifest["dataset_id"], "target": manifest["target"], "horizons": horizons,
                   "rows": {k: manifest["splits"][k]["windows"] for k in ("train", "validation")},
                   "splits": manifest["declared_split"], "purge_bars": manifest.get("purge_bars"),
                   "purge_seconds": manifest.get("purge_seconds"),
                   "train_fitted_transforms": manifest["scaler_identity"],
                   "models": "grouped32 and per_feature modular R0 (lane D corrected default) and flatten_mlp control",
                   "seeds": seeds, "budget": {"max_epochs": DEFAULT_FLAT["train.max_epochs"],
                                             "patience": DEFAULT_FLAT["train.patience"],
                                             "batch_size": DEFAULT_FLAT["train.batch_size"],
                                             "learning_rate": DEFAULT_FLAT["train.learning_rate"],
                                             **base["evaluator_fixed"]},
                   "selection_rule": "lowest mean validation MAE (z_train, all horizons) across both paired seeds, "
                                     "verified cells only; naive comparisons never enter the ranking",
                   "frozen_metric": {"primary": "MAE", "secondary": "MSE"}},
        "amendments": []}
    root = Path(args.root)
    mdc.Campaign.create(root, declaration)
    print(json.dumps({"created": str(root), "cells": declaration["cells"], "features": len(features)}))


def amend(args):
    root = Path(args.root)
    path = root / "CAMPAIGN.json"
    text = path.read_text()
    declaration = json.loads(text)
    before = hashlib.sha256(text.encode()).hexdigest()
    patch = json.loads(Path(args.patch).read_text())

    def merge(target, update):
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value
    for section in ("resources", "control"):
        if section in patch:
            caps_before = {k: v.get("cap") for k, v in declaration.get("resources", {}).items()}
            merge(declaration[section], patch[section])
            if section == "resources":
                for kind, cap in caps_before.items():
                    new = declaration["resources"][kind].get("cap")
                    if cap and new and parse_size(new) < parse_size(cap):
                        raise ValueError(f"{kind} cap {cap} -> {new} would be lowered; caps are never lowered")
    declaration["amendments"].append({"n": len(declaration["amendments"]) + 1, "date": time.strftime("%Y-%m-%d"),
                                      "change": args.change, "from_sha256": before, "patch": patch})
    path.write_text(json.dumps(declaration, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"amended": str(path), "resources": declaration["resources"], "control": declaration["control"]}))


# ------------------------------------------------------------------- enqueue --

def enqueue(args):
    campaign = mdc.Campaign(args.root)
    decl = campaign.declaration
    seeds = list(decl["paired_seeds"]) if not args.seeds else [int(s) for s in args.seeds.split(",")]
    added = []
    for cell in args.cells.split(","):
        arch, loss, opt = parse_cell(cell)
        if arch == CONTROL:
            hidden = decl["control"]["hidden"]
            if not hidden:
                raise ValueError("control hidden widths are not amended yet (needs the grouped32 pilot parameter count)")
            for seed in seeds:
                nested = control_candidate(decl["base"], loss, opt, seed, hidden)
                added += _insert_control(campaign, nested, seed, cell)
            continue
        flat = cell_flat(arch, loss, opt, len(decl["base"]["feature_names"]), regime_of(cell))
        if cell.endswith(RESIDUAL_SUFFIX):
            for seed in seeds:
                nested = with_residual(ss.from_flat({**flat, "train.seed": seed}, decl["base"], decl["search_space"]))
                added += _insert_custom(campaign, nested, seed, cell, {**flat, "model.target_residual": "seasonal_naive_cumulative_p6"})
            continue
        if seeds != list(decl["paired_seeds"]):
            added += _enqueue_with_seeds(campaign, flat, cell, seeds)
        else:
            added += campaign.enqueue(flat, cell)
    print(json.dumps({"enqueued": len(added), "cids": [c[:16] for c in added]}))


def _enqueue_with_seeds(campaign, flat_without_seed, label, seeds):
    """Paired-seed enqueue for an explicit seed list (extra seeds for the best configurations)."""
    saved = campaign.declaration["paired_seeds"]
    campaign.declaration["paired_seeds"] = list(seeds)
    try:
        return campaign.enqueue(flat_without_seed, label)
    finally:
        campaign.declaration["paired_seeds"] = saved


def _insert_control(campaign, nested, seed, label):
    flat = {"control": nested["control"], "train.loss": nested["evaluator"]["loss"],
            "train.weight_decay": nested["evaluator"]["weight_decay"]}
    return _insert_custom(campaign, nested, seed, label, flat)


def _insert_custom(campaign, nested, seed, label, flat):
    """Enqueue a nested candidate the flat space cannot express (control, residual variant)."""
    cid, config_id = ss.digest(nested), ss.config_identity(nested)
    flat = {**flat, "train.seed": seed}
    db = campaign.db
    db.execute("BEGIN IMMEDIATE")
    try:
        if db.execute("SELECT 1 FROM candidates WHERE cid=?", (cid,)).fetchone():
            db.execute("COMMIT")
            return []
        position = db.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM candidates").fetchone()[0]
        db.execute("INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   (cid, position, config_id, seed, label, ss.canonical(flat), ss.canonical(nested), "queued", None,
                    mdc.now(), None, mdc.now()))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return [cid]


# ------------------------------------------------------------------ executor --

class LocalCrispdmExecutor:
    """Runs each attempt on THIS host through crispdm-run (measured cap, wall, name)."""

    def __init__(self, declaration, host):
        self.host = host
        self.e = {**declaration["executor"], **declaration.get("hosts", {}).get(host, {}).get("executor", {})}
        self.resources = {**declaration["resources"], **declaration.get("hosts", {}).get(host, {}).get("resources", {})}
        for kind in ("train", "verify"):
            if not self.resources[kind].get("cap"):
                raise RuntimeError(f"{kind} cap is not amended from a measured pilot; nothing is dispatched")

    def _launch(self, argv, output_root, kind):
        res = self.resources[kind]
        env_prefix = ["env", f"CUDA_VISIBLE_DEVICES={self.e.get('cuda_visible_devices', '')}",
                      f"M04_HOST_ROLE={self.host}", f"F2_HOST_ROLE={self.host}", "PYTHONUNBUFFERED=1",
                      f"PYTHONPATH={self.e['predictor_checkout']}",
                      *[f"{k}={v}" for k, v in sorted(self.e.get("extra_env", {}).items())]]
        command = [self.e["crispdm_run"], "-m", res["cap"], "-t", res["wall"], "-q", "-W",
                   str(res.get("queue_seconds", 3600)), "-n", f"f2-{kind}-{output_root.parent.name[:8]}",
                   "--", *env_prefix, *argv]
        (output_root / "command.json").write_text(json.dumps(command, indent=1) + "\n")
        started = time.monotonic()
        with open(output_root / "launcher.log", "w") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=self.e["predictor_checkout"]).returncode
        return code, time.monotonic() - started

    def train(self, nested, output_root, declaration, row=None):
        (output_root / "candidate.json").write_text(json.dumps(nested, indent=1, sort_keys=True) + "\n")
        argv = [self.e["predictor_python"], "-u", "tools/eth_cell_runner.py", "--candidate",
                str(output_root / "candidate.json"), "--train", declaration["data"]["train"]["path"],
                "--validation", declaration["data"]["validation"]["path"], "--out", str(output_root / "cell"),
                "--revision", self.e["predictor_revision"], "--campaign-id", declaration["campaign_id"],
                "--manifest", declaration["data"]["manifest"]["path"], "--seasonal-period", str(declaration.get("seasonal_period") or SEASONAL_PERIOD),
                "--heartbeat-interval", "30"]
        if self.e.get("cuda_visible_devices"):
            argv += ["--gpu-uuid", self.e["cuda_visible_devices"]]
        if row is not None:
            argv += ["--cid", row["cid"], "--label", row["label"], "--config-id", row["config_id"]]
        code, elapsed = self._launch(argv, output_root, "train")
        return mdc.summarize_train(output_root, code, elapsed)

    def verify(self, receipt_path, output_root, declaration):
        argv = [self.e["predictor_python"], "-u", "tools/modular_checkpoint_scorer.py", "--receipt", receipt_path,
                "--validation", declaration["data"]["validation"]["path"], "--output",
                str(output_root / "verification.json")]
        code, elapsed = self._launch(argv, output_root, "verify")
        return mdc.summarize_verify(output_root, code, elapsed)


class F2Campaign(mdc.Campaign):
    """M04's queue with the candidate row passed to the executor (cell identity in the receipt)."""

    def execute(self, row, kind, executor, attempt, output_root):
        if kind != "train":
            return super().execute(row, kind, executor, attempt, output_root)
        cid = row["cid"]
        output_root.mkdir(parents=True, exist_ok=False)
        try:
            outcome = executor.train(json.loads(row["nested"]), output_root, self.declaration, row=row)
        except Exception as exc:
            outcome = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        (output_root / "OUTCOME.json").write_text(json.dumps(outcome, indent=1, default=str) + "\n")
        if outcome.get("status") == "completed":
            declared = {s: self.declaration["data"][s]["sha256"] for s in ("train", "validation")}
            if outcome.get("data_sha256") != declared:
                outcome = {**outcome, "status": "failed",
                           "error": f"receipt data digests {outcome.get('data_sha256')} != declared {declared}"}
        self._record(row, kind, attempt, outcome)
        return outcome


def write_status(campaign, path, extra=None):
    status = campaign.status()
    brief = {"schema": "f2.status.v1", "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "campaign": status["campaign"], "counts": status["counts"], "total": status["total"],
             "incumbent": status["incumbent"], "objective": status["objective"],
             "running": [{"cid": a["cid"][:16], "kind": a["kind"], "started": a["started"], "cap": a["cap"],
                          "output_root": a["output_root"], "host": a["host"]}
                         for a in status["attempts"] if a["status"] == "running"],
             "standings": [{k: t[k] for k in ("label", "eligible", "mean_objective", "per_seed", "statuses")}
                           for t in status["standings"]], **(extra or {})}
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(brief, indent=1, default=str) + "\n")
    os.replace(tmp, path)
    return brief


def run(args):
    campaign = F2Campaign(args.root)
    executor = LocalCrispdmExecutor(campaign.declaration, args.host_role)
    status_path = Path(args.status) if args.status else Path(args.root) / "STATUS.json"
    stop = threading.Event()

    def cadence():
        while not stop.wait(args.status_seconds):
            try:
                write_status(mdc.Campaign(args.root), status_path, {"loop_pid": os.getpid()})
            except Exception:
                pass
    thread = threading.Thread(target=cadence, daemon=True)
    thread.start()
    try:
        done = campaign.run(executor, args.max, args.stop_file)
    finally:
        stop.set()
        write_status(campaign, status_path, {"loop_pid": os.getpid(), "loop": "exited"})
    print(json.dumps({"trained": done, "host": args.host_role}))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("declare")
    p.add_argument("--root", required=True)
    p.add_argument("--campaign-id", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--data-manifest", required=True)
    p.add_argument("--predictor-checkout", required=True)
    p.add_argument("--predictor-python", required=True)
    p.add_argument("--predictor-revision", required=True)
    p.add_argument("--crispdm-run", default=str(Path.home() / ".local/bin/crispdm-run"))
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--ld-library-path", required=True)
    p.add_argument("--host-role", required=True)
    p.add_argument("--seasonal-period", type=int, default=None, help="rows; default 6 (ETH 4h); FX 1h uses 24")
    p.add_argument("--donor-dir", default=None, help="M02-format donor directory (branch_i.keras, core.keras) for R1/R2 cells")
    p = sub.add_parser("materialize")
    p.add_argument("--root", required=True)
    p.add_argument("--cell", required=True, help="<architecture>_<loss>_<optimizer>")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("amend")
    p.add_argument("--root", required=True)
    p.add_argument("--patch", required=True, help="JSON with resources and/or control fields")
    p.add_argument("--change", required=True)
    p = sub.add_parser("control-size")
    p.add_argument("--root", required=True)
    p.add_argument("--parameter-target", type=int, required=True)
    p = sub.add_parser("enqueue")
    p.add_argument("--root", required=True)
    p.add_argument("--cells", required=True, help="comma list of <architecture>_<loss>_<optimizer>")
    p.add_argument("--seeds", help="comma list; default the paired seeds")
    p = sub.add_parser("run")
    p.add_argument("--root", required=True)
    p.add_argument("--host-role", required=True)
    p.add_argument("--max", type=int)
    p.add_argument("--stop-file")
    p.add_argument("--status")
    p.add_argument("--status-seconds", type=int, default=300)
    p = sub.add_parser("hold", help="park queued cells by label (never dropped); e.g. cells placed on another host")
    p.add_argument("--root", required=True)
    p.add_argument("--labels", required=True, help="comma list of cell labels")
    p.add_argument("--reason", required=True)
    p = sub.add_parser("release-hold")
    p.add_argument("--root", required=True)
    p.add_argument("--reason", required=True)
    p = sub.add_parser("status")
    p.add_argument("--root", required=True)
    p.add_argument("--write")
    args = parser.parse_args()
    if args.command == "hold":
        campaign = mdc.Campaign(args.root)
        labels = set(args.labels.split(","))
        rows = {r["cid"]: r["label"] for r in campaign.db.execute("SELECT cid, label FROM candidates WHERE status='queued'")}
        held = campaign.hold(lambda flat: False, args.reason)  # predicate on flat is not label-aware; hold by cid below
        campaign.db.execute("BEGIN IMMEDIATE")
        for cid, label in rows.items():
            if label in labels:
                campaign.db.execute("UPDATE candidates SET status='blocked', blocked_reason=?, updated=? WHERE cid=? "
                                    "AND status='queued'", ("HOLD:" + args.reason, mdc.now(), cid))
                held.append(cid)
        campaign.db.execute("COMMIT")
        print(json.dumps({"held": len(held), "labels": sorted(labels)}))
        return
    if args.command == "release-hold":
        print(json.dumps({"released": mdc.Campaign(args.root).release_hold(args.reason)}))
        return
    if args.command == "declare":
        declare(args)
    elif args.command == "materialize":
        campaign = mdc.Campaign(args.root)
        decl = campaign.declaration
        arch, loss, opt = parse_cell(args.cell)
        if arch == CONTROL:
            nested = control_candidate(decl["base"], loss, opt, args.seed, decl["control"]["hidden"])
        else:
            nested = ss.from_flat({**cell_flat(arch, loss, opt, len(decl["base"]["feature_names"]), regime_of(args.cell)), "train.seed": args.seed}, decl["base"], decl["search_space"])
            if args.cell.endswith(RESIDUAL_SUFFIX):
                nested = with_residual(nested)
        Path(args.out).write_text(json.dumps(nested, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"cid": ss.digest(nested), "config_id": ss.config_identity(nested), "out": args.out}))
    elif args.command == "amend":
        amend(args)
    elif args.command == "control-size":
        decl = mdc.Campaign(args.root).declaration
        base = decl["base"]
        hidden, count = control_hidden_for(args.parameter_target, base["window"], len(base["feature_names"]),
                                           len(base["horizons"]), len(base["target_feature_indices"]))
        print(json.dumps({"hidden": hidden, "parameters": count, "target": args.parameter_target}))
    elif args.command == "enqueue":
        enqueue(args)
    elif args.command == "run":
        run(args)
    else:
        campaign = mdc.Campaign(args.root)
        brief = write_status(campaign, args.write) if args.write else campaign.status()
        print(json.dumps(brief, indent=1, default=str))


if __name__ == "__main__":
    main()
