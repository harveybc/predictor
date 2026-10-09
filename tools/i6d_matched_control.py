"""Resumable fleet CLI for the I6-D Dense-versus-Conv matched control."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

from predictor_plugins.modular_temporal.matched_control import (
    CONFIG_SCHEMA, array_sha256, build_matched_control, canonical_sha256,
    normalize_matched_config, train_only_plumbing,
)

TERMINAL_SCHEMA = "predictor.i6d.arm_terminal.v1"
DESIGN_SCHEMA = "predictor.i6d.campaign.v1"


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _read_json(path):
    return json.loads(Path(path).read_text())


def _terminal_path(root, arm):
    return Path(root) / "arms" / arm / "terminal.json"


def _sealed(value, field):
    body = {key: item for key, item in value.items() if key != field}
    return {**body, field: canonical_sha256(body)}


def _verify_seal(value, field):
    if value.get(field) != canonical_sha256({key: item for key, item in value.items() if key != field}):
        raise ValueError(f"{field} mismatch")
    return value


def initialize_campaign(config, output, *, train_path, validation_path):
    """Freeze paths and architecture without opening any data file."""
    root = Path(output)
    if root.exists():
        raise ValueError("campaign output already exists")
    normalized = normalize_matched_config(config)
    architecture = build_matched_control(normalized).report
    body = {
        "schema": DESIGN_SCHEMA,
        "status": "READY",
        "config": normalized,
        "config_sha256": canonical_sha256(normalized),
        "fit_sha256": canonical_sha256(normalized["fit"]),
        "train_path": str(Path(train_path).expanduser().absolute()),
        "validation_path": str(Path(validation_path).expanduser().absolute()),
        "test_path": None,
        "test_read": False,
        "arms": ["CONV", "DENSE"],
        "architecture": architecture,
    }
    design = _sealed(body, "design_sha256")
    root.mkdir(parents=True)
    _atomic_json(root / "DESIGN.json", design)
    _atomic_json(root / "STATUS.json", campaign_status(root))
    return design


def _read_design(root):
    design = _read_json(Path(root) / "DESIGN.json")
    if design.get("schema") != DESIGN_SCHEMA:
        raise ValueError("campaign design schema mismatch")
    return _verify_seal(design, "design_sha256")


def _read_terminal(root, arm):
    terminal = _read_json(_terminal_path(root, arm))
    if terminal.get("schema") != TERMINAL_SCHEMA or terminal.get("arm") != arm:
        raise ValueError(f"{arm} terminal identity mismatch")
    return _verify_seal(terminal, "terminal_sha256")


def campaign_status(output):
    """Return status from durable terminals; no model or data is loaded."""
    root = Path(output)
    design = _read_design(root) if (root / "DESIGN.json").exists() else None
    arms = ["CONV", "DENSE"] if design is None else design["arms"]
    complete, running, invalid = [], [], []
    for arm in arms:
        terminal = _terminal_path(root, arm)
        if terminal.exists():
            try:
                _read_terminal(root, arm)
                complete.append(arm)
            except (ValueError, json.JSONDecodeError):
                invalid.append(arm)
        if (root / "arms" / arm / "heartbeat.json").exists() and arm not in complete:
            running.append(arm)
    if invalid:
        state = "INVALID_EVIDENCE"
    elif len(complete) == len(arms):
        state = "COMPLETE"
    elif running:
        state = "RUNNING"
    else:
        state = "READY"
    return {"schema": "predictor.i6d.status.v1", "state": state,
            "complete_arms": sorted(complete),
            "pending_arms": sorted(set(arms) - set(complete)),
            "running_arms": sorted(running), "invalid_arms": sorted(invalid),
            "design_sha256": None if design is None else design["design_sha256"],
            "test_read": False}


def _shared_data(design):
    """Open TRAIN and VALIDATION once, establishing the rows shared by both arms."""
    config = design["config"]
    train = _load_split(design["train_path"], "train", config)
    validation = _load_split(design["validation_path"], "validation", config)
    if (train["dataset_id"] != validation["dataset_id"]
            or train["windows"].shape[1:] != validation["windows"].shape[1:]
            or np.intersect1d(train["row_ids"], validation["row_ids"]).size
            or train["target_timestamps"].max() >= validation["input_start"].min()):
        raise ValueError("TRAIN/VALIDATION identity, rows or chronological boundary disagree")
    for key in ("metric_space", "scaler_identity", "scaler_scale"):
        if not np.array_equal(train[key], validation[key]):
            raise ValueError(f"TRAIN/VALIDATION {key} disagree")
    identity = {
        "dataset_id": train["dataset_id"],
        "train_sha256": train["sha256"],
        "validation_sha256": validation["sha256"],
        "train_rows_sha256": array_sha256(train["row_ids"]),
        "validation_rows_sha256": array_sha256(validation["row_ids"]),
        "train_rows": len(train["windows"]),
        "validation_rows": len(validation["windows"]),
    }
    return train, validation, {**identity, "sha256": canonical_sha256(identity)}


def _load_split(path, split, config):
    """Load explicit matched rows; prediction targets need not be input columns."""
    raw = Path(path).read_bytes()
    required = {"windows", "targets", "baseline", "row_ids", "timestamps",
                "target_timestamps", "dataset_id", "split", "feature_names",
                "target_names", "horizons", "timestamp_unit", "metric_space",
                "scaler_identity", "scaler_scale"}
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError(f"{split} NPZ missing/unexpected fields: {set(archive.files) ^ required}")
        data = {key: archive[key] for key in required}
    x, y, baseline = data["windows"], data["targets"], data["baseline"]
    target_shape = (len(x), len(config["horizons"]), len(config["target_names"]))
    if (x.shape != (len(x), 24, len(config["feature_names"]))
            or y.shape != target_shape or baseline.shape != target_shape or min(target_shape) < 1
            or x.dtype.kind not in "fi" or y.dtype.kind not in "fi"
            or baseline.dtype.kind not in "fi"):
        raise ValueError("windows/targets/baseline violate the matched shape contract")
    for key in ("windows", "targets", "baseline"):
        data[key] = np.asarray(data[key], dtype="float32")
        if not np.all(np.isfinite(data[key])):
            raise ValueError(f"{key} must be finite")
    for key, expected in (("feature_names", config["feature_names"]),
                          ("target_names", config["target_names"])):
        if data[key].dtype.kind not in "US" or data[key].tolist() != expected:
            raise ValueError(f"{key} disagree with the design")
    if (data["horizons"].dtype.kind not in "iu"
            or data["horizons"].tolist() != config["horizons"]):
        raise ValueError("horizons disagree with the design")
    rows = data["row_ids"]
    if rows.shape != (len(x),) or rows.dtype.kind not in "US" or len(set(rows.tolist())) != len(rows):
        raise ValueError("row_ids must be unique strings")
    if data["timestamps"].shape != (len(x),) or data["timestamps"].dtype.kind != "i":
        raise ValueError("timestamps must be epoch-second integers")
    if (data["target_timestamps"].shape != (len(x), len(config["horizons"]))
            or data["target_timestamps"].dtype.kind != "i"):
        raise ValueError("target_timestamps shape/dtype disagree")
    expected_times = data["timestamps"][:, None] + np.asarray(config["horizons"]) * 3600
    if (not np.array_equal(data["target_timestamps"], expected_times)
            or np.any(data["timestamps"][1:] <= data["timestamps"][:-1])):
        raise ValueError("timestamps violate exact hourly horizon alignment")
    scalars = {}
    for key in ("dataset_id", "split", "timestamp_unit", "metric_space", "scaler_identity"):
        if data[key].shape != () or data[key].dtype.kind not in "US":
            raise ValueError(f"{key} must be a string scalar")
        scalars[key] = data[key].item()
    if scalars["split"] != split or scalars["timestamp_unit"] != "seconds":
        raise ValueError("split/timestamp_unit disagree")
    scale = data["scaler_scale"]
    if (scale.shape != (len(config["target_names"]),) or scale.dtype.kind not in "fi"
            or np.any(~np.isfinite(scale)) or np.any(scale <= 0)):
        raise ValueError("scaler_scale must be finite positive per target")
    data.update(scalars)
    data["input_start"] = data["timestamps"] - 23 * 3600
    data["sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def run_arm(output, arm):
    """Execute one arm exactly once; a process lock makes restart concurrency safe."""
    if arm not in ("CONV", "DENSE"):
        raise ValueError("arm must be CONV or DENSE")
    root = Path(output)
    design = _read_design(root)
    destination = root / "arms" / arm
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / "run.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ValueError(f"{arm} is already running") from exc
    if _terminal_path(root, arm).exists():
        return _read_terminal(root, arm)

    train, validation, data_identity = _shared_data(design)
    harness = build_matched_control(design["config"])
    selected = harness.dense if arm == "DENSE" else harness.conv
    heartbeat = destination / "heartbeat.json"

    def progress(value):
        _atomic_json(heartbeat, {"arm": arm, "state": "RUNNING", **value})

    from tools.modular_candidate_evaluator import (_metrics, _predict,
                                                    fit_with_early_stopping)
    settings = {**design["config"]["fit"], "seed": design["config"]["seed"],
                "progress": progress}
    training = fit_with_early_stopping(
        selected.model, train["windows"], train["targets"],
        validation["windows"], validation["targets"], settings)
    prediction = _predict(selected.model, validation["windows"],
                          validation["targets"], settings["batch_size"])
    metrics = _metrics(validation["targets"], prediction, validation["baseline"])
    artifact = destination / "best.keras"
    selected.model.save(artifact)
    import tensorflow as tf
    restored = tf.keras.models.load_model(artifact, compile=False)
    replay = _predict(restored, validation["windows"], validation["targets"],
                      settings["batch_size"])
    if not np.allclose(prediction, replay, rtol=1e-5, atol=1e-6):
        raise ValueError("saved-model replay differs")
    body = {
        "schema": TERMINAL_SCHEMA, "status": "COMPLETED", "arm": arm,
        "design_sha256": design["design_sha256"],
        "data_identity_sha256": data_identity["sha256"],
        "data_identity": data_identity,
        "fit_sha256": design["fit_sha256"], "seed": design["config"]["seed"],
        "architecture": design["architecture"]["arms"][arm],
        "training": training, "metrics": metrics,
        "prediction_sha256": array_sha256(prediction),
        "model_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "reload_parity": True, "test_read": False,
    }
    terminal = _sealed(body, "terminal_sha256")
    _atomic_json(_terminal_path(root, arm), terminal)
    _atomic_json(heartbeat, {"arm": arm, "state": "COMPLETED"})
    _atomic_json(root / "STATUS.json", campaign_status(root))
    return terminal


def write_terminal_for_test(output, terminal):
    """Seal a fixture terminal through the same durable writer."""
    value = _sealed(terminal, "terminal_sha256")
    _atomic_json(_terminal_path(output, terminal["arm"]), value)
    return value


def close_campaign(output):
    """Require two authentic terminals and reconcile every matched attribute."""
    root = Path(output)
    design = _read_design(root)
    terminals = {arm: _read_terminal(root, arm) for arm in design["arms"]}
    fields = ("design_sha256", "data_identity_sha256", "fit_sha256", "seed")
    for field in fields:
        values = {terminal[field] for terminal in terminals.values()}
        if len(values) != 1:
            raise ValueError(f"arm terminals disagree on {field}")
    body = {"schema": "predictor.i6d.closure.v1", "state": "COMPLETE",
            "arms": {arm: terminal["terminal_sha256"] for arm, terminal in terminals.items()},
            "matched": {field: next(iter({t[field] for t in terminals.values()}))
                        for field in fields},
            "architectures": design["architecture"]["arms"], "test_read": False}
    closure = _sealed(body, "closure_sha256")
    _atomic_json(root / "CLOSURE.json", closure)
    _atomic_json(root / "STATUS.json", campaign_status(root))
    return closure


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--config", required=True)
    init = commands.add_parser("init")
    init.add_argument("--config", required=True)
    init.add_argument("--output", required=True)
    init.add_argument("--train", required=True)
    init.add_argument("--validation", required=True)
    run = commands.add_parser("run-arm")
    run.add_argument("--output", required=True)
    run.add_argument("--arm", choices=("CONV", "DENSE"), required=True)
    status = commands.add_parser("status")
    status.add_argument("--output", required=True)
    close = commands.add_parser("close")
    close.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "plan":
        result = build_matched_control(_read_json(args.config)).report
    elif args.command == "init":
        result = initialize_campaign(_read_json(args.config), args.output,
                                     train_path=args.train, validation_path=args.validation)
    elif args.command == "run-arm":
        result = run_arm(args.output, args.arm)
    elif args.command == "status":
        result = campaign_status(args.output)
    else:
        result = close_campaign(args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
