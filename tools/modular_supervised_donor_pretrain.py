#!/usr/bin/env python3
"""Pretrain modular branches and core on a TRAIN-only seasonal-residual forecast.

The temporary forecasting head is discarded. The exported branch and core
models retain their temporal axes and can be loaded under R1, R2, or R3.
Outer validation, test, and holdout paths are intentionally absent from this
program's interface.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np


SCHEMA = "modular.supervised_donor.pretrain.v1"
RECIPE_SCHEMA = "modular.supervised_donor.recipe.v1"
OBJECTIVE_NAME = "seasonal_residual_forecast_P24"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(document):
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic(path, document):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_train_npz(path):
    """Load and validate the sole admissible input: a finite TRAIN NPZ."""
    path = Path(path)
    required = {
        "windows", "targets", "row_ids", "timestamps", "target_timestamps",
        "dataset_id", "split", "feature_names", "target_names", "horizons",
        "timestamp_unit", "metric_space", "scaler_identity", "scaler_scale",
    }
    with np.load(path, allow_pickle=False) as source:
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"TRAIN_SCHEMA_MISSING: {sorted(missing)}")
        if str(source["split"]) != "train":
            raise ValueError("TRAIN_ONLY: supervised donors accept only split='train'")
        data = {key: np.asarray(source[key]) for key in required}
    if data["windows"].ndim != 3 or data["targets"].ndim != 3:
        raise ValueError("windows and targets must be rank-three")
    if len(data["windows"]) != len(data["targets"]) or len(data["timestamps"]) != len(data["windows"]):
        raise ValueError("TRAIN_SCHEMA_LENGTH: windows, targets and timestamps differ")
    if data["target_timestamps"].shape[:1] != data["timestamps"].shape:
        raise ValueError("TRAIN_SCHEMA_LENGTH: target timestamps differ")
    if not np.isfinite(data["windows"]).all() or not np.isfinite(data["targets"]).all():
        raise ValueError("TRAIN arrays must be finite")
    if str(data["timestamp_unit"]) != "seconds":
        raise ValueError("timestamps must declare seconds")
    data["path"] = str(path.resolve())
    data["sha256"] = _sha256(path)
    return data


def internal_split_indices(timestamps, target_timestamps, *, validation_fraction, purge_origins):
    """Return chronological fit/validation rows with disjoint label support."""
    if isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)):
        raise ValueError("validation_fraction must be numeric")
    if not 0.0 < float(validation_fraction) < 0.5:
        raise ValueError("validation_fraction must be between zero and 0.5")
    if isinstance(purge_origins, bool) or not isinstance(purge_origins, int) or purge_origins < 1:
        raise ValueError("purge_origins must be a positive integer")
    timestamps = np.asarray(timestamps, dtype=np.int64)
    target_timestamps = np.asarray(target_timestamps, dtype=np.int64)
    if timestamps.ndim != 1 or target_timestamps.ndim != 2 or len(timestamps) != len(target_timestamps):
        raise ValueError("timestamp arrays have incompatible shapes")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("origin timestamps must be strictly increasing")
    validation_count = max(1, int(math.ceil(len(timestamps) * float(validation_fraction))))
    validation_start = len(timestamps) - validation_count
    train_stop = validation_start - purge_origins
    if train_stop < 1:
        raise ValueError("internal split leaves no training rows")
    train = np.arange(train_stop, dtype=np.int64)
    validation = np.arange(validation_start, len(timestamps), dtype=np.int64)
    if int(target_timestamps[train].max()) >= int(timestamps[validation].min()):
        raise ValueError("LABEL_OVERLAP: increase purge_origins")
    return {"train": train, "validation": validation,
            "purged": np.arange(train_stop, validation_start, dtype=np.int64)}


def expand_recipe(recipe, feature_names):
    """Expand a compact, reviewable recipe into the strict engine config."""
    if recipe.get("schema") != RECIPE_SCHEMA:
        raise ValueError(f"recipe schema must be {RECIPE_SCHEMA}")
    names = [str(name) for name in feature_names]
    if not names or len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    branch_template = copy.deepcopy(recipe["branch"])
    branches = []
    for index, feature in enumerate(names):
        branches.append({"name": f"branch_{index}", "features": [feature],
                         **copy.deepcopy(branch_template), "regime": "R0", "donor": None})
    period = int(recipe["target_residual_period"])
    return {
        "schema": "predictor.modular.v1",
        "window": int(recipe["window"]),
        "sample_hours": recipe["sample_hours"],
        "feature_names": names,
        "branches": branches,
        "branch_steps": int(recipe["window"]),
        "core": {**copy.deepcopy(recipe["core"]), "regime": "R0", "donor": None},
        "fusion": copy.deepcopy(recipe["fusion"]),
        "head": copy.deepcopy(recipe["head"]),
        "output_steps": int(recipe["output_steps"]),
        "output_channels": int(recipe["output_channels"]),
        "horizons": list(recipe["horizons"]),
        "target_count": len(names),
        "target_residual": {"kind": "seasonal_naive", "period": period, "target_features": names},
    }


def regime_configs(model_config, donors, *, freeze_epochs, unfreeze_learning_rate):
    """Build strict R1/R2/R3 model configs from a complete donor index."""
    result = {}
    for regime in ("R1", "R2", "R3"):
        config = copy.deepcopy(model_config)
        for branch in config["branches"]:
            branch.update(regime=regime, donor=donors["branches"][branch["name"]])
            if regime == "R3":
                branch.update(freeze_epochs=freeze_epochs,
                              unfreeze_learning_rate=unfreeze_learning_rate)
        config["core"].update(regime=regime, donor=donors["core"])
        if regime == "R3":
            config["core"].update(freeze_epochs=freeze_epochs,
                                  unfreeze_learning_rate=unfreeze_learning_rate)
        result[regime] = config
    return result


def _objective_identity(train, split, recipe_sha256):
    document = {
        "group": "modular.supervised_donor",
        "name": OBJECTIVE_NAME,
        "version": "1.0.0",
        "params": {"loss": "mae", "period": 24, "horizons": train["horizons"].tolist()},
        "train_sha256": train["sha256"],
        "recipe_sha256": recipe_sha256,
        "support": {"fit": [int(split["train"][0]), int(split["train"][-1])],
                    "validation": [int(split["validation"][0]), int(split["validation"][-1])],
                    "purged_origins": int(len(split["purged"]))},
    }
    return {**document, "sha256": identity(document)}


def run(train_npz, recipe_path, output_dir, fit_settings, *, validation_fraction=0.10,
        purge_origins=24, freeze_epochs=3, unfreeze_learning_rate=1e-4):
    """Fit once, export all trained temporal components, and verify reload parity."""
    from predictor_plugins.modular_temporal import build_modular, load_donor, save_donor, weights_hash
    from tools.modular_candidate_evaluator import fit_with_early_stopping
    import tensorflow as tf

    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("output directory already exists")
    recipe_path = Path(recipe_path)
    recipe = json.loads(recipe_path.read_text())
    recipe_sha = _sha256(recipe_path)
    data = load_train_npz(train_npz)
    split = internal_split_indices(data["timestamps"], data["target_timestamps"],
                                   validation_fraction=validation_fraction, purge_origins=purge_origins)
    model_config = expand_recipe(recipe, data["feature_names"].tolist())
    if list(model_config["horizons"]) != data["horizons"].tolist():
        raise ValueError("recipe horizons differ from TRAIN NPZ")
    if tuple(data["windows"].shape[1:]) != (model_config["window"], len(model_config["feature_names"])):
        raise ValueError("recipe dimensions differ from TRAIN NPZ")
    if data["targets"].shape[1:] != (len(model_config["horizons"]), model_config["target_count"]):
        raise ValueError("target dimensions differ from recipe")

    settings = dict(fit_settings)
    tf.keras.utils.set_random_seed(int(settings["seed"]))
    bundle = build_modular(model_config)
    fit = split["train"]
    validation = split["validation"]
    started = time.monotonic()
    training = fit_with_early_stopping(
        bundle.forecast_model,
        data["windows"][fit], data["targets"][fit],
        data["windows"][validation], data["targets"][validation],
        settings,
    )
    output.mkdir(parents=True)
    donors_dir = output / "donors"
    donors_dir.mkdir()
    objective = _objective_identity(data, split, recipe_sha)
    provenance = {
        "conditioning_contract": "OPERATIONAL",
        "learned_corpus": {
            "kind": "TRAIN_ONLY", "dataset_id": str(data["dataset_id"]),
            "data_sha256": data["sha256"],
            "support": "TRAIN origins only; final tail is internal validation; purged origins excluded",
            "pretrained_weights_source": None,
        },
        "reconstruction": {"state": "NOT_APPLICABLE"},
    }
    donor_paths = {"branches": {}}
    donor_records = []
    for name, model in bundle.branch_models.items():
        path = donors_dir / f"{name}.keras"
        sidecar = save_donor(model, path, bundle.donor_manifest("branch", name),
                             objective=objective, provenance=provenance)
        loaded = load_donor(path, bundle.donor_manifest("branch", name), require_contract="OPERATIONAL")
        if weights_hash(loaded) != weights_hash(model):
            raise ValueError(f"branch donor reload parity failed: {name}")
        donor_paths["branches"][name] = str(path)
        donor_records.append({"role": "branch", "name": name, "path": str(path),
                              "model_sha256": sidecar["model_sha256"],
                              "weights_sha256": sidecar["weights_sha256"]})
    core_path = donors_dir / "core.keras"
    core_manifest = bundle.donor_manifest("core")
    core_sidecar = save_donor(bundle.core_model, core_path, core_manifest,
                              objective=objective, provenance=provenance)
    loaded_core = load_donor(core_path, core_manifest, require_contract="OPERATIONAL")
    if weights_hash(loaded_core) != weights_hash(bundle.core_model):
        raise ValueError("core donor reload parity failed")
    donor_paths["core"] = str(core_path)
    donor_records.append({"role": "core", "name": "core", "path": str(core_path),
                          "model_sha256": core_sidecar["model_sha256"],
                          "weights_sha256": core_sidecar["weights_sha256"]})

    regimes = regime_configs(model_config, donor_paths, freeze_epochs=freeze_epochs,
                             unfreeze_learning_rate=unfreeze_learning_rate)
    for regime, config in regimes.items():
        _atomic(output / f"MODEL_{regime}.json", config)
        build_modular(config)  # all donors must load under their strict manifests
    receipt = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "objective": objective,
        "data": {"path": data["path"], "sha256": data["sha256"],
                 "dataset_id": str(data["dataset_id"]), "split": "train",
                 "rows": int(len(data["windows"])), "test_used": False,
                 "outer_validation_used": False, "holdout_used": False},
        "internal_split": {"fit_rows": [int(fit[0]), int(fit[-1])],
                           "validation_rows": [int(validation[0]), int(validation[-1])],
                           "purged_rows": split["purged"].tolist(),
                           "fit_target_end": int(data["target_timestamps"][fit].max()),
                           "validation_input_start": int(data["timestamps"][validation].min())},
        "training": training,
        "donors": donor_records,
        "regime_configs": {key: str(output / f"MODEL_{key}.json") for key in regimes},
        "wall_seconds": time.monotonic() - started,
    }
    _atomic(output / "PRETRAIN.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--internal-validation-fraction", type=float, default=0.10)
    parser.add_argument("--purge-origins", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--max-updates", type=int, default=1_000_000)
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    fit = {"loss": "mae", "seed": args.seed, "batch_size": args.batch_size,
           "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
           "max_epochs": args.max_epochs, "patience": args.patience,
           "min_delta": args.min_delta, "max_updates": args.max_updates,
           "max_seconds": args.max_seconds}
    receipt = run(args.train_npz, args.recipe, args.output, fit,
                  validation_fraction=args.internal_validation_fraction,
                  purge_origins=args.purge_origins)
    print(json.dumps({"status": receipt["status"], "objective": receipt["objective"]["sha256"],
                      "donors": len(receipt["donors"]), "wall_seconds": receipt["wall_seconds"]}))


if __name__ == "__main__":
    main()
