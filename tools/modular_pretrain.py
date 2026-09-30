"""Train branch donors, materialize their fused sequences, then train a core donor.

Only explicit train and internal train-validation populations are accepted.
Population boundaries are caller declarations; array digests are computed here.
This module does not grant data-governance authority or read an external test.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _population_pair(train, validation):
    for pop, expected in ((train, "train"), (validation, "train_validation")):
        if pop.get("split") != expected or pop.get("time_unit") != "seconds":
            raise ValueError("pretraining requires train and internal train_validation in seconds")
        if not isinstance(pop.get("dataset_id"), str) or not pop["dataset_id"]:
            raise ValueError("dataset_id is required")
        for field in ("support_start", "support_end"):
            value = pop.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid {field}")
        if pop["support_start"] > pop["support_end"]:
            raise ValueError("reversed support")
    if train["dataset_id"] != validation["dataset_id"]:
        raise ValueError("dataset identity differs")
    if train["support_end"] >= validation["support_start"]:
        raise ValueError("train and internal validation supports overlap")


def _array_identity(values, batch_size=64):
    if len(values.shape) != 3 or not len(values) or any(d <= 0 for d in values.shape):
        raise ValueError("nonempty rank-three windows required")
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(values.shape), "dtype": str(values.dtype)}, sort_keys=True).encode())
    for start in range(0, len(values), batch_size):
        block = np.ascontiguousarray(values[start:start + batch_size])
        if not np.issubdtype(block.dtype, np.floating) or not np.isfinite(block).all():
            raise ValueError("finite floating point windows required")
        digest.update(block.tobytes())
    return {"shape": list(values.shape), "dtype": str(values.dtype), "sha256": digest.hexdigest()}


class FeatureView:
    """Select channels only for the batch being consumed."""
    def __init__(self, source, indices):
        self.source, self.indices = source, indices
        self.shape = (*source.shape[:2], len(indices))
        self.dtype = source.dtype

    def __len__(self):
        return len(self.source)

    def __getitem__(self, key):
        return self.source[key][..., self.indices]


def _materialize(model, values, path, batch_size):
    shape = (len(values), *tuple(int(n) for n in model.output_shape[1:]))
    output = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=shape)
    for start in range(0, len(values), batch_size):
        encoded = np.asarray(model(values[start:start + batch_size], training=False))
        if not np.isfinite(encoded).all():
            raise ValueError("nonfinite fused representation")
        output[start:start + len(encoded)] = encoded
    output.flush()
    return output


def pretrain_components(config, train_x, validation_x, output_dir, fit_config,
                        train_population, validation_population):
    """Return manifests and an R2 config, preserving the caller's random config."""
    _population_pair(train_population, validation_population)
    train_identity = _array_identity(train_x)
    validation_identity = _array_identity(validation_x)
    if train_x.shape[1:] != validation_x.shape[1:]:
        raise ValueError("train/validation window schemas differ")
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty")
    from predictor_plugins.modular_temporal import build_autoencoder, build_modular, save_donor
    from tools.modular_candidate_evaluator import fit_with_early_stopping

    base = copy.deepcopy(config)
    if any(c.get("regime", "R0") != "R0" or c.get("donor")
           for c in [*base.get("branches", []), base.get("core", {})]):
        raise ValueError("fresh pretraining requires R0 components without donors")
    bundle = build_modular(base)
    if tuple(bundle.forecast_model.input_shape[1:]) != tuple(train_x.shape[1:]):
        raise ValueError("data does not match model input")
    out.mkdir(parents=True, exist_ok=True)
    resolved = copy.deepcopy(bundle.config)
    names = resolved["feature_names"]
    records = []
    for index, spec in enumerate(resolved["branches"]):
        name = spec["name"]
        columns = [names.index(feature) for feature in spec["features"]]
        train_view, val_view = FeatureView(train_x, columns), FeatureView(validation_x, columns)
        encoder = bundle.branch_models[name]
        ae = build_autoencoder(encoder)
        training = fit_with_early_stopping(ae, train_view, train_view, val_view, val_view, fit_config)
        donor = out / f"branch_{index:03d}.keras"
        save_donor(encoder, donor, bundle.donor_manifest("branch", name))
        records.append({"name": name, "donor": str(donor), "training": training})
        spec.update(regime="R2", donor=str(donor))

    batch_size = int(fit_config.get("batch_size", 32))
    fused_train = _materialize(bundle.fusion_model, train_x, out / "fused_train.npy", batch_size)
    fused_validation = _materialize(bundle.fusion_model, validation_x, out / "fused_train_validation.npy", batch_size)
    core_ae = build_autoencoder(bundle.core_model)
    core_training = fit_with_early_stopping(core_ae, fused_train, fused_train,
                                           fused_validation, fused_validation, fit_config)
    donor = out / "core.keras"
    save_donor(bundle.core_model, donor, bundle.donor_manifest("core"))
    resolved["core"].update(regime="R2", donor=str(donor))
    result = {"schema": "modular.pretrain.v1", "branches": records,
              "core": {"donor": str(donor), "training": core_training},
              "train_population": train_population, "validation_population": validation_population,
              "train_input": train_identity, "validation_input": validation_identity,
              "fused_train": _array_identity(fused_train),
              "fused_validation": _array_identity(fused_validation),
              "fine_tune_config": resolved,
              "scope": "representation pretraining; downstream utility unmeasured"}
    (out / "PRETRAIN.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
