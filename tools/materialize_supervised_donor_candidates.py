#!/usr/bin/env python3
"""Materialize evaluator candidates from one complete supervised-donor receipt.

The bridge consumes, verifies, and binds the producer's existing PRETRAIN.json,
MODEL_R1/R2/R3 files, Keras archives, and schema-2 sidecars. It never rewrites
those artifacts or creates a historical amendment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


BINDING_SCHEMA = "predictor.modular.donor_binding.v2"
PRETRAIN_SCHEMA = "modular.supervised_donor.pretrain.v1"
REGIMES = ("R1", "R2", "R3")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(document) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(payload.encode("utf-8"))


def _read_json(path, label):
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(document, allow_nan=False)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label}_INVALID: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label}_INVALID: expected an object")
    return document


def _load_and_hash_weights(path, expected_manifest, require_contract=None):
    """Deserialize through the strict donor loader and return ordered weight identity."""
    from predictor_plugins.modular_temporal import load_donor, weights_hash

    model = load_donor(path, expected_manifest, require_contract=require_contract)
    return weights_hash(model)


def _validated_settings(settings):
    from tools.modular_candidate_evaluator import _settings

    if not isinstance(settings, dict):
        raise ValueError("settings: expected an object")
    try:
        validated = _settings({"evaluator": settings})
        json.dumps(validated, allow_nan=False)
    except (TypeError, ValueError) as exc:
        message = str(exc)
        key = next((name for name in settings if name in message), "settings")
        raise ValueError(f"{key}: {message}") from exc
    return validated


def _validated_objective(objective):
    expected = {"metric", "split", "higher_is_better", "unit"}
    directions = {"MAE": False, "MSE": False, "skill_MAE": True, "skill_MSE": True}
    if (not isinstance(objective, dict) or set(objective) != expected
            or objective.get("metric") not in directions
            or objective.get("split") != "validation"
            or objective.get("higher_is_better") is not directions.get(objective.get("metric"))
            or not isinstance(objective.get("unit"), str) or not objective["unit"].strip()):
        raise ValueError("OBJECTIVE_INVALID: expected the evaluator's explicit validation objective")
    return json.loads(json.dumps(objective, sort_keys=True, allow_nan=False))


def _validate_objective_identity(objective):
    if not isinstance(objective, dict) or set(objective) != {
            "group", "name", "version", "params", "train_sha256", "recipe_sha256", "support", "sha256"}:
        raise ValueError("PRETRAIN_OBJECTIVE_INVALID")
    body = {key: value for key, value in objective.items() if key != "sha256"}
    if _canonical_sha256(body) != objective["sha256"]:
        raise ValueError("PRETRAIN_OBJECTIVE_DIGEST_MISMATCH")


def _donor_records(receipt, root):
    records = receipt.get("donors")
    if not isinstance(records, list) or not records:
        raise ValueError("PRETRAIN_DONORS_INVALID")
    by_path = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
                "role", "name", "path", "model_sha256", "weights_sha256"}:
            raise ValueError("PRETRAIN_DONOR_RECORD_INVALID")
        path = Path(record["path"]).resolve()
        if path.parent != root / "donors" or path.suffix != ".keras" or str(path) in by_path:
            raise ValueError("PRETRAIN_DONOR_PATH_INVALID")
        if record["role"] not in ("branch", "core"):
            raise ValueError("PRETRAIN_DONOR_ROLE_INVALID")
        by_path[str(path)] = record
    return by_path


def _verify_donor(path, record):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"DONOR_MODEL_MISSING: {path}")
    if _sha256_file(path) != record["model_sha256"]:
        raise ValueError(f"DONOR_MODEL_SHA256_MISMATCH: {path.name}")
    sidecar_path = path.with_suffix(".manifest.json")
    if not sidecar_path.is_file():
        raise ValueError(f"DONOR_MANIFEST_MISSING: {sidecar_path}")
    sidecar = _read_json(sidecar_path, "DONOR_MANIFEST")
    required = {"schema", "manifest", "manifest_sha256", "model_sha256", "weights_sha256", "provenance"}
    if set(sidecar) != required or sidecar.get("schema") != 2:
        raise ValueError(f"DONOR_MANIFEST_SCHEMA_NOT_2: {path.name}")
    if sidecar["model_sha256"] != record["model_sha256"]:
        raise ValueError(f"DONOR_MODEL_SHA256_MISMATCH: {path.name}")
    if sidecar["weights_sha256"] != record["weights_sha256"]:
        raise ValueError(f"DONOR_WEIGHTS_SHA256_MISMATCH: {path.name}")
    if _canonical_sha256(sidecar["manifest"]) != sidecar["manifest_sha256"]:
        raise ValueError(f"DONOR_MANIFEST_DIGEST_MISMATCH: {path.name}")
    if (sidecar["manifest"].get("role") != record["role"]
            or sidecar["manifest"].get("name") != record["name"]):
        raise ValueError(f"DONOR_MANIFEST_IDENTITY_MISMATCH: {path.name}")
    if (not isinstance(sidecar["provenance"], dict)
            or sidecar["provenance"].get("conditioning_contract") != "OPERATIONAL"):
        raise ValueError(f"DONOR_CONTRACT_NOT_OPERATIONAL: {path.name}")
    actual_weights = _load_and_hash_weights(path, sidecar["manifest"], require_contract="OPERATIONAL")
    if actual_weights != sidecar["weights_sha256"]:
        raise ValueError(f"DONOR_WEIGHTS_SHA256_MISMATCH: {path.name}")
    return {"keras": sidecar["model_sha256"], "manifest": _sha256_file(sidecar_path),
            "weights": sidecar["weights_sha256"]}


def _model_donor_paths(model, regime):
    if not isinstance(model.get("branches"), list) or not model["branches"] or not isinstance(model.get("core"), dict):
        raise ValueError(f"MODEL_{regime}_INVALID")
    components = [*model["branches"], model["core"]]
    paths = []
    for component in components:
        if component.get("regime") != regime or not isinstance(component.get("donor"), str):
            raise ValueError(f"MODEL_{regime}_REGIME_OR_DONOR_INVALID")
        paths.append(str(Path(component["donor"]).resolve()))
    if len(paths) != len(set(paths)):
        raise ValueError(f"MODEL_{regime}_DUPLICATE_DONOR")
    return paths


def _target_identity(receipt, names, target_count):
    """Resolve targets from the receipt, or its digest-bound TRAIN NPZ."""
    data = receipt.get("data", {})
    target_indices = data.get("target_feature_indices")
    target_names = data.get("target_names")
    if target_indices is None and target_count == len(names):
        return list(range(target_count)), list(names)
    if target_indices is None:
        path = Path(data.get("path", ""))
        if not path.is_file() or _sha256_file(path) != data.get("sha256"):
            raise ValueError("PRETRAIN_TARGET_SOURCE_INVALID")
        try:
            with np.load(path, allow_pickle=False) as source:
                source_names = [str(value) for value in source["feature_names"].tolist()]
                target_names = [str(value) for value in source["target_names"].tolist()]
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError("PRETRAIN_TARGET_SOURCE_INVALID") from exc
        if source_names != names:
            raise ValueError("PRETRAIN_TARGET_SOURCE_FEATURES_MISMATCH")
        try:
            target_indices = [names.index(name) for name in target_names]
        except ValueError as exc:
            raise ValueError("PRETRAIN_TARGET_SOURCE_TARGET_UNKNOWN") from exc
    if (not isinstance(target_indices, list) or len(target_indices) != target_count
            or any(type(index) is not int or index < 0 or index >= len(names)
                   for index in target_indices)
            or len(set(target_indices)) != len(target_indices)
            or not isinstance(target_names, list)
            or target_names != [names[index] for index in target_indices]):
        raise ValueError("PRETRAIN_TARGET_IDENTITY_INVALID")
    return target_indices, target_names


def materialize(pretrain_path, evaluator_settings, objective):
    """Return complete R1/R2/R3 evaluator candidates after strict verification."""
    pretrain_path = Path(pretrain_path).resolve()
    root = pretrain_path.parent
    receipt = _read_json(pretrain_path, "PRETRAIN")
    if receipt.get("schema") != PRETRAIN_SCHEMA or receipt.get("status") != "COMPLETE":
        raise ValueError("PRETRAIN_NOT_COMPLETE")
    _validate_objective_identity(receipt.get("objective"))
    settings = _validated_settings(evaluator_settings)
    candidate_objective = _validated_objective(objective)
    records = _donor_records(receipt, root)

    verified = {path: _verify_donor(path, record) for path, record in records.items()}
    configs = receipt.get("regime_configs")
    if not isinstance(configs, dict) or set(configs) != set(REGIMES):
        raise ValueError("PRETRAIN_REGIME_CONFIGS_INVALID")
    candidates = {}
    for regime in REGIMES:
        model_path = Path(configs[regime]).resolve()
        expected_path = root / f"MODEL_{regime}.json"
        if model_path != expected_path or not model_path.is_file():
            raise ValueError(f"MODEL_{regime}_PATH_INVALID")
        model = _read_json(model_path, f"MODEL_{regime}")
        used = _model_donor_paths(model, regime)
        missing = [path for path in used if path not in records]
        if missing:
            raise ValueError(f"MODEL_DONOR_NOT_IN_PRETRAIN: {missing[0]}")
        if set(used) != set(records):
            raise ValueError(f"MODEL_{regime}_DONOR_SET_INCOMPLETE")
        for component in model["branches"]:
            record = records[str(Path(component["donor"]).resolve())]
            if record["role"] != "branch" or record["name"] != component.get("name"):
                raise ValueError(f"MODEL_{regime}_BRANCH_DONOR_IDENTITY_MISMATCH")
        core_record = records[str(Path(model["core"]["donor"]).resolve())]
        if core_record["role"] != "core" or core_record["name"] != "core":
            raise ValueError(f"MODEL_{regime}_CORE_DONOR_IDENTITY_MISMATCH")
        names = model.get("feature_names")
        target_count = model.get("target_count")
        if (not isinstance(names, list) or not names or len(set(names)) != len(names)
                or type(target_count) is not int or not 0 < target_count <= len(names)):
            raise ValueError(f"MODEL_{regime}_TARGETS_INVALID")
        target_indices, _ = _target_identity(receipt, names, target_count)
        binding = {
            "schema": BINDING_SCHEMA,
            "required_contract": "OPERATIONAL",
            "pretrain": {"path": str(pretrain_path), "sha256": _sha256_file(pretrain_path)},
            "regime_config": {"path": str(model_path), "sha256": _sha256_file(model_path)},
            "donor_objective_sha256": receipt["objective"]["sha256"],
            "donors": {path: verified[path] for path in used},
        }
        candidates[regime] = {
            "modular_candidate": {"schema": "modular.candidate.v1", "donor_binding": binding},
            "model": model,
            "evaluator": settings,
            "target_feature_indices": target_indices,
            "objective": candidate_objective,
        }
        json.dumps(candidates[regime], allow_nan=False)
    return candidates


def write_candidates(pretrain_path, evaluator_settings, objective, output_dir):
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ValueError("output directory already exists")
    candidates = materialize(pretrain_path, evaluator_settings, objective)
    destination.mkdir(parents=True)
    files = {}
    for regime, candidate in candidates.items():
        path = destination / f"CANDIDATE_{regime}.json"
        path.write_text(json.dumps(candidate, indent=2, sort_keys=True, allow_nan=False) + "\n")
        files[regime] = {"path": str(path), "sha256": _sha256_file(path)}
    receipt = {"schema": "predictor.modular.donor_materialization.v1", "status": "COMPLETE",
               "source_pretrain_sha256": _sha256_file(pretrain_path), "candidates": files}
    temporary = destination / "MATERIALIZATION.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, destination / "MATERIALIZATION.json")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--evaluator-json", required=True)
    parser.add_argument("--objective-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluator = _read_json(args.evaluator_json, "EVALUATOR")
    objective = _read_json(args.objective_json, "OBJECTIVE")
    result = write_candidates(args.pretrain, evaluator, objective, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
