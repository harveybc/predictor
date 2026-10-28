#!/usr/bin/env python3
"""Audit retained FS4 extractors before branch-pretraining regimes use them.

The FS4 autoencoders are valuable extractibility evidence, but their HDF5
weights are not automatically modular-temporal branch donors.  This command
binds every selected feature/fold to retained bytes and reports compatibility
without deserializing or silently adapting an incompatible model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


SCHEMA = "predictor.i6d.branch_donor_audit.v1"
DEFAULT_FOLDS = tuple(f"inner_{year}" for year in range(2019, 2024))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED: {path}")
    return value


def _target_contract(design: dict) -> dict:
    groups = design.get("config", {}).get("feature_groups")
    contract = design.get("branch_only_contract", {})
    conv = contract.get("arms", {}).get("CONV", {})
    grid = contract.get("shared", {}).get("branch_time_grid")
    if not isinstance(groups, list) or not groups or not isinstance(grid, list) or not grid:
        raise ValueError("INVALID_BRANCH_ONLY_DESIGN")
    if conv.get("plugin") != "causal_conv1d":
        raise ValueError("CONV_BRANCH_PLUGIN_NOT_CAUSAL_CONV1D")
    params = conv.get("params", {})
    channels = params.get("channels")
    if type(channels) is not int or channels <= 0:
        raise ValueError("INVALID_CONV_CHANNELS")
    names = [item.get("name") for item in groups]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("INVALID_SELECTED_FEATURE_POPULATION")
    return {
        "features": names,
        "plugin": conv["plugin"],
        "params": params,
        "output_shape": [len(grid), channels],
        "output_grid": grid,
    }


def _index_results(root: Path, selected: set[str]) -> dict[tuple[str, str], tuple[dict, Path]]:
    indexed: dict[tuple[str, str], tuple[dict, Path]] = {}
    for path in sorted(root.glob("*/result.json")):
        result = _load(path)
        if (result.get("status") != "COMPLETE"
                or result.get("arm") != "TRAINED_ENCODER"
                or result.get("feature_id") not in selected):
            continue
        key = (result["feature_id"], result.get("fold_id"))
        if key in indexed:
            raise ValueError(f"DUPLICATE_TRAINED_ENCODER_RESULT: {key[0]} {key[1]}")
        indexed[key] = (result, path.parent)
    return indexed


def _audit_cell(result: dict, directory: Path, target: dict) -> dict:
    reasons: list[str] = []
    weights = directory / "chosen.weights.h5"
    declared = result.get("artifacts", {}).get("chosen_weights_file_sha256")
    if not weights.is_file():
        reasons.append("WEIGHTS_MISSING")
    elif not isinstance(declared, str) or _sha256(weights) != declared:
        reasons.append("WEIGHTS_DIGEST_MISMATCH")
    if reasons:
        return {"fold_id": result.get("fold_id"), "state": "INVALID_ARTIFACT", "reasons": reasons}

    architecture = result.get("architecture", {})
    hyper = result.get("hyper", {})
    source_shape = architecture.get("latent_shape")
    if hyper.get("window") != len(target["output_grid"]):
        reasons.append("INPUT_WINDOW_MISMATCH")
    if source_shape != target["output_shape"]:
        reasons.append("OUTPUT_TEMPORAL_GRID_MISMATCH")
    # FS4 stores a whole autoencoder checkpoint in HDF5.  The modular runtime
    # admits only a safe-mode .keras component plus its digest-bound sidecar.
    reasons.append("NOT_A_SEALED_MODULAR_DONOR")
    return {
        "fold_id": result.get("fold_id"),
        "state": "EVIDENCE_ONLY_INCOMPATIBLE" if reasons else "COMPATIBLE",
        "reasons": sorted(set(reasons)),
        "source": {
            "architecture_id": architecture.get("id"),
            "output_shape": source_shape,
            "weights_sha256": declared,
            "result_file": str(directory / "result.json"),
        },
    }


def audit_branch_donors(
    design: dict,
    runner_results: Path,
    *,
    expected_folds: Iterable[str] = DEFAULT_FOLDS,
) -> dict:
    """Return a deterministic compatibility audit for selected feature/folds."""
    target = _target_contract(design)
    folds = tuple(expected_folds)
    if not folds or len(set(folds)) != len(folds):
        raise ValueError("INVALID_EXPECTED_FOLDS")
    indexed = _index_results(Path(runner_results), set(target["features"]))
    features = []
    verified = compatible = missing = 0
    for feature in target["features"]:
        rows = []
        for fold in folds:
            found = indexed.get((feature, fold))
            if found is None:
                rows.append({"fold_id": fold, "state": "MISSING", "reasons": ["TRAINED_ENCODER_RESULT_MISSING"]})
                missing += 1
                continue
            cell = _audit_cell(*found, target)
            rows.append(cell)
            if cell["state"] != "INVALID_ARTIFACT":
                verified += 1
            if cell["state"] == "COMPATIBLE":
                compatible += 1
        features.append({"feature_id": feature, "folds": rows})
    return {
        "schema": SCHEMA,
        "design_sha256": design.get("design_sha256"),
        "target_branch": target,
        "features": features,
        "summary": {
            "features": len(features),
            "expected_feature_folds": len(features) * len(folds),
            "verified_extractibility_artifacts": verified,
            "compatible_branch_donors": compatible,
            "missing_feature_folds": missing,
        },
        "decision": "PRETRAIN_MATCHING_BRANCH_DONORS" if compatible == 0 else "REVIEW_COMPATIBLE_DONORS",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True)
    parser.add_argument("--runner-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", action="append", dest="folds")
    args = parser.parse_args(argv)
    report = audit_branch_donors(
        _load(Path(args.design)),
        Path(args.runner_results),
        expected_folds=args.folds or DEFAULT_FOLDS,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
