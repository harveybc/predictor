"""Materialize branch-only or core-only donor ablations from a bound candidate."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from tools.modular_heartbeat import verify_donor_binding


MODES = ("branches_only", "core_only")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def materialize(source_path, mode, output_dir):
    """Create one mixed R0/R2 candidate while retaining exact donor provenance."""
    if mode not in MODES:
        raise ValueError(f"unknown ablation mode: {mode}")
    source_path = Path(source_path).resolve()
    candidate = json.loads(source_path.read_text())
    verify_donor_binding(candidate)
    candidate = copy.deepcopy(candidate)
    model = candidate["model"]
    binding = candidate["modular_candidate"]["donor_binding"]

    retained = model["branches"] if mode == "branches_only" else [model["core"]]
    randomized = [model["core"]] if mode == "branches_only" else model["branches"]
    retained_paths = {component["donor"] for component in retained}
    if None in retained_paths or retained_paths - set(binding["donors"]):
        raise ValueError("retained component has no bound donor")
    for component in randomized:
        component["regime"] = "R0"
        component["donor"] = None
        component.pop("freeze_epochs", None)
        component.pop("unfreeze_learning_rate", None)

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ValueError("output directory already exists")
    destination.mkdir(parents=True)
    model_path = destination / f"MODEL_{mode}.json"
    _write_atomic(model_path, model)
    binding["regime_config"] = {"path": str(model_path), "sha256": _sha256(model_path)}
    binding["donors"] = {path: record for path, record in binding["donors"].items()
                         if path in retained_paths}
    verify_donor_binding(candidate)

    candidate_path = destination / f"CANDIDATE_{mode}.json"
    _write_atomic(candidate_path, candidate)
    receipt = {
        "schema": "predictor.modular.donor_ablation.v1",
        "status": "COMPLETE",
        "mode": mode,
        "source_candidate": {"path": str(source_path), "sha256": _sha256(source_path)},
        "candidate": {"path": str(candidate_path), "sha256": _sha256(candidate_path)},
        "retained_donors": len(retained_paths),
    }
    _write_atomic(destination / "MATERIALIZATION.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source, args.mode, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
