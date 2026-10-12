"""Prepare the H72 data artifacts without STLFSI4 (owner order 2026-10-12); never fits.

Derives, into a NEW staging data root (never overwriting a published one):

``inputs/<name>-no-stlfsi4/``  INPUT.npz with the STLFSI4 column removed (every
    other column byte-identical, same ts/price), CLOCK.json bound to the new
    input and the 19-feature list, CALENDAR/CLOCK_SCOPE/PREPARATION copied
    unchanged, and DERIVATION.json proving the column drop.  The feature is
    removed, not shifted by an assumed lag.
``c/plan-v6-no-stlfsi4/cells/*_h72.json``  every H72 cell (four inner folds and
    the weekly units) rewritten as an ``i6n.historical_cell.v2`` cell: the
    19 gated features, the new input/clock pins, the manifest and availability
    contract digests.  Every other key is unchanged.
``c/plan-v6-no-stlfsi4/VIEWS.h72.json`` / ``ARTIFACTS.json``  the H72 units with
    their new cell pins and the superseded cell digests.

Feature scaling stays inside each fit (TRAIN rows only); there is no scaler
prep artifact to recompute.  Frozen supports are sealed afterwards with
``tools.i6n_historical_evaluator freeze --data-root`` on the staged cells.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from tools.i6m_data import file_hash
from tools.i6n_contracts import feature_availability as availability
from tools.i6n_historical_cell import SCHEMA_V1, SCHEMA_V2

SUFFIX = "-no-stlfsi4"
PLAN_DIR = "plan-v6" + SUFFIX
COPIED = ("CALENDAR.json", "CLOCK_SCOPE.json", "PREPARATION.json")


def _json(path: Path, value) -> str:
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return file_hash(path)


def derive_input(source: Path, target: Path, drop=availability.EXCLUDED_FEATURES) -> dict:
    """Write ``target`` (must not exist) = ``source`` minus the ``drop`` columns."""
    source, target = Path(source), Path(target)
    clock = json.loads((source / "CLOCK.json").read_text(encoding="utf-8"))
    if clock.get("input_sha256") != file_hash(source / "INPUT.npz"):
        raise ValueError("source CLOCK does not bind the source INPUT")
    features = list(clock["features"])
    if features != list(availability.LEGACY_MANIFEST_20):
        raise ValueError("source input is not the sealed 20-feature manifest")
    keep = [i for i, f in enumerate(features) if f not in drop]
    dropped = [i for i, f in enumerate(features) if f in drop]
    if len(dropped) != len(drop):
        raise ValueError("a feature to drop is absent from the source manifest")
    with np.load(source / "INPUT.npz", allow_pickle=False) as archive:
        ts, x, price = (archive[k].copy() for k in ("ts", "x", "price"))
    reduced = np.ascontiguousarray(x[:, keep])
    for new_index, old_index in enumerate(keep):  # every retained column is untouched
        if reduced[:, new_index].tobytes() != x[:, old_index].tobytes():
            raise AssertionError("retained column changed")
    target.mkdir(parents=True, exist_ok=False)
    np.savez(target / "INPUT.npz", ts=ts, x=reduced, price=price)
    new_features = [features[i] for i in keep]
    if new_features != list(availability.MANIFEST_NO_STLFSI4):
        raise ValueError("derived feature order differs from the gated manifest")
    for name in COPIED:
        shutil.copyfile(source / name, target / name)
    new_clock = dict(clock, features=new_features, input_sha256=file_hash(target / "INPUT.npz"),
                     clock_policy=(clock["clock_policy"] + "; STLFSI4 column removed by owner ruling "
                                   "2026-10-12 (point-in-time exclusion, no lag shift); every other value "
                                   "byte-identical to the source input " + clock["input_sha256"]))
    clock_sha = _json(target / "CLOCK.json", new_clock)
    derivation = {
        "schema": "i6n.input_feature_exclusion.v1",
        "excluded_features": [features[i] for i in dropped],
        "excluded_column_indices": dropped,
        "shifted_by_assumed_lag": False,
        "source": {"input_sha256": clock["input_sha256"], "clock_sha256": file_hash(source / "CLOCK.json"),
                   "features": features},
        "target": {"input_sha256": new_clock["input_sha256"], "clock_sha256": clock_sha,
                   "features": new_features,
                   "feature_manifest_sha256": availability.manifest_sha256(new_features)},
        "unchanged": {"ts": True, "price": True, "retained_columns_bytewise": True,
                      "copied_files_sha256": {n: file_hash(target / n) for n in COPIED}},
        "rows": int(len(ts)),
    }
    _json(target / "DERIVATION.json", derivation)
    return derivation


def convert_cell(cell: dict, input_pins: dict, contract_sha256: str) -> dict:
    """v1 20-feature H72 cell -> v2 gated 19-feature cell; nothing else changes."""
    if cell.get("schema") != SCHEMA_V1 or cell["features"] != list(availability.LEGACY_MANIFEST_20):
        raise ValueError("expected a sealed 20-feature v1 cell")
    old_input = cell["input"]["path"]
    if old_input not in input_pins:
        raise ValueError(f"no derived input for {old_input}")
    pins = input_pins[old_input]
    features = list(availability.MANIFEST_NO_STLFSI4)
    out = deepcopy(cell)
    out.update(schema=SCHEMA_V2, features=features,
               input={"path": pins["input_path"], "sha256": pins["input_sha256"]},
               source_clock={"path": pins["clock_path"], "sha256": pins["clock_sha256"]},
               feature_manifest_sha256=availability.manifest_sha256(features),
               availability_contract_sha256=contract_sha256)
    return out


def prepare(source_root: Path, stage_root: Path, horizon: int = 72) -> dict:
    source_root, stage_root = Path(source_root).resolve(), Path(stage_root).resolve()
    if stage_root.exists():
        raise FileExistsError("stage root already exists; never overwrite a published root")
    contract = availability.load_contract()
    availability.gate(list(availability.MANIFEST_NO_STLFSI4), contract)
    cells_dir = source_root / "c" / "plan-v6" / "cells"
    cells = sorted(cells_dir.glob(f"*_h{horizon}.json"))
    if not cells:
        raise ValueError("no source cells for this horizon")
    input_pins, derivations = {}, {}
    for path in cells:
        cell = json.loads(path.read_text(encoding="utf-8"))
        rel = cell["input"]["path"]
        if rel in input_pins:
            continue
        source_dir = (path.parent / rel).resolve().parent
        name = source_dir.name
        target_dir = stage_root / "inputs" / (name + SUFFIX)
        derivations[name] = derive_input(source_dir, target_dir)
        if cell["input"]["sha256"] != derivations[name]["source"]["input_sha256"]:
            raise ValueError(f"cell {path.name} pins a different source input")
        input_pins[rel] = {
            "input_path": f"../../../inputs/{name}{SUFFIX}/INPUT.npz",
            "input_sha256": derivations[name]["target"]["input_sha256"],
            "clock_path": f"../../../inputs/{name}{SUFFIX}/CLOCK.json",
            "clock_sha256": derivations[name]["target"]["clock_sha256"],
        }
    out_dir = stage_root / "c" / PLAN_DIR / "cells"
    out_dir.mkdir(parents=True)
    units = []
    for path in cells:
        cell = json.loads(path.read_text(encoding="utf-8"))
        new = convert_cell(cell, input_pins, contract["contract_sha256"])
        new_path = out_dir / path.name
        sha = _json(new_path, new)
        units.append({
            "unit_id": path.name[: -len(f"_h{horizon}.json")],
            "kind": "INNER_FOLD" if path.name.startswith("2023Q") else "WEEKLY_REFIT",
            "horizon_hours": horizon,
            "cell_config": {"relpath": new_path.relative_to(stage_root).as_posix(), "sha256": sha},
            "superseded_cell_sha256": file_hash(path),
            "input": derivations[Path(cell["input"]["path"]).parent.name]["target"]["input_sha256"],
        })
    views = {
        "schema": "i6n.weekly_campaign.views.h72_no_stlfsi4.v1",
        "horizon_hours": horizon,
        "features": list(availability.MANIFEST_NO_STLFSI4),
        "feature_manifest_sha256": availability.manifest_sha256(availability.MANIFEST_NO_STLFSI4),
        "availability_contract_sha256": contract["contract_sha256"],
        "folds": [u for u in units if u["kind"] == "INNER_FOLD"],
        "weeks": [u for u in units if u["kind"] == "WEEKLY_REFIT"],
    }
    views_sha = _json(stage_root / "c" / PLAN_DIR / "VIEWS.h72.json", views)
    artifacts = {
        "schema": "i6n.h72_no_stlfsi4.artifacts.v1",
        "cells": len(units),
        "views_sha256": views_sha,
        "inputs": {name: d["target"] for name, d in sorted(derivations.items())},
        "cell_sha256": {u["unit_id"]: u["cell_config"]["sha256"] for u in units},
    }
    _json(stage_root / "c" / PLAN_DIR / "ARTIFACTS.json", artifacts)
    return {"views_sha256": views_sha, "cells": len(units), "inputs": sorted(derivations),
            "folds": [u["unit_id"] for u in views["folds"]]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=72)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.source_root, args.stage_root, args.horizon), sort_keys=True))


if __name__ == "__main__":
    main()
