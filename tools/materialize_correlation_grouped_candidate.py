#!/usr/bin/env python3
"""Regroup an R0 modular candidate by TRAIN-only absolute correlation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


SCHEMA = "predictor.modular.correlation_grouping.v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def correlation_groups(windows, feature_names, group_count):
    """Cluster features from one observation per TRAIN origin, never targets."""
    if isinstance(group_count, bool) or not isinstance(group_count, int) or group_count < 2:
        raise ValueError("group_count must be an integer >= 2")
    x = np.asarray(windows)
    names = [str(name) for name in feature_names]
    if x.ndim != 3 or x.shape[2] != len(names) or group_count > len(names):
        raise ValueError("windows and feature names are incompatible")
    observations = np.asarray(x[:, -1, :], dtype=np.float64)
    if not np.isfinite(observations).all():
        raise ValueError("TRAIN windows must be finite")
    correlation = np.corrcoef(observations, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)
    distance = np.clip(1.0 - np.abs(correlation), 0.0, 1.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    labels = fcluster(linkage(squareform(distance, checks=True), method="average"),
                      t=group_count, criterion="maxclust")
    groups = [[name for name, label in zip(names, labels) if label == group]
              for group in sorted(set(labels.tolist()))]
    if len(groups) != group_count or sorted(sum(groups, [])) != sorted(names):
        raise ValueError("clustering did not produce the requested partition")
    return groups


def materialize(source_path, train_npz, output_dir, group_count=3):
    source_path, train_npz = Path(source_path).resolve(), Path(train_npz).resolve()
    candidate = json.loads(source_path.read_text())
    model = candidate["model"]
    components = [*model["branches"], model["core"]]
    if any(component.get("regime") != "R0" or component.get("donor") is not None
           for component in components):
        raise ValueError("source candidate must be entirely R0")
    with np.load(train_npz, allow_pickle=False) as source:
        if str(source["split"]) != "train":
            raise ValueError("only TRAIN data may define groups")
        names = source["feature_names"].astype(str).tolist()
        groups = correlation_groups(source["windows"], names, group_count)
    if names != model["feature_names"]:
        raise ValueError("candidate and TRAIN feature identities differ")
    templates = model["branches"]
    if len(templates) != group_count:
        raise ValueError("parameter matching requires the same branch count")
    candidate = copy.deepcopy(candidate)
    for index, (branch, features) in enumerate(zip(candidate["model"]["branches"], groups)):
        branch["name"] = f"correlation_{index}"
        branch["features"] = features
    candidate["modular_candidate"]["grouping"] = {
        "schema": SCHEMA,
        "method": "average_linkage_1_minus_absolute_pearson",
        "observations": "last input step of each TRAIN origin",
        "train_npz": {"path": str(train_npz), "sha256": _sha256(train_npz)},
        "groups": groups,
    }
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ValueError("output directory already exists")
    destination.mkdir(parents=True)
    path = destination / "CANDIDATE_correlation.json"
    _atomic(path, candidate)
    receipt = {"schema": SCHEMA, "status": "COMPLETE",
               "source": {"path": str(source_path), "sha256": _sha256(source_path)},
               "candidate": {"path": str(path), "sha256": _sha256(path)},
               "group_sizes": [len(group) for group in groups]}
    _atomic(destination / "MATERIALIZATION.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source, args.train_npz, args.output, args.groups),
                     sort_keys=True))


if __name__ == "__main__":
    main()
