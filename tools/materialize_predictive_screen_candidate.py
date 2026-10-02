#!/usr/bin/env python3
"""Create an exploratory R0 candidate from TRAIN-only lead association."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from tools.materialize_correlation_grouped_candidate import (
    SCHEMA as GROUPING_SCHEMA,
    _atomic,
    _sha256,
    correlation_groups,
)


SCHEMA = "predictor.modular.predictive_screen.v1"


def screen(windows, targets, feature_names, *, top_k, fit_stop):
    """Rank features by maximum absolute Spearman association across horizons."""
    x, y = np.asarray(windows), np.asarray(targets)
    names = [str(name) for name in feature_names]
    if (x.ndim != 3 or y.ndim != 3 or len(x) != len(y)
            or x.shape[2] != len(names) or y.shape[2] != 1):
        raise ValueError("TRAIN windows, targets and feature names are incompatible")
    if (isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(names)
            or isinstance(fit_stop, bool) or not isinstance(fit_stop, int)
            or not 2 <= fit_stop <= len(x)):
        raise ValueError("invalid top_k or fit_stop")
    observations = np.asarray(x[:fit_stop, -1, :], dtype=np.float64)
    outcomes = np.asarray(y[:fit_stop, :, 0], dtype=np.float64)
    scores = []
    for index, name in enumerate(names):
        by_horizon = []
        for horizon in range(outcomes.shape[1]):
            value = float(spearmanr(observations[:, index], outcomes[:, horizon]).statistic)
            by_horizon.append(0.0 if not np.isfinite(value) else value)
        scores.append({"feature": name, "index": index, "by_horizon": by_horizon,
                       "score": max(abs(value) for value in by_horizon)})
    ranked = sorted(scores, key=lambda row: (-row["score"], row["index"]))
    selected = {row["feature"] for row in ranked[:top_k]}
    return [name for name in names if name in selected], scores


def materialize(source_path, train_npz, output_dir, *, top_k=32, group_count=3,
                validation_fraction=0.1, purge_origins=36):
    source_path, train_npz = Path(source_path).resolve(), Path(train_npz).resolve()
    candidate = json.loads(source_path.read_text())
    model = candidate["model"]
    if any(component.get("regime") != "R0" or component.get("donor") is not None
           for component in [*model["branches"], model["core"]]):
        raise ValueError("source candidate must be entirely R0")
    if len(model["branches"]) != group_count:
        raise ValueError("parameter matching requires the source branch count")
    with np.load(train_npz, allow_pickle=False) as source:
        if str(source["split"]) != "train":
            raise ValueError("only TRAIN data may define the screen")
        names = source["feature_names"].astype(str).tolist()
        rows = len(source["windows"])
        validation_rows = max(1, int(np.ceil(rows * validation_fraction)))
        fit_stop = rows - validation_rows - purge_origins
        selected, scores = screen(source["windows"], source["targets"], names,
                                  top_k=top_k, fit_stop=fit_stop)
        selected_indices = [names.index(name) for name in selected]
        groups = correlation_groups(source["windows"][:fit_stop, :, selected_indices],
                                    selected, group_count)
    if names != model["feature_names"]:
        raise ValueError("candidate and TRAIN feature identities differ")
    candidate = copy.deepcopy(candidate)
    for index, (branch, features) in enumerate(zip(candidate["model"]["branches"], groups)):
        branch["name"] = f"screened_{index}"
        branch["features"] = features
    excluded = sorted(set(names) - set(selected), key=names.index)
    candidate["model"]["excluded_features"] = {
        name: "exploratory TRAIN-only lead-association screen outside top_k"
        for name in excluded
    }
    candidate["modular_candidate"]["screening"] = {
        "schema": SCHEMA,
        "status": "EXPLORATORY_ASSOCIATION_NOT_CAUSAL_SELECTION",
        "statistic": "max_absolute_spearman_last_input_vs_each_target_horizon",
        "fit_rows": [0, fit_stop - 1],
        "purge_origins": purge_origins,
        "top_k": top_k,
        "selected": selected,
        "scores": scores,
        "train_npz": {"path": str(train_npz), "sha256": _sha256(train_npz)},
        "grouping": {"schema": GROUPING_SCHEMA, "group_sizes": [len(group) for group in groups]},
    }
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ValueError("output directory already exists")
    destination.mkdir(parents=True)
    path = destination / "CANDIDATE_screened.json"
    _atomic(path, candidate)
    receipt = {"schema": SCHEMA, "status": "COMPLETE",
               "candidate": {"path": str(path), "sha256": _sha256(path)},
               "selected": len(selected), "excluded": len(excluded),
               "group_sizes": [len(group) for group in groups]}
    _atomic(destination / "MATERIALIZATION.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--purge-origins", type=int, default=36)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source, args.train_npz, args.output,
                                 top_k=args.top_k, group_count=args.groups,
                                 purge_origins=args.purge_origins), sort_keys=True))


if __name__ == "__main__":
    main()
