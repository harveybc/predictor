#!/usr/bin/env python3
"""Run resumable I6-B branch-only pretraining from a sealed TRAIN NPZ."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tools.modular_pretrain import Heartbeat, pretrain_from_train_npz


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-updates", type=int, default=20000)
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    parser.add_argument("--branch-indices", default=None,
                        help="comma-separated disjoint branch indices for a fleet shard")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    config = json.loads(Path(args.config).read_text())
    if args.branch_indices:
        try:
            indices = [int(value) for value in args.branch_indices.split(",")]
        except ValueError as exc:
            raise ValueError("branch indices must be comma-separated integers") from exc
        if not indices or len(indices) != len(set(indices)) or min(indices) < 0 \
                or max(indices) >= len(config["branches"]):
            raise ValueError("branch indices are empty, duplicated, negative or out of range")
        full = copy.deepcopy(config)
        config["branches"] = [full["branches"][index] for index in indices]
    fit = {"max_epochs": args.max_epochs, "patience": args.patience, "min_delta": 1e-4,
           "monitor_every": 1, "max_updates": args.max_updates, "max_seconds": args.max_seconds,
           "batch_size": args.batch_size, "learning_rate": args.learning_rate, "loss": "mse"}
    output.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output.with_name(output.name + ".heartbeat.jsonl")
    with Heartbeat(heartbeat_path, 30.0) as heartbeat:
        result = pretrain_from_train_npz(
            args.train_npz, output, fit, provenance="local_file", config=config,
            seed=args.seed, heartbeat=heartbeat, resume=args.resume,
            stop_after_branches=True,
        )
    print(json.dumps({"status": result["status"], "branches": len(result["branches"]),
                      "branch_names": [row["name"] for row in result["branches"]],
                      "resumed": len(result["resumed_branches"]),
                      "next_stage": result["next_stage"]}))


if __name__ == "__main__":
    main()
