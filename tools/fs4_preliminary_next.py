#!/usr/bin/env python3
"""Choose one Phase-4 GPU task from a pinned Phase-3 preliminary wave."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path


def choose_task(tasks: list[dict], manifest: dict) -> str | None:
    if manifest.get("schema") != "fs3.preliminary_gpu_triage_bundle.v1" or manifest.get("final_selection") is not False:
        raise ValueError("INVALID_PRELIMINARY_MANIFEST")
    by_population = {row["population_id"]: set(row["gpu_feature_ids"])
                     for row in manifest["populations"]}
    if not by_population or len(by_population) != len(manifest["populations"]):
        raise ValueError("INVALID_PRELIMINARY_POPULATIONS")
    candidates = [task for task in tasks
                  if task["state"] == "PENDING" and task["arm"] == "TRAINED_ENCODER"
                  and task["population_id"] in by_population
                  and task["feature_id"] in by_population[task["population_id"]]]
    candidates.sort(key=lambda task: (task["population_id"] != "EURUSD", task["feature_id"],
                                      task["fold_id"], task["task_id"]))
    return candidates[0]["task_id"] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--db", required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.sha256:
        raise ValueError("PRELIMINARY_MANIFEST_DIGEST_MISMATCH")
    manifest = json.loads(raw)
    argv = [args.python, args.controller, "--db", args.db, "list", "--arm", "TRAINED_ENCODER"]
    remote = " ".join(shlex.quote(word) for word in argv)
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                             args.coordinator, remote], capture_output=True, text=True,
                            timeout=90, check=True)
    tasks = json.loads(result.stdout)
    task_id = choose_task(tasks, manifest)
    if task_id is not None:
        print(task_id)


if __name__ == "__main__":
    main()
