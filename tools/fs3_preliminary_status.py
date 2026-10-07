#!/usr/bin/env python3
"""Count the pinned first wave from the coordinator task store, without writes."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


ARMS = ("RAW", "RANDOM_ENCODER", "TRAINED_ENCODER")
TERMINAL = {"COMPLETE", "NOT_AVAILABLE_FOR_TRAIN"}


def summarize(queue: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "fs3.preliminary_gpu_triage_bundle.v1":
        raise ValueError("INVALID_PRELIMINARY_MANIFEST")
    selected = {row["population_id"]: set(row["gpu_feature_ids"])
                for row in manifest["populations"]}
    counts = {arm: {state: 0 for state in ("COMPLETE", "NOT_AVAILABLE_FOR_TRAIN", "PENDING", "LEASED", "FAILED")}
              for arm in ARMS}
    by_population = {name: {arm: 0 for arm in ARMS} for name in selected}
    typed_refused = {arm: 0 for arm in ARMS}
    recent_gpu_durations = []
    now = time.time()
    with sqlite3.connect(f"file:{queue}?mode=ro", uri=True) as con:
        rows = con.execute("SELECT state,payload,result,started_at,finished_at FROM tasks").fetchall()
    for state, payload_json, result_json, started_at, finished_at in rows:
        payload = json.loads(payload_json)
        pop, arm, feature = payload["population_id"], payload["arm"], payload["feature_id"]
        if pop not in selected or feature not in selected[pop]:
            continue
        if arm not in counts or state not in counts[arm]:
            raise ValueError(f"UNRECOGNIZED_TASK_STATE:{state}:{arm}")
        counts[arm][state] += 1
        if (arm == "TRAINED_ENCODER" and state == "COMPLETE" and started_at and finished_at
                and finished_at >= now - 3600):
            recent_gpu_durations.append(finished_at - started_at)
        if state == "FAILED" and result_json:
            reason = str(json.loads(result_json).get("reason", ""))
            if reason.startswith(("NO_TRAIN_OBSERVATIONS", "INSUFFICIENT_TRAIN_")):
                typed_refused[arm] += 1
        by_population[pop][arm] += 1
    expected = sum(len(features) for features in selected.values()) * 5
    if any(sum(counts[arm].values()) != expected for arm in ARMS):
        raise ValueError("FIRST_WAVE_TASK_DENOMINATOR_MISMATCH")
    trained_done = sum(counts["TRAINED_ENCODER"][state] for state in TERMINAL) + typed_refused["TRAINED_ENCODER"]
    eta = None
    if len(recent_gpu_durations) >= 5:
        eta = round((expected - trained_done) * (statistics.median(recent_gpu_durations) + 120) / 2)
    return {"schema": "fs3.preliminary_wave_status.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path), "queue": str(queue),
            "features": {pop: len(features) for pop, features in selected.items()},
            "tasks_per_arm": expected, "counts": counts,
            "done_per_arm": {arm: sum(counts[arm][state] for state in TERMINAL) + typed_refused[arm]
                             for arm in ARMS},
            "typed_refused_per_arm": typed_refused,
            "gpu_eta_seconds": eta,
            "gpu_eta_basis": "last-hour median task duration + 120s timer, two uninterrupted GPU slots",
            "by_population_tasks": by_population, "final_selection": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.queue, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
