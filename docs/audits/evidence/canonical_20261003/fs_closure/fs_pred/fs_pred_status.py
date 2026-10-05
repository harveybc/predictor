#!/usr/bin/env python3
"""Render FS-PRED lane status from worker_b evidence (read-only; never touches STATUS.json).

Usage: fs_pred_status.py <local copy of out/> <dest dir>
Writes LANE_STATUS.json, method_walls.csv and primary_k24_sets.csv into <dest dir>.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


def watchdog_state(progress: dict, progress_age_seconds: float, probe: dict | None, blocked: bool) -> dict:
    """Decide RUNNING / STALLED / DONE / BLOCKED / IDLE from evidence, never from a live PID.

    STALLED when progress.json has not changed for max(10 min, 3 x p90 method wall) AND the worker
    process is under memory pressure (any D-state process in its scope, cgroup memory.events high
    count rising, or cgroup memory pressure); a live PID alone never proves progress.
    """

    done, total = int(progress.get("done") or 0), int(progress.get("total") or 0)
    if blocked:
        return {"state": "BLOCKED", "reason": "BLOCKED_CAP_UNMEASURED present"}
    if total and done >= total:
        return {"state": "DONE", "reason": "done == total"}
    p90 = float(progress.get("observed_p90_method_wall_seconds") or 0.0)
    threshold = max(600.0, 3.0 * p90)
    probe = probe or {}
    pressure = bool(probe.get("d_state_pids")) or bool(probe.get("high_events_rising")) or (
        float(probe.get("cgroup_some_avg10") or 0.0) > 0.0
    )
    stale = progress_age_seconds > threshold
    if stale and pressure:
        return {"state": "STALLED", "reason": f"progress unchanged for {progress_age_seconds:.0f}s > {threshold:.0f}s and the worker scope is under memory pressure", "threshold_seconds": threshold}
    if not probe.get("scope_active"):
        return {"state": "IDLE", "reason": "no FS-PRED scope active on the worker", "threshold_seconds": threshold}
    if stale:
        return {"state": "SUSPECT", "reason": f"progress unchanged for {progress_age_seconds:.0f}s > {threshold:.0f}s without observed memory pressure", "threshold_seconds": threshold}
    return {"state": "RUNNING", "reason": "progress advanced within the threshold and a scope is active", "threshold_seconds": threshold}


def eta_after_post_cells(progress: dict, post_walls: list[float], relaunch_epoch: float | None) -> dict:
    """ETA is recomputed only from cells terminal AFTER the relaunch; fewer than three -> no ETA."""

    remaining = int(progress.get("method_cells_total") or 0) - int(progress.get("method_cells_done") or 0)
    if relaunch_epoch is None:
        return {"eta_utc": progress.get("eta_utc"), "basis": "runner estimate (no relaunch epoch)"}
    if len(post_walls) < 3:
        return {"eta_utc": None, "basis": f"{len(post_walls)} POST cells; ETA withheld until three", "post_cells": len(post_walls)}
    median = statistics.median(post_walls)
    eta = datetime.now(timezone.utc).timestamp() + remaining * median
    return {"eta_utc": datetime.fromtimestamp(eta, timezone.utc).isoformat(), "basis": f"median of {len(post_walls)} POST method walls x {remaining} remaining method cells, one worker", "post_cells": len(post_walls), "post_median_wall_seconds": median}


def main(out_dir: str, dest_dir: str) -> int:
    out, dest = Path(out_dir), Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    progress = json.loads((out / "progress.json").read_text()) if (out / "progress.json").exists() else {}
    contract = json.loads((out / "run_contract.json").read_text()) if (out / "run_contract.json").exists() else {}
    walls: dict[str, list[float]] = {}
    rows, sets = [], []
    for record_path in sorted(out.glob("cells/*/*/*.json")):
        if record_path.name == "population.json":
            continue
        record = json.loads(record_path.read_text())
        rows.append({
            "method": record["method"], "target": record["target"], "fold": record["fold"],
            "disposition": record["disposition"], "wall_seconds": round(float(record["wall_seconds"]), 2),
            "implementation": record.get("implementation", ""), "rows": record["population"]["rows"],
            "row_digest": record["identity"]["row_digest"][:16], "plan_digest": record["identity"]["plan_digest"][:16],
            "k_failures": len(record.get("k_failures", [])), "failure": record.get("failure_message") or "",
        })
        if record["disposition"] == "COMPLETE":
            walls.setdefault(record["method"], []).append(float(record["wall_seconds"]))
            for k, selection in record.get("selected_by_k", []):
                if int(k) == 24:
                    sets.append({"method": record["method"], "target": record["target"], "fold": record["fold"],
                                 "K": 24, "selected": "|".join(selection)})
    with (dest / "method_walls.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["method"])
        writer.writeheader(); writer.writerows(rows)
    with (dest / "primary_k24_sets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "target", "fold", "K", "selected"])
        writer.writeheader(); writer.writerows(sets)
    probe = json.loads((out.parent / "worker_probe.json").read_text()) if (out.parent / "worker_probe.json").exists() else None
    blocked = (out.parent / "BLOCKED_CAP_UNMEASURED").exists() or (out / "BLOCKED_CAP_UNMEASURED").exists()
    progress_age = (datetime.now(timezone.utc).timestamp() - (out / "progress.json").stat().st_mtime) if (out / "progress.json").exists() else 0.0
    if probe and probe.get("progress_mtime_epoch"):
        progress_age = float(probe.get("probe_epoch", datetime.now(timezone.utc).timestamp())) - float(probe["progress_mtime_epoch"])
    relaunch_epoch = float(probe["relaunch_epoch"]) if probe and probe.get("relaunch_epoch") else None
    post_walls = []
    if relaunch_epoch is not None:
        for record_path in out.glob("cells/*/*/*.json"):
            if record_path.name == "population.json":
                continue
            try:
                record = json.loads(record_path.read_text())
            except (OSError, ValueError):
                continue
            written = record.get("written_utc")
            if record.get("disposition") == "COMPLETE" and written and datetime.fromisoformat(written).timestamp() >= relaunch_epoch:
                post_walls.append(float(record["wall_seconds"]))
    watchdog = watchdog_state(progress, progress_age, probe, blocked)
    eta = eta_after_post_cells(progress, post_walls, relaunch_epoch)
    status = {
        "schema": "fs_pred_lane_status.v2",
        "state": watchdog["state"],
        "watchdog": dict(watchdog, progress_age_seconds=round(progress_age, 1), probe=probe),
        "eta": eta,
        "rendered_utc": datetime.now(timezone.utc).isoformat(),
        "host_role": "worker_b (CPU)",
        "progress": {k: progress.get(k) for k in ("done", "total", "method_cells_done", "method_cells_total", "n_failures",
                                                   "observed_median_method_wall_seconds", "observed_p90_method_wall_seconds",
                                                   "eta_seconds", "eta_utc", "updated_utc", "workers")},
        "per_method": progress.get("per_method"),
        "per_target": progress.get("per_target"),
        "failures": progress.get("failures"),
        "method_wall_seconds": {m: {"n": len(v), "median": statistics.median(v), "max": max(v)} for m, v in sorted(walls.items())},
        "contract": {k: contract.get(k) for k in ("denominator", "n_features", "train_rows", "train_end_utc_exclusive", "k_primary",
                                                   "sealed_ks", "seed", "methods", "chronoepilogi_implementation", "input_sha256")},
        "records": len(rows),
        "k24_sets": len(sets),
    }
    (dest / "LANE_STATUS.json").write_text(json.dumps(status, indent=1, sort_keys=True, default=str))
    print(json.dumps(status["progress"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
