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
    status = {
        "schema": "fs_pred_lane_status.v1",
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
