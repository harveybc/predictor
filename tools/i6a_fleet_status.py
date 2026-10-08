"""Read all I6-A collector status files in one command; no log tailing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def campaign_status(target, root):
    path = root / "STATUS.json"
    if not path.is_file():
        return {"target": target, "state": "NOT_STARTED"}
    try:
        status = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"target": target, "state": "INVALID_STATUS"}
    if status.get("target") != target:
        return {"target": target, "state": "INVALID_STATUS"}
    state = status.get("state")
    row = {"target": target, "state": state, "status_age_seconds": max(0, int(time.time() - path.stat().st_mtime))}
    if state == "WAITING":
        workers = status.get("workers", {})
        if not isinstance(workers, dict) or not workers:
            return {"target": target, "state": "INVALID_STATUS"}
        row["completed"] = sum(w.get("completed", 0) for w in workers.values())
        row["total"] = sum(w.get("total", 0) for w in workers.values())
        etas = [w.get("eta_seconds") for w in workers.values()]
        row["eta_seconds"] = max(etas) if all(type(eta) is int for eta in etas) else None
        row["workers"] = workers
    elif state == "PUBLISHED":
        closure_path = root / "CLOSURE.json"
        if not closure_path.is_file():
            return {"target": target, "state": "INVALID_STATUS", "reason": "CLOSURE_MISSING"}
        closure = json.loads(closure_path.read_text())
        if (closure.get("state") != "COMPLETE" or closure.get("sha256") != status.get("closure_sha256")
                or closure.get("verified_cells") != 208 or status.get("reports") != 208):
            return {"target": target, "state": "INVALID_STATUS", "reason": "CLOSURE_MISMATCH"}
        row["completed"] = 208
        row["total"] = 208
        row["eta_seconds"] = 0
        row["closure_sha256"] = closure["sha256"]
    else:
        row["reason"] = status.get("problems")
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", action="append", required=True,
                    help="TARGET=/absolute/path/to/collected/results")
    args = ap.parse_args(argv)
    rows = []
    for item in args.campaign:
        target, sep, path = item.partition("=")
        if not sep or not path.startswith("/"):
            raise ValueError("INVALID_CAMPAIGN_SPEC")
        rows.append(campaign_status(target, Path(path)))
    print(json.dumps({"schema": "i6a.fleet_status.v1", "campaigns": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
