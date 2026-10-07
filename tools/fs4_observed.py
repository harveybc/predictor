#!/usr/bin/env python3
"""FS4 observed status: the closure follower's STATUS.json plus read-only checks, nothing narrated.

Every field is measured when the command runs:
  * the closure STATUS.json (itself generated from the controller's task store);
  * per worker role: can the worker reach the controller THROUGH ITS OWN SSH GATE (the worker runs
    ``fs4_worker.py --status``, which is the exact path claim/complete/fail use), and which
    ``fs4-worker@<slot>.timer`` units are enabled there;
  * the warehouse, from the closure's own last submit/readback pass and the age of that pass.

``credential_items`` is empty unless one of those observations FAILED; an item then names the check, the
role and the observed error text. No credential is read, requested or handled here: SSH is used with the
operator's existing keys, and the warehouse is not contacted at all (the closure follower is the only
client that holds its token).

    fs4_observed.py --state-root <closure dir> --hosts-env <file with WORKER_A_SSH/WORKER_B_SSH> [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "fs4.status_observed.v1"
MIN_COMPLETE_FOR_ETA = 5
STALE_SECONDS = 600
WORKER_STATUS_SCRIPT = r"""
f=$(ls "$HOME"/.config/fs4/*.env 2>/dev/null | grep -v AUTHORIZE | head -1)
[ -n "$f" ] || { echo "NO_SLOT_ENV_FILE" >&2; exit 3; }
set -a; . "$f"; set +a
"$FS4_PYTHON" "$FS4_CODE/tools/fs4_worker.py" --coordinator "$FS4_COORDINATOR" --controller "$FS4_CONTROLLER" \
  --python "${FS4_COORDINATOR_PYTHON:-python3}" --db "$FS4_DB" --owner "$FS4_OWNER" --runner "$FS4_RUNNER" \
  --cap "$FS4_CAP" --output-root "$FS4_OUTPUT_ROOT" --status
echo "---TIMERS---"
systemctl --user list-timers 'fs4-worker@*' --no-legend --no-pager 2>/dev/null | grep -o 'fs4-worker@[^.]*' | sort -u
"""


def read_hosts(path: Path) -> dict[str, str]:
    aliases = {}
    for line in Path(path).read_text().splitlines():
        match = re.match(r"^(WORKER_[AB])_SSH=(.+)$", line.strip())
        if match:
            aliases["worker_" + match.group(1)[-1].lower()] = match.group(2).strip().strip("'\"")
    return aliases


def ssh_run(alias: str, script: str, timeout: int = 90) -> tuple[int, str, str]:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", alias, "bash", "-s"],
                            input=script, text=True, capture_output=True, timeout=timeout, check=False)
    return result.returncode, result.stdout, result.stderr


def observe_worker(role: str, alias: str, runner=ssh_run) -> dict:
    try:
        rc, out, err = runner(alias, WORKER_STATUS_SCRIPT)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"role": role, "gate_reachable": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    head, _, tail = out.partition("---TIMERS---")
    try:
        queue = json.loads(head.strip())
        ok = rc == 0 and queue.get("schema") == "fs4.status.v1"
    except json.JSONDecodeError:
        queue, ok = None, False
    record = {"role": role, "gate_reachable": ok}
    if ok:
        record["queue_total"] = queue["total"]
        record["queue_complete"] = queue["complete"]
        record["timers"] = sorted(t for t in tail.split() if t.startswith("fs4-worker@"))
    else:
        record["error"] = (err.strip() or head.strip() or f"rc={rc}")[-300:]
    return record


def observe_warehouse(status: dict | None, now: float) -> dict:
    if not status:
        return {"healthy": False, "error": "closure STATUS.json is missing or unreadable"}
    warehouse = status.get("warehouse") or {}
    age = now - float(status.get("generated_epoch") or 0)
    last = warehouse.get("last_pass")
    problems = []
    if warehouse.get("error"):
        problems.append(f"closure reports: {warehouse['error']}")
    if not last:
        problems.append("no submit/readback pass recorded")
    elif last.get("errors"):
        problems.append(f"last pass errors: {last['errors'][:2]}")
    if age > STALE_SECONDS:
        problems.append(f"closure STATUS is {int(age)} s old (follower not running)")
    return {"healthy": not problems, "status_age_seconds": round(age), "last_pass": last,
            "verified_receipts": warehouse.get("verified_receipts"), "pending_submit": warehouse.get("pending_submit"),
            "quarantined": warehouse.get("quarantined"), "error": "; ".join(problems) or None}


def build(status: dict | None, workers: list[dict], now: float) -> dict:
    warehouse = observe_warehouse(status, now)
    items = []
    for worker in workers:
        if not worker["gate_reachable"]:
            items.append({"item": "worker_ssh_gate", "role": worker["role"], "observed_failure": worker.get("error")})
    if not warehouse["healthy"]:
        items.append({"item": "warehouse_readback", "observed_failure": warehouse["error"]})
    complete = (status or {}).get("complete", 0)
    if status and complete >= MIN_COMPLETE_FOR_ETA:
        eta = {"seconds": status.get("eta_seconds"), "basis": status.get("eta_reason"), "completed_tasks": complete}
    else:
        eta = {"seconds": None, "basis": f"fewer than {MIN_COMPLETE_FOR_ETA} completed tasks", "completed_tasks": complete}
    return {"schema": SCHEMA, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "generated_epoch": now, "closure_status": status, "extractibility_eta": eta,
            "observed": {"workers": workers, "warehouse": warehouse}, "credential_items": items}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--hosts-env", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    now = time.time()
    try:
        status = json.loads((args.state_root / "STATUS.json").read_text())
    except (OSError, json.JSONDecodeError):
        status = None
    workers = [observe_worker(role, alias) for role, alias in sorted(read_hosts(args.hosts_env).items())]
    document = build(status, workers, now)
    text = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.out:
        temp = args.out.with_name(args.out.name + ".partial")
        temp.write_text(text)
        temp.replace(args.out)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
