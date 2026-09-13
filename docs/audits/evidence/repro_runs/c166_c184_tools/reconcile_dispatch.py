#!/usr/bin/env python3
"""Reconcile a stopped dispatch with what physically ran, read-only on every host, then release finished units.

  reconcile_dispatch.py --dispatch <dispatch root> --unit-glob 'crispdm-dispatch-d2_fresh_c174_v1*' \
      --shards-out .local/state/crispdm-data-foundation/d2_fresh_c174_v1 --members <shards dir> \
      --out <write-once RECONCILIATION.json> [--release] [--cause '...']

Per role and per unit matching the glob: the unit's final systemd properties, its shard root on that role
(terminals by status, invalidation markers, members) and the dispatcher's own launch records. Units whose
dispatcher was stopped have no terminal receipt; this file is the durable record instead. A unit is released
only after it is recorded and only when inactive/failed/exited. Hosts by role only; no home paths.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

HOME = Path.home()
ROLES = json.loads((HOME / ".config/crispdm/host_roles.json").read_text())
PROPS = ("Id", "LoadState", "ActiveState", "SubState", "Result", "ExecMainCode", "ExecMainStatus", "MemoryPeak",
         "ActiveEnterTimestamp", "ActiveExitTimestamp")


def on(role: str, script: str) -> str:
    alias = ROLES[role]["ssh"]
    argv = ["bash", "-c", script] if alias is None else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                                         alias, script]
    return subprocess.run(argv, capture_output=True, text=True, timeout=300).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispatch", type=Path, required=True)
    ap.add_argument("--unit-glob", required=True)
    ap.add_argument("--shards-out", required=True, help="home-relative path of the <ROLE>/shard_NN roots")
    ap.add_argument("--members", type=Path, required=True, help="shards dir holding shard_NN/MEMBERS.txt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--cause", default="")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit("REFUSED: reconciliation is write-once")
    launches, recorded = {}, {}
    for p in sorted((a.dispatch / "receipts").glob("*.launch.json")):
        d = json.loads(p.read_text())
        launches.setdefault(d["job_id"], []).append({"attempt": d["attempt"], "role": d["role"], "unit": d["unit"]})
    for p in sorted((a.dispatch / "receipts").glob("*.attempt-*.json")):
        if p.name.endswith(".launch.json"):
            continue
        d = json.loads(p.read_text())
        recorded.setdefault(d["job_id"], []).append({"attempt": d.get("attempt"), "status": d["status"]})
    shards: dict = {}
    for role in ("WORKER_A", "WORKER_B", "COORDINATOR"):
        units = on(role, f'systemctl --user list-units --no-legend --plain --all "{a.unit_glob}" | awk "{{print \\$1}}"').split()
        for u in units:
            props = dict(ln.split("=", 1) for ln in on(role, f"systemctl --user show {u} " +
                                                         " ".join(f"-p {p}" for p in PROPS)).splitlines() if "=" in ln)
            m = re.search(r"(shard\d+)", u)
            shard = m.group(1).replace("shard", "shard_") if m else u
            kv = dict(ln.split("=", 1) for ln in on(role, (
                f'd=$HOME/{a.shards_out}/{role}/{shard}; '
                'echo "terminals=$(ls $d/terminals 2>/dev/null | wc -l)"; '
                'echo "statuses=$(cat $d/terminals/*.json 2>/dev/null | grep -o "\\"status\\": \\"[A-Z_]*\\"" | sort | uniq -c | tr -s " " | tr "\\n" ";")"; '
                'echo "invalidated=$(ls $d/ROOT_INVALIDATED__* 2>/dev/null | wc -l)"')).splitlines() if "=" in ln)
            members = len((a.members / shard / "MEMBERS.txt").read_text().split()) if (a.members / shard / "MEMBERS.txt").is_file() else None
            entry = {"role": role, "unit": u, "systemd": props, "unit_terminals": int(kv.get("terminals", 0)),
                     "unit_statuses": kv.get("statuses", ""), "invalidation_markers": int(kv.get("invalidated", 0)),
                     "members": members, "complete": members is not None and int(kv.get("terminals", 0)) == members}
            shards.setdefault(shard, []).append(entry)
            finished = props.get("ActiveState") in ("inactive", "failed") or props.get("SubState") == "exited"
            if a.release and finished and entry["complete"]:
                verb = "reset-failed" if props.get("ActiveState") == "failed" else "stop"
                entry["released_with"] = verb
                on(role, f"systemctl --user {verb} {u}")
    dup = {s: [e["role"] for e in es] for s, es in shards.items() if len(es) > 1}
    doc = {"schema": "crispdm.data_foundation.dispatch_reconciliation.v2", "dispatch_root": a.dispatch.name,
           "cause": a.cause, "dispatcher_launch_records": launches, "dispatcher_recorded_terminals": recorded,
           "physical_shards": shards, "shards_running_on_more_than_one_role": dup, "shards_found": len(shards),
           "all_complete": all(e["complete"] for es in shards.values() for e in es)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True).replace(str(HOME), "~") + "\n")
    print(json.dumps({"shards_found": len(shards), "duplicates": dup, "all_complete": doc["all_complete"],
                      "states": sorted({(e["systemd"].get("ActiveState"), e["systemd"].get("Result"),
                                         e["systemd"].get("ExecMainStatus")) for es in shards.values() for e in es})}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
