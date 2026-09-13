#!/usr/bin/env python3
"""Reconcile the first C172 reanalysis dispatch with what physically ran.

The dispatcher (code e1ae37b) recorded LAUNCH_NOT_FOUND for shard units that were in fact running under a unit name
derived from the role-bound argv. This reconciliation reads, per role and read-only:
  * each shard's systemd user unit (bound-name), its final ActiveState/Result/ExecMainStatus/MemoryPeak;
  * each shard root on that role (RUN_MANIFEST, terminals per unit, invalidation markers);
and writes a write-once RECONCILIATION.json mapping shard -> role -> unit -> final state -> unit terminals by status,
plus the dispatcher's recorded (wrong) receipts, so the evidence chain is physical, never inferred from receipts.
It releases (stops) each finished unit only after recording it. Hosts by role only; no home paths.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
S = HOME / ".local/state/crispdm-data-foundation"
ROLES = json.loads((HOME / ".config/crispdm/host_roles.json").read_text())
DISPATCH = S / "d2_dispatch_c172_v1"
SHARD_OUT = ".local/state/crispdm-data-foundation/d2_reanalysis_c172_v1"
PROPS = ("Id", "LoadState", "ActiveState", "SubState", "Result", "ExecMainCode", "ExecMainStatus", "MemoryPeak",
         "ActiveEnterTimestamp", "ActiveExitTimestamp")


def on(role: str, script: str) -> str:
    alias = ROLES[role]["ssh"]
    argv = ["bash", "-c", script] if alias is None else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                                         alias, script]
    return subprocess.run(argv, capture_output=True, text=True, timeout=300).stdout


def main(out: Path, release: bool) -> int:
    if out.exists():
        raise SystemExit("REFUSED: reconciliation is write-once")
    launches = {}
    for p in sorted((DISPATCH / "receipts").glob("*.launch.json")):
        d = json.loads(p.read_text())
        bound = re.search(r"--unit=(\S+)", d["command"]).group(1)
        launches.setdefault(d["job_id"], []).append({"attempt": d["attempt"], "role": d["role"],
                                                     "recorded_unit": d["unit"], "launched_unit": bound})
    recorded_terminals = {}
    for p in sorted((DISPATCH / "receipts").glob("*.attempt-*.json")):
        if p.name.endswith(".launch.json"):
            continue
        d = json.loads(p.read_text())
        recorded_terminals.setdefault(d["job_id"], []).append({"attempt": d.get("attempt"), "status": d["status"]})
    shards = {}
    for role in ("WORKER_A", "WORKER_B", "COORDINATOR"):
        units = on(role, 'systemctl --user list-units --no-legend --plain --all "crispdm-dispatch-d2_reanalysis*" '
                         '| awk "{print \\$1}"').split()
        for u in units:
            props = dict(ln.split("=", 1) for ln in on(role, f"systemctl --user show {u} " +
                                                         " ".join(f"-p {p}" for p in PROPS)).splitlines() if "=" in ln)
            m = re.search(r"(shard\d+)", u)
            shard = m.group(1).replace("shard", "shard_") if m else u
            root_summary = on(role, (
                f'd=$HOME/{SHARD_OUT}/{role}/{shard}; '
                'echo "terminals=$(ls $d/terminals 2>/dev/null | wc -l)"; '
                'echo "statuses=$(cat $d/terminals/*.json 2>/dev/null | grep -o "\\"status\\": \\"[A-Z_]*\\"" | sort | uniq -c | tr -s " " | tr "\\n" ";")"; '
                'echo "invalidated=$(ls $d/ROOT_INVALIDATED__* 2>/dev/null | wc -l)"; '
                'echo "members=$(grep -c . $HOME/.local/state/crispdm-data-foundation/d2_shards_c172_v1/' + shard + '/MEMBERS.txt)"'))
            kv = dict(ln.split("=", 1) for ln in root_summary.splitlines() if "=" in ln)
            entry = {"role": role, "unit": u, "systemd": props, "unit_terminals": int(kv.get("terminals", 0)),
                     "unit_statuses": kv.get("statuses", ""), "invalidation_markers": int(kv.get("invalidated", 0)),
                     "members": int(kv.get("members", 0))}
            shards.setdefault(shard, []).append(entry)
            if release and props.get("ActiveState") in ("inactive", "failed") or (release and props.get("SubState") == "exited"):
                verb = "reset-failed" if props.get("ActiveState") == "failed" else "stop"
                entry["released_with"] = verb
                on(role, f"systemctl --user {verb} {u}")
    dup = {s: [e["role"] for e in es] for s, es in shards.items() if len(es) > 1}
    doc = {"schema": "crispdm.data_foundation.c172_dispatch_reconciliation.v1",
           "cause": "dispatcher e1ae37b launched units under a name derived from the role-bound argv; it polled the "
                    "template name, recorded LAUNCH_NOT_FOUND and requeued while units were running (fixed in 734bd55)",
           "dispatcher_launch_records": launches, "dispatcher_recorded_terminals": recorded_terminals,
           "physical_shards": shards, "shards_running_on_more_than_one_role": dup,
           "shards_found": len(shards)}
    text = json.dumps(doc, indent=1, sort_keys=True).replace(str(HOME), "~")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(json.dumps({"shards_found": len(shards), "duplicates": dup,
                      "states": sorted({(e["systemd"].get("ActiveState"), e["systemd"].get("Result"),
                                         e["systemd"].get("ExecMainStatus")) for es in shards.values() for e in es})}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), "--release" in sys.argv))
