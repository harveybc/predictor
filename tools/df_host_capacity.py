#!/usr/bin/env python3
"""C174 (order 2026-09-13): live, read-only capacity inventory of every host role.

Hosts are named by role only (COORDINATOR, WORKER_A, WORKER_B). The map from
role to ssh alias lives outside Git (default ~/.config/crispdm/host_roles.json;
a null alias means this machine). No alias, host name or address appears in an
output: probe errors are redacted before they are recorded.

One probe per role (one ssh call for a worker, a local bash for the
COORDINATOR) prints marked sections; every section is parsed by a pure function
that tests feed with captured text:

* cpus (nproc --all) and load average;
* /proc/meminfo: MemTotal, MemAvailable, SwapTotal, SwapFree;
* crispdm-batch.slice: MemoryMax, MemoryHigh, memory.current, and each running
  crispdm-*.scope in it with memory.current and memory.peak;
* crispdm-memguard.service state and whether crispdm-run is installed;
* GPUs: `nvidia-smi --query-gpu` (index, name, uuid, pci bus, total/free/used
  VRAM) and `--query-compute-apps` (processes per GPU). A GPU whose handle
  cannot be obtained ("Unable to determine the device handle ...") or that a
  declared quarantine entry names is QUARANTINED_NOT_SCHEDULABLE.

Declared quarantine: DEFAULT_QUARANTINE below, plus entries from an optional
private file (default ~/.config/crispdm/gpu_quarantine.json, a JSON list of
{"role", "gpu_index" | "pci_bus_id" | "uuid", "reason"}). A quarantine entry
whose GPU is not even enumerated is still reported, as a quarantined device
without a handle.

Nothing is started, stopped or written on any host. `--dry-run` prints the
inventory; `--out FILE` also writes it write-once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "crispdm.data_foundation.host_capacity.v1"
ROLES = ("COORDINATOR", "WORKER_A", "WORKER_B")
DEFAULT_ROLES = Path.home() / ".config/crispdm/host_roles.json"
DEFAULT_QUARANTINE_FILE = Path.home() / ".config/crispdm/gpu_quarantine.json"
MIB = 1 << 20
QUARANTINED = "QUARANTINED_NOT_SCHEDULABLE"
SCHEDULABLE = "SCHEDULABLE"
# The auditor's standing quarantine: WORKER_B's second GPU has no NVML handle and is never schedulable.
DEFAULT_QUARANTINE = ({"role": "WORKER_B", "gpu_index": 1,
                       "reason": "no NVML device handle; quarantined by the auditor, never schedulable"},)
PROBE_TIMEOUT_SECONDS = 60

# Runs identically on every host (bash, read only). Sections are delimited by '@@name' lines.
PROBE = r"""
set -u
echo "@@cpus"; nproc --all
echo "@@loadavg"; cat /proc/loadavg
echo "@@meminfo"; grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
echo "@@slice"; systemctl --user show crispdm-batch.slice -p MemoryMax -p MemoryHigh -p ControlGroup -p ActiveState 2>&1
CG=$(systemctl --user show crispdm-batch.slice -p ControlGroup --value 2>/dev/null)
echo "@@slice_cgroup"
if [ -n "$CG" ] && [ -d "/sys/fs/cgroup$CG" ]; then
  echo "memory.current $(cat /sys/fs/cgroup$CG/memory.current 2>/dev/null)"
  echo "memory.max $(cat /sys/fs/cgroup$CG/memory.max 2>/dev/null)"
fi
echo "@@scopes"
if [ -n "$CG" ] && [ -d "/sys/fs/cgroup$CG" ]; then
  for d in /sys/fs/cgroup$CG/*.scope; do
    [ -d "$d" ] || continue
    echo "$(basename "$d") $(cat "$d/memory.current" 2>/dev/null || echo -1) $(cat "$d/memory.peak" 2>/dev/null || echo -1)"
  done
fi
echo "@@memguard"; systemctl --user is-active crispdm-memguard.service 2>&1
echo "@@crispdm_run"; [ -x "$HOME/.local/bin/crispdm-run" ] && echo PRESENT || echo MISSING
echo "@@gpus"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.free,memory.used --format=csv,noheader,nounits 2>&1
  echo "rc=$?"
else
  echo "NO_NVIDIA_SMI"
fi
echo "@@apps"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>&1
fi
echo "@@end"
"""

HANDLE_ERROR_RE = re.compile(r"Unable to determine the device handle for (?:GPU|gpu)\s*(\d+)?:?\s*([0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f])?\s*:?\s*(.*)")
IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


# ------------------------------------------------------------------ parsing
def split_sections(text: str) -> dict:
    out, cur = {}, None
    for line in text.splitlines():
        if line.startswith("@@"):
            cur = line[2:].strip()
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def parse_meminfo(text: str) -> dict:
    """/proc/meminfo lines -> bytes, keyed mem_total/mem_available/swap_total/swap_free."""
    keys = {"MemTotal": "mem_total_bytes", "MemAvailable": "mem_available_bytes", "SwapTotal": "swap_total_bytes",
            "SwapFree": "swap_free_bytes"}
    out = {v: None for v in keys.values()}
    for line in text.splitlines():
        m = re.match(r"^(\w+):\s+(\d+)\s*(kB)?", line.strip())
        if m and m.group(1) in keys:
            out[keys[m.group(1)]] = int(m.group(2)) * (1024 if m.group(3) else 1)
    return out


def parse_loadavg(text: str) -> dict:
    parts = text.split()
    try:
        return {"load1": float(parts[0]), "load5": float(parts[1]), "load15": float(parts[2])}
    except (IndexError, ValueError):
        return {"load1": None, "load5": None, "load15": None}


def _bytes_or_none(v: str):
    v = (v or "").strip()
    if v in ("", "infinity", "max", "[not set]"):
        return None
    try:
        return int(v)
    except ValueError:
        return None


def parse_slice(show_text: str, cgroup_text: str) -> dict:
    """`systemctl show` of the batch slice plus its cgroup files. Unlimited values are None."""
    props = dict(line.split("=", 1) for line in show_text.splitlines() if "=" in line)
    cg = dict(line.split(" ", 1) for line in cgroup_text.splitlines() if " " in line)
    max_ = _bytes_or_none(props.get("MemoryMax", ""))
    if max_ is None:
        max_ = _bytes_or_none(cg.get("memory.max", ""))
    return {"present": bool(props.get("ControlGroup")), "active_state": props.get("ActiveState"),
            "memory_max_bytes": max_, "memory_high_bytes": _bytes_or_none(props.get("MemoryHigh", "")),
            "memory_current_bytes": _bytes_or_none(cg.get("memory.current", ""))}


def parse_scopes(text: str) -> list:
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) == 3 and p[0].endswith(".scope"):
            cur, peak = int(p[1]), int(p[2])
            out.append({"unit": p[0], "memory_current_bytes": cur if cur >= 0 else None,
                        "memory_peak_bytes": peak if peak >= 0 else None})
    return out


def parse_nvidia_gpus(text: str) -> tuple:
    """-> (gpus, handle_errors). `text` is `nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,
    memory.free,memory.used --format=csv,noheader,nounits` with stderr merged (MiB)."""
    gpus, errors = [], []
    if "NO_NVIDIA_SMI" in text:
        return gpus, errors
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("rc="):
            continue
        m = HANDLE_ERROR_RE.search(line)
        if m:
            errors.append({"gpu_index": int(m.group(1)) if m.group(1) is not None else None,
                           "pci_bus_id": m.group(2), "error": line[:200]})
            continue
        p = [x.strip() for x in line.split(",")]
        if len(p) != 7:
            errors.append({"gpu_index": None, "pci_bus_id": None, "error": line[:200]})
            continue
        try:
            total, free, used = (int(float(x)) * MIB for x in p[4:7])
            gpus.append({"index": int(p[0]), "name": p[1], "uuid": p[2], "pci_bus_id": p[3],
                         "vram_total_bytes": total, "vram_free_bytes": free, "vram_used_bytes": used})
        except ValueError:
            errors.append({"gpu_index": None, "pci_bus_id": None, "error": line[:200]})
    # the same handle error is printed by both queries; keep one per device
    seen, uniq = set(), []
    for e in errors:
        k = (e["gpu_index"], e["pci_bus_id"], e["error"] if e["gpu_index"] is None and e["pci_bus_id"] is None else "")
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return gpus, uniq


def parse_compute_apps(text: str) -> list:
    out = []
    for line in text.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 3 and p[0].startswith("GPU-"):
            try:
                out.append({"gpu_uuid": p[0], "pid": int(p[1]), "used_bytes": int(float(p[2])) * MIB})
            except ValueError:
                continue
    return out


def _quarantine_match(q: dict, role: str, gpu: dict) -> bool:
    if q.get("role") != role:
        return False
    if "uuid" in q and q["uuid"] == gpu.get("uuid"):
        return True
    if "pci_bus_id" in q and gpu.get("pci_bus_id") and q["pci_bus_id"].lower() == gpu["pci_bus_id"].lower():
        return True
    return "gpu_index" in q and q["gpu_index"] == gpu.get("index")


def build_role_inventory(role: str, probe_text: str, quarantine=DEFAULT_QUARANTINE) -> dict:
    """Pure: one role's raw probe output -> its inventory record."""
    s = split_sections(probe_text)
    if "end" not in s:
        return {"role": role, "reachable": False, "error": "PROBE_OUTPUT_INCOMPLETE"}
    mem = parse_meminfo(s.get("meminfo", ""))
    try:
        cpus = int(s.get("cpus", "").strip())
    except ValueError:
        cpus = None
    gpus, errors = parse_nvidia_gpus(s.get("gpus", ""))
    apps = parse_compute_apps(s.get("apps", ""))
    q_role = [q for q in quarantine if q.get("role") == role]
    out_gpus = []
    for g in gpus:
        procs = [a for a in apps if a["gpu_uuid"] == g["uuid"]]
        hit = [q for q in q_role if _quarantine_match(q, role, g)]
        g = dict(g, compute_processes=len(procs), compute_used_bytes=sum(a["used_bytes"] for a in procs),
                 handle_error=None, status=QUARANTINED if hit else SCHEDULABLE,
                 quarantine_reason=hit[0].get("reason", "DECLARED_QUARANTINE") if hit else None)
        out_gpus.append(g)
    for e in errors:
        hit = [q for q in q_role if _quarantine_match(q, role, {"index": e["gpu_index"], "pci_bus_id": e["pci_bus_id"]})]
        out_gpus.append({"index": e["gpu_index"], "name": None, "uuid": None, "pci_bus_id": e["pci_bus_id"],
                         "vram_total_bytes": None, "vram_free_bytes": None, "vram_used_bytes": None,
                         "compute_processes": None, "compute_used_bytes": None, "handle_error": e["error"],
                         "status": QUARANTINED,
                         "quarantine_reason": (hit[0].get("reason") + "; " if hit else "") + "NO_DEVICE_HANDLE"})
    listed = {g["index"] for g in out_gpus if g["index"] is not None}
    for q in q_role:
        if "gpu_index" in q and q["gpu_index"] not in listed:
            out_gpus.append({"index": q["gpu_index"], "name": None, "uuid": None, "pci_bus_id": None,
                             "vram_total_bytes": None, "vram_free_bytes": None, "vram_used_bytes": None,
                             "compute_processes": None, "compute_used_bytes": None,
                             "handle_error": "NOT_ENUMERATED", "status": QUARANTINED,
                             "quarantine_reason": q.get("reason", "DECLARED_QUARANTINE")})
    out_gpus.sort(key=lambda g: (g["index"] is None, g["index"] if g["index"] is not None else 0))
    return {"role": role, "reachable": True, "error": None, "cpus": cpus, **parse_loadavg(s.get("loadavg", "")),
            **mem, "batch_slice": parse_slice(s.get("slice", ""), s.get("slice_cgroup", "")),
            "batch_scopes": parse_scopes(s.get("scopes", "")),
            "memguard_active": s.get("memguard", "").strip() == "active",
            "crispdm_run_present": s.get("crispdm_run", "").strip() == "PRESENT",
            "gpus": out_gpus}


# ------------------------------------------------------------------- probing
def redact(text: str, secrets=()) -> str:
    text = text.replace(str(Path.home()), "~")
    for sec in secrets:
        if sec:
            text = text.replace(sec, "<host>")
    return IPV4_RE.sub("<address>", text)


def run_probe(alias, runner=subprocess.run, timeout: float = PROBE_TIMEOUT_SECONDS) -> tuple:
    """-> (returncode, stdout, stderr). alias None = this host."""
    if alias:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", alias, "bash -s"]
    else:
        cmd = ["bash", "-s"]
    try:
        r = runner(cmd, input=PROBE, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "PROBE_TIMEOUT"
    except OSError as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"


def load_quarantine(path: Path | None) -> list:
    entries = [dict(q) for q in DEFAULT_QUARANTINE]
    if path is not None and Path(path).is_file():
        doc = json.loads(Path(path).read_text())
        if not isinstance(doc, list):
            raise SystemExit("REFUSED: the GPU quarantine file must be a JSON list")
        entries += [q for q in doc if isinstance(q, dict) and q.get("role") in ROLES]
    return entries


def inventory_digest(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_inventory(roles_map: dict, quarantine=DEFAULT_QUARANTINE, runner=subprocess.run, only=None) -> dict:
    """Live inventory of every role in `roles_map` ({role: {"ssh": alias|null}}). Read only."""
    roles = {}
    for role in ROLES:
        if role not in roles_map or (only and role not in only):
            continue
        alias = roles_map[role].get("ssh") if role != "COORDINATOR" else None
        rc, out, err = run_probe(alias, runner)
        if rc != 0 and "@@end" not in out:
            roles[role] = {"role": role, "reachable": False,
                           "error": redact(f"PROBE_FAILED exit={rc}: {err.strip()[-200:]}", [alias])}
            continue
        roles[role] = build_role_inventory(role, out, quarantine)
    doc = {"schema": SCHEMA, "read_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "quarantine": [dict(q) for q in quarantine], "roles": roles}
    doc["digest_sha256"] = inventory_digest({k: v for k, v in doc.items() if k != "digest_sha256"})
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    ap.add_argument("--quarantine-file", type=Path, default=DEFAULT_QUARANTINE_FILE)
    ap.add_argument("--role", action="append", choices=ROLES, help="probe only these roles")
    ap.add_argument("--dry-run", action="store_true", help="print the live inventory (the default; never launches)")
    ap.add_argument("--out", type=Path, help="also write the inventory here, write-once")
    a = ap.parse_args(argv)
    roles_map = json.loads(a.roles.read_text())
    doc = read_inventory(roles_map, load_quarantine(a.quarantine_file), only=a.role)
    text = json.dumps(doc, indent=1, sort_keys=True)
    for cfg in roles_map.values():
        if cfg.get("ssh") and cfg["ssh"] in text:
            raise SystemExit("REFUSED: an alias leaked into the inventory")
    if a.out:
        if a.out.exists():
            raise SystemExit(f"REFUSED: {a.out.name} exists; inventories are write-once")
        a.out.write_text(text + "\n")
    print(text)
    return 0 if all(r.get("reachable") for r in doc["roles"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
