#!/usr/bin/env python3
"""Read-only fleet probe, run ON a host (locally or as `ssh host python3 - < this`). Prints one JSON.

It reconciles, on this host only:
  * launchers  - crispdm-run processes, found by an EXACT argv test on /proc/<pid>/cmdline
                 (argv[0] is the crispdm-run script, or argv[0] is a shell and argv[1] is the
                 script).  Options are parsed with the launcher's own option set
                 `m:t:n:W:L:E:P:S:qh`, up to the first `--`.  Shell wrappers (`bash -c '...'`),
                 ssh commands, `timeout` and `env` are never parsed, so an ssh alias or a word
                 like "venv" can never become a job name or a memory size.
  * leases     - the admission store's lease files, joined to a launcher by the launcher pid that
                 ends the lease label (`crispdm-<name>-<ts>-<launcher pid>`).
  * processes  - the lease cgroup's own processes (cgroup.procs) with their argv.
  * heartbeats - heartbeat.json / heartbeat.jsonl in a directory the job's own processes name
                 after --out / --out-dir / --output-root / --root, or in their cwd.
  * GPUs       - nvidia-smi (optionally skipped) and its compute processes, each mapped to the
                 cgroup of its pid.
  * incidents  - admission incident records from the last 24 h (terminal stops).
It writes nothing.
"""
import glob
import json
import os
import subprocess
import time

QUERY_GPU = os.environ.get("M06_QUERY_GPU", "1") == "1"
HOME = os.path.expanduser("~")
STORE = os.environ.get("CRISPDM_ADMISSION_DIR") or HOME + "/.local/state/crispdm/admission"
ADM = HOME + "/.local/libexec/crispdm/crispdm_admission.py"
OPTS_WITH_ARG = set("mtnWLEPS")
OPTS_FLAG = set("qh")
OUT_FLAGS = ("--out", "--out-dir", "--output-root", "--root", "--output", "--outdir")


def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def argv_of(pid):
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
    except OSError:
        return None
    return [a.decode(errors="replace") for a in raw.split(b"\0") if a != b""] or None


def etimes(pid):
    try:
        st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        clk = os.sysconf("SC_CLK_TCK")
        up = float(open("/proc/uptime").read().split()[0])
        return int(up - int(st[19]) / clk)
    except Exception:
        return None


def cgroup_of(pid):
    try:
        for line in open(f"/proc/{pid}/cgroup"):
            if line.startswith("0::"):
                return line[3:].strip().lstrip("/")
    except OSError:
        return None


def parse_launcher(argv):
    """argv -> {name, mem, wall, wait, queue, slice, command} or None if not a crispdm-run call."""
    if not argv:
        return None
    i = 0
    if argv[0].endswith("/crispdm-run") or argv[0] == "crispdm-run":
        i = 1
    elif os.path.basename(argv[0]) in ("bash", "sh", "dash") and len(argv) > 1 and \
            (argv[1].endswith("/crispdm-run") or argv[1] == "crispdm-run"):
        i = 2
    else:
        return None
    out = {"name": None, "mem": None, "wall": None, "wait": None, "queue": False, "slice": None,
           "label": None, "command": []}
    keys = {"n": "name", "m": "mem", "t": "wall", "W": "wait", "S": "slice", "L": "label"}
    while i < len(argv):
        a = argv[i]
        if a == "--":
            out["command"] = argv[i + 1:]
            break
        if a.startswith("-") and len(a) >= 2 and a[1] in OPTS_WITH_ARG | OPTS_FLAG:
            # getopts clusters: -qm 4G, -m4G
            j = 1
            while j < len(a):
                c = a[j]
                if c in OPTS_FLAG:
                    if c == "q":
                        out["queue"] = True
                    j += 1
                    continue
                if c in OPTS_WITH_ARG:
                    val = a[j + 1:] if j + 1 < len(a) else (argv[i + 1] if i + 1 < len(argv) else None)
                    if j + 1 >= len(a):
                        i += 1
                    if c in keys:
                        out[keys[c]] = val
                    break
                break
            i += 1
            continue
        out["command"] = argv[i:]       # first non-option word starts the command
        break
    return out


def read_heartbeat(dirs):
    for d in dirs:
        for name in ("heartbeat.json", "heartbeat.jsonl"):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                try:
                    if name.endswith(".jsonl"):
                        with open(p, "rb") as fh:
                            fh.seek(0, 2)
                            size = fh.tell()
                            fh.seek(max(0, size - 400000))
                            lines = [l for l in fh.read().decode(errors="replace").splitlines() if l.strip()]
                        hist = []
                        for l in lines[-400:]:
                            try:
                                hist.append(json.loads(l))
                            except ValueError:
                                pass
                        last = hist[-1] if hist else None
                    else:
                        last = json.load(open(p))
                        hist = []
                    return {"path": p, "mtime": os.path.getmtime(p), "last": last,
                            "history": [{"t": h.get("time_unix"), "stage": h.get("stage"),
                                         "completed": (h.get("resume_point") or {}).get("completed_branches")}
                                        for h in hist]}
                except Exception as e:  # unreadable is recorded, not fatal
                    return {"path": p, "error": type(e).__name__}
    return None


def main():
    now = time.time()
    out = {"now": now, "uptime_s": float(open("/proc/uptime").read().split()[0]),
           "boot_id": open("/proc/sys/kernel/random/boot_id").read().strip()}
    mi = {}
    for l in open("/proc/meminfo"):
        k, v = l.split(":")
        mi[k] = int(v.split()[0]) * 1024
    out["mem"] = {k: mi.get(k) for k in ("MemTotal", "MemAvailable", "SUnreclaim", "SReclaimable", "Shmem")}
    st = os.statvfs(HOME)
    out["disk_home"] = {"free_bytes": st.f_bavail * st.f_frsize, "total_bytes": st.f_blocks * st.f_frsize}
    try:
        out["psi"] = open("/proc/pressure/memory").read().strip()
    except OSError:
        out["psi"] = None
    try:
        out["admission"] = json.loads(sh(f"python3 {ADM} state 2>/dev/null") or "{}")
        out["admission"].pop("live", None)
    except ValueError:
        out["admission"] = {}
    # GPUs
    out["gpus"], out["gpu_procs"] = [], []
    if QUERY_GPU:
        for l in sh("nvidia-smi --query-gpu=uuid,name,temperature.gpu,utilization.gpu,clocks.sm,clocks.max.sm,"
                    "memory.used,memory.total,power.draw,clocks_throttle_reasons.active "
                    "--format=csv,noheader,nounits").strip().splitlines():
            f = [x.strip() for x in l.split(",")]
            if len(f) >= 10:
                out["gpus"].append(f)
        for l in sh("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits").strip().splitlines():
            f = [x.strip() for x in l.split(",")]
            if len(f) == 3 and f[1].isdigit():
                pid = int(f[1])
                a = argv_of(pid) or []
                out["gpu_procs"].append({"uuid": f[0], "pid": pid, "used_mib": f[2], "cgroup": cgroup_of(pid),
                                         "exe": os.path.basename(a[0]) if a else None})
    out["gpu_reinit_this_boot"] = sh("journalctl -k -b 0 --no-pager -q 2>/dev/null | grep -c kbifInitLtr").strip()
    # launchers
    launchers, unparsed = {}, []
    me = os.getpid()
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        a = argv_of(int(d))
        p = parse_launcher(a)
        if a and len(a) > 1 and a[1].endswith("crispdm_admission.py"):
            continue                     # the admission module itself (monitor/acquire): part of a lease
        if p is None and a and any("crispdm-run" in x for x in a):
            # mentions the launcher but is not a launcher argv (a shell -c string, an ssh wrapper,
            # a waiter loop): reported, never guessed into a job
            unparsed.append({"pid": int(d), "argv_len": len(a), "argv0": os.path.basename(a[0]),
                             "elapsed_s": etimes(int(d))})
            continue
        if p:
            p["pid"] = int(d)
            p["elapsed_s"] = etimes(int(d))
            p["command_head"] = [os.path.basename(x) if i == 0 else x for i, x in enumerate(p.pop("command")[:6])]
            launchers[int(d)] = p
    # leases
    leases = []
    for f in sorted(glob.glob(STORE + "/leases/*.json")):
        try:
            L = json.load(open(f))
        except ValueError:
            continue
        lp = None
        try:
            lp = int(str(L.get("label", "")).rsplit("-", 1)[1])
        except (IndexError, ValueError):
            pass
        cg = L.get("cgroup")
        procs = []
        try:
            for pid in open(f"/sys/fs/cgroup/{cg}/cgroup.procs").read().split():
                a = argv_of(int(pid))
                if a:
                    procs.append({"pid": int(pid), "argv": a[:40], "cwd": os.readlink(f"/proc/{pid}/cwd")
                                  if os.path.exists(f"/proc/{pid}/cwd") else None})
        except (OSError, TypeError):
            pass
        dirs = []
        for pr in procs:
            a = pr["argv"]
            for k, w in enumerate(a[:-1]):
                if w in OUT_FLAGS:
                    dirs.append(os.path.expanduser(a[k + 1]))
            if pr.get("cwd"):
                dirs.append(pr["cwd"])
        cur = None
        try:
            cur = int(open(f"/sys/fs/cgroup/{cg}/memory.current").read())
            peak = int(open(f"/sys/fs/cgroup/{cg}/memory.peak").read())
        except (OSError, TypeError, ValueError):
            peak = None
        leases.append({"lease_id": L.get("lease_id"), "name": L.get("name"), "cap_bytes": L.get("cap_bytes"),
                       "armed": L.get("armed"), "created_at": L.get("created_at"), "expires_at": L.get("expires_at"),
                       "launcher_pid": lp, "launcher_alive": lp in launchers, "cgroup": cg,
                       "cgroup_current": cur, "cgroup_peak": peak, "nprocs": len(procs),
                       "child_heads": [[os.path.basename(pr["argv"][0])] + pr["argv"][1:4] for pr in procs][:6],
                       "heartbeat": read_heartbeat(dirs)})
    out["leases"] = leases
    leased_pids = {l["launcher_pid"] for l in leases}
    out["launchers_without_lease"] = [v for k, v in launchers.items() if k not in leased_pids]
    out["unparsed"] = unparsed
    # incidents (terminal stops) in the last 24 h
    inc = []
    for f in glob.glob(STORE + "/incidents/*.json"):
        m = os.path.getmtime(f)
        if now - m < 86400:
            try:
                d = json.load(open(f))
            except ValueError:
                d = {}
            inc.append({"lease_id": d.get("lease_id") or os.path.basename(f)[:-5], "name": d.get("name"),
                        "at": m, "exit_cause": d.get("exit_cause"), "rule": d.get("rule"),
                        "tree_peak_bytes": d.get("observed_tree_peak_bytes"), "cap_bytes": d.get("cap_bytes")})
    out["incidents_24h"] = sorted(inc, key=lambda x: x["at"])
    print(json.dumps(out))


if __name__ == "__main__":
    main()
