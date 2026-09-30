#!/usr/bin/env python3
"""M06 fleet status writer (modular.program.status.v1), stdlib only, no LLM in the loop.

Reads, never writes, the fleet: GPU telemetry, crispdm admission state, systemd unit state,
Traffic cell checkpoints/logs/records, lane worktree tips.  Writes STATUS.json atomically
(temp file + os.replace) every --period seconds and immediately when any job's state changes.

Hosts are recorded by ROLE only.  The role -> ssh alias map lives OUTSIDE the repository
(~/.config/m06/hosts.json); no host name, address or account identifier is written here.

Heartbeat: one unbuffered line per cycle to stdout (journal), plus HEARTBEAT.json beside STATUS.
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
ADM = HOME + "/.local/libexec/crispdm/crispdm_admission.py"
TRAFFIC_ROOT = HOME + "/.local/state/crispdm-data-foundation/traffic_scored_codex_20260930"
GITHUB = HOME + "/Documents/GitHub"
TOTAL_EPOCHS = 30          # sealed train_epochs of the Traffic L96 h96 design
PATIENCE = 3               # author early-stopping patience in the sealed args

# The adopted Traffic cells: (job id, host role, seed, unit name, device uuid)
TRAFFIC = [
    ("traffic_L96_h96_s2021", "coordinator", 2021, "codex-traffic-h96-s2021.service",
     "GPU-612d1e0c-33de-d5cc-56eb-06c0ae424326"),
    ("traffic_L96_h96_s2022", "worker_b", 2022, None,
     "GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9"),
    ("traffic_L96_h96_s2023", "worker_b", 2023, "codex-traffic-h96-s2023.service",
     "GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9"),
]

# One remote probe script: everything we read from a host in one ssh round-trip, read-only.
PROBE = r'''
import json, os, subprocess, time, glob
QUERY_GPU = %s
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=20).stdout
    except Exception as e: return ""
out = {"now": time.time()}
out["gpus"] = "" if not QUERY_GPU else sh("nvidia-smi --query-gpu=uuid,name,temperature.gpu,utilization.gpu,clocks.sm,clocks.max.sm,memory.used,memory.total,power.draw,clocks_throttle_reasons.active --format=csv,noheader,nounits")
out["apps"] = "" if not QUERY_GPU else sh("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits")
mi = {}
for l in open("/proc/meminfo"):
    k, v = l.split(":"); mi[k] = int(v.split()[0]) * 1024
out["mem"] = {k: mi.get(k) for k in ("MemTotal", "MemAvailable", "SUnreclaim", "SReclaimable")}
try: out["psi"] = open("/proc/pressure/memory").read()
except Exception: out["psi"] = ""
out["uptime_s"] = float(open("/proc/uptime").read().split()[0])
out["adm"] = sh("python3 %s state 2>/dev/null")
units = {}
for u in %s:
    if u: units[u] = sh("systemctl --user show -p ActiveState -p SubState -p ActiveEnterTimestampMonotonic -p ExecMainStatus " + u)
out["units"] = units
cells = {}
root = %r
for d in glob.glob(root + "/cell_traffic_h96_s*"):
    seed = d.rsplit("_s", 1)[1]
    c = {}
    log = d + "/author_stdout.log"
    if os.path.exists(log):
        c["log_mtime"] = os.path.getmtime(log)
        c["log"] = open(log, errors="replace").read()[-200000:]
    ck = glob.glob(d + "/checkpoints/*/checkpoint.pth")
    if ck: c["ckpt_mtime"] = os.path.getmtime(ck[0]); c["ckpt_bytes"] = os.path.getsize(ck[0])
    cd = glob.glob(d + "/checkpoints/*")
    if cd: c["ckptdir_mtime"] = os.path.getmtime(cd[0])
    c["dir_mtime"] = os.path.getmtime(d)
    rec = root + "/CELLS/traffic_L96_h96_s%%s.json" %% seed
    if os.path.exists(rec):
        c["record"] = rec; c["record_mtime"] = os.path.getmtime(rec)
        try:
            R = json.load(open(rec)); c["window"] = [R.get("started_at"), R.get("finished_at")]
        except Exception: pass
    cells[seed] = c
out["cells"] = cells
ps = sh("ps -eo pid,etimes,pcpu,rss,args | grep df_tsl_execute.py | grep -v grep | grep -v timeout | grep -v crispdm-run")
cr = sh("ps -eo pid,etimes,args | grep -E 'bin/crispdm-run ' | grep -v grep")
out["launchers"] = []
for l in cr.splitlines():
    f = l.split(None, 2)
    if len(f) == 3:
        a = f[2].split()
        nm = a[a.index("-n") + 1] if "-n" in a else None
        mem = a[a.index("-m") + 1] if "-m" in a else None
        out["launchers"].append({"pid": int(f[0]), "elapsed_s": int(f[1]), "name": nm, "declared_mem": mem})
out["slab_kB"] = out["mem"]["SUnreclaim"] // 1024 if out["mem"].get("SUnreclaim") else None
if QUERY_GPU or True:
    out["gpu_reinit_this_boot"] = sh("journalctl -k -b 0 --no-pager -q 2>/dev/null | grep -c kbifInitLtr").strip()
out["boot_id"] = open("/proc/sys/kernel/random/boot_id").read().strip()
out["children"] = [l.split(None, 4) for l in ps.splitlines() if "--seed" in l]
print(json.dumps(out))
'''


def iso(t):
    if t is None:
        return None
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


GPU_CACHE = {}
# Every nvidia-smi call on a host whose GPUs run without persistence mode re-initialises the
# idle GPU (one "kbifInitLtr" line per init on the 5090 host).  M06's diagnosis suspects a
# per-init kernel-memory leak there, so this writer queries that host's GPUs at most every
# 10 minutes and serves the cached reading in between.
GPU_MIN_INTERVAL = {"worker_a": 600}


def probe(role, hosts):
    units = repr([u for (_, r, _, u, _) in TRAFFIC if r == role and u])
    last = GPU_CACHE.get(role, {}).get("at", 0)
    query = time.time() - last >= GPU_MIN_INTERVAL.get(role, 0)
    code = PROBE % (query, ADM, units, TRAFFIC_ROOT)
    cmd = [sys.executable, "-c", code] if role == "coordinator" else \
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", hosts[role], "python3", "-"]
    err = None
    for _ in range(2):  # one retry: a transient ssh hiccup must not blank a host
        try:
            r = subprocess.run(cmd, input=None if role == "coordinator" else code,
                               capture_output=True, text=True, timeout=60)
            out = json.loads(r.stdout)
            if query:
                GPU_CACHE[role] = {"at": time.time(), "gpus": out.get("gpus"), "apps": out.get("apps")}
            elif role in GPU_CACHE:
                out["gpus"], out["apps"] = GPU_CACHE[role]["gpus"], GPU_CACHE[role]["apps"]
                out["gpu_reading_cached_at"] = GPU_CACHE[role]["at"]
            return out
        except Exception as e:  # unreachable host is recorded, not fatal
            err = type(e).__name__
    return {"error": err}


def parse_log(txt):
    """Epoch costs, eval overhead estimate and last completed epoch from the author's log."""
    costs = [float(m[1]) for m in re.findall(r"Epoch: (\d+) cost time: ([\d.]+)", txt)]
    done = [int(x) for x in re.findall(r"Epoch: (\d+), Steps:", txt)]
    # eval overhead: first 100-iter speed of an epoch minus the steady speed after it
    firsts = re.findall(r"iters: 100, epoch: (\d+) \| loss: [\d.]+\s+speed: ([\d.]+)s/iter", txt)
    seconds = re.findall(r"iters: 200, epoch: (\d+) \| loss: [\d.]+\s+speed: ([\d.]+)s/iter", txt)
    s2 = {int(e): float(s) for e, s in seconds}
    ov = [(float(s) - s2[int(e)]) * 100 for e, s in firsts if int(e) > 1 and int(e) in s2]
    overhead = sorted(ov)[len(ov) // 2] if ov else None
    return {"costs": costs, "last_epoch_logged": max(done) if done else 0,
            "eval_overhead_s": overhead,
            "stopped_early": "Early stopping" in txt}


def gpu_rows(p):
    rows = []
    for l in (p.get("gpus") or "").strip().splitlines():
        f = [x.strip() for x in l.split(",")]
        if len(f) < 10:
            continue
        def num(x):
            try:
                return float(x)
            except ValueError:
                return None
        rows.append({"uuid": f[0], "name": f[1], "temperature_c": num(f[2]),
                     "utilization_percent": num(f[3]), "sm_clock_mhz": num(f[4]),
                     "sm_clock_max_mhz": num(f[5]), "vram_used_mib": num(f[6]),
                     "vram_total_mib": num(f[7]), "power_w": num(f[8]),
                     "throttle_reasons_hex": f[9], "throttle_reasons": decode_throttle(f[9])})
    return rows


def decode_throttle(h):
    names = {0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap",
             0x8: "hw_slowdown", 0x10: "sync_boost", 0x20: "sw_thermal_slowdown",
             0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake_slowdown",
             0x100: "display_clock_setting"}
    try:
        v = int(h, 16)
    except ValueError:
        return []
    return [n for b, n in names.items() if v & b]


def traffic_job(jid, role, seed, unit, uuid, p, now):
    cell = (p.get("cells") or {}).get(str(seed), {})
    unit_state = None
    if unit and unit in (p.get("units") or {}):
        kv = dict(l.split("=", 1) for l in p["units"][unit].splitlines() if "=" in l)
        unit_state = kv.get("ActiveState")
    child = [c for c in p.get("children") or [] if f"--seed {seed} " in c[4] + " "]
    lg = parse_log(cell.get("log", "")) if cell.get("log") else {"costs": [], "last_epoch_logged": 0,
                                                                "eval_overhead_s": None, "stopped_early": False}
    if cell.get("record"):
        state, stage = "completed", "record_sealed"
    elif child:
        state, stage = "running", "training_or_scoring"
    elif unit_state in ("activating",):
        state, stage = "queued", "admission_queue"
    elif not cell:
        state, stage = "unknown", "no_cell_directory"
    else:
        state, stage = "failed", f"no_child_no_record(unit={unit_state})"
    job = {"id": jid, "lane": "M06-adopted-traffic", "state": state if state != "unknown" else "queued",
           "stage": stage, "producer_commit": "author TimeFilter dffde87e + df_tsl_execute (staged copy)",
           "host_alias": role, "device": uuid, "seed": seed, "unit": unit, "unit_active_state": unit_state,
           "heartbeat_at": iso(max([t for t in (cell.get("log_mtime"), cell.get("ckpt_mtime")) if t] or [0]) or None),
           "heartbeat_source": "latest of author_stdout.log mtime (block-buffered) and checkpoint.pth mtime (written only on validation improvement); no <=60 s heartbeat exists",
           "last_checkpoint_at": iso(cell.get("ckpt_mtime")),
           "progress": {"unit": "epochs", "completed": None, "total": TOTAL_EPOCHS},
           "eta": {"earliest": None, "latest": None, "basis": "not_estimable", "assumptions": []}}
    if state == "completed":
        job["progress"]["completed"] = lg["last_epoch_logged"] or None
        job["record_path"] = cell["record"].replace(HOME, "~")
        job["window"] = cell.get("window")
    if child:
        job["resources"] = {"pid": int(child[0][0]), "elapsed_s": int(child[0][1]),
                            "cpu_percent": float(child[0][2]), "rss_bytes": int(child[0][3]) * 1024}
    if state == "running" and lg["costs"] and cell.get("ckptdir_mtime"):
        cost = sum(lg["costs"][-5:]) / len(lg["costs"][-5:])
        ov = lg["eval_overhead_s"] or 0.0
        W = cost + ov
        t0 = cell["ckptdir_mtime"] - W  # the dir is created by the first save, at epoch 1's end
        est_done = min(TOTAL_EPOCHS, max(lg["last_epoch_logged"], int((now - t0) // W)))
        remaining = TOTAL_EPOCHS - est_done
        test_s = 2 * W
        early = max(now, (cell.get("ckpt_mtime") or now) + PATIENCE * W) + test_s * 0.5
        latest = now + (remaining + 1) * W * 1.10 + test_s
        job["progress"]["completed"] = est_done
        job["progress"]["logged_epochs"] = lg["last_epoch_logged"]
        job["eta"] = {"earliest": iso(min(early, latest)), "latest": iso(latest), "basis": "observed_throughput",
                      "assumptions": [
                          f"per-epoch wall W={W:.0f}s = mean train cost of last 5 logged epochs {cost:.0f}s + median eval overhead {ov:.0f}s (first-100-iter speed excess)",
                          f"epochs completed estimated from elapsed time since first checkpoint directory / W (log is block-buffered, logged={lg['last_epoch_logged']})",
                          f"sealed budget {TOTAL_EPOCHS} epochs; earliest = early stop {PATIENCE} epochs after last checkpoint save (validation improvement)",
                          "latest = remaining epochs +1 at W*1.10 (thermal/clock slack) + final test and float64 independent scoring ~2W",
                          "clock/thermal state held as observed now"]}
    return job


def lanes():
    out = []
    for repo in sorted(os.listdir(GITHUB)):
        path = os.path.join(GITHUB, repo)
        if not os.path.isdir(os.path.join(path, ".git")):
            continue
        try:
            wl = subprocess.run(["git", "-C", path, "worktree", "list", "--porcelain"],
                                capture_output=True, text=True, timeout=10).stdout
        except Exception:
            continue
        for block in wl.strip().split("\n\n"):
            kv = {}
            for l in block.splitlines():
                k, _, v = l.partition(" ")
                kv[k] = v
            br = kv.get("branch", "").replace("refs/heads/", "")
            m = re.search(r"\b(m0[1-6]|c07|s07)-", br)
            if m and "20260930" in br:
                wt = kv.get("worktree", "")
                out.append({"lane": m.group(1).upper(), "repo": repo, "branch": br,
                            "worktree": wt.replace(HOME, "~"), "tip": kv.get("HEAD", "")[:12]})
    return out


SLAB_SERIES = None


def lane_of(name):
    m = re.search(r"(m0[1-6]|c07|s07)", name or "")
    return m.group(1).upper() if m else "unknown"


def build(hosts, reg):
    now = time.time()
    probes = {r: probe(r, hosts) for r in ("coordinator", "worker_a", "worker_b")}
    jobs, devices = [], []
    for (jid, role, seed, unit, uuid) in TRAFFIC:
        jobs.append(traffic_job(jid, role, seed, unit, uuid, probes[role], now))
    for role, p in probes.items():
        adm = {}
        try:
            adm = json.loads(p.get("adm") or "{}")
        except ValueError:
            pass
        for g in gpu_rows(p):
            run = [j for j in jobs if j["device"] == g["uuid"] and j["state"] == "running"]
            reason = None
            if run:
                reason = "adopted Traffic cell"
            elif role == "worker_a":
                reason = reg.get("worker_a_idle_reason_by_uuid", {}).get(g["uuid"]) or reg.get("worker_a_idle_reason")
            else:
                reason = "idle: free for the next admissible slot"
            devices.append({
                "host_alias": role, "device": g["uuid"], "name": g["name"],
                "state": "running" if run else ("idle" if g["utilization_percent"] is not None else "unknown"),
                "job_id": run[0]["id"] if run else None, "reason": reason, "observed_at": iso(p.get("now", now)),
                "temperature_c": g["temperature_c"], "utilization_percent": g["utilization_percent"],
                "host_available_bytes": (p.get("mem") or {}).get("MemAvailable"),
                "gpu": g,
                "host": {"mem": p.get("mem"), "unreclaimable_slab_bytes": (p.get("mem") or {}).get("SUnreclaim"),
                         "psi_memory": (p.get("psi") or "").strip(), "uptime_s": p.get("uptime_s")},
                "admission": {k: adm.get(k) for k in ("host_free_for_new_bytes", "held_unrealised_bytes",
                                                      "desktop_reserve_bytes", "pressure_some_avg10",
                                                      "pressure_full_avg10", "pressure_admit_max",
                                                      "slice_memory_current", "slice_memory_max")},
                "leases": [{k: l.get(k) for k in ("lease_id", "name", "cap_bytes", "observed_bytes", "armed", "expires_at")}
                           for l in adm.get("live", [])],
            })
        if "error" in p:
            devices.append({"host_alias": role, "device": "unreachable", "state": "unknown", "job_id": None,
                            "reason": "probe failed: " + p["error"], "observed_at": iso(now)})
    queued, seen = [], set()
    for role, p in probes.items():
        try:
            live = {l.get("name") for l in json.loads(p.get("adm") or "{}").get("live", [])}
        except ValueError:
            live = set()
        for L in p.get("launchers") or []:
            if L["name"] and L["name"] not in live and (role, L["name"]) not in seen:
                seen.add((role, L["name"]))
                queued.append({"id": L["name"], "lane": lane_of(L["name"]), "state": "queued",
                               "stage": "queued_for_admission", "host_alias": role,
                               "declared_mem": L["declared_mem"], "waiting_s": L["elapsed_s"],
                               "producer_commit": "unknown (launcher argv only)", "heartbeat_at": iso(p.get("now")),
                               "progress": {"unit": "unknown", "completed": None, "total": None},
                               "eta": {"earliest": None, "latest": None, "basis": "not_estimable",
                                       "assumptions": ["admission waits for a live lease on this host to release"]}})
    jobs += queued
    wa = probes.get("worker_a", {})
    if "error" not in wa:
        with open(SLAB_SERIES, "a") as f:
            f.write(json.dumps({"at": iso(wa.get("now")), "boot_id": wa.get("boot_id"), "uptime_s": wa.get("uptime_s"),
                                "SUnreclaim_kB": wa.get("slab_kB"), "gpu_reinit_this_boot": wa.get("gpu_reinit_this_boot"),
                                "MemAvailable_B": (wa.get("mem") or {}).get("MemAvailable")}) + "\n")
    agents = reg.get("agents", [])
    wts = lanes()
    for a in agents:
        for w in wts:
            if w["lane"] == a["lane"]:
                a.setdefault("worktrees", [])
                if w not in a["worktrees"]:
                    a["worktrees"].append(w)
                a["commit"] = w["tip"]
    return {"schema": "modular.program.status.v1", "observed_at": iso(now),
            "plan_revision": reg.get("plan_revision", "dc72170e"),
            "writer": {"script": "tools/m06_status_writer.py", "lane": "M06", "pid": os.getpid()},
            "agents": agents, "devices": devices, "jobs": jobs + reg.get("extra_jobs", []),
            "results": reg.get("results", []), "milestones": reg.get("milestones", []),
            "next_actions": reg.get("next_actions", []), "lane_worktrees": wts,
            "events": reg.get("events", []),
            "worker_a_slab_series": {"file": os.path.basename(SLAB_SERIES), "reading": "one sample per writer cycle since the post-reboot baseline"},
            "heartbeat_audit": reg.get("heartbeat_audit", [])}


def atomic_write(path, obj):
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--registry", required=True, help="hand-maintained agents/results/milestones JSON")
    ap.add_argument("--hosts", default=HOME + "/.config/m06/hosts.json")
    ap.add_argument("--period", type=int, default=120)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    hosts = json.load(open(a.hosts))
    global SLAB_SERIES
    SLAB_SERIES = os.path.join(a.out_dir, "worker_a_slab_series.jsonl")
    last_sig, last_write = None, 0.0
    while True:
        try:
            reg = json.load(open(a.registry))
        except Exception:
            reg = {}
        st = build(hosts, reg)
        sig = json.dumps([(j["id"], j["state"], j["stage"]) for j in st["jobs"]])
        now = time.time()
        changed = sig != last_sig
        if changed or now - last_write >= a.period or a.once:
            atomic_write(os.path.join(a.out_dir, "STATUS.json"), st)
            last_write = now
        atomic_write(os.path.join(a.out_dir, "HEARTBEAT.json"),
                     {"at": iso(now), "pid": os.getpid(), "status_written_at": iso(last_write), "job_signature": sig})
        print(iso(now), "status" if now == last_write else "poll", "changed" if changed else "same", sig, flush=True)
        last_sig = sig
        if a.once:
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
