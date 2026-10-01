#!/usr/bin/env python3
"""M06 fleet STATUS writer v2 (modular.program.status.v1 + reconciled fields), stdlib only, no LLM.

Every cycle it runs tools/m06_fleet_probe.py on each host (locally on the coordinator, through
`ssh <alias> python3 -` on the workers) and RECONCILES, per host role and per GPU UUID:

  job  = a lease (joined to its launcher by the launcher pid in the lease label)
         or a launcher without a lease (queued for admission).
  phase, from facts only, in this order:
         failed       an admission incident record for the job in the last 24 h and no live lease
         queued       launcher alive, no lease
         <override>   a dated registry override for that job id (e.g. STOPPING_AT_SAFE_BOUNDARY)
         building / fitting / validating / transferring
                      from the job's own heartbeat stage/event text
         running_stage_unknown
                      leased and alive, but no heartbeat found (named, never guessed)
  devices: running (a compute process in a crispdm scope), desktop_client (a compute process
           outside crispdm, e.g. the desktop's file manager), idle.
Processes that mention crispdm-run but are not launcher argv are listed as UNPARSED with their
argv length only.  Free text never states a job's state; only probes do.

STATUS.json is written atomically every --period seconds and immediately when the job signature
changes.  ETA per job = remaining work / measured rate over the heartbeat history, or the exact
missing measurement.  Hosts are recorded by role; aliases live outside the repository.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
GITHUB = HOME + "/Documents/GitHub"
HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "m06_fleet_probe.py")
GPU_MIN_INTERVAL = {"worker_a": 600}   # each nvidia-smi re-initialises worker_a's idle GPUs (slab hypothesis)
HEARTBEAT_STALE_S = 180
GPU_CACHE: dict = {}
LANE_RE = re.compile(r"\b(m0[1-6]|c07|s07|[a-f]-[a-z]+)-", re.I)


def iso(t):
    return None if t is None else dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_probe(role, hosts):
    query = time.time() - GPU_CACHE.get(role, {}).get("at", 0) >= GPU_MIN_INTERVAL.get(role, 0)
    env = {**os.environ, "M06_QUERY_GPU": "1" if query else "0"}
    if role == "coordinator":
        cmd, inp = [sys.executable, PROBE], None
    else:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", hosts[role],
               f"M06_QUERY_GPU={'1' if query else '0'} python3 -"]
        inp = open(PROBE).read()
    err = None
    for _ in range(2):
        try:
            r = subprocess.run(cmd, input=inp, capture_output=True, text=True, timeout=90, env=env)
            out = json.loads(r.stdout)
            if query:
                GPU_CACHE[role] = {"at": time.time(), "gpus": out["gpus"], "gpu_procs": out["gpu_procs"]}
            elif role in GPU_CACHE:
                out["gpus"], out["gpu_procs"] = GPU_CACHE[role]["gpus"], GPU_CACHE[role]["gpu_procs"]
                out["gpu_reading_at"] = iso(GPU_CACHE[role]["at"])
            return out
        except Exception as e:
            err = type(e).__name__
    return {"error": err}


def decode_throttle(h):
    names = {0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
             0x10: "sync_boost", 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown",
             0x80: "hw_power_brake_slowdown", 0x100: "display_clock_setting"}
    try:
        v = int(h, 16)
    except ValueError:
        return []
    return [n for b, n in names.items() if v & b]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def phase_from_heartbeat(hb):
    last = (hb or {}).get("last") or {}
    text = " ".join(str(x) for x in (last.get("stage"), (last.get("progress") or {}).get("stage"),
                                     (last.get("fit") or {}).get("event"), last.get("event")) if x).lower()
    if not text:
        return None, None
    for words, ph in ((("transfer", "rsync", "copy", "upload"), "transferring"),
                      (("valid", "verify", "rescor", "score"), "validating"),
                      (("build", "compile", "trace", "load"), "building"),
                      (("fit", "train", "update", "epoch", "branch_ae", "core_ae"), "fitting")):
        if any(w in text for w in words):
            return ph, text
    return "running_stage_unknown", text


def eta_from_history(hb, total, now):
    """remaining x measured rate, from (time, completed) pairs of the heartbeat history."""
    pts = [(h["t"], h["completed"]) for h in (hb or {}).get("history", []) if h.get("t") and h.get("completed") is not None]
    if not total:
        return {"earliest": None, "latest": None, "basis": "not_estimable",
                "assumptions": ["missing measurement: the job's total work units are not declared to M06"]}
    if len(pts) < 2 or pts[-1][1] == pts[0][1]:
        return {"earliest": None, "latest": None, "basis": "not_estimable",
                "assumptions": ["missing measurement: fewer than two distinct completed counts in the heartbeat history"]}
    recent = [p for p in pts if p[0] >= pts[-1][0] - 3600] or pts
    if recent[-1][1] == recent[0][1]:
        recent = pts
    rate = (recent[-1][1] - recent[0][1]) / max(1.0, recent[-1][0] - recent[0][0])   # units/s
    remaining = max(0, total - pts[-1][1])
    sec = remaining / rate if rate > 0 else None
    if sec is None:
        return {"earliest": None, "latest": None, "basis": "not_estimable", "assumptions": ["measured rate is zero"]}
    return {"earliest": iso(now + 0.8 * sec), "latest": iso(now + 1.25 * sec), "basis": "observed_throughput",
            "assumptions": [f"rate {rate * 3600:.1f} units/h measured over {(recent[-1][0] - recent[0][0]) / 60:.0f} min "
                            f"of heartbeat history ({recent[0][1]}->{recent[-1][1]})",
                            f"remaining {remaining} of {total} units; interval x0.8..x1.25 of the point estimate",
                            "covers THIS stage only; later stages (e.g. a core AE after the branches) are not included"]}


def campaign_progress(reg, now):
    """Read-only counts and ETA for each registered campaign queue on this (coordinator) host."""
    import sqlite3
    out = []
    for q in reg.get("campaign_queues", []):
        path = os.path.expanduser(q["path"])
        try:
            c = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
            st = dict(c.execute("select status, count(*) from candidates group by status").fetchall())
            att = c.execute("select kind, status, started, finished, elapsed_seconds, cgroup_peak_bytes, host "
                            "from attempts").fetchall()
            inc = c.execute("select config_id, mean_objective, time from incumbent_changes order by seq desc limit 1").fetchone()
            c.close()
        except Exception as e:
            out.append({"campaign": q["name"], "error": type(e).__name__})
            continue
        excluded = {k: v for k, v in st.items() if k in q.get("excluded_statuses", [])}
        live = {k: v for k, v in st.items() if k not in excluded}
        done = live.get("verified", 0) + live.get("failed", 0)
        remaining = live.get("queued", 0) + live.get("running", 0) + live.get("trained", 0)
        fin = [a for a in att if a[1] == "completed" and a[4]]
        per_cell = None
        if fin:
            tr = [a[4] for a in fin if a[0] == "train"]
            ve = [a[4] for a in fin if a[0] == "verify"]
            per_cell = (sum(tr) / len(tr) if tr else 0) + (sum(ve) / len(ve) if ve else 0)
        hosts = max(1, int(q.get("parallel_hosts", 1)))
        if per_cell and remaining:
            sec = remaining * per_cell / hosts
            eta = {"earliest": iso(now + 0.8 * sec), "latest": iso(now + 1.5 * sec), "basis": "observed_throughput",
                   "assumptions": [f"mean completed train+verify wall {per_cell:.0f} s per cell over {len(fin)} attempts",
                                   f"{remaining} cells remaining (queued+running) across {hosts} host(s); x0.8..x1.5",
                                   "admission waits and pressure stops not modelled"]}
        else:
            eta = {"earliest": None, "latest": None, "basis": "not_estimable",
                   "assumptions": ["missing measurement: no completed train+verify attempt in this campaign yet"
                                   if not per_cell else "no remaining cells"]}
        out.append({"campaign": q["name"], "status_counts": st, "excluded": excluded,
                    "cells_done": done, "cells_planned": sum(live.values()), "eta": eta,
                    "attempts": {"total": len(att), "completed": len(fin),
                                 "max_cgroup_peak_bytes": max([a[5] or 0 for a in att] or [0])},
                    "incumbent": {"config_id": inc[0][:8], "mean_objective": inc[1], "at": iso(inc[2])} if inc else None})
    return out


def coverage_block(reg, campaigns_out):
    """Three counts kept apart (owner addendum 256c61a6): catalogue coverage, model-input count,
    evaluated-candidate count.  A complete-coverage claim is REFUSED while any known source or
    transform family is absent from lane B's accounting."""
    import sqlite3
    cat = reg.get("coverage_catalogue", {})
    absent = cat.get("known_absent_from_accounting", [])
    inputs = []
    for q in reg.get("campaign_queues", []):
        try:
            c = sqlite3.connect(f"file:{os.path.expanduser(q['path'])}?mode=ro", uri=True, timeout=5)
            rows = c.execute("select label, status, nested from candidates where status not in (%s)" %
                             ",".join("?" * len(q.get("excluded_statuses", []) or [""])),
                             q.get("excluded_statuses", []) or [""]).fetchall()
            c.close()
        except Exception as e:
            inputs.append({"campaign": q["name"], "error": type(e).__name__})
            continue
        per = {}
        for label, status, nested in rows:
            m = (json.loads(nested) if nested else {}).get("model", {})
            chans = sorted({f for b in m.get("branches", []) for f in b.get("features", [])})
            per.setdefault(label, {"input_channels": len(chans), "branches": len(m.get("branches", [])),
                                   "targets": m.get("target_count"), "statuses": []})["statuses"].append(status)
        inputs.append({"campaign": q["name"], "per_config": per,
                       "distinct_input_channel_counts": sorted({v["input_channels"] for v in per.values()})})
    evaluated = [{"campaign": c["campaign"], "verified_rows": (c.get("status_counts") or {}).get("verified", 0)}
                 for c in campaigns_out if "status_counts" in c]
    evaluated += reg.get("coverage_evaluated_retained", [])
    return {"catalogue_coverage": {**cat, "complete_coverage_claim":
                                   "REFUSED: known sources/families absent from lane B's accounting: " + "; ".join(absent)
                                   if absent else "not asserted (no census of every usable source exists)"},
            "model_input_count": inputs, "evaluated_candidate_count": evaluated,
            "rule": "three counts are never summed or substituted for one another"}


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
            kv = dict((l.partition(" ")[0], l.partition(" ")[2]) for l in block.splitlines())
            br = kv.get("branch", "").replace("refs/heads/", "")
            if re.search(r"(m0[1-6]|c07|s07|-engine-|corrected-queue|continuation-ack)", br) and \
                    ("20260930" in br or "20261001" in br):
                out.append({"repo": repo, "branch": br, "worktree": kv.get("worktree", "").replace(HOME, "~"),
                            "tip": kv.get("HEAD", "")[:12]})
    return out


def build(hosts, reg):
    now = time.time()
    probes = {r: run_probe(r, hosts) for r in ("coordinator", "worker_a", "worker_b")}
    overrides = reg.get("job_overrides", {})
    hints = reg.get("job_hints", {})
    jobs, devices, unparsed, failed = [], [], [], []
    for role, p in probes.items():
        if "error" in p:
            devices.append({"host_alias": role, "device": "unreachable", "state": "unknown", "job_id": None,
                            "reason": "probe failed: " + str(p["error"]), "observed_at": iso(now)})
            continue
        live_names = set()
        for L in p.get("leases", []):
            live_names.add(L["name"])
            hb = L.get("heartbeat")
            ph, text = phase_from_heartbeat(hb)
            hb_at = (hb or {}).get("mtime")
            if re.search(r"(^|-)verify-", L["name"]):
                ph, text = "validating", "independent checkpoint rescoring (job kind from its launcher name)"
            if L["name"] in overrides:
                ph = overrides[L["name"]]["phase"]
            elif ph is None:
                ph = "running_stage_unknown"
            hint = next((v for k, v in hints.items() if re.fullmatch(k, L["name"])), {})
            last = (hb or {}).get("last") or {}
            done = (last.get("resume_point") or {}).get("completed_branches")
            jobs.append({
                "id": L["name"], "lane": hint.get("lane") or (LANE_RE.search(L["name"] + "-").group(1).upper()
                                                             if LANE_RE.search(L["name"] + "-") else "unknown"),
                "state": "running", "phase": ph, "stage": text or "no heartbeat stage",
                "producer_commit": hint.get("producer_commit", "unknown"), "host_alias": role,
                "lease_id": L["lease_id"], "launcher_pid": L["launcher_pid"], "launcher_alive": L["launcher_alive"],
                "child_heads": L["child_heads"], "cap_bytes": L["cap_bytes"],
                "cgroup_current_bytes": L["cgroup_current"], "cgroup_peak_bytes": L["cgroup_peak"],
                "heartbeat_at": iso(hb_at), "heartbeat_path": (hb or {}).get("path", "").replace(HOME, "~") or None,
                "heartbeat_stale": (hb_at is not None and now - hb_at > HEARTBEAT_STALE_S),
                "progress": {"unit": hint.get("unit", "unknown"), "completed": done, "total": hint.get("total")},
                "eta": eta_from_history(hb, hint.get("total"), now),
                "override": overrides.get(L["name"])})
        for Q in p.get("launchers_without_lease", []):
            jobs.append({"id": Q["name"] or f"unnamed-launcher-{Q['pid']}", "lane": "unknown" if not Q["name"] else
                         (LANE_RE.search(Q["name"] + "-").group(1).upper() if LANE_RE.search(Q["name"] + "-") else "unknown"),
                         "state": "queued", "phase": "queued", "stage": "waiting for admission",
                         "producer_commit": "unknown", "host_alias": role, "launcher_pid": Q["pid"],
                         "declared_mem": Q["mem"], "declared_wall": Q["wall"], "queue_wait_s": Q["wait"],
                         "waiting_s": Q["elapsed_s"], "command_head": Q["command_head"], "heartbeat_at": None,
                         "progress": {"unit": "unknown", "completed": None, "total": None},
                         "eta": {"earliest": None, "latest": None, "basis": "not_estimable",
                                 "assumptions": ["admission order depends on live leases releasing"]}})
        for u in p.get("unparsed", []):
            unparsed.append({**u, "host_alias": role, "classification": "UNPARSED (mentions crispdm-run; not a launcher argv)"})
        for i in p.get("incidents_24h", []):
            failed.append({**i, "host_alias": role, "at": iso(i["at"]), "live_again": i["name"] in live_names})
        # devices
        adm = p.get("admission", {})
        for f in p.get("gpus", []):
            uuid = f[0]
            procs = [g for g in p.get("gpu_procs", []) if g["uuid"] == uuid]
            jprocs = [g for g in procs if g.get("cgroup") and "crispdm-batch.slice" in g["cgroup"]]
            job = next((j for j in jobs if j.get("host_alias") == role and any(
                (j.get("lease_id") or "") and g["cgroup"].endswith(".scope") and j["id"] in g["cgroup"] for g in jprocs)), None)
            state = "running" if jprocs else "idle"
            devices.append({
                "host_alias": role, "device": uuid, "name": f[1], "state": state,
                "job_id": job["id"] if job else (jprocs[0]["cgroup"].rsplit("/", 1)[-1] if jprocs else None),
                "reason": ("compute process in a crispdm job scope" if jprocs else
                           ("desktop client only: " + ", ".join(g["exe"] or "?" for g in procs) if procs
                            else "no compute process and no GPU job lease on this device")),
                "observed_at": p.get("gpu_reading_at") or iso(p["now"]),
                "temperature_c": num(f[2]), "utilization_percent": num(f[3]),
                "host_available_bytes": (p.get("mem") or {}).get("MemAvailable"),
                "gpu": {"sm_clock_mhz": num(f[4]), "sm_clock_max_mhz": num(f[5]), "vram_used_mib": num(f[6]),
                        "vram_total_mib": num(f[7]), "power_w": num(f[8]), "throttle_reasons": decode_throttle(f[9]),
                        "compute_processes": procs},
                "host": {"mem": p.get("mem"), "psi_memory": p.get("psi"), "uptime_s": p.get("uptime_s"),
                         "gpu_reinit_this_boot": p.get("gpu_reinit_this_boot"),
                         "shares_host_ram_with": [g[0] for g in p.get("gpus", []) if g[0] != uuid]},
                "admission": {k: adm.get(k) for k in ("host_free_for_new_bytes", "held_unrealised_bytes",
                                                      "desktop_reserve_bytes", "slice_memory_current",
                                                      "slice_charged_bytes", "slice_memory_max",
                                                      "pressure_some_avg10", "pressure_full_avg10", "pressure_admit_max")},
                "next_task": reg.get("slot_plan", {}).get(uuid)})
    # jobs that ended: an override for a job id no longer leased becomes its terminal record
    for name, ov in overrides.items():
        if not any(j["id"] == name for j in jobs):
            jobs.append({"id": name, "lane": ov.get("lane", "unknown"), "state": ov.get("terminal_state", "completed"),
                         "phase": ov.get("terminal_phase", "ended"), "stage": ov.get("terminal_stage", "ended"),
                         "producer_commit": ov.get("producer_commit", "unknown"), "host_alias": ov.get("host_alias"),
                         "heartbeat_at": ov.get("last_heartbeat_at"),
                         "progress": ov.get("progress", {"unit": "unknown", "completed": None, "total": None}),
                         "eta": {"earliest": None, "latest": None, "basis": "not_estimable", "assumptions": ["job ended"]},
                         "facts": ov.get("facts")})
    jobs += reg.get("completed_jobs", [])
    wa = probes.get("worker_a", {})
    if "error" not in wa and SLAB_SERIES:
        with open(SLAB_SERIES, "a") as fh:
            fh.write(json.dumps({"at": iso(wa["now"]), "boot_id": wa.get("boot_id"), "uptime_s": wa.get("uptime_s"),
                                 "SUnreclaim_kB": (wa.get("mem") or {}).get("SUnreclaim", 0) // 1024,
                                 "gpu_reinit_this_boot": wa.get("gpu_reinit_this_boot"),
                                 "MemAvailable_B": (wa.get("mem") or {}).get("MemAvailable")}) + "\n")
    quotas = {role: {**{k: (p.get("admission") or {}).get(k) for k in
                        ("host_free_for_new_bytes", "held_unrealised_bytes", "desktop_reserve_bytes", "slice_memory_max",
                         "slice_memory_current", "slice_charged_bytes", "pressure_some_avg10")},
                     "mem_available_bytes": (p.get("mem") or {}).get("MemAvailable"),
                     "disk_home_free_bytes": (p.get("disk_home") or {}).get("free_bytes"),
                     "disk_home_total_bytes": (p.get("disk_home") or {}).get("total_bytes")}
              for role, p in probes.items() if "error" not in p}
    return {"schema": "modular.program.status.v1", "observed_at": iso(now),
            "plan_revision": reg.get("plan_revision"), "plan_links": reg.get("plan_links", {}),
            "writer": {"script": "tools/m06_status_writer.py + tools/m06_fleet_probe.py", "version": 2, "pid": os.getpid()},
            "agents": reg.get("agents", []), "lanes": reg.get("lanes", []), "devices": devices, "jobs": jobs,
            "campaigns": (camps := campaign_progress(reg, now)),
            "coverage": coverage_block(reg, camps),
            "unparsed_processes": unparsed, "terminal_incidents_24h": failed, "quotas_measured": quotas,
            "results": reg.get("results", []), "milestones": reg.get("milestones", []),
            "next_actions": reg.get("next_actions", []), "events": reg.get("events", []),
            "corrections": reg.get("corrections", []), "lane_worktrees": lanes(),
            "admission_findings": reg.get("admission_findings", []),
            "open_owner_questions": reg.get("open_owner_questions"),
            "independent_checks": reg.get("independent_checks", []),
            "heartbeat_audit": reg.get("heartbeat_audit", []),
            "worker_a_slab_series": "worker_a_slab_series.jsonl"}


SLAB_SERIES = None


def scrub(obj, names):
    """Replace every host name (aliases, the local host name) in every string with <host>.
    Argv of child processes (e.g. a compiler's temp file) can carry the host name."""
    pat = re.compile("|".join(re.escape(n) for n in sorted(names, key=len, reverse=True) if n))
    def go(x):
        if isinstance(x, str):
            return pat.sub("<host>", x)
        if isinstance(x, list):
            return [go(v) for v in x]
        if isinstance(x, dict):
            return {go(k): go(v) for k, v in x.items()}
        return x
    return go(obj) if pat.pattern else obj


def atomic_write(path, obj):
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    global SLAB_SERIES
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--hosts", default=HOME + "/.config/m06/hosts.json")
    ap.add_argument("--period", type=int, default=120)
    ap.add_argument("--poll", type=int, default=45)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    hosts = json.load(open(a.hosts))
    SLAB_SERIES = os.path.join(a.out_dir, "worker_a_slab_series.jsonl")
    last_sig, last_write = None, 0.0
    while True:
        try:
            reg = json.load(open(a.registry))
        except Exception:
            reg = {}
        import socket
        st = scrub(build(hosts, reg), set(hosts.values()) | {socket.gethostname()} | set(reg.get("scrub_names_extra", [])))
        sig = json.dumps(sorted((j["id"], j["state"], j.get("phase")) for j in st["jobs"]))
        now = time.time()
        changed = sig != last_sig
        if changed or now - last_write >= a.period or a.once:
            atomic_write(os.path.join(a.out_dir, "STATUS.json"), st)
            last_write = now
        atomic_write(os.path.join(a.out_dir, "HEARTBEAT.json"),
                     {"at": iso(now), "pid": os.getpid(), "status_written_at": iso(last_write), "job_signature": sig})
        print(iso(now), "status" if now == last_write else "poll", "changed" if changed else "same", sig[:300], flush=True)
        last_sig = sig
        if a.once:
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
