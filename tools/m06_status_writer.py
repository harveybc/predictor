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
sys.path.insert(0, HERE)
import m06_observed_state as obs  # noqa: E402  pure declared-vs-observed logic (corrective order 2026-10-03 §5)

QUEUE_READER = r"""
import json, sqlite3, sys
p = sys.argv[1]
try:
    c = sqlite3.connect("file:%s?mode=ro" % p, uri=True, timeout=5)
    out = {"status_counts": dict(c.execute("select status, count(*) from candidates group by status").fetchall()),
           "candidates": [list(r) for r in c.execute("select cid, label, seed, status from candidates order by position")],
           "attempts": [list(r) for r in c.execute("select cid, attempt, kind, status, elapsed_seconds, cgroup_peak_bytes from attempts")]}
    c.close()
except Exception as e:
    out = {"error": type(e).__name__}
print(json.dumps(out))
"""
GPU_MIN_INTERVAL = {"worker_a": 60}    # owner order 2026-10-01 (no idle GPU): 60 s sampling for the idle alarm.
                                       # Trade-off recorded: each query of an idle, non-persistent GPU re-initialises it
                                       # (worker_a slab hypothesis, UNVERIFIED); the slab series is watched (alert >= 4 GB).
IDLE_STATE: dict = {}                  # uuid -> {"since": t or None, "events": [(t, idle_dt)], "last_t": t}
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
            att = c.execute("select kind, status, started, finished, elapsed_seconds, cgroup_peak_bytes, host, cid "
                            "from attempts").fetchall()
            labels = dict(c.execute("select cid, label from candidates").fetchall())
            pending = [r[0] for r in c.execute("select label from candidates where status in "
                                               "('queued','running','trained','verifying','completed')").fetchall()]
            inc = c.execute("select config_id, mean_objective, time from incumbent_changes order by seq desc limit 1").fetchone()
            c.close()
        except Exception as e:
            out.append({"campaign": q["name"], "error": type(e).__name__})
            continue
        excluded = {k: v for k, v in st.items() if k in q.get("excluded_statuses", [])}
        live = {k: v for k, v in st.items() if k not in excluded}
        done = live.get("verified", 0) + live.get("failed", 0)
        remaining = sum(v for k, v in live.items() if k in ("queued", "running", "trained", "verifying", "completed"))
        dependency_held = {k: v for k, v in live.items() if k == "blocked" or k.startswith("HOLD")}
        fin = [a for a in att if a[1] == "completed" and a[4]]
        def family(lbl):
            # the label without its loss suffix: corrected_default_R0_huber -> corrected_default_R0,
            # corrected_default_branch_R1_huber -> corrected_default_branch_R1 (regime and donor scope matter)
            parts = (lbl or "").split("_")
            return "_".join(parts[:-1]) if len(parts) > 1 else (lbl or "")
        cell_s = {}
        for a_ in fin:
            fam = family(labels.get(a_[7]))
            cell_s.setdefault(fam, {"train": [], "verify": []})[a_[0]].append(a_[4])
        def fam_cost(fam):
            d = cell_s.get(fam)
            if not d or not d["train"]:
                return None
            return sum(d["train"]) / len(d["train"]) + (sum(d["verify"]) / len(d["verify"]) if d["verify"] else 0)
        per_cell = None
        hosts = max(1, int(q.get("parallel_hosts", 1)))
        costs = [fam_cost(family(l)) for l in pending]
        if pending and all(x is not None for x in costs):
            sec = sum(costs) / hosts
            per_cell = sec / len(pending)
            fams = sorted({family(l) for l in pending})
            eta = {"earliest": iso(now + 0.8 * sec), "latest": iso(now + 1.5 * sec), "basis": "observed_throughput",
                   "assumptions": [f"per-family measured train+verify wall: " + "; ".join(
                                       f"{f} {fam_cost(f):.0f} s" for f in fams),
                                   f"{len(pending)} cells pending (queued+running) across {hosts} host(s); x0.8..x1.5",
                                   "admission waits and pressure stops not modelled"]}
        elif pending:
            missing = sorted({family(l) for l, x in zip(pending, costs) if x is None})
            known = [x for x in costs if x is not None]
            part = (f"partial lower bound from measured families: {len(known)} cells x measured wall = "
                    f"{sum(known) / hosts / 60:.0f} min (" + "; ".join(f"{f} {fam_cost(f):.0f} s/cell" for f in
                    sorted({family(l) for l, x in zip(pending, costs) if x is not None})) + ")") if known else "no family measured"
            eta = {"earliest": iso(now + sum(known) / hosts) if known else None, "latest": None, "basis": "not_estimable",
                   "assumptions": [f"missing measurement: no completed train+verify attempt yet for family {missing} "
                                   f"({sum(1 for x in costs if x is None)} cells)", part,
                                   "earliest = the measured part only; no latest until every pending family is measured"]}
        else:
            eta = {"earliest": None, "latest": None, "basis": "not_estimable", "assumptions": ["no runnable cells remain"]}
        if dependency_held:
            eta["assumptions"].append(f"{sum(dependency_held.values())} cells held ({dependency_held}): "
                                      + (q.get("dependency_note") or "not estimable until their pilot/donors exist"))
        out.append({"campaign": q["name"], "identity": q.get("identity"), "status_counts": st, "excluded": excluded,
                    "dependency_held": dependency_held,
                    "cells_done": done, "cells_planned": sum(live.values()), "eta": eta,
                    "attempts": {"total": len(att), "completed": len(fin),
                                 "max_cgroup_peak_bytes": max([a[5] or 0 for a in att] or [0])},
                    "incumbent": {"config_id": inc[0][:8], "mean_objective": inc[1], "at": iso(inc[2])} if inc else None})
    out += reg.get("campaigns_retained", [])
    return out


def coverage_block(reg, campaigns_out):
    """Three counts kept apart (owner addendum 256c61a6): catalogue coverage, model-input count,
    evaluated-candidate count.  A complete-coverage claim is REFUSED while any known source or
    transform family is absent from lane B's accounting."""
    import sqlite3
    cat = dict(reg.get("coverage_catalogue", {}))
    gen = reg.get("coverage_generated_file")
    if gen and os.path.exists(gen):
        try:
            g = json.load(open(gen))
            cat["new_source_and_transform_denominators"] = {
                "generated_from": os.path.basename(gen), "new_denominator": g["new_denominator"],
                "old_denominator": g["old_denominator"], "grain_note": g.get("grain_note"),
                "transform_ledger": g["transform_ledger"], "sources": g["sources"], "dag": g["dag"], "inputs": g["inputs"]}
        except Exception as e:
            cat["new_source_and_transform_denominators"] = "unreadable generated file: " + type(e).__name__
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
    three = {"distinct_column_profiles": cat.get("old_denominator"),
             "metric_family_cells": reg.get("coverage_metric_family_cells") or
             "NOT_PUBLISHED: missing measurement; owner lane B (M03); no number is substituted",
             "source_recipe_catalogue": (cat.get("new_source_and_transform_denominators") or {}).get("new_denominator")
             if isinstance(cat.get("new_source_and_transform_denominators"), dict) else None,
             "rule": "three denominators; none substitutes for the others (orders b327b771 item 5)"}
    return {"three_denominators": three,
            "catalogue_coverage": {**cat, "complete_coverage_claim":
                                   "REFUSED: known sources/families absent from lane B's accounting: " + "; ".join(absent)
                                   if absent else "not asserted (no census of every usable source exists)"},
            "model_input_count": inputs, "evaluated_candidate_count": evaluated,
            "rule": "three counts are never summed or substituted for one another"}


def sampler_idle(uuid, fallback):
    """Idle counters from m06-gpu-sampler (15 s sampling; idle = 0 % or no compute process), else the
    writer's own coarser counter."""
    try:
        d = json.load(open(os.path.expanduser("~/.local/state/m06/gpu_idle.json"))).get(uuid)
    except Exception:
        d = None
    if d:
        return {"gpu_idle_seconds_24h": d.get("idle_24h_s"), "gpu_idle_current_seconds": d.get("idle_current_s"),
                "sampler_temperature_c": d.get("temp_c"), "sampler_temperature_max_since_start_c": d.get("temp_max_since_start_c"),
                "sampler_vram_used_mib": d.get("mem_used_mib"), "sampler_vram_total_mib": d.get("mem_total_mib"),
                "sampler_power_w": d.get("power_w"),
                "sampler_sample_at": (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(d["last_t"])) if d.get("last_t") else None),
                "gpu_idle_note": "m06-gpu-sampler: 15 s sampling; idle = 0 % utilization or no compute process; alarm at 120 s"}
    f = fallback.get(uuid, {})
    return {"gpu_idle_seconds_24h": int(sum(e[1] for e in f.get("events", []))),
            "gpu_idle_current_seconds": (int(time.time() - f["since"]) if f.get("since") else 0),
            "gpu_idle_note": "writer cycle sampling (sampler unavailable for this device)"}


def psi_val(text, kind, key):
    for line in (text or "").splitlines():
        if line.startswith(kind):
            for part in line.split():
                if part.startswith(key + "="):
                    try:
                        return float(part.split("=", 1)[1])
                    except ValueError:
                        return None
    return None


def gpu_vitals(dv, job):
    """Temperature (15 s sampler first, else the writer's probe), VRAM, power, and the running job's
    cap / cgroup peak / progress, for the capacity board and the 30-minute report."""
    g = dv.get("gpu") or {}
    t, src = dv.get("sampler_temperature_c"), "m06-gpu-sampler 15 s"
    if t is None:
        t, src = dv.get("temperature_c"), "writer probe"
    return {"temperature_c": t, "temperature_source": src,
            "temperature_max_since_sampler_start_c": dv.get("sampler_temperature_max_since_start_c"),
            "vram_used_mib": dv.get("sampler_vram_used_mib") if dv.get("sampler_vram_used_mib") is not None else g.get("vram_used_mib"),
            "vram_total_mib": dv.get("sampler_vram_total_mib") if dv.get("sampler_vram_total_mib") is not None else g.get("vram_total_mib"),
            "power_w": dv.get("sampler_power_w") if dv.get("sampler_power_w") is not None else g.get("power_w"),
            "job_cap_bytes": (job or {}).get("cap_bytes"), "job_peak_bytes": (job or {}).get("cgroup_peak_bytes"),
            "job_progress": (job or {}).get("progress"), "job_phase": (job or {}).get("phase"),
            "job_stage": (job or {}).get("stage"), "job_eta": (job or {}).get("eta")}


def capacity_board(reg, probes, devices, jobs, camps):
    """Owner orders rev 1 §6: every agent, CPU and GPU with current work, NEXT PREPARED work, and the
    measurable cause when free.  IDLE without an assigned successor is flagged ORCHESTRATION_DEFECT."""
    nxt = reg.get("next_prepared", {})            # lane -> text
    owner_of = reg.get("gpu_owner_lane", {})       # uuid -> lane
    agents = {L["lane"]: L for L in reg.get("lanes", [])}
    run = [j for j in jobs if j["state"] == "running"]
    queued = [j for j in jobs if j["state"] == "queued"]
    rows = []
    for dv in devices:
        if not str(dv.get("device", "")).startswith("GPU-"):
            continue
        uuid, role = dv["device"], dv["host_alias"]
        lane = owner_of.get(uuid)
        dev_cgs = {g.get("cgroup") for g in (dv.get("gpu") or {}).get("compute_processes", []) if g.get("cgroup")}
        cur = [j for j in run if j.get("host_alias") == role and j.get("cgroup") in dev_cgs]
        if role == "coordinator":
            rows.append({"lane": "-", "agent": "-", "host_alias": role, "resource": uuid, "name": dv.get("name"),
                         "current_job": cur[0]["id"] if cur else None, "heartbeat": None,
                         **gpu_vitals(dv, cur[0] if cur else None),
                         "next_prepared": None, "state": "RESERVED_FOR_DESKTOP" if not cur else "RUNNING",
                         "cause_if_idle": "owner's desktop GPU: never batch work"})
            continue
        q = [j for j in queued if j.get("host_alias") == role and j.get("gpu_request")]
        free = ((probes.get(role) or {}).get("admission") or {}).get("host_free_for_new_bytes")
        cause = None
        if not cur:
            if q:
                cause = f"queued for admission: {q[0]['id']} needs {q[0].get('declared_mem')}, host free-for-new {(free or 0) / 1e9:.2f} GB"
            elif lane and nxt.get(lane):
                cause = f"between jobs; next prepared by {lane}: {nxt[lane]}"
            else:
                cause = "ORCHESTRATION_DEFECT: idle with no assigned successor"
        rows.append({"lane": lane or (cur[0].get("lane") if cur else None), "agent": ", ".join(agents.get(lane or "", {}).get("agents", [])) or None,
                     "host_alias": role, "resource": uuid, "name": dv.get("name"),
                     "current_job": cur[0]["id"] if cur else None,
                     "heartbeat": (cur[0].get("heartbeat_status"), cur[0].get("heartbeat_age_s")) if cur else None,
                     "next_prepared": (q[0]["id"] if q else nxt.get(lane or "")),
                     "state": "RUNNING" if cur else "IDLE", "idle_seconds": dv.get("gpu_idle_current_seconds"),
                     **gpu_vitals(dv, cur[0] if cur else None),
                     "cause_if_idle": cause})
    for role, p in probes.items():
        if "error" in p:
            continue
        cpu_jobs = [j for j in run if j.get("host_alias") == role and not j.get("uses_gpu") and not j["id"].startswith("m06-")]
        rows.append({"lane": ", ".join(sorted({j.get("lane") or "?" for j in cpu_jobs})) or None, "agent": None,
                     "host_alias": role, "resource": f"CPU x{p.get('ncpu')} (load {' '.join(p.get('loadavg') or [])})",
                     "current_job": [f"{j['id']} [{j.get('lane')}] cap {(j.get('cap_bytes') or 0) / 1e9:.2f} GB / peak {(j.get('cgroup_peak_bytes') or 0) / 1e9:.2f} GB" for j in cpu_jobs],
                     "heartbeat": [(j["id"], j.get("heartbeat_status")) for j in cpu_jobs],
                     "next_prepared": [j["id"] for j in queued if j.get("host_alias") == role and not j.get("gpu_request")],
                     "state": ("DESKTOP_RESERVED_MARGIN" if role == "coordinator" else ("RUNNING" if cpu_jobs else "IDLE")),
                     "cause_if_idle": (None if cpu_jobs or role == "coordinator" else "no CPU job running or queued on this host")})
    for L in reg.get("lanes", []):
        mine = [j for j in run if j.get("lane") == L["lane"] and not j["id"].startswith("m06-")]
        rows.append({"lane": L["lane"], "agent": ", ".join(L.get("agents", [])), "host_alias": None, "resource": "AGENT",
                     "current_job": [j["id"] for j in mine], "heartbeat": [(j["id"], j.get("heartbeat_status")) for j in mine],
                     "next_prepared": nxt.get(L["lane"]), "state": "RUNNING" if mine else ("HOLD" if "HOLD" in str(L.get("declared_state")) else "NO_JOB"),
                     "cause_if_idle": (None if mine else (L.get("declared_state") if "HOLD" in str(L.get("declared_state")) else
                                       ("next prepared: " + nxt[L["lane"]] if nxt.get(L["lane"]) else "ORCHESTRATION_DEFECT: no job and no prepared successor")))})
    return rows


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


def read_lane_queue(q, hosts):
    """Read-only sqlite read of a lane queue, locally or on a worker through its own python (the
    workers have no sqlite3 CLI; the registry names the interpreter)."""
    py = q.get("python", "python3")
    role = q.get("host_role", "coordinator")
    try:
        if role == "coordinator":
            r = subprocess.run([sys.executable, "-c", QUEUE_READER, os.path.expanduser(q["path"])],
                               capture_output=True, text=True, timeout=30)
        else:
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", hosts[role],
                                f"{py} - {q['path']}"], input=QUEUE_READER, capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout)
    except Exception as e:
        return {"error": type(e).__name__}


def lane_queues(reg, hosts, now):
    out = {}
    for q in reg.get("lane_queues", []):
        raw = read_lane_queue(q, hosts)
        if "error" in raw:
            out[q["name"]] = {"queue": q["name"], "error": raw["error"], "host_role": q.get("host_role")}
            continue
        out[q["name"]] = {**obs.queue_summary(q["name"], raw, now, q.get("parallel_hosts", 1), q.get("planned_total"),
                                               tuple(q.get("excluded_statuses", []))),
                          "host_role": q.get("host_role"), "lane": q.get("lane"), "note": q.get("note")}
        e = out[q["name"]]["eta"]
        if e.get("first_estimable_event") == "nothing pending in this queue" and q.get("pending_event"):
            e["first_estimable_event"] = q["pending_event"]
    return out


def branch_tips(rows):
    tips = {}
    for L in rows:
        for b in (L.get("observe") or {}).get("branches", []):
            key = f"{b['repo']}:{b['branch']}"
            if key in tips:
                continue
            path = os.path.join(GITHUB, b["repo"])
            for ref in (b["branch"], "origin/" + b["branch"]):
                try:
                    r = subprocess.run(["git", "-C", path, "log", "-1", "--format=%ct %H", ref],
                                       capture_output=True, text=True, timeout=10)
                    ct, sha = r.stdout.split()
                    if key not in tips or int(ct) > tips[key][0]:
                        tips[key] = (int(ct), sha)
                except Exception:
                    pass
    return tips


def results_present(reg):
    """Declared result commits/artifacts that exist on this host: {id: evidence path}."""
    out = {}
    for rid, r in (reg.get("lane_results") or {}).items():
        if r.get("commit"):
            try:
                ok = subprocess.run(["git", "-C", os.path.join(GITHUB, r["repo"]), "cat-file", "-e", r["commit"] + "^{commit}"],
                                    capture_output=True, timeout=10).returncode == 0
            except Exception:
                ok = False
            out[rid] = f"{r['repo']}@{r['commit']}" if ok else None
        elif r.get("path"):
            out[rid] = r["path"] if os.path.exists(os.path.expanduser(r["path"])) else None
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
                "cap_vs_observed_peak": (round(L["cap_bytes"] / L["cgroup_peak"], 2) if L.get("cap_bytes") and L.get("cgroup_peak") else None),
                "uses_gpu": any(g.get("cgroup") == L.get("cgroup") for g in p.get("gpu_procs", [])),
                "cgroup": L.get("cgroup"),
                "heartbeat_at": iso(hb_at), "heartbeat_path": (hb or {}).get("path", "").replace(HOME, "~") or None,
                "heartbeat_stale": (hb_at is not None and now - hb_at > HEARTBEAT_STALE_S),
                "heartbeat_status": ("NONE" if hb_at is None else
                                     "NO_PROGRESS" if now - hb_at > 120 else "OK"),
                "heartbeat_age_s": (int(now - hb_at) if hb_at is not None else None),
                "last_advance": {k: ((hb or {}).get("last") or {}).get(k) for k in ("stage", "progress", "fit", "resume_point", "resources")
                                 if ((hb or {}).get("last") or {}).get(k) is not None},
                "progress": {"unit": hint.get("unit", "unknown"), "completed": done, "total": hint.get("total")},
                "eta": eta_from_history(hb, hint.get("total"), now),
                "override": overrides.get(L["name"])})
        for Q in p.get("launchers_without_lease", []):
            qh = next((v for k, v in hints.items() if Q["name"] and re.fullmatch(k, Q["name"])), {})
            jobs.append({"id": Q["name"] or f"unnamed-launcher-{Q['pid']}", "lane": qh.get("lane") or ("unknown" if not Q["name"] else
                         (LANE_RE.search(Q["name"] + "-").group(1).upper() if LANE_RE.search(Q["name"] + "-") else "unknown")),
                         "state": "queued", "phase": "queued", "stage": "waiting for admission",
                         "producer_commit": "unknown", "host_alias": role, "launcher_pid": Q["pid"],
                         "declared_mem": Q["mem"], "declared_wall": Q["wall"], "queue_wait_s": Q["wait"],
                         "gpu_request": Q.get("gpu_request"),
                         "waiting_s": Q["elapsed_s"], "command_head": Q["command_head"], "heartbeat_at": None,
                         "progress": {"unit": "unknown", "completed": None, "total": None},
                         "eta": {"earliest": None, "latest": None, "basis": "not_estimable",
                                 "assumptions": ["admission order depends on live leases releasing"]}})
        for u in p.get("unleased_scopes", []):
            unparsed.append({**u, "host_alias": role, "classification": "UNLEASED_SCOPE (in crispdm-batch.slice without a crispdm-run lease)"})
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
            util = num(f[3])
            st_ = IDLE_STATE.setdefault(uuid, {"since": None, "events": [], "last_t": None})
            t_obs = time.time()
            if st_["last_t"] is not None and util is not None and util == 0:
                st_["events"].append((t_obs, t_obs - st_["last_t"]))
            st_["events"] = [e for e in st_["events"] if e[0] > t_obs - 86400]
            if util is not None and util == 0:
                st_["since"] = st_["since"] or t_obs
            elif util is not None:
                st_["since"] = None
            st_["last_t"] = t_obs
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
                "next_task": reg.get("slot_plan", {}).get(uuid),
                **sampler_idle(uuid, IDLE_STATE)})
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
    for role in ("worker_a", "worker_b"):
        wa = probes.get(role, {})
        if "error" in wa or not SLAB_SERIES:
            continue
        path = SLAB_SERIES if role == "worker_a" else SLAB_SERIES.replace("worker_a_", "worker_b_")
        with open(path, "a") as fh:
            fh.write(json.dumps({"at": iso(wa["now"]), "boot_id": wa.get("boot_id"), "uptime_s": wa.get("uptime_s"),
                                 "SUnreclaim_kB": (wa.get("mem") or {}).get("SUnreclaim", 0) // 1024,
                                 "gpu_reinit_this_boot": wa.get("gpu_reinit_this_boot"),
                                 "SwapFree_B": (wa.get("mem") or {}).get("SwapFree"),
                                 "MemAvailable_B": (wa.get("mem") or {}).get("MemAvailable")}) + "\n")
    quotas = {role: {**{k: (p.get("admission") or {}).get(k) for k in
                        ("host_free_for_new_bytes", "held_unrealised_bytes", "desktop_reserve_bytes", "slice_memory_max",
                         "slice_memory_current", "slice_charged_bytes", "pressure_some_avg10")},
                     "mem_available_bytes": (p.get("mem") or {}).get("MemAvailable"),
                     "psi_some_avg10": psi_val(p.get("psi"), "some", "avg10"),
                     "psi_some_avg60": psi_val(p.get("psi"), "some", "avg60"),
                     "swap_free_bytes": (p.get("mem") or {}).get("SwapFree"),
                     "swap_total_bytes": (p.get("mem") or {}).get("SwapTotal"),
                     "unreclaimable_slab_bytes": (p.get("mem") or {}).get("SUnreclaim"),
                     "disk_home_free_bytes": (p.get("disk_home") or {}).get("free_bytes"),
                     "disk_home_total_bytes": (p.get("disk_home") or {}).get("total_bytes")}
              for role, p in probes.items() if "error" not in p}
    return {"schema": "modular.program.status.v1", "observed_at": iso(now),
            "plan_revision": reg.get("plan_revision"), "plan_links": reg.get("plan_links", {}),
            "writer": {"script": "tools/m06_status_writer.py + tools/m06_fleet_probe.py", "version": 3, "pid": os.getpid()},
            "agents": obs.observe_agents(reg.get("agents", []), (lanes_obs := obs.lane_rows(
                reg.get("lanes", []), jobs, (lq := lane_queues(reg, hosts, now)), branch_tips(reg.get("lanes", [])),
                results_present(reg), now))),
            "lanes": lanes_obs, "lane_queues": lq,
            "state_semantics": "declared_state is the registry's; observed_state comes only from process, lease, heartbeat, "
                               "queue or a branch tip after assignment; STALE_DECLARATION when nothing current is observed",
            "devices": devices, "jobs": jobs,
            "campaigns": (camps := campaign_progress(reg, now)),
            "coverage": coverage_block(reg, camps),
            "capacity_board": capacity_board(reg, probes, devices, jobs, camps),
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
    ipv4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")   # ssh mux names carry addresses
    def go(x):
        if isinstance(x, str):
            return ipv4.sub("<ip>", pat.sub("<host>", x) if pat.pattern else x)
        if isinstance(x, list):
            return [go(v) for v in x]
        if isinstance(x, dict):
            return {go(k): go(v) for k, v in x.items()}
        return x
    return go(obj)


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
    idle_path = os.path.join(os.path.expanduser("~/.local/state/m06"), "gpu_idle_state.json")
    try:
        IDLE_STATE.update({k: {**v, "events": [tuple(e) for e in v["events"]]} for k, v in json.load(open(idle_path)).items()})
    except Exception:
        pass
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
        try:
            atomic_write(idle_path, IDLE_STATE)
        except Exception:
            pass
        atomic_write(os.path.join(a.out_dir, "HEARTBEAT.json"),
                     {"at": iso(now), "pid": os.getpid(), "status_written_at": iso(last_write), "job_signature": sig})
        print(iso(now), "status" if now == last_write else "poll", "changed" if changed else "same", sig[:300], flush=True)
        last_sig = sig
        if a.once:
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
