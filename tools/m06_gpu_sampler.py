#!/usr/bin/env python3
"""M06 GPU sampler: every 15 s, query utilization on worker_a and worker_b (nvidia-smi only, one ssh
ControlMaster per host), keep consecutive-idle and 24 h idle counters per UUID, write gpu_idle.json
atomically, and append an ALERT line to events.log after 120 s of consecutive 0 % samples.
Host aliases come from ~/.config/m06/hosts.json; nothing host-named is written."""
import json, os, subprocess, time
HOME = os.path.expanduser("~")
hosts = json.load(open(HOME + "/.config/m06/hosts.json"))
OUT = HOME + "/.local/state/m06/gpu_idle.json"; EV = HOME + "/.local/state/m06/events.log"
STATUS = HOME + "/Documents/GitHub/.worktrees/predictor-m06-20260930/docs/audits/evidence/MODULAR_CAMPAIGN_20260930/STATUS.json"
PERIOD, THRESH = 15, 120
TEMP_ALERT_C = 83  # thermal alarm after 2 consecutive samples at or above this (owner thermal rule; report, never act)
try:
    state = json.load(open(OUT))
except Exception:
    state = {}
alerted = set(); hot = {}
def ts(): return time.strftime("%H:%M:%SZ", time.gmtime())
def context(uuid, role):
    try:
        d = json.load(open(STATUS))
    except Exception:
        return "queue n/a", "n/a"
    q = [j["id"] for j in d.get("jobs", []) if j.get("host_alias") == role and j["state"] == "queued"]
    camps = "; ".join(f"{c['campaign'][:28]} {c.get('status_counts')}" for c in d.get("campaigns", []) if "identity" in c and "status_counts" in c)
    nxt = next((x.get("next_task") for x in d.get("devices", []) if x.get("device") == uuid), None)
    return f"queued on host: {q or 'none'}; campaigns: {camps}", nxt or "UNASSIGNED"
while True:
    now = time.time()
    for role, alias in hosts.items():
        try:
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ControlMaster=auto",
                                "-o", f"ControlPath={HOME}/.local/state/m06/cm-%r@%h", "-o", "ControlPersist=120", alias,
                                "nvidia-smi --query-gpu=uuid,name,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits; echo APPS; nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader"],
                               capture_output=True, text=True, timeout=20)
            head, _, apps = r.stdout.partition("APPS")
            rows = [(l.split(", ") + ["", "", "", ""])[:7] for l in head.strip().splitlines() if l.count(",") >= 2]
            busy = {l.strip() for l in apps.strip().splitlines() if l.strip()}
        except Exception:
            rows = []
        for uuid, name, util, temp, mused, mtot, pw in rows:
            st = state.setdefault(uuid, {"role": role, "name": name, "since": None, "last_t": None, "events": []})
            try:
                u = float(util)
            except ValueError:
                continue
            # idle = 0 % utilization, or no compute process at all (a display/query can show a few %)
            idle = (u == 0) or (uuid not in busy)
            st["compute_processes"] = uuid in busy
            if st["last_t"] is not None and idle:
                st["events"].append([now, now - st["last_t"]])
            st["events"] = [e for e in st["events"] if e[0] > now - 86400]
            st["since"] = (st["since"] or now) if idle else None
            st["last_t"] = now; st["util"] = u
            def num(x):
                try: return float(x)
                except ValueError: return None
            st["temp_c"], st["mem_used_mib"], st["mem_total_mib"], st["power_w"] = num(temp), num(mused), num(mtot), num(pw)
            st["temp_max_since_start_c"] = max([t for t in (st.get("temp_max_since_start_c"), st["temp_c"]) if t is not None], default=None)
            if st["temp_c"] is not None and st["temp_c"] >= TEMP_ALERT_C:
                hot[uuid] = hot.get(uuid, 0) + 1
                if hot[uuid] == 2:
                    with open(EV, "a") as fh:
                        fh.write(f"{ts()} ALERT GPU HOT {role} {name} {uuid[:16]} {st['temp_c']:.0f} C (>= {TEMP_ALERT_C} C, 2 samples); util {u:.0f} %; report only\n")
            else:
                hot[uuid] = 0
            st["idle_current_s"] = int(now - st["since"]) if st["since"] else 0
            st["idle_24h_s"] = int(sum(e[1] for e in st["events"]))
            if st["idle_current_s"] >= THRESH and uuid not in alerted:
                qs, nxt = context(uuid, role)
                with open(EV, "a") as fh:
                    fh.write(f"{ts()} ALERT GPU IDLE {role} {name} {uuid[:16]} idle {st['idle_current_s']}s (>= {THRESH} s, 15 s sampling); {qs}; owner/next useful job: {nxt}\n")
                alerted.add(uuid)
            if not idle:
                alerted.discard(uuid)
    tmp = OUT + ".tmp"; json.dump(state, open(tmp, "w")); os.replace(tmp, OUT)
    time.sleep(max(1, PERIOD - (time.time() - now)))
