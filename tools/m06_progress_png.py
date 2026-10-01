#!/usr/bin/env python3
"""PROGRESS.png generated from STATUS.json only (writer v2 fields).

Panels: (1) devices by role/UUID: state, temperature, clock, throttle, VRAM, host RAM, and the job
on it; (2) jobs with phase, progress with its denominator, ETA window and basis (or the missing
measurement), heartbeat age; (3) lanes A-F, measured results, open owner decisions.
No weighted completion percentage is drawn: tests and prototypes are not scientific progress.
"""
import datetime as dt
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STATE_COL = {"running": "#2a78d6", "idle": "#b4b2aa", "unknown": "#e34948"}
PHASE_COL = {"fitting": "#2a78d6", "building": "#eda100", "validating": "#1baf7a", "transferring": "#4a3aa7",
             "queued": "#b4b2aa", "failed": "#e34948"}


def T(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc) if s else None


def gib(b):
    return f"{b / 2**30:.1f}" if isinstance(b, (int, float)) else "?"


def main(status_path, out_path):
    st = json.load(open(status_path))
    now = T(st["observed_at"])
    fig = plt.figure(figsize=(13, 10), facecolor=SURF)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.93])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y = 0.985
    ax.text(0, y, f"Modular program status, plan {st.get('plan_revision')}, observed {st['observed_at']} "
                  f"(generated from STATUS.json; no weighted percentage)", fontsize=11, color=INK, va="top")
    y -= 0.04
    ax.text(0, y, "Devices (role / GPU UUID)", fontsize=10, color=INK, weight="bold", va="top")
    y -= 0.03
    for d in st["devices"]:
        g = d.get("gpu") or {}
        h = (d.get("host") or {}).get("mem") or {}
        col = STATE_COL.get(d["state"], "#b4b2aa")
        ax.add_patch(plt.Rectangle((0, y - 0.016), 0.012, 0.016, color=col))
        thr = ",".join(t for t in g.get("throttle_reasons", []) if t != "gpu_idle") or "none"
        ax.text(0.018, y, f"{d['host_alias']:<12} {d.get('name', '').replace('NVIDIA GeForce ', '')[:24]:<24} {d['device'][:16]}  {d['state']:<8} "
                          f"{d.get('temperature_c') or 0:>3.0f}C  SM {g.get('sm_clock_mhz') or 0:>5.0f}/{g.get('sm_clock_max_mhz') or 0:.0f} MHz  "
                          f"thr {thr:<20} VRAM {g.get('vram_used_mib') or 0:>6.0f}/{g.get('vram_total_mib') or 0:.0f} MiB  "
                          f"host avail {gib(h.get('MemAvailable'))}/{gib(h.get('MemTotal'))} GiB",
                fontsize=7.6, family="monospace", color=INK, va="top")
        y -= 0.022
        ax.text(0.03, y, f"job: {d.get('job_id') or '-'}; {d.get('reason')}"[:150], fontsize=7, family="monospace",
                color=INK2, va="top")
        y -= 0.024
    y -= 0.01
    ax.text(0, y, "Jobs (phase from probes; ETA = remaining x measured rate, or the missing measurement)",
            fontsize=10, color=INK, weight="bold", va="top")
    y -= 0.03
    for j in sorted(st["jobs"], key=lambda j: (j["state"] != "running", j["state"] != "queued", j["id"])):
        ph = j.get("phase") or j["state"]
        col = PHASE_COL.get(ph.split()[0].lower(), "#1baf7a" if j["state"] == "completed" else "#b4b2aa")
        ax.add_patch(plt.Rectangle((0, y - 0.014), 0.012, 0.014, color=col))
        p = j.get("progress") or {}
        prog = f"{p.get('completed')}/{p.get('total')} {p.get('unit')}" if p.get("completed") is not None else "-"
        e = j.get("eta") or {}
        if e.get("earliest"):
            eta = f"ETA {e['earliest'][11:16]}-{e['latest'][11:16]}Z ({e['basis']})"
        else:
            eta = "ETA n/a: " + ((e.get("assumptions") or ["-"])[0])[:60]
        hb = j.get("heartbeat_at")
        age = f"hb {int((now - T(hb)).total_seconds())}s ago" if hb and j["state"] == "running" else ""
        ax.text(0.018, y, f"{j['id'][:44]:<44} {str(j.get('host_alias') or ''):<11} {ph[:34]:<34} {prog:<20} {eta[:70]} {age}",
                fontsize=7.2, family="monospace", color=INK, va="top")
        y -= 0.021
    y -= 0.01
    ax.text(0, y, "Lanes", fontsize=10, color=INK, weight="bold", va="top")
    y -= 0.028
    for L in st.get("lanes", []):
        extra = L.get("first_deliverable") or L.get("tests") or ""
        ax.text(0.018, y, f"{L['lane']}  {L['state']:<32} {', '.join(L['agents'])[:52]:<52} {extra[:70]}",
                fontsize=7.2, family="monospace", color=INK, va="top")
        y -= 0.02
    y -= 0.01
    ax.text(0, y, "Measured results (classes as recorded; validation vs test kept apart)", fontsize=10, color=INK,
            weight="bold", va="top")
    y -= 0.028
    seen = set()
    for r in st.get("results", []):
        if not any(w in r.get("model", "") for w in ("mean", "incumbent")):
            continue
        ax.text(0.018, y, f"{r['task'][:52]} [{r['split']}]  {r['metric']} {r['value']:.6f}  naive {r.get('naive')}  "
                          f"lit {r['literature']['value']}  {r.get('class') or r['literature']['comparability']}  "
                          f"{r.get('note', '')[:40]}", fontsize=7.0, family="monospace", color=INK, va="top")
        y -= 0.02
    y -= 0.03
    ax.text(0, y, "Open owner decisions", fontsize=10, color=INK, weight="bold", va="top")
    y -= 0.028
    for n in [a for a in st.get("next_actions", []) if a.startswith("owner")][:6]:
        ax.text(0.018, y, n[:150], fontsize=7.2, family="monospace", color=INK, va="top")
        y -= 0.02
    fig.savefig(out_path + ".tmp.png", dpi=105, facecolor=SURF)
    os.replace(out_path + ".tmp.png", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
