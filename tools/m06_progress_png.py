#!/usr/bin/env python3
"""PROGRESS.png from STATUS.json (same evidence, nothing else).

One row per device: elapsed run time (solid), ETA window earliest->latest (light), completed
cells (solid, darker), queued-for-admission counts; dated events and milestones below; a
"now" rule.  No weighted completion percentage is drawn: tests and prototypes are not
measured scientific progress.
"""
import datetime as dt
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RUN, ETA, DONE, QUE = "#2a78d6", "#a9c8f0", "#1baf7a", "#b4b2aa"


def T(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc) if s else None


def main(status_path, out_path):
    st = json.load(open(status_path))
    now = T(st["observed_at"])
    devs = [d for d in st["devices"] if d.get("device", "").startswith("GPU-")]
    labels = [f"{d['host_alias']}  {d.get('name', '').replace('NVIDIA GeForce ', '')}\n{d['device'][:12]}  "
              f"{d.get('temperature_c') or 0:.0f} °C  {(d.get('gpu') or {}).get('sm_clock_mhz') or 0:.0f} MHz"
              f"{'  SW-thermal' if 'sw_thermal_slowdown' in (d.get('gpu') or {}).get('throttle_reasons', []) else ''}"
              for d in devs]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7.2), gridspec_kw={"height_ratios": [3, 1.6]},
                                  facecolor=SURF)
    for a in (ax, ax2):
        a.set_facecolor(SURF)
        for s in a.spines.values():
            s.set_visible(False)
    t0 = now - dt.timedelta(hours=2.2)
    t1 = now + dt.timedelta(hours=1.4)
    for i, d in enumerate(devs):
        y = len(devs) - 1 - i
        for j in st["jobs"]:
            if j.get("device") != d["device"]:
                continue
            if j["state"] == "running" and j.get("resources"):
                start = now - dt.timedelta(seconds=j["resources"]["elapsed_s"])
                ax.barh(y, now - start, left=start, height=0.42, color=RUN, zorder=3)
                e0, e1 = T(j["eta"]["earliest"]), T(j["eta"]["latest"])
                if e0 and e1:
                    ax.barh(y, e1 - now, left=now, height=0.42, color=ETA, zorder=2)
                    ax.plot([e0, e0], [y - 0.21, y + 0.21], color=INK2, lw=1.5, zorder=4)
                    ax.text(e1 + dt.timedelta(minutes=2), y,
                            f"{j['id'].replace('traffic_L96_', '')}  ~{j['progress']['completed']}/{j['progress']['total']} ep\n"
                            f"ETA {e0:%H:%M}–{e1:%H:%M}Z ({j['eta']['basis']})", va="center", fontsize=8, color=INK)
            elif j["state"] == "completed" and j.get("window"):
                a, b = T(j["window"][0]), T(j["window"][1])
                ax.barh(y, b - a, left=a, height=0.42, color=DONE, zorder=3)
                ax.text(a, y + 0.32, f"{j['id'].replace('traffic_L96_', '')} done {b:%H:%M}Z", fontsize=8, color=INK)
        if d["state"] == "idle":
            why = (d.get("reason") or "").removeprefix("idle: ")
            ax.text(now + dt.timedelta(minutes=3), y, ("idle — " + why)[:70], va="center",
                    fontsize=7.5, color=INK2)
    q = [j for j in st["jobs"] if j.get("stage") == "queued_for_admission"]
    if q:
        by = {}
        for j in q:
            by.setdefault(j["host_alias"], []).append(f"{j['id']} {j.get('declared_mem')}")
        ax.text(0.0, -0.13, "queued for admission — " + "; ".join(f"{h}: {', '.join(v)}" for h, v in by.items()),
                transform=ax.transAxes, fontsize=7.5, color=INK2, wrap=True)
    ax.axvline(now, color=INK, lw=1, zorder=5)
    ax.text(now, -0.62, f" now {now:%H:%M}Z", fontsize=8, color=INK)
    ax.set_yticks(range(len(devs)))
    ax.set_yticklabels(labels[::-1], fontsize=8, color=INK)
    ax.set_xlim(t0, t1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_title(f"Modular campaign — devices and jobs, {now:%Y-%m-%d} (UTC). Solid: elapsed; light: ETA window "
                 f"(tick = earliest); green: completed cell", fontsize=10, color=INK, loc="left")
    # events and milestones
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    lines = []
    for e in st.get("events", [])[-6:]:
        lines.append(f"{e['at'][11:16]}Z  [{e['host_alias']}] {e['event'][:100]}")
    lines.append("")
    for m in st.get("milestones", []):
        mark = {"completed": "[done]", "in_progress": "[in progress]", "not_started": "[not started]"}[m["state"]]
        lines.append(f"{mark:14s} {m['id']}: {m['completion_criterion'][:88]}")
    res = st.get("results", [])
    lines.append("")
    lines.append(f"results rows: {len(res)} measured "
                 f"({', '.join(sorted({r.get('cell', r['task'])[:26] for r in res}))}); synthetic checks and unit tests are NOT counted as scientific progress")
    ax2.text(0, 1, "\n".join(lines), va="top", fontsize=7.6, color=INK, family="monospace")
    fig.tight_layout()
    fig.savefig(out_path + ".tmp.png", dpi=110, facecolor=SURF)
    import os
    os.replace(out_path + ".tmp.png", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
