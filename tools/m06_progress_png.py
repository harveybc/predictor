#!/usr/bin/env python3
"""PROGRESS.png in the owner's continuation-order format (e9d689e6), generated from artifacts only.

Inputs: STATUS.json (capacity board, campaigns), registry.json (critical_path, lanes), the method
record's current_state_20261001 block (front tips, eligibility), and RESULTS/corrected_v2_config_summary.json.
Panels: newest result with its naive and class; critical path 1-7; fronts A-I; resources per GPU with
temperature; campaign queues with ETA or the missing measurement; strategy/deployment eligibility.
No weighted completion percentage is drawn."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
COL = {"RUNNING": "#2a78d6", "IDLE": "#e34948", "RESERVED_FOR_DESKTOP": "#b4b2aa", "OPEN": "#eda100",
       "COMPLETE": "#1baf7a", "NOT_STARTED": "#b4b2aa", "BLOCKED_ON_STEP_4": "#e34948", "HOLD": "#b4b2aa"}
FRONT_LANE = {"A": "A2", "B": "B2", "C": "C2", "D": "D2", "D/F2": "F2", "E": "E2", "F": "F", "G": "G2", "H": "H1", "I": "I1"}


def load(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def render(status, registry, method, v2, out_path):
    st, reg, ms = load(status, {}), load(registry, {}), load(method, {})
    cur = ms.get("current_state_20261001", {})
    v2d = load(v2, {})
    fig = plt.figure(figsize=(14, 11), facecolor=SURF)
    ax = fig.add_axes([0.015, 0.02, 0.97, 0.95]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    y = 0.99

    def line(txt, size=7.4, color=INK, x=0.012, step=0.019, mono=True, bold=False):
        nonlocal y
        ax.text(x, y, txt, fontsize=size, color=color, va="top", family="monospace" if mono else None,
                weight="bold" if bold else None)
        y -= step

    def box(state):
        ax.add_patch(plt.Rectangle((0, y - 0.013), 0.008, 0.013, color=COL.get(str(state).split()[0], "#b4b2aa")))

    line(f"Program progress, observed {st.get('observed_at')} UTC (generated from STATUS, registry, method record; no weighted %)",
         10.5, mono=False, step=0.03)
    line("1. Newest result", 9.5, mono=False, bold=True, step=0.024)
    if v2d.get("configs"):
        c0 = v2d["configs"][0]; nv = v2d["naive_same_rows"]
        line(f"{c0['label']} ({c0['config_id']}): validation MAE_z {c0['mean_mae_z']:.6f} (sd {c0['sd']:.6f}, {c0['n']} seeds) "
             f"vs same-row 24 h seasonal {nv['seasonal_24h']:.6f} -> skill {c0['skill_vs_seasonal24_same_rows']:+.4f}; "
             f"persistence {nv['persistence']:.6f} -> {c0['skill_vs_persistence_same_rows']:+.4f}")
        line(f"population: {v2d['population']}", color=INK2)
        line(f"class: {v2d['evidence_class']}; {v2d['comparability']}"[:200], color=INK2, step=0.028)
    line("2. Critical path (owner order e9d689e6)", 9.5, mono=False, bold=True, step=0.024)
    steps = (reg.get("critical_path") or {}).get("steps", [])
    done = sum(1 for s in steps if s["state"] == "COMPLETE")
    line(f"completed {done}/{len(steps)}", color=INK2)
    for s in steps:
        box(s["state"])
        line(f"{s['n']} {s['name'][:52]:<52} {s['state'][:34]:<34} next: {s['next'][:78]}")
    y -= 0.01
    line("3. Fronts A-I (tips read from git; IMPLEMENTED = code at the tip, not a result)", 9.5, mono=False, bold=True, step=0.024)
    lanes = {L["lane"]: L for L in st.get("lanes", [])}
    board_agents = {r.get("lane"): r for r in st.get("capacity_board", []) if r.get("resource") == "AGENT"}
    seen = {}
    for f in cur.get("fronts", []):
        seen.setdefault((f["front"], f["owner"]), []).append(f"{f['repo']}@{(f['tip'] or '')[:8]}")
    for (fr, owner), tips in seen.items():
        ln = FRONT_LANE.get(fr, fr); L = lanes.get(ln, {}); b = board_agents.get(ln, {})
        jobs = b.get("current_job") or []
        state = b.get("state") or L.get("state", "?")
        box(state)
        line(f"{fr:<5}{owner[:28]:<28} {', '.join(tips)[:52]:<52} {str(state)[:10]:<10} {', '.join(jobs)[:60] if isinstance(jobs, list) else jobs}")
    box("RUNNING"); line(f"{'F':<5}{'M06 evidence/resources':<28} {'predictor m06 branch':<52} {'RUNNING':<10} writer, watcher, sampler")
    y -= 0.01
    line("4. Resources per GPU (temperature from the 15 s sampler)", 9.5, mono=False, bold=True, step=0.024)
    for r in st.get("capacity_board", []):
        if not str(r.get("resource", "")).startswith("GPU-"):
            continue
        box(r.get("state"))
        t = r.get("temperature_c"); vu, vt = r.get("vram_used_mib"), r.get("vram_total_mib")
        line(f"{r['host_alias']:<11} {r['resource'][:20]:<20} {(r.get('name') or '').replace('NVIDIA GeForce ', '')[:22]:<22} "
             f"{r.get('state', '')[:20]:<20} {('%.0f C' % t) if t is not None else '? C':>5}  VRAM {vu or 0:>6.0f}/{vt or 0:.0f}  "
             f"job {str(r.get('current_job') or '-')[:26]:<26} next {str(r.get('next_prepared') or '-')[:44]}")
    y -= 0.01
    line("5-6. Queues and ETA (versioned denominators; ETA interval or the missing measurement)", 9.5, mono=False, bold=True, step=0.024)
    for c in st.get("campaigns", []):
        e = c.get("eta") or {}
        eta = (f"ETA {e['earliest'][11:16]}-{e['latest'][11:16]}Z" if e.get("earliest") and e.get("latest")
               else "ETA n/a: " + ("; ".join(e.get("assumptions") or ["-"]))[:90])
        line(f"{c['campaign'][:46]:<46} {json.dumps(c.get('status_counts'))[:70]:<70} {eta}"[:205])
    y -= 0.01
    line("Eligibility for strategy and deployment", 9.5, mono=False, bold=True, step=0.024)
    for k, v in (cur.get("eligibility_for_strategy_and_deployment") or {}).items():
        line(f"{k:<24} {v}")
    fig.savefig(out_path + ".tmp.png", dpi=100, facecolor=SURF)
    os.replace(out_path + ".tmp.png", out_path)


if __name__ == "__main__":
    render(*sys.argv[1:6])
