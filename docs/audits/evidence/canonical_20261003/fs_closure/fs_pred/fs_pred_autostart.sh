#!/bin/bash
# FS-PRED pilot -> production handoff, unattended (worker_b). Runs as the transient unit fs-pred-autostart.
#  1. waits until no pilot launcher (crispdm-run -n fs-pred-pilot-cell) and no pilot scope is alive;
#  2. reads every retained lease of the pilot cell and takes the largest cgroup tree peak;
#  3. refuses to size production if any pilot incident was memory-caused (own cgroup OOM/MemoryMax):
#     writes BLOCKED_CAP_UNMEASURED and stops; a host-wide PRESSURE_STOP with zero cgroup pressure is not that;
#  4. cap = max(4G, 1.25 x peak), writes peak_evidence.json (cgroup scope) and runner.env;
#  5. links + enables + starts fs-pred-runner.service (sequential per target, resumable; exit 75 stops it).
set -uo pipefail
ROOT="$HOME/.local/state/canonical_20261003/fs_pred"
ADM="$HOME/.local/state/crispdm/admission"
LOG="$ROOT/logs/autostart.log"
INC="$ROOT/INCIDENTS.jsonl"
mkdir -p "$ROOT/logs"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "autostart waiting for the pilot launcher to exit"
while pgrep -f '[c]rispdm-run .* -n fs-pred-pilot-cell' >/dev/null 2>&1 \
   || systemctl --user list-units --no-pager --plain 2>/dev/null | grep -q 'crispdm-fs-pred-pilot-cell'; do
  sleep 60
done
log "pilot launcher gone; evaluating retained records"
python3 - "$ROOT" "$ADM" "$INC" <<'PY'
import glob, json, math, os, sys, time
root, adm, inc = sys.argv[1:4]
retained = sorted(glob.glob(os.path.join(adm, "retained", "fs-pred-pilot-cell-*.json")))
incidents = sorted(glob.glob(os.path.join(adm, "incidents", "fs-pred-pilot-cell-*.json")))
peaks, sources = [], []
for path in retained:
    doc = json.load(open(path))
    peak = doc.get("observed_peak_bytes")
    if peak:
        peaks.append(int(peak)); sources.append({"record": os.path.basename(path), "observed_peak_bytes": int(peak), "reclaim_cause": doc.get("reclaim_cause")})
memory_caused = []
for path in incidents:
    doc = json.load(open(path))
    cause = str(doc.get("exit_cause", "")).upper()
    cg = max([float(s.get("cgroup_some_avg10", 0.0)) for s in doc.get("samples", [])] or [0.0])
    if any(tok in cause for tok in ("OOM", "MEMORYMAX", "MEMORY_MAX", "CGROUP")) or ("PRESSURE" in cause and cg > 25.0):
        memory_caused.append({"record": os.path.basename(path), "exit_cause": cause, "max_cgroup_some_avg10": cg})
cells = os.path.join(root, "out", "cells", "Y_s_1h", "inner_2019")
records = [f for f in glob.glob(os.path.join(cells, "*.json")) if not f.endswith("population.json")]
complete = sum(1 for f in records if json.load(open(f)).get("disposition") == "COMPLETE")
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def incident(**kw):
    with open(inc, "a") as fh:
        fh.write(json.dumps(dict(schema="fs_pred_incident.v1", utc=now, host_role="worker_b", **kw), sort_keys=True) + "\n")
if not peaks:
    incident(stage="autostart", verdict="BLOCKED_CAP_UNMEASURED", reason="no retained pilot lease carries an observed peak")
    open(os.path.join(root, "BLOCKED_CAP_UNMEASURED"), "w").write(now + "\n"); sys.exit(3)
if memory_caused:
    incident(stage="autostart", verdict="BLOCKED_CAP_UNMEASURED", reason="pilot incident was memory-caused in its own cgroup", incidents=memory_caused, peaks=sources)
    open(os.path.join(root, "BLOCKED_CAP_UNMEASURED"), "w").write(now + "\n"); sys.exit(3)
peak = max(peaks)
cap_mib = max(4096, int(math.ceil(1.25 * peak / 2**20)))
evidence = {"schema": "fs_pred_peak_evidence.v1", "peak_scope": "cgroup", "peak_bytes": peak,
            "rule": "max observed cgroup peak over every retained pilot lease; cap = max(4G, 1.25 x peak)", "sources": sources, "written_utc": now}
ev_path = os.path.join(root, "peak_evidence.json")
json.dump(evidence, open(ev_path, "w"), indent=1, sort_keys=True)
with open(os.path.join(root, "runner.env"), "w") as fh:
    fh.write(f"FS_PRED_CAP={cap_mib}M\nFS_PRED_PEAK_EVIDENCE={ev_path}\nFS_PRED_PEAK_BYTES={peak}\n")
state = "pilot_complete" if complete >= 14 else f"pilot_incomplete_{complete}_of_14_methods (runner resumes it)"
incident(stage="autostart", verdict="PRODUCTION_STARTED", pilot_state=state, peak_bytes=peak, cap=f"{cap_mib}M", incidents_seen=[os.path.basename(p) for p in incidents])
print(f"peak={peak} cap={cap_mib}M state={state}")
PY
rc=$?
if [ "$rc" != 0 ]; then log "BLOCKED_CAP_UNMEASURED (rc=$rc); production not started"; exit 3; fi
log "sizing done: $(cat "$ROOT/runner.env" | tr '\n' ' ')"
mkdir -p "$HOME/.config/systemd/user"
ln -sf "$ROOT/fs-pred-runner.service" "$HOME/.config/systemd/user/fs-pred-runner.service"
systemctl --user daemon-reload
systemctl --user enable fs-pred-runner.service >> "$LOG" 2>&1
systemctl --user start fs-pred-runner.service >> "$LOG" 2>&1
log "fs-pred-runner.service started: $(systemctl --user is-active fs-pred-runner.service)"
