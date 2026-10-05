#!/bin/bash
# ---- FS-PRED status block (coordinator). Pulls progress.json, selector_sets.json and the per-cell records
# from worker_b (ionice idle), renders status/LANE_STATUS.json, method_walls.csv, primary_k24_sets.csv into the
# coordination worktree and commits its own paths at most every 30 min. Idempotent; safe to run every cycle.
set -uo pipefail
REPO="${FS_PRED_REPO:-$HOME/Documents/GitHub/.worktrees/predictor-canonical-exec-20261003}"
WORKER="${FS_PRED_WORKER:-dragon}"
STATE="$HOME/.local/state/canonical_20261003/fs_pred_status"; mkdir -p "$STATE/out"
DEST="$REPO/docs/audits/evidence/canonical_20261003/fs_closure/fs_pred"
PYTHON="${CRISPDM_PYTHON:-python3}"
ionice -c3 nice -n 19 rsync -q -a --delete --include='*/' --include='*.json' --exclude='*' \
  "$WORKER:.local/state/canonical_20261003/fs_pred/out/" "$STATE/out/" 2>>"$STATE/pull.err" || true
ionice -c3 rsync -q -a "$WORKER:.local/state/canonical_20261003/fs_pred/INCIDENTS.jsonl" "$STATE/INCIDENTS.worker_b.jsonl" 2>/dev/null || true
# worker probe: scope state, D-state pids, memory.events high (rising?), cgroup pressure, progress mtime, relaunch epoch
ssh -o BatchMode=yes -o ConnectTimeout=20 "$WORKER" 'python3 - <<PY
import json,os,glob,subprocess,time
root=os.path.expanduser("~/.local/state/canonical_20261003/fs_pred"); out={"probe_epoch":time.time()}
units=subprocess.run(["systemctl","--user","list-units","--no-pager","--plain"],capture_output=True,text=True).stdout
scopes=[l.split()[0] for l in units.splitlines() if l.strip().startswith("crispdm-fs-pred-")]
out["scope_active"]=bool(scopes); out["scopes"]=scopes
if scopes:
    cg=subprocess.run(["systemctl","--user","show","-p","ControlGroup","--value",scopes[0]],capture_output=True,text=True).stdout.strip()
    base="/sys/fs/cgroup"+cg
    try:
        ev=dict(l.split() for l in open(base+"/memory.events").read().split("\n") if l); out["memory_events_high"]=int(ev.get("high",0))
        out["cgroup_some_avg10"]=float(open(base+"/memory.pressure").readline().split()[1].split("=")[1])
        out["memory_peak"]=int(open(base+"/memory.peak").read()); out["memory_max"]=open(base+"/memory.max").read().strip()
        pids=open(base+"/cgroup.procs").read().split()
        out["d_state_pids"]=[p for p in pids if open(f"/proc/{p}/stat").read().split()[2]=="D"]
    except Exception as e: out["probe_error"]=str(e)
prev=os.path.join(root,"logs","probe_prev.json")
try: old=json.load(open(prev))
except Exception: old={}
out["high_events_rising"]=bool(out.get("memory_events_high",0)>old.get("memory_events_high",0)) if "memory_events_high" in out and old.get("scopes")==scopes else False
p=os.path.join(root,"out","progress.json"); out["progress_mtime_epoch"]=os.stat(p).st_mtime if os.path.exists(p) else None
r=os.path.join(root,"RELAUNCH_EPOCH"); out["relaunch_epoch"]=float(open(r).read()) if os.path.exists(r) else None
out["blocked"]=os.path.exists(os.path.join(root,"BLOCKED_CAP_UNMEASURED"))
json.dump(out,open(prev,"w")); print(json.dumps(out))
PY' > "$STATE/worker_probe.json.tmp" 2>>"$STATE/pull.err" && mv "$STATE/worker_probe.json.tmp" "$STATE/worker_probe.json" || true
CRISPDM_PYTHON="$PYTHON" "$HOME/.local/bin/crispdm-run" -m 1G -t 10m -n fs_pred_status -- \
  "$PYTHON" "$DEST/fs_pred_status.py" "$STATE/out" "$DEST/status" > "$STATE/last_cycle.json" 2> "$STATE/last_cycle.err" || true
[ -f "$STATE/INCIDENTS.worker_b.jsonl" ] && cp "$STATE/INCIDENTS.worker_b.jsonl" "$DEST/status/INCIDENTS.worker_b.jsonl"
STAMP="$STATE/last_commit_epoch"
if [ ! -f "$STAMP" ] || [ $(( $(date +%s) - $(cat "$STAMP") )) -ge 1800 ]; then
  ( cd "$REPO" && git add docs/audits/evidence/canonical_20261003/fs_closure/fs_pred/status 2>/dev/null \
    && git diff --cached --quiet || git commit -q -m "FS-PRED: automatic status refresh

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" ; git pull --no-rebase --no-edit -q 2>/dev/null; git push -q 2>/dev/null ) || true
  date +%s > "$STAMP"
fi
