#!/usr/bin/env bash
# Host health and available-memory measurement for the phase 2/3 selection deployment (§E.1).
#
#   host_health.sh ROLE [PYTHON_CANDIDATE ...]
#
# Prints ONE JSON document on stdout and never changes anything on the host: it reads
# /proc/meminfo, the crispdm-batch.slice accounting, the admission ledger (live leases),
# CPU count and load, and the version of the numeric stack under every python interpreter
# given (the system python3 is always probed). GPUs are only READ through nvidia-smi so the
# receipt shows what is holding them; nothing is launched. The document names the host by
# ROLE only (coordinator / worker_a / worker_b): no host name, user or address is recorded,
# so the receipt can live in the repository. Runs unchanged over `ssh HOST bash -s -- ROLE`
# because it needs only bash, coreutils, systemd and a python3 (stdlib only).
set -uo pipefail
ROLE="${1:?role (coordinator|worker_a|worker_b)}"; shift || true
ADM_CANDIDATES=("${CRISPDM_ADMISSION_MODULE:-}" "$HOME/.local/libexec/crispdm/crispdm_admission.py")

slice_prop() { systemctl --user show crispdm-batch.slice -p "$1" --value 2>/dev/null || echo ""; }
SLICE_CUR="$(slice_prop MemoryCurrent)"; SLICE_HIGH="$(slice_prop MemoryHigh)"; SLICE_MAX="$(slice_prop MemoryMax)"
SLICE_ACTIVE="$(systemctl --user is-active crispdm-batch.slice 2>/dev/null || echo unknown)"
SCOPES="$(systemctl --user list-units --type=scope --no-legend --plain 2>/dev/null | awk '{print $1}' | grep '^crispdm-' | tr '\n' ' ')"
ADM_STATE=""
for c in "${ADM_CANDIDATES[@]}"; do
  if [[ -n "$c" && -f "$c" ]]; then ADM_STATE="$(python3 "$c" state 2>/dev/null)"; break; fi
done
GPU="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')"
CRISPDM_RUN="$(command -v crispdm-run 2>/dev/null || ls "$HOME/.local/bin/crispdm-run" 2>/dev/null || echo "")"

python3 - "$ROLE" "$SLICE_CUR" "$SLICE_HIGH" "$SLICE_MAX" "$SLICE_ACTIVE" "$SCOPES" "$ADM_STATE" "$GPU" "$CRISPDM_RUN" python3 "$@" <<'PY'
import json, os, subprocess, sys, time
role, s_cur, s_high, s_max, s_active, scopes, adm_state, gpu, crun, *pythons = sys.argv[1:]
def meminfo():
    out = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        out[k] = int(v.split()[0]) * 1024
    return out
def ival(x):
    try: return int(x)
    except (TypeError, ValueError): return None
mi = meminfo()
PROBE = ("import json,sys,platform\n"
         "r={'executable':sys.executable,'python':platform.python_version(),'packages':{}}\n"
         "for m in ('numpy','scipy','sklearn','duckdb','pandas','pyarrow'):\n"
         "  try:\n    mod=__import__(m); r['packages'][m]=getattr(mod,'__version__','?')\n"
         "  except Exception as e:\n    r['packages'][m]='MISSING:'+type(e).__name__\n"
         "print(json.dumps(r))")
envs = {}
seen = set()
for p in pythons:
    p = os.path.expanduser(p)
    if p in seen: continue
    seen.add(p)
    try:
        r = subprocess.run([p, "-c", PROBE], capture_output=True, text=True, timeout=60)
        envs[p] = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {"error": (r.stderr or "no output").strip()[-300:]}
    except Exception as e:
        envs[p] = {"error": f"{type(e).__name__}: {e}"}
try: adm = json.loads(adm_state) if adm_state else None
except ValueError: adm = {"unparsed": adm_state[-300:]}
leases = []
if adm and isinstance(adm.get("live"), list):
    for l in adm["live"]:
        if isinstance(l, dict):
            leases.append({k: l.get(k) for k in ("lease_id", "name", "label", "cap_bytes", "reserved_bytes", "slice", "armed", "unit") if k in l})
        else:
            leases.append(l)
gpus = []
for row in filter(None, gpu.split(";")):
    f = [x.strip() for x in row.split(",")]
    if len(f) >= 4:
        gpus.append({"index": ival(f[0]), "memory_used_mib": ival(f[1]), "memory_total_mib": ival(f[2]), "utilization_pct": ival(f[3])})
doc = {
    "schema": "fs_phase23_host_health.v1",
    "role": role,
    "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "cpu": {"cores": os.cpu_count(), "loadavg_1_5_15": list(os.getloadavg())},
    "memory": {"total_bytes": mi.get("MemTotal"), "available_bytes": mi.get("MemAvailable"), "free_bytes": mi.get("MemFree"),
               "cached_bytes": mi.get("Cached"), "slab_unreclaim_bytes": mi.get("SUnreclaim"), "swap_total_bytes": mi.get("SwapTotal"), "swap_free_bytes": mi.get("SwapFree")},
    "crispdm_batch_slice": {"active": s_active, "memory_current_bytes": ival(s_cur), "memory_high_bytes": ival(s_high), "memory_max_bytes": ival(s_max),
                            "scopes": sorted(filter(None, scopes.split()))},
    "admission": None if adm is None else {"host_free_for_new_bytes": adm.get("host_free_for_new_bytes"), "desktop_reserve_bytes": adm.get("desktop_reserve_bytes"),
                                           "mem_available_bytes": adm.get("mem_available_bytes"), "pressure_some_avg10": adm.get("pressure_some_avg10"),
                                           "pressure_full_avg10": adm.get("pressure_full_avg10"), "pressure_admit_max": adm.get("pressure_admit_max"),
                                           "live_lease_count": len(adm.get("live") or []), "live_leases": leases},
    "crispdm_run_installed": bool(crun),
    "gpus_read_only": gpus,
    "python_envs": envs,
}
text = json.dumps(doc, indent=1, sort_keys=True)
# roles only: the receipt must carry no private path, so the home directory is written as "~"
print(text.replace(os.path.expanduser("~"), "~"))
PY
