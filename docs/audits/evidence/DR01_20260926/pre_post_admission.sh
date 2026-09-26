#!/usr/bin/env bash
# DR01 PRE/POST: two requests of 8 GiB against ONE reading of 12 GiB available.
# Nothing is allocated, no process is started and real memory is never pressured: both sides run
# against a SIMULATED host.  PRE is the arithmetic the launcher carried until 2026-09-26; POST is
# tools/crispdm_admission.py.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd -- "$here/../../../.." && pwd)
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
GIB=$((1<<30))

cat > "$work/readings.json" <<EOF
{"mem_available_bytes": $((12*GIB)), "mem_total_bytes": $((32*GIB)),
 "slice_memory_max": $((14*GIB)), "slice_memory_current": 0, "pressure_some_avg10": 0.0,
 "alive": {}, "cgroup_current": {}, "cgroup_peak": {}}
EOF

echo "=== PRE: the rule the launcher carried until 2026-09-26 ==="
echo "    req <= MemAvailable - 3G, and req <= slice MemoryMax.  Nothing is written between the"
echo "    reading and the launch, so every caller sees the same 12 GiB."
pre() {
  req=$1
  avail=$((12*GIB)); reserve=$((3*GIB)); slice=$((14*GIB))
  if [ "$req" -gt $((avail - reserve)) ]; then echo "  request $1: REFUSED (host)"; return; fi
  if [ "$req" -gt "$slice" ]; then echo "  request $1: REFUSED (slice)"; return; fi
  echo "  request $1: ADMITTED"
}
pre $((8*GIB)); pre $((8*GIB))
echo "    -> 16 GiB admitted against 12 GiB available.  This is Musashi F1."

echo
echo "=== POST: tools/crispdm_admission.py, same simulated host ==="
export CRISPDM_ADMISSION_RESOURCES_JSON="$work/readings.json"
export CRISPDM_ADMISSION_DIR="$work/store"
export CRISPDM_ADMISSION_NOW=1790400000
for label in first second; do
  set +e
  out=$(python3 "$repo/tools/crispdm_admission.py" acquire -n "$label" -m 8G --label "$label")
  rc=$?
  set -e
  printf '  %-7s exit=%-3s %s\n' "$label" "$rc" \
    "$(printf '%s' "$out" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["verdict"], d["code"], "--", d["reason"])')"
done
echo "    live reservations after both calls:"
python3 "$repo/tools/crispdm_admission.py" state \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);print("     ", len(d["live"]), "lease(s);", "host_free_for_new_bytes =", d["host_free_for_new_bytes"])'
