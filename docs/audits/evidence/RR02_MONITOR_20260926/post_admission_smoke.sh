#!/usr/bin/env bash
# RR02 (order 2026-09-26): the post-admission monitor at the REAL wrapper boundary.
#
# What this does, and what it deliberately does not do.  The launcher, the transient scope, the
# cgroup and the child are all REAL.  The memory READINGS are simulated, from a JSON file: the
# order forbids validating this by exhausting a host, and a bounded synthetic child is sufficient
# for an integration smoke.  The child is `sleep`, which allocates nothing, so no host is put under
# pressure by this script and no OOM is provoked.  A private lease store is used, so the fleet's own
# store and its ledger are not touched.
#
#   case 1  calm host          -> the child runs to its own end, exit 0, nothing stopped
#   case 2  sustained pressure -> the monitor stops THIS scope only, before the host's oomd would act
#   case 3  one dip below the admission limit in the middle of sustained pressure -> still stopped:
#           a single sample is not a sustained recovery (the 52.18 -> 24.5 crossing the order names)
set -uo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TOOLS=$(cd -- "$here/../../../../tools" && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/rr02-smoke.XXXXXX")
trap 'rm -rf "$work"' EXIT

export CRISPDM_ADMISSION_MODULE="$TOOLS/crispdm_admission.py"
export CRISPDM_ADMISSION_DIR="$work/store"
export CRISPDM_ADMISSION_RESOURCES_JSON="$work/readings.json"

pressure() {
  cat > "$work/readings.json" <<JSON
{"mem_available_bytes": 34359738368, "mem_total_bytes": 68719476736,
 "slice_memory_max": null, "slice_memory_current": 0,
 "pressure_some_avg10": $1, "pressure_full_avg10": $1,
 "alive": {}, "cgroup_current": {}, "cgroup_peak": {}}
JSON
}

echo "=== policy in force (derived; see RETAINED_PRESSURE_SERIES.json) ==="
python3 "$TOOLS/crispdm_admission.py" policy

echo
echo "=== case 1: a calm host.  The child ends on its own terms. ==="
pressure 0.0
/usr/bin/time -f "  wall %es" bash "$TOOLS/crispdm-run" -m 256M -t 2m -n rr02calm -- sleep 8
echo "  exit=$?"
python3 "$TOOLS/crispdm_admission.py" state | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print("  live leases:", len(d["live"]), "| retained bodies:", d["retained_bodies"])'

echo
echo "=== case 2: admitted on a calm host, then pressure rises.  The real sequence: on the"
echo "===         previous boot the second scope was admitted at PSI 24.50 and the host went to 74.96. ==="
pressure 0.0
bash "$TOOLS/crispdm-run" -m 256M -t 5m -n rr02stop -- sleep 600 &
launcher=$!
sleep 3; pressure 60.0            # admitted first, as it was on the day; pressure rises afterwards
wait "$launcher"; echo "  exit=$? (the status the launcher's own timeout wrapper reports when the job is terminated; 124 on this coreutils)"
for f in $(compgen -G "$CRISPDM_ADMISSION_DIR/incidents/*.json" || true); do
  python3 - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("  incident:", d["exit_cause"], "|", d["rule"])
print("  detail  :", d["detail"])
print("  samples :", len(d["samples"]), "| scope_only:", d["scope_only"],
      "| partial evidence retained:", d["partial_evidence_retained"])
print("  acted on:", json.dumps(d.get("acted", "<not yet written when this file was read>"), sort_keys=True))
PY
done
python3 "$TOOLS/crispdm_admission.py" state | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print("  live leases after:", len(d["live"]))'

echo
echo "=== case 3: pressure rises, dips below the admission limit, and returns. ==="
rm -f "$CRISPDM_ADMISSION_DIR"/incidents/*.json
pressure 0.0
bash "$TOOLS/crispdm-run" -m 256M -t 5m -n rr02dip -- sleep 600 &
launcher=$!
sleep 3;  pressure 60.0
sleep 12; pressure 24.5           # the crossing the order names: one sample below 25.00
sleep 6;  pressure 60.0           # and pressure returns, as it did in the retained series
wait "$launcher"; echo "  exit=$? (still stopped: one dip is not a recovery)"
for f in $(compgen -G "$CRISPDM_ADMISSION_DIR/incidents/*.json" || true); do
  python3 - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
low = [s for s in d["samples"] if s["host_some_avg10"] <= 25.0]
print("  incident:", d["exit_cause"], "| samples at or below 25.00 during the run:", len(low))
print("  recoveries confirmed:", d["recoveries_confirmed"], "(a confirmed recovery needs 120 s)")
PY
done
echo
echo "=== nothing outside the smoke's own scopes was touched ==="
echo "  the fleet store was not used: CRISPDM_ADMISSION_DIR=$CRISPDM_ADMISSION_DIR"
echo "  no limit, ceiling, cache, swap or oomd setting was read into a change"
