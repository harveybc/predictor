#!/usr/bin/env bash
# Activate only pinned weekly RAW slots after the successor controller has a plan.
set -euo pipefail
config="$HOME/.config/fs4/weekly-activation.env"
test -f "$config"
set -a
# shellcheck disable=SC1090
source "$config"
set +a
for key in FS4_CODE FS4_COORDINATOR FS4_CONTROLLER FS4_DB FS4_WEEKLY_SLOTS; do
  : "${!key:?$key is required}"
done
status="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$FS4_COORDINATOR" \
  python3 "$FS4_CONTROLLER" --db "$FS4_DB" status 2>/dev/null || true)"
if ! printf '%s' "$status" | python3 -c \
  'import json,sys; d=json.load(sys.stdin); assert d["schema"]=="fs4.weekly_status.v1" and d["expected"]>0' \
  >/dev/null 2>&1; then
  echo WEEKLY_PLAN_NOT_READY
  exit 0
fi
for slot in $FS4_WEEKLY_SLOTS; do
  if systemctl --user is-active --quiet "fs4-weekly@${slot}.timer"; then
    continue
  fi
  if ! bash "$FS4_CODE/tools/fs4_deploy/weekly/install_weekly_host.sh" "$slot"; then
    echo "WEEKLY_SLOT_NOT_ADMITTED:$slot" >&2
  fi
done
