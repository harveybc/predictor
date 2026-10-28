#!/usr/bin/env bash
# One timer tick of one FS4 slot: health check, at most one claim, one runner under crispdm-run,
# local terminal first, SSH delivery to the coordinator. Installed by install_host.sh as
# ~/.local/bin/fs4-worker-once and invoked by fs4-worker@<slot>.service.
set -euo pipefail

slot="${1:?slot required}"
config="$HOME/.config/fs4/${slot}.env"
test -f "$config" || { echo "Missing $config" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$config"
set +a
: "${FS4_PYTHON:?}"
: "${FS4_CODE:?}"
: "${FS4_COORDINATOR:?}"
: "${FS4_CONTROLLER:?}"
: "${FS4_DB:?}"
: "${FS4_OWNER:?}"
: "${FS4_RUNNER:?}"
: "${FS4_CAP:?}"
: "${FS4_ARM:?}"
: "${FS4_OUTPUT_ROOT:?}"
case "$FS4_OUTPUT_ROOT" in /tmp|/tmp/*|/dev/shm*) echo "FS4_OUTPUT_ROOT is not durable" >&2; exit 2 ;; esac

args=(
  --coordinator "$FS4_COORDINATOR" --controller "$FS4_CONTROLLER"
  --python "${FS4_COORDINATOR_PYTHON:-python3}" --db "$FS4_DB" --owner "$FS4_OWNER"
  --runner "$FS4_RUNNER" --cap "$FS4_CAP" --arm "$FS4_ARM"
  --wall "${FS4_WALL:-2h}" --timeout "${FS4_TIMEOUT:-7500}"
  --output-root "$FS4_OUTPUT_ROOT" --max-tasks 1
)
if [[ "$FS4_ARM" == TRAINED_ENCODER ]]; then
  : "${FS4_GPU_UUID:?trained arm needs physical UUID}"
  args+=(--gpu-uuid "$FS4_GPU_UUID" --max-gpu-temp "${FS4_MAX_GPU_TEMP:-75}")
fi
mkdir -p "$FS4_OUTPUT_ROOT/logs"
exec "$FS4_PYTHON" "$FS4_CODE/tools/fs4_worker.py" "${args[@]}" \
  >> "$FS4_OUTPUT_ROOT/logs/${slot}.jsonl" 2>> "$FS4_OUTPUT_ROOT/logs/${slot}.stderr"
