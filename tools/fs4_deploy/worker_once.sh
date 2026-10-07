#!/usr/bin/env bash
set -euo pipefail

slot="${1:?slot required}"
config="$HOME/.config/fs4/${slot}.env"
test -f "$config" || { echo "Missing $config" >&2; exit 2; }
set -a
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

args=(
  --coordinator "$FS4_COORDINATOR" --controller "$FS4_CONTROLLER"
  --python "$FS4_PYTHON" --db "$FS4_DB" --owner "$FS4_OWNER"
  --runner "$FS4_RUNNER" --cap "$FS4_CAP" --arm "$FS4_ARM"
  --output-root "$FS4_OUTPUT_ROOT" --max-tasks 1
)
if [[ "$FS4_ARM" == TRAINED_ENCODER ]]; then
  : "${FS4_GPU_UUID:?trained arm needs physical UUID}"
  args+=(--gpu-uuid "$FS4_GPU_UUID")
fi
exec "$FS4_PYTHON" "$FS4_CODE/tools/fs4_worker.py" "${args[@]}"
