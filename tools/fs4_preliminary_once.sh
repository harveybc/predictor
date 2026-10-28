#!/usr/bin/env bash
# Run one pinned first-wave task through the existing admitted FS4 worker.
set -euo pipefail
slot="${1:?slot required}"
config="$HOME/.config/fs4/${slot}.env"
test -f "$config"
set -a
source "$config"
set +a
: "${FS4_PRELIMINARY_MANIFEST:?}"
: "${FS4_PRELIMINARY_SHA256:?}"
task_id="$(python3 "$HOME/.local/bin/fs4-preliminary-next.py" \
    --manifest "$FS4_PRELIMINARY_MANIFEST" --sha256 "$FS4_PRELIMINARY_SHA256" \
    --coordinator "$FS4_COORDINATOR" --python "$FS4_COORDINATOR_PYTHON" \
    --controller "$FS4_CONTROLLER" --db "$FS4_DB" --arm "$FS4_ARM")"
[[ -n "$task_id" ]] || exit 0
args=(
    --coordinator "$FS4_COORDINATOR" --controller "$FS4_CONTROLLER"
    --python "$FS4_COORDINATOR_PYTHON" --db "$FS4_DB" --owner "$FS4_OWNER"
    --runner "$FS4_RUNNER" --cap "$FS4_CAP" --arm "$FS4_ARM"
    --wall "$FS4_WALL" --timeout "$FS4_TIMEOUT"
    --output-root "$FS4_OUTPUT_ROOT" --max-tasks 1 --task-id "$task_id"
)
if [[ "$FS4_ARM" == TRAINED_ENCODER ]]; then
  args+=(--gpu-uuid "$FS4_GPU_UUID" --max-gpu-temp "$FS4_MAX_GPU_TEMP")
fi
exec "$FS4_PYTHON" "$FS4_CODE/tools/fs4_worker.py" "${args[@]}"
