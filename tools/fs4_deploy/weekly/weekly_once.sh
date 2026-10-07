#!/usr/bin/env bash
# One timer tick of one weekly slot: health check, at most one claim, one wrapper run under crispdm-run,
# local terminal first, SSH delivery to the coordinator. Installed by install_weekly_host.sh as
# ~/.local/bin/fs4-weekly-once and invoked by fs4-weekly@<slot>.service. Same env-file pattern as worker_once.sh.
set -euo pipefail

slot="${1:?slot required}"
config="$HOME/.config/fs4/${slot}.env"
test -f "$config" || { echo "Missing $config" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$config"
set +a
for v in FS4_PYTHON FS4_CODE FS4_COORDINATOR FS4_CONTROLLER FS4_DB FS4_OWNER FS4_CAP FS4_INPUT_MODE FS4_OUTPUT_ROOT \
         FS4_POPULATION FS4_BAR_HOURS FS4_TRAIN_FEATURES FS4_TRAIN_TARGETS; do
  : "${!v:?$v is required in $config}"
done
case "$FS4_OUTPUT_ROOT" in /tmp|/tmp/*|/dev/shm*) echo "FS4_OUTPUT_ROOT is not durable" >&2; exit 2 ;; esac

# shellcheck disable=SC2206
args=(
  --coordinator "$FS4_COORDINATOR" --controller "$FS4_CONTROLLER" --python "${FS4_COORDINATOR_PYTHON:-python3}"
  --python-local "$FS4_PYTHON" --wrapper "$FS4_CODE/tools/fs4_weekly_wrapper.py" --db "$FS4_DB" --owner "$FS4_OWNER"
  --cap "$FS4_CAP" --wall "${FS4_WALL:-3h}" --timeout "${FS4_TIMEOUT:-10500}" --output-root "$FS4_OUTPUT_ROOT" --max-tasks 1
  --population "$FS4_POPULATION" --bar-hours "$FS4_BAR_HOURS" --input-mode "$FS4_INPUT_MODE" --split "${FS4_SPLIT:-validation}"
  --train-features $FS4_TRAIN_FEATURES --train-targets "$FS4_TRAIN_TARGETS"
)
[[ -n "${FS4_VAL_FEATURES:-}" ]] && args+=(--val-features $FS4_VAL_FEATURES --val-targets "${FS4_VAL_TARGETS:?FS4_VAL_TARGETS}")
[[ -n "${FS4_RUNNER_RESULTS:-}" ]] && args+=(--runner-results "$FS4_RUNNER_RESULTS" --extractor-code "${FS4_EXTRACTOR_CODE:?FS4_EXTRACTOR_CODE}")
[[ -n "${FS4_MIN_FEATURES:-}" ]] && args+=(--min-features "$FS4_MIN_FEATURES")
[[ -n "${FS4_MAX_FEATURES:-}" ]] && args+=(--max-features "$FS4_MAX_FEATURES")
[[ -n "${FS4_TEST_FREEZE:-}" ]] && args+=(--test-freeze "$FS4_TEST_FREEZE")
[[ -n "${FS4_GPU_UUID:-}" ]] && args+=(--gpu-uuid "$FS4_GPU_UUID" --max-gpu-temp "${FS4_MAX_GPU_TEMP:-75}")
mkdir -p "$FS4_OUTPUT_ROOT/logs"
exec "$FS4_PYTHON" "$FS4_CODE/tools/fs4_weekly_worker.py" "${args[@]}" \
  >> "$FS4_OUTPUT_ROOT/logs/${slot}.jsonl" 2>> "$FS4_OUTPUT_ROOT/logs/${slot}.stderr"
