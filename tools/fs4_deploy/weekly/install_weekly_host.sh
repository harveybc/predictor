#!/usr/bin/env bash
# Install and enable ONE weekly slot on this host as a systemd --user timer (2 min, --max-tasks 1).
#
#   install_weekly_host.sh <slot>              # reads ~/.config/fs4/<slot>.env (untracked, mode 0600)
#   install_weekly_host.sh <slot> --check-only # verify everything, install nothing
#
# Extends tools/fs4_deploy/install_host.sh (same env-file pattern and refusals). REFUSES before touching systemd when:
#   - the env file is missing, not mode 0600/0400, or lacks a required variable, or a value is a PENDING_* placeholder;
#   - the cap is not a size, or the output root is under /tmp or /dev/shm;
#   - FS4_CODE has no tools/fs4_weekly_worker.py / fs4_weekly_wrapper.py, or FS4_PYTHON cannot import tensorflow;
#   - the TRAIN (and, for validation slots, VALIDATION) input files named by the slot do not exist (existence only, never read);
#   - an encoder slot has no runner-results directory or no pinned extractor checkout;
#   - a GPU UUID is set but is not physical, not listed by nvidia-smi -L, or TensorFlow does not see exactly one device;
#   - FS4_GPU_UUID is set on a RAW slot (RAW is CPU only);
#   - the coordinator's weekly controller does not answer `status` over SSH for the configured DB;
#   - this host fails the health check (memory available >= cap + desktop reserve).
set -euo pipefail

slot="${1:?usage: install_weekly_host.sh <slot> [--check-only]}"
check_only=0; [[ "${2:-}" == "--check-only" ]] && check_only=1
source_dir="$(cd "$(dirname "$0")" && pwd)"
deploy_dir="$(dirname "$source_dir")"
config="$HOME/.config/fs4/${slot}.env"
refuse() { echo "REFUSED: $*" >&2; exit 2; }

test -f "$config" || refuse "missing $config"
perm="$(stat -c '%a' "$config")"
[[ "$perm" == "600" || "$perm" == "400" ]] || refuse "$config must be mode 0600 (is $perm)"
set -a
# shellcheck disable=SC1090
source "$config"
set +a
for var in FS4_PYTHON FS4_CODE FS4_COORDINATOR FS4_CONTROLLER FS4_DB FS4_OWNER FS4_CAP FS4_INPUT_MODE FS4_OUTPUT_ROOT \
           FS4_POPULATION FS4_BAR_HOURS FS4_TRAIN_FEATURES FS4_TRAIN_TARGETS; do
  [[ -n "${!var:-}" ]] || refuse "$var is empty in $config"
  [[ "${!var}" == PENDING* ]] && refuse "$var is still a placeholder (${!var})"
done
case "$FS4_INPUT_MODE" in RAW|RANDOM_ENCODER|TRAINED_ENCODER) ;; *) refuse "FS4_INPUT_MODE must be RAW, RANDOM_ENCODER or TRAINED_ENCODER" ;; esac
[[ "$FS4_OWNER" =~ ^(worker_a|worker_b|coordinator)-[a-z0-9-]+$ ]] || refuse "FS4_OWNER must be <role>-<slot> with role worker_a|worker_b|coordinator"
[[ "$FS4_CAP" =~ ^[0-9]+[KMGT]?$ ]] || refuse "FS4_CAP must be a size such as 2G (is $FS4_CAP)"
[[ "$FS4_BAR_HOURS" =~ ^[0-9]+$ ]] || refuse "FS4_BAR_HOURS must be an integer (is $FS4_BAR_HOURS)"
case "$FS4_OUTPUT_ROOT" in /tmp|/tmp/*|/dev/shm*) refuse "FS4_OUTPUT_ROOT is not durable: $FS4_OUTPUT_ROOT" ;; esac
test -x "$FS4_PYTHON" || refuse "FS4_PYTHON is not executable: $FS4_PYTHON"
test -f "$FS4_CODE/tools/fs4_weekly_worker.py" || refuse "FS4_CODE has no tools/fs4_weekly_worker.py: $FS4_CODE"
test -f "$FS4_CODE/tools/fs4_weekly_wrapper.py" || refuse "FS4_CODE has no tools/fs4_weekly_wrapper.py: $FS4_CODE"
test -x "$HOME/.local/bin/crispdm-run" || refuse "crispdm-run is not installed"
"$FS4_PYTHON" -c 'import tensorflow, pyarrow, numpy' >/dev/null 2>&1 || refuse "FS4_PYTHON cannot import tensorflow, pyarrow and numpy"

# shellcheck disable=SC2086
for f in $FS4_TRAIN_FEATURES "$FS4_TRAIN_TARGETS"; do test -f "$f" || refuse "TRAIN input missing: $f"; done
if [[ "${FS4_SPLIT:-validation}" == validation ]]; then
  [[ -n "${FS4_VAL_FEATURES:-}" && -n "${FS4_VAL_TARGETS:-}" ]] || refuse "validation slots need FS4_VAL_FEATURES and FS4_VAL_TARGETS"
  # shellcheck disable=SC2086
  for f in $FS4_VAL_FEATURES "$FS4_VAL_TARGETS"; do test -f "$f" || refuse "VALIDATION input missing (existence checked, content never read): $f"; done
fi
if [[ "$FS4_INPUT_MODE" != RAW ]]; then
  test -d "${FS4_RUNNER_RESULTS:-/nonexistent}" || refuse "encoder slot needs FS4_RUNNER_RESULTS (the runner's retained terminals)"
  test -f "${FS4_EXTRACTOR_CODE:-/nonexistent}/app/fs4_extractibility.py" || refuse "encoder slot needs FS4_EXTRACTOR_CODE (pinned feature-extractor checkout)"
fi
if [[ -n "${FS4_GPU_UUID:-}" ]]; then
  [[ "$FS4_INPUT_MODE" != RAW ]] || refuse "RAW slots are CPU only; unset FS4_GPU_UUID"
  [[ "$FS4_GPU_UUID" =~ ^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || refuse "FS4_GPU_UUID is not a physical UUID: $FS4_GPU_UUID"
  nvidia-smi -L 2>/dev/null | grep -F "(UUID: $FS4_GPU_UUID)" >/dev/null || refuse "nvidia-smi -L does not list $FS4_GPU_UUID"
  "$HOME/.local/bin/crispdm-run" -m 3G -t 5m -n "fs4-weekly-probe-$slot" -- \
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$FS4_GPU_UUID" "$FS4_PYTHON" "$deploy_dir/tf_probe.py" --expect-uuid "$FS4_GPU_UUID" >/dev/null \
    || refuse "TensorFlow does not see exactly one device under CUDA_VISIBLE_DEVICES=$FS4_GPU_UUID"
fi

common=(--coordinator "$FS4_COORDINATOR" --controller "$FS4_CONTROLLER" --python "${FS4_COORDINATOR_PYTHON:-python3}" --db "$FS4_DB"
        --owner "$FS4_OWNER" --cap "$FS4_CAP" --output-root "$FS4_OUTPUT_ROOT" --population "$FS4_POPULATION" --bar-hours "$FS4_BAR_HOURS"
        --input-mode "$FS4_INPUT_MODE" --train-features $FS4_TRAIN_FEATURES --train-targets "$FS4_TRAIN_TARGETS")
[[ -n "${FS4_VAL_FEATURES:-}" ]] && common+=(--val-features $FS4_VAL_FEATURES --val-targets "$FS4_VAL_TARGETS")
[[ "$FS4_INPUT_MODE" != RAW ]] && common+=(--runner-results "$FS4_RUNNER_RESULTS" --extractor-code "$FS4_EXTRACTOR_CODE")
"$FS4_PYTHON" "$FS4_CODE/tools/fs4_weekly_worker.py" "${common[@]}" --status \
  | "$FS4_PYTHON" -c 'import json,sys; d=json.load(sys.stdin); assert d["schema"]=="fs4.weekly_status.v1" and d["expected"]>0' \
  || refuse "coordinator weekly controller did not answer status for $FS4_DB"
"$FS4_PYTHON" "$FS4_CODE/tools/fs4_weekly_worker.py" "${common[@]}" ${FS4_GPU_UUID:+--gpu-uuid "$FS4_GPU_UUID"} --health || refuse "host health check failed for slot $slot"

echo "weekly slot $slot verified: owner=$FS4_OWNER mode=$FS4_INPUT_MODE population=$FS4_POPULATION cap=$FS4_CAP gpu=${FS4_GPU_UUID:-cpu}"
[[ $check_only -eq 1 ]] && exit 0

mkdir -p "$FS4_OUTPUT_ROOT"
install -D -m 755 "$source_dir/weekly_once.sh" "$HOME/.local/bin/fs4-weekly-once"
install -D -m 644 "$source_dir/fs4-weekly@.service" "$HOME/.config/systemd/user/fs4-weekly@.service"
install -D -m 644 "$source_dir/fs4-weekly@.timer" "$HOME/.config/systemd/user/fs4-weekly@.timer"
systemctl --user daemon-reload
systemctl --user enable --now "fs4-weekly@${slot}.timer"
systemctl --user status "fs4-weekly@${slot}.timer" --no-pager
