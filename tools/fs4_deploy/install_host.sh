#!/usr/bin/env bash
# Install and enable ONE FS4 slot on this host as a systemd --user timer (2 min, --max-tasks 1).
#
#   install_host.sh <slot>              # reads ~/.config/fs4/<slot>.env (untracked, mode 0600)
#   install_host.sh <slot> --check-only # verify everything, install nothing
#
# The slot file names the role-only owner, the pinned runner executable, the measured cap and
# (for TRAINED_ENCODER) the physical GPU UUID. The installer REFUSES before touching systemd when:
#   - the env file is missing, world/group readable, or lacks a required variable;
#   - the runner is not a regular executable, or does not answer `--self-check` with JSON
#     naming a real code commit (a placeholder such as PENDING_* is not a runner);
#   - the cap is not a size (e.g. 4650M) or is a placeholder;
#   - the output root is under /tmp or /dev/shm (not durable);
#   - for TRAINED_ENCODER: the UUID is not of the physical form, is not listed by nvidia-smi -L,
#     or TensorFlow (FS4_PYTHON) does not see exactly one device under CUDA_VISIBLE_DEVICES=UUID
#     (and still sees one under a bogus UUID, which would mean a fallback);
#   - the coordinator's controller does not answer `status` over SSH with the configured DB.
set -euo pipefail

slot="${1:?usage: install_host.sh <slot> [--check-only]}"
check_only=0; [[ "${2:-}" == "--check-only" ]] && check_only=1
source_dir="$(cd "$(dirname "$0")" && pwd)"
config="$HOME/.config/fs4/${slot}.env"
refuse() { echo "REFUSED: $*" >&2; exit 2; }

test -f "$config" || refuse "missing $config"
perm="$(stat -c '%a' "$config")"
[[ "$perm" == "600" || "$perm" == "400" ]] || refuse "$config must be mode 0600 (is $perm)"
set -a
# shellcheck disable=SC1090
source "$config"
set +a
for var in FS4_PYTHON FS4_CODE FS4_COORDINATOR FS4_CONTROLLER FS4_DB FS4_OWNER FS4_RUNNER FS4_CAP FS4_ARM FS4_OUTPUT_ROOT; do
  [[ -n "${!var:-}" ]] || refuse "$var is empty in $config"
  [[ "${!var}" == PENDING* ]] && refuse "$var is still a placeholder (${!var})"
done
case "$FS4_ARM" in RAW|RANDOM_ENCODER|TRAINED_ENCODER) ;; *) refuse "FS4_ARM must be RAW, RANDOM_ENCODER or TRAINED_ENCODER" ;; esac
[[ "$FS4_OWNER" =~ ^(worker_a|worker_b|coordinator)-[a-z0-9-]+$ ]] || refuse "FS4_OWNER must be <role>-<slot> with role worker_a|worker_b|coordinator"
[[ "$FS4_CAP" =~ ^[0-9]+[KMGT]?$ ]] || refuse "FS4_CAP must be a size such as 4650M (is $FS4_CAP)"
case "$FS4_OUTPUT_ROOT" in /tmp|/tmp/*|/dev/shm*) refuse "FS4_OUTPUT_ROOT is not durable: $FS4_OUTPUT_ROOT" ;; esac
test -x "$FS4_PYTHON" || refuse "FS4_PYTHON is not executable: $FS4_PYTHON"
test -f "$FS4_CODE/tools/fs4_worker.py" || refuse "FS4_CODE has no tools/fs4_worker.py: $FS4_CODE"
test -x "$HOME/.local/bin/crispdm-run" || refuse "crispdm-run is not installed"

# the runner must be a real pinned executable, not a path that merely exists
[[ -f "$FS4_RUNNER" && -x "$FS4_RUNNER" ]] || refuse "runner is not a regular executable: $FS4_RUNNER"
self_check="$("$FS4_RUNNER" --self-check 2>/dev/null || true)"
[[ -n "$self_check" ]] || refuse "runner did not answer --self-check: $FS4_RUNNER"
commit="$(printf '%s' "$self_check" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("code_commit") or "")' 2>/dev/null || true)"
[[ "$commit" =~ ^[0-9a-f]{7,40}$ ]] || refuse "runner --self-check carries no real code_commit: ${self_check:0:200}"

if [[ "$FS4_ARM" == TRAINED_ENCODER ]]; then
  [[ -n "${FS4_GPU_UUID:-}" ]] || refuse "TRAINED_ENCODER needs FS4_GPU_UUID"
  [[ "$FS4_GPU_UUID" =~ ^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || refuse "FS4_GPU_UUID is not a physical UUID: $FS4_GPU_UUID"
  # A fault on another GPU can make nvidia-smi exit nonzero after listing this one.
  gpu_listing="$(nvidia-smi -L 2>/dev/null || true)"
  grep -F "(UUID: $FS4_GPU_UUID)" <<<"$gpu_listing" >/dev/null || refuse "nvidia-smi -L does not list $FS4_GPU_UUID"
  probe="$source_dir/tf_probe.py"
  ldp="${LD_LIBRARY_PATH:-}"; [[ -n "${FS4_LD_LIBRARY_PATH_FILE:-}" && -f "$FS4_LD_LIBRARY_PATH_FILE" ]] && ldp="$(< "$FS4_LD_LIBRARY_PATH_FILE")"
  "$HOME/.local/bin/crispdm-run" -m 3G -t 5m -n "fs4-install-probe-$slot" -- \
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$FS4_GPU_UUID" LD_LIBRARY_PATH="$ldp" "$FS4_PYTHON" "$probe" --expect-uuid "$FS4_GPU_UUID" >/dev/null \
    || refuse "TensorFlow does not see exactly one device under CUDA_VISIBLE_DEVICES=$FS4_GPU_UUID"
  "$HOME/.local/bin/crispdm-run" -m 3G -t 5m -n "fs4-install-probe-none-$slot" -- \
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-00000000-0000-0000-0000-000000000000 LD_LIBRARY_PATH="$ldp" "$FS4_PYTHON" "$probe" --expect-none >/dev/null \
    || refuse "TensorFlow still sees a device under a bogus UUID: a fallback would be possible"
else
  [[ -z "${FS4_GPU_UUID:-}" ]] || refuse "CPU arm $FS4_ARM must not set FS4_GPU_UUID"
fi

# the coordinator must answer with the configured queue, and this host must be fit right now
"$FS4_PYTHON" "$FS4_CODE/tools/fs4_worker.py" --coordinator "$FS4_COORDINATOR" --controller "$FS4_CONTROLLER" \
  --db "$FS4_DB" --owner "$FS4_OWNER" --runner "$FS4_RUNNER" --cap "$FS4_CAP" --output-root "$FS4_OUTPUT_ROOT" --status \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["schema"]=="fs4.status.v1" and d["total"]>0' \
  || refuse "coordinator controller did not answer status for $FS4_DB"
health_args=(--coordinator "$FS4_COORDINATOR" --controller "$FS4_CONTROLLER" --db "$FS4_DB" --owner "$FS4_OWNER"
             --runner "$FS4_RUNNER" --cap "$FS4_CAP" --output-root "$FS4_OUTPUT_ROOT" --arm "$FS4_ARM" --health)
[[ "$FS4_ARM" == TRAINED_ENCODER ]] && health_args+=(--gpu-uuid "$FS4_GPU_UUID")
"$FS4_PYTHON" "$FS4_CODE/tools/fs4_worker.py" "${health_args[@]}" || refuse "host health check failed for slot $slot"

echo "slot $slot verified: owner=$FS4_OWNER arm=$FS4_ARM cap=$FS4_CAP runner_commit=$commit gpu=${FS4_GPU_UUID:-cpu}"
[[ $check_only -eq 1 ]] && exit 0

mkdir -p "$FS4_OUTPUT_ROOT"
install -D -m 755 "$source_dir/worker_once.sh" "$HOME/.local/bin/fs4-worker-once"
install -D -m 644 "$source_dir/fs4-worker@.service" "$HOME/.config/systemd/user/fs4-worker@.service"
install -D -m 644 "$source_dir/fs4-worker@.timer" "$HOME/.config/systemd/user/fs4-worker@.timer"
systemctl --user daemon-reload
systemctl --user enable --now "fs4-worker@${slot}.timer"
systemctl --user status "fs4-worker@${slot}.timer" --no-pager
