#!/usr/bin/env bash
set -euo pipefail

slot="${1:?usage: install_host.sh <slot>}"
source_dir="$(cd "$(dirname "$0")" && pwd)"
config="$HOME/.config/fs4/${slot}.env"
test -f "$config" || { echo "Missing $config" >&2; exit 2; }
set -a
source "$config"
set +a
: "${FS4_RUNNER:?}"
: "${FS4_ARM:?}"
test -x "$FS4_RUNNER" || { echo "Runner is not executable: $FS4_RUNNER" >&2; exit 2; }
if [[ "$FS4_ARM" == TRAINED_ENCODER ]]; then
  : "${FS4_GPU_UUID:?}"
  nvidia-smi -L | grep -F "$FS4_GPU_UUID" >/dev/null || {
    echo "Requested physical GPU is not visible" >&2; exit 2;
  }
fi
install -D -m 755 "$source_dir/worker_once.sh" "$HOME/.local/bin/fs4-worker-once"
install -D -m 644 "$source_dir/fs4-worker@.service" "$HOME/.config/systemd/user/fs4-worker@.service"
install -D -m 644 "$source_dir/fs4-worker@.timer" "$HOME/.config/systemd/user/fs4-worker@.timer"
systemctl --user daemon-reload
systemctl --user enable --now "fs4-worker@${slot}.timer"
systemctl --user status "fs4-worker@${slot}.timer" --no-pager
