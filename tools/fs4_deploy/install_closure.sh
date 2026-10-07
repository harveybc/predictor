#!/usr/bin/env bash
# Install the FS4 closure follower on the coordinator (systemd --user). Refuses before enabling
# the timer when the env file, the controller DB, the checkout or the interpreter is not real.
# Never deploys the warehouse service; see LIVE_DEPLOY_STEPS_FS4.md for that.
set -euo pipefail
source_dir="$(cd "$(dirname "$0")" && pwd)"
config="$HOME/.config/fs4/closure.env"
test -f "$config" || { echo "Missing $config (see closure_tick.sh header for its variables)" >&2; exit 2; }
perms=$(stat -c '%a' "$config"); [[ "$perms" == "600" ]] || { echo "$config must be mode 0600 (is $perms)" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$config"
set +a
: "${FS4_PYTHON:?}" "${FS4_CODE:?}" "${FS4_DB:?}" "${FS4_WAREHOUSE:?}" "${FS4_STATE:?}"
test -x "$FS4_PYTHON" || { echo "Interpreter is not executable: $FS4_PYTHON" >&2; exit 2; }
test -f "$FS4_CODE/tools/fs4_closure.py" || { echo "Checkout lacks tools/fs4_closure.py: $FS4_CODE" >&2; exit 2; }
test -f "$FS4_DB" || { echo "Controller DB missing: $FS4_DB" >&2; exit 2; }
case "$FS4_WAREHOUSE" in
  http://*|https://*) : "${WAREHOUSE_TOKEN:?WAREHOUSE_TOKEN required for a service warehouse}" ;;
esac
test -x "$HOME/.local/bin/crispdm-run" || { echo "crispdm-run missing" >&2; exit 2; }
PYTHONPATH="$FS4_CODE/olap/store/src:$FS4_CODE" "$FS4_PYTHON" "$FS4_CODE/tools/fs4_closure.py" \
  --db "$FS4_DB" --warehouse "$FS4_WAREHOUSE" --state-root "$FS4_STATE" expected >/dev/null \
  || { echo "The controller DB does not answer expected counts; not installing" >&2; exit 2; }
install -D -m 644 "$source_dir/fs4-closure.service" "$HOME/.config/systemd/user/fs4-closure.service"
install -D -m 644 "$source_dir/fs4-closure.timer" "$HOME/.config/systemd/user/fs4-closure.timer"
systemctl --user daemon-reload
systemctl --user enable --now fs4-closure.timer
systemctl --user status fs4-closure.timer --no-pager
