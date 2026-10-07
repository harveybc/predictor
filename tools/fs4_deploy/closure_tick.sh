#!/usr/bin/env bash
# One closure pass (fs4-closure.timer fires it every two minutes on the coordinator): submit each
# COMPLETE terminal from the controller's task store to the warehouse, verify the readback, write
# STATUS.json from the task store, and write EXTRACTIBILITY_COMPLETE.json when the evidence allows
# it. Runs tools/fs4_closure.py under crispdm-run with FS4_CLOSURE_CAP (<= 1G). No loop here;
# systemd continues the queue. Nothing here touches the controller DB for writing.
#
# Environment (~/.config/fs4/closure.env, mode 0600; roles only, never host names in the repo):
#   FS4_PYTHON        interpreter with duckdb / sqlalchemy (the follower's own env)
#   FS4_CODE          predictor checkout on the coordinator (this branch or its merge)
#   FS4_DB            <state>/fs4/queue_v2.sqlite (the controller's SQLite; read-only here)
#   FS4_WAREHOUSE     http://127.0.0.1:<port> of the warehouse service, or a local .duckdb file
#   FS4_STATE         durable directory for receipts/, quarantine/, STATUS.json and the closure
#   FS4_CLOSURE_CAP   crispdm-run memory cap, default 1G (never above)
#   WAREHOUSE_TOKEN   the service token; lives in the env file only, never printed or committed
set -uo pipefail
if [[ -z "${FS4_STATE:-}" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.config/fs4/closure.env"
fi
: "${FS4_PYTHON:?}" "${FS4_CODE:?}" "${FS4_DB:?}" "${FS4_WAREHOUSE:?}" "${FS4_STATE:?}"
FS4_CLOSURE_CAP="${FS4_CLOSURE_CAP:-1G}"
case "$FS4_WAREHOUSE" in
  http://*|https://*) : "${WAREHOUSE_TOKEN:?WAREHOUSE_TOKEN must be set in the env file for a service warehouse}" ;;
esac
export PYTHONPATH="$FS4_CODE/olap/store/src:$FS4_CODE${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1
mkdir -p "$FS4_STATE"
if [[ -f "$FS4_STATE/EXTRACTIBILITY_COMPLETE.json" ]]; then
  # closed: refresh STATUS.json from the task store only and exit; nothing is resubmitted
  cd "$FS4_CODE" && exec "$HOME/.local/bin/crispdm-run" -q -m "$FS4_CLOSURE_CAP" -t 5m -n fs4-closure-status -- \
    "$FS4_PYTHON" tools/fs4_closure.py --db "$FS4_DB" --warehouse "$FS4_WAREHOUSE" --state-root "$FS4_STATE" status \
    > /dev/null 2>>"$FS4_STATE/closure_tick.err"
fi
cd "$FS4_CODE" && exec "$HOME/.local/bin/crispdm-run" -q -m "$FS4_CLOSURE_CAP" -t 20m -n fs4-closure-tick -- \
  "$FS4_PYTHON" tools/fs4_closure.py --db "$FS4_DB" --warehouse "$FS4_WAREHOUSE" --state-root "$FS4_STATE" tick \
  > /dev/null 2>>"$FS4_STATE/closure_tick.err"
