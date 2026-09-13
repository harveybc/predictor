#!/bin/sh
ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
TOKEN_FILE="$ROOT/../../../data-gov/var/lake_token"
if [ -f "$TOKEN_FILE" ]; then
  DATA_GOV_LAKE_TOKEN="$(tr -d '\n' < "$TOKEN_FILE")"
  export DATA_GOV_LAKE_TOKEN
fi
if [ -z "$PGPASSWORD" ]; then
  echo "PGPASSWORD is required for the live cube (same as olap/init_db.py)." >&2
fi
exec python3 -m app.main --load_config examples/config/default.json "$@"
