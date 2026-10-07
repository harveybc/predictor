#!/usr/bin/env bash
set -euo pipefail
config="$HOME/.config/fs4/wave-successor.env"
test -f "$config"
set -a
# shellcheck disable=SC1090
source "$config"
set +a
for key in FS4_CODE FS4_PYTHON FS4_QUEUE FS4_WAVE_MANIFEST FS4_RECEIPT_ROOT FS4_COVERAGE_REPORT \
           FS4_EURUSD_CONSOLIDATED FS4_ETH_CONSOLIDATED FS4_SUCCESSOR_OUT FS4_WEEKLY_DB FS4_ALIGNMENT_CERT; do
  : "${!key:?$key is required}"
done
cd "$FS4_CODE"
exec "$FS4_PYTHON" -m tools.fs4_wave_successor \
  --queue "$FS4_QUEUE" --manifest "$FS4_WAVE_MANIFEST" --receipt-root "$FS4_RECEIPT_ROOT" \
  --coverage-report "$FS4_COVERAGE_REPORT" \
  --consolidated "$FS4_EURUSD_CONSOLIDATED" --consolidated "$FS4_ETH_CONSOLIDATED" \
  --out-dir "$FS4_SUCCESSOR_OUT" --weekly-db "$FS4_WEEKLY_DB" \
  --alignment-cert "$FS4_ALIGNMENT_CERT"
