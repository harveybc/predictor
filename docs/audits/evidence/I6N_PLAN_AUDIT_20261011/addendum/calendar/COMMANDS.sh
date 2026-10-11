#!/bin/sh
# Lane "calendar" of the I6-N audit ADDENDUM. Run date: 2026-10-11. CPU only.
# Interpreter: Python 3.12.7, numpy 1.26.4, pandas 2.2.2 (pandas only for bar_label_check part 2).
# Paths: STATE=~/.local/state ; AUDITED = checkout of the audited object
# (branch codex/i6d-weekly-walk-forward-20261009, HEAD 269c2d13 + uncommitted draft;
# every module imported below is committed and unmodified at 269c2d13).
# AUD = docs/audits/evidence/I6N_PLAN_AUDIT_20261011 (commit 4d3af09d); OUT = $AUD/addendum/calendar
# Each job: crispdm-run -q -m <cap> -t <wall> -n i6n-add-calendar, one at a time.
set -eu
STATE=${STATE:-$HOME/.local/state}
I6E=$STATE/canonical_20261003/i6e
OUT=$AUD/addendum/calendar
export CUDA_VISIBLE_DEVICES=""

# 1. Identify the INPUT the original sub-auditor read (ts member only).
crispdm-run -q -m 1G -t 5m -n i6n-add-calendar -- python3 -I $OUT/identify_input.py "$STATE" \
  canonical_20261003/i6e/statistical72_v1/INPUT.npz \
  canonical_20261003/i6e/multihorizon_2024_v1/INPUT.npz \
  canonical_20261003/i6e/multihorizon_2024_v2/INPUT.npz > $OUT/identify_input.out

# 2. Rerun the three retained calendar probes UNMODIFIED; outputs must be byte-identical.
for s in bar_clock gaps_2024; do
  crispdm-run -q -m 1G -t 5m -n i6n-add-calendar -- python3 -I $AUD/probes/a/$s.py $I6E/statistical72_v1/INPUT.npz > $OUT/rerun_$s.out 2>&1
  cmp $OUT/rerun_$s.out $AUD/probes/a/$s.out
done
crispdm-run -q -m 1G -t 10m -n i6n-add-calendar -- python3 -I $AUD/probes/a/wmc_calendar.py \
  $I6E/statistical72_v1/INPUT.npz $OUT/rerun_wmc_calendar.json > $OUT/rerun_wmc_calendar.out 2>&1
cmp $OUT/rerun_wmc_calendar.out $AUD/probes/a/wmc_calendar.out
cmp $OUT/rerun_wmc_calendar.json $AUD/probes/a/wmc_calendar.json

# 3. Parameterized calendar attribution (19 calendars) + I6-E segment/row reconciliation.
crispdm-run -q -m 2G -t 15m -n i6n-add-calendar -- python3 -I $OUT/calendar_attribution.py \
  $I6E/statistical72_v1/INPUT.npz "$AUDITED" $I6E/multihorizon_2024_v2/DESIGN.json \
  $OUT/calendar_attribution.json > $OUT/calendar_attribution.out 2>&1

# 4. Bar-label / DST-gap clock check (ts only) + known-input converter control.
crispdm-run -q -m 2G -t 10m -n i6n-add-calendar -- python3 -I $OUT/bar_label_check.py \
  $I6E/statistical72_v1/INPUT.npz "$AUDITED" > $OUT/bar_label_check.out 2>&1

# Steps 3 and 4 were each run twice; both reruns were byte-identical (determinism check).
