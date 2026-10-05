#!/bin/bash
# ---- FS-PRED status block (coordinator). Pulls progress.json, selector_sets.json and the per-cell records
# from worker_b (ionice idle), renders status/LANE_STATUS.json, method_walls.csv, primary_k24_sets.csv into the
# coordination worktree and commits its own paths at most every 30 min. Idempotent; safe to run every cycle.
set -uo pipefail
REPO="${FS_PRED_REPO:-$HOME/Documents/GitHub/.worktrees/predictor-canonical-exec-20261003}"
WORKER="${FS_PRED_WORKER:-dragon}"
STATE="$HOME/.local/state/canonical_20261003/fs_pred_status"; mkdir -p "$STATE/out"
DEST="$REPO/docs/audits/evidence/canonical_20261003/fs_closure/fs_pred"
PYTHON="${CRISPDM_PYTHON:-python3}"
ionice -c3 nice -n 19 rsync -q -a --delete --include='*/' --include='*.json' --exclude='*' \
  "$WORKER:.local/state/canonical_20261003/fs_pred/out/" "$STATE/out/" 2>>"$STATE/pull.err" || true
ionice -c3 rsync -q -a "$WORKER:.local/state/canonical_20261003/fs_pred/INCIDENTS.jsonl" "$STATE/INCIDENTS.worker_b.jsonl" 2>/dev/null || true
CRISPDM_PYTHON="$PYTHON" "$HOME/.local/bin/crispdm-run" -m 1G -t 10m -n fs_pred_status -- \
  "$PYTHON" "$DEST/fs_pred_status.py" "$STATE/out" "$DEST/status" > "$STATE/last_cycle.json" 2> "$STATE/last_cycle.err" || true
[ -f "$STATE/INCIDENTS.worker_b.jsonl" ] && cp "$STATE/INCIDENTS.worker_b.jsonl" "$DEST/status/INCIDENTS.worker_b.jsonl"
STAMP="$STATE/last_commit_epoch"
if [ ! -f "$STAMP" ] || [ $(( $(date +%s) - $(cat "$STAMP") )) -ge 1800 ]; then
  ( cd "$REPO" && git add docs/audits/evidence/canonical_20261003/fs_closure/fs_pred/status 2>/dev/null \
    && git diff --cached --quiet || git commit -q -m "FS-PRED: automatic status refresh

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" ; git pull --no-rebase --no-edit -q 2>/dev/null; git push -q 2>/dev/null ) || true
  date +%s > "$STAMP"
fi
