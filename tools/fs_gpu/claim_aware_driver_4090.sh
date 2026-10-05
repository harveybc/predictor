#!/usr/bin/env bash
# FS-GPU §3.3 claim-aware successor of worker_b's (RTX 4090 host) batch_002 baseline driver.
#
# Installed as its own user service that waits for the legacy driver service
# (feature-selection-4090.service) to become inactive. The legacy loop is ended at a cell
# boundary by truncating its plan file IN PLACE to the already-consumed prefix; its running
# cell is never touched and finishes under the legacy loop. This driver then regenerates the
# same plan (same generator, same order) and walks it FORWARD, skipping every cell that is
# terminal (own run, relayed peer terminal, FAILED marker) or claimed by worker_a, claiming
# each cell it takes under the shared protocol (tools/fs_gpu/ps3r_claims.py). As the home
# role of batch_002 it wins timestamp ties and, if the relay stops advancing, proceeds
# after the settle timeout (the stealer cannot claim without a fresh heartbeat).
# Recipe is byte-identical to the legacy driver: code fx_8987c57, cap 8000M, 6h wall,
# families identity,random,ae,dae, seed 0, window 168, latent 8, 16384 fit windows.
set -uo pipefail

ROOT="$HOME/.local/state/canonical_20261003/ps3r/dragon"
INPUT="$ROOT/input/batch_002"
BATCH="batch_002"
OUT="$ROOT/runs/$BATCH"
CODE="$ROOT/fx_8987c57"
PLAN="$ROOT/claim_aware_plan.txt"
LOG="$ROOT/claim_aware_driver.log"
CLAIMS="$ROOT/claims"
PEER="$ROOT/peer_terminals/$BATCH"
HEARTBEAT="$ROOT/RELAY_HEARTBEAT.json"
TOOLS="$HOME/.local/state/canonical_20261003/ps3r/tools"
ROLE="worker_b"
FAMILIES="identity,random,ae,dae"
CAP="8000M"
STALE_AFTER="${STALE_AFTER:-36000}"
SETTLE_POLL="${SETTLE_POLL:-20}"
SETTLE_TIMEOUT="${SETTLE_TIMEOUT:-1500}"
LEGACY_SERVICE="feature-selection-4090.service"

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
mkdir -p "$OUT" "$CLAIMS" "$PEER"

log "WAIT for $LEGACY_SERVICE to finish its current cell (waiter, not GPU work)"
while systemctl --user is-active --quiet "$LEGACY_SERVICE"; do
  sleep 30
done

GPU_UUID="$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/RTX 4090/{print $2; exit}')"
CODE_COMMIT="$(git -C "$CODE" rev-parse HEAD 2>/dev/null || echo unknown)"
INPUT_DIGEST="$(sha256sum "$INPUT/series.npz" | cut -d' ' -f1)"

python3 - "$INPUT" > "$PLAN" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
priority = json.loads((p / "ps2_extractor_priority.json").read_text())
allowed = set(json.loads((p / "batch_manifest.json").read_text())["features"])
seen = set()
for stage in ("tier_1", "exploration", "tier_2"):
    for feature in priority[stage]:
        if feature in allowed and not feature.startswith("cal.") and feature not in seen:
            seen.add(feature)
            print(stage, feature)
PY

valid_terminal() {
  python3 - "$1" <<'PY'
import hashlib, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
try:
    m = json.loads((p / "run_manifest.json").read_text())
    data = (p / "results.jsonl").read_bytes()
except (OSError, ValueError):
    raise SystemExit(1)
expected = m.get("results_sha256") or m.get("results_digest")
raise SystemExit(0 if m.get("status") == "COMPLETED" and expected == hashlib.sha256(data).hexdigest() else 1)
PY
}

claims() { python3 "$TOOLS/ps3r_claims.py" "$@"; }

log "START claim-aware role=$ROLE gpu=$GPU_UUID code=$CODE_COMMIT input=$INPUT_DIGEST plan=$(wc -l < "$PLAN") cells (forward order)"
n_run=0; n_skip=0; n_lose=0; n_fail=0
while read -r stage feature; do
  [[ -n "${feature:-}" ]] || continue
  out="$OUT/$feature"
  decision="$(claims decide --claims-root "$CLAIMS" --batch "$BATCH" --feature "$feature" --role "$ROLE" \
    --out-dir "$out" --peer-terminal-dir "$PEER/$feature" --heartbeat "$HEARTBEAT" \
    --stale-after-seconds "$STALE_AFTER" \
    --extra "{\"code_commit\":\"$CODE_COMMIT\",\"input_digest\":\"$INPUT_DIGEST\",\"families\":\"$FAMILIES\",\"stage\":\"$stage\",\"cap\":\"$CAP\",\"service\":\"feature-selection-4090-claims\"}")"
  rc=$?
  if [[ $rc -eq 10 ]]; then n_skip=$((n_skip+1)); log "SKIP $stage $BATCH $feature $decision"; continue; fi
  if [[ $rc -ne 0 ]]; then log "ERROR decide rc=$rc $feature $decision"; sleep 60; continue; fi
  log "CLAIMED $stage $BATCH $feature $decision"

  settled=0; verdict=""
  while (( settled < SETTLE_TIMEOUT )); do
    verdict="$(claims settle --claims-root "$CLAIMS" --batch "$BATCH" --feature "$feature" --role "$ROLE" \
      --heartbeat "$HEARTBEAT" --min-cycle-advance 3 --stale-after-seconds "$STALE_AFTER")"
    rc=$?
    [[ $rc -eq 12 ]] || break
    sleep "$SETTLE_POLL"; settled=$(( settled + SETTLE_POLL ))
  done
  if [[ $rc -eq 12 ]]; then
    # relay not advancing: the home role proceeds; the stealer cannot claim without a fresh heartbeat
    verdict="$(claims settle --claims-root "$CLAIMS" --batch "$BATCH" --feature "$feature" --role "$ROLE" --stale-after-seconds "$STALE_AFTER")"
    rc=$?
    log "HOME_PROCEEDS_WITHOUT_RELAY after ${SETTLE_TIMEOUT}s $feature $verdict"
  fi
  if [[ $rc -ne 0 ]]; then n_lose=$((n_lose+1)); log "LOSE $stage $BATCH $feature $verdict"; continue; fi
  if valid_terminal "$out" || valid_terminal "$PEER/$feature"; then
    claims mark --claims-root "$CLAIMS" --batch "$BATCH" --feature "$feature" --role "$ROLE" --state ABANDONED --extra '{"reason":"TERMINAL_APPEARED_DURING_SETTLE"}' >/dev/null
    n_skip=$((n_skip+1)); log "SKIP terminal appeared during settle $feature"; continue
  fi

  claims mark --claims-root "$CLAIMS" --batch "$BATCH" --feature "$feature" --role "$ROLE" --state RUNNING >/dev/null
  mkdir -p "$out"
  log "BEGIN cap=$CAP $stage $BATCH $feature"
  "$HOME/.local/bin/crispdm-run" -q -W 86400 -m "$CAP" -t 6h \
    -n "codexB_${feature//./_}" -- \
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_UUID" \
      TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 \
      LD_LIBRARY_PATH="$(cat "$HOME/.local/state/scratch/m07/ld_library_path.txt")" \
      bash -lc "cd '$CODE' && exec '$HOME/anaconda3/envs/tensorflow/bin/python' -m app.univariate_temporal_pilot --batch_dir '$INPUT' --out_dir '$out' --features '$feature' --window 168 --latent_dim 8 --seed 0 --families identity,random,ae,dae --max_fit_windows 16384 --max_val_windows 0 --max_ref_windows 256 --probe_lags 0,1,2,23" \
      > "$out/stdout.codex.log" 2>&1
  rc=$?
  log "END rc=$rc $stage $BATCH $feature"
  if valid_terminal "$out"; then
    claims mark --claims-root "$CLAIMS" --batch "$BATCH" --feature "$feature" --role "$ROLE" --state COMPLETED --results-dir "$out" >/dev/null
    n_run=$((n_run+1))
  else
    printf '{"status":"FAILED","rc":%s,"feature":"%s","families":"identity,random,ae,dae","seed":0,"host_role":"%s"}\n' "$rc" "$feature" "$ROLE" > "$out/FAILED.codex.json"
    claims mark --claims-root "$CLAIMS" --batch "$BATCH" --feature "$feature" --role "$ROLE" --state FAILED --extra "{\"rc\":$rc,\"marker\":\"$out/FAILED.codex.json\"}" >/dev/null
    n_fail=$((n_fail+1))
  fi
done < "$PLAN"
log "DONE claim-aware completed=$n_run failed=$n_fail skipped=$n_skip lost=$n_lose"
