#!/usr/bin/env bash
# FS-GPU §3.3 work-stealing stage for worker_a (RTX 5090 host).
#
# Appended to the 5090 successor chain as its own user service: it waits until the
# alternatives service and the baseline-successor service are both inactive, then walks
# batch_002 in REVERSE plan order (worker_b walks forward) taking still-pending baseline
# cells under the shared claim protocol (tools/fs_gpu/ps3r_claims.py). A cell is skipped
# when terminal (own output, relayed peer terminal, FAILED marker) or claimed by the peer.
# The settle gate needs three relay cycles; the stealer never claims without a fresh
# relay heartbeat and abandons on a settle timeout, so a relay outage can never produce
# a duplicate. Recipe (code fx_8987c57, families identity,random,ae,dae, seed 0, window
# 168, latent 8, batch_size 64, 16384 fit windows) is byte-identical to worker_b's.
# Cap: 1.25x the largest cgroup peak observed for this recipe, floor 8000M, monotonic.
# Idempotent and restart-safe: every decision is re-derived from files on disk.
set -uo pipefail

ROOT="$HOME/.local/state/canonical_20261003"
SUCCESSOR="$ROOT/selection_successor"
CODE="$SUCCESSOR/fx_8987c57"
PS2="$ROOT/ps2"
BATCH="batch_002"
OUT_ROOT="$SUCCESSOR/baseline/$BATCH"
CLAIMS="$ROOT/ps3r/claims"
PEER="$ROOT/ps3r/peer_terminals/$BATCH"
HEARTBEAT="$ROOT/ps3r/RELAY_HEARTBEAT.json"
TOOLS="$ROOT/ps3r/tools"
PLAN="$SUCCESSOR/steal_batch_002_plan.txt"
LOG="$SUCCESSOR/steal_batch_002_5090.log"
CAP_FILE="$SUCCESSOR/CAP_steal_batch_002"
ROLE="worker_a"
FAMILIES="identity,random,ae,dae"
MAX_HB_AGE="${MAX_HB_AGE:-300}"
STALE_AFTER="${STALE_AFTER:-36000}"
SETTLE_POLL="${SETTLE_POLL:-20}"
SETTLE_TIMEOUT="${SETTLE_TIMEOUT:-1500}"
RELAY_WAIT="${RELAY_WAIT:-120}"
RELAY_GIVEUP="${RELAY_GIVEUP:-172800}"
PYTHON_BIN="$HOME/anaconda3/envs/tensorflow/bin/python"
CRISPDM_RUN="$HOME/.local/bin/crispdm-run"
LD_FILE="$HOME/.local/state/scratch/m07/ld_library_path.txt"

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
mkdir -p "$OUT_ROOT" "$CLAIMS" "$PEER" "$(dirname "$LOG")"

log "WAIT for feature-selection-5090.service and feature-selection-baseline-successor-5090.service to finish (waiter, not GPU work)"
while systemctl --user is-active --quiet feature-selection-5090.service \
   || systemctl --user is-active --quiet feature-selection-baseline-successor-5090.service; do
  sleep 30
done

GPU_UUID="$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/RTX 5090/{print $2; exit}')"
CODE_COMMIT="$(git -C "$CODE" rev-parse HEAD 2>/dev/null || echo unknown)"
INPUT_DIGEST="$(sha256sum "$PS2/$BATCH/series.npz" | cut -d' ' -f1)"

# Same plan generator as worker_b's driver, reversed: the two walkers meet in the middle.
python3 - "$PS2/$BATCH" <<'PY' | tac > "$PLAN"
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

compute_cap() {
  python3 - "$SUCCESSOR/baseline" "$PEER" "$CAP_FILE" <<'PY'
import json, math, pathlib, sys
peak = 0
for root in sys.argv[1:3]:
    for m in pathlib.Path(root).rglob("run_manifest.json"):
        try:
            d = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
        if d.get("status") == "COMPLETED" and d.get("families") == ["identity", "random", "ae", "dae"]:
            peak = max(peak, int(d.get("cgroup_peak_bytes") or 0))
cap = max(8000, math.ceil(peak * 1.25 / 1048576))
prev = pathlib.Path(sys.argv[3])
if prev.is_file():
    try:
        cap = max(cap, int(prev.read_text().strip().rstrip("M")))
    except ValueError:
        pass
prev.write_text(f"{cap}M\n")
print(f"{cap}M")
PY
}

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

log "START steal role=$ROLE gpu=$GPU_UUID code=$CODE_COMMIT input=$INPUT_DIGEST plan=$(wc -l < "$PLAN") cells (reverse order)"
n_run=0; n_skip=0; n_lose=0; n_fail=0
while read -r stage feature; do
  [[ -n "${feature:-}" ]] || continue
  out="$OUT_ROOT/$feature"
  waited=0
  while :; do
    decision="$(claims decide --claims-root "$CLAIMS" --batch "$BATCH" --feature "$feature" --role "$ROLE" \
      --out-dir "$out" --peer-terminal-dir "$PEER/$feature" --heartbeat "$HEARTBEAT" \
      --max-heartbeat-age "$MAX_HB_AGE" --stale-after-seconds "$STALE_AFTER" \
      --extra "{\"code_commit\":\"$CODE_COMMIT\",\"input_digest\":\"$INPUT_DIGEST\",\"families\":\"$FAMILIES\",\"stage\":\"$stage\",\"service\":\"feature-selection-steal-5090\"}")"
    rc=$?
    if [[ $rc -eq 10 && "$decision" == *SKIP_RELAY_STALE* ]]; then
      if (( waited >= RELAY_GIVEUP )); then log "GIVEUP relay stale for ${waited}s at $feature"; break; fi
      (( waited == 0 )) && log "RELAY_STALE waiting before claiming $feature: $decision"
      sleep "$RELAY_WAIT"; waited=$(( waited + RELAY_WAIT )); continue
    fi
    break
  done
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
    claims mark --claims-root "$CLAIMS" --batch "$BATCH" --feature "$feature" --role "$ROLE" --state ABANDONED --extra '{"reason":"SETTLE_TIMEOUT_RELAY"}' >/dev/null
    n_lose=$((n_lose+1)); log "ABANDON settle timeout (relay did not advance 3 cycles in ${SETTLE_TIMEOUT}s) $feature"; continue
  fi
  if [[ $rc -ne 0 ]]; then n_lose=$((n_lose+1)); log "LOSE $stage $BATCH $feature $verdict"; continue; fi
  if valid_terminal "$out" || valid_terminal "$PEER/$feature"; then
    claims mark --claims-root "$CLAIMS" --batch "$BATCH" --feature "$feature" --role "$ROLE" --state ABANDONED --extra '{"reason":"TERMINAL_APPEARED_DURING_SETTLE"}' >/dev/null
    n_skip=$((n_skip+1)); log "SKIP terminal appeared during settle $feature"; continue
  fi

  cap="$(compute_cap)"
  claims mark --claims-root "$CLAIMS" --batch "$BATCH" --feature "$feature" --role "$ROLE" --state RUNNING --extra "{\"cap\":\"$cap\"}" >/dev/null
  mkdir -p "$out"
  log "BEGIN cap=$cap $stage $BATCH $feature"
  rc=0
  "$CRISPDM_RUN" -q -W 86400 -m "$cap" -t 8h -n "steal_${feature//./_}" -- \
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_UUID" \
      TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 \
      LD_LIBRARY_PATH="$(cat "$LD_FILE")" \
      bash -lc "cd '$CODE' && exec '$PYTHON_BIN' -m app.univariate_temporal_pilot --batch_dir '$PS2/$BATCH' --out_dir '$out' --features '$feature' --window 168 --latent_dim 8 --seed 0 --families '$FAMILIES' --batch_size 64 --max_fit_windows 16384 --max_val_windows 0 --max_ref_windows 256 --probe_lags 0,1,2,23" \
      > "$out/stdout.log" 2>&1 || rc=$?
  log "END rc=$rc $stage $BATCH $feature"
  if valid_terminal "$out"; then
    claims mark --claims-root "$CLAIMS" --batch "$BATCH" --feature "$feature" --role "$ROLE" --state COMPLETED --results-dir "$out" >/dev/null
    n_run=$((n_run+1))
  else
    printf '{"status":"FAILED","rc":%s,"feature":"%s","families":"%s","seed":0,"host_role":"%s"}\n' "$rc" "$feature" "$FAMILIES" "$ROLE" > "$out/FAILED.json"
    claims mark --claims-root "$CLAIMS" --batch "$BATCH" --feature "$feature" --role "$ROLE" --state FAILED --extra "{\"rc\":$rc,\"marker\":\"$out/FAILED.json\"}" >/dev/null
    n_fail=$((n_fail+1))
  fi
done < "$PLAN"
log "DONE steal completed=$n_run failed=$n_fail skipped=$n_skip lost=$n_lose"
