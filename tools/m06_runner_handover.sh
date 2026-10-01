#!/usr/bin/env bash
# M06 durable-runner handover (runs on the worker as its own transient unit, so it survives agent death).
# usage: handover.sh NAME CWD MATCH STOPFILE|- WAITFILE|- WAITSTR|- ENVSPEC -- command...
#  1. optionally wait until WAITFILE contains WAITSTR (<= 2 h);
#  2. optionally touch STOPFILE so the current loop finishes its current attempt and exits;
#  3. wait until no process matches MATCH (<= 6 h);
#  4. remove STOPFILE; refuse if MATCH is alive again; start the durable unit NAME through crispdm-run 512M.
set -euo pipefail
NAME=$1 CWD=$2 MATCH=$3 STOPF=$4 WAITF=$5 WAITS=$6 ENVSPEC=$7; shift 7; [ "$1" = "--" ] && shift
log(){ echo "$(date -u +%FT%TZ) handover $NAME: $*"; }
alive(){  # a process matching MATCH that is not a handover/pgrep/ssh/bash -c wrapper carrying the pattern as text
  for p in $(pgrep -f -- "$MATCH"); do
    [ "$p" = "$$" ] && continue
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null) || continue
    case "$c" in *handover.sh*|bash\ -c*|*systemd-run*|pgrep*) continue;; esac
    return 0
  done; return 1; }
if [ "$WAITF" != "-" ]; then
  for i in $(seq 1 720); do grep -q -- "$WAITS" "$WAITF" 2>/dev/null && break; sleep 10; done
  grep -q -- "$WAITS" "$WAITF" || { log "precondition '$WAITS' not seen in $WAITF after 2 h; abort"; exit 3; }
  log "precondition seen"
fi
if [ "$STOPF" != "-" ]; then touch "$STOPF"; log "STOP touched"; fi
for i in $(seq 1 2160); do alive || break; sleep 10; done
alive && { log "old loop still alive after 6 h; abort"; exit 4; }
[ "$STOPF" != "-" ] && rm -f "$STOPF" && log "STOP removed"
sleep 2; alive && { log "a loop matching appeared again; abort (no second loop)"; exit 5; }
mkdir -p "$HOME/.local/state/runners"
ENVARGS=(); IFS=';' read -ra kv <<< "$ENVSPEC"; for e in "${kv[@]}"; do [ -n "$e" ] && ENVARGS+=(-E "$e"); done
systemd-run --user --unit "$NAME" --description "durable runner $NAME (M06 handover)" \
  -p Restart=on-failure -p RestartSec=30 -p WorkingDirectory="$CWD" "${ENVARGS[@]}" \
  -p StandardOutput=append:$HOME/.local/state/runners/$NAME.log -p StandardError=append:$HOME/.local/state/runners/$NAME.log \
  $HOME/.local/bin/crispdm-run -q -W 3600 -m 512M -t 24h -n "$NAME" -- "$@"
sleep 5; log "unit $(systemctl --user is-active "$NAME")"
