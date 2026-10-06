#!/usr/bin/env bash
# Per-host deployment step for the phase 2/3 selection campaign. deploy.sh runs it on the
# coordinator directly and on each worker over `ssh ALIAS bash -s -- ARGS`, so it needs only
# bash, coreutils, tar, systemd --user and one python 3.12 interpreter on the host.
#
#   deploy_host_step.sh --role ROLE --commit SHA --tar-sha256 HEX --base-python PY
#                       [--cap 2G] [--slots N] [--threads 1] [--steal 0|1] [--wall 12h]
#                       [--follow-cap 2G] [--env KEY=VALUE ...]   (KEY=VALUE appended to runner.env;
#                       FS23_WAREHOUSE_OVERRIDE=... in the environment points the follower elsewhere)
#
# The archive must already be at $STATE/code/incoming.<SHA>.tar. Steps, all idempotent:
#   1. verify the archive digest, extract into code/<SHA>/ (once), point code/CURRENT at it,
#      and compute the TREE digest (sha256 over the sorted sha256 of every file) -- the same
#      command on every host, so equality means the same bytes were installed;
#   2. build/refresh venv/ from the base interpreter and install EXACTLY requirements.lock;
#      record python version, the lock digest and a digest of `pip freeze` restricted to the
#      locked packages;
#   3. install the unit files and the role drop-in into ~/.config/systemd/user (no enable, no
#      start: deploy.sh --start does that in phase B), daemon-reload;
#   4. write runner.env (absolute paths, this role's cap/slots/threads) and the directory tree.
# Prints ONE JSON receipt on stdout with the home directory written as "~" (roles only).
set -euo pipefail
ROLE=""; COMMIT=""; TAR_SHA=""; BASE_PY="python3"; CAP="2G"; SLOTS="1"; THREADS="1"; STEAL="1"; WALL="12h"; FOLLOW_CAP="2G"
EXTRA_ENV=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) ROLE="$2"; shift 2 ;; --commit) COMMIT="$2"; shift 2 ;; --tar-sha256) TAR_SHA="$2"; shift 2 ;;
    --base-python) BASE_PY="$2"; shift 2 ;; --cap) CAP="$2"; shift 2 ;; --slots) SLOTS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;; --steal) STEAL="$2"; shift 2 ;; --wall) WALL="$2"; shift 2 ;;
    --follow-cap) FOLLOW_CAP="$2"; shift 2 ;; --env) EXTRA_ENV+=("$2"); shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$ROLE" && -n "$COMMIT" && -n "$TAR_SHA" ]] || { echo "--role --commit --tar-sha256 required" >&2; exit 2; }
BASE_PY="${BASE_PY/#\~/$HOME}"
STATE="$HOME/.local/state/canonical_20261003/fs_phase23"
CODE_ROOT="$STATE/code"; CODE="$CODE_ROOT/$COMMIT"; VENV="$STATE/venv"
mkdir -p "$CODE_ROOT" "$STATE"/{eurusd,eth,data,peer_terminals,peer_status,logs,warehouse}

# 1. code
TAR="$CODE_ROOT/incoming.$COMMIT.tar"
[[ -f "$TAR" ]] || { echo "archive missing: $TAR" >&2; exit 3; }
actual="$(sha256sum "$TAR" | cut -d' ' -f1)"
[[ "$actual" == "$TAR_SHA" ]] || { echo "archive digest mismatch: $actual != $TAR_SHA" >&2; exit 3; }
if [[ ! -f "$CODE/.deployed" ]]; then
  rm -rf "$CODE"; mkdir -p "$CODE_ROOT"
  tar -xf "$TAR" -C "$CODE_ROOT"      # the archive carries the prefix <COMMIT>/
  [[ -d "$CODE" ]] || { echo "archive did not produce $CODE" >&2; exit 3; }
  echo "$COMMIT" > "$CODE/.deployed"
fi
ln -sfn "$COMMIT" "$CODE_ROOT/CURRENT"
rm -f "$TAR"
tree_digest="$(cd "$CODE" && find . -type f ! -name .deployed -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"

# 2. venv
LOCK="$CODE/tools/fs_phase23_deploy/requirements.lock"
lock_sha="$(sha256sum "$LOCK" | cut -d' ' -f1)"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$BASE_PY" -m venv "$VENV"
fi
(cd "$CODE" && "$VENV/bin/python" -m pip install -q --disable-pip-version-check --require-virtualenv -r "$LOCK")
py_version="$("$VENV/bin/python" -c 'import platform; print(platform.python_version())')"
pkgs="$(grep -vE '^\s*(#|$)' "$LOCK" | cut -d= -f1 | tr 'A-Z_' 'a-z-' | sort)"
pkgs="$(printf '%s\n' "$pkgs" | grep -v '/' ; echo predictor-olap-store; echo predictor-duckdb-store)"
freeze="$("$VENV/bin/python" -m pip freeze --disable-pip-version-check 2>/dev/null | tr 'A-Z_' 'a-z-' | sed 's/ @ .*//' | grep -E "^($(echo "$pkgs" | paste -sd'|'))(==|$)" | sort)"
freeze_sha="$(printf '%s\n' "$freeze" | sha256sum | cut -d' ' -f1)"
numeric="$("$VENV/bin/python" -c 'import json,numpy,scipy,sklearn,pandas,pyarrow,duckdb; print(json.dumps({"numpy":numpy.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"pandas":pandas.__version__,"pyarrow":pyarrow.__version__,"duckdb":duckdb.__version__}))')"

# 3. units
UNITS_SRC="$CODE/tools/fs_phase23_deploy/units"
UNITS_DST="$HOME/.config/systemd/user"
mkdir -p "$UNITS_DST/fs-phase23-worker@.service.d"
install -m 0644 "$UNITS_SRC/fs-phase23-worker@.service" "$UNITS_SRC/fs-phase23-status.service" "$UNITS_SRC/fs-phase23-status.timer" "$UNITS_DST/"
if [[ "$ROLE" == "coordinator" ]]; then
  install -m 0644 "$UNITS_SRC/fs-phase23-follower@.service" "$UNITS_SRC/fs-phase23-relay.service" "$UNITS_DST/"
  install -m 0644 "$UNITS_SRC/role-coordinator.conf" "$UNITS_DST/fs-phase23-worker@.service.d/role.conf"
else
  install -m 0644 "$UNITS_SRC/role-worker.conf" "$UNITS_DST/fs-phase23-worker@.service.d/role.conf"
fi
systemctl --user daemon-reload
units_digest="$(cd "$UNITS_DST" && sha256sum fs-phase23-worker@.service fs-phase23-status.service fs-phase23-status.timer fs-phase23-worker@.service.d/role.conf $( [[ "$ROLE" == coordinator ]] && echo fs-phase23-follower@.service fs-phase23-relay.service ) | sha256sum | cut -d' ' -f1)"
unit_src_digest="$(cd "$UNITS_SRC" && sha256sum fs-phase23-worker@.service fs-phase23-status.service fs-phase23-status.timer fs-phase23-follower@.service fs-phase23-relay.service role-coordinator.conf role-worker.conf | sha256sum | cut -d' ' -f1)"

# 4. runner.env (absolute paths: systemd EnvironmentFile does not expand variables)
{
  echo "# written by deploy_host_step.sh; role-only configuration of this host for the phase 2/3 campaign"
  echo "FS23_ROLE=$ROLE"
  echo "FS23_STATE=$STATE"
  echo "FS23_CODE=$CODE_ROOT/CURRENT"
  echo "FS23_CODE_COMMIT=$COMMIT"
  echo "FS23_PYTHON=$VENV/bin/python"
  echo "FS23_DATA=$STATE/data"
  echo "FS23_POPULATIONS=eurusd eth"
  echo "FS23_WAREHOUSE=${FS23_WAREHOUSE_OVERRIDE:-$STATE/warehouse/fs_phase23_adapter.duckdb}"
  echo "FS23_CAP=$CAP"
  echo "FS23_WALL=$WALL"
  echo "FS23_THREADS=$THREADS"
  echo "FS23_SLOTS=$SLOTS"
  echo "FS23_STEAL=$STEAL"
  echo "FS23_FOLLOW_CAP=$FOLLOW_CAP"
  echo "FS23_FOLLOW_EVERY=60"
  echo "FS23_PHASE3_WORKERS=2"
  echo "FS23_ADMIT_WAIT=86400"
  echo "FS23_IDLE_SLEEP=300"
  for kv in "${EXTRA_ENV[@]}"; do echo "$kv"; done
} > "$STATE/runner.env.tmp" && mv -f "$STATE/runner.env.tmp" "$STATE/runner.env"
chmod +x "$CODE"/tools/fs_phase23_deploy/*.sh 2>/dev/null || true

python3 - "$ROLE" "$COMMIT" "$TAR_SHA" "$tree_digest" "$py_version" "$lock_sha" "$freeze_sha" "$numeric" "$units_digest" "$unit_src_digest" "$CAP" "$SLOTS" "$THREADS" "$STEAL" "$FOLLOW_CAP" "$BASE_PY" "$(nproc)" "$(uname -r)" <<'PY'
import json, os, sys, time
a = sys.argv[1:]
home = os.path.expanduser("~")
doc = {
    "schema": "fs_phase23_host_deploy_receipt.v1", "role": a[0], "commit": a[1], "archive_sha256": a[2],
    "tree_sha256": a[3], "python_version": a[4], "requirements_lock_sha256": a[5], "pip_freeze_locked_sha256": a[6],
    "numeric_stack": json.loads(a[7]), "installed_units_sha256": a[8], "unit_sources_sha256": a[9],
    "runner": {"cap": a[10], "slots": int(a[11]), "threads": int(a[12]), "steal": a[13] == "1", "follow_cap": a[14]},
    "base_python": a[15].replace(home, "~"), "cpu_cores": int(a[16]), "kernel": a[17],
    "state_dir": "~/.local/state/canonical_20261003/fs_phase23", "installed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
print(json.dumps(doc, sort_keys=True))
PY
