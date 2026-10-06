#!/usr/bin/env bash
# Deploy ONE commit of this repository to the three roles of the phase 2/3 selection campaign
# (§E.2) and write a deployment receipt that proves parity.
#
#   deploy.sh --commit SHA --hosts-env FILE --receipt-dir DIR
#             [--roles coordinator,worker_a,worker_b] [--assignment assignment.json]
#             [--cap-coordinator 1G] [--cap-worker 2G] [--threads 2] [--wall 6h]
#             [--start] [--repo PATH]
#
# hosts.env is UNTRACKED (lives in the state directory) and defines the ssh aliases
# WORKER_A_SSH / WORKER_B_SSH and optionally COORD_PYTHON / WORKER_A_PYTHON / WORKER_B_PYTHON
# (the base interpreter the venv is built from; defaults: python3 on the coordinator,
# ~/anaconda3/envs/tensorflow/bin/python on the workers). The repository names roles only.
#
# Steps: `git archive` of the exact commit (one tar, one digest) -> placed on every role ->
# deploy_host_step.sh on every role (extract, venv from requirements.lock, units, runner.env)
# -> receipts compared: archive, tree, lock, locked-freeze and unit digests must be equal on
# every role or the script exits 4. Slots per role come from --assignment when given (the shard
# policy's output), else 1. With --start the units are enabled and started (phase B only):
# the relay first, then the status timer and the worker slots, then the follower.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$(readlink -f -- "$0")")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
COMMIT=""; HOSTS_ENV=""; RECEIPT_DIR=""; ROLES="coordinator,worker_a,worker_b"; ASSIGNMENT=""
CAP_COORD="1G"; CAP_WORKER="2G"; THREADS="2"; WALL="6h"; START=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit) COMMIT="$2"; shift 2 ;; --hosts-env) HOSTS_ENV="$2"; shift 2 ;; --receipt-dir) RECEIPT_DIR="$2"; shift 2 ;;
    --roles) ROLES="$2"; shift 2 ;; --assignment) ASSIGNMENT="$2"; shift 2 ;; --cap-coordinator) CAP_COORD="$2"; shift 2 ;;
    --cap-worker) CAP_WORKER="$2"; shift 2 ;; --threads) THREADS="$2"; shift 2 ;; --wall) WALL="$2"; shift 2 ;;
    --start) START=1; shift ;; --repo) REPO="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$COMMIT" && -n "$HOSTS_ENV" && -n "$RECEIPT_DIR" ]] || { echo "--commit --hosts-env --receipt-dir required" >&2; exit 2; }
# shellcheck disable=SC1090
source "$HOSTS_ENV"
: "${WORKER_A_SSH:?}" "${WORKER_B_SSH:?}"
COMMIT="$(git -C "$REPO" rev-parse --verify "$COMMIT^{commit}")"
STATE="$HOME/.local/state/canonical_20261003/fs_phase23"
REMOTE_STATE=".local/state/canonical_20261003/fs_phase23"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/fs23-deploy.XXXXXX")"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$RECEIPT_DIR" "$STATE/code"

git -C "$REPO" archive --format=tar --prefix="$COMMIT/" "$COMMIT" > "$WORK/code.tar"
TAR_SHA="$(sha256sum "$WORK/code.tar" | cut -d' ' -f1)"
STEP="$REPO/tools/fs_phase23_deploy/deploy_host_step.sh"

slots_for() {  # role -> slots from the assignment, else 1
  if [[ -n "$ASSIGNMENT" && -f "$ASSIGNMENT" ]]; then
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["roles"][sys.argv[2]]["slots"])' "$ASSIGNMENT" "$1"
  else echo 1; fi
}
declare -A RECEIPT=() RC=()
IFS=',' read -r -a ROLE_LIST <<< "$ROLES"
for role in "${ROLE_LIST[@]}"; do
  case "$role" in
    coordinator) ssh_alias=""; base_py="${COORD_PYTHON:-python3}"; cap="$CAP_COORD"; steal=0 ;;
    worker_a)    ssh_alias="$WORKER_A_SSH"; base_py="${WORKER_A_PYTHON:-~/anaconda3/envs/tensorflow/bin/python}"; cap="$CAP_WORKER"; steal=1 ;;
    worker_b)    ssh_alias="$WORKER_B_SSH"; base_py="${WORKER_B_PYTHON:-~/anaconda3/envs/tensorflow/bin/python}"; cap="$CAP_WORKER"; steal=1 ;;
    *) echo "unknown role $role" >&2; exit 2 ;;
  esac
  slots="$(slots_for "$role")"
  args=(--role "$role" --commit "$COMMIT" --tar-sha256 "$TAR_SHA" --base-python "$base_py" --cap "$cap" --slots "$slots" --threads "$THREADS" --steal "$steal" --wall "$WALL" --follow-cap "$CAP_COORD")
  echo "== $role: placing archive and running the host step" >&2
  if [[ -z "$ssh_alias" ]]; then
    cp "$WORK/code.tar" "$STATE/code/incoming.$COMMIT.tar"
    out="$(bash "$STEP" "${args[@]}" 2>"$WORK/$role.err")"; RC[$role]=$?
  else
    ssh -o ConnectTimeout=20 "$ssh_alias" "mkdir -p $REMOTE_STATE/code && cat > $REMOTE_STATE/code/incoming.$COMMIT.tar" < "$WORK/code.tar"
    out="$(ssh -o ConnectTimeout=20 "$ssh_alias" bash -s -- "${args[@]}" < "$STEP" 2>"$WORK/$role.err")"; RC[$role]=$?
  fi
  if [[ "${RC[$role]}" -ne 0 ]]; then echo "host step FAILED on $role (rc ${RC[$role]}):" >&2; tail -20 "$WORK/$role.err" >&2; fi
  RECEIPT[$role]="$out"
done

# parity + receipt
python3 - "$COMMIT" "$TAR_SHA" "$RECEIPT_DIR" "$(for r in "${ROLE_LIST[@]}"; do printf '%s\t%s\t%s\n' "$r" "${RC[$r]}" "${RECEIPT[$r]}"; done)" <<'PY'
import json, pathlib, sys, time
commit, tar_sha, receipt_dir, rows = sys.argv[1:5]
hosts, failures = {}, {}
for line in rows.splitlines():
    role, rc, body = line.split("\t", 2)
    if rc != "0" or not body.strip():
        failures[role] = f"host step rc={rc}"
        continue
    hosts[role] = json.loads(body)
keys = ("archive_sha256", "tree_sha256", "requirements_lock_sha256", "pip_freeze_locked_sha256", "unit_sources_sha256")
parity = {k: len({h[k] for h in hosts.values()}) == 1 for k in keys} if hosts else {k: False for k in keys}
parity["python_version_equal"] = len({h["python_version"] for h in hosts.values()}) == 1 if hosts else False
parity["numeric_stack_equal"] = len({json.dumps(h["numeric_stack"], sort_keys=True) for h in hosts.values()}) == 1 if hosts else False
required_ok = all(parity[k] for k in keys) and not failures
doc = {
    "schema": "fs_phase23_deploy_receipt.v1", "commit": commit, "archive_sha256": tar_sha,
    "deployed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "roles": hosts, "failures": failures, "parity": parity,
    "parity_required": list(keys) + ["numeric_stack_equal"],
    "parity_ok": required_ok and parity["numeric_stack_equal"],
    "note": "python_version may differ at patch level across roles (same minor 3.12); the locked numeric stack must be identical",
}
out = pathlib.Path(receipt_dir) / f"DEPLOY_RECEIPT_{commit[:12]}.json"
out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
print(f"receipt: {out}")
for role, h in hosts.items():
    print(f"{role}: tree {h['tree_sha256'][:12]} freeze {h['pip_freeze_locked_sha256'][:12]} py {h['python_version']} units {h['installed_units_sha256'][:12]} slots {h['runner']['slots']} cap {h['runner']['cap']}")
for role, why in failures.items():
    print(f"{role}: FAILED {why}")
print("parity:", json.dumps(parity))
sys.exit(0 if doc["parity_ok"] else 4)
PY
rc=$?
[[ $rc -eq 0 ]] || exit $rc

if [[ $START -eq 1 ]]; then
  echo "== starting units" >&2
  for role in "${ROLE_LIST[@]}"; do
    slots="$(slots_for "$role")"
    units="fs-phase23-status.timer"
    for ((i = 1; i <= slots; i++)); do units="$units fs-phase23-worker@$i.service"; done
    cmd="systemctl --user daemon-reload && systemctl --user enable --now $units"
    case "$role" in
      coordinator) systemctl --user daemon-reload && systemctl --user enable --now fs-phase23-relay.service && eval "$cmd" && systemctl --user enable --now fs-phase23-follower.service ;;
      worker_a) ssh -o ConnectTimeout=20 "$WORKER_A_SSH" "$cmd" ;;
      worker_b) ssh -o ConnectTimeout=20 "$WORKER_B_SSH" "$cmd" ;;
    esac
  done
fi
