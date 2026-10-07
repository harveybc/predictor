#!/usr/bin/env bash
# Phase B: publish this repository's worker code and the pinned runner wrapper on ONE worker.
#
#   deploy_runner.sh --ssh ALIAS --predictor-commit SHA --fe-commit FULL_SHA --python PY
#                    [--ld-file FILE] --input ROLE=PATH ... [--state ~/.local/state/canonical_20261003/fs4]
#
# The feature-extractor checkout <state>/code/fe_<first8 of FE_COMMIT> must already exist on the worker as a
# git clone whose HEAD is exactly FE_COMMIT and whose tree is clean (so the runner's code_commit is real).
# Steps (idempotent): `git archive` of the predictor commit -> <state>/code/predictor-<SHA>/ (tree digest),
# PREDICTOR_CURRENT symlink; <state>/bin/fs4-runner written: it answers `--self-check` with JSON naming the
# pinned code_commit and the interpreter's pins, and otherwise exports FS4_RUNNER_PYTHON,
# FS4_LD_LIBRARY_PATH_FILE and FS4_INPUT_<ROLE> then execs <fe>/tools/fs4_runner.sh. Prints ONE JSON receipt
# (home written as "~"): both workers must show the same fe commit and the same pins digest for parity.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$(readlink -f -- "$0")")" && pwd)"
PRED_REPO="$(cd "$HERE/../.." && pwd)"
SSH=""; PRED_COMMIT=""; FE_COMMIT=""; PY=""; LD_FILE=""; STATE="~/.local/state/canonical_20261003/fs4"; INPUTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh) SSH="$2"; shift 2 ;; --predictor-commit) PRED_COMMIT="$2"; shift 2 ;; --fe-commit) FE_COMMIT="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;; --ld-file) LD_FILE="$2"; shift 2 ;; --state) STATE="$2"; shift 2 ;;
    --input) INPUTS+=("$2"); shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$SSH" && -n "$PRED_COMMIT" && "$FE_COMMIT" =~ ^[0-9a-f]{40}$ && -n "$PY" && ${#INPUTS[@]} -gt 0 ]] || { echo "--ssh --predictor-commit --fe-commit(40 hex) --python --input required" >&2; exit 2; }
PRED_COMMIT="$(git -C "$PRED_REPO" rev-parse --verify "$PRED_COMMIT^{commit}")"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/fs4-deploy.XXXXXX")"; trap 'rm -rf "$WORK"' EXIT
git -C "$PRED_REPO" archive --format=tar --prefix="predictor-$PRED_COMMIT/" "$PRED_COMMIT" > "$WORK/pred.tar"
PRED_SHA="$(sha256sum "$WORK/pred.tar" | cut -d' ' -f1)"
ssh -o BatchMode=yes -o ConnectTimeout=20 "$SSH" "mkdir -p $STATE/code && cat > $STATE/code/incoming.pred.tar" < "$WORK/pred.tar"
ssh -o BatchMode=yes -o ConnectTimeout=20 "$SSH" bash -s -- "$FE_COMMIT" "$PRED_COMMIT" "$PRED_SHA" "$PY" "$LD_FILE" "$STATE" "${INPUTS[@]}" <<'REMOTE'
set -euo pipefail
FE="$1"; PRED="$2"; PRED_SHA="$3"; PY="${4/#\~/$HOME}"; LDF="${5/#\~/$HOME}"; STATE="${6/#\~/$HOME}"; shift 6
CODE="$STATE/code"; BIN="$STATE/bin"; FEDIR="$CODE/fe_${FE:0:8}"
mkdir -p "$BIN" "$STATE/terminals" "$STATE/logs"
[[ "$(git -C "$FEDIR" rev-parse HEAD)" == "$FE" ]] || { echo "feature-extractor checkout is not at $FE" >&2; exit 3; }
[[ -z "$(git -C "$FEDIR" status --porcelain --untracked-files=no)" ]] || { echo "feature-extractor checkout is not clean" >&2; exit 3; }
[[ -x "$FEDIR/tools/fs4_runner.sh" ]] || { echo "runner wrapper missing in checkout" >&2; exit 3; }
test -x "$PY" || { echo "python not executable: $PY" >&2; exit 3; }
tar_path="$CODE/incoming.pred.tar"
[[ "$(sha256sum "$tar_path" | cut -d' ' -f1)" == "$PRED_SHA" ]] || { echo "predictor archive digest mismatch" >&2; exit 3; }
dir="$CODE/predictor-$PRED"
if [[ ! -f "$dir/.deployed" ]]; then rm -rf "$dir"; tar -xf "$tar_path" -C "$CODE"; echo "$PRED" > "$dir/.deployed"; fi
rm -f "$tar_path"
ln -sfn "predictor-$PRED" "$CODE/PREDICTOR_CURRENT"
tree="$(cd "$dir" && find . -type f ! -name .deployed -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
pins="$("$PY" -m pip freeze --disable-pip-version-check 2>/dev/null | tr 'A-Z_' 'a-z-' | grep -E '^(tensorflow|keras|numpy|pandas|pyarrow|scipy|scikit-learn|h5py|ml-dtypes)==' | sort | tr '\n' ' ')"
pins_sha="$(printf '%s' "$pins" | sha256sum | cut -d' ' -f1)"
pyv="$("$PY" -c 'import platform; print(platform.python_version())')"
{
  echo '#!/usr/bin/env bash'
  echo "# pinned FS4 runner: feature-extractor@$FE (written by deploy_runner.sh)"
  echo 'set -euo pipefail'
  echo 'if [[ "${1:-}" == "--self-check" ]]; then'
  echo "  printf '{\"schema\":\"fs4.runner_self_check.v1\",\"code_commit\":\"%s\",\"python\":\"%s\",\"pins_sha256\":\"%s\"}\\n' \"\$(git -C '$FEDIR' rev-parse HEAD)\" '$pyv' '$pins_sha'"
  echo '  exit 0'
  echo 'fi'
  echo "export FS4_RUNNER_PYTHON='$PY'"
  [[ -n "$LDF" ]] && echo "export FS4_LD_LIBRARY_PATH_FILE='$LDF'"
  for item in "$@"; do role="${item%%=*}"; path="${item#*=}"; echo "export FS4_INPUT_$(printf '%s' "$role" | tr 'a-z' 'A-Z')='${path/#\~/$HOME}'"; done
  echo "exec '$FEDIR/tools/fs4_runner.sh' \"\$@\""
} > "$BIN/fs4-runner.tmp"
chmod 755 "$BIN/fs4-runner.tmp"; mv -f "$BIN/fs4-runner.tmp" "$BIN/fs4-runner"
"$BIN/fs4-runner" --self-check >/dev/null
python3 - "$FE" "$PRED" "$tree" "$pyv" "$pins_sha" "$pins" "$BIN/fs4-runner" "$STATE" <<'PY'
import json, os, sys, time
a = sys.argv[1:]; home = os.path.expanduser("~")
print(json.dumps({"schema": "fs4_runner_deploy_receipt.v2", "fe_commit": a[0], "predictor_commit": a[1],
    "predictor_tree_sha256": a[2], "python_version": a[3], "pins_sha256": a[4], "pins": a[5].split(),
    "runner": a[6].replace(home, "~"), "state_dir": a[7].replace(home, "~"),
    "installed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, sort_keys=True))
PY
REMOTE
