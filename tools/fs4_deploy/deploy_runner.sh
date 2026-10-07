#!/usr/bin/env bash
# Phase B: publish the pinned feature-extractor runner on one worker with an isolated env.
#
#   deploy_runner.sh --ssh ALIAS --fe-repo PATH --commit SHA --predictor-commit SHA
#                    [--base-python ~/anaconda3/envs/tensorflow/bin/python] [--state ~/.local/state/canonical_20261003/fs4]
#
# Steps on the worker (all idempotent, roles only in anything written back here):
#   1. `git archive` of the feature-extractor commit and of the predictor commit (this kit)
#      -> <state>/code/fe-<SHA>/ and <state>/code/predictor-<SHA>/, each with a .deployed marker and a
#      tree digest (sha256 over the sorted sha256 of every file), CURRENT symlinks pointed at them;
#   2. <state>/venv built from the base interpreter and `pip install -r tools/fs4_deploy/requirements.lock`
#      (exactly), `pip freeze` digest recorded;
#   3. <state>/bin/fs4-runner written: an executable wrapper that answers `--self-check` with the code
#      commit and otherwise execs `<venv>/bin/python -m app.fs4_task_runner` inside the fe checkout with
#      LD_LIBRARY_PATH set from the venv's nvidia wheels. The wrapper re-reads FS4_EXPECTED_GPU_UUID so the
#      child can verify its device; it never clears CUDA_VISIBLE_DEVICES.
# Prints ONE JSON receipt on stdout (home written as "~").
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$(readlink -f -- "$0")")" && pwd)"
PRED_REPO="$(cd "$HERE/../.." && pwd)"
SSH=""; FE_REPO=""; COMMIT=""; PRED_COMMIT=""; BASE_PY="~/anaconda3/envs/tensorflow/bin/python"; STATE="~/.local/state/canonical_20261003/fs4"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh) SSH="$2"; shift 2 ;; --fe-repo) FE_REPO="$2"; shift 2 ;; --commit) COMMIT="$2"; shift 2 ;;
    --predictor-commit) PRED_COMMIT="$2"; shift 2 ;; --base-python) BASE_PY="$2"; shift 2 ;; --state) STATE="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$SSH" && -n "$FE_REPO" && -n "$COMMIT" && -n "$PRED_COMMIT" ]] || { echo "--ssh --fe-repo --commit --predictor-commit required" >&2; exit 2; }
COMMIT="$(git -C "$FE_REPO" rev-parse --verify "$COMMIT^{commit}")"
PRED_COMMIT="$(git -C "$PRED_REPO" rev-parse --verify "$PRED_COMMIT^{commit}")"
git -C "$FE_REPO" cat-file -e "$COMMIT:app/fs4_task_runner.py" || { echo "feature-extractor@$COMMIT has no app/fs4_task_runner.py" >&2; exit 3; }
WORK="$(mktemp -d "${TMPDIR:-/tmp}/fs4-deploy.XXXXXX")"; trap 'rm -rf "$WORK"' EXIT
git -C "$FE_REPO" archive --format=tar --prefix="fe-$COMMIT/" "$COMMIT" > "$WORK/fe.tar"
git -C "$PRED_REPO" archive --format=tar --prefix="predictor-$PRED_COMMIT/" "$PRED_COMMIT" > "$WORK/pred.tar"
FE_SHA="$(sha256sum "$WORK/fe.tar" | cut -d' ' -f1)"; PRED_SHA="$(sha256sum "$WORK/pred.tar" | cut -d' ' -f1)"
ssh -o BatchMode=yes -o ConnectTimeout=20 "$SSH" "mkdir -p $STATE/code && cat > $STATE/code/incoming.fe.tar" < "$WORK/fe.tar"
ssh -o BatchMode=yes -o ConnectTimeout=20 "$SSH" "cat > $STATE/code/incoming.pred.tar" < "$WORK/pred.tar"
ssh -o BatchMode=yes -o ConnectTimeout=20 "$SSH" bash -s -- "$COMMIT" "$FE_SHA" "$PRED_COMMIT" "$PRED_SHA" "$BASE_PY" "$STATE" <<'REMOTE'
set -euo pipefail
COMMIT="$1"; FE_SHA="$2"; PRED_COMMIT="$3"; PRED_SHA="$4"; BASE_PY="${5/#\~/$HOME}"; STATE="${6/#\~/$HOME}"
CODE="$STATE/code"; VENV="$STATE/venv"; BIN="$STATE/bin"
mkdir -p "$CODE" "$BIN" "$STATE/terminals" "$STATE/logs"
extract() {  # prefix sha tar
  local dir="$CODE/$1" tar="$CODE/$3"
  [[ "$(sha256sum "$tar" | cut -d' ' -f1)" == "$2" ]] || { echo "archive digest mismatch for $1" >&2; exit 3; }
  if [[ ! -f "$dir/.deployed" ]]; then rm -rf "$dir"; tar -xf "$tar" -C "$CODE"; echo "$1" > "$dir/.deployed"; fi
  rm -f "$tar"
  (cd "$dir" && find . -type f ! -name .deployed -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}
fe_tree="$(extract "fe-$COMMIT" "$FE_SHA" incoming.fe.tar)"
pred_tree="$(extract "predictor-$PRED_COMMIT" "$PRED_SHA" incoming.pred.tar)"
ln -sfn "fe-$COMMIT" "$CODE/FE_CURRENT"; ln -sfn "predictor-$PRED_COMMIT" "$CODE/PREDICTOR_CURRENT"
LOCK="$CODE/predictor-$PRED_COMMIT/tools/fs4_deploy/requirements.lock"
[[ -x "$VENV/bin/python" ]] || "$BASE_PY" -m venv "$VENV"
"$HOME/.local/bin/crispdm-run" -q -m 4G -t 45m -n fs4-venv-lock -- "$VENV/bin/python" -m pip install -q --disable-pip-version-check --require-virtualenv -r "$LOCK" >"$STATE/logs/venv_lock.log" 2>&1
lock_sha="$(sha256sum "$LOCK" | cut -d' ' -f1)"
freeze_sha="$("$VENV/bin/python" -m pip freeze --disable-pip-version-check 2>/dev/null | sort | sha256sum | cut -d' ' -f1)"
py_version="$("$VENV/bin/python" -c 'import platform; print(platform.python_version())')"
ldp="$("$VENV/bin/python" - <<'PY'
import glob, os, site
roots = []
for sp in site.getsitepackages():
    roots.extend(sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib"))))
print(":".join(roots))
PY
)"
cat > "$BIN/fs4-runner.tmp" <<EOF
#!/usr/bin/env bash
# pinned FS4 runner: feature-extractor@$COMMIT under $VENV (written by deploy_runner.sh)
set -euo pipefail
if [[ "\${1:-}" == "--self-check" ]]; then
  printf '{"schema":"fs4.runner_self_check.v1","code_commit":"%s","fe_tree_sha256":"%s","python":"%s","venv_freeze_sha256":"%s"}\n' "$COMMIT" "$fe_tree" "$py_version" "$freeze_sha"
  exit 0
fi
export LD_LIBRARY_PATH="$ldp\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export TF_CPP_MIN_LOG_LEVEL="\${TF_CPP_MIN_LOG_LEVEL:-2}"
export FS4_CODE_COMMIT="$COMMIT"
cd "$CODE/fe-$COMMIT"
exec "$VENV/bin/python" -m app.fs4_task_runner "\$@"
EOF
chmod 755 "$BIN/fs4-runner.tmp"; mv -f "$BIN/fs4-runner.tmp" "$BIN/fs4-runner"
"$BIN/fs4-runner" --self-check >/dev/null
python3 - "$COMMIT" "$fe_tree" "$PRED_COMMIT" "$pred_tree" "$py_version" "$lock_sha" "$freeze_sha" "$BIN/fs4-runner" "$STATE" <<'PY'
import json, os, sys, time
a = sys.argv[1:]
home = os.path.expanduser("~")
doc = {"schema": "fs4_runner_deploy_receipt.v1", "fe_commit": a[0], "fe_tree_sha256": a[1],
       "predictor_commit": a[2], "predictor_tree_sha256": a[3], "python_version": a[4],
       "requirements_lock_sha256": a[5], "pip_freeze_sha256": a[6], "runner": a[7].replace(home, "~"),
       "state_dir": a[8].replace(home, "~"), "installed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
print(json.dumps(doc, sort_keys=True))
PY
REMOTE
