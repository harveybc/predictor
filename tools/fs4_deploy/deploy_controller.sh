#!/usr/bin/env bash
# Place one commit of this repository on the COORDINATOR as the controller the workers call
# over SSH (FS4_CONTROLLER=<state>/code/PREDICTOR_CURRENT/tools/fs4_campaign.py).
#
#   deploy_controller.sh --commit SHA [--state ~/.local/state/canonical_20261003/fs4]
#
# `git archive` -> <state>/code/predictor-<SHA>/ (tree digest recorded) -> PREDICTOR_CURRENT symlink.
# The controller is stdlib-only and runs with the coordinator's python3; no venv is built here.
# Prints ONE JSON receipt (home written as "~").
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$(readlink -f -- "$0")")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
COMMIT=""; STATE="$HOME/.local/state/canonical_20261003/fs4"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit) COMMIT="$2"; shift 2 ;; --state) STATE="${2/#\~/$HOME}"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$COMMIT" ]] || { echo "--commit required" >&2; exit 2; }
COMMIT="$(git -C "$REPO" rev-parse --verify "$COMMIT^{commit}")"
CODE="$STATE/code"; DIR="$CODE/predictor-$COMMIT"
mkdir -p "$CODE"
if [[ ! -f "$DIR/.deployed" ]]; then
  rm -rf "$DIR"
  git -C "$REPO" archive --format=tar --prefix="predictor-$COMMIT/" "$COMMIT" | tar -xf - -C "$CODE"
  echo "$COMMIT" > "$DIR/.deployed"
fi
ln -sfn "predictor-$COMMIT" "$CODE/PREDICTOR_CURRENT"
tree="$(cd "$DIR" && find . -type f ! -name .deployed -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
python3 "$DIR/tools/fs4_campaign.py" --db "$STATE/queue_v2.sqlite" status >/dev/null
printf '{"schema":"fs4_controller_deploy_receipt.v1","commit":"%s","tree_sha256":"%s","controller":"%s","db":"%s"}\n' \
  "$COMMIT" "$tree" "${CODE/#$HOME/\~}/PREDICTOR_CURRENT/tools/fs4_campaign.py" "${STATE/#$HOME/\~}/queue_v2.sqlite"
