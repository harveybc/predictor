#!/usr/bin/env bash
# Wait for disjoint remote I6-B shards, copy them, and authenticate one closure.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if (( $# < 6 || ($# - 3) % 3 != 0 )); then
  echo "usage: $0 CONFIG OUTPUT STAGING HOST UNIT REMOTE_DIR [HOST UNIT REMOTE_DIR ...]" >&2
  exit 64
fi

config=$1
output=$2
staging=$3
shift 3
mkdir -p "$staging"

sources=()
while (( $# )); do
  host=$1 unit=$2 remote_dir=$3
  shift 3
  local_dir="$staging/$unit"
  sources+=("$local_dir")
  while :; do
    state=$(ssh "$host" "systemctl --user show '$unit' -p ActiveState --value")
    result=$(ssh "$host" "systemctl --user show '$unit' -p Result --value")
    case "$state:$result" in
      active:*|activating:*) sleep 30 ;;
      inactive:success|inactive:)
        mkdir -p "$local_dir"
        rsync -a --delete --exclude='.*.work/' "$host:$remote_dir/" "$local_dir/"
        break
        ;;
      *) echo "remote shard failed: $host $unit state=$state result=$result" >&2; exit 1 ;;
    esac
  done
done

args=()
for source in "${sources[@]}"; do args+=(--shard "$source"); done
cd "$repo_root"
python3 -m tools.i6b_close --config "$config" "${args[@]}" --output "$output"
