#!/usr/bin/env bash
set -euo pipefail

base=/mnt/windows/experiment-logs
storage_path=${1:-$base}
required_bytes=${2:?usage: experiment_storage_preflight.sh [path-under-experiment-logs] required_bytes [reserve_bytes]}
reserve_bytes=${3:-5368709120}

for value in "$required_bytes" "$reserve_bytes"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { printf 'INVALID_BYTE_COUNT\n' >&2; exit 2; }
done
case "$storage_path" in
    "$base"|"$base"/*) ;;
    *) printf 'PATH_OUTSIDE_EXPERIMENT_STORAGE\n' >&2; exit 2 ;;
esac

[[ -d "$storage_path" ]] || { printf 'OUTPUT_DIRECTORY_MISSING\n' >&2; exit 2; }
[[ -w "$storage_path" ]] || { printf 'OUTPUT_DIRECTORY_NOT_WRITABLE\n' >&2; exit 2; }

mount_target=$(findmnt -rn -T "$storage_path" -o TARGET)
mount_type=$(findmnt -rn -T "$storage_path" -o FSTYPE)
mount_options=$(findmnt -rn -T "$storage_path" -o OPTIONS)
[[ "$mount_target" == /mnt/windows ]] || { printf 'WRONG_FILESYSTEM_TARGET\n' >&2; exit 3; }
[[ "$mount_type" == fuseblk || "$mount_type" == ntfs3 ]] || { printf 'WRONG_FILESYSTEM_TYPE\n' >&2; exit 3; }
[[ ",$mount_options," == *,rw,* ]] || { printf 'FILESYSTEM_NOT_RW\n' >&2; exit 3; }

available_bytes=$(df -PB1 "$storage_path" | awk 'NR == 2 {print $4}')
[[ "$available_bytes" =~ ^[0-9]+$ ]] || { printf 'FREE_SPACE_UNREADABLE\n' >&2; exit 4; }
(( available_bytes >= required_bytes + reserve_bytes )) || {
    printf 'INSUFFICIENT_SPACE available=%s required=%s reserve=%s\n' \
        "$available_bytes" "$required_bytes" "$reserve_bytes" >&2
    exit 5
}

umask 077
probe=$(mktemp "$storage_path/.preflight.XXXXXXXX")
trap 'rm -f -- "$probe"' EXIT
printf 'i6dmo-storage-preflight\n' > "$probe"
sync -f "$probe"
rm -f -- "$probe"
trap - EXIT
printf 'STORAGE_READY path=%s type=%s available_bytes=%s required_bytes=%s reserve_bytes=%s\n' \
    "$storage_path" "$mount_type" "$available_bytes" "$required_bytes" "$reserve_bytes"
