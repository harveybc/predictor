# Local experiment output storage

On omega, new large generated experiment outputs use `/mnt/windows/experiment-logs`.
The local user environment advertises this through `EXPERIMENT_STORAGE_ROOT`.
The variable is a path convention; each application must explicitly route its
output there. Existing inputs, repositories, worktrees and live service data
are not moved by this policy.

## Layout

- `i6dmo/`: immutable run artifacts and retained predictions/checkpoints.
- `financial-data/experiments/`: future experiment outputs.
- `financial-data/features/`: future materialized feature outputs.
- `crispdm/`: future generated CRISP-DM run artifacts.
- `agent-multi/`: future generated campaign artifacts.

Keep source code and Git repositories on ext4. Keep active SQLite/PostgreSQL/
DuckDB databases, POSIX lock files, private deployment specs, credentials and
live mutable campaign state on ext4. For I6-DMO, the coordinator root and
`state/*.sqlite3` remain on ext4; archive large immutable terminal artifacts
to the NTFS output path only through a versioned, digest-verified storage
adapter. Do not make the campaign root or its SQLite state a directory on
NTFS. Remote workers keep their own scratch roots and return authenticated
terminals to the coordinator before archival.

Before a workload starts, pass its conservative output-size estimate and
reserve to `tools/experiment_storage_preflight.sh`. It fails closed if the
volume is absent, mounted read-only, unwritable, the target is outside the
approved tree, or free space is below estimate plus reserve. It also performs
a temporary write and sync. A failed check means the new output workload
must not start or silently fall back to `/home`.

Current mount observed 2026-10-15: NTFS at `/mnt/windows`, read/write, 711 GB
available. This is a point-in-time observation; every workload rechecks.
Windows must remain shut down without Fast Startup. A future read-only mount
is a hard refusal, not a reason to use a different partition implicitly.
