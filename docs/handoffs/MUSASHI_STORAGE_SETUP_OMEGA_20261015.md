# Satoshi: use omega's new experiment storage

The user prepared `/mnt/windows/experiment-logs` on omega (711 GB free at the
2026-10-15 check). The ext4 home filesystem has only 15 GB free. Storage
directories exist for `i6dmo`, `financial-data/experiments`,
`financial-data/features`, `crispdm`, and `agent-multi`.

Musashi configured `EXPERIMENT_STORAGE_ROOT=/mnt/windows/experiment-logs` in
`~/.config/environment.d/90-experiment-storage.conf` and imported it into the
current systemd user manager. No existing service was restarted. The bounded
preflight performed a real temporary write+sync and observed 763,253,288,960
free bytes at execution; repeat the check for every workload.

Read `docs/EXPERIMENT_STORAGE_POLICY.md`. The environment variable
`EXPERIMENT_STORAGE_ROOT` advertises the base path to new services. The
versioned preflight is `tools/experiment_storage_preflight.sh`; require it to
pass immediately before each new output workload with that workload's
conservative output estimate and reserve. Do not start or fall back to `/home`
if it fails.

For the next I6-DMO campaign, keep mutable campaign root, queue/outbox
SQLite, claims/locks, deployment secrets and private specs on ext4. Add a
versioned storage adapter that routes large immutable terminal artifacts
(checkpoints, validation prediction arrays, trainer output) to
`$EXPERIMENT_STORAGE_ROOT/i6dmo/<campaign-id>/...` after they return to omega.
The adapter must verify hashes before and after copy, preserve relative-path
identity in terminal receipts, support restart/re-adoption, publish/mirror the
archival path, and refuse failed/read-only mounts. Never use NTFS as a SQLite,
PostgreSQL or DuckDB live database filesystem. Remote worker scratch remains on
each worker's own configured filesystem; this omega partition is not mounted
on Gamma or Dragon.

Route future outputs for other applications to their matching subdirectories
only through each repository's supported output configuration. Do not move
historical data, running campaign roots, feature sources, CRISP-DM databases,
agent-multi campaigns, Git repositories or worktrees in this change. Do not
rewrite a running service's environment or interrupt active work.

Integrate this storage policy with the approved MG2/MG12 successor while
implementing its campaign lifecycle. Run storage checks before init/deploy and
again before a job claims an artifact-writing unit. Report local vs worker
disk separately in preflight and status, including `UNAVAILABLE` if a worker
cannot report disk. Do not treat omega's 711 GB as free space on another host.

Return: paths/config changes, owners, PRE/POST, proof that live DB paths remain
on ext4, failure-injection result for a read-only/missing mount, artifact hash
round-trip, disk estimates and one finite status command. Do not monitor logs.
