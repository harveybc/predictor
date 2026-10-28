# Coordinator Memory Recovery

2026-10-19, post-owner restart. Previous-boot kernel log identifies the killed
codebase-memory process at anonymous RSS22382132KiB (about21.3GiB). It coincides
with the auditor's unbounded tools index attempt, which timed out. Do not
attribute this OOM to the SAC experiment or to Satoshi's evaluator.

Applied locally: each MCP server launches in a user systemd scope with2GiB
MemoryHigh,3GiB MemoryMax and256MiB swap. All such scopes share a slice with
4GiB MemoryHigh,6GiB MemoryMax,512MiB swap and150% CPU. All three known client
configs use the bounded launcher. The two already running server PIDs were
attached to bounded scopes without restarting their clients; cgroup membership
and limits verified. Bounded --version returns0.9.0. No unbounded index rerun.

Post-reboot: about24GiB RAM available, swap unused; /home83GiB free, archive
675GiB free and mounted. Warehouse active, collection timers resumed. Retired
multihorizon status timer pointed to a missing queue and was disabled. The old
native acquisition service tried to reuse a changed identity with all six jobs
already terminal FAILED; disabled its obsolete boot restart, retaining all
statuses/errors. Its separate gap-recovery/publication timers remain active.
This is not a claim that the broader external data catalogue is admitted.

The preferred GPU worker remained up: L2 and combination closed with42/42 warehouse deliveries
each; BatchNorm started automatically. One heavy worker; other GPU not used
as a second training slot. The secondary GPU worker answered health checks with14GiB RAM available,
515GiB disk free and a visible4090; its existing swap occupancy was3.9GiB,
without proof of active memory pressure. No worker restart was required.

An unrelated social-enrichment service subsequently reported failure; its own
timer remains enabled. This recovery does not certify that application's health
or change its behavior. Core warehouse, research and collection paths were checked.
