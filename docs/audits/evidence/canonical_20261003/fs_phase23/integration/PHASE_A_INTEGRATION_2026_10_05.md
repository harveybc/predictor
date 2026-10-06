# Phase 2/3 selection: integration lane, Phase A (2026-10-05)

Order: `MUSASHI_TO_SATOSHI_FS_PHASE2_PHASE3_AUTOMATED_2026_10_05.md` §A, §E, §F, §G and
`program_v3/FEATURE_SELECTION_PHASE2_PHASE3_WORK_PLAN_2026_10_05.md` §5. Branch
`satoshi/fs-phase23-integration-20261005` from `origin/satoshi/canonical-exec-20261003`
(eb9df9eb). Write set: `tools/fs_phase23_deploy/**` and this directory. The driver
(engineering agent) and the warehouse module (data agent) are NOT in this branch; the
units call them by the command lines recorded in §6 below, which are the only adaptation
points for Phase B.

Roles only. Hosts are `coordinator`, `worker_a`, `worker_b`; ssh aliases live in the
untracked `~/.local/state/canonical_20261003/fs_phase23/hosts.env`.

## 1. Health check and available memory (§E.1) — measured, no GPU used, nothing stopped

Probe: `tools/fs_phase23_deploy/host_health.sh ROLE [python ...]`, run on each role; the
three documents are in `health/` (home directory written as `~`, no host names).

| role | cores | load1 | MemTotal | MemAvailable | SUnreclaim | batch slice cur / high / max | admission free-for-new | live leases | pressure some10 | GPU (used/total MiB, read-only) |
|---|---|---|---|---|---|---|---|---|---|---|
| coordinator | 16 | 1.12 | 30.53 GiB | 20.27 GiB | 0.74 GiB | 0.04 / 12 / 14 GiB | 17.26 GiB | 0 | 0.0 | 706 / 8188 |
| worker_a | 32 | 0.28 | 14.29 GiB | 9.49 GiB | 1.35 GiB | 0.02 / 7 / 8 GiB | 6.17 GiB | 0 | 0.0 | 14 / 12227 and 48 / 32607 |
| worker_b | 32 | 0.32 | 30.58 GiB | 16.17 GiB | 7.93 GiB | 0.02 / 18 / 20 GiB | 12.82 GiB | 0 | 0.0 | 14 / 16376 |

Measured state versus the order's description: at measurement time (2026-10-06 00:55Z)
no role held a live crispdm lease, both batch slices were at ~0.02 GiB, the worker GPUs
were at idle memory (no PS3-R 5090 cell, no 4090 cell), and `fs-pred-runner.service` on
worker_b was `inactive`. The slice ceilings the order names are confirmed (worker_a 7G/8G,
worker_b 18G/20G widened by drop-in). worker_b's unreclaimable slab is 7.93 GiB (the
known slab growth). Nothing was started, stopped or restarted.

Python / package parity BEFORE deployment (gap): the coordinator has no `tensorflow`
conda env; its system python 3.12.7 carries numpy 1.26.4 / scipy 1.13.1 / sklearn 1.5.1;
the workers' conda env is python 3.12.13 with numpy 2.4.3 / scipy 1.17.1 / sklearn 1.8.0;
`duckdb` was missing on all three. AFTER deployment (§4) every role runs the campaign
from one venv built from `requirements.lock` with an identical numeric stack
(numpy 2.4.3, scipy 1.17.1, scikit-learn 1.8.0, pandas 3.0.1, pyarrow 23.0.1,
duckdb 1.5.6); the only remaining difference is the interpreter patch level
(3.12.7 on the coordinator, 3.12.13 on the workers), recorded as non-required parity.

## 2. Units (§E.6-E.8) — `tools/fs_phase23_deploy/units/`

| unit | where | what | restart |
|---|---|---|---|
| `fs-phase23-worker@SLOT.service` | every role | `worker_loop.sh SLOT`: claim → settle → `crispdm-run -q -m $FS23_CAP -t $FS23_WALL` → mark; one shard at a time per slot; OMP/OPENBLAS/MKL threads = `FS23_THREADS` | `on-failure`, 120 s; exit 75 (admission refused) never restarts; exit 13 (plan all terminal) is success |
| `fs-phase23-worker@.service.d/role.conf` | per role | coordinator: Nice=19, IOSchedulingClass=idle, CPUWeight=20, CPUQuota=200%; workers: Nice=10, best-effort IO prio 7, CPUWeight=50 | — |
| `fs-phase23-relay.service` | coordinator | `relay_loop.sh`: pull claims + terminals + STATUS from both workers, push plan/assignment and the other roles' claims, heartbeat with monotonic cycle (the settle gate) | `always`, 60 s |
| `fs-phase23-follower.service` | coordinator | `follower_loop.sh`: driver `follow` under `crispdm-run -m 1G` every 60 s: load terminals, verify readback, chain close-phase2 → run-phase3 → close-phase3; exits 0 on `PHASE_3_FILTER_COMPLETE.json` | `on-failure`, 120 s |
| `fs-phase23-status.timer` + `.service` | every role | `status_tick.sh` every minute (`OnCalendar=*:*:00`): `CLAIMS_LEDGER.json` always; `STATUS.json` from the driver's status module when present, else a fallback derived from the ledger with a schema that says so | timer |

A failed shard stops only itself: the loop writes `terminals/<shard>/FAILED.json` (exit
code, reason, log tail, role, slot) and marks the claim FAILED; other slots and hosts
continue; nothing is retried blindly.

## 3. Shard-assignment policy (§E.4) — `shard_policy.py`

* `coordinator` gets the cheapest third by COUNT (shards sorted by `est_cost` ascending,
  first `ceil(N/3)`), reported with its cost share (well under a third). Desktop-safe cap,
  documented and enforced: ONE slot, `FS23_CAP=1G` per process through `crispdm-run`
  (MemoryMax 1 GiB inside `crispdm-batch.slice`, admission-gated, `-q` queued never
  refused into a hot loop), Nice 19 / idle IO / CPUWeight 20 / CPUQuota 200%, no
  stealing. The owner's standing rule stays: this order assigns the smallest third
  explicitly and the footprint is the minimum that still executes it.
* `worker_a` / `worker_b` take the rest by weighted LPT; weight = slots =
  `min(floor(free_cores / threads_per_slot), floor(admission_free_for_new * 0.5 / cap), 8)`
  with cap 2 GiB and 2 threads per slot. With the measured headroom: worker_a 1 slot
  (6.17 GiB × 0.5 / 2 GiB), worker_b 3 slots (12.82 GiB × 0.5 / 2 GiB). The policy reads
  the admission monitor's `host_free_for_new_bytes`, never raw MemAvailable; a host
  without the monitor gets zero slots.
* Output `assignment.json` is exclusive and complete (checked); deterministic regardless
  of input order. Tests: `test_shard_policy.py` (10).

## 4. Exclusive claims (§E.6) — `shard_claims.py`

Claim files `claims/<shard>/claim.<role>.json` written only by the owning host and relayed
through the coordinator into every other host's `peer_claims/`. Local slot exclusivity is
the O_EXCL creation of the file; a live own claim whose pid is gone is an orphan and is
re-claimed (reported). Own-assigned shards win at once (home role). A steal (only when
every own shard is terminal or taken; stealers walk the others' lists backwards) waits
three relay cycles then arbitrates: home role wins, else earliest claim; the loser marks
ABANDONED before computing. Valid terminals (manifest `status: COMPLETED` with a
results digest that verifies) are never recomputed; FAILED markers are never retried
blindly; ALL_DONE counts relayed COMPLETED/FAILED claims (workers receive claims, not
results). Tests: `test_shard_claims.py` (14). All 24 tests pass under
`crispdm-run -m 512M`.

## 5. Deployment (§E.2) — `deploy.sh` + `deploy_host_step.sh`

`git archive` of the exact commit → one tar, one digest → placed on every role → per-host
step (verify digest, extract to `code/<commit>/`, `code/CURRENT` symlink, tree digest,
venv from `requirements.lock`, units + role drop-in, `daemon-reload`, `runner.env` with
absolute paths) → receipt with parity. Executed for commit `c0a6cd52` (this kit, before
the driver): `DEPLOY_RECEIPT_c0a6cd520820.json`.

| parity | result |
|---|---|
| archive sha256 | equal on 3 roles |
| tree sha256 (sorted sha256 of every file) | `aa2bdf96556971f8…` on 3 roles |
| requirements.lock sha256 | equal |
| `pip freeze` restricted to the lock | `29f3f6316b51…` on 3 roles |
| numeric stack | identical |
| unit sources sha256 | equal (installed sets differ by design: the coordinator also installs follower + relay and its own drop-in) |
| python version | 3.12.7 / 3.12.13 / 3.12.13 (non-required) |

Units were installed and verified loadable (`systemctl --user list-unit-files`,
`systemd-analyze --user verify`) on all three roles; NONE was enabled or started
(Phase B: `deploy.sh --start`). A manual `status_tick.sh` wrote `STATUS.json` on every
role.

## 6. Worker-loop exercise (local, stub driver, `loop_exercise/`)

A scratch plan of three shards (two good, one that fails) assigned to the coordinator,
cap 256M: `s0` and `s1` COMPLETED with verified digests, `bad` FAILED with
`FAILED.json` (exit 3, reason, log tail), then ALL_DONE exit 0; a second run of the loop
claimed nothing and exited ALL_DONE in 0 s (restart parity of the loop); the status tick
reported expected 3 / complete 2 / failed 1 / active 0.

Adaptation points for Phase B (the only lines that know the driver's CLI):

* `worker_loop.sh --run-shard`: `python tools/feature_pairwise_campaign.py run-worker --plan P --shard S --out-dir terminals/S --threads N --host-role R $FS23_WORKER_ARGS`
* `follower_loop.sh --pass`: `python tools/feature_pairwise_campaign.py follow --plan P --campaign-root C --terminals-root terminals --terminals-root peer_terminals --chain-phase3 $FS23_FOLLOW_ARGS`
* `status_tick.sh`: `python tools/feature_selection_phase23_status.py --state-dir S --plan P --assignment A --host-role R --out STATUS.json $FS23_STATUS_ARGS`
* terminal validity: `run_manifest.json` (or `terminal.json`) with `status: COMPLETED`, `results_sha256`, optional `results_file` (default `results.jsonl`).

## 7. Phase B checklist (after the engineering commit arrives)

1. `deploy.sh --commit <eng commit> --hosts-env … --receipt-dir …` (parity receipt).
2. `driver plan` on the coordinator → `~/.local/state/canonical_20261003/fs_phase23/plan/plan.json` (all shards before any launch, §E.3).
3. `shard_policy.py --plan … --health health/*.json --out assignment.json` (fresh health first).
4. Re-run `deploy.sh … --assignment assignment.json --start` (slots per role, units enabled: relay → status timer → worker slots → follower).
5. Verify: first terminal lands in `peer_terminals/` on the coordinator, follower readback passes, `STATUS.json` advances every minute on every role, then report once per §G.
