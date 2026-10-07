# Phase 2/3 selection: integration lane, Phase B (2026-10-06)

Branch `satoshi/fs-phase23-integrated-20261006` = eb9df9eb + driver fc026d04 + data 6859b07a
(re-merged at 18865d4c) + integration 9e613bd3. Deployed commit **f71d328e** on all three
roles (receipt `DEPLOY_RECEIPT_f71d328e170c.json`: archive / tree / lock / locked-freeze /
unit-source digests equal; numeric stack identical; interpreter 3.12.7 vs 3.12.13).
Test suites on the merged tree: 70 passed (`TESTS_INTEGRATED_7f0d570b.txt` is the first
merged run; the warehouse suite grew by one with 18865d4c), kit 26 passed.

Roles only; ssh aliases and the token live outside the repository.

## 1. Hosts (fresh measurement before launch, `health/`)

| role | cores | load1 | MemAvailable | admission free-for-new | live leases | slots | cap/process | threads | steal |
|---|---|---|---|---|---|---|---|---|---|
| coordinator | 16 | 0.92 | 20.21 GiB | 17.22 GiB | 0 | 1 | 1 GiB (Nice 19, idle IO, CPUWeight 20, CPUQuota 200%) | 1 | no |
| worker_a | 32 | 2.61 | 9.47 GiB | 6.15 GiB | 0 | 1 | 2 GiB | 1 | yes |
| worker_b | 32 | 0.29 | 16.09 GiB | 13.09 GiB | 0 | 3 | 2 GiB | 1 | yes |

Slots from `shard_policy.py` (`SLOT_POLICY_eurusd_e3ed414c.json`): floor(headroom × 0.5 / 2 GiB)
capped by free cores. Follower cap 2 GiB (coordinator; the 1 GiB rule is for worker processes).
Data: the phase-1 TRAIN bundle (6 files, 82 MB) copied to every role's `data/` and all six
manifest digests verified on each host.

## 2. Plans (all shards before any launch, §E.3)

| population | identity | shards | pairs | coordinator (small) | worker_a | worker_b |
|---|---|---|---|---|---|---|
| EURUSD | `phase1-eurusd-final:94d20c038d55e152` | 256 (plan `d5bcf199…`) | 66,795 | 85 shards, 31.0 % of cost, 20,845 pairs | 86 / 34.7 % / 23,150 | 85 / 34.3 % / 22,800 |
| ETH | `phase1-final:ETH:29d2f745f5d9e87c` | 32 (plan `9d12e136…`) | 3,403 | 11 / 31.0 % / 1,054 | 11 / 36.1 % / 1,228 | 10 / 32.9 % / 1,121 |

The driver's rule ("smallest third of shards by estimated cost to size_class=small") is the
order's §E.4; the coordinator carries 31 % of the cost on one capped slot.

## 3. Units running

coordinator: `fs-phase23-relay`, `fs-phase23-status.timer`, `fs-phase23-worker@1`,
`fs-phase23-follower@eurusd`, `fs-phase23-follower@eth`. worker_a: status timer + `worker@1`.
worker_b: status timer + `worker@{1,2,3}`. Started 01:26Z; every run-worker pass admitted by
the host's admission monitor (`crispdm-run -q`), no refusal, no OOM, no failure marker.

## 4. First terminals and transport

* coordinator: first EURUSD terminal at 01:30Z (unit `9bcdfbcc…`, 249 pairs → 249 gate /
  26,892 metric / 4,482 stability rows = 249 × 6 × 18).
* worker_a and worker_b: plan + manifest relayed at cycle 1, passes started 01:31:15Z, first
  authored terminals visible in the coordinator mirrors at 01:34:45Z (2 each).
* Relay: 60 s cycles, pulls terminals/failures/claims/quarantine/STATUS per role, pushes plan,
  manifest and peer terminals; `pushes_ok` true from cycle 2 once the lazy subdirectories
  were created (fix 92c17145).

## 5. Follower, warehouse and readback

* 01:26-01:43Z on the driver's adapter file: first receipt accepted with readback_count =
  inserted = row_count for all three tables.
* 01:43Z switched to the DATA module (`tools/fs_phase23_warehouse.open_warehouse`, DuckDB file
  mode, one file per population since a DuckDB file takes one writer): receipts moved aside
  (`receipts.adapter_20261006T014321Z`), every adopted terminal resubmitted. Run registered
  with expected counts (EURUSD 66,795 / 7,213,860 / 1,202,310; ETH 3,403 / 245,016 / 61,254).
* Readback on a copy of the file: all ten relations present; `reconcile` reports count vs
  expected per table with the campaign digest rule; `host_role` stored per computing role
  (coordinator / worker_a / worker_b); `read_run(unit_id=…)` returns exactly that unit's rows.
* One receipt written by a stale follower process (its crispdm-run scope survived the unit
  stop) was detected by comparing receipts with stored units and removed; both loops now
  forward SIGTERM to the child (f71d328e). At 01:49Z exactly one `follow` per population.

## 6. STATUS at 01:50Z (coordinator `STATUS.json`, regenerated every minute on every role)

| population | complete / expected | active | failed | pairs | rate | ETA |
|---|---|---|---|---|---|---|
| EURUSD | 39 / 256 | 5 | 0 | 10,180 / 66,795 | 108 units/h | 2.0 h |
| ETH | 0 / 32 | 0 | 0 | 0 / 3,403 | — | starts when a host finishes its EURUSD pass |

by_host (complete, active): coordinator (11, 1), worker_a (13, 1), worker_b (15, 3).
Follower eurusd: adopted / already / submitted per cycle, submit_errors [], quarantined 0,
foreign 0. Worker STATUS files relayed within the minute.

## 7. Live-cube gate (step 5) — BLOCKED on the token

Done without the token: data module merged and deployed; store packages 0.1.5 / 0.1.4 in the
campaign venv; follower unit reads an optional untracked `follower.secret.env`
(`WAREHOUSE_TOKEN=…`, mode 0600); runs registered; file-mode receipt + readback + reconcile
proven. NOT done: the policy classifier denied reading the service token from the warehouse
host's environment and querying the live cube with it, so the live steps that need it
(verify capabilities / information_schema, one-row smoke, pointing the follower at the
service URL, live receipt + readback + reconcile) wait for the owner to place the token in
`~/.local/state/canonical_20261003/fs_phase23/follower.secret.env`. The service venv still
runs olap-store 0.1.4 / duckdb-store 0.1.3 / data-warehouse-service 0.1.0; the package bump
and the user-unit restart (`crispdm-data-warehouse-olap.service` is a `systemd --user` unit,
not a system one) were not applied because their verification needs the same token.

Switch procedure once the token exists: `systemctl --user stop fs-phase23-follower@{eurusd,eth}`
(stops the scopes too now), move each `receipts/` aside, set `FS23_WAREHOUSE=http://127.0.0.1:5057`
in `runner.env`, start both followers: every adopted terminal is resubmitted (the store's unique
key makes it idempotent) and `reconcile` on the live cube judges completeness.
