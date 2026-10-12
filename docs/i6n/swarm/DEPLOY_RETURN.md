# I6-N swarm production campaign `i6nw1`: deployment return

Status: **PREPARED AND VERIFIED, NOT STARTED.** The permission classifier
denied the production deploy step (start of the coordinator and secondary
units) with the reason `[Production Deploy]`. Per the orders it was not
bypassed. No unit was installed or started, no fit ran, and no optimizer
update exists. The S3 pilot unit was not touched.

## Pins

| Component | Pin |
| --- | --- |
| predictor (deploy commit; code pin 68e6a7e2) | 2ded0406f25c357719adb55e4a54d0c1cf77403a |
| doin-core | 4a321cdc680d8a91ff4fb3722b04e6da2160cd5f |
| doin-node | df6aad0b0ee5a932845828d682af547c2e170d46 |
| doin-plugins | 566c5af189075eedd81e43c10c660d24d2f30faf |
| data-lake | fbb6e58d37e0a7045ede5a975e08ffea330b5071 |
| data-warehouse | 463ad5671bd458f6869128260b729176069edf9d |
| E search plan | plan_sha256 606a355567257ff76810cec2dba3cd2952b0b6c3fe4ad592ea568402d8feda3c (v1.2.0) |
| C weekly plan | plan-v6 f58d906d41f039c1863d868cf067aff58f9884c932d8f2d829e320e38894be11 (seals all 681 cells) |
| Seed / direction | 6102026 / minimize |

## Done and verified

- **Code:** all six trees were exported with `git archive` into
  `/var/tmp/i6n-swarm-code-2ded0406f25c` (883 files, sha256 manifest), with
  pinned per-host python wrappers.
- **Effective private config:** derived and validated (no placeholders, pin
  digests, ports, seed, mode). Capabilities: VALIDATION_OUTER_WEEK and
  DIRECT_MULTI_OUTPUT_ARM.
- **Class admission:**
  - S1 cap 3,023,954,012 B: 1.25 × (pilot peak 2,410,315,776 + coverage 8,847,433).
  - S2 cap 3,044,231,417 B: 1.25 × (2,390,552,576 + 44,832,557).
  - S3 not admitted: its pilot is still RUNNING and has no completed receipt.
- **Lane B `plan`:** sealed 52 DAG nodes. Rehearsal `preflight` was READY on
  all 12 checks.
- **Replication:** code, data root (701 files), campaign root and ledger token
  were copied to the secondary and the preferred and verified byte-for-byte on
  each host with `sha256sum -c` (1,591 files per host). The deploy commit is
  present in each worker's repo (refs only).
- **`crosshost_verify --pin 2ded0406…`:** preferred and secondary
  `identical=true`, 701/701 files, pin present. `control` is unreachable and is
  not a worker in this campaign.

## To start (the step that was denied)

```bash
cd <deploy branch checkout>   # satoshi/i6n-deploy-20261012
python -m tools.i6n_swarm_deploy --spec <private deploy spec>
```

The command is idempotent. It runs:

1. `preflight` (must be READY), then `install` and `start` for the coordinator
   (ledger and DAG tick).
2. Re-verification of the secondary, then `install` and `start` (admitted
   S1/S2 search begins with freeze, then DOIN).
3. `install` on the preferred, plus the coordinator timer
   `i6n-swarm-i6nw1-join-preferred.timer`. The timer starts the preferred only
   after the S3 unit is inactive and its RESULT sha256 is verified. If S3
   failed, it starts the preferred anyway with the cause recorded.

## Status

```bash
python -m tools.i6n_swarm_campaign status --root /var/tmp/i6n-swarm-prod-20261012 --json
python -m tools.i6n_swarm_deploy --spec <private deploy spec> --status
```

`STATUS.json` here was generated with `--write-status`. The swarm state is
`QUEUED: NOT_STARTED: run start`. After start, the coordinator tick rewrites
`status/STATUS.json` every 5 minutes.

## Pending DAG nodes and reasons

- **All 52 nodes:** QUEUED / NOT_STARTED. The ledger has no units because
  start was denied.
- **`search:h*`:** depends on `freeze_folds:h*`, which runs after start on the
  secondary.
- **`annual:h*`:** depends on `search:h*` and `freeze_weeks:h*`.
- **`multi_output:short|long`:** waits for search finalists. The arms
  template (212 arms) is WAIT_DEPENDENCY FINALISTS_UNRESOLVED.
- **S3 class:** NOT_ADMITTED until its pilot completes and is verified.
  Admitting it needs a new plan with the S3 receipt; the cells already carry
  the S3 resource guards.

## OLAP

No campaign publication yet. The pilot OLAP state is unchanged and the cost
collector timer was preserved.
