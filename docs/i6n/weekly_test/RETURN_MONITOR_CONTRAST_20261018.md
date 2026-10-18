# Monitor contrast launch return

Date: 2026-10-18. Source deployed: `dc818635b75916763c17658d7a4dd703832d6e47`.
Packet: `37ba91fe7395c8f2b14f45be6d33c7af219975a3cadcd268ac33020b20207958`.
Execution identity: `3ecd269063177dc12b1a8df8c126ca5f56bbea76904dc28490ddd75a45347286`.

The old window/core campaign was paused after its current fit finished:
316/408 actual fits and 541/541 reports read back, zero pending/discrepant.
Both old timers are disabled. Historical artifacts, original source runtimes,
and their sealed campaign identities are preserved. This is a priority change,
not a finding that the old mean monitor was implemented incorrectly.

New campaign: 50 chronological supported weeks x two families x three monitors
x BN off/on = 600 weekly predict/score/update units. Preferred GPU runs short7;
secondary GPU runs long6. One heavy job per host. The second GPU sharing the
preferred host is not used. Each family retains its device class across arms.
Initial January calibration checkpoint and each later donor are authenticated.
Each forecast is sealed before reading its week's observed labels. Source
feature selection is retrospective development; no independent OOS claim.

## Verification

- PRE: new monitor tests failed collection because the component did not exist.
- POST: 15 focused tests passed, including real-trainer synthetic CPU tests.
- Each worker's actual TF 2.21 runtime passed all three synthetic GPU tests:
  preferred 18.01 s, secondary 18.99 s. These are engineering tests, not fits on
  scientific data. Existing layer tensors are preserved during BN insertion;
  BN inference does not update moving statistics; archive reload reproduces
  predictions; both monitors leave the two-epoch gradient trajectory identical.
- Broader regressions: 104 passed, one failed, one deselected. The failure is
  the old test requiring `slots=25` to be rejected, although the prior window
  experiment already supports it. `build_config`, `_validated` and their lower
  validator are unchanged from base `d5f5f03a` (AST/byte checks). The deselected
  historical source-pin test intentionally rejects this newly changed trainer;
  the original deployed campaign files are not changed. Neither is hidden as a
  green suite. Keras emitted deprecation warnings during GPU tests.
- `git diff --check` passed. CLI preparation authenticated source files, checked
  four-calendar-year gradient windows and label maturity for all 100 units.
- Start verified on both GPUs. The first unit of each family completed without
  failure and both continued automatically to the next week. Two reports were
  published and read back, zero pending/discrepant. This is not a full-lot
  result or a measured stable ETA.

MC01/02/04/05/06 have direct focused behavioral tests. MC03 reuses the tested
prequential order and guards (included in broader regression). MC07 has digest
and immutable-result tests plus the existing donor/forecast guards. MC08 has an
explicit null TRAIN_REPLAY test. MC09 has production start evidence and bounded
attempt/lock logic; forced worker death during a real fit has not been tested.

## Operation

The worker timer executes one weekly unit per invocation with 4352 MiB governed
memory and a 3600 s wall limit, under a 6 GiB/zero-swap parent service. Committed
source digests, data, result files and donor weights are checked before reuse.
Three process-start attempts maximum; untyped exceptions become explicit FAILED
terminals and block that arm's descendants. An already written update receipt
can be reused without a second fit. Independent arms remain schedulable.

The coordinator timer copies results to the large artifact volume, publishes
metrics with the existing durable outbox and warehouse readback, and stops only
this campaign's timers after both families finish and delivery is complete.
SQLite and locks remain on the native filesystem; large snapshots have copies
on worker and coordinator. The worker's local STATUS supplies resource waits
and retry states. No agent polling is needed to advance the queue.

One-shot status, on the coordinator with its private config:

```bash
python -m tools.i6dmo.monitor_campaign status --spec <coordinator-RUN.private.json>
```

Reproduction: the operator config identifies the two source-unit environments,
runtime paths, dataset packet, warehouse EnvironmentFile path, and role hosts.
It contains no credentials. See `monitor_ops.py --help` and the frozen design.
Archive/runtime deployment must pass the three GPU engineering tests before
the deployment command; it is not sufficient merely to copy code.

## Scope still pending

This experiment compares weekly forward VALIDATION performance. The owner's
full retrospective TRAIN_REPLAY is still pending; FIT and observed calibration
metrics are saved under their own names and cannot substitute for it. Therefore
no TRAIN_REPLAY-versus-VALIDATION gap is claimed. Smaller gradient histories
(one year, six months, three months) remain later contrasts. TEST is not read.
No experiment here promotes a model to trading automatically.
