# Q2 resource successor and independent lanes

Musashi, 2026-09-29. Operational dispatch following the owner's restored lake.
This is a documentary review and implementation order, NOT an independent
certification of Satoshi's artifacts or authorization of confirmatory fits.

## Evidence and corrections

- Q2 return and MEMORY_CORRECTION.json: `6de538e2`. Two governed mechanical
  units completed; no new forecast accuracy was measured.
- The 7.4 G journal figure is a killed multi-child wrapper's cgroup peak,
  not RSS and not a cell peak. The separate 8,458,399,744 B figure is child RSS.
- The 1,463,877,632 B pilot peak belongs to a specific 20,000-window data
  materialization. Do not call it a universal lower bound or use it as a
  training cap: the production loader gathers batches and the model was absent.
- RSS and cgroup accounting measure different quantities. Their observed ordering
  does not establish an invariant. Read cgroup identity, ancestry and lifetime.
- The instrument at `84bcd605` and v2 seal must be integrated into the actual
  published execution lineage. Presence on master is not intrinsically required;
  reproducible reachability from the selected runner commit is required.
- Traffic `8d461628` reports bounded evaluation at 3.600 GiB, not a trained
  model result. Weather `f328db3a` reports replay separately from authorization.

## QRM01: integrate and test, CPU lane

Use an isolated integration worktree. Preserve existing worktrees and all old
design bytes. Integrate the required runner, instrument and seal with explicit
commit provenance; do not cherry-pick a seal and silently reinterpret old cells.
Use the existing admission/launcher API, not another scheduler. Every cell must
have a fresh exclusive scope enclosing its complete process tree, with its own
reservation. One sequential child in a reused driver scope is NOT sufficient.
Record host RAM cgroup peak separately from GPU allocated/reserved memory,
scope identity, kernel limit, optimizer updates, CPU and wall time, and stage.
Read peak before scope removal; an external supervisor must retain termination
status on child failure. Missing peak is unknown, never zero or success.
Tests first: two concurrent children, reused-scope rejection, missing peak,
failed child, aggregate admission and reservation release. Use bounded fixtures;
never trigger host OOM to test recovery. Verify deployed runner identity too.

## QRM02: actual training-path cost pilot

After QRM01, seal a TRAIN-only pilot using the exact intended architecture,
batch, dtype, optimizer, loader and W1440 configurations. State stage budgets,
host/device caps and remaining allocation BEFORE dispatch. Do not invent a
new allocation or reduce a cap to pass admission; reconcile existing authority
first and report an exact additional request if necessary.
Build the model; run forward/backward and optimizer steps that instantiate slots;
cover warmup, steady batches, the largest final-batch shape, checkpoint write/
reload, and validation mechanics on a TRAIN-only fixture of the planned shape.
Measure the stages in the same cell scope. No test access or accuracy selection.
Cost both missing architectures unless a justified bound covers both. Report
peak and projection with explicit headroom, not a claim of maximum memory proven.

## QRM03: successor ready for review

Produce full costs and residual budget for six missing cells and any genuinely
required fresh controls. Preserve the matched-budget estimand and 600-update
contract unless a distinct scientific successor is explicitly approved.
Never promote the twelve historical cells through a new seal. Give a fieldwise
design diff, integration commit and executable reproduction command. Full fits
remain pending this costed successor's review; no serial documentary rework of
unchanged lake restoration is required.

## Parallel dispatch, not a programme-wide stop

Owner's explicit follow-up: orchestrate agents concurrently, not a serial queue
of repairs. Satoshi owns dispatch and integration; delegate immediately after
checking live processes, reservations and current published tips. Do not wait
for all lanes to close before delivering or dispatching the next eligible unit.

| Agent | Independent assignment | Deliverable / dependency |
|---|---|---|
| A: Q2 runtime | QRM01 integration and cell-scope tests | Published runner and measured isolation; unlocks QRM02 |
| B: Traffic | Reconcile `8d461628`, prepare and execute its authorized TRAIN-only cost pilot | Actual training footprint, step cost and trained-output parity; does not wait for A unless it uses A's defective launch path |
| C: classification | Reconcile `19a37baf` and `40224a3a`; finish outstanding native-reference/warehouse work under CB orders | Recount actual retained results, distinguish retained projection from accepted warehouse rows; no duplicate reproduction |
| D: M5PHET | Inspect current product plan, select its next independently executable user-facing acceptance case | End-to-end real-provider result with input, output, persistence and restart proof; fixtures clearly separated; no live service restart |
| Satoshi | Integrate each completed lane, keep dispatch moving, prepare QRM02 design while A works | One updated queue, actual leases and budgets; resolve conflicts rather than making every agent wait |

Each agent uses a separate worktree and owns its files. Shared-runner changes
have one owner (A); other agents pin its accepted published revision or use an
already verified independent runner. Hermes instances may fill these roles when
available; otherwise use the actual available subagents. Do not claim dispatch
without a task/session identifier and worker acknowledgement.

Start CPU/protocol/product work concurrently. GPU jobs queue through the shared
atomic admission gate; use distinct eligible devices concurrently only when
aggregate host RAM, VRAM, disk, thermal state and existing allocations permit.
Keep a ready successor queued for each eligible device. No artificial GPU load
to report utilization, no resurrection of a completed campaign, no queue that
holds a reservation while waiting for an unrelated code review.

At initial dispatch publish agent/task id, worktree, source tip, first command,
acceptance test, budget and actual blocker for each lane. Refresh the queue when
any job ends or fails; start the next independent admitted job without waiting
for the round's final report. Report elapsed idle time with its concrete cause
if no eligible experiment can run. Never present unissued assignments as active.

While QRM01 runs, progress the independent Traffic TRAIN-only pilot already
listed in the queue, after its own sealed cost limits and fresh admission;
prove bounded-evaluator parity on trained outputs where the author path fits.
Do not replace the author metric or transfer Weather's training footprint.
Continue classification/M5PHET deliverables with their existing authority and
own tests. Do not duplicate completed jobs or treat this as M4 approval.
The external 5090 stays preferred only when eligible; its reported host-memory
restriction is NOT lifted by this order. Use another admitted worker without
displacing live work. Coordinator stays light; no host reboot or broker changes.

Update each affected queue row with these source commits, real dependencies,
reservation and next executable action, preserving prior observations. Lead
the return with new scientific measurements if any; otherwise explicitly say
no new accuracy measurement, then give measured resource results. Include model
error, paired naive and literature comparison only for genuinely scored tasks.
