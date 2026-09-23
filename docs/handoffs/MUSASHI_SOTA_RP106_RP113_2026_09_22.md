# Satoshi: RP106-RP113, finish exact scoring and trustworthy retention

Base reviewed: `55a339ce`. Read the [independent review](../audits/work_plan/MUSASHI_RP98_RP105_REVIEW_2026_09_22.md)
and its executable PRE. This supersedes RP98-RP105. Existing owner authorization
covers the scoped work; finish all feasible blocks and return one consolidated
review request. Update persistent state as each block finishes. Do not stop at
each successful test to seek another approval. Respect actual tool refusals.

**Already done by Musashi:** warehouse BIGINT adoption, preserved-content check,
and resend/reconciliation of all three original T=720 pending terminals.
Do NOT restart again for this, create replacement envelopes, or retrain cells.
All 12 record/checkpoint identities were checked against the live warehouse.

**Placement (owner clarification, 22-sep):** the ONLY GPU eligible for new work
is WORKER_B's EXTERNAL RTX 5090, with its independent cooler. WORKER_B's internal
RTX 5070 Ti and WORKER_A's GPU remain held; coordinator heavy CPU/GPU work also
remains held. The owner's expected return tomorrow afternoon is NOT a release
time. Explicit cooling-restored confirmation is required for those devices.
Select the external device by verified physical CUDA UUID, never a guessed index
or model name alone; assert the actual child uses that UUID. If it is disconnected,
unavailable or fails admission, refuse GPU work without internal-GPU/host fallback.
Before dispatch, check connection, thermals, free RAM/VRAM, competing workload and
disk. Its independent cooler does not cool the host CPU: bound CPU preprocessing,
thread counts and concurrency, monitor host thermals as well, and stop safely on
failed admission or thermal limits. No VM interruption, larger slice, compression,
new storage, full-array transfer home or reduced recipe. Continue lightweight
repairs/custody work while an original-device replay is held; a 5090 replay of a
checkpoint trained elsewhere remains CROSS_DEVICE, never SAME_DEVICE.

## RP106: preserve PRE, make the real path fail correctly

Preserve Musashi's probe/results unchanged. Before implementation, freeze tests
for (1) rewritten historical metrics with missing vault/checkpoint, (2) an
unresolvable original report, and (3) an extra-root file of different identity
sharing a unit name. Include valid retained-history and identical-copy controls.
Tests must call the real closure and deletion entry points. No benchmark fitting.

## RP107: historical evidence must be bound, not self-declared

Resolve the actual original pre-deletion report by digest. Bind its accepted
design/population/preparation, unit, record, checkpoint, vault, predictions and
targets to the retained files and the accepted terminal chain. A new hash of a
candidate file or a `verified_at_deletion` boolean is not acceptance. Wrong
design, absent/changed vault, forged/unresolved report, changed record or
unaccepted terminal must produce a typed refusal and no verified pooled score.
Version successor verification without rewriting historic reports or receipts.
Keep historical verification distinct from current-array verification.

Recheck all six real deleted cells against the retained pre-deletion report and
Musashi's live-artifact/custody evidence. The independently checked metadata
backup is already available; preserve its manifest and scope. Do not claim
every estimator was independently validated just because its file is unchanged.

## RP108: content-specific deletion with durable prior acceptance

For every candidate path, require the verified artifact identity, not just a
folder name. Preflight the entire unit's inventory before any unlink. A conflicting
attempt, changed copy, symlink/alias ambiguity, active reader/writer, or unverified
catalog refuses that unit. Recheck identity at deletion under an appropriate
exclusive boundary. Interrupted/partial deletion must retain accurate per-path
status and never claim that all copies were removed. Test failure and resumption.

Bind the deletion approval to the exact independently accepted catalog/report
and backup verification. Preserve content-addressed reports before replacing any
REPORT.json. Never make a deletion marker its own scientific authority. No new
production deletion until RP107/RP108 and the per-cell scientific prerequisites
below pass. Owner's no-compression/no-permanent-prediction-archive policy remains.

## RP109: complete the author's metric within actual resources

Implement and validate a bounded route for the ORIGINAL float32 MAE/MSE reduction,
including dtype, layout, axes, full population and rounding. Compare with the
unmodified author's scorer on real-author fixtures and retained H96 arrays;
test final partial blocks and adverse cancellation/dynamic ranges. A mean of
chunk means is not presumed equivalent. Disk-backed intermediates are a candidate,
not proof; measure page-cache/cgroup and disk peaks before larger admission.

Retain the independent float64 metric as a separately named check. Never put
float64 fallback under `author_metric_float32`. If exact parity is not established,
report the precise remaining limitation rather than silently changing precision,
the original rule, model, dataset or batch recipe.

For T=336, regenerate predictions by inference from retained checkpoints only
when needed, with original inputs/ordering/device evidence and resource admission.
Label them REGENERATED, compare preserved prediction/target digests, and preserve
any mismatch. A new replay is not the deleted original. No training or metric
editing to force agreement. Prefer any still-inventoried matching copy only where
its location/thermal policy allows; do not lift the coordinator hold for this.

## RP110: device attribution and operational patch scope

Separate MEASURED actual CUDA UUID from INFERRED/UNKNOWN historical identity.
Unrelated GPU memory growth, multiple GPUs, missing measurements and numeric
visibility masks must not prove same-device equality. Preserve original records;
append evidence and provenance for any stronger historical attribution.

Run a bounded real-author CPU parity test for DataLoader worker change over
multiple optimizer steps: batch order/values, RNG states, weights, optimizer state
and stopping/validation inputs. Check the allocator option reaches the real child
and leaves the scientific recipe untouched. Report any unproven equivalence.
Qualify routing diagnostics by selected rows/batches and device pair; no global
causal explanation from three selected windows. Preserve every failed replay.

## RP111: finish remaining numerical checks, no duplicate training

Reconcile the already accepted T=720 terminals, then complete its original-rule
replays, corrected official scoring, independent reductions and finite catalog on
the admitted worker. Freeze the bounded run ledger from measured costs; no unlimited
loop. H96 original-device replays remain conditional on cooling confirmation;
first check whether existing independently accepted replay evidence satisfies
the same requirement without new execution. Do not upgrade unanchored history.
Never call a cross-device replay same-device or replace a failed pointwise rule
by equality of aggregate metrics. Work on other blocks while cooling is unknown.

## RP112: independent finite-catalog acceptance and authorized cleanup

Before deleting remaining original or regenerated temporary arrays, independently
check the catalog's required estimator families with numeric oracles, denominator,
ordering, population and undefined/approximate states. Produce the approved paired
contrasts and per-time/channel/horizon summaries. Store/check the small durable
catalog backup, accepted manifest and report. Then invoke only the corrected
per-content deletion path, with no active consumers and explicit per-copy receipts.
Do not retroactively claim this was done for the six already deleted originals;
state what can now be checked and what requires controlled regeneration.

Coordinator copies stay held unless their bounded administrative deletion meets
both the accepted retention gate and the existing thermal policy. Do not hash or
replay many GB locally as a workaround. Inventory any remaining copies and their
space; report measured free disk, not projected reclaimed space. Retain checkpoints,
configs, metric catalogs and accepted receipts within the existing disk envelope.

## RP113: one complete, scoped return

Report all 12 cells, official normalized MSE/MAE, independently checked float64
columns, matched persistence/seasonal baselines, source table, per-seed values,
sample SD with host scope, and four-horizon means only on complete matched seeds.
Separate training completion, custody, original scorer, same-device repeatability,
cross-device portability and operational agreement. No missing horizon disappears
from the denominator. No operational margin is called statistical equivalence.

Keep protocol B's exact Table-9 recipe/choice mapping as a separate explicit
deliverable; neither execute an assumed fixed-L512 substitute nor declare it
completed by protocol A. No financial or household fits resume during this block.
Run focal and relevant integration suites with interpreter/dependency scope and
resource accounting, freeze POST against PRE, reconcile content through the live
warehouse, update plan/state (supported stage name), commit/push and synchronize
workers. One final return, with any genuine physical/evidence limit stated; do not
invent completion and do not defer independent work to the owner.
