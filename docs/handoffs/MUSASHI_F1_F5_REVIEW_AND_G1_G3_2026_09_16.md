# F1-F5 review and bounded continuation G1-G3

Reviewed: predictor e99c6ba. Date: 2026-09-16. Reviewer: Musashi.
Executor: Satoshi. Existing owner authorization remains valid.

## Verified progress

Independently reran the previous two probes using the migration module and the
installed DuckDB interpreter, on disposable databases only:

- source ids [1,2,3], destination [1]: copy_relation now produces [1,2,3].
- existing terminal with no children: replay_between now restores one metric,
  one dataset and one artifact from the source fixture.

These two exact defects are corrected. Warehouse and successor loader both
report active/running, NRestarts=0. No production process, data or WAL was changed.
No full-suite or independent PostgreSQL rollback run is claimed in this review.

## High-priority finding: no-loss test does not compare expected child content

tools/incident_evidence_reconcile.py reads independent terminal identities and
statuses, but computes child hashes only from the recovered cube. It neither
compares those hashes with independently retained expected content nor fails
when a COMPLETED terminal has lost its children. It also fetches campaign, unit
and generation from the cube without comparing them with accounting.

Independent reproducer through its main(): build_fixture_cube with one terminal,
children and a contract; create a temporary accounting database with that exact
terminal identity/status. Then run three times:

| Cube contents | Exit | committed | missing | children_present_for |
|---|---|---|---|---|
| Original fixture | 0 | 1 | 0 | 1 |
| Metric value changed to 999999 | 0 | 1 | 0 | 1 |
| All three child tables emptied | 0 | 1 | 0 | 0 |

This disproves the verifier's claimed child-content guarantee, NOT actual
production preservation or loss. The incident's outcome remains to be established
at that scope. Preserve the honest admission that the WAL root cause is unknown.

## Additional incomplete paths

copy_relation resumes from MAX(key) without proving that the destination is a
complete matching prefix. Reproduced source [1,2,3], destination [1,3]: returns
zero copied, leaves id 2 absent. This helper must repair the hole or refuse it;
a higher-layer final validator does not make its resume claim true.

snapshot_database still takes owner_stopped as an assertion and copies main/WAL
sequentially, then verifies one count. The source function is unchanged from the
prior review. Requiring the flag does not establish a writer boundary. No
production snapshot loss is alleged by this source review.

## G1 - Establish exactly what survived

Freeze the three-case reconciler reproducer before edits. Derive expected child
content from independently retained accepted terminal payloads/receipts/outboxes,
with the actual canonical terminal contract and exact campaign/unit/generation.
Compare every expected field and child multiset, including absence, additions,
numeric changes, contract links and legitimate empty outcomes. Do not derive the
expected answer from the recovered rows themselves. Reuse production validators.

Where independent expected content is unavailable, report CONTENT_UNVERIFIABLE
with its precise population, not preserved or lost by inference. Keep identity
presence, status agreement, content preservation and replayability separate.
Outbox availability must match the exact generation/digest and a valid recoverable
payload; a filename or same campaign/unit is insufficient. An empty accounting
population must not produce a blanket no-loss result for a nonempty campaign.

Test the shipped CLI on altered metric, missing dataset/artifact, changed parent
identity fields, wrong generation, legitimate empty refusal, unavailable expected
payload and empty populations. Reconcile the real recovered cube read-only against
an engine-consistent snapshot or owned query interface. Publish scoped counts
and unresolved cases, not a binary claim beyond the evidence.

## G2 - Finish resume and snapshot correctness

Prove a matching contiguous prefix before MAX(key) resume, or use key/content
reconciliation. Test holes, modified existing rows, extra rows and selected-run
subsets via the actual export CLI. Repair missing rows or refuse precisely; do
not silently report completion. Keep duplicate/bag semantics explicit.

Implement a measured engine/owner snapshot boundary, or rename the current path
UNVERIFIED_COPY and use a genuinely consistent path for recovery evidence. Merely
adding another required caller flag is not a correction. Test coordinated writer
drain, rejected overlapping writes, full-content/dependency equality and reopen
on disposable databases. Preserve original WAL and recovered production files.

## G3 - Complete and return, without opening new campaigns

Finish independent blocks continuously under the existing authorization; do not
request permission at every step. Keep healthy production running except for
necessary tested scoped cutovers. No live trading, training or scientific replay.
No original incident-WAL replay is needed. Root-cause uncertainty alone is not a
request to stop the healthy warehouse, and Metabase is not a blocker.

Update the return's no-loss wording now to distinguish identity/status evidence
from unproven content. Preserve previous reports in Git; link a dated correction.
Maintain method state, work plan and test/evidence matrix. Return exact command
paths, real-engine tests, improved snapshot/reconciliation results and any
remaining unverifiable population. Do not reopen the two corrected defects or
expand this into a general infrastructure redesign.
