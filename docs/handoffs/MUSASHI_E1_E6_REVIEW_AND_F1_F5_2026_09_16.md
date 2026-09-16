# Review of d35bb21 and corrective orders F1-F5

Date: 2026-09-16. Reviewer: Musashi. Executor: Satoshi.
Disposition: operational progress observed; recovery and migration closeout NOT accepted.
The owner's scoped implementation/deployment authorization remains in force.

## Independent evidence

The warehouse and successor DuckDB loader are active/running with NRestarts=0,
as read from systemd. This does not independently prove the reported 55 terminals
or healthy ingestion. No production data, configuration, service or quarantined
WAL was modified in this audit. Full suite counts remain Satoshi's report.

Attempting the migration pytest file with the installed production interpreter
failed because pytest is not installed there. The interpreter was NOT modified.
The following direct behavioral probes ran using that interpreter, importing
the reviewed tools/olap_duckdb_migrate.py, on disposable databases only:

1. source(id PRIMARY KEY) = [1,2,3], target(id PRIMARY KEY) = [1]. Calling
   copy_relation(con, 'source', 'target', 3, 'id', 1) raises ConstraintException:
   Duplicate key id:1. It restarts at the first source row rather than skipping
   the completed prefix. Without a key constraint, duplication remains possible.
2. build_fixture_cube(source, terminals=1, with_children=True, with_contract=True)
   and build_fixture_cube(target, terminals=1), then replay_between(source,target)
   reports zero terminals and zero children replayed, one contract replayed;
   target metrics=0 and datasets=0 afterwards. An existing parent hides missing
   children, so replay does not reconcile a partially restored outcome.

## Source-level findings

- cmd_rollback now restores to a DuckDB file. It does not implement the ordered
  replay into the former PostgreSQL backend; a successful DuckDB fixture is not
  a cross-engine rollback rehearsal. No PostgreSQL incident rollback was attempted
  by this reviewer.
- cmd_catchup still skips tables without received_at and does not call the new
  replay helper. Its production CLI has not acquired the claimed child closure.
- cmd_export calls row-count equality COMPLETE without a content check, continues
  on OVERFILLED_REFUSED and returns zero. Its partial path invokes the duplicate
  insertion reproduced above. Helper tests do not establish CLI behavior.
- snapshot_database trusts a caller-supplied owner_stopped boolean; it does not
  establish the coordinated boundary itself and still verifies only a terminal
  count. This is a claim of coordination, not proof of full snapshot contents.
- The incident's source explanation says startup DDL is checkpointed before
  envelopes. The provider still calls ensure_envelope_tables inside
  write_foundation_envelope and checkpoints only after load_envelope; startup
  _ensure_schema has no checkpoint. This discrepancy requires an executable
  failure/recovery test, not a claim that the original engine defect recurred.

## F1 - Real command paths and complete recovery

Freeze these probes before edits. Add tests through main()/the shipped CLI for
export, catchup and rollback, using real providers and disposable engines.
Implement actual PostgreSQL rollback destination support: create the missing
contract schema with its provider, restore parents and all children, validate
content and resolve retained contracts in the destination. Keep DuckDB-to-DuckDB
copy as a separately named operation, not a substitute for rollback.

Existing parent identities must not hide missing children or conflicting content.
Reconcile exact keys/content, repair missing rows transactionally or refuse with
a precise unresolved state before declaring success. Prove interrupt/resume,
second-run idempotency and a target whose parent exists but children are missing.
Do not run rollback on production. No source evidence is discarded.

## F2 - Correct resumability and catch-up

Implement completed-content checks, transactional visibility or explicit durable
checkpoints, and key-based resume that does not reinsert prior rows. Test both
constrained tables and bag-valued tables, equal-count different-content targets,
partial targets, extra rows and interrupted batches. CLI must exit nonzero when
required work is refused/incomplete. Never use row count alone as completion.

Route the actual catchup command through complete parent/child/contract closure.
Test a new outcome after the watermark with timestamp-free children, and partial
existing outcomes. A green helper and an unchanged CLI do not close this order.

## F3 - Incident and snapshot evidence, without replaying production WAL

Keep the quarantined WAL and recovered main file untouched. Record their digests
privately. Reconcile pre-incident accepted delivery/terminal/outbox identities and
their child CONTENT against the recovered database; separate committed evidence
from pending or replayable writes. Counts alone do not support no-loss claims.
Do not presume the WAL contained only tests without supporting receipts.

Create a fresh, minimal disposable reproducer for the claimed schema/WAL failure.
If it cannot reproduce, label the root cause a hypothesis and the checkpoint
change a mitigation. Verify abrupt interruption between schema creation, envelope
commit, checkpoint and response; reopen and reconcile accepted outcomes. A clean
close/reopen is not the same as recovery after interrupted writes.

Snapshot code must establish a real engine/owner boundary or clearly produce an
UNVERIFIED_COPY. Do not accept a boolean as a measured boundary. Compare full
selected content and dependency closure, not just terminal count. Test writes
near the boundary entirely on disposable databases. No live WAL replay is needed.

## F4 - Population and scientific views

Preserve the improved run-level selection, but separate CURRENT membership from
MECHANICAL/EXPLORATORY/INVALIDATED/scientifically admissible status in executable
views. Including 52 current terminals does not make all 52 scientific evidence.
Reconcile the exclusion of B4, T2 and M4 against the current work-plan scope;
being reviewed separately or NON_CONFIRMATORY is not evidence of obsolete work.
Give precise order/artifact links for inclusions/exclusions. Do not promote
results or discard archives. Keep rejected/negative results within their proper
campaign scope rather than selecting by favorable outcome.

## F5 - Deploy only needed fixes and finish

Use isolated test environments; do not install test dependencies in the live
service environment. Continue independent CPU work across available workers with
separate disposable databases. Correct and publish code, tests, method state,
work plan and runbooks. Scoped tested updates are already authorized; no requests
to continue between steps. Preserve unrelated PostgreSQL and all live campaigns.

After fixes, verify the successor loader with a real bounded envelope, query its
persisted result and reconcile the receipt/outbox, including an idempotent retry.
Keep the console as the accepted analytics route for now; do not block this work
on a Metabase driver decision. Historical terms research remains independent.

Return exact revisions, CLI-based failing-before/passing-after evidence, actual
cross-engine rollback results and full-content recovery/snapshot checks. Do not
declare the whole order closed while a required operation is only simulated,
implemented in an unused helper, or left to a future round.
