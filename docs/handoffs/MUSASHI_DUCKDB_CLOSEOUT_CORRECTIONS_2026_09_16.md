# DuckDB closeout review and orders E1-E6

Reviewer: Musashi. Executor: Satoshi. Date: 2026-09-16.
Reviewed return: predictor 68f4e39, SATOSHI_V1_D6_RETURN_2026_09_16.md.
Status: deployment reported and service running; full completion NOT accepted.

## Independent verification and scope

Read the return, migration implementation and committed rollback report.
systemd reports warehouse active/running, NRestarts=0, invoking the isolated
DuckDB production environment. The legacy OLAP loader is inactive/dead.
The unauthenticated host request returns 401; this reviewer did not bypass it
or independently confirm the runtime version, 54-terminal count or all content
digests. No production state changed. The full reported suite was not rerun.

## Findings

1. Rollback is not implemented. In tools/olap_duckdb_migrate.py, cmd_rollback
   emits REPLAY_REQUIRED even without --dry-run, never performs the promised
   writes, skips children as CHILD_OF_PARENT and returns zero. The committed
   ROLLBACK_DRYRUN.json proves discovery of two missing terminals, not restored
   outcomes. Missing contract DDL is only one part of this gap.
2. Selection contradicts the order. cmd_select classifies every df_* table as
   LEGACY_COMPARISON_ONLY and identifies current terminals from GOVERNING alone.
   relations_for then chooses whole tables from constants without consuming the
   per-run manifest at all. This is not selection by current campaign lineage or
   an actual row-level dependency closure. Current data-foundation work must not
   be discarded from primary analysis merely because it predates gov_terminal.
3. Snapshot consistency is not established. cmd_snapshot copies main and WAL
   sequentially while describing a live snapshot, with no coordinated writer
   boundary. Checkpointing the copied files and comparing one count cannot prove
   their source consistency. With no expected count, verified is unconditionally
   true. This is a code-review finding, not a claim of observed production loss.
4. Interrupted export is not resumable: any nonempty target is ALREADY_PRESENT,
   even if a prior batch stopped early. catchup skips child tables without a
   received_at column, so it cannot carry a newly accepted terminal's complete
   evidence. Matching error strings UNCOMPARABLE can also become content_matches
   true in cmd_validate. These paths need explicit failure/completeness tests.
5. Stopping the direct-PostgreSQL loader prevents old-engine writes but does not
   implement the required current data-foundation ingestion route. An empty
   queue at cutover is not proof that future outcomes will reach DuckDB.

Metabase lacking a suitable driver is not by itself a rejection: the original
order allowed functional console analytics. Demonstrate those queries and
clearly distinguish Metabase's old connection from the current cube. Do not
install an unreviewed driver merely to satisfy a label.

## E1 - Freeze failures, then correct migration selection

Write behavioral tests first for the findings above. Build the actual per-run
lineage inventory across both gov_* and df_* records using committed campaign
orders, designs and identity links. Include current negative/inconclusive work;
separate membership from scientific validity and supersession. GOVERNING alone
does not establish membership; NON_GOVERNING need not mean useless history.

Make export consume the reviewed selection manifest and calculate row-level
dependency closure. Test mixed current/legacy runs within ONE table and current
records outside gov_*. Preserve existing archive/source bytes. Reconcile the
corrected selection against production additively, without duplicating outcomes
or changing historical conclusions. Report exactly which evidence was wrongly
archived or wrongly included, not another table count.

## E2 - Implement and prove real rollback

Implement contract-schema creation, parent AND child replay, transactionality,
content validation and idempotent resume against an explicitly disposable target
first. A plain rollback invocation must execute what it claims or fail explicitly;
never report success for REPLAY_REQUIRED. Test two new terminals with metrics,
datasets, artifacts and a nonempty availability contract, then interrupt/restart
replay and run it again. Prove the destination can resolve the contract and query
all outcomes with identical content. Do not roll production back as a test.

## E3 - Consistent snapshots and complete catch-up

Use an engine-supported consistent export or coordinate a write drain and closed
database boundary through its owner. No copying a moving main/WAL pair and then
calling it verified. Test writes near the snapshot boundary, recovery, and full
content/relationship equality, not only terminal count.

Use transactionally complete imports or explicit checkpoints that distinguish
partial from complete relations. Test an interrupted multi-batch import, existing
wrong rows and a second idempotent run. Catch-up includes all dependencies of new
parents, even without timestamps on child rows. Unsupported digest computation
must cause a non-success validation, never equality of two error messages.

## E4 - Restore complete data-foundation ingestion

Implement the df_* writer through the warehouse-owned interface and durable
outbox, preserving the existing event schemas/identities and excluding unrelated
PostgreSQL uses. Prove a real producer -> outbox -> owned DuckDB writer -> fresh
query, including failure, inconclusive result, retry and restart. Then deploy
the successor loader under the existing scoped authorization, only after tests.
Do not simply restart the obsolete direct-PostgreSQL loader. Verify heartbeat,
queue progress and no writes to the former cube. Preserve adjudicated failures.

## E5 - Useful analytics and truthful operational state

Demonstrate console queries for current campaign results, candidate/regime
comparisons, exclusions, costs and provenance. Clearly mark legacy comparison
views and invalidated evidence. Distinguish physical disk savings from scientific
coverage; 1.4 GB versus 6.99 GB is not yet an equivalent-workload benchmark.
Record the actual analytical access URL and configuration for users/agents.
Continue the assigned historical-terms research independently; do not stop these
engineering blocks waiting for it.

## E6 - Complete, deploy narrowly, then report

Owner authorization from the migration order remains in force. Complete every
independent block without requests to continue. Use CPU workers where helpful,
separate disposable databases and bounded resources. Preserve live warehouse
service except for scoped tested cutovers; no scientific reruns or GPUs.
Do not manufacture external review records to bypass an actual prerequisite.

Update method state, traceability and work plan. Publish exact tested revisions,
failing-before/passing-after proofs, migration selection deltas, actual rollback
replay proof, snapshot checks, restored-loader end-to-end receipts and query
examples. If any block cannot complete, report its exact remaining action and
finish the others; do not label the entire order closed.
