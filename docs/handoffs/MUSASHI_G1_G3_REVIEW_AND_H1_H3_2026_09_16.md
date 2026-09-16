# G1-G3 scoped acceptance and remaining H1-H3

Reviewer: Musashi. Executor: Satoshi. Reviewed return: predictor 6b80923.
Date: 2026-09-16. Existing owner authorization remains valid.

## Accepted evidence, independently checked

Read-only queries through the live warehouse show the two repaired metrics on
each of the two named incident terminals. Their names and values match the
independent data-gov accounting payloads: bytes_delivered=228801 and
delivery_from_cache=1 for both. This accepts restoration of those four metric
values, not blanket preservation of every terminal field or all historical data.

Ran tests/test_incident_reconciler_contract.py and
tests/test_migration_cli_contract.py with the separate store-hosts-duckdb test
interpreter: 23 passed in 3.82 seconds. Full suites and PostgreSQL tests were not
rerun. Warehouse and successor loader are active/running, NRestarts=0.
No production write, service restart, or original WAL access was performed.

## Remaining findings

1. Reconciler completeness: PARENT_FIELDS omits costs, code identity, timestamps,
   tags and other persisted payload fields; dataset CHILD_SHAPES omits
   availability_contract_sha256. Independent disposable probes through main()
   still return NO_LOSS_FOR_THE_COMPARED_POPULATION after changing costs_json
   to wall_seconds=999999 or changing the stored availability contract digest
   to 64 zeroes. This shows unchecked fields, not a new production loss.
2. Snapshot boundary: measure_boundary checkpoints and CLOSES its connection
   before snapshot_database copies the files. A writer can open after the probe
   returns and before copying. This source-level finding is distinct from the
   disclosed same-process limit. The stored copy hashes are not compared against
   independently measured source hashes under a held boundary.
3. Causal attribution: replaying a payload correctly on a disposable cube does
   not establish that the missing rows were in the quarantined WAL or exclude
   the deployed writer/version/transaction path. The recovered missing metrics
   and the reason for their absence are separate findings.

## H1 - Close coverage against the actual persistence contract

Derive the expected parent and child schema from the production terminal
contract and its storage mapping, including optional fields and renamed columns.
Document deliberately non-persisted or service-generated fields. Do not add
another small handpicked list and call it full content.

For each persisted field, test independent mutation/removal/addition through
the real reconciler CLI. Verify costs, executed code identity, time fields and
availability links explicitly. Retained payloads must validate and correspond
to their recorded digest; malformed partial expectations are unverifiable,
not silently skipped. Replayability needs the exact terminal digest/generation
and a validated payload, not just schema/status presence.

Use structured, typed canonical comparison, preserving integer identities and
multiset multiplicity. Repair remains limited to verified missing rows; conflicts
must not be overwritten. Reconcile production read-only at the resulting scope.
Report verified fields, unrepresented fields and unverifiable rows explicitly.
Do not redo the already established four-row restoration.

## H2 - Hold the snapshot boundary through the operation

Hold the actual engine/owner coordination until copying and full source-to-copy
comparison finish, or use an engine-supported transactional snapshot/export.
Closing the probe before copy cannot establish this property. Test a second
process attempting a write exactly after boundary acquisition and before copy;
the outcome must be a consistent snapshot or explicit failure, never a stale
successful probe. Test the actual CLI on disposable databases. Preserve the
original WAL and recovered production files; no incident replay is required.

## H3 - Accurate closure and continued operation

Correct WAL attribution to an unproven hypothesis while retaining the measured
four-row recovery. Unknown root cause alone does not block healthy operation.
Keep the console as the current analytics route; Metabase is not a blocker.
Continue already assigned primary-source terms research separately.

Complete H1-H2 tests and necessary scoped deployment without per-step approval
requests. Preserve production otherwise, no GPU/scientific reruns. Update method
state and work plan; return complete field coverage, snapshot boundary proof
and independently grounded reconciliation. No broad infrastructure redesign.
