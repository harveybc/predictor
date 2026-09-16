# H1-H3 review: code progress accepted, live duplicates require reconciliation

Reviewer: Musashi. Executor: Satoshi. Reviewed source: predictor 383fbe7.
Date: 2026-09-16. Existing scoped repair/deployment authorization remains valid.

## Independent checks

Ran the three migration/reconciler test files with the separate DuckDB test
interpreter: 60 passed in 9.04 seconds. This includes field-change detection and
the second-process writer test during snapshot copying. The prior H1/H2 defects
are corrected at the tested scope. No full-suite or PostgreSQL rerun is claimed.
Warehouse and successor loader are active/running with NRestarts=0.

Then compared live warehouse API results against data-gov's independent
governed_terminals.body_json, validating retained payload digests and comparing
parent fields and child multisets using the reviewed mapping. Queried all rows
and checked counts against the returned population; no pagination truncation.

Observed live:
- 55 terminals, 537 metric rows, 97 artifact rows, 92 dataset rows.
- 53 accepted terminals match; 2 differ in their metric multisets only.
- Each affected terminal expects bytes_delivered=228801 and delivery_from_cache=1
  once; the live cube holds each row twice. Four surplus metric rows in total.
- These are the same two terminals identified in the prior incident repair,
  privately named in G1_REPAIR_PROD.json. No terminal identity was absent.
- The parent identity population remained stable across the read sequence.

This was a read-only sequence through the live API, NOT a transactional snapshot.
An initial query with LIMIT 10000 was refused; the successful read used LIMIT
1000 and separate count checks, with all four tables below that bound. No state
was modified and no original WAL was opened.

The committed H1 report may describe an earlier consistent snapshot. It does
not describe these current live contents. The reviewer does not infer when or
which process inserted the surplus rows. Do not claim the repaired rows were
in the incident WAL or that any particular repair caused the duplication without
evidence. A total terminal count of 55 alone cannot detect this discrepancy.

## I1 - Freeze and explain the current discrepancy

Before edits or repair, preserve a consistent evidence copy and the accepted
payloads, using the now-tested snapshot path and a scoped writer drain. Record
exact identities privately, row multisets, timestamps and relevant write/repair/
loader receipts. Compare the committed H1 snapshot with the present source.
Determine the duplicate-producing path if evidence permits; otherwise retain an
explicit unknown cause rather than inventing one.

Do not reopen solved H1/H2 work or replay the original quarantined WAL. Keep the
existing healthy services running except for a necessary coordinated repair.

## I2 - Repair only proven surplus multiplicity and prevent recurrence

Create an idempotent reconciliation/repair operation through the owning process
or a coordinated maintenance boundary. Derive expected multiplicity from validated
accepted payloads, never SELECT DISTINCT or a global deduplication rule. Legitimate
duplicates in a contract must be preserved. Rows with conflicting values are not
automatically repairable.

Rehearse on a disposable copy: the two incident terminals begin with two copies
of each expected row and end with exactly the accepted multiset. Preserve all
before-images and an explicit repair receipt. Make the correction atomic, verify
all unaffected rows unchanged, and prove a second invocation changes nothing.
Test overlapping retries, interruption and the actual deployed writer path.
Inspect whether repair, rollback or normal ingestion can append an already
present child; fix the responsible path only after reproduction.

The owner authorizes this bounded repair under the existing migration/recovery
order. Preserve the surplus rows in the incident evidence before removing them
from the active operational projection. This is not permission to erase history,
change scientific outcomes or deduplicate unrelated tables. A production result
must be reconciled after the write through the service, not only on a rehearsal.

## I3 - Close on measured live content

Re-run independent-payload reconciliation after repair and after a subsequent
loader/retry cycle. Report every expected and observed multiplicity. Present the
actual live result even if new legitimate terminals arrive; do not pin success
to the number 55. Keep scientific and operational populations separate.

Publish code/tests, scoped repair evidence, method-state/work-plan updates and
current live verification. Continue without per-step permission requests. No GPU,
training, original-WAL replay or Metabase driver installation is required. Console
analytics remains accepted; terms research and unexplained WAL failure continue
independently and do not block this repair.
