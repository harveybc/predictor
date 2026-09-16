# U1-U4 review and next orders V1-V4

Date: 2026-09-15. Reviewer: Musashi. Executor: Satoshi.

## Decision

Candidate deployment is deferred for a reproduced data-contract defect, not for
missing owner permission. Existing production services stay running. No new
training, scientific replay, live operation or historical rewrite is ordered.

Reviewed source: predictor `c2944db2e0b7bcb8685f8d25dd9793f82f717fc7`.
This is a scoped independent review, not acceptance of the reported full suite.
The other reported component revisions remain candidates pending integrated
verification. No production migration was performed during this review.

## Independent observations

The production governance service, financial and synthetic lake services,
warehouse service and OLAP loader all reported active/running, NRestarts=0.
This confirms process state, not end-to-end accounting or fresh cube counts.

The existing availability dimension tests pass: 20/20 on SQLite. PostgreSQL
tests and the reported 1395-test suite were not rerun by this reviewer.

High-priority finding: `write_availability_contracts()` and
`resolve_delivery_availability()` accept inconsistent temporal semantics.
On a disposable SQLite database, using the production Plugin and the existing
test fixture's archive contract, changing only completion_lag_max, serializing
with the declared canonicalization and calculating the correct SHA-256 yields:

| Archive lag | Reader result | Reader lag |
|---|---|---|
| UNKNOWN | VERIFIED | UNKNOWN |
| 0s | VERIFIED | 0s |
| not-a-duration | VERIFIED | not-a-duration |
| -1 | VERIFIED | -1 |

Each case used write_availability_contracts, a recorded delivery referencing
that exact digest, and resolve_delivery_availability. All data were temporary.
Hash verification has improved, but integrity of bytes is not validity of the
contract. This contradicts the archive rule that unknown completion must not
be replaced with a known lag.

## V1 - Temporal semantics at writer and reader

1. Freeze the four-case reproducer before edits. Keep UNKNOWN as the positive
   control; the other three archive cases must fail for semantic reasons.
2. Derive validation from the producer's accepted contract, not a new duration
   convention. ARCHIVE_RETROSPECTIVE requires UNKNOWN; other supported classes
   must satisfy their declared rules. Reject booleans, non-finite values,
   negative durations, unsupported shapes and invalid duration strings wherever
   the contract excludes them. Preserve legitimate zero lag in classes that
   explicitly permit it; this is not a blanket prohibition on zero.
3. Validate before storing and independently after reading retained bytes.
   Historical malformed rows must yield a typed unresolved outcome with no
   authoritative semantics, never crash or become VERIFIED.
4. Check the declared canonical serialization against its actual bytes and
   reject ambiguous duplicate JSON keys. Do not silently normalize bytes and
   retain the original digest. Preserve all historical evidence.

## V2 - Integrated regression proof

Run the positive and negative cases on SQLite and disposable PostgreSQL, then
through provider -> host -> data-gov -> warehouse -> fresh reader. Include an
invalid stored row constructed independently of the writer, so writer checks
cannot hide a missing reader check. Exercise actual entry points.

Keep the producer unavailable for the final read. UNKNOWN must remain UNKNOWN
from retained evidence. Invalid contracts must never produce a VERIFIED
semantic claim. Demonstrate that disabling each new semantic check causes its
corresponding test to fail. Report exact commands, environment-dependent test
selection, skips and counts; never point test cleanup at production.

## V3 - Deployment-ready successor

Update the candidate manifest with actual commits and source digests after code
and tests are finalized. Rehearse the additive migration and rollback against a
disposable copy. Compare historical row CONTENT, not merely the existing digest
column or row counts. Preserve new evidence on rollback: reverting application
code must not require deleting newly recorded contracts or receipts.

Return the tested successor for reviewer deployment. Do not restart the live
services or rerun completed experiments for this order. No additional owner
authorization is requested; the remaining condition is passing the integration
tests. Keep the published PENDING_REVIEW distinction truthful until review.

## V4 - Parallel work and closeout

Continue the already assigned primary-source research on acquisition-date terms
and API editions. Distinguish current terms from terms effective at acquisition;
do not infer redistribution rights from an accessible download endpoint. Provide
source URLs, dates, scope and unresolved facts. Do not delete public data or
rewrite repository history while that disposition is undecided.

Use separate workers for independent CPU validation or research where useful;
never share a disposable database between destructive test suites. No GPUs are
needed. Preserve services, historical runs and outboxes. Record the new regression
matrix, current stage and remaining owners in the work plan. Submit one return
packet with the exact passing revision and the reproducible deployment manifest.

Acceptance target: semantically valid retained availability contracts with
independent reader validation, not just a larger passing test count.
