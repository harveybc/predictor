# Return: H1–H3

Order: `docs/handoffs/MUSASHI_G1_G3_REVIEW_AND_H1_H3_2026_09_16.md` (`4b28354`), over the review
of `6b80923`. Executed without pausing between blocks. The already-verified four-row
restoration was **not** redone.

## H1 — coverage derived from the contract, not chosen by me

Musashi's two probes were exact: changing `costs_json` to `wall_seconds=999999`, or a dataset's
availability contract digest to 64 zeroes, both still returned `NO_LOSS`. My `PARENT_FIELDS`
was another handpicked list, and the dataset shape omitted the contract link.

The comparison set is now **derived from the store's own contract constants**
(`TERMINAL_KEYS`, `TERMINAL_DATASET_KEYS`) plus an explicit column map for the three fields
stored as JSON text, and an explicit list of payload fields that are deliberately not on the
parent row, each with its reason. Result: **17 parent fields** and **all 13 dataset fields**.

Every report now carries `field_coverage`, so the claim is checkable rather than asserted —
measured on production evidence:

| relation | compared | excused | stored but unrepresented |
|---|---|---|---|
| `gov_terminal` | 18 | 1 (`received_at`, service-generated) | **0** |
| `gov_terminal_metric` | 9 | 0 | **0** |
| `gov_terminal_artifact` | 4 | 0 | **0** |
| `gov_terminal_dataset` | 14 | 0 | **0** |

Comparison is **typed**: an integer identity is not the float of equal magnitude, JSON columns
are compared parsed rather than as text, multiset multiplicity is preserved, and child
differences **name the fields that changed** instead of rendering positional values. A retained
payload must itself validate against the digest it is filed under and carry the full key set,
or it is `CONTENT_UNVERIFIABLE` — an expectation that does not hash to its own identity is not
an expectation. Replayability already required the exact generation and a recoverable payload.

**The fixture had to become a real terminal.** It invented its digest (`sha256("terminal-0")`)
and wrote only some dataset columns, so it could not serve as an expectation at all. It is now
built from a contract-valid payload, and the tests take their expectation from the builder
rather than from a copy beside it.

**Production reconciled read-only** at this scope, from a boundary-held snapshot:
**55 accepted, 55 matching, 0 differing, 0 unverifiable, 0 orphan rows.** No new loss.

## H2 — the boundary is held through the operation

The probe used to checkpoint and **close** before the files were copied, so anything could
write in between: the measurement described a moment that had already passed. The exclusive
connection is now taken and **kept** across the checkpoint, the measurement of the source, the
copy of both files and the comparison of the copy against that measurement.

Two rules cover it: a second **process** attempting a write exactly during the copy is
**refused**, and the copy's content digests must equal the source's, per relation. The outcome
is named `VERIFIED_SNAPSHOT` or `UNVERIFIED_COPY`; the caller's `owner_stopped` claim is
recorded beside the measurement, never trusted. The documented limit stands: DuckDB locks per
process.

## H3 — a retracted attribution

I wrote that the four missing rows "were in the write-ahead log I quarantined". **That does not
follow**, and Musashi is right to separate it. A payload replaying correctly on a disposable
cube rules out the provider *logic*; it says nothing about where the rows went, and does not
exclude the deployed writer, its version or the transaction path at the time.

- **Established:** four metric rows accepted by governance were absent from the cube, and have
  been restored from the independent record; the cube now reconciles 55/55 at full field scope.
- **Unproven:** where they were lost, and why the write-ahead log could not be replayed.

Corrected in place in the G1–G3 return, beside the original sentence rather than over it.

Unknown root cause is not a reason to stop a healthy warehouse, the console remains the
analytics route, and the terms research continues separately.

## Suites

| suite | scope | result |
|---|---|---|
| store + migration + reconciler files | three engines (`U2_DUCKDB_PATH=1`, `U2_PG_DATABASE=<disposable>`) | **206 passed, 1 skipped** |
| predictor `tests` + `olap/store/tests` | trading-stack, `--ignore=tests/unit_tests --ignore=tests/integration_tests` | **1454 passed, 5 skipped, 0 failed** in 7m10s |

The five skips are the three DuckDB-dependent files under trading-stack, which has no `duckdb`;
those same files contribute 69 of the 206 rules in the DuckDB environment.

Failing-before evidence: `H_FAILING_BEFORE.txt` (17 red). No test dependency was installed in
the live service environment.

## Open, with owners

| item | owner |
|---|---|
| where the four rows were lost, and why the log could not be replayed — both unproven | Satoshi; quarantined log preserved, digests recorded |
| Metabase driver decision (not blocking) | Satoshi |
| acquisition-era terms and API editions | Satoshi |
