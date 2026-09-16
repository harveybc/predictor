# Return: G1–G3

Order: `docs/handoffs/MUSASHI_F1_F5_REVIEW_AND_G1_G3_2026_09_16.md` (`d20dc13`), over the review
of `e99c6ba`. Executed without pausing between blocks.

## The headline: there WAS loss, and my evidence could not have found it

Musashi's finding was exact. `incident_evidence_reconcile.py` computed child hashes from the
recovered cube and compared them with **themselves**. He ran the shipped `main()` three times —
the original fixture, a metric silently changed to `999999`, and all three child tables emptied
— and got **exit 0** every time. A verifier that cannot fail cannot verify.

Rebuilt so that expectations come from `data-gov`'s accounting —
`governed_terminals.body_json` is the canonical payload **as accepted**, in a database the
incident never touched — and run against production evidence, it found:

> **Two terminals accepted by governance were missing four metric rows.**
> `omega-probe-duckdb-cutover-2` and `omega-probe-post-incident-1`, two metrics each.

Both are DuckDB-era writes. Replaying their exact accepted payloads through the provider on a
disposable cube stores both metrics correctly, so the provider is not the cause: those rows
were in the write-ahead log **I quarantined**.

**So "nothing governed was lost" was wrong.** It is corrected in place in the E1–E6 and F1–F5
returns, as a dated note beside the original paragraph rather than an edit over it.

**Repaired** from the independent record: only missing rows added, only under a parent whose
fields already match, inside a transaction, never editing or deleting. Rehearsed on a copy
(55/55) before being applied with the owner stopped. The cube now reconciles **55/55, zero
differences**, verified through the service.

## G1 — what the verifier now does

| separated, because they mean different things | |
|---|---|
| identity present | the terminal is in the cube at all |
| status agreement | its status matches what governance accepted |
| content preservation | parent fields field-by-field; children as **multisets** over their identifying columns, so an absence, an addition and a changed number are each differences |
| replayability | an outbox counts only for the **matching generation** with a recoverable payload — not a filename, not the same campaign and unit |
| unverifiable | no retained payload: `CONTENT_UNVERIFIABLE`, never "preserved" |

An empty accounting can no longer produce a blanket no-loss verdict, and cube rows with no
accepted record are counted and reported. Eight rules cover the reviewer's three cases plus
altered parent fields, wrong generation, legitimate empty refusals, an unavailable payload and
empty populations.

## G2 — resume and snapshot

* **Resume proves its prefix.** `max(key)` says where the destination stops, not that
  everything below it is present: source `[1,2,3]` into `[1,3]` used to copy nothing and leave
  `2` absent. The destination is now reconciled by identity **and** content first; a hole is
  repaired, an extra row or a modified row is refused with the reason.
* **The snapshot measures its boundary.** It takes the exclusive lock itself and checkpoints
  the source before copying; the result is `VERIFIED_SNAPSHOT` or `UNVERIFIED_COPY`, and the
  caller's `owner_stopped` claim is recorded *beside* the measurement rather than trusted.
  Its limit is stated because it is real: DuckDB locks per **process**, so this detects another
  process — the case that matters — and not a writer inside the caller. My first test asserted
  the opposite and was wrong, not the code.

## A second live defect, found on the way

While isolating the missing metrics I wrote rules for child persistence on all three engines.
They pass — which is what told me the provider was not at fault and sent me to the log instead.
Those rules stay: `olap/store/tests/test_terminal_children_are_persisted.py`, six rules.

## Suites

| suite | scope | result |
|---|---|---|
| store + all migration/reconciler files | three engines (`U2_DUCKDB_PATH=1`, `U2_PG_DATABASE=<disposable>`) | **187 passed, 1 skipped** |
| predictor `tests` + `olap/store/tests` | trading-stack, `--ignore=tests/unit_tests --ignore=tests/integration_tests` | **1454 passed, 5 skipped, 0 failed** in 6m32s |

The five skips are the three DuckDB-dependent files under trading-stack, which has no `duckdb`;
those same files contribute 50 of the 187 rules in the DuckDB environment. Two lines, because
the environment decides how many rules run.

Failing-before evidence: `G_FAILING_BEFORE.txt` (13 red). No test dependency was installed into
the live service environment.

## State

Production healthy throughout except for two scoped, tested stops (the repair and the snapshot),
each followed by a verified restart. The quarantined write-ahead log and the recovered main file
are untouched; their digests are recorded. **The root cause of the log failure remains unknown**
— the reproducer still does not reproduce it — and that is not a reason to stop a healthy
warehouse.

## Open, with owners

| item | owner |
|---|---|
| why the write-ahead log could not be replayed | Satoshi; unreproduced, mitigation in place, quarantined log preserved |
| Metabase driver decision (not blocking) | Satoshi |
| acquisition-era terms and API editions | Satoshi |
