# Return: F1–F5

Order: `docs/handoffs/MUSASHI_E1_E6_REVIEW_AND_F1_F5_2026_09_16.md` (`0b1b78e`), over the review
of `d35bb21`. Executed without pausing between blocks. Counts are mine.

## The five findings were correct, and the method behind them matters more than the bugs

Musashi did not read my receipts. He imported the tool and called it, and two defects fell out
in minutes. Both are frozen as tests at `F1_FAILING_BEFORE.txt` — seven rules, all red — before
any fix.

The pattern across these rounds, now stated plainly: **I prove things at the layer I happened to
write, and the layer that ships goes unexercised.** First the probe instead of the reader, then
integrity instead of validity, now helpers instead of commands.

## F1 — the two probes, and the command paths

* `copy_relation` restarted at the first source row. On a constrained table that raises
  `Duplicate key`; on an unconstrained one it **silently doubles rows**. It now resumes from the
  destination's own high-water mark, and a keyless relation that already holds rows is
  **refused** — without an identity, a resumed copy and a duplicated one are the same thing.
* `replay_between` matched parents by identity alone, so a destination holding the parent was
  declared in sync while its metrics, datasets and artifacts were missing. Shared parents are
  reconciled, children compared by content, and a shared identity with **different** content is
  `CONFLICT_REFUSED` rather than silently resolved one way.

**Real cross-engine rollback.** `rollback --target-engine postgres` is the operation; the
DuckDB-to-DuckDB path is now `copy-cube`, named for what it is. Rehearsed on a disposable
database restored from the pre-cutover dump: the PostgreSQL **provider** creates the missing
`gov_availability_contract`, **2 terminals and 3 child rows** are restored, the destination
holds **54** terminals, and a second run replays nothing.

Rules now run through `main()` for export, catchup, rollback and copy-cube.

## F2 — resumability and catch-up

Row-count equality no longer declares completion: `CONTENT_DIFFERS` is its own state and is
refused. An overfilled destination is refused. `catchup` routes through the same parent/child
closure, so children with no `received_at` travel with their parent, and a partially present
outcome is completed. Refused or unresolved work **exits non-zero** instead of reading as
success.

## F3 — the incident, and a root cause I must demote

**The no-loss claim, now supported.** `data-gov`'s own accounting is a separate database the
incident never touched. Reconciled against a **verified snapshot** at child level:

| | |
|---|---|
| accepted by governance | **55** |
| committed in the cube | **55** |
| missing | **0** |
| status disagreements | **0** |

Child content present for 45; the other **10 are exactly the REFUSED** terminals, which have no
deliveries or metrics by construction — checked, not assumed. Both WAL copies are byte-identical
and neither was opened.

> **Correction, 2026-09-16 (G1).** The no-loss statement above was **wrong**, and the evidence
> offered for it could not have detected the error: the verifier hashed the recovered cube and
> compared it with itself. Rebuilt to compare against the canonical payloads `data-gov`'s
> accounting retained, it found **two terminals missing four metric rows** — both DuckDB-era
> writes whose rows were in the write-ahead log I quarantined. They were restored from that
> independent record and the cube now reconciles 55/55 with zero differences. See
> `SATOSHI_G1_G3_RETURN_2026_09_16.md` and `G1_EVIDENCE_RECONCILE.json`. This paragraph is left
> in place rather than edited away.

**The correction.** I stated the root cause as fact. A minimal disposable reproducer — four
arrangements differing in where the schema is created and whether the process is killed before
it can checkpoint — **does not reproduce the failure** on DuckDB 1.5.5; every case reopens. So
the explanation is a **hypothesis** and the start-up schema plus per-write checkpoint are a
**mitigation**. I do not know why the production log could not be replayed. The E1–E6 return is
corrected in place, and the quarantined log is preserved because it is the only artefact that
could still answer the question.

**Snapshots** no longer accept a bare assertion of a boundary: without `--owner-stopped` and an
expected count, the result is unverified with the reason named.

## F4 — membership is not admissibility

I conflated them in **both** directions, and each error is now a column.

Including 52 governed terminals as current did not make them scientific evidence: 48 declared
`NON_GOVERNING` before they ran, which grants nothing scientific by their own declaration. And
B4, T2 and M4 were listed as not current because they are reviewed elsewhere — but the master
plan assigns each a role: T2 belongs to gate **I4** and is explicitly *not mixed with the new
inventory*; M3/M4 enter as **experimental diagnostics** in I2–I3; B4 is **adjudicated under the
contract it ran with**. They are members. Each entry cites the plan line that decides it.

Executable, not documentary: `gov_campaign_disposition` plus three views, published into the
cube and queried through the service — **73 members, 48 `MECHANICAL_OPERATIONAL`, 24
`CURRENT_SCIENTIFIC`, 1 `NON_CONFIRMATORY_HISTORICAL`**. The scientific view holds **19**, not
24, because five of those runs are members whose disposition is legacy. That gap is the two
dimensions working.

## F5 — verification and state

Successor loader verified **after** the fixes with a real envelope: loaded, the persisted result
queried through the service, an idempotent retry reporting `skipped_existing`, and the outbox
reconciled — healthy, backlog zero, the 16 adjudicated failures preserved.

| suite | scope | result |
|---|---|---|
| `olap/store/tests` + both migration files | three engines (`U2_DUCKDB_PATH=1`, `U2_PG_DATABASE=<disposable>`) | **165 passed, 1 skipped** |
| predictor `tests` + `olap/store/tests` | trading-stack, `--ignore=tests/unit_tests --ignore=tests/integration_tests` | **1448 passed, 4 skipped, 0 failed** in 6m38s |

The four skips are the two migration files under trading-stack, which has no `duckdb`; those
same files contribute 28 of the 165 rules in the DuckDB environment. Two lines, because the
environment decides how many rules run and a single number would not say so.

No test dependency was installed into the live service environment: the production interpreter
is untouched, and the test runs use the separate DuckDB environment.

## Open, with owners

| item | owner |
|---|---|
| **why the production write-ahead log could not be replayed** — unreproduced; mitigation in place, cause unknown | Satoshi; the quarantined log is the only remaining artefact |
| Metabase driver decision (not blocking, per the order) | Satoshi |
| acquisition-era terms and API editions | Satoshi — unchanged |
