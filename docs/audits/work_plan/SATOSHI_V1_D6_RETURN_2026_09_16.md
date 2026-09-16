# Return: V1–V4 and D0–D6 — DuckDB is the OLAP engine

Orders: `MUSASHI_U1_U4_REVIEW_AND_V1_V4_2026_09_15.md` (`a2538d3`) and
`MUSASHI_DUCKDB_WAREHOUSE_MIGRATION_ORDER_2026_09_16.md` (`757fd5c`). Executed without pausing
between blocks, as instructed. Counts are mine and await independent verification.

## The finding I was wrong about, again, and what it cost

Musashi's reproduction was exact. I had made the reader verify that retained bytes hash to the
digest they are filed under, and treated that as validity. It is not: an
`ARCHIVE_RETROSPECTIVE` declaring `0s`, `-1` or `not-a-duration` hashes perfectly and came back
**VERIFIED**, with the lag handed over as though it meant something.

The reproducer was frozen **before** any edit (`REPRODUCER_BEFORE.json`): control survives,
three counterexamples wrongly verified. After (`REPRODUCER_AFTER.json`): control survives,
all three refused with semantic reasons.

That is twice now — first hashing in the probe instead of the reader, now treating integrity as
validity. The pattern in both is the same: I checked the property I had built rather than the
property that was asked for.

## V1 — temporal semantics, at both seams

The rules are **the producer's own** (`financial_data_store.inventory.availability_scope`),
mirrored rather than reinvented, because a contract that is valid where it is written and
meaningless where it is read is the defect in another costume:

* a retrospective archive declares `UNKNOWN` and nothing else;
* every other class needs a finite, non-negative, parseable duration;
* `LIVE_EQUIVALENT` still requires exactly zero with a known label and a producer time-zone
  statement — so zero stays legitimate where the contract permits it. This is not a ban on zero.

Enforced on write **and independently on read**, because a reader that inherits the writer's
checks cannot see a missing one. Retained bytes are also checked against the canonical form
they declare, and duplicate JSON keys refused; neither is normalised under the original digest,
and no historical evidence is rewritten.

**Eleven mutation rules** prove each check is what refuses. One corrected a claim of mine:
removing the duplicate-key hook alone does **not** admit an ambiguous contract — the
canonical-form check catches it too. The hook names the reason; the two together are the floor.

## V2 — integrated regression

Positive and negative on SQLite, disposable PostgreSQL **and** DuckDB. Through the real route:
the host refuses a temporally invalid contract with **400** and the semantic reason, and a row
planted straight into the store — constructed independently of the writer — is refused again on
read as `UNRESOLVED_INVALID_TEMPORAL_SEMANTICS`, with no availability claim.

## D0 — what was actually there

85 relations, 6.99 GB. `gov_*` 768 rows; `df_*` 6.56M. Writers: the warehouse host and
`tools/olap_loader.py`. Mentions in code were kept distinct from calls a running unit makes.

## D1 — selection by declared identity

4 `INCLUDED_CURRENT` (all GOVERNING), 48 `MECHANICAL_ONLY` keeping their real statuses (20
FAILED, 9 REFUSED, 19 COMPLETED), 14 closure relations, 64 `LEGACY_COMPARISON_ONLY`. Said
plainly so it cannot look like filtering: **all four governing campaigns happen to be
COMPLETED**; no governed negative exists to include and none was excluded.

## D2–D3 — the provider and the tooling

`predictor-duckdb-store` is an external provider through the host's existing entry-point group,
built in an **isolated** environment; the live venv was never mutated. It subclasses the cube
rather than copying it.

Defects found by running it:

* DuckDB returns `rowcount = -1` for `INSERT … ON CONFLICT DO NOTHING`, so the idempotency
  counter reported "-1 stored, 2 already stored" for one new contract. Insertions are now
  **counted** with `RETURNING`;
* my migration batches used `LIMIT/OFFSET` with no `ORDER BY`, which is not pagination. Nine
  archive relations came out with the **right row count and the wrong rows** — a count-only
  validation would have passed them. Replaced with keyset pagination.

After the fix: cube 21/21 relations (1,351 rows), archive 64/64 (**6,860,035** rows), all by
order-independent content digest, both sides computed by the same engine. A second export
copies zero.

## D4–D5 — acceptance and deployment

Disposable stack with the real host, provider, data-gov and lake: whole archive delivered and
closed; range, point-in-time and live-tail refused 422 for the declared reason; plain resource
delivered as regression; the negative contract refused at the host and again on read. Then the
lake **and the warehouse service** were stopped, and the fresh reader still answered
`VERIFIED` / `ARCHIVE_RETROSPECTIVE` / `UNKNOWN` — evidence survives the service, not only the
producer.

Cutover: backup (424 MB) plus counts and an md5 over all terminal digests; all three outboxes
drained to **0 pending**; owner stopped; export and content validation; catch-up added 0 rows;
unit repointed at the isolated production venv; service started.

Two more defects, each found only by doing it:

* **constraints do not survive `CREATE TABLE AS SELECT`**. The migrated `gov_terminal` had no
  primary key, so `ON CONFLICT` had no conflict target: the cube answered every query and
  **refused every governed terminal**. Governed tables are now created by the provider's DDL;
* `describe()` reads `storage()["root"]`, absent from the DuckDB `storage()`, so
  `/api/v1/describe` answered 500 on a healthy service.

The unit stranded by the first was **closed as REFUSED** with the cause in its reason.

Production acceptance: governed delivery and terminal **201**, reconciliation empty on all
three lists, delivery `VERIFIED_CACHE`. The cube holds **54** terminals and serves
representative queries including the S1 compute-contract metrics. `/api/v1/host` reports
DuckDB 1.5.5 and the open file — the running engine, not the configuration.

## Rollback — and the third defect

`cp cube.duckdb` is not a snapshot: recent transactions live in the WAL, and a copy taken while
two freshly accepted terminals were outstanding contained **neither**; the rehearsal against it
reported "nothing to replay". The `snapshot` command now copies both files, `CHECKPOINT`s,
reopens and verifies against what the service reports.

Rehearsed against a disposable database restored from the pre-cutover dump: it correctly names
the **2** DuckDB-era terminals a naive rollback would lose, and `gov_availability_contract` as
**ABSENT_IN_POSTGRES** — it must be created there before any replay. New evidence is never
dropped to make rollback easier.

## Scope kept

PostgreSQL was replaced **only** as the OLAP warehouse engine. Metabase's application database,
the `fxpg` container and every other use are untouched; no PostgreSQL was stopped or
uninstalled; the pre-cutover database is intact as the rollback source. Data-gov accounting and
both lakes are unchanged. No scientific campaign was re-run, no GPU used, no result promoted.

## Declared gaps, with owners

| gap | owner |
|---|---|
| Metabase v0.56.3 ships **no** DuckDB driver — verified against its own `/api/session/properties`, not assumed. Analytics today are the host console and `/api/v1/query`. PostgreSQL is **not** kept quietly serving the cube | Satoshi, pending a decision on third-party drivers |
| `tools/olap_loader.py` writes `df_*` straight to PostgreSQL. Stopped and disabled: it was the only other OLAP writer and its outbox is empty. Future `df_*` work needs a route through the owned interface first | Satoshi, when `df_*` work resumes |
| the availability dimension does not exist in the deployed PostgreSQL schema, so rollback must create it before replaying | in the rollback report |
| acquisition-era terms and the separate API editions (V4) | Satoshi — unchanged, still blocked on documents a browser session would fetch |

## Suites and exact scope

| suite | command scope | result |
|---|---|---|
| `olap/store/tests`, three engines | `U2_DUCKDB_PATH=1 U2_PG_DATABASE=<disposable>` under the DuckDB venv | **136 passed, 1 skipped** |
| predictor `tests` + `olap/store/tests` | trading-stack, `--ignore=tests/unit_tests --ignore=tests/integration_tests` (stale per AGENTS.md) | **1447 passed, 2 skipped, 0 failed** in 6m44s |
| data-gov / data-warehouse / data-lake / financial-data store | unchanged from U1–U4 | 167 / 28 / 33 / 22 |

The two environment variables decide what runs: without `U2_DUCKDB_PATH` the DuckDB half is
absent, and without `U2_PG_DATABASE` the PostgreSQL half is. Same files, fewer rules, and the
count alone would not say so.
