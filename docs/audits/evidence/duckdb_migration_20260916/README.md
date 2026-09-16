# DuckDB is the OLAP engine (D0–D6)

Executed 2026-09-16 on omega. PostgreSQL was replaced **only** as the OLAP warehouse storage
engine. Nothing else that uses PostgreSQL was touched: Metabase's own application database,
the `fxpg` container and every other project are exactly as they were, and no PostgreSQL was
stopped, uninstalled or rewritten. The pre-cutover database is intact as the rollback source.

## What runs now

    warehouse host  :5057   data-warehouse-service, venv ~/.venvs/store-hosts-duckdb-prod
    backend                 predictor-duckdb-store 0.1.0, entry point predictor_duckdb
    engine                  DuckDB 1.5.5
    database                ~/.local/state/crispdm-duckdb/prod/cube.duckdb
    archive                 ~/.local/state/crispdm-duckdb/prod/archive.duckdb  (separate file)

Reported by the **running process** through `/api/v1/host`, not read from configuration.
`:5055`, `:5056`, `:5057` and `:5058` all answer 200.

## D0 — what was actually there

85 relations, 6.99 GB. `gov_*` held **768** rows; `df_*` held **6.56M**. That asymmetry is what
separates the current campaign from the archive, and it is measured rather than assumed.
Writers found: the warehouse host (governed `gov_*`) and `tools/olap_loader.py` (legacy `df_*`,
direct to PostgreSQL). Readers: data-gov, the host console, Metabase as a data source.

## D1 — selection by declared identity

`D1_SELECTION.json`. Membership comes from the classification each campaign **declared before
execution**, never from a date, a model family, a metric's sign, defaults or a table name.

| disposition | count |
|---|---|
| `INCLUDED_CURRENT` runs | 4 (all GOVERNING) |
| `MECHANICAL_ONLY` runs | 48 |
| `INCLUDED_CURRENT` relations (closure) | 14 |
| `LEGACY_COMPARISON_ONLY` relations | 64 |

All 48 mechanical runs keep their real statuses — 20 FAILED, 9 REFUSED, 19 COMPLETED — because
dropping them would be selecting by outcome. Stated plainly so it cannot look like filtering:
**all four GOVERNING campaigns happen to be COMPLETED.** No governed negative or failed outcome
exists to include; none was excluded.

## D3 — migration, and the defect the content check caught

Source opened `READ_ONLY` through DuckDB's own `postgres` extension, so both halves of every
comparison are computed by the same engine and compare data rather than formatting.

The defect: my batches used `LIMIT/OFFSET` with no `ORDER BY`, which is not pagination. Nine
archive relations came out with the **right row count and the wrong rows**. A count-only
validation would have called it a success; the order's insistence on content is what found it.
Replaced with keyset pagination on a verified-unique column.

After the fix:

| scope | relations | rows | content digests |
|---|---|---|---|
| cube | 21 / 21 | 1,351 | all match |
| archive | 64 / 64 | 6,860,035 | all match |

A second export copies **0** rows and reports every relation `ALREADY_PRESENT`.

## D5 — the cutover, and two more defects

Backup `cube.before-duckdb.dump` (424 MB) plus counts and an md5 over all terminal digests.
All three outboxes drained to **0 pending** before the owner was stopped. Catch-up from the
watermark added 0 rows, and the final content validation was 21/21.

* **Constraints do not survive `CREATE TABLE AS SELECT`.** The migrated `gov_terminal` had no
  primary key, so `ON CONFLICT (terminal_sha256)` had no conflict target: the cube answered
  every query and **refused every governed terminal**. Only running a real governed campaign in
  production could find this. Governed tables are now created by the provider's own DDL.
* **`describe()` reads `storage()["root"]`**, which the DuckDB `storage()` did not return, so
  `/api/v1/describe` answered 500 on an otherwise healthy service.

The unit that was stranded by the first defect was **closed as REFUSED** with the cause in its
reason, not left open: `2ad77e6b…` / `omega-probe-duckdb-cutover-1`.

## D4 — acceptance

On a disposable stack with the real host, provider, data-gov and lake: archive delivered and
closed; range, point-in-time and live-tail each refused 422 for the declared reason; a plain
resource delivered as a regression; a temporally invalid contract refused **400** by the host
over HTTP and refused again on read when planted directly. Then the lake **and** the warehouse
service were stopped, and the fresh reader still answered `VERIFIED` /
`ARCHIVE_RETROSPECTIVE` / `UNKNOWN` — the evidence survives the service, not only the producer.

In production: a bounded governed delivery and terminal, **201**, reconciliation empty on all
three lists, delivery `VERIFIED_CACHE`. The cube now holds **54** terminals and answers
representative queries, including the S1 compute-contract metrics.

## Rollback, rehearsed — and the third defect

`cp cube.duckdb` is **not** a snapshot: recent transactions live in `cube.duckdb.wal`. A copy
taken while two freshly accepted terminals were outstanding contained **neither**, and the
rollback rehearsal against that copy reported "nothing to replay". The `snapshot` command now
copies both files, `CHECKPOINT`s, reopens and verifies against what the service reports.

Against a disposable database restored from the pre-cutover dump, the rehearsal correctly
reports:

* `gov_terminal` — **2** DuckDB-era terminals absent from PostgreSQL, which a rollback must
  replay or lose;
* `gov_availability_contract` — **ABSENT_IN_POSTGRES**: the relation does not exist there at
  all and must be created before any replay.

The DuckDB evidence is never dropped to make rollback easier.

## Declared gaps, with owners

| gap | owner |
|---|---|
| Metabase v0.56.3 ships no DuckDB driver (verified against its own API). Analytics today are the host console and `/api/v1/query` | Satoshi, on a decision about third-party drivers |
| `tools/olap_loader.py` writes `df_*` straight to PostgreSQL. It is **stopped and disabled** — it was the only other OLAP writer and its outbox is empty. Any future `df_*` work needs a DuckDB-side route through the owned interface first | Satoshi, when `df_*` work resumes |
| the availability dimension is not in the deployed PostgreSQL schema, so rollback must create it | recorded in the rollback report |
