# Return: E1–E6

Order: `docs/handoffs/MUSASHI_DUCKDB_CLOSEOUT_CORRECTIONS_2026_09_16.md` (`eed1077`), over the
review of `68f4e39`. Executed without pausing between blocks. Counts are mine.

## Musashi's five findings were all correct, and all mine

Each is the same failure in a different place: **the tooling reported an outcome it had not
performed.** That is the third round in which the correction I needed was to check the property
that was asked for rather than the one I had built.

Failing-before evidence is committed at `E1_FAILING_BEFORE.txt`: thirteen rules, all red,
written before any fix.

## E1 — selection from lineage, and what my previous selection got wrong

`df_` is a naming convention, not a campaign. Membership now comes from each run's recorded
campaign, checked against a **reviewable declaration** —
`docs/audits/work_plan/CURRENT_CAMPAIGNS.json` — of the campaigns the committed orders name.
Governed terminals are listed individually there rather than matched by prefix, precisely
because a prefix is the criterion being corrected.

`dim_campaign.run_id` does **not** join to the data-foundation runs at all; measured, not
assumed. Their identity is the order item each run recorded in `module` (C130, D0, D2 …), and
that is what is used.

**The delta, stated as the order requires — which evidence was wrongly archived:**

| | before | now |
|---|---|---|
| governed terminals | 4 `INCLUDED_CURRENT`, 48 `MECHANICAL_ONLY` | **52 `INCLUDED_CURRENT`**, statuses preserved (20 FAILED, 10 REFUSED, 22 COMPLETED) |
| data-foundation runs | not inventoried at all | 16 current, 6 legacy |
| relations | 64 archived by prefix | 32 whole-included, 51 legacy, **2 MIXED** |
| **current rows wrongly archived** | — | **1,757,695** |

Two relations hold both current and legacy runs (`df_dim_run`, `df_fact_load_receipt`), which a
whole-table decision cannot express; the export now consumes the manifest and selects by
`run_id`. Reconciled into production **additively**: 1,757,695 rows added, 15 relations left
untouched, no duplication, no historical conclusion changed.

Nothing was wrongly *included*: `df_fact_d2_unit_denoising` (2,941,709 rows) has zero current
rows and remains in the archive, where it is verified present.

## E2 — a rollback that acts

`cmd_rollback` used to emit `REPLAY_REQUIRED` and exit zero having written nothing. It now
performs the replay: parents and children in **one transaction**, a relation the destination
lacks created through the provider's own DDL, idempotent and resumable by identity. `--dry-run`
is the only reporting mode and says `WOULD_REPLAY`.

Proved on disposable targets: two terminals with metrics, datasets, artifacts and a non-empty
availability contract; an interrupted replay resumed and completed; a second run replayed
nothing; the destination **resolves the replayed contract** as `VERIFIED` /
`ARCHIVE_RETROSPECTIVE` / `UNKNOWN`; and every relation's content digest matches after replay.

## E3 — snapshots with a boundary, and honest comparisons

A snapshot is verified only with a **coordinated writer boundary** and a count to compare
against. With the owner running, or with no expected count, `verified` is now **false** with the
reason named — it used to be unconditionally true.

Two uncomparable digests are no longer a match: both sides failing the same way is not
agreement. An empty relation is distinguished from an unreadable one. An interrupted import
**resumes** instead of being declared `ALREADY_PRESENT`, and a destination holding more rows
than the source is refused. Catch-up carries a new parent's children, which have no timestamp.

After the cutover the destination legitimately leads the legacy source. That is reported as
`DESTINATION_AHEAD` only when the superset is **proved** by anti-join: measured, **0** source
rows absent from the destination, with 2 destination-only terminals.

## E4 — the ingestion route, and an incident I caused

The `df_*` writer now goes through the owned interface. Proved with three real envelopes:
loaded, idempotent on a second drain, held pending while the owner was down, drained after
restart. The successor unit `crispdm-olap-loader-duckdb.service` is enabled and active,
heartbeat healthy, queue zero, and the 16 adjudicated failures preserved.

**The incident.** Creating the foundation schema *during a write* produced a write-ahead log
DuckDB could not replay on reopen, and the production cube became **unopenable**. I quarantined
the log — copied twice, never deleted — and the main file opened intact: 54 terminals, 16
foundation runs, 440,694 coverage rows. **Nothing governed was lost**; what the log held were
my own test envelope writes, which are re-drainable.

Root cause fixed: the schema is created at start-up, where it is checkpointed before any
envelope arrives, and each envelope write is checkpointed. The exact failure is now a check
that writes, closes and reopens. A governed delivery and terminal afterwards: **201**,
reconciliation empty.

Two further portability defects, both found by running it: `JSONB` does not exist in DuckDB,
and `rowcount` is `-1` there for `INSERT … ON CONFLICT`, so the loader reported `-1` units and
`-1` consumption rows for a successful envelope. Both counted with `RETURNING`. The PostgreSQL
path is verified unbroken against a disposable database.

## E5 — analytics, and an honest comparison

Six representative console queries are recorded with their results in
`E5_CONSOLE_QUERIES.txt`: current campaign results, the successor run's compute contract, costs
by project, delivery provenance, **exclusions with their reasons**, and data-foundation
coverage.

**Access:** `http://127.0.0.1:5057/` for the console, `GET /api/v1/query` with `LIMIT` required
and a bearer token from the host's environment file. Legacy evidence is in a **separate
database file** and is not reachable from those queries at all, which is a stronger separation
than a marked view.

**The comparison I overstated:** 1.4 GB against 6.99 GB is disk, not coverage, and the two
figures do not describe the same content — the cube holds the current campaign, the archive
holds the rest. No equivalent-workload benchmark was run, and none is claimed.

## E6 — state

| suite | scope | result |
|---|---|---|
| `olap/store/tests` + `tests/test_olap_duckdb_migrate.py` | three engines (`U2_DUCKDB_PATH=1`, `U2_PG_DATABASE=<disposable>`) | **155 passed, 1 skipped** |
| predictor `tests` + `olap/store/tests` | trading-stack, legacy dirs excluded | recorded below |

Services: `:5055`, `:5056`, `:5057`, `:5058` all 200; the successor loader active. The legacy
direct-PostgreSQL loader remains stopped and disabled. PostgreSQL itself is untouched as the
rollback source, and no unrelated PostgreSQL use changed.

## Open, with owners

| item | owner |
|---|---|
| Metabase v0.56.3 has no DuckDB driver; analytics are the console and `/api/v1/query`. Installing a third-party driver is a decision, not an engineering step | Satoshi, on that decision |
| acquisition-era terms and the separate API editions (V4) | Satoshi — unchanged, blocked on documents a browser session would fetch |
| the quarantined incident log is kept as evidence and is not replayed | Musashi, if he wants it examined |
