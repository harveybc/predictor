# Live warehouse migration plan: fs_phase23_0001 (dry; NOT applied)

Status 2026-10-05: tested on throwaway DuckDB files and dry-run read-only against the
local phase-1 snapshot copy. **Nothing has been applied to the live cube.** The live cube
has one owner, the warehouse service; this agent never opened it and never restarts it.

## The deployed target

- Service role: `crispdm-data-warehouse-olap.service` (system unit, port 5057), a
  `data-warehouse` host loading backend `predictor_duckdb` from
  `olap/duckdb_store` (`predictor-duckdb-store 0.1.3`) over `olap/store`
  (`predictor-olap-store 0.1.4`), in the warehouse host's own virtualenv.
- Engine: DuckDB, schema `main`, single file plus write-ahead log. Memory limit and
  thread count come from the host configuration.
- Its *only* schema mechanism is `PredictorDuckdbStore._ensure_schema`, which runs every
  statement of `Plugin._ddl()` (all `CREATE ... IF NOT EXISTS`) at service start, before any
  envelope write. Phase-1 tables arrived exactly this way
  (`feature_selection_store.ddl()` appended in `query.py::_ddl`). There is no separate
  migration runner for the live file. The legacy `olap/etl_migrate_v2.py` and the Postgres
  `.sql` files target a different, incompatible schema and are not used.

## The migration

`olap/migrations/fs_phase23/0001_fs_phase23_additive.sql`
(sha256 `207379ff49b777e9a69c9e7e8645e4b9c42952d28e4c0fbfa332af75bb1d6473`):
7 statements, 6 `CREATE TABLE IF NOT EXISTS`, 1 `CREATE OR REPLACE VIEW`, 0 destructive.

| relation | grain | UNIQUE identity key |
|---|---|---|
| `fs_phase23_run` | one phase-2/3 run bound to one population and the contract digest | `run_id`; `run_sha256` |
| `fs_phase23_load_receipt` | one submission | `receipt_sha256` |
| `feature_pair_metrics` | ordered pair x fold x lag x method x params/code/input digests | run_id, population_id, feature_left, feature_right, fold, lag, method, params_sha256, code_sha256, input_sha256 |
| `feature_alias_groups` | feature membership in an alias group | run_id, population_id, group_id, feature_id, fold, method, params_sha256, code_sha256, input_sha256 |
| `feature_redundancy_clusters` | feature membership in a cluster | run_id, population_id, fold, method, params_sha256, code_sha256, input_sha256, cluster_id, feature_id |
| `feature_filter_rankings` | feature rank per target x horizon x fold x method | run_id, population_id, target_id, horizon, fold, method, params_sha256, code_sha256, input_sha256, feature_id |
| `fs_phase23_coverage` (view) | per run counts | - |

Every fact row also carries `row_identity_sha256` (PRIMARY KEY, digest of the identity
columns), `row_sha256` (content digest), `split` fixed to `'train'`, `shared_support`,
`state` in {MEASURED, INSUFFICIENT_SUPPORT, NOT_APPLICABLE, FAILED}, `host_role` (role only),
`shard_id`, `terminal_sha256`, `extra_json`, `stored_at`.

Dry run against the local phase-1 snapshot copy (read-only, 308,555,776 bytes,
sha256 `0e9fb23d...7918`, the file the published release asset decompresses to):
`MIGRATION_DRYRUN_phase1_snapshot.json` - would create all six relations, zero already
present, zero name collisions with the 8 phase-1 tables, 6 phase-1 views and the `gov_*`
relations.

## Steps for the integration agent (the governed path)

1. Register the DDL in the backend so the service's own start-up mechanism creates the
   relations: in `olap/store/src/predictor_olap_store/query.py::_ddl`, append the
   statements of the SQL file after `feature_selection_ddl(t, dialect)` (load them with
   `tools/fs_phase23_warehouse.load_ddl()` or an equivalent packaged copy; the file is the
   single source and `tests/test_fs_phase23_warehouse.py` pins its content). Bump
   `predictor-olap-store` to 0.1.5 and `predictor-duckdb-store` to 0.1.4, update the parity
   digest the store tests pin, run `olap/store/tests` and `tests/test_fs_phase23_warehouse.py`.
2. Install the bumped packages into the warehouse host's virtualenv on the coordinator.
3. **Service restart** (`systemctl restart crispdm-data-warehouse-olap.service`) so
   `_ensure_schema` runs the additive DDL at start-up, checkpointed before any write. This is
   the integration agent's step; it was not executed here.
4. Verify through the running service's read-only route:
   `SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'feature_%' OR table_name LIKE 'fs_phase23%' LIMIT 20`
   must list the six relations.
5. Apply to the local snapshot copy only AFTER the live cube has it, never before; the
   snapshot is a copy of the cube, not a second source of truth.

Alternative if the backend route is delayed: `tools/fs_phase23_warehouse.py migrate --duckdb <cube>`
applies the same file directly, but it can only run while the owning service is stopped
(DuckDB single writer). That is a stop-apply-start sequence owned by the integration agent;
it is recorded here as the fallback, not the plan.

## Write path (open item, not in this repository)

The host exposes typed write routes only (`/api/v2/terminals`,
`/api/v2/feature-selection-envelopes`, `/api/v2/feature-selection-reconcile`) plus the
read-only `/api/v1/query`. No route accepts phase-2/3 rows. `submit_rows`/`verify_receipt`
in `tools/fs_phase23_warehouse.py` are the backend-side semantics; wiring them as a
capability `write_fs_phase23_rows` + host route `/api/v2/fs-phase23/rows` requires a change
in the `data-warehouse` repository (sibling) and a host redeploy. Until then the follower
writes terminals locally, and the warehouse load happens through that route once it exists;
**readback and reconciliation already work today** through `/api/v1/query` (same SQL
expression as the local path, see `READBACK_REPORT_FORMAT.md`).

## Second open item

`tools/olap_duckdb_migrate.py snapshot` verifies a boundary over `GOVERNANCE` +
`FEATURE_SELECTION_TABLES`; add the six fs_phase23 relations to that boundary so a phase-2
snapshot is `VERIFIED` rather than `UNVERIFIED_COPY`. One tuple edit plus its test.
