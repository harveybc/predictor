# Live warehouse migration plan: fs_phase23_0001 (dry; NOT applied)

Status 2026-10-06: proved on throwaway DuckDB/SQLite files (packaged backend and local tool)
and dry-run read-only against the local phase-1 snapshot copy. **Nothing has been applied to
the live cube and no service was restarted.** The exact install/restart sequence is in
`LIVE_DEPLOY_STEPS.md` for the integration agent.

## The deployed target

- Service role: `crispdm-data-warehouse-olap.service` (system unit, port 5057): a
  `data-warehouse` host (deployed at data-warehouse revision 2d4550d) loading backend
  `predictor_duckdb` (`olap/duckdb_store`, predictor-duckdb-store 0.1.3 deployed) over
  `olap/store` (predictor-olap-store 0.1.4 deployed), in the warehouse host's own virtualenv.
- Engine: DuckDB, schema `main`, single file plus write-ahead log; one owner process.
- The only schema mechanism of the live file is `PredictorDuckdbStore._ensure_schema`, which
  runs every statement of `Plugin._ddl()` (all `CREATE ... IF NOT EXISTS`) at service start,
  checkpointed before any write. Phase-1 tables arrived exactly this way. The legacy
  `olap/etl_migrate_v2.py` and the Postgres `.sql` files target an incompatible schema.

## The migration (single source: `predictor_olap_store.fs_phase23_store.ddl()`)

`olap/migrations/fs_phase23/0001_fs_phase23_additive.sql` is GENERATED from that function
(`fs_phase23_warehouse.py render-migration`; a test fails when they differ). 10 statements:
9 `CREATE TABLE IF NOT EXISTS`, 1 `CREATE OR REPLACE VIEW` (`CREATE VIEW IF NOT EXISTS` on
SQLite), 0 destructive. `query.py::_ddl` appends them after the phase-1 DDL, so the service's
own start-up creates them.

| relation | grain | keys |
|---|---|---|
| `fs_phase23_run` | run bound to one population; CONTRACT_BOUND or FIRST_SUBMISSION | `run_id` PK, `run_sha256` UNIQUE |
| `fs_phase23_load_receipt` | one submission | `receipt_sha256` PK |
| `feature_pair_metrics` | ordered pair x fold x metric x lag x method x params/code digests | PK `row_identity_sha256`; UNIQUE (run_id, row_key); UNIQUE (run_id, population_id, feature_left, feature_right, fold, metric, lag, method, params_sha256, code_sha256) |
| `feature_pair_stability` | ordered pair x metric x lag (across inner folds) | ... UNIQUE (run_id, population_id, feature_left, feature_right, metric, lag, method, params_sha256, code_sha256) |
| `feature_pair_gate` | ordered pair identity/equivalence gate on TRAIN | ... UNIQUE (run_id, population_id, feature_left, feature_right, fold, method, params_sha256, code_sha256) |
| `feature_alias_groups` | one alias group (members, representative, disposition) | ... UNIQUE (run_id, population_id, group_id) |
| `feature_redundancy_clusters` | one cluster (members, representative, rule) | ... UNIQUE (run_id, population_id, cluster_id) |
| `feature_filter_rankings` | feature rank per target x horizon x method | ... UNIQUE (run_id, population_id, target_id, horizon, method, params_sha256, feature_id) |
| `feature_filter_subsets` | K-subset per target x horizon x method | ... UNIQUE (run_id, population_id, target_id, horizon, method, params_sha256, k) |
| `fs_phase23_coverage` (view) | per-run counts | - |

Every fact row carries `row_identity_sha256` (digest of the typed identity columns),
`row_sha256` (sha256 of the canonical JSON of the submitted row), `row_key` (the driver's
key), `unit_id`, typed identity + queryable value columns, `host_role` (roles only),
`shard_id`, `payload` (the submitted row, returned verbatim by readback), `stored_at`.
CHECKs: `feature_left < feature_right`, `lag >= 0`, `state` vocabulary.

Dry run against the local phase-1 snapshot copy (read-only, 308,555,776 bytes,
sha256 `0e9fb23d...7918`): `MIGRATION_DRYRUN_phase1_snapshot.json` - would create all ten
relations, zero already present, zero collisions with the phase-1 and `gov_*` relations.

## Write path (implemented, NOT deployed)

data-warehouse branch `satoshi/fs-phase23-write-route-20261006` (from the deployed 2d4550d):
capabilities `write_fs_phase23_rows`, `read_fs_phase23_rows`, `reconcile_fs_phase23`; routes
`POST /api/v2/fs-phase23/rows` (201 inserted / 200 pure replay / 400 typed refusal / 422 no
capability / 401 no token), `GET /api/v2/fs-phase23/rows?run_id&table&unit_id&after&limit`
(paged by row_key, rows exactly as submitted), `POST /api/v2/fs-phase23/reconcile`.
predictor backends 0.1.5 / 0.1.4 implement the three methods over the same
`fs_phase23_store` functions the local tool uses, under the DuckDB provider's write lock in
one transaction.

## Snapshot boundary

`tools/olap_duckdb_migrate.py` now lists the ten relations in `SNAPSHOT_RELATIONS`; a copy
lacking them is an `UNVERIFIED_COPY` with the reason named (test added).
