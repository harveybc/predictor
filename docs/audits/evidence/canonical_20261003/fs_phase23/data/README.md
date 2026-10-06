# fs_phase23/data: contract, warehouse schema, interface, readback, snapshot (data lane)

Order: `docs/handoffs/MUSASHI_TO_SATOSHI_FS_PHASE2_PHASE3_AUTOMATED_2026_10_05.md` B, D, F, G,
plus the Phase-B interface alignment with the engineering driver (branch
`satoshi/fs-phase23-driver-20261005`, `docs/FS_PHASE23_DRIVER.md`).

Code: `olap/store/src/predictor_olap_store/fs_phase23_store.py` (single implementation),
`tools/fs_phase23_warehouse.py` (driver entry point + CLI), `olap/migrations/fs_phase23/0001_fs_phase23_additive.sql`
(generated), `tests/test_fs_phase23_warehouse.py`, `tests/test_olap_duckdb_migrate.py` (boundary),
data-warehouse `tests/test_fs_phase23_routes.py`.

## Interface contract as implemented

```
wh = tools.fs_phase23_warehouse.open_warehouse(path_or_url)   # DuckDB file (migration applied) or http(s) URL
wh.submit_rows(run_id, table, rows, host_role=None, shard_id=None) -> receipt (v2, fields above)
wh.read_run(run_id, table, unit_id=None) -> [row as submitted], ordered by row_key
wh.reconcile(run_id, receipts=None, expected=None) -> {"run_id", "tables": {t: {"count", "rows_sha256", ...}}, ...}
wh.register_run({run_id, population_id, phase, contract_sha256, campaign_sha256, code_sha256, input_sha256, expected_json})
wh.verify_receipt(receipt); wh.query(sql)
```

Tables: `feature_pair_metrics`, `feature_pair_stability`, `feature_pair_gate`,
`feature_alias_groups`, `feature_redundancy_clusters`, `feature_filter_rankings`,
`feature_filter_subsets`. Every row needs `run_id`, `population_id`, `row_key` and its typed
identity fields (`IDENTITY` in `fs_phase23_store`: e.g. `left`, `right`, `fold_id`, `metric`,
`lag_hours`, `method`, `params_sha256`, `code_sha256` for metrics; `alias_group_id` /
`cluster_id`; `target_id`, `horizon_hours`, `method`, `params_sha256`, `feature_id` / `k`).
Refused before any write: foreign `run_id`, foreign `population_id`, unordered pair, a
validation/test fold, a host name instead of a role, an unknown state, one identity with
two contents, a `row_key` reused under another identity. Identical replay: `inserted 0`,
`duplicates_ignored n`, same `rows_sha256`. The service token is `WAREHOUSE_TOKEN` in the
environment only.

| file | what |
|---|---|
| `CONTRACT.json` | frozen populations from phase-1 artifacts: EURUSD 366/14/5 folds/66,795 pairs, ETH 83/6/3 folds/3,403 pairs; phase-1 digests; CAUSAL_SUPPORTED candidates (12 + 3); identity columns per table |
| `MIGRATION_PLAN.md`, `MIGRATION_DRYRUN_phase1_snapshot.json` | additive migration, dry-run evidence, write path |
| `LIVE_DEPLOY_STEPS.md` | the integration agent's install/restart/verify sequence (not executed) |
| `READBACK_REPORT_FORMAT.md`, `EXAMPLE_*.json` | receipt v2, reconcile v2, readback report (examples synthetic) |
| `SNAPSHOT_PROCEDURE.md` | `.duckdb.zst` + SHA-256 + release asset, owner-credential boundary |
| `TEST_RESULTS.txt` | suite results under crispdm-run |
