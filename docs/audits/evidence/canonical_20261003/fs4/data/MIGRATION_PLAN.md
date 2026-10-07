# FS4 migration plan (additive; the code is the single DDL source)

`predictor_olap_store.fs4_store.ddl()` renders five statements, every one
`CREATE TABLE IF NOT EXISTS` or `CREATE OR REPLACE VIEW` (`CREATE VIEW IF NOT EXISTS` on
SQLite). The checked-in file `olap/migrations/fs4/0001_fs4_extractibility_additive.sql`
(SHA-256 `bad9467d34f8311307e956aaae0274ce7cd3db39075da0640fe57267a09cf602`) is generated from it and the test
`test_migration_is_additive_idempotent_and_the_sql_file_equals_the_code` fails when they differ.
`tools/fs4_warehouse.py migrate --dry-run` on an empty file: `MIGRATION_DRYRUN_empty_file.json`
(would_create = the five relations, destructive_statements = 0).

| Relation | Grain | Keys / constraints |
|---|---|---|
| `feature_extractibility_v1` | one controller task (population x identity x feature x fold x arm, seed 0) | PK `task_id` = sha256(canonical task payload), recomputed on submit; UNIQUE (plan_sha256, population_id, identity, feature_id, fold_id, arm, seed); CHECK arm in {RAW, RANDOM_ENCODER, TRAINED_ENCODER}, seed = 0, population_n >= 1, mae / naive_mae / mse finite and >= 0, cost fields >= 0, host_role in {coordinator, worker_a, worker_b} |
| `fs4_load_receipt` | one submission | PK receipt_sha256; plan_sha256, terminal_count, inserted, duplicates_ignored, terminals_sha256, host_role, submitted_at |
| `fs4_load_receipt_task` | receipt x task | PK (receipt_sha256, task_id); terminal_sha256 |
| `fs4_extractibility_coverage` (view) | plan x population x arm | terminals, features, folds, min/max mae |
| `fs4_extractibility_triples` (view) | plan x population x identity x feature x fold | arms, distinct rows/mask/input digests, population_n, naive_mae (FS4-09 readout) |

Columns of `feature_extractibility_v1`: task_id, plan_sha256, population_id, identity,
feature_id, fold_id, arm, seed, rows_sha256, mask_sha256, input_sha256, code_sha256,
model_sha256, population_n, mae, naive_mae, mse, chosen_epoch, updates, cpu_s, wall_s,
peak_ram_bytes, peak_vram_bytes, host_role, owner, attempt, started_at, finished_at,
terminal_sha256, task_json, result_json, stored_at.

Source of each value in the terminal document (`EXAMPLE_TERMINAL_DOCUMENT.json`): identity
columns from `task`; digests, population_n and metrics from `result` (the controller's
`_validate_result` fields); `mse` from `result.metrics.mse` when present; `chosen_epoch` /
`updates` from `result.training.{chosen_epoch|best_epoch, updates|update_count}`; cost from
`result.cost.{cpu_s, wall_s, peak_ram_bytes|peak_ram, peak_vram_bytes|peak_vram}`; owner,
attempt, started_at, finished_at from the controller row. Everything nullable is nullable; the
metrics are not. `result_json` keeps the result byte-for-byte canonical, so a readback returns
exactly what the controller stored.
