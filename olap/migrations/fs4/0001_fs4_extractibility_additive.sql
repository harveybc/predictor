-- fs4_0001: additive schema for feature-selection phase 4 (extractibility).
-- GENERATED from predictor_olap_store.fs4_store.ddl(); do not edit by hand.
-- Every statement is CREATE ... IF NOT EXISTS. Nothing is dropped, altered or rewritten.
-- Identity: task_id (PRIMARY KEY) = sha256 of the controller's canonical task payload,
-- AND UNIQUE over (plan_sha256, population_id, identity, feature_id, fold_id, arm, seed).
-- CHECK: arm in {RAW, RANDOM_ENCODER, TRAINED_ENCODER}, seed = 0, population_n >= 1,
-- mae / naive_mae / mse finite and non-negative. terminal_sha256 digests the result as
-- the controller stored it; result_json holds that JSON; stored_at is never digested.

CREATE TABLE IF NOT EXISTS feature_extractibility_v1 ( task_id TEXT PRIMARY KEY, plan_sha256 TEXT NOT NULL, population_id TEXT NOT NULL, identity TEXT NOT NULL, feature_id TEXT NOT NULL, fold_id TEXT NOT NULL, arm TEXT NOT NULL CHECK (arm IN ('RAW', 'RANDOM_ENCODER', 'TRAINED_ENCODER')), seed INTEGER NOT NULL CHECK (seed = 0), rows_sha256 TEXT NOT NULL, mask_sha256 TEXT NOT NULL, input_sha256 TEXT NOT NULL, code_sha256 TEXT NOT NULL, model_sha256 TEXT NOT NULL, population_n BIGINT NOT NULL, mae DOUBLE NOT NULL, naive_mae DOUBLE NOT NULL, mse DOUBLE, chosen_epoch INTEGER, updates BIGINT, cpu_s DOUBLE, wall_s DOUBLE, peak_ram_bytes BIGINT, peak_vram_bytes BIGINT, host_role TEXT, owner TEXT, attempt INTEGER, started_at DOUBLE, finished_at DOUBLE, terminal_sha256 TEXT NOT NULL, task_json TEXT NOT NULL, result_json TEXT NOT NULL, stored_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, CHECK (population_n >= 1), CHECK (isfinite(mae) AND mae >= 0), CHECK (isfinite(naive_mae) AND naive_mae >= 0), CHECK (mse IS NULL OR (isfinite(mse) AND mse >= 0)), CHECK (chosen_epoch IS NULL OR chosen_epoch >= 0), CHECK (updates IS NULL OR updates >= 0), CHECK (cpu_s IS NULL OR cpu_s >= 0), CHECK (wall_s IS NULL OR wall_s >= 0), CHECK (peak_ram_bytes IS NULL OR peak_ram_bytes >= 0), CHECK (peak_vram_bytes IS NULL OR peak_vram_bytes >= 0), CHECK (host_role IS NULL OR host_role IN ('coordinator', 'worker_a', 'worker_b')), UNIQUE (plan_sha256, population_id, identity, feature_id, fold_id, arm, seed));

CREATE TABLE IF NOT EXISTS fs4_load_receipt ( receipt_sha256 TEXT PRIMARY KEY, plan_sha256 TEXT NOT NULL, terminal_count INTEGER NOT NULL, inserted INTEGER NOT NULL, duplicates_ignored INTEGER NOT NULL, terminals_sha256 TEXT NOT NULL, host_role TEXT, submitted_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS fs4_load_receipt_task ( receipt_sha256 TEXT NOT NULL, task_id TEXT NOT NULL, terminal_sha256 TEXT NOT NULL, PRIMARY KEY (receipt_sha256, task_id));

CREATE OR REPLACE VIEW fs4_extractibility_coverage AS SELECT plan_sha256, population_id, arm, count(*) AS terminals, count(DISTINCT feature_id) AS features, count(DISTINCT fold_id) AS folds, min(mae) AS min_mae, max(mae) AS max_mae FROM feature_extractibility_v1 GROUP BY plan_sha256, population_id, arm;

CREATE OR REPLACE VIEW fs4_extractibility_triples AS SELECT plan_sha256, population_id, identity, feature_id, fold_id, count(*) AS arms, count(DISTINCT rows_sha256) AS rows_digests, count(DISTINCT mask_sha256) AS mask_digests, count(DISTINCT input_sha256) AS input_digests, count(DISTINCT population_n) AS population_ns, count(DISTINCT naive_mae) AS naive_maes FROM feature_extractibility_v1 GROUP BY plan_sha256, population_id, identity, feature_id, fold_id;
