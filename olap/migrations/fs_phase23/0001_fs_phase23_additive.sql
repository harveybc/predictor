-- fs_phase23_0001: additive schema for feature-selection phases 2 and 3.
-- Deployed target: the DuckDB warehouse served by the data-warehouse host (schema main).
-- Every statement is CREATE ... IF NOT EXISTS. Nothing is dropped, altered or rewritten.
-- Parsed by tools/fs_phase23_warehouse.py (statements split on ';', '--' lines ignored).
-- Identity columns per table are the UNIQUE key of subplan section 2; row_identity_sha256 is
-- the SHA-256 of exactly those columns and is the PRIMARY KEY; row_sha256 digests the content.
-- stored_at / submitted_at never enter a digest.

CREATE TABLE IF NOT EXISTS fs_phase23_run (
  run_id TEXT PRIMARY KEY,
  run_sha256 TEXT NOT NULL UNIQUE,
  population_id TEXT NOT NULL,
  phase TEXT NOT NULL CHECK (phase IN ('PHASE_2', 'PHASE_3')),
  contract_sha256 TEXT NOT NULL,
  campaign_sha256 TEXT NOT NULL,
  code_sha256 TEXT NOT NULL,
  input_sha256 TEXT NOT NULL,
  expected_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fs_phase23_load_receipt (
  receipt_sha256 TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES fs_phase23_run(run_id),
  population_id TEXT NOT NULL,
  table_name TEXT NOT NULL,
  submitted INTEGER NOT NULL,
  distinct_identities INTEGER NOT NULL,
  stored_new INTEGER NOT NULL,
  already_stored INTEGER NOT NULL,
  rows_sha256 TEXT NOT NULL,
  host_role TEXT,
  shard_id TEXT,
  submitted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_pair_metrics (
  row_identity_sha256 TEXT PRIMARY KEY,
  row_sha256 TEXT NOT NULL,
  run_id TEXT NOT NULL REFERENCES fs_phase23_run(run_id),
  population_id TEXT NOT NULL,
  feature_left TEXT NOT NULL,
  feature_right TEXT NOT NULL,
  split TEXT NOT NULL DEFAULT 'train' CHECK (split = 'train'),
  fold TEXT NOT NULL,
  lag INTEGER NOT NULL CHECK (lag >= 0),
  method TEXT NOT NULL,
  params_sha256 TEXT NOT NULL,
  code_sha256 TEXT NOT NULL,
  input_sha256 TEXT NOT NULL,
  shared_support INTEGER NOT NULL CHECK (shared_support >= 0),
  metric_value DOUBLE,
  state TEXT NOT NULL CHECK (state IN ('MEASURED', 'INSUFFICIENT_SUPPORT', 'NOT_APPLICABLE', 'FAILED')),
  host_role TEXT,
  shard_id TEXT,
  terminal_sha256 TEXT,
  extra_json TEXT,
  stored_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK (feature_left < feature_right),
  UNIQUE (run_id, population_id, feature_left, feature_right, fold, lag, method, params_sha256, code_sha256, input_sha256)
);

CREATE TABLE IF NOT EXISTS feature_alias_groups (
  row_identity_sha256 TEXT PRIMARY KEY,
  row_sha256 TEXT NOT NULL,
  run_id TEXT NOT NULL REFERENCES fs_phase23_run(run_id),
  population_id TEXT NOT NULL,
  group_id TEXT NOT NULL,
  feature_id TEXT NOT NULL,
  split TEXT NOT NULL DEFAULT 'train' CHECK (split = 'train'),
  fold TEXT NOT NULL,
  method TEXT NOT NULL,
  params_sha256 TEXT NOT NULL,
  code_sha256 TEXT NOT NULL,
  input_sha256 TEXT NOT NULL,
  disposition TEXT NOT NULL CHECK (disposition IN ('ALIAS_BYTE_EXACT', 'ALIAS_NUMERIC_TOLERANCE', 'AFFINE_EXACT', 'MONOTONE_NEAR_PERFECT', 'DISTINCT')),
  representative BOOLEAN NOT NULL,
  shared_support INTEGER,
  host_role TEXT,
  shard_id TEXT,
  terminal_sha256 TEXT,
  extra_json TEXT,
  stored_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (run_id, population_id, group_id, feature_id, fold, method, params_sha256, code_sha256, input_sha256)
);

CREATE TABLE IF NOT EXISTS feature_redundancy_clusters (
  row_identity_sha256 TEXT PRIMARY KEY,
  row_sha256 TEXT NOT NULL,
  run_id TEXT NOT NULL REFERENCES fs_phase23_run(run_id),
  population_id TEXT NOT NULL,
  split TEXT NOT NULL DEFAULT 'train' CHECK (split = 'train'),
  fold TEXT NOT NULL,
  method TEXT NOT NULL,
  params_sha256 TEXT NOT NULL,
  code_sha256 TEXT NOT NULL,
  input_sha256 TEXT NOT NULL,
  cluster_id TEXT NOT NULL,
  feature_id TEXT NOT NULL,
  representative BOOLEAN NOT NULL,
  linkage_distance DOUBLE,
  threshold DOUBLE,
  host_role TEXT,
  shard_id TEXT,
  terminal_sha256 TEXT,
  extra_json TEXT,
  stored_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (run_id, population_id, fold, method, params_sha256, code_sha256, input_sha256, cluster_id, feature_id)
);

CREATE TABLE IF NOT EXISTS feature_filter_rankings (
  row_identity_sha256 TEXT PRIMARY KEY,
  row_sha256 TEXT NOT NULL,
  run_id TEXT NOT NULL REFERENCES fs_phase23_run(run_id),
  population_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  horizon INTEGER NOT NULL,
  split TEXT NOT NULL DEFAULT 'train' CHECK (split = 'train'),
  fold TEXT NOT NULL,
  method TEXT NOT NULL CHECK (method IN ('clustering_spearman', 'mrmr_mi', 'jmi', 'ALL_ADMISSIBLE', 'univariate_mi', 'CAUSAL_SUPPORTED', 'RANDOM_K')),
  params_sha256 TEXT NOT NULL,
  code_sha256 TEXT NOT NULL,
  input_sha256 TEXT NOT NULL,
  feature_id TEXT NOT NULL,
  rank INTEGER NOT NULL CHECK (rank >= 1),
  score DOUBLE,
  relevance_term DOUBLE,
  redundancy_term DOUBLE,
  complementarity_term DOUBLE,
  causal_term DOUBLE,
  cost_term DOUBLE,
  k_membership_json TEXT,
  state TEXT NOT NULL CHECK (state IN ('MEASURED', 'INSUFFICIENT_SUPPORT', 'NOT_APPLICABLE', 'FAILED')),
  host_role TEXT,
  shard_id TEXT,
  terminal_sha256 TEXT,
  extra_json TEXT,
  stored_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (run_id, population_id, target_id, horizon, fold, method, params_sha256, code_sha256, input_sha256, feature_id)
);

CREATE OR REPLACE VIEW fs_phase23_coverage AS
SELECT r.run_id, r.population_id, r.phase,
       (SELECT count(*) FROM feature_pair_metrics p WHERE p.run_id = r.run_id) AS pair_metric_rows,
       (SELECT count(*) FROM feature_alias_groups a WHERE a.run_id = r.run_id) AS alias_rows,
       (SELECT count(*) FROM feature_redundancy_clusters c WHERE c.run_id = r.run_id) AS cluster_rows,
       (SELECT count(*) FROM feature_filter_rankings f WHERE f.run_id = r.run_id) AS ranking_rows,
       (SELECT count(*) FROM fs_phase23_load_receipt l WHERE l.run_id = r.run_id) AS receipts
FROM fs_phase23_run r;
