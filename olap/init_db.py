#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file init_db.py
@brief Initialize PostgreSQL schema and tables required for OLAP cube.
       Safe to run multiple times (idempotent).
"""

import os
import logging
from sqlalchemy import create_engine

SCHEMA = "public"

DDL = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA};

-- Project dimension
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_project (
  project_key TEXT PRIMARY KEY
);

-- Phase dimension
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_phase (
  phase_key   TEXT PRIMARY KEY,
  project_key TEXT NOT NULL REFERENCES {SCHEMA}.dim_project(project_key) ON DELETE CASCADE
);

-- Experiment dimension (config & extracted params)
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_experiment (
  experiment_key TEXT PRIMARY KEY,
  project_key    TEXT NOT NULL REFERENCES {SCHEMA}.dim_project(project_key) ON DELETE CASCADE,
  phase_key      TEXT NOT NULL REFERENCES {SCHEMA}.dim_phase(phase_key)   ON DELETE CASCADE,
  config_json    JSONB,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Extracted fields for filtering
  max_steps_train    INTEGER,
  max_steps_test     INTEGER,
  intermediate_layers INTEGER,
  initial_layer_size INTEGER,
  layer_size_divisor INTEGER,
  learning_rate      DOUBLE PRECISION,
  activation         TEXT,
  l2_reg             DOUBLE PRECISION,
  kl_weight          DOUBLE PRECISION,
  kl_anneal_epochs   INTEGER,
  early_patience     INTEGER,
  start_from_epoch   INTEGER,
  use_returns        BOOLEAN,
  mmd_lambda         DOUBLE PRECISION,
  window_size        INTEGER,
  batch_size         INTEGER,
  min_delta          DOUBLE PRECISION,
  epochs             INTEGER,
  stl_period         INTEGER,
  predicted_horizons JSONB,
  use_stl            BOOLEAN,
  use_wavelets       BOOLEAN,
  use_multi_tapper   BOOLEAN,
  predictor_plugin   TEXT,
  optimizer_plugin   TEXT,
  pipeline_plugin    TEXT,
  preprocessor_plugin TEXT,
  use_strategy       BOOLEAN,
  use_daily          BOOLEAN,
  mc_samples         INTEGER
);

-- Dataset split dimension
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_dataset_split (
  split_key TEXT PRIMARY KEY,
  description TEXT
);

-- Horizon dimension
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_horizon (
  horizon_key INTEGER PRIMARY KEY,
  description TEXT
);

-- Metric dimension
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_metric (
  metric_key TEXT PRIMARY KEY,
  metric_type TEXT,
  direction   TEXT  -- "lower_is_better" or "higher_is_better"
);

-- Fact table: experiment × split × horizon × metric
CREATE TABLE IF NOT EXISTS {SCHEMA}.fact_performance (
  id             BIGSERIAL PRIMARY KEY,
  experiment_key TEXT NOT NULL REFERENCES {SCHEMA}.dim_experiment(experiment_key) ON DELETE CASCADE,
  split_key      TEXT NOT NULL REFERENCES {SCHEMA}.dim_dataset_split(split_key),
  horizon_key    INTEGER NOT NULL REFERENCES {SCHEMA}.dim_horizon(horizon_key),
  metric_key     TEXT NOT NULL REFERENCES {SCHEMA}.dim_metric(metric_key),
  metric_value   DOUBLE PRECISION NOT NULL,
  loaded_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (experiment_key, split_key, horizon_key, metric_key)
);

-- Useful indexes
CREATE INDEX IF NOT EXISTS idx_fact_perf_experiment
  ON {SCHEMA}.fact_performance (experiment_key);

CREATE INDEX IF NOT EXISTS idx_fact_perf_metric
  ON {SCHEMA}.fact_performance (metric_key);

CREATE INDEX IF NOT EXISTS idx_dim_experiment_cfg_gin
  ON {SCHEMA}.dim_experiment USING gin (config_json);
"""

SEED = f"""
INSERT INTO {SCHEMA}.dim_dataset_split (split_key, description) VALUES
  ('train','Training set'), ('validation','Validation set'), ('test','Test set')
ON CONFLICT DO NOTHING;

INSERT INTO {SCHEMA}.dim_metric (metric_key, metric_type, direction) VALUES
  ('MAE','error','lower_is_better'),
  ('R2','fit','higher_is_better'),
  ('SNR','signal_to_noise','higher_is_better'),
  ('Uncertainty','uncertainty','lower_is_better'),
  ('Naive_MAE','baseline','lower_is_better')
ON CONFLICT DO NOTHING;
"""


def build_engine_from_pg_env():
    """
    Construct a SQLAlchemy engine from PG* environment variables
    with sane defaults for local dev.
    """
    host = os.getenv("PGHOST", "127.0.0.1")
    port = int(os.getenv("PGPORT", "5432"))
    dbname = os.getenv("PGDATABASE", "predictor_olap")   # default fallback
    user = os.getenv("PGUSER", "metabase")
    password = os.getenv("PGPASSWORD", "metabase_pass")

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(dsn, pool_pre_ping=True, future=True)

    logging.info("Connected to PostgreSQL at %s:%s, database=%s, user=%s",
                 host, port, dbname, user)
    return engine


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = build_engine_from_pg_env()
    logging.info("Creating schema and tables in database (see DSN above)...")
    with engine.begin() as conn:
        conn.exec_driver_sql(DDL)
        conn.exec_driver_sql(SEED)

    logging.info("Database initialization complete.")


if __name__ == "__main__":
    main()
