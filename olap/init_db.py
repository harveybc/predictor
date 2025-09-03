#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file init_db.py
@brief Initialize PostgreSQL schema and tables required for OLAP cube.
       Safe to run multiple times (idempotent).
"""

import os
import sys
import logging
from sqlalchemy import create_engine

SCHEMA = "public"

DDL = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA};

CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_project (
  project_key TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_phase (
  phase_key   TEXT PRIMARY KEY,
  project_key TEXT NOT NULL REFERENCES {SCHEMA}.dim_project(project_key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_experiment (
  experiment_key TEXT PRIMARY KEY,
  project_key    TEXT NOT NULL REFERENCES {SCHEMA}.dim_project(project_key) ON DELETE CASCADE,
  phase_key      TEXT NOT NULL REFERENCES {SCHEMA}.dim_phase(phase_key)   ON DELETE CASCADE,
  config_json    JSONB,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  -- Optional extracted columns for easy filtering in Metabase:
  model          TEXT,
  learning_rate  DOUBLE PRECISION,
  optimizer      TEXT,
  batch_size     INTEGER,
  epochs         INTEGER
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.fact_results_summary (
  id             BIGSERIAL PRIMARY KEY,
  experiment_key TEXT NOT NULL REFERENCES {SCHEMA}.dim_experiment(experiment_key) ON DELETE CASCADE,
  metric         TEXT NOT NULL,
  avg_value      DOUBLE PRECISION,
  std_dev        DOUBLE PRECISION,
  min_value      DOUBLE PRECISION,
  max_value      DOUBLE PRECISION,
  loaded_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (experiment_key, metric)
);

CREATE INDEX IF NOT EXISTS idx_fact_results_summary_experiment
  ON {SCHEMA}.fact_results_summary (experiment_key);

CREATE INDEX IF NOT EXISTS idx_dim_experiment_cfg_gin
  ON {SCHEMA}.dim_experiment USING gin (config_json);
"""

def build_engine():
    """Construct SQLAlchemy engine from PG* env vars."""
    host = os.getenv("PGHOST", "127.0.0.1")
    port = int(os.getenv("PGPORT", "5432"))
    dbname = os.getenv("PGDATABASE", "metabase_db")
    user = os.getenv("PGUSER", "metabase")
    password = os.getenv("PGPASSWORD", "metabase_pass")

    if not dbname or not user:
        sys.stderr.write("ERROR: PGDATABASE and PGUSER must be set in env.\n")
        sys.exit(2)

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(dsn, pool_pre_ping=True, future=True)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = build_engine()
    logging.info("Creating schema and tables in database '%s'...", os.getenv("PGDATABASE"))
    with engine.begin() as conn:
        conn.exec_driver_sql(DDL)
    logging.info("Database initialization complete.")

if __name__ == "__main__":
    main()
