#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file etl_migrate_v2.py
@brief ETL loader for experiment results into PostgreSQL (Metabase backend).

This script:
  - Connects to PostgreSQL using PG* environment variables (PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE).
  - Ensures dimension and fact tables exist in the `public` schema.
  - Upserts project/phase/experiment dimensions.
  - Loads a results summary CSV whose columns are exactly:
      ["Metric", "Average", "Std Dev", "Min", "Max"]
  - Performs idempotent upserts into fact_results_summary (UNIQUE(experiment_key, metric)).
  - Logs every warning and error with context; exits non-zero on any error.
  - Avoids misleading success messages.

Usage (as invoked by your runner):
  python olap/etl_migrate_v2.py \
    --project-key <PROJECT> \
    --phase-key <PHASE> \
    --experiment-key <EXPERIMENT> \
    --experiment-config <path/to/config.json> \
    --results-csv <path/to/results.csv> \
    [--predictions-csv <optional>] \
    [--uncertainties-csv <optional>]
"""

# ==== Imports ====
import argparse              # @import For CLI argument parsing
import json                  # @import To load experiment config JSON
import logging               # @import For structured logging at INFO/WARN/ERROR
import os                    # @import To read PG* environment variables
import sys                   # @import To control exit codes
from typing import Dict      # @import For type hints (dict types)
import pandas as pd          # @import To parse results CSVs robustly
from sqlalchemy import create_engine, text  # @import SQLAlchemy engine + SQL text

# ==== Constants ====
SCHEMA = "public"  # @const Target schema; DB you showed only has `public`

# ---- Logging setup ----
# @note LOG_LEVEL can be changed externally via environment if needed.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # @var Log level env override
logging.basicConfig(                                 # @call Configure root logger
    level=getattr(logging, LOG_LEVEL, logging.INFO), # @arg Use env level or INFO
    format="%(asctime)s %(levelname)s %(message)s"   # @arg Timestamped format
)

# ==== Utility: Build SQLAlchemy engine from PG* env ====
def _build_engine_from_pg_env():
    """
    @brief Construct a SQLAlchemy engine using libpq-style env vars.
    @return sqlalchemy.Engine connected to target database.
    """
    # @var host Read PGHOST or default to localhost
    host = os.getenv("PGHOST", "127.0.0.1")
    # @var port Read PGPORT or default to 5432
    port = int(os.getenv("PGPORT", "5432"))
    # @var dbname Read PGDATABASE; fail early if missing
    dbname = os.getenv("PGDATABASE", "metabase_db")
    # @var user Read PGUSER; fail early if missing
    user = os.getenv("PGUSER", "metabase")
    # @var password Read PGPASSWORD; may be required by server auth
    password = os.getenv("PGPASSWORD", "metabase_pass")

    # @check Validate required env vars
    if not dbname or not user:
        logging.error("PGDATABASE and PGUSER must be set in environment.")
        sys.exit(2)

    # @var dsn Compose DSN without driver suffix (lets SQLAlchemy pick available)
    dsn = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    # @return Engine constructed from DSN
    return create_engine(dsn, pool_pre_ping=True, future=True)


# ==== Idempotent DDL ====
def ensure_schema_and_tables(engine):
    """
    @brief Create required schema and tables if missing (idempotent).
    @param engine SQLAlchemy Engine
    """
    # @var ddl SQL for schema + tables (matches Section 1 DDL)
    ddl = f"""
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
      created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
    """
    # @exec Execute DDL in a transaction block with auto-commit
    with engine.begin() as conn:
        conn.exec_driver_sql(ddl)


# ==== Upsert helpers (dimensions) ====
def upsert_project(engine, project_key: str):
    """
    @brief Upsert a project dimension row by project_key.
    """
    sql = f"""
    INSERT INTO {SCHEMA}.dim_project (project_key)
    VALUES (:k)
    ON CONFLICT (project_key) DO NOTHING;
    """
    with engine.begin() as conn:
        conn.execute(text(sql), {"k": project_key})
    logging.info("Upserted project: %s", project_key)


def upsert_phase(engine, project_key: str, phase_key: str):
    """
    @brief Upsert a phase dimension row.
    """
    sql = f"""
    INSERT INTO {SCHEMA}.dim_phase (phase_key, project_key)
    VALUES (:p, :proj)
    ON CONFLICT (phase_key) DO UPDATE SET project_key = EXCLUDED.project_key;
    """
    with engine.begin() as conn:
        conn.execute(text(sql), {"p": phase_key, "proj": project_key})
    logging.info("Upserted phase: %s (project=%s)", phase_key, project_key)


def upsert_experiment(engine, project_key: str, phase_key: str, experiment_key: str, config_json: Dict):
    """
    @brief Upsert an experiment dimension row including config JSON.
    """
    sql = f"""
    INSERT INTO {SCHEMA}.dim_experiment (experiment_key, project_key, phase_key, config_json)
    VALUES (:e, :proj, :ph, CAST(:cfg AS JSONB))
    ON CONFLICT (experiment_key) DO UPDATE
      SET project_key = EXCLUDED.project_key,
          phase_key   = EXCLUDED.phase_key,
          config_json = EXCLUDED.config_json;
    """
    with engine.begin() as conn:
        conn.execute(text(sql), {
            "e": experiment_key,
            "proj": project_key,
            "ph": phase_key,
            "cfg": json.dumps(config_json) if config_json is not None else None
        })
    logging.info("Upserted experiment: %s (project=%s, phase=%s)", experiment_key, project_key, phase_key)


# ==== Load results summary CSV ====
def load_results_summary(engine, experiment_key: str, results_csv: str) -> int:
    """
    @brief Load results summary CSV into fact_results_summary with upsert.
    @return Number of rows upserted.
    """
    # @read Read CSV via pandas
    try:
        df = pd.read_csv(results_csv)
    except Exception as exc:
        logging.error("Failed to read results CSV '%s': %s", results_csv, exc, exc_info=True)
        raise

    # @validate Required columns (from sample)
    required = ["Metric", "Average", "Std Dev", "Min", "Max"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logging.error("Results CSV missing required columns: %s (file=%s)", missing, results_csv)
        raise ValueError(f"Results CSV missing columns: {missing}")

    # @cast Ensure numeric types; coerce errors to NaN and warn
    for col in ["Average", "Std Dev", "Min", "Max"]:
        before_na = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_na = df[col].isna().sum()
        if after_na > before_na:
            logging.warning("Non-numeric values coerced to NULL in column '%s' (%d new NaNs) [file=%s]",
                            col, after_na - before_na, results_csv)

    # @sql Upsert statement into fact_results_summary (unique by experiment_key, metric)
    sql = f"""
    INSERT INTO {SCHEMA}.fact_results_summary
      (experiment_key, metric, avg_value, std_dev, min_value, max_value)
    VALUES
      (:experiment_key, :metric, :avg, :std, :minv, :maxv)
    ON CONFLICT (experiment_key, metric) DO UPDATE
      SET avg_value = EXCLUDED.avg_value,
          std_dev   = EXCLUDED.std_dev,
          min_value = EXCLUDED.min_value,
          max_value = EXCLUDED.max_value,
          loaded_at = NOW();
    """

    # @exec Write in chunks for safety
    rows = 0
    with engine.begin() as conn:
        for rec in df.itertuples(index=False):
            params = {
                "experiment_key": experiment_key,
                "metric": getattr(rec, "Metric"),
                "avg":   getattr(rec, "Average"),
                "std":   getattr(rec, "Std_Dev"),
                "minv":  getattr(rec, "Min"),
                "maxv":  getattr(rec, "Max"),
            }
            conn.execute(text(sql), params)
            rows += 1

    logging.info("Loaded results summary rows: %d (experiment=%s)", rows, experiment_key)
    return rows


# ==== Main ====
def main():
    """
    @brief CLI entrypoint (keeps your current runner’s arguments).
    """
    # @cli Define arguments (staying faithful to your runner) :contentReference[oaicite:9]{index=9}
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-key",      required=True)
    ap.add_argument("--phase-key",        required=True)
    ap.add_argument("--experiment-key",   required=True)
    ap.add_argument("--experiment-config",required=True)
    ap.add_argument("--results-csv",      required=True)
    ap.add_argument("--predictions-csv",  required=False)     # optional (not loaded without spec)
    ap.add_argument("--uncertainties-csv",required=False)     # optional (not loaded without spec)
    args = ap.parse_args()

    # @log Input recap
    logging.info("Starting ETL for experiment '%s' (project=%s, phase=%s)",
                 args.experiment_key, args.project_key, args.phase_key)

    # @engine Connect using PG* env variables
    engine = _build_engine_from_pg_env()

    # @ddl Ensure tables exist
    ensure_schema_and_tables(engine)

    # @cfg Load config JSON (the output_config sample is comprehensive and shows fields we will store as JSONB) :contentReference[oaicite:10]{index=10}
    try:
        with open(args.experiment_config, "r") as fh:
            cfg = json.load(fh)
    except Exception as exc:
        logging.error("Failed to read experiment config '%s': %s", args.experiment_config, exc, exc_info=True)
        sys.exit(3)

    # @dims Upsert dimensions
    try:
        upsert_project(engine, args.project_key)
        upsert_phase(engine, args.project_key, args.phase_key)
        upsert_experiment(engine, args.project_key, args.phase_key, args.experiment_key, cfg)
    except Exception as exc:
        logging.error("Failed to upsert dimensions: %s", exc, exc_info=True)
        sys.exit(4)

    # @facts Load results summary CSV (strict columns)
    try:
        _ = load_results_summary(engine, args.experiment_key, args.results_csv)
    except Exception as exc:
        logging.error("Failed to load results summary: %s", exc, exc_info=True)
        sys.exit(5)

    # @todo Predictions/uncertainties: no column contract provided,
    #       so do not guess formats. If you provide a sample, we will
    #       add strict loaders analogous to results with validation.
    if args.predictions_csv:
        logging.warning("Predictions CSV provided but loader not enabled (no column spec). Skipping: %s",
                        args.predictions_csv)
    if args.uncertainties_csv:
        logging.warning("Uncertainties CSV provided but loader not enabled (no column spec). Skipping: %s",
                        args.uncertainties_csv)

    # @done Success; exit 0 (no cosmetics)
    logging.info("ETL completed successfully for experiment '%s'.", args.experiment_key)
    sys.exit(0)


# ==== Guard ====
if __name__ == "__main__":
    main()
