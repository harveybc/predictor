#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file etl_migrate_v2.py
@brief ETL loader for experiment results into PostgreSQL (OLAP cube for Metabase).

This script:
  - Connects to PostgreSQL using PG* environment variables (PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE).
  - Ensures dimension and fact tables exist in the `public` schema.
  - Upserts project/phase/experiment dimensions (with config fields extracted).
  - Loads a results summary CSV (strict contract).
  - Logs every warning and error with context; exits non-zero on any error.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict

import pandas as pd
from sqlalchemy import create_engine, text

# ==== Constants ====
SCHEMA = "public"

# ---- Logging setup ----
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s"
)

# ==== Utility: Build SQLAlchemy engine from PG* env ====
def build_engine_from_pg_env():
    """
    Construct a SQLAlchemy engine from PG* environment variables
    with sane defaults for local dev.
    """
    host = os.getenv("PGHOST", "127.0.0.1")
    port = int(os.getenv("PGPORT", "5432"))
    dbname = os.getenv("PGDATABASE", "predictor_olap")
    user = os.getenv("PGUSER", "metabase")
    password = os.getenv("PGPASSWORD", "metabase_pass")

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    safe_dsn = f"postgresql://{user}:*****@{host}:{port}/{dbname}"

    engine = create_engine(dsn, pool_pre_ping=True, future=True)
    logging.info("Connected to PostgreSQL via DSN: %s", safe_dsn)
    return engine

# ==== Idempotent DDL ====
def ensure_schema_and_tables(engine):
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
      use_multi_tapper   BOOLEAN
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
    with engine.begin() as conn:
        conn.exec_driver_sql(ddl)

# ==== Upsert helpers ====
def upsert_project(engine, project_key: str):
    sql = f"""
    INSERT INTO {SCHEMA}.dim_project (project_key)
    VALUES (:k)
    ON CONFLICT (project_key) DO NOTHING;
    """
    with engine.begin() as conn:
        conn.execute(text(sql), {"k": project_key})
    logging.info("Upserted project: %s", project_key)

def upsert_phase(engine, project_key: str, phase_key: str):
    sql = f"""
    INSERT INTO {SCHEMA}.dim_phase (phase_key, project_key)
    VALUES (:p, :proj)
    ON CONFLICT (phase_key) DO UPDATE SET project_key = EXCLUDED.project_key;
    """
    with engine.begin() as conn:
        conn.execute(text(sql), {"p": phase_key, "proj": project_key})
    logging.info("Upserted phase: %s (project=%s)", phase_key, project_key)

def upsert_experiment(engine, project_key: str, phase_key: str,
                      experiment_key: str, config_json: Dict):
    # Map config fields (only sample subset shown here)
    extracted = {
        "max_steps_train": config_json.get("max_steps_train"),
        "max_steps_test": config_json.get("max_steps_test"),
        "intermediate_layers": config_json.get("intermediate_layers"),
        "initial_layer_size": config_json.get("initial_layer_size"),
        "layer_size_divisor": config_json.get("layer_size_divisor"),
        "learning_rate": config_json.get("learning_rate"),
        "activation": config_json.get("activation"),
        "l2_reg": config_json.get("l2_reg"),
        "kl_weight": config_json.get("kl_weight"),
        "kl_anneal_epochs": config_json.get("kl_anneal_epochs"),
        "early_patience": config_json.get("early_patience"),
        "start_from_epoch": config_json.get("start_from_epoch"),
        "use_returns": config_json.get("use_returns"),
        "mmd_lambda": config_json.get("mmd_lambda"),
        "window_size": config_json.get("window_size"),
        "batch_size": config_json.get("batch_size"),
        "min_delta": config_json.get("min_delta"),
        "epochs": config_json.get("epochs"),
        "stl_period": config_json.get("stl_period"),
        "predicted_horizons": json.dumps(config_json.get("predicted_horizons")),
        "use_stl": config_json.get("use_stl"),
        "use_wavelets": config_json.get("use_wavelets"),
        "use_multi_tapper": config_json.get("use_multi_tapper"),
        "predictor_plugin": config_json.get("predictor_plugin"),
        "optimizer_plugin": config_json.get("optimizer_plugin"),
        "pipeline_plugin": config_json.get("pipeline_plugin"),
        "preprocessor_plugin": config_json.get("preprocessor_plugin"),
        "use_strategy": config_json.get("use_strategy"),
        "use_daily": config_json.get("use_daily"),
        "mc_samples": config_json.get("mc_samples"),
    }

    sql = f"""
    INSERT INTO {SCHEMA}.dim_experiment
      (experiment_key, project_key, phase_key, config_json,
       max_steps_train, max_steps_test, intermediate_layers, initial_layer_size,
       layer_size_divisor, learning_rate, activation, l2_reg, kl_weight,
       kl_anneal_epochs, early_patience, start_from_epoch, use_returns,
       mmd_lambda, window_size, batch_size, min_delta, epochs, stl_period,
       predicted_horizons, use_stl, use_wavelets, use_multi_tapper)
    VALUES
      (:e, :proj, :ph, CAST(:cfg AS JSONB),
       :max_steps_train, :max_steps_test, :intermediate_layers, :initial_layer_size,
       :layer_size_divisor, :learning_rate, :activation, :l2_reg, :kl_weight,
       :kl_anneal_epochs, :early_patience, :start_from_epoch, :use_returns,
       :mmd_lambda, :window_size, :batch_size, :min_delta, :epochs, :stl_period,
       CAST(:predicted_horizons AS JSONB), :use_stl, :use_wavelets, :use_multi_tapper)
    ON CONFLICT (experiment_key) DO UPDATE
      SET project_key = EXCLUDED.project_key,
          phase_key = EXCLUDED.phase_key,
          config_json = EXCLUDED.config_json,
          max_steps_train = EXCLUDED.max_steps_train,
          max_steps_test = EXCLUDED.max_steps_test,
          intermediate_layers = EXCLUDED.intermediate_layers,
          initial_layer_size = EXCLUDED.initial_layer_size,
          layer_size_divisor = EXCLUDED.layer_size_divisor,
          learning_rate = EXCLUDED.learning_rate,
          activation = EXCLUDED.activation,
          l2_reg = EXCLUDED.l2_reg,
          kl_weight = EXCLUDED.kl_weight,
          kl_anneal_epochs = EXCLUDED.kl_anneal_epochs,
          early_patience = EXCLUDED.early_patience,
          start_from_epoch = EXCLUDED.start_from_epoch,
          use_returns = EXCLUDED.use_returns,
          mmd_lambda = EXCLUDED.mmd_lambda,
          window_size = EXCLUDED.window_size,
          batch_size = EXCLUDED.batch_size,
          min_delta = EXCLUDED.min_delta,
          epochs = EXCLUDED.epochs,
          stl_period = EXCLUDED.stl_period,
          predicted_horizons = EXCLUDED.predicted_horizons,
          use_stl = EXCLUDED.use_stl,
          use_wavelets = EXCLUDED.use_wavelets,
          use_multi_tapper = EXCLUDED.use_multi_tapper;
    """

    params = {"e": experiment_key, "proj": project_key, "ph": phase_key,
              "cfg": json.dumps(config_json)}
    params.update(extracted)

    with engine.begin() as conn:
        conn.execute(text(sql), params)

    logging.info("Upserted experiment: %s", experiment_key)

# ==== Load results summary CSV ====
def load_results_summary(engine, experiment_key: str, results_csv: str) -> int:
    try:
        df = pd.read_csv(results_csv)
        # Normalize headers: strip spaces, replace with underscores
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    except Exception as exc:
        logging.error("Failed to read results CSV '%s': %s", results_csv, exc, exc_info=True)
        raise

    required = ["Metric", "Average", "Std_Dev", "Min", "Max"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logging.error("Results CSV missing required columns: %s (file=%s)", missing, results_csv)
        raise ValueError(f"Missing columns: {missing}")

    for col in ["Average", "Std_Dev", "Min", "Max"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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

    rows = 0
    with engine.begin() as conn:
        for rec in df.itertuples(index=False):
            params = {
                "experiment_key": experiment_key,
                "metric": rec.Metric,
                "avg":   rec.Average,
                "std":   rec.Std_Dev,
                "minv":  rec.Min,
                "maxv":  rec.Max,
            }
            conn.execute(text(sql), params)
            rows += 1

    logging.info("Loaded results summary rows: %d (experiment=%s)", rows, experiment_key)
    return rows


# ==== Main ====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-key",      required=True)
    ap.add_argument("--phase-key",        required=True)
    ap.add_argument("--experiment-key",   required=True)
    ap.add_argument("--experiment-config",required=True)
    ap.add_argument("--results-csv",      required=True)
    ap.add_argument("--predictions-csv",  required=False)
    ap.add_argument("--uncertainties-csv",required=False)
    args = ap.parse_args()

    engine = build_engine_from_pg_env()
    ensure_schema_and_tables(engine)

    try:
        with open(args.experiment_config, "r") as fh:
            cfg = json.load(fh)
    except Exception as exc:
        logging.error("Failed to read config: %s", exc, exc_info=True)
        sys.exit(3)

    upsert_project(engine, args.project_key)
    upsert_phase(engine, args.project_key, args.phase_key)
    upsert_experiment(engine, args.project_key, args.phase_key, args.experiment_key, cfg)

    try:
        _ = load_results_summary(engine, args.experiment_key, args.results_csv)
    except Exception as exc:
        logging.error("Failed to load results: %s", exc, exc_info=True)
        sys.exit(5)

    if args.predictions_csv:
        logging.warning("Predictions CSV loader not implemented. Skipping: %s", args.predictions_csv)
    if args.uncertainties_csv:
        logging.warning("Uncertainties CSV loader not implemented. Skipping: %s", args.uncertainties_csv)

    logging.info("ETL completed successfully for experiment '%s'.", args.experiment_key)
    sys.exit(0)

if __name__ == "__main__":
    main()
