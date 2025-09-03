#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
etl_migrate_v2.py
Professional ETL migrator for experiment configs + per-horizon metrics into PostgreSQL OLAP schema.

Behavior summary:
 - Connects to Postgres via PG* env vars (PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE).
 - Ensures star-schema (dim_project, dim_phase, dim_experiment, dim_dataset_split,
   dim_horizon, dim_metric, fact_performance) exists and seeds lookup dims.
 - Upserts project/phase/experiment. Extracts and stores the canonical list of config fields
   (typed columns) and the full config as JSONB.
 - Loads results CSVs that contain per-split/per-metric/per-horizon columns in a
   wide format (e.g. 'train_MAE_h1', 'validation_R2_h3', 'test_SNR_h6') and writes
   them into fact_performance with idempotent upsert.
 - Logs every step with INFO/WARNING/ERROR and returns non-zero on fatal errors.

Usage (keeps compatibility with your runner):
  python olap/etl_migrate_v2.py \
    --project-key <PROJECT> \
    --phase-key <PHASE> \
    --experiment-key <EXPERIMENT> \
    --experiment-config <path/to/config.json> \
    --results-csv <path/to/results.csv>

"""

# Standard library imports
import argparse                # CLI argument parsing
import json                    # JSON serialization/deserialization
import logging                 # Structured logging
import os                      # Environment variable access
import re                      # Regex for column parsing
import sys                     # Exit codes and sys.stderr
from typing import Dict        # Type hints for dictionaries

# Third-party imports
import pandas as pd            # CSV parsing and dataframe operations
from sqlalchemy import create_engine, text  # DB connectivity and SQL execution

# Target schema name (change here if you later want 'olap' instead)
SCHEMA = "public"

# Logging setup (timestamped + level from env)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, "INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")

# Regex to match results column names like: split_metric_hX (e.g. train_MAE_h1)
# - group 1: split (train|validation|test or any string without underscore)
# - group 2: metric name (word chars or containing hyphen)
# - group 3: horizon number
_COL_REGEX = re.compile(r"^([^_]+)_([^_]+)_h(\d+)$", flags=re.IGNORECASE)


# === Utility: Build SQLAlchemy engine from PG* env ===
def _build_engine_from_pg_env():
    """
    Construct a SQLAlchemy engine from PG* environment variables.
    Exits the process with code 2 if required vars are missing.
    """
    host = os.getenv("PGHOST", "127.0.0.1")                      # default host
    port = int(os.getenv("PGPORT", "5432"))                      # default port
    dbname = os.getenv("PGDATABASE", "predictor_olap")                       # DB name must be set
    user = os.getenv("PGUSER", "metabase")                             # DB user must be set
    password = os.getenv("PGPASSWORD", "metabase_pass")                       # optional password

    if not dbname or not user:
        logging.error("PGDATABASE and PGUSER must be set in environment.")
        sys.exit(2)

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"  # DSN for SQLAlchemy
    # create_engine with pool_pre_ping to avoid stale connections in long runners
    return create_engine(dsn, pool_pre_ping=True, future=True)


# === Ensure schema and tables (idempotent DDL + seed) ===
def ensure_schema_and_tables(engine):
    """
    Create schema, dimensions, and fact table if missing. Also seed dataset splits and metrics.
    This function is idempotent and safe to run repeatedly.
    """
    # DDL: create dims and fact_performance with the chosen typed columns for dim_experiment.
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

      -- Extracted fields (typed) for filtering in Metabase
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

    CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_dataset_split (
      split_key TEXT PRIMARY KEY,
      description TEXT
    );

    CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_horizon (
      horizon_key INTEGER PRIMARY KEY,
      description TEXT
    );

    CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_metric (
      metric_key TEXT PRIMARY KEY,
      metric_type TEXT,
      direction   TEXT
    );

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

    CREATE INDEX IF NOT EXISTS idx_fact_perf_experiment
      ON {SCHEMA}.fact_performance (experiment_key);

    CREATE INDEX IF NOT EXISTS idx_fact_perf_metric
      ON {SCHEMA}.fact_performance (metric_key);

    CREATE INDEX IF NOT EXISTS idx_dim_experiment_cfg_gin
      ON {SCHEMA}.dim_experiment USING gin (config_json);
    """

    # Seed statements for dataset splits and canonical metrics
    seed = f"""
    INSERT INTO {SCHEMA}.dim_dataset_split (split_key, description) VALUES
      ('train','Training set'),
      ('validation','Validation set'),
      ('test','Test set')
    ON CONFLICT DO NOTHING;

    -- Seed commonly-used metrics (you can extend this list)
    INSERT INTO {SCHEMA}.dim_metric (metric_key, metric_type, direction) VALUES
      ('MAE','error','lower_is_better'),
      ('R2','fit','higher_is_better'),
      ('SNR','signal_to_noise','higher_is_better'),
      ('Uncertainty','uncertainty','lower_is_better'),
      ('Naive_MAE','baseline','lower_is_better')
    ON CONFLICT DO NOTHING;
    """

    # Execute DDL + seed in a single transaction
    with engine.begin() as conn:
        conn.exec_driver_sql(ddl)
        conn.exec_driver_sql(seed)

    # Log completion
    logging.info("Schema and tables are ensured and seed data inserted (if needed).")


# === Dimension upsert helpers ===
def upsert_project(engine, project_key: str):
    """
    Upsert the project dimension row. Does nothing if project exists.
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
    Upsert the phase dimension row (phase_key, project_key).
    If phase exists, update its project linkage.
    """
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
    """
    Upsert an experiment row into dim_experiment.

    - Stores full config JSON in config_json (JSONB).
    - Extracts the canonical set of config fields and writes them into typed columns,
      enabling click-and-drag filtering in Metabase without JSON queries.

    The canonical extracted fields are:
      max_steps_train, max_steps_test, intermediate_layers, initial_layer_size,
      layer_size_divisor, learning_rate, activation, l2_reg, kl_weight,
      kl_anneal_epochs, early_patience, start_from_epoch, use_returns, mmd_lambda,
      window_size, batch_size, min_delta, epochs, stl_period,
      predicted_horizons, use_stl, use_wavelets, use_multi_tapper
    """
    # Defensive: ensure config_json is a dict
    cfg = config_json or {}

    # Extract fields with default None if missing
    extracted = {
        "max_steps_train":     cfg.get("max_steps_train"),
        "max_steps_test":      cfg.get("max_steps_test"),
        "intermediate_layers": cfg.get("intermediate_layers"),
        "initial_layer_size":  cfg.get("initial_layer_size"),
        "layer_size_divisor":  cfg.get("layer_size_divisor"),
        "learning_rate":       cfg.get("learning_rate"),
        "activation":          cfg.get("activation"),
        "l2_reg":              cfg.get("l2_reg"),
        "kl_weight":           cfg.get("kl_weight"),
        "kl_anneal_epochs":    cfg.get("kl_anneal_epochs"),
        "early_patience":      cfg.get("early_patience"),
        "start_from_epoch":    cfg.get("start_from_epoch"),
        "use_returns":         cfg.get("use_returns"),
        "mmd_lambda":          cfg.get("mmd_lambda"),
        "window_size":         cfg.get("window_size"),
        "batch_size":          cfg.get("batch_size"),
        "min_delta":           cfg.get("min_delta"),
        "epochs":              cfg.get("epochs"),
        "stl_period":          cfg.get("stl_period"),
        "predicted_horizons":  cfg.get("predicted_horizons"),
        "use_stl":             cfg.get("use_stl"),
        "use_wavelets":        cfg.get("use_wavelets"),
        "use_multi_tapper":    cfg.get("use_multi_tapper"),
    }

    # SQL to insert/update the experiment dimension with all extracted fields
    sql = f"""
    INSERT INTO {SCHEMA}.dim_experiment (
      experiment_key, project_key, phase_key, config_json,
      max_steps_train, max_steps_test, intermediate_layers, initial_layer_size, layer_size_divisor,
      learning_rate, activation, l2_reg, kl_weight, kl_anneal_epochs, early_patience, start_from_epoch,
      use_returns, mmd_lambda, window_size, batch_size, min_delta, epochs, stl_period,
      predicted_horizons, use_stl, use_wavelets, use_multi_tapper
    )
    VALUES (
      :e, :proj, :ph, CAST(:cfg AS JSONB),
      :max_steps_train, :max_steps_test, :intermediate_layers, :initial_layer_size, :layer_size_divisor,
      :learning_rate, :activation, :l2_reg, :kl_weight, :kl_anneal_epochs, :early_patience, :start_from_epoch,
      :use_returns, :mmd_lambda, :window_size, :batch_size, :min_delta, :epochs, :stl_period,
      CAST(:predicted_horizons AS JSONB), :use_stl, :use_wavelets, :use_multi_tapper
    )
    ON CONFLICT (experiment_key) DO UPDATE SET
      project_key        = EXCLUDED.project_key,
      phase_key          = EXCLUDED.phase_key,
      config_json        = EXCLUDED.config_json,
      max_steps_train    = EXCLUDED.max_steps_train,
      max_steps_test     = EXCLUDED.max_steps_test,
      intermediate_layers = EXCLUDED.intermediate_layers,
      initial_layer_size = EXCLUDED.initial_layer_size,
      layer_size_divisor = EXCLUDED.layer_size_divisor,
      learning_rate      = EXCLUDED.learning_rate,
      activation         = EXCLUDED.activation,
      l2_reg             = EXCLUDED.l2_reg,
      kl_weight          = EXCLUDED.kl_weight,
      kl_anneal_epochs   = EXCLUDED.kl_anneal_epochs,
      early_patience     = EXCLUDED.early_patience,
      start_from_epoch   = EXCLUDED.start_from_epoch,
      use_returns        = EXCLUDED.use_returns,
      mmd_lambda         = EXCLUDED.mmd_lambda,
      window_size        = EXCLUDED.window_size,
      batch_size         = EXCLUDED.batch_size,
      min_delta          = EXCLUDED.min_delta,
      epochs             = EXCLUDED.epochs,
      stl_period         = EXCLUDED.stl_period,
      predicted_horizons = EXCLUDED.predicted_horizons,
      use_stl            = EXCLUDED.use_stl,
      use_wavelets       = EXCLUDED.use_wavelets,
      use_multi_tapper   = EXCLUDED.use_multi_tapper;
    """

    # Execute inside a transaction and log outcome
    with engine.begin() as conn:
        conn.execute(text(sql), {
            "e": experiment_key,
            "proj": project_key,
            "ph": phase_key,
            "cfg": json.dumps(cfg) if cfg is not None else None,
            # pass extracted fields (SQLAlchemy will convert Python types to DB)
            "max_steps_train": extracted["max_steps_train"],
            "max_steps_test": extracted["max_steps_test"],
            "intermediate_layers": extracted["intermediate_layers"],
            "initial_layer_size": extracted["initial_layer_size"],
            "layer_size_divisor": extracted["layer_size_divisor"],
            "learning_rate": extracted["learning_rate"],
            "activation": extracted["activation"],
            "l2_reg": extracted["l2_reg"],
            "kl_weight": extracted["kl_weight"],
            "kl_anneal_epochs": extracted["kl_anneal_epochs"],
            "early_patience": extracted["early_patience"],
            "start_from_epoch": extracted["start_from_epoch"],
            "use_returns": extracted["use_returns"],
            "mmd_lambda": extracted["mmd_lambda"],
            "window_size": extracted["window_size"],
            "batch_size": extracted["batch_size"],
            "min_delta": extracted["min_delta"],
            "epochs": extracted["epochs"],
            "stl_period": extracted["stl_period"],
            "predicted_horizons": json.dumps(extracted["predicted_horizons"]) if extracted["predicted_horizons"] is not None else None,
            "use_stl": extracted["use_stl"],
            "use_wavelets": extracted["use_wavelets"],
            "use_multi_tapper": extracted["use_multi_tapper"]
        })

    logging.info("Upserted experiment '%s' (extracted fields stored).", experiment_key)


# === Results CSV loader: pivot wide->long and upsert into fact_performance ===
def load_results_csv(engine, experiment_key: str, results_csv: str) -> int:
    """
    Load a results CSV and insert metrics into fact_performance.

    Expected CSV header pattern per metric column:
      <split>_<metric>_h<horizon>    e.g. "train_MAE_h1", "validation_R2_h3", "test_SNR_h6"

    The function:
      - reads the CSV (pandas)
      - scans columns and matches those following the pattern
      - for each matching column, iterates rows and upserts metric values into fact_performance
    Returns:
      number of metric values upserted
    """
    # Read file via pandas, with robust error handling
    try:
        df = pd.read_csv(results_csv)
    except Exception as exc:
        logging.error("Failed to read results CSV '%s': %s", results_csv, exc, exc_info=True)
        raise

    # If the CSV is empty, warn and return 0
    if df.shape[0] == 0:
        logging.warning("Results CSV '%s' is empty.", results_csv)
        return 0

    # Identify metric columns by regex
    metric_columns = []
    for col in df.columns:
        m = _COL_REGEX.match(col)
        if m:
            metric_columns.append((col, m.group(1), m.group(2), int(m.group(3))))
        else:
            logging.debug("Skipping non-metric column: %s", col)

    # If no metric columns found, fallback: try older summary format detection
    if not metric_columns:
        logging.error("No per-horizon metric columns found in '%s' (expected pattern split_metric_hN)", results_csv)
        raise ValueError("No per-horizon metric columns found in results CSV.")

    # Upsert each metric value. Use transaction per-file for atomicity.
    rows_upserted = 0
    with engine.begin() as conn:
        # Prepare SQL upsert for fact_performance
        upsert_sql = f"""
        INSERT INTO {SCHEMA}.fact_performance
          (experiment_key, split_key, horizon_key, metric_key, metric_value)
        VALUES (:e, :split, :h, :metric, :value)
        ON CONFLICT (experiment_key, split_key, horizon_key, metric_key)
        DO UPDATE SET metric_value = EXCLUDED.metric_value, loaded_at = NOW();
        """

        # Iterate over each detected metric column
        for col_name, split, metric, horizon in metric_columns:
            # convert column values to numeric; non-numeric will be NaN and skipped
            series = pd.to_numeric(df[col_name], errors="coerce")
            # iterate rows: there may be multiple rows (e.g., per-fold), insert each
            for val in series.dropna().tolist():
                try:
                    conn.execute(text(upsert_sql), {
                        "e": experiment_key,
                        "split": split,
                        "h": horizon,
                        "metric": metric,
                        "value": float(val)
                    })
                    rows_upserted += 1
                except Exception as exc:
                    # Log but continue inserting other values in the file
                    logging.error(
                        "Failed to upsert metric for experiment=%s split=%s metric=%s horizon=%s value=%s: %s",
                        experiment_key, split, metric, horizon, val, exc, exc_info=True
                    )

    logging.info("Loaded %d metric values from '%s' (experiment=%s).", rows_upserted, results_csv, experiment_key)
    return rows_upserted


# --- Backwards-compatible loader for old "results summary" CSVs (optional) ---
def load_results_summary(engine, experiment_key: str, results_csv: str) -> int:
    """
    Legacy loader for older summary CSVs that have columns: Metric, Average, Std Dev, Min, Max.
    This is kept for backwards compatibility but not promoted as the canonical loader.
    """
    try:
        df = pd.read_csv(results_csv)
    except Exception as exc:
        logging.error("Failed to read legacy results CSV '%s': %s", results_csv, exc, exc_info=True)
        raise

    required = ["Metric", "Average", "Std Dev", "Min", "Max"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logging.error("Legacy summary CSV missing required columns: %s (file=%s)", missing, results_csv)
        raise ValueError(f"Summary CSV missing columns: {missing}")

    # Upsert each summary metric into fact_results_summary (kept for compatibility)
    sql = f"""
    INSERT INTO {SCHEMA}.fact_results_summary
      (experiment_key, metric, avg_value, std_dev, min_value, max_value)
    VALUES (:experiment_key, :metric, :avg, :std, :minv, :maxv)
    ON CONFLICT (experiment_key, metric) DO UPDATE
      SET avg_value = EXCLUDED.avg_value, std_dev = EXCLUDED.std_dev,
          min_value = EXCLUDED.min_value, max_value = EXCLUDED.max_value, loaded_at = NOW();
    """

    rows = 0
    with engine.begin() as conn:
        for rec in df.itertuples(index=False):
            try:
                conn.execute(text(sql), {
                    "experiment_key": experiment_key,
                    "metric": getattr(rec, "Metric"),
                    "avg": float(getattr(rec, "Average")),
                    "std": float(getattr(rec, "Std_Dev")),
                    "minv": float(getattr(rec, "Min")),
                    "maxv": float(getattr(rec, "Max"))
                })
                rows += 1
            except Exception as exc:
                logging.error("Failed to upsert legacy summary row for experiment %s: %s", experiment_key, exc, exc_info=True)
    logging.info("Loaded %d legacy summary rows from '%s' (experiment=%s).", rows, results_csv, experiment_key)
    return rows


# === CLI / main ===
def main():
    """
    CLI entrypoint. Parses arguments, connects to DB, ensures schema, upserts dims, and loads results.
    Exits with non-zero codes on failure.
    """
    ap = argparse.ArgumentParser(description="ETL loader: experiment config + per-horizon metrics -> Postgres OLAP")
    ap.add_argument("--project-key", required=True, help="Project key string")
    ap.add_argument("--phase-key", required=True, help="Phase key string")
    ap.add_argument("--experiment-key", required=True, help="Experiment key string")
    ap.add_argument("--experiment-config", required=True, help="Path to experiment JSON config")
    ap.add_argument("--results-csv", required=True, help="Path to results CSV (per-horizon wide format)")
    args = ap.parse_args()

    logging.info("Starting ETL for experiment '%s' (project=%s, phase=%s)",
                 args.experiment_key, args.project_key, args.phase_key)

    # Build DB engine (will exit if PGDATABASE or PGUSER missing)
    engine = _build_engine_from_pg_env()

    # Ensure DB objects exist (idempotent)
    try:
        ensure_schema_and_tables(engine)
    except Exception as exc:
        logging.error("Failed to ensure schema/tables: %s", exc, exc_info=True)
        sys.exit(10)

    # Load experiment config JSON
    try:
        with open(args.experiment_config, "r") as fh:
            cfg = json.load(fh)
    except Exception as exc:
        logging.error("Failed to read experiment config '%s': %s", args.experiment_config, exc, exc_info=True)
        sys.exit(11)

    # Upsert dims (project, phase, experiment)
    try:
        upsert_project(engine, args.project_key)
        upsert_phase(engine, args.project_key, args.phase_key)
        upsert_experiment(engine, args.project_key, args.phase_key, args.experiment_key, cfg)
    except Exception as exc:
        logging.error("Failed to upsert dimensions: %s", exc, exc_info=True)
        sys.exit(12)

    # Load per-horizon results into fact_performance (preferred canonical loader)
    try:
        n = load_results_csv(engine, args.experiment_key, args.results_csv)
        if n == 0:
            logging.warning("No metric values were loaded from '%s' (experiment=%s).", args.results_csv, args.experiment_key)
    except ValueError as exc:
        # explicit pattern mismatch - try legacy summary loader as fallback
        logging.warning("Primary loader failed due to column patterns: %s. Attempting legacy summary loader.", exc)
        try:
            _ = load_results_summary(engine, args.experiment_key, args.results_csv)
        except Exception as exc2:
            logging.error("Legacy summary loader also failed: %s", exc2, exc_info=True)
            sys.exit(13)
    except Exception as exc:
        logging.error("Failed to load results CSV '%s': %s", args.results_csv, exc, exc_info=True)
        sys.exit(14)

    # Done
    logging.info("ETL completed successfully for experiment '%s'.", args.experiment_key)
    sys.exit(0)


# Guard
if __name__ == "__main__":
    main()
