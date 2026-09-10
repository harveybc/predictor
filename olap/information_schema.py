"""DDL and loaders for CRISP-DM inventory and information diagnostics."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime

from sqlalchemy import text


SCHEMA = "public"
INVENTORY_SCHEMA = "predictor.crispdm_dataset_inventory.v1"


INFORMATION_DDL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_dataset (
  dataset_key          TEXT PRIMARY KEY,
  source_class         TEXT NOT NULL,
  domain_key           TEXT NOT NULL,
  provider             TEXT NOT NULL,
  dataset_version      TEXT NOT NULL,
  license_id           TEXT NOT NULL,
  availability_policy  TEXT NOT NULL,
  exposure_status      TEXT NOT NULL,
  metadata_json        JSONB NOT NULL DEFAULT '{{}}'::jsonb
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_series (
  dataset_key          TEXT NOT NULL REFERENCES {SCHEMA}.dim_dataset(dataset_key) ON DELETE CASCADE,
  series_key           TEXT NOT NULL,
  panel_key            TEXT,
  metadata_json        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
  PRIMARY KEY (dataset_key, series_key)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_variable (
  dataset_key          TEXT NOT NULL REFERENCES {SCHEMA}.dim_dataset(dataset_key) ON DELETE CASCADE,
  variable_key         TEXT NOT NULL,
  role_key             TEXT NOT NULL,
  unit_key             TEXT NOT NULL,
  physical_dtype       TEXT,
  metadata_json        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
  PRIMARY KEY (dataset_key, variable_key)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.fact_dataset_inventory (
  dataset_key          TEXT NOT NULL REFERENCES {SCHEMA}.dim_dataset(dataset_key) ON DELETE CASCADE,
  inventory_sha256     TEXT NOT NULL,
  physical_sha256      TEXT NOT NULL,
  row_count            BIGINT NOT NULL CHECK (row_count >= 0),
  variable_count       INTEGER NOT NULL CHECK (variable_count >= 0),
  start_at             TIMESTAMPTZ,
  end_at               TIMESTAMPTZ,
  sampling_seconds     DOUBLE PRECISION,
  missing_cell_count   BIGINT NOT NULL CHECK (missing_cell_count >= 0),
  profile_status       TEXT NOT NULL,
  profile_json         JSONB NOT NULL,
  inventoried_at       TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (dataset_key, inventory_sha256)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.fact_variable_profile (
  dataset_key          TEXT NOT NULL,
  variable_key         TEXT NOT NULL,
  inventory_sha256     TEXT NOT NULL,
  observation_count    BIGINT NOT NULL CHECK (observation_count >= 0),
  missing_count        BIGINT NOT NULL CHECK (missing_count >= 0),
  finite_count         BIGINT NOT NULL CHECK (finite_count >= 0),
  unique_count         BIGINT NOT NULL CHECK (unique_count >= 0),
  numeric_fraction     DOUBLE PRECISION CHECK (numeric_fraction BETWEEN 0 AND 1),
  constant_value       BOOLEAN NOT NULL,
  minimum_value        DOUBLE PRECISION,
  maximum_value        DOUBLE PRECISION,
  mean_value           DOUBLE PRECISION,
  std_value            DOUBLE PRECISION,
  profile_json         JSONB NOT NULL,
  PRIMARY KEY (dataset_key, variable_key, inventory_sha256),
  FOREIGN KEY (dataset_key, variable_key)
    REFERENCES {SCHEMA}.dim_variable(dataset_key, variable_key) ON DELETE CASCADE,
  FOREIGN KEY (dataset_key, inventory_sha256)
    REFERENCES {SCHEMA}.fact_dataset_inventory(dataset_key, inventory_sha256) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.fact_split_information (
  experiment_key       TEXT NOT NULL REFERENCES {SCHEMA}.dim_experiment(experiment_key) ON DELETE CASCADE,
  dataset_key          TEXT NOT NULL,
  series_key           TEXT NOT NULL,
  variable_key         TEXT NOT NULL,
  split_key            TEXT NOT NULL REFERENCES {SCHEMA}.dim_dataset_split(split_key),
  measurement_sha256   TEXT NOT NULL,
  raw_bytes            BIGINT NOT NULL CHECK (raw_bytes >= 0),
  zstd_bytes           BIGINT,
  lzma_bytes           BIGINT,
  bz2_bytes            BIGINT,
  entropy_order0_bits  DOUBLE PRECISION,
  context_rate_bits    DOUBLE PRECISION,
  snr_estimate_db      DOUBLE PRECISION,
  estimated_signal_information_bits DOUBLE PRECISION,
  measurement_json     JSONB NOT NULL,
  PRIMARY KEY (experiment_key, dataset_key, series_key, variable_key, split_key, measurement_sha256),
  FOREIGN KEY (dataset_key, series_key)
    REFERENCES {SCHEMA}.dim_series(dataset_key, series_key) ON DELETE RESTRICT,
  FOREIGN KEY (dataset_key, variable_key)
    REFERENCES {SCHEMA}.dim_variable(dataset_key, variable_key) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.fact_run_epoch (
  experiment_key       TEXT NOT NULL REFERENCES {SCHEMA}.dim_experiment(experiment_key) ON DELETE CASCADE,
  seed                  INTEGER NOT NULL,
  epoch                 INTEGER NOT NULL CHECK (epoch >= 0),
  snapshot_sha256       TEXT NOT NULL,
  train_loss            DOUBLE PRECISION,
  validation_loss       DOUBLE PRECISION,
  model_parameter_count BIGINT,
  model_raw_bytes       BIGINT,
  model_zstd_bytes      BIGINT,
  model_lzma_bytes      BIGINT,
  model_compression_ratio DOUBLE PRECISION,
  graph_density         DOUBLE PRECISION,
  graph_weight_entropy  DOUBLE PRECISION,
  graph_spectral_radius DOUBLE PRECISION,
  graph_effective_rank  DOUBLE PRECISION,
  stopped               BOOLEAN NOT NULL DEFAULT FALSE,
  stop_reason           TEXT,
  snapshot_json         JSONB NOT NULL,
  PRIMARY KEY (experiment_key, seed, epoch, snapshot_sha256)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.fact_ingestion_receipt (
  receipt_sha256       TEXT PRIMARY KEY,
  artifact_kind       TEXT NOT NULL,
  artifact_sha256     TEXT NOT NULL,
  artifact_as_of      TIMESTAMPTZ NOT NULL,
  dataset_count       INTEGER NOT NULL CHECK (dataset_count >= 0),
  variable_count      INTEGER NOT NULL CHECK (variable_count >= 0),
  receipt_json        JSONB NOT NULL,
  loaded_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fact_dataset_inventory_status
  ON {SCHEMA}.fact_dataset_inventory (profile_status);
CREATE INDEX IF NOT EXISTS idx_fact_variable_profile_dataset
  ON {SCHEMA}.fact_variable_profile (dataset_key);
CREATE INDEX IF NOT EXISTS idx_fact_split_information_experiment
  ON {SCHEMA}.fact_split_information (experiment_key);
CREATE INDEX IF NOT EXISTS idx_fact_run_epoch_experiment
  ON {SCHEMA}.fact_run_epoch (experiment_key, seed, epoch);
CREATE INDEX IF NOT EXISTS idx_fact_ingestion_receipt_artifact
  ON {SCHEMA}.fact_ingestion_receipt (artifact_kind, artifact_sha256);
"""


def ensure_information_tables(engine) -> None:
    """Create the additive laboratory tables without altering existing facts."""
    with engine.begin() as conn:
        conn.exec_driver_sql(INFORMATION_DDL)


def _require_exact_keys(value: Mapping, required: set[str], allowed: set[str], path: str) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    extra = sorted(keys - allowed)
    if missing or extra:
        raise ValueError(f"{path}: schema mismatch; missing={missing}, extra={extra}")


def _finite_or_none(value, path: str):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: expected a finite number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path}: expected a finite number or null")
    return number


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _non_negative_integer(value, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path}: expected a non-negative integer")
    return value


def _non_empty_string(value, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: expected a non-empty string")
    return value


def validate_inventory_document(document: Mapping) -> None:
    """Validate the inventory envelope before any database write."""
    required = {"schema", "inventory_sha256", "inventoried_at", "scope", "datasets", "summary"}
    _require_exact_keys(document, required, required, "inventory")
    if document["schema"] != INVENTORY_SCHEMA:
        raise ValueError(f"inventory.schema: expected {INVENTORY_SCHEMA}")
    if not _is_sha256(document["inventory_sha256"]):
        raise ValueError("inventory.inventory_sha256: expected SHA-256")
    try:
        inventoried_at = datetime.fromisoformat(document["inventoried_at"].replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("inventory.inventoried_at: expected RFC3339 timestamp") from exc
    if inventoried_at.tzinfo is None:
        raise ValueError("inventory.inventoried_at: timezone is required")
    if not isinstance(document["scope"], Mapping):
        raise ValueError("inventory.scope: expected an object")
    if not isinstance(document["summary"], Mapping):
        raise ValueError("inventory.summary: expected an object")
    summary_keys = {"dataset_count", "variable_count", "row_count", "status_counts"}
    _require_exact_keys(document["summary"], summary_keys, summary_keys, "inventory.summary")
    for key in ("dataset_count", "variable_count", "row_count"):
        _non_negative_integer(document["summary"][key], f"inventory.summary.{key}")
    if not isinstance(document["summary"]["status_counts"], Mapping):
        raise ValueError("inventory.summary.status_counts: expected an object")
    for status, count in document["summary"]["status_counts"].items():
        _non_empty_string(status, "inventory.summary.status_counts key")
        _non_negative_integer(count, f"inventory.summary.status_counts.{status}")
    digest_input = copy.deepcopy(dict(document))
    claimed_inventory_digest = digest_input.pop("inventory_sha256")
    computed_inventory_digest = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    if claimed_inventory_digest != computed_inventory_digest:
        raise ValueError("inventory.inventory_sha256: digest mismatch")
    if not isinstance(document["datasets"], list) or not document["datasets"]:
        raise ValueError("inventory.datasets: expected a non-empty list")

    dataset_ids: set[str] = set()
    for index, dataset in enumerate(document["datasets"]):
        path = f"inventory.datasets[{index}]"
        required_dataset = {
            "dataset_id", "source_class", "domain", "provider", "version", "license_id",
            "availability_policy", "exposure_status", "root_id", "relative_path", "format",
            "physical_sha256", "row_count", "variable_count", "time_profile", "variables",
            "profile_status", "metadata_issues", "profile_sha256",
        }
        _require_exact_keys(dataset, required_dataset, required_dataset, path)
        dataset_id = dataset["dataset_id"]
        _non_empty_string(dataset_id, f"{path}.dataset_id")
        if dataset_id in dataset_ids:
            raise ValueError(f"{path}.dataset_id: duplicate {dataset_id}")
        dataset_ids.add(dataset_id)
        for key in (
            "source_class", "domain", "provider", "version", "license_id",
            "availability_policy", "exposure_status", "root_id", "relative_path",
            "format", "profile_status",
        ):
            _non_empty_string(dataset[key], f"{path}.{key}")
        if dataset["source_class"] not in {"financial", "public", "synthetic"}:
            raise ValueError(f"{path}.source_class: unsupported value")
        _non_negative_integer(dataset["row_count"], f"{path}.row_count")
        _non_negative_integer(dataset["variable_count"], f"{path}.variable_count")
        if not isinstance(dataset["variables"], list) or not dataset["variables"]:
            raise ValueError(f"{path}.variables: expected a non-empty list")
        if dataset["variable_count"] != len(dataset["variables"]):
            raise ValueError(f"{path}.variable_count: does not match variables")
        if not _is_sha256(dataset["physical_sha256"]):
            raise ValueError(f"{path}.physical_sha256: expected SHA-256")
        if not isinstance(dataset["metadata_issues"], list) or not all(
            isinstance(issue, str) and issue for issue in dataset["metadata_issues"]
        ):
            raise ValueError(f"{path}.metadata_issues: expected a list of strings")

        required_time_profile = {
            "timestamp_column", "start_at", "end_at", "invalid_timestamp_count",
            "duplicate_timestamp_count", "monotonic_non_decreasing",
            "median_sampling_seconds", "irregular_interval_fraction",
        }
        if not isinstance(dataset["time_profile"], Mapping):
            raise ValueError(f"{path}.time_profile: expected an object")
        _require_exact_keys(
            dataset["time_profile"], required_time_profile, required_time_profile,
            f"{path}.time_profile",
        )
        _non_empty_string(
            dataset["time_profile"]["timestamp_column"],
            f"{path}.time_profile.timestamp_column",
        )
        _non_negative_integer(
            dataset["time_profile"]["invalid_timestamp_count"],
            f"{path}.time_profile.invalid_timestamp_count",
        )
        _non_negative_integer(
            dataset["time_profile"]["duplicate_timestamp_count"],
            f"{path}.time_profile.duplicate_timestamp_count",
        )
        if type(dataset["time_profile"]["monotonic_non_decreasing"]) is not bool:
            raise ValueError(f"{path}.time_profile.monotonic_non_decreasing: expected boolean")
        _finite_or_none(
            dataset["time_profile"]["median_sampling_seconds"],
            f"{path}.time_profile.median_sampling_seconds",
        )
        for key in ("start_at", "end_at"):
            value = dataset["time_profile"][key]
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{path}.time_profile.{key}: expected timestamp string or null")
        irregular = _finite_or_none(
            dataset["time_profile"]["irregular_interval_fraction"],
            f"{path}.time_profile.irregular_interval_fraction",
        )
        if irregular is not None and not 0 <= irregular <= 1:
            raise ValueError(f"{path}.time_profile.irregular_interval_fraction: expected [0, 1]")

        dataset_digest_input = dict(dataset)
        claimed_dataset_digest = dataset_digest_input.pop("profile_sha256")
        computed_dataset_digest = hashlib.sha256(
            json.dumps(dataset_digest_input, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        ).hexdigest()
        if not _is_sha256(claimed_dataset_digest) or claimed_dataset_digest != computed_dataset_digest:
            raise ValueError(f"{path}.profile_sha256: digest mismatch")

        names: set[str] = set()
        for variable_index, variable in enumerate(dataset["variables"]):
            variable_path = f"{path}.variables[{variable_index}]"
            required_variable = {
                "name", "role", "unit", "physical_dtype", "observation_count", "missing_count",
                "finite_count", "unique_count", "numeric_fraction", "constant", "minimum",
                "maximum", "mean", "std", "profile_sha256",
            }
            _require_exact_keys(variable, required_variable, required_variable, variable_path)
            name = variable["name"]
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(f"{variable_path}.name: invalid or duplicate")
            names.add(name)
            for key in ("role", "unit", "physical_dtype"):
                _non_empty_string(variable[key], f"{variable_path}.{key}")
            observation_count = _non_negative_integer(
                variable["observation_count"], f"{variable_path}.observation_count"
            )
            missing_count = _non_negative_integer(
                variable["missing_count"], f"{variable_path}.missing_count"
            )
            finite_count = _non_negative_integer(
                variable["finite_count"], f"{variable_path}.finite_count"
            )
            unique_count = _non_negative_integer(
                variable["unique_count"], f"{variable_path}.unique_count"
            )
            if missing_count > observation_count:
                raise ValueError(f"{variable_path}.missing_count: exceeds observations")
            non_missing_count = observation_count - missing_count
            if finite_count > non_missing_count:
                raise ValueError(f"{variable_path}.finite_count: exceeds non-missing observations")
            if unique_count > non_missing_count:
                raise ValueError(f"{variable_path}.unique_count: exceeds non-missing observations")
            if type(variable["constant"]) is not bool:
                raise ValueError(f"{variable_path}.constant: expected boolean")
            for key in ("minimum", "maximum", "mean", "std", "numeric_fraction"):
                _finite_or_none(variable[key], f"{variable_path}.{key}")
            numeric_fraction = variable["numeric_fraction"]
            if numeric_fraction is not None and not 0 <= numeric_fraction <= 1:
                raise ValueError(f"{variable_path}.numeric_fraction: expected [0, 1]")
            variable_digest_input = dict(variable)
            claimed_variable_digest = variable_digest_input.pop("profile_sha256")
            computed_variable_digest = hashlib.sha256(
                json.dumps(variable_digest_input, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
            ).hexdigest()
            if not _is_sha256(claimed_variable_digest) or claimed_variable_digest != computed_variable_digest:
                raise ValueError(f"{variable_path}.profile_sha256: digest mismatch")

    expected_summary = {
        "dataset_count": len(document["datasets"]),
        "variable_count": sum(dataset["variable_count"] for dataset in document["datasets"]),
        "row_count": sum(dataset["row_count"] for dataset in document["datasets"]),
        "status_counts": {},
    }
    for dataset in document["datasets"]:
        status = dataset["profile_status"]
        expected_summary["status_counts"][status] = expected_summary["status_counts"].get(status, 0) + 1
    expected_summary["status_counts"] = dict(sorted(expected_summary["status_counts"].items()))
    if document["summary"] != expected_summary:
        raise ValueError("inventory.summary: does not match re-derived counts")


def load_inventory_document(engine, document: Mapping) -> dict[str, int | str]:
    """Load one validated inventory atomically into the additive OLAP tables."""
    validate_inventory_document(document)
    ensure_information_tables(engine)
    counts = {
        "datasets": 0,
        "series": 0,
        "variables": 0,
        "dataset_profiles": 0,
        "variable_profiles": 0,
    }
    receipt_body = {
        "schema": "predictor.olap_ingestion_receipt.v1",
        "artifact_kind": "CRISPDM_DATASET_INVENTORY",
        "artifact_sha256": document["inventory_sha256"],
        "artifact_as_of": document["inventoried_at"],
        "dataset_count": document["summary"]["dataset_count"],
        "variable_count": document["summary"]["variable_count"],
    }
    receipt_sha256 = hashlib.sha256(
        json.dumps(receipt_body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()

    dataset_sql = text(f"""
        INSERT INTO {SCHEMA}.dim_dataset
          (dataset_key, source_class, domain_key, provider, dataset_version, license_id,
           availability_policy, exposure_status, metadata_json)
        VALUES
          (:dataset_key, :source_class, :domain_key, :provider, :dataset_version, :license_id,
           :availability_policy, :exposure_status, CAST(:metadata_json AS JSONB))
        ON CONFLICT (dataset_key) DO UPDATE SET
          source_class = EXCLUDED.source_class,
          domain_key = EXCLUDED.domain_key,
          provider = EXCLUDED.provider,
          dataset_version = EXCLUDED.dataset_version,
          license_id = EXCLUDED.license_id,
          availability_policy = EXCLUDED.availability_policy,
          exposure_status = EXCLUDED.exposure_status,
          metadata_json = EXCLUDED.metadata_json;
    """)
    series_sql = text(f"""
        INSERT INTO {SCHEMA}.dim_series
          (dataset_key, series_key, panel_key, metadata_json)
        VALUES
          (:dataset_key, :series_key, :panel_key, CAST(:metadata_json AS JSONB))
        ON CONFLICT (dataset_key, series_key) DO UPDATE SET
          panel_key = EXCLUDED.panel_key,
          metadata_json = EXCLUDED.metadata_json;
    """)
    variable_sql = text(f"""
        INSERT INTO {SCHEMA}.dim_variable
          (dataset_key, variable_key, role_key, unit_key, physical_dtype, metadata_json)
        VALUES
          (:dataset_key, :variable_key, :role_key, :unit_key, :physical_dtype, CAST(:metadata_json AS JSONB))
        ON CONFLICT (dataset_key, variable_key) DO UPDATE SET
          role_key = EXCLUDED.role_key,
          unit_key = EXCLUDED.unit_key,
          physical_dtype = EXCLUDED.physical_dtype,
          metadata_json = EXCLUDED.metadata_json;
    """)
    dataset_profile_sql = text(f"""
        INSERT INTO {SCHEMA}.fact_dataset_inventory
          (dataset_key, inventory_sha256, physical_sha256, row_count, variable_count,
           start_at, end_at, sampling_seconds, missing_cell_count, profile_status,
           profile_json, inventoried_at)
        VALUES
          (:dataset_key, :inventory_sha256, :physical_sha256, :row_count, :variable_count,
           :start_at, :end_at, :sampling_seconds, :missing_cell_count, :profile_status,
           CAST(:profile_json AS JSONB), :inventoried_at)
        ON CONFLICT (dataset_key, inventory_sha256) DO NOTHING;
    """)
    variable_profile_sql = text(f"""
        INSERT INTO {SCHEMA}.fact_variable_profile
          (dataset_key, variable_key, inventory_sha256, observation_count, missing_count,
           finite_count, unique_count, numeric_fraction, constant_value, minimum_value,
           maximum_value, mean_value, std_value, profile_json)
        VALUES
          (:dataset_key, :variable_key, :inventory_sha256, :observation_count, :missing_count,
           :finite_count, :unique_count, :numeric_fraction, :constant_value, :minimum_value,
           :maximum_value, :mean_value, :std_value, CAST(:profile_json AS JSONB))
        ON CONFLICT (dataset_key, variable_key, inventory_sha256) DO NOTHING;
    """)
    receipt_sql = text(f"""
        INSERT INTO {SCHEMA}.fact_ingestion_receipt
          (receipt_sha256, artifact_kind, artifact_sha256, artifact_as_of,
           dataset_count, variable_count, receipt_json)
        VALUES
          (:receipt_sha256, :artifact_kind, :artifact_sha256, :artifact_as_of,
           :dataset_count, :variable_count, CAST(:receipt_json AS JSONB))
        ON CONFLICT (receipt_sha256) DO NOTHING;
    """)

    with engine.begin() as conn:
        for dataset in document["datasets"]:
            dataset_metadata = {
                "root_id": dataset["root_id"],
                "relative_path": dataset["relative_path"],
                "format": dataset["format"],
                "metadata_issues": dataset["metadata_issues"],
                "profile_sha256": dataset["profile_sha256"],
            }
            conn.execute(dataset_sql, {
                "dataset_key": dataset["dataset_id"],
                "source_class": dataset["source_class"],
                "domain_key": dataset["domain"],
                "provider": dataset["provider"],
                "dataset_version": dataset["version"],
                "license_id": dataset["license_id"],
                "availability_policy": dataset["availability_policy"],
                "exposure_status": dataset["exposure_status"],
                "metadata_json": json.dumps(dataset_metadata, sort_keys=True),
            })
            counts["datasets"] += 1

            conn.execute(series_sql, {
                "dataset_key": dataset["dataset_id"],
                "series_key": "__dataset__",
                "panel_key": None,
                "metadata_json": json.dumps({
                    "meaning": "dataset-level aggregate; no panel series inventory supplied"
                }),
            })
            counts["series"] += 1

            for variable in dataset["variables"]:
                conn.execute(variable_sql, {
                    "dataset_key": dataset["dataset_id"],
                    "variable_key": variable["name"],
                    "role_key": variable["role"],
                    "unit_key": variable["unit"],
                    "physical_dtype": variable["physical_dtype"],
                    "metadata_json": json.dumps({"profile_sha256": variable["profile_sha256"]}),
                })
                counts["variables"] += 1

            time_profile = dataset["time_profile"]
            conn.execute(dataset_profile_sql, {
                "dataset_key": dataset["dataset_id"],
                "inventory_sha256": document["inventory_sha256"],
                "physical_sha256": dataset["physical_sha256"],
                "row_count": dataset["row_count"],
                "variable_count": dataset["variable_count"],
                "start_at": time_profile["start_at"],
                "end_at": time_profile["end_at"],
                "sampling_seconds": time_profile["median_sampling_seconds"],
                "missing_cell_count": sum(v["missing_count"] for v in dataset["variables"]),
                "profile_status": dataset["profile_status"],
                "profile_json": json.dumps(dataset, sort_keys=True),
                "inventoried_at": document["inventoried_at"],
            })
            counts["dataset_profiles"] += 1

            for variable in dataset["variables"]:
                conn.execute(variable_profile_sql, {
                    "dataset_key": dataset["dataset_id"],
                    "variable_key": variable["name"],
                    "inventory_sha256": document["inventory_sha256"],
                    "observation_count": variable["observation_count"],
                    "missing_count": variable["missing_count"],
                    "finite_count": variable["finite_count"],
                    "unique_count": variable["unique_count"],
                    "numeric_fraction": variable["numeric_fraction"],
                    "constant_value": variable["constant"],
                    "minimum_value": variable["minimum"],
                    "maximum_value": variable["maximum"],
                    "mean_value": variable["mean"],
                    "std_value": variable["std"],
                    "profile_json": json.dumps(variable, sort_keys=True),
                })
                counts["variable_profiles"] += 1

        conn.execute(receipt_sql, {
            "receipt_sha256": receipt_sha256,
            "artifact_kind": receipt_body["artifact_kind"],
            "artifact_sha256": receipt_body["artifact_sha256"],
            "artifact_as_of": receipt_body["artifact_as_of"],
            "dataset_count": receipt_body["dataset_count"],
            "variable_count": receipt_body["variable_count"],
            "receipt_json": json.dumps(receipt_body, sort_keys=True),
        })

    counts["receipt_sha256"] = receipt_sha256

    return counts
