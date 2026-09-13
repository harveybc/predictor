"""Cube access for data-gov: SELECT-only `query`, append-only `write_metrics`
on the gov_* tables (docs/04_FLOW_V2.md §3). Never truncates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

_WRITE = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|copy|grant|revoke|create|vacuum|call|do)\b",
    re.I,
)
_BANNED = re.compile(
    r"\b(pg_sleep|pg_read_file|pg_ls_dir|lo_import|dblink|file_fdw)\b",
    re.I,
)
_LIMIT = re.compile(r"\blimit\s+(\d+)\s*;?\s*$", re.I)
MAX_LIMIT = 5000

_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_LINEAGE = ("VERIFIED", "UNVERIFIED")
METRIC_FIELDS = ("metric", "value", "split", "horizon", "std_dev", "min_value", "max_value", "unit")
DATASET_FIELDS = ("lake", "resource", "sha256", "role")
DATASET_LINEAGE_FIELDS = (
    "lineage", "reason", "event_id", "source_sha256",
    "range_from", "range_to", "delivery", "time_column",
)

GOV_TABLES = ("gov_report", "gov_metric", "gov_dataset")
GOV_VIEW = "gov_metric_current"


def _opt_str(obj, name):
    value = obj.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _req_str(obj, name):
    value = obj.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is required")
    return value


def _opt_float(obj, name):
    value = obj.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or null")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite value")
    return value


def _opt_int(obj, name):
    value = obj.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer or null")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer or null")
    return int(value)


def normalise_report(report) -> dict:
    """Validate a metrics report (§3) and normalise it: numbers coerced,
    duplicate datasets collapsed, metrics and datasets sorted. ValueError on
    anything invalid (the HTTP layer maps it to 400)."""
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    key = _req_str(report, "experiment_key")
    if not _KEY.match(key):
        raise ValueError("invalid experiment_key")
    set_key = _opt_str(report, "experiment_set_key")
    if set_key is not None and not _KEY.match(set_key):
        raise ValueError("invalid experiment_set_key")
    tags = report.get("tags")
    if tags is None:
        tags = {}
    if not isinstance(tags, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in tags.items()
    ):
        raise ValueError("tags must be a map of strings")

    metrics_in = report.get("metrics")
    if not isinstance(metrics_in, list) or not metrics_in:
        raise ValueError("metrics must be a non-empty list")
    metrics = []
    for item in metrics_in:
        if not isinstance(item, dict):
            raise ValueError("metric entries must be objects")
        metrics.append({
            "metric": _req_str(item, "metric"),
            "value": _opt_float(item, "value"),
            "split": _opt_str(item, "split"),
            "horizon": _opt_int(item, "horizon"),
            "std_dev": _opt_float(item, "std_dev"),
            "min_value": _opt_float(item, "min_value"),
            "max_value": _opt_float(item, "max_value"),
            "unit": _opt_str(item, "unit"),
        })
    metrics.sort(key=lambda m: (
        m["metric"], m["split"] or "", m["horizon"] if m["horizon"] is not None else -1
    ))

    datasets_in = report.get("datasets")
    if datasets_in is None:
        datasets_in = []
    if not isinstance(datasets_in, list):
        raise ValueError("datasets must be a list")
    seen = {}
    for item in datasets_in:
        if not isinstance(item, dict):
            raise ValueError("dataset entries must be objects")
        entry = {
            "lake": _req_str(item, "lake"),
            "resource": _req_str(item, "resource"),
            "sha256": _req_str(item, "sha256"),
            "role": _opt_str(item, "role"),
        }
        if not _HEX64.match(entry["sha256"]):
            raise ValueError("dataset sha256 must be 64 hex characters")
        ident = tuple(entry[k] for k in DATASET_FIELDS)
        if ident in seen:
            continue
        lineage = _opt_str(item, "lineage") or "UNVERIFIED"
        if lineage not in _LINEAGE:
            raise ValueError("dataset lineage must be VERIFIED or UNVERIFIED")
        # data-gov's accounting detail names the range "from"/"to"; the
        # gov_dataset columns are range_from/range_to. Both are accepted.
        entry.update({
            "lineage": lineage,
            "reason": _opt_str(item, "reason"),
            "event_id": _opt_int(item, "event_id"),
            "source_sha256": _opt_str(item, "source_sha256"),
            "range_from": _opt_str(item, "range_from") if "range_from" in item else _opt_str(item, "from"),
            "range_to": _opt_str(item, "range_to") if "range_to" in item else _opt_str(item, "to"),
            "delivery": _opt_str(item, "delivery"),
            "time_column": _opt_str(item, "time_column"),
        })
        seen[ident] = entry
    datasets = sorted(
        seen.values(), key=lambda d: (d["lake"], d["resource"], d["sha256"], d["role"] or "")
    )

    lineage = _opt_str(report, "lineage")
    if lineage is None:
        lineage = "VERIFIED" if datasets and all(
            d["lineage"] == "VERIFIED" for d in datasets
        ) else "UNVERIFIED"
    elif lineage not in _LINEAGE:
        raise ValueError("lineage must be VERIFIED or UNVERIFIED")

    return {
        "experiment_key": key,
        "experiment_set_key": set_key,
        "actor": _req_str(report, "actor"),
        "lake": _req_str(report, "lake"),
        "config_sha256": _opt_str(report, "config_sha256"),
        "code_commit": _opt_str(report, "code_commit"),
        "project": _opt_str(report, "project"),
        "phase": _opt_str(report, "phase"),
        "tags": tags,
        "datasets": datasets,
        "metrics": metrics,
        "lineage": lineage,
        "received_at": _opt_str(report, "received_at"),
    }


def canonical_body(normalised: dict) -> dict:
    """The hashed body of §3 'Report identity': lineage, receipt time and
    the per-dataset lineage fields are left out."""
    return {
        "experiment_key": normalised["experiment_key"],
        "experiment_set_key": normalised["experiment_set_key"],
        "actor": normalised["actor"],
        "lake": normalised["lake"],
        "config_sha256": normalised["config_sha256"],
        "code_commit": normalised["code_commit"],
        "project": normalised["project"],
        "phase": normalised["phase"],
        "tags": normalised["tags"],
        "datasets": [{k: d[k] for k in DATASET_FIELDS} for d in normalised["datasets"]],
        "metrics": [{k: m[k] for k in METRIC_FIELDS} for m in normalised["metrics"]],
    }


def canonical_text(body: dict) -> str:
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def report_sha256(report) -> str:
    """sha256 of the canonical body of a report (raw or normalised)."""
    normalised = normalise_report(report)
    return hashlib.sha256(canonical_text(canonical_body(normalised)).encode()).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Plugin:
    plugin_params = {
        "sqlite_path": None,
        "schema": "public",
        "holdout_start": "2025-01-01",
        "time_column": None,
        "lake_id": "olap_cube",
        "title": "predictor OLAP",
        "description": "",
        "kind": "sql_olap",
    }

    def __init__(self):
        self.params = dict(self.plugin_params)
        self._engine = None
        self._write_engine = None
        self._schema_error = None

    def set_params(self, **kwargs):
        self.params.update(kwargs)
        self._engine = None
        self._write_engine = None
        self._schema_error = None

    def _pg_url(self, user, password):
        host = os.getenv("PGHOST", "127.0.0.1")
        port = os.getenv("PGPORT", "5432")
        db = os.getenv("PGDATABASE", "predictor_olap")
        if not password:
            raise RuntimeError("PGPASSWORD is not set")
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

    def engine(self):
        if self._engine is not None:
            return self._engine
        sqlite_path = self.params.get("sqlite_path")
        if sqlite_path:
            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(f"sqlite:///{sqlite_path}", connect_args={"timeout": 30})
        else:
            user = os.getenv("PGUSER") or "metabase"
            engine = create_engine(
                self._pg_url(user, os.getenv("PGPASSWORD")), pool_pre_ping=True
            )
        self._engine = engine
        # The gov_* DDL runs once per engine, in autocommit, under the read
        # role (the write role may hold INSERT only). A failure here must
        # not take the SELECT path down, so it is kept for write_metrics.
        self._schema_error = None
        try:
            self._ensure_schema(engine)
        except Exception as exc:
            self._schema_error = exc
            print(f"gov_* schema not ready: {exc}", file=sys.stderr)
        return engine

    def write_engine(self):
        """Engine for write_metrics: PGUSER_WRITE / PGPASSWORD_WRITE when set,
        else the read engine (same credentials, same SQLite file)."""
        self.engine()
        if self._write_engine is not None:
            return self._write_engine
        user = os.getenv("PGUSER_WRITE")
        if self.params.get("sqlite_path") or not user:
            self._write_engine = self._engine
        else:
            password = os.getenv("PGPASSWORD_WRITE") or os.getenv("PGPASSWORD")
            self._write_engine = create_engine(self._pg_url(user, password), pool_pre_ping=True)
        return self._write_engine

    def _qualified(self, name: str) -> str:
        if self.params.get("sqlite_path"):
            return f'"{name}"'
        schema = self.params.get("schema") or "public"
        return f'"{schema}"."{name}"'

    def _ddl(self, dialect: str):
        t = self._qualified
        view_select = (
            "SELECT m.report_sha256, m.experiment_key, r.lake_id, r.received_at, m.metric,"
            " m.value, m.split, m.horizon, m.std_dev, m.min_value, m.max_value, m.unit"
            f" FROM {t('gov_metric')} m JOIN {t('gov_report')} r"
            " ON r.report_sha256 = m.report_sha256"
            " WHERE r.received_at = (SELECT MAX(r2.received_at)"
            f" FROM {t('gov_report')} r2 WHERE r2.experiment_key = r.experiment_key"
            " AND r2.lake_id = r.lake_id)"
        )
        create_view = (
            "CREATE VIEW IF NOT EXISTS" if dialect == "sqlite" else "CREATE OR REPLACE VIEW"
        )
        return [
            f"CREATE TABLE IF NOT EXISTS {t('gov_report')} ("
            " report_sha256 TEXT PRIMARY KEY, experiment_key TEXT NOT NULL,"
            " experiment_set_key TEXT, actor TEXT NOT NULL, lake_id TEXT NOT NULL,"
            " received_at TEXT NOT NULL, lineage TEXT NOT NULL,"
            " config_sha256 TEXT, code_commit TEXT, project TEXT, phase TEXT, tags_json TEXT,"
            " n_metrics INTEGER NOT NULL, n_datasets INTEGER NOT NULL)",
            f"CREATE TABLE IF NOT EXISTS {t('gov_metric')} ("
            " report_sha256 TEXT NOT NULL, experiment_key TEXT NOT NULL, metric TEXT NOT NULL,"
            " value DOUBLE PRECISION, split TEXT, horizon INTEGER, std_dev DOUBLE PRECISION,"
            " min_value DOUBLE PRECISION, max_value DOUBLE PRECISION, unit TEXT)",
            f"CREATE TABLE IF NOT EXISTS {t('gov_dataset')} ("
            " report_sha256 TEXT NOT NULL, experiment_key TEXT NOT NULL, lake_id TEXT NOT NULL,"
            " resource_id TEXT NOT NULL, sha256 TEXT NOT NULL, role TEXT, lineage TEXT NOT NULL,"
            " reason TEXT, event_id INTEGER, source_sha256 TEXT, range_from TEXT, range_to TEXT,"
            " delivery TEXT, time_column TEXT)",
            f"CREATE INDEX IF NOT EXISTS gov_metric_report_idx ON {t('gov_metric')} (report_sha256)",
            f"CREATE INDEX IF NOT EXISTS gov_dataset_report_idx ON {t('gov_dataset')} (report_sha256)",
            f"CREATE INDEX IF NOT EXISTS gov_dataset_sha256_idx ON {t('gov_dataset')} (sha256)",
            f"CREATE INDEX IF NOT EXISTS gov_report_experiment_idx ON {t('gov_report')}"
            " (experiment_key, received_at)",
            f"{create_view} {t(GOV_VIEW)} AS {view_select}",
        ]

    def _ensure_schema(self, engine):
        dialect = engine.dialect.name
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            if dialect == "sqlite":
                conn.execute(text("PRAGMA journal_mode=WAL"))
            for statement in self._ddl(dialect):
                conn.execute(text(statement))

    def write_metrics(self, report):
        """Store one report (§3 'Lake side'). Returns
        {"stored", "already_stored", "lineage"}; ValueError on an invalid
        report or a report_sha256 that does not match the canonical body."""
        normalised = normalise_report(report)
        given = report.get("report_sha256")
        if not isinstance(given, str) or not _HEX64.match(given):
            raise ValueError("report_sha256 is required")
        digest = hashlib.sha256(
            canonical_text(canonical_body(normalised)).encode()
        ).hexdigest()
        if given.lower() != digest:
            raise ValueError("report_sha256 mismatch")
        if self._engine is None:
            self.engine()
        if self._schema_error is not None:
            raise RuntimeError(f"gov_* schema not ready: {self._schema_error}")
        t = self._qualified
        key = normalised["experiment_key"]
        report_row = {
            "report_sha256": digest,
            "experiment_key": key,
            "experiment_set_key": normalised["experiment_set_key"],
            "actor": normalised["actor"],
            "lake_id": normalised["lake"],
            "received_at": normalised["received_at"] or _now_utc(),
            "lineage": normalised["lineage"],
            "config_sha256": normalised["config_sha256"],
            "code_commit": normalised["code_commit"],
            "project": normalised["project"],
            "phase": normalised["phase"],
            "tags_json": canonical_text(normalised["tags"]),
            "n_metrics": len(normalised["metrics"]),
            "n_datasets": len(normalised["datasets"]),
        }
        metric_rows = [
            {"report_sha256": digest, "experiment_key": key, **m} for m in normalised["metrics"]
        ]
        dataset_rows = [
            {
                "report_sha256": digest,
                "experiment_key": key,
                "lake_id": d["lake"],
                "resource_id": d["resource"],
                "sha256": d["sha256"],
                "role": d["role"],
                **{k: d[k] for k in DATASET_LINEAGE_FIELDS},
            }
            for d in normalised["datasets"]
        ]
        with self.write_engine().begin() as conn:
            inserted = conn.execute(
                text(
                    f"INSERT INTO {t('gov_report')} (report_sha256, experiment_key,"
                    " experiment_set_key, actor, lake_id, received_at, lineage, config_sha256,"
                    " code_commit, project, phase, tags_json, n_metrics, n_datasets)"
                    " VALUES (:report_sha256, :experiment_key, :experiment_set_key, :actor,"
                    " :lake_id, :received_at, :lineage, :config_sha256, :code_commit, :project,"
                    " :phase, :tags_json, :n_metrics, :n_datasets)"
                    " ON CONFLICT (report_sha256) DO NOTHING"
                ),
                report_row,
            ).rowcount
            if inserted:
                conn.execute(
                    text(
                        f"INSERT INTO {t('gov_metric')} (report_sha256, experiment_key, metric,"
                        " value, split, horizon, std_dev, min_value, max_value, unit)"
                        " VALUES (:report_sha256, :experiment_key, :metric, :value, :split,"
                        " :horizon, :std_dev, :min_value, :max_value, :unit)"
                    ),
                    metric_rows,
                )
                if dataset_rows:
                    conn.execute(
                        text(
                            f"INSERT INTO {t('gov_dataset')} (report_sha256, experiment_key,"
                            " lake_id, resource_id, sha256, role, lineage, reason, event_id,"
                            " source_sha256, range_from, range_to, delivery, time_column)"
                            " VALUES (:report_sha256, :experiment_key, :lake_id, :resource_id,"
                            " :sha256, :role, :lineage, :reason, :event_id, :source_sha256,"
                            " :range_from, :range_to, :delivery, :time_column)"
                        ),
                        dataset_rows,
                    )
        if inserted:
            return {"stored": True, "already_stored": False, "lineage": normalised["lineage"]}
        # Read through the read engine: the write role may hold INSERT only.
        with self.engine().connect() as conn:
            stored = conn.execute(
                text(f"SELECT lineage FROM {t('gov_report')} WHERE report_sha256 = :h"),
                {"h": digest},
            ).scalar()
        return {"stored": False, "already_stored": True, "lineage": stored}

    def discover(self):
        insp = inspect(self.engine())
        dialect = self.engine().dialect.name
        schema = None if dialect == "sqlite" else (self.params.get("schema") or "public")
        names = insp.get_table_names(schema=schema)
        items = []
        with self.engine().connect() as conn:
            for name in names:
                if schema:
                    q = text(f'SELECT COUNT(*) FROM "{schema}"."{name}"')
                else:
                    q = text(f'SELECT COUNT(*) FROM "{name}"')
                n = conn.execute(q).scalar()
                items.append({"resource_id": name, "kind": "table", "rows": int(n or 0)})
        return items

    def list_resources(self):
        return self.discover()

    def _guard(self, sql: str) -> str:
        text_sql = (sql or "").strip().rstrip(";").strip()
        if ";" in text_sql:
            raise ValueError("multiple statements are not allowed")
        head = text_sql.split(None, 1)[0].lower() if text_sql else ""
        if head not in {"select", "with"}:
            raise ValueError("only SELECT/WITH is allowed")
        if _WRITE.search(text_sql):
            raise ValueError("write SQL is not allowed")
        if _BANNED.search(text_sql):
            raise ValueError("function is not allowed")
        match = _LIMIT.search(text_sql)
        if not match:
            raise ValueError("LIMIT n is required")
        if int(match.group(1)) > MAX_LIMIT:
            raise ValueError("LIMIT too large")
        return text_sql

    def query(self, sql: str):
        guarded = self._guard(sql)
        with self.engine().connect() as conn:
            result = conn.execute(text(guarded))
            rows = [dict(row._mapping) for row in result]
        holdout = self.params.get("holdout_start")
        if holdout:
            limit = date.fromisoformat(str(holdout)[:10])
            time_cols = [self.params.get("time_column")] if self.params.get("time_column") else []
            if rows:
                for key in rows[0]:
                    lk = str(key).lower()
                    if lk in {"ts", "time", "date", "datetime", "timestamp"} or "time" in lk or "date" in lk:
                        if key not in time_cols:
                            time_cols.append(key)
            for row in rows:
                for col in time_cols:
                    if not col or col not in row or not row[col]:
                        continue
                    try:
                        day = date.fromisoformat(str(row[col])[:10])
                    except ValueError:
                        continue
                    if day >= limit:
                        raise PermissionError("holdout")
        canonical = json.dumps(rows, default=str, sort_keys=True, separators=(",", ":"))
        return {
            "rows": rows,
            "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "bytes": len(canonical.encode()),
        }

    def storage(self):
        import shutil

        sqlite_path = self.params.get("sqlite_path")
        if sqlite_path:
            path = Path(sqlite_path)
            usage = shutil.disk_usage(path.parent if path.parent.exists() else Path("."))
            size = path.stat().st_size if path.exists() else 0
            root = str(path)
        else:
            usage = shutil.disk_usage(".")
            size = 0
            root = "postgresql (PGHOST)"
        return {
            "root": root,
            "host_total": usage.total,
            "host_used": usage.used,
            "host_free": usage.free,
            "lake_bytes": size,
        }

    def describe(self):
        try:
            n_resources = len(self.discover())
        except Exception:
            n_resources = None
        return {
            "lake_id": self.params.get("lake_id"),
            "title": self.params.get("title"),
            "description": self.params.get("description"),
            "kind": self.params.get("kind"),
            "root_path": self.storage()["root"],
            "holdout_start": self.params.get("holdout_start"),
            "n_resources": n_resources,
        }
